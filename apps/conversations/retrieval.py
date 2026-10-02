"""Recuperação de trechos dos materiais (RAG): busca híbrida, vizinhança e refs opacas.

- `search`: vetorial (pgvector, cosseno) + textual (tsvector 'portuguese'), fundidas
  por RRF. Sempre filtrada pelas permissões do usuário e, se pedido, por curso.
- `neighbors`: chunks adjacentes de um chunk já recuperado (mesmo documento).
- `RefRegistry` / `format_context`: o LLM só vê referências opacas ("T1", "T2"...) e
  o título da seção; o título do DOCUMENTO nunca entra no texto, então não há o que
  vazar. O mapa ref -> chunk fica no servidor.
"""

import logging
import re
from dataclasses import dataclass, field

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F
from pgvector.django import CosineDistance

from apps.ai_providers import services as ai_providers
from apps.documents.models import DocumentChunk
from apps.documents.views import get_documents_queryset

from .guard import _normalize as _normalize_title

logger = logging.getLogger(__name__)

SEARCH_CONFIG = "portuguese"
RRF_K = 60
CANDIDATES_PER_SIDE = 20
DEFAULT_TOP_K = 6
CURRICULUM_TOP_K = 8
CURRICULUM_EXTRA_DISTANCE = 0.05
# Acerto textual por "qualquer termo" (consulta relaxada) é evidência fraca: só entra
# se o trecho também não for semanticamente distante demais (cutoff + esta margem).
RELAXED_TEXT_DISTANCE_MARGIN = 0.15
MAX_RELAXED_TERMS = 12
COURSE_FILTER_CODES = ("cc", "ads")  # "ambos"/"indefinido"/None: sem filtro de curso

_WORD_RE = re.compile(r"\w{2,}", re.UNICODE)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    heading: str
    content: str
    score: float  # RRF (ou 0.0 em `neighbors`); maior = melhor
    distance: float | None  # cosseno; None quando não calculada
    order: int | None  # `DocumentChunk.index` (posição dentro do documento)
    # Só para limpar o heading (ver `_clean_heading`); nunca é exibido nem vai ao LLM.
    document_title: str = field(default="", repr=False, compare=False)

    @property
    def id(self) -> int:
        """Compatibilidade com o pipeline legacy, que usava o model `DocumentChunk`."""
        return self.chunk_id


# --- permissões e filtros ----------------------------------------------------------


def get_accessible_chunks_queryset(user):
    """Chunks prontos, com embedding, de documentos que o usuário pode ver."""
    return DocumentChunk.objects.filter(
        document__in=get_documents_queryset(user),
        document__status="ready",
        embedding__isnull=False,
    ).select_related("document")


def _scoped_queryset(user, course_code: str | None):
    queryset = get_accessible_chunks_queryset(user)
    if course_code in COURSE_FILTER_CODES:
        queryset = queryset.filter(document__courses__code=course_code)
    return queryset


_FIELDS = ("id", "document_id", "heading", "content", "index", "document__title")
HEADING_SEPARATOR = " > "


def _is_specific(normalized: str) -> bool:
    """Texto normalizado específico o bastante (2+ palavras e 8+ caracteres) para que
    uma coincidência parcial com o título do documento valha como vazamento dele."""
    return len(normalized) >= 8 and len(normalized.split()) >= 2


def _clean_heading(heading: str, document_title: str) -> str:
    """Tira do breadcrumb ("A > B > C") os segmentos que repetem o título do documento
    (o primeiro heading de um PDF costuma ser o próprio título/nome do arquivo). Assim
    o rótulo `[T1 · seção: ...]` nunca entrega o título ao LLM. Descarta o segmento se
    for igual ao título, se o título (específico) estiver contido nele ou se ele (com
    2+ palavras) estiver contido no título. Sobrando nada, o heading fica vazio."""
    title = _normalize_title(document_title or "")
    segments = [part.strip() for part in (heading or "").split(HEADING_SEPARATOR.strip()) if part.strip()]
    if not title or not segments:
        return " > ".join(segments)

    def repeats_title(segment: str) -> bool:
        norm = _normalize_title(segment)
        if not norm:
            return False
        if norm == title:
            return True
        return (_is_specific(title) and title in norm) or (_is_specific(norm) and norm in title)

    return HEADING_SEPARATOR.join(segment for segment in segments if not repeats_title(segment))


def _to_chunk(row: dict, *, score: float = 0.0) -> RetrievedChunk:
    distance = row.get("distance")
    document_title = row.get("document__title") or ""
    return RetrievedChunk(
        chunk_id=row["id"],
        document_id=row["document_id"],
        heading=_clean_heading(row["heading"] or "", document_title),
        content=row["content"],
        score=score,
        distance=None if distance is None else float(distance),
        order=row["index"],
        document_title=document_title,
    )


# --- fusão -------------------------------------------------------------------------


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(id) = soma de 1/(k + posição) em cada ranking
    (posição começa em 1). Função pura: recebe listas de ids ordenadas do melhor ao
    pior e devolve {id: score}."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


# --- buscas ------------------------------------------------------------------------


def _embed_query(query: str) -> list[float]:
    vector = ai_providers.get_embedding_model().embed_query(ai_providers.embedding_text_for("query", query))
    ai_providers.validate_embedding_dimensions(vector)
    return vector


def _vector_candidates(queryset, query_vector, max_distance: float) -> list[dict]:
    return list(
        queryset.annotate(distance=CosineDistance("embedding", query_vector))
        .filter(distance__lte=max_distance)
        .order_by("distance")
        .values(*_FIELDS, "distance")[:CANDIDATES_PER_SIDE]
    )


def _relaxed_query_text(query: str) -> str:
    """"termo1 or termo2 or ..." (sintaxe websearch): perguntas em linguagem natural
    quase nunca têm TODOS os termos num único trecho. Stopwords saem sozinhas pela
    config 'portuguese'."""
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(query.lower()):
        seen.setdefault(word)
    return " or ".join(list(seen)[:MAX_RELAXED_TERMS])


def _text_search(queryset, query: str, query_vector) -> list[dict]:
    """Ranking textual (SearchRank). Primeiro a consulta estrita (todos os termos);
    se nada casar, a relaxada (qualquer termo)."""

    def run(text: str):
        search_query = SearchQuery(text, config=SEARCH_CONFIG, search_type="websearch")
        annotated = queryset.annotate(rank=SearchRank(F("search_vector"), search_query))
        if query_vector is not None:
            annotated = annotated.annotate(distance=CosineDistance("embedding", query_vector))
        fields = (*_FIELDS, "distance") if query_vector is not None else _FIELDS
        return list(
            annotated.filter(search_vector=search_query, rank__gt=0)
            .order_by("-rank", "id")
            .values(*fields)[:CANDIDATES_PER_SIDE]
        )

    rows = run(query)
    if rows:
        return rows
    relaxed = _relaxed_query_text(query)
    if not relaxed or relaxed == query:
        return []
    max_distance = settings.RAG_MAX_DISTANCE + RELAXED_TEXT_DISTANCE_MARGIN
    return [
        row
        for row in run(relaxed)
        if row.get("distance") is None or row["distance"] <= max_distance
    ]


def search(
    user,
    query: str,
    *,
    course_code: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    curriculum: bool = False,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """Trechos mais relevantes para `query` que o usuário pode ver.

    - `course_code` "cc"/"ads" filtra por curso; "ambos"/"indefinido"/None não filtra.
    - `curriculum=True` (pergunta de grade): top_k mínimo 8 e corte de distância +0,05,
      porque a grade se espalha por vários trechos, todos perto do limite.
    - `max_distance` sobrescreve `RAG_MAX_DISTANCE` (usado pelo pipeline legacy).
    - Com `RAG_HYBRID_ENABLED`, funde vetorial + textual por RRF; sem ela, só vetorial.
      Se o embedding da consulta falhar no modo híbrido, segue só com a busca textual."""
    query = (query or "").strip()
    if not query:
        return []

    if curriculum:
        top_k = max(top_k, CURRICULUM_TOP_K)
    if max_distance is None:
        max_distance = settings.RAG_MAX_DISTANCE + (CURRICULUM_EXTRA_DISTANCE if curriculum else 0)

    queryset = _scoped_queryset(user, course_code)
    hybrid = settings.RAG_HYBRID_ENABLED

    query_vector = None
    try:
        query_vector = _embed_query(query)
    except Exception:
        if not hybrid:
            raise
        logger.warning("Embedding da consulta falhou; usando só a busca textual.", exc_info=True)

    vector_rows = _vector_candidates(queryset, query_vector, max_distance) if query_vector is not None else []
    text_rows = _text_search(queryset, query, query_vector) if hybrid else []

    rows_by_id = {row["id"]: row for row in [*text_rows, *vector_rows]}  # vetorial vence (mesmos campos)
    scores = rrf_fuse([[r["id"] for r in vector_rows], [r["id"] for r in text_rows]])

    def sort_key(chunk_id: int):
        distance = rows_by_id[chunk_id].get("distance")
        return (-scores[chunk_id], distance if distance is not None else 1.0, chunk_id)

    ranked = [_to_chunk(rows_by_id[i], score=scores[i]) for i in sorted(scores, key=sort_key)]
    if settings.RAG_RERANK_ENABLED:
        ranked = rerank(query, ranked)
    return ranked[:top_k]


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Gancho de reranking (LLM barato ou cross-encoder), atrás de `RAG_RERANK_ENABLED`.
    Por enquanto é a identidade: só deve ganhar uma implementação se o eval mostrar
    ganho que justifique a latência/custo extra."""
    return chunks


def neighbors(user, chunk_id: int, window: int = 1) -> list[RetrievedChunk]:
    """Chunks do mesmo documento com `index` em [i-window, i+window] (inclui o próprio),
    em ordem de leitura. Respeita as permissões: chunk inacessível -> lista vazia."""
    base = get_accessible_chunks_queryset(user).filter(id=chunk_id).values("document_id", "index").first()
    if base is None:
        return []
    window = max(0, window)
    rows = (
        get_accessible_chunks_queryset(user)
        .filter(
            document_id=base["document_id"],
            index__gte=base["index"] - window,
            index__lte=base["index"] + window,
        )
        .order_by("index")
        .values(*_FIELDS)
    )
    return [_to_chunk({**row, "distance": None}) for row in rows]


# --- referências opacas ------------------------------------------------------------


class RefRegistry:
    """Referências opacas por resposta: "T1", "T2"... O LLM só enxerga a ref (e o
    título da seção); quem resolve ref -> chunk é o servidor. Chunks repetidos
    (mesmo `chunk_id`) reaproveitam a ref que já tinham."""

    def __init__(self) -> None:
        self._by_ref: dict[str, RetrievedChunk] = {}
        self._ref_by_chunk: dict[int, str] = {}

    def add(self, chunks) -> list[tuple[str, RetrievedChunk]]:
        """Registra os chunks e devolve (ref, chunk) para cada um, sem repetir
        chunk_id dentro da mesma chamada. Os já registrados antes mantêm a ref."""
        pairs: list[tuple[str, RetrievedChunk]] = []
        seen: set[int] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ref = self._ref_by_chunk.get(chunk.chunk_id)
            if ref is None:
                ref = f"T{len(self._by_ref) + 1}"
                self._by_ref[ref] = chunk
                self._ref_by_chunk[chunk.chunk_id] = ref
            pairs.append((ref, self._by_ref[ref]))
        return pairs

    def get(self, ref: str) -> RetrievedChunk | None:
        return self._by_ref.get((ref or "").strip().upper())

    @property
    def chunk_ids(self) -> list[int]:
        return list(self._ref_by_chunk)


def format_context(pairs: list[tuple[str, RetrievedChunk]]) -> str:
    """"[T1 · seção: <heading>]\\n<conteúdo>" separados por "\\n\\n---\\n\\n". O título
    do documento NUNCA entra; a ref existe só para o modelo se orientar e não deve
    ser citada na resposta (a guarda de saída a remove)."""
    parts = []
    for ref, chunk in pairs:
        heading = " ".join((getattr(chunk, "heading", "") or "").split())
        label = f"[{ref} · seção: {heading}]" if heading else f"[{ref}]"
        parts.append(f"{label}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)
