"""Divisão dos materiais em chunks.

Decisão: usamos o `HybridChunker` do Docling (respeita a estrutura do documento,
limita por tokens e repete o cabeçalho de uma tabela em cada pedaço dela), mas com
um tokenizador LEVE e local (`ApproxTokenizer`: conta palavras e pontuação) em vez
do padrão do Docling, que baixaria um modelo do HuggingFace (sentence-transformers)
na primeira execução — dependência de rede/disco que não queremos na ingestão. A
contagem aproximada é suficiente: o objetivo é só manter chunks de tamanho
parecido, não casar com o limite de contexto de um modelo específico. Texto de
PT-BR vira ~1,3 token real por "token aproximado", e `RAG_CHUNK_MAX_TOKENS=300`
fica na faixa dos ~1500 caracteres usados antes.

Mudar o chunking ou `RAG_CHUNK_MAX_TOKENS` exige `manage.py reprocess_documents`.
"""

import re
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer, BaseSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc import ContentLayer
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from . import cleaning, disciplinas, meses

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MAX_TITLE_CHARS = 150


@dataclass
class Chunk:
    content: str
    heading: str = ""
    # Plano/disciplina vigente (contexto "pegajoso", ver disciplinas.py): o nome já está
    # no `heading`; o rótulo completo (semestre, C/H) entra só no texto do embedding.
    discipline: "disciplinas.DisciplineContext | None" = None


class ApproxTokenizer(BaseTokenizer):
    """Contador de tokens sem modelo: cada palavra e cada sinal de pontuação conta 1.
    Também serve de "tokenizador" para o semchunk, que aceita um contador de tokens."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


class _MarkdownTableSerializerProvider(BaseSerializerProvider):
    """Serializa tabelas como Markdown (uma linha por linha da tabela) em vez do
    formato "triplas" padrão do chunker. Assim o HybridChunker parte tabelas grandes
    SÓ entre linhas e repete o cabeçalho (colunas) no topo de cada pedaço — grade
    curricular e horários continuam interpretáveis em qualquer chunk."""

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
                compact_tables=True,  # sem o preenchimento de espaços das colunas
            ),
        )


def chunk_text(
    text: str, *, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Divisão simples por parágrafo — usada só para TXT, que não tem estrutura
    (cabeçalhos, seções) para o Docling entender."""
    paragraphs = [p.strip() for p in cleaning.clean_text(text).split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    result = [Chunk(content=c) for c in chunks]
    disciplinas.contexts_for_text_chunks(result)
    return result


def chunk_docling_document(
    document: DoclingDocument, *, max_tokens: int | None = None
) -> list[Chunk]:
    """Divisão com consciência de estrutura (cabeçalhos, seções, tabelas) via
    `HybridChunker` — usada para PDF/DOCX/PPTX/MD. Cada chunk carrega o caminho de
    seções (heading) a que pertence, ex.: "5. Modelo ER > 5.1 Entidades". Uma
    tabela maior que o limite é partida por linhas, repetindo o cabeçalho dela em
    cada pedaço (grade e horário continuam legíveis em qualquer chunk)."""
    tokenizer = ApproxTokenizer(max_tokens=max_tokens or settings.RAG_CHUNK_MAX_TOKENS)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        repeat_table_header=True,
        serializer_provider=_MarkdownTableSerializerProvider(),
    )

    # Antes do chunker: rodapé/paginação saem do texto e os cabeçalhos de plano de ensino
    # viram fronteira de seção, com o contexto da disciplina por item.
    title = document_title(document)  # antes da limpeza, que esconde cabeçalhos como "Legenda"
    cleaning.mark_boilerplate_items(document)
    cleaning.mark_header_only_tables(document)
    contexts = disciplinas.prepare_document(document)
    months = meses.prepare_document(document)  # calendário: coluna "MÊS" nas tabelas

    chunks: list[Chunk] = []
    for doc_chunk in chunker.chunk(document):
        headings = doc_chunk.meta.headings or []
        items = list(doc_chunk.meta.doc_items or [])
        for text, items, month in _split_by_month(document, doc_chunk.text, items, months):
            # Rede de segurança: itens com várias linhas (OCR) escapam da checagem por item.
            text = cleaning.clean_text(text, repeated_footer=False).strip()
            if text:
                chunks.append(_build_chunk(text, items, headings, title=title, contexts=contexts, month=month))
    return chunks


def _split_by_month(
    document: DoclingDocument, text: str, items: list, months: dict[str, str]
) -> list[tuple[str, list, str]]:
    """Calendário (ver meses.py): um chunk que atravessa meses vira um pedaço por mês, com
    o mês no heading. O chunker junta itens pequenos vizinhos (a legenda de um mês, as
    linhas soltas de outro, uma tabela curta), então o corte é no texto do chunk, onde
    começa cada item. Tabela vira pedaço próprio e sem mês no heading: as linhas dela já
    levam o mês na coluna "MÊS"."""
    if not months or not items:
        return [(text, items, "")]
    # O meta do chunk guarda cópias genéricas (`DocItem`, sem texto): o item de verdade
    # vem do documento.
    resolved = [RefItem(cref=i.self_ref).resolve(document) for i in items]
    runs: list[tuple[str | None, list, int]] = []  # (mês ou None p/ tabela, itens, início)
    cursor = 0
    for item in resolved:
        if isinstance(item, TableItem):
            key, start = None, cursor  # o texto da tabela vem logo depois do item anterior
        elif isinstance(item, TextItem) and item.text:
            start = text.find(item.text, cursor)
            if start < 0:  # texto reformatado pelo serializador: não arrisca o corte
                return [(text, items, "")]
            key, cursor = months.get(item.self_ref, ""), start + len(item.text)
        else:  # figura vazia: fica no pedaço corrente
            key, start = (runs[-1][0] if runs else ""), cursor
        if runs and runs[-1][0] == key:
            runs[-1][1].append(item)
        else:
            runs.append((key, [item], start))
    bounds = [0, *[start for _, _, start in runs[1:]], len(text)]
    return [
        (text[begin:end], run_items, key or "")
        for (key, run_items, _), begin, end in zip(runs, bounds, bounds[1:])
    ]


def _build_chunk(text: str, items: list, headings: list[str], *, title: str, contexts, month: str) -> Chunk:
    ctx = contexts.get(items[0].self_ref) if items else None
    headings = [h for h in headings if not cleaning.is_boilerplate_heading(h)]
    if ctx is not None:
        headings = [ctx.nome, *[h for h in headings if h != ctx.header_text and h != ctx.nome]]
        if items and all(i.self_ref in ctx.member_refs for i in items):
            text = ctx.summary()  # o texto grudado do cabeçalho vira linhas legíveis
    if title and title not in headings:
        headings = [title, *headings]
    if month and month not in headings:
        headings = [*headings, month]
    return Chunk(content=text, heading=" > ".join(headings), discipline=ctx)


def document_title(document: DoclingDocument) -> str:
    """Título impresso no documento, que vira a raiz do caminho de seções de TODO chunk.

    Sem isso, num PDF em que o título e as seções são cabeçalhos do mesmo nível
    (cronogramas: título e "Legenda"), a primeira seção apaga o título da hierarquia e
    os chunks de documentos parecidos (o cronograma de cada disciplina) ficam
    indistinguíveis para o LLM. É o texto do próprio documento, nunca o nome do arquivo.

    Usa a posição na PRIMEIRA PÁGINA, não a ordem de leitura, que às vezes sai trocada
    (o título lido depois da tabela). Conta como título um `TitleItem` no topo, ou um
    cabeçalho no topo com outro cabeçalho logo abaixo (bloco de título antes da primeira
    seção); cabeçalho seguido de conteúdo é só a primeira seção. Figuras (o logo) não
    contam. Só para PDF: em Markdown/DOCX a hierarquia já vem explícita no arquivo."""
    origin = getattr(document, "origin", None)
    if origin is None or origin.mimetype != "application/pdf":
        return ""
    page = document.pages.get(1)
    if page is None:
        return ""
    placed = []
    for item, _ in document.iterate_items():
        if item.content_layer != ContentLayer.BODY or isinstance(item, PictureItem):
            continue
        if isinstance(item, TextItem) and not item.text.strip():
            continue
        prov = next((p for p in item.prov if p.page_no == 1), None)
        if prov is not None:
            top = prov.bbox.to_top_left_origin(page_height=page.size.height).t
            placed.append((top, item))
    placed.sort(key=lambda pair: pair[0])
    if not placed or not isinstance(placed[0][1], (TitleItem, SectionHeaderItem)):
        return ""
    first = placed[0][1]
    title = " ".join(first.text.split())
    if len(title) > MAX_TITLE_CHARS:
        return ""
    if isinstance(first, TitleItem):
        return title
    following = placed[1][1] if len(placed) > 1 else None
    return title if isinstance(following, SectionHeaderItem) else ""

def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(max_chars - overlap, 1)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]
