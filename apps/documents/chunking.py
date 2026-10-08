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
from docling_core.types.doc.document import DoclingDocument

from . import cleaning, disciplinas

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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
    cleaning.mark_boilerplate_items(document)
    contexts = disciplinas.prepare_document(document)

    chunks: list[Chunk] = []
    for doc_chunk in chunker.chunk(document):
        # Rede de segurança: itens com várias linhas (OCR) escapam da checagem por item.
        text = cleaning.clean_text(doc_chunk.text, repeated_footer=False).strip()
        if not text:
            continue
        items = list(doc_chunk.meta.doc_items or [])
        ctx = contexts.get(items[0].self_ref) if items else None
        headings = [h for h in (doc_chunk.meta.headings or []) if not cleaning.is_boilerplate_heading(h)]
        if ctx is not None:
            headings = [ctx.nome, *[h for h in headings if h != ctx.header_text and h != ctx.nome]]
            if items and all(i.self_ref in ctx.member_refs for i in items):
                text = ctx.summary()  # o texto grudado do cabeçalho vira linhas legíveis
        chunks.append(Chunk(content=text, heading=" > ".join(headings), discipline=ctx))
    return chunks


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(max_chars - overlap, 1)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]
