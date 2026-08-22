from dataclasses import dataclass

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200


@dataclass
class Chunk:
    content: str
    heading: str = ""


def chunk_text(
    text: str, *, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Divisão simples por parágrafo — usada só para TXT, que não tem estrutura
    (cabeçalhos, seções) para o Docling entender."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
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

    return [Chunk(content=c) for c in chunks]


def chunk_docling_document(
    document: DoclingDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Divisão com consciência de estrutura (cabeçalhos, seções, tabelas) via
    Docling — usada para PDF/DOCX/PPTX. Cada chunk carrega o caminho de
    seções (heading) a que pertence, ex.: "5. Modelo ER > 5.1 Entidades"."""
    chunks: list[Chunk] = []
    for doc_chunk in HierarchicalChunker().chunk(document):
        text = doc_chunk.text.strip()
        if not text:
            continue
        heading = " > ".join(doc_chunk.meta.headings) if doc_chunk.meta.headings else ""

        if len(text) <= max_chars:
            chunks.append(Chunk(content=text, heading=heading))
        else:
            chunks.extend(
                Chunk(content=piece, heading=heading)
                for piece in _split_long_text(text, max_chars, overlap)
            )

    return chunks


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(max_chars - overlap, 1)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]
