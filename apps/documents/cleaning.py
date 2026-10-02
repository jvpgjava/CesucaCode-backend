"""Limpeza de boilerplate de página na ingestão (rodapés, paginação, endereço).

PDFs exportados para Markdown (como o seed dos planos de ensino) repetem em TODA
página um rodapé: site, endereço/telefone, "Credenciamento Institucional" + parágrafo
da Portaria e a paginação "N / M". Sem limpeza, isso vira heading (102 chunks com o
heading "Credenciamento Institucional") e conteúdo de chunk, e esconde o nome real da
seção/disciplina. Aqui ficam as regras, em dois mecanismos:

1. padrões declarados abaixo (linhas isoladas e blocos "cabeçalho de rodapé + corpo");
2. heurística opcional de "linha repetida na página": linhas logo antes de um marcador
   de paginação que se repetem na maioria das páginas são rodapé, mesmo sem padrão.

Tudo trabalha sobre uma lista de textos (itens do Docling ou linhas de um TXT) e devolve
flags por posição (`boilerplate_flags`), então o mesmo código serve aos dois formatos.
A remoção nunca apaga conteúdo "no escuro": os padrões são ancorados na linha inteira,
e os blocos da Portaria só são removidos logo depois do seu cabeçalho.
"""

import re
import unicodedata
from collections import Counter

from docling_core.types.doc import ContentLayer
from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TextItem

# Marcador de paginação isolado: "2 / 102", "Página 3 de 10", "pág. 4/9".
PAGINATION_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
    re.compile(r"^\s*p[áa]g(?:ina)?\.?\s*\d{1,4}\s*(?:de|/)\s*\d{1,4}\s*$", re.IGNORECASE),
]

# Linhas de rodapé isoladas (a linha inteira deve casar).
BOILERPLATE_PATTERNS = [
    *PAGINATION_PATTERNS,
    re.compile(r"^\s*(?:https?://)?(?:www\.)?cesuca\.edu\.br/?\s*$", re.IGNORECASE),
    # Endereço + telefone: "Rua X, 160 | 94940 243 Cidade UF | T 51 3396 1000"
    re.compile(r"^.{0,120}\b\d{5}[-\s]?\d{3}\b.{0,60}\|\s*T\s*\(?\d{2}\)?[\s\d.-]{8,}\s*$"),
    re.compile(r"^\s*Rua\s+Silv[ée]rio\s+Manoel\s+da\s+Silva\b.*$", re.IGNORECASE),
    # Rodapé de timbre do OCR: "...Cachoeirinha / Rio Grande do Sul - RS - CEP: 94940-243" e
    # "~www.cesuca.edu.br(51)3396-1000" (site e telefone colados).
    re.compile(r"^.{0,60}Cachoeirinha\s*/\s*Rio\s+Grande\s+do\s+Sul\b.{0,20}CEP\s*:?\s*\d{5}-?\d{3}\s*$", re.IGNORECASE),
    re.compile(r"^\W{0,3}(?:https?://)?(?:www\.)?cesuca\.edu\.br\s*\(?\d{2}\)?[\s\d.-]{8,}$", re.IGNORECASE),
    # Linha de credenciamento no rodapé de um quadro: "Credenciado pela Portaria ... DOU ..."
    re.compile(r"^\s*Credenciad[oa]\s+pela\s+Portaria\b.*\bDOU\b.*$", re.IGNORECASE),
]

# Bloco de credenciamento: o cabeçalho abre o bloco e as linhas seguintes que casam com
# o corpo (Portaria/DOU) fazem parte dele. Um parágrafo de Portaria SOLTO, fora desse
# bloco, é conteúdo legítimo (ex.: um regulamento que cita uma Portaria) e fica.
BLOCK_HEADER_PATTERNS = [re.compile(r"^\s*Credenciamento\s+Institucional\s*$", re.IGNORECASE)]
BLOCK_BODY_PATTERNS = [
    re.compile(r"^\s*Portaria\s+(?:Ministerial|SERES|MEC|Normativa)\b", re.IGNORECASE),
    re.compile(r"\bDOU\s+n[ºo°]\s*\d+", re.IGNORECASE),
    re.compile(r"\bse[çc][ãa]o\s+\d+,\s*p\.\s*[\d-]+", re.IGNORECASE),
]
MAX_BLOCK_BODY_LINES = 4

# Heurística de rodapé repetido: janela de linhas antes do marcador de paginação e
# fração mínima das páginas em que a linha precisa aparecer ali.
REPEATED_FOOTER_WINDOW = 3
REPEATED_FOOTER_MIN_REPEATS = 5
REPEATED_FOOTER_MIN_PAGE_FRACTION = 0.5
REPEATED_FOOTER_MIN_CHARS = 8


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


def _matches(text: str, patterns) -> bool:
    return any(pattern.search(text or "") for pattern in patterns)


def is_pagination(text: str) -> bool:
    return _matches(text, PAGINATION_PATTERNS)


def is_boilerplate_line(text: str) -> bool:
    """A linha, sozinha, é rodapé/paginação? (Não vê o contexto: blocos de Portaria
    e a heurística de repetição ficam em `boilerplate_flags`.)"""
    return _matches(text, BOILERPLATE_PATTERNS) or _matches(text, BLOCK_HEADER_PATTERNS)


def boilerplate_flags(
    texts: list[str],
    *,
    repeated_footer: bool = True,
    min_repeats: int = REPEATED_FOOTER_MIN_REPEATS,
    min_page_fraction: float = REPEATED_FOOTER_MIN_PAGE_FRACTION,
) -> list[bool]:
    """Para cada texto, True se for boilerplate de página. Função pura."""
    flags = [_matches(t, BOILERPLATE_PATTERNS) for t in texts]

    # Blocos "Credenciamento Institucional" + corpo da Portaria.
    index = 0
    while index < len(texts):
        if _matches(texts[index], BLOCK_HEADER_PATTERNS):
            flags[index] = True
            cursor = index + 1
            while (
                cursor < len(texts)
                and cursor - index <= MAX_BLOCK_BODY_LINES
                and _matches(texts[cursor], BLOCK_BODY_PATTERNS)
            ):
                flags[cursor] = True
                cursor += 1
            index = cursor
        else:
            index += 1

    if repeated_footer:
        _flag_repeated_footers(texts, flags, min_repeats, min_page_fraction)
    return flags


def _flag_repeated_footers(texts: list[str], flags: list[bool], min_repeats: int, min_page_fraction: float) -> None:
    """Linhas que, em boa parte das páginas, vêm logo antes do marcador de paginação."""
    markers = [i for i, text in enumerate(texts) if is_pagination(text)]
    if len(markers) < min_repeats:
        return
    candidates: Counter[str] = Counter()
    positions: dict[str, list[int]] = {}
    for marker in markers:
        seen_here: set[str] = set()
        cursor = marker - 1
        examined = 0
        while cursor >= 0 and examined < REPEATED_FOOTER_WINDOW:
            if not flags[cursor]:
                examined += 1
                key = _normalize(texts[cursor])
                if len(key) >= REPEATED_FOOTER_MIN_CHARS and key not in seen_here:
                    seen_here.add(key)
                    candidates[key] += 1
                    positions.setdefault(key, []).append(cursor)
            cursor -= 1
    needed = max(min_repeats, int(len(markers) * min_page_fraction))
    for key, count in candidates.items():
        if count >= needed:
            for position in positions[key]:
                flags[position] = True


def clean_text(text: str, *, repeated_footer: bool = True) -> str:
    """Tira o boilerplate de um texto corrido (TXT) linha a linha."""
    lines = text.split("\n")
    flags = boilerplate_flags(lines, repeated_footer=repeated_footer)
    return "\n".join(line for line, drop in zip(lines, flags) if not drop)


def is_boilerplate_heading(heading: str) -> bool:
    return is_boilerplate_line(heading)


def clean_heading_path(heading: str, separator: str = ">") -> str:
    """Remove de um breadcrumb "A > B" os segmentos que são boilerplate."""
    segments = [part.strip() for part in (heading or "").split(separator) if part.strip()]
    return f" {separator} ".join(s for s in segments if not is_boilerplate_heading(s))


def mark_boilerplate_items(document: DoclingDocument, *, repeated_footer: bool = True) -> int:
    """Move para a camada FURNITURE (que o chunker ignora) os itens de texto/cabeçalho
    que são boilerplate de página. Devolve quantos itens foram marcados. Não mexe na
    estrutura da árvore: só na camada, então é seguro e reversível."""
    items = [item for item, _ in document.iterate_items() if isinstance(item, (TextItem, SectionHeaderItem))]
    flags = boilerplate_flags([item.text for item in items], repeated_footer=repeated_footer)
    marked = 0
    for item, drop in zip(items, flags):
        if drop:
            item.content_layer = ContentLayer.FURNITURE
            marked += 1
    return marked
