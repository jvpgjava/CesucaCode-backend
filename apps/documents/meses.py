"""Mês vigente em calendários: cada linha de atividade passa a levar o mês a que pertence.

O calendário acadêmico em PDF é uma sequência de blocos, um por mês: à esquerda o
mini-calendário com o rótulo "Janeiro 2026", à direita a tabela `DATAS | ATIVIDADE`
só com o dia ("02 a 31", "13 e 14"). O rótulo do mês fica DENTRO da figura do
mini-calendário (que o chunker serializa vazia) ou numa célula da tabela vizinha, quando
o Docling junta dois meses numa tabela só; a ordem de leitura também não ajuda (os
mini-calendários de julho a outubro são lidos antes das linhas de junho). Sem isso, o
chunk diz "10 a 30 | Rematrícula" e ninguém sabe de que mês.

Por isso o mês vem da POSIÇÃO na página, não da ordem de leitura: um rótulo de mês vale
para tudo o que está da altura dele para baixo, em qualquer coluna, até o próximo rótulo;
o que está acima do primeiro rótulo de uma página continua o mês da página anterior (bloco
que quebrou de página). Só liga com `MIN_LABELS` rótulos ou mais (um calendário de fato),
e só em documentos com posição (PDF).

Efeitos (ver `prepare_document`): tabelas ganham a coluna "MÊS" na frente, linha a linha
(o chunker repete o cabeçalho e cada linha vira um registro autocontido, mesmo numa
tabela que mistura dois meses); itens de texto soltos (linhas de tabela que o Docling não
reconheceu, legendas de feriados) recebem o mês em `dict[self_ref -> mês]`, que o
chunking usa no heading."""

import re
from dataclasses import dataclass

from docling_core.types.doc import BoundingBox, ContentLayer, TableCell
from docling_core.types.doc.document import DoclingDocument, TableItem, TextItem

MONTHS = (
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)
_WEEKDAY = r"(?:dom|seg|ter|qua|qui|sex|s[áa]b)\.?"
# O rótulo inteiro é "Mês [de] AAAA"; numa célula ele pode vir colado aos dias da semana
# do mini-calendário ("Dom Seg Ter Qua Qui Sex Sáb Maio 2026"). Frases que só citam um
# mês ("Recesso de julho de 2026") não casam.
LABEL_RE = re.compile(
    rf"^(?:{_WEEKDAY}\s+)*(?P<mes>{'|'.join(MONTHS)}|marco)(?:\s+de)?\s+(?P<ano>\d{{4}})$",
    re.IGNORECASE,
)
MIN_LABELS = 2
COLUMN_HEADER = "MÊS"


@dataclass(frozen=True, order=True)
class _Label:
    page: int
    top: float  # origem no topo da página: menor = mais alto
    text: str


def parse_label(text: str) -> str:
    """'Dom Seg ... Sáb Maio 2026' -> 'Maio 2026'; '' se o texto não é um rótulo de mês."""
    match = LABEL_RE.match(" ".join(text.split()))
    if match is None:
        return ""
    mes = match.group("mes").lower()
    return f"{('março' if mes == 'marco' else mes).capitalize()} {match.group('ano')}"


def _top_left(document: DoclingDocument, page_no: int, bbox: BoundingBox) -> BoundingBox | None:
    page = document.pages.get(page_no)
    return bbox.to_top_left_origin(page_height=page.size.height) if page is not None else None


def _labels(document: DoclingDocument) -> list[_Label]:
    """Rótulos de mês em qualquer texto (inclusive os de dentro de figuras e da camada
    que o chunker ignora) e em células de tabela, com a posição na página."""
    found: list[_Label] = []
    for item in document.texts:
        label = parse_label(item.text)
        for prov in item.prov[:1] if label else []:
            bbox = _top_left(document, prov.page_no, prov.bbox)
            if bbox is not None:
                found.append(_Label(prov.page_no, bbox.t, label))
    for table in document.tables:
        if not table.prov:
            continue
        page_no = table.prov[0].page_no
        for cell in table.data.table_cells:
            label = parse_label(cell.text)
            bbox = _top_left(document, page_no, cell.bbox) if label and cell.bbox else None
            if bbox is not None:
                found.append(_Label(page_no, bbox.t, label))
    return sorted(found)


def _month_at(labels: list[_Label], page_no: int, center: float) -> str:
    """Último rótulo, em ordem de página e altura, que começa acima do centro do trecho."""
    current = ""
    for label in labels:
        if (label.page, label.top) > (page_no, center):
            break
        current = label.text
    return current


def _center(bbox: BoundingBox) -> float:
    return (bbox.t + bbox.b) / 2


def _add_month_column(document: DoclingDocument, table: TableItem, labels: list[_Label]) -> bool:
    """Põe a coluna "MÊS" na frente da tabela, com o mês de cada linha pela altura das
    células dela. Linhas sem posição (ou acima do primeiro rótulo) ficam em branco."""
    if not table.prov:
        return False
    page_no = table.prov[0].page_no
    centers: dict[int, list[float]] = {}
    for cell in table.data.table_cells:
        bbox = _top_left(document, page_no, cell.bbox) if cell.bbox else None
        if bbox is not None:
            centers.setdefault(cell.start_row_offset_idx, []).append(_center(bbox))
    months = {
        row: _month_at(labels, page_no, sum(values) / len(values)) for row, values in centers.items()
    }
    if not any(months.values()):
        return False
    for cell in table.data.table_cells:
        cell.start_col_offset_idx += 1
        cell.end_col_offset_idx += 1
    header = any(c.column_header for c in table.data.table_cells if c.start_row_offset_idx == 0)
    for row in range(table.data.num_rows):
        # Tabela que continua da página anterior começa direto nos dados, sem cabeçalho.
        text = COLUMN_HEADER if row == 0 and header else months.get(row, "")
        table.data.table_cells.append(
            TableCell(
                text=text,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                column_header=row == 0 and header,
            )
        )
    table.data.num_cols += 1
    return True


def prepare_document(document: DoclingDocument) -> dict[str, str]:
    """Marca o mês de cada trecho de um calendário. Tabelas recebem a coluna "MÊS" (no
    lugar); devolve {self_ref -> mês} dos itens de texto do corpo. Sem rótulos suficientes
    (não é calendário), não mexe em nada e devolve {}."""
    labels = _labels(document)
    if len({label.text for label in labels}) < MIN_LABELS:
        return {}
    months: dict[str, str] = {}
    for item, _ in document.iterate_items(included_content_layers={ContentLayer.BODY}):
        if isinstance(item, TableItem):
            _add_month_column(document, item, labels)
        elif isinstance(item, TextItem) and item.prov:
            prov = item.prov[0]
            bbox = _top_left(document, prov.page_no, prov.bbox)
            month = _month_at(labels, prov.page_no, _center(bbox)) if bbox is not None else ""
            if month:
                months[item.self_ref] = month
    return months
