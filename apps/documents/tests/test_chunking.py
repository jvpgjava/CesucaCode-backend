import io

import pytest
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

from apps.documents import chunking

TABELA = (
    "| Dia | Hora | Disciplina |\n|---|---|---|\n"
    + "\n".join(f"| Seg{i} | 19:00 | Disciplina número {i} com nome longo de teste |" for i in range(60))
)
MD = f"# Plano\n\n## Horário\n\n{TABELA}\n\n## Avaliação\n\n" + ("A média final deve ser sete. " * 80)


@pytest.fixture(scope="module")
def chunks():
    doc = DocumentConverter().convert(DocumentStream(name="a.md", stream=io.BytesIO(MD.encode()))).document
    return chunking.chunk_docling_document(doc, max_tokens=300)


def test_tabela_grande_repete_cabecalho_e_nao_parte_linhas(chunks):
    tabela = [c for c in chunks if c.heading.endswith("Horário")]
    assert len(tabela) > 1
    for c in tabela:
        assert c.content.startswith("| Dia | Hora | Disciplina |")
        # Toda linha termina inteira: nenhuma linha de dados foi cortada no meio.
        for line in c.content.splitlines():
            assert line.startswith("|") and line.endswith("|")
    todas = "\n".join(c.content for c in tabela)
    assert all(f"| Seg{i} |" in todas for i in range(60))


def test_respeita_limite_de_tokens_e_guarda_secao(chunks):
    tokenizer = chunking.ApproxTokenizer(max_tokens=300)
    assert all(tokenizer.count_tokens(c.content) <= 300 for c in chunks)
    assert {c.heading for c in chunks} == {"Plano > Horário", "Plano > Avaliação"}
    assert len([c for c in chunks if c.heading.endswith("Avaliação")]) >= 2


def test_txt_simples_continua_funcionando():
    result = chunking.chunk_text("Um parágrafo.\n\nOutro parágrafo.")
    assert [c.content for c in result] == ["Um parágrafo.\n\nOutro parágrafo."]


# ---------------------------------------------------------------- título e legenda (cronogramas em PDF)

from docling_core.types.doc import (  # noqa: E402
    BoundingBox,
    CoordOrigin,
    DocumentOrigin,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import DoclingDocument  # noqa: E402

TITULO = "CRONOGRAMA - PROJ. E DESENV. DE SISTEMAS MÓVEIS"


def _tabela(linhas):
    cells = [
        TableCell(text=texto, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                  start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=r == 0)
        for r, linha in enumerate(linhas) for c, texto in enumerate(linha)
    ]
    return TableData(num_rows=len(linhas), num_cols=len(linhas[0]), table_cells=cells)


def _em(topo, altura=20):
    """Posição na página 1 (origem no topo): quanto menor `topo`, mais alto na página."""
    bbox = BoundingBox(l=50, t=topo, r=550, b=topo + altura, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 0))


def _pdf(nome="cronograma", *, pdf=True):
    doc = DoclingDocument(name=nome)
    doc.origin = DocumentOrigin(
        mimetype="application/pdf" if pdf else "text/markdown", binary_hash=1, filename="x"
    )
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    return doc


def _cronograma_pdf(*, titulo_lido_por_ultimo=False, pdf=True):
    """Mesma forma dos PDFs reais: logo, título e "Legenda" como cabeçalhos do MESMO
    nível, a legenda de cores como tabela só de cabeçalho e a tabela de aulas. No
    cronograma de IHC o Docling lê o título por último, embora ele esteja no topo."""
    doc = _pdf(pdf=pdf)
    doc.add_picture(prov=_em(10, 60))
    if not titulo_lido_por_ultimo:
        doc.add_heading(TITULO, level=1, prov=_em(90))
    doc.add_heading("Legenda", level=1, prov=_em(130))
    doc.add_table(data=_tabela([["Feriado", "Prova", "Trabalho", "AF"]]), prov=_em(150))
    doc.add_table(data=_tabela([["Aula", "Data", "Conteúdo"], ["9", "05/10", "Avaliação A1 - 3,5"]]), prov=_em(200, 400))
    if titulo_lido_por_ultimo:
        doc.add_heading(TITULO, level=1, prov=_em(90))
    return doc


@pytest.mark.parametrize("titulo_lido_por_ultimo", [False, True])
def test_titulo_do_pdf_vira_raiz_do_heading_e_legenda_some(titulo_lido_por_ultimo):
    chunks = chunking.chunk_docling_document(_cronograma_pdf(titulo_lido_por_ultimo=titulo_lido_por_ultimo))
    assert chunks
    for c in chunks:
        assert c.heading == TITULO
        assert "Feriado | Prova" not in c.content
    assert any("| 9 | 05/10 | Avaliação A1 - 3,5 |" in c.content for c in chunks)


def test_document_title_usa_posicao_na_pagina_e_ignora_logo():
    assert chunking.document_title(_cronograma_pdf()) == TITULO
    assert chunking.document_title(_cronograma_pdf(titulo_lido_por_ultimo=True)) == TITULO


def test_cabecalho_seguido_de_conteudo_nao_e_titulo():
    doc = _pdf("horario")
    doc.add_heading("SEGUNDA-FEIRA", level=1, prov=_em(50))
    doc.add_table(data=_tabela([["Hora", "Disciplina"], ["19:00", "IHC"]]), prov=_em(80, 200))
    doc.add_heading("TERÇA-FEIRA", level=1, prov=_em(300))
    assert chunking.document_title(doc) == ""


def test_document_title_so_para_pdf():
    assert chunking.document_title(_cronograma_pdf(pdf=False)) == ""
