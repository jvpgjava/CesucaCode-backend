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

from docling_core.types.doc import DocumentOrigin, TableCell, TableData  # noqa: E402
from docling_core.types.doc.document import DoclingDocument  # noqa: E402


def _tabela(linhas):
    cells = [
        TableCell(text=texto, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                  start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=r == 0)
        for r, linha in enumerate(linhas) for c, texto in enumerate(linha)
    ]
    return TableData(num_rows=len(linhas), num_cols=len(linhas[0]), table_cells=cells)


def _cronograma_pdf(titulo="CRONOGRAMA - PROJ. E DESENV. DE SISTEMAS MÓVEIS", *, pdf=True):
    """Mesma forma do PDF real: logo, título e "Legenda" como cabeçalhos do MESMO nível,
    a legenda de cores como tabela só de cabeçalho e a tabela de aulas."""
    doc = DoclingDocument(name="cronograma")
    doc.origin = DocumentOrigin(
        mimetype="application/pdf" if pdf else "text/markdown", binary_hash=1, filename="x"
    )
    doc.add_picture()
    doc.add_heading(titulo, level=1)
    doc.add_heading("Legenda", level=1)
    doc.add_table(data=_tabela([["Feriado", "Prova", "Trabalho", "AF"]]))
    doc.add_table(data=_tabela([["Aula", "Data", "Conteúdo"], ["9", "05/10", "Avaliação A1 - 3,5"]]))
    return doc


def test_titulo_do_pdf_vira_raiz_do_heading_e_legenda_some():
    chunks = chunking.chunk_docling_document(_cronograma_pdf())
    assert chunks
    for c in chunks:
        assert c.heading == "CRONOGRAMA - PROJ. E DESENV. DE SISTEMAS MÓVEIS"
        assert "Legenda" not in c.heading and "Feriado | Prova" not in c.content
    assert any("| 9 | 05/10 | Avaliação A1 - 3,5 |" in c.content for c in chunks)


def test_document_title_ignora_logo_e_exige_bloco_de_titulo():
    assert chunking.document_title(_cronograma_pdf()) == "CRONOGRAMA - PROJ. E DESENV. DE SISTEMAS MÓVEIS"
    # Cabeçalho seguido de conteúdo é só a primeira seção, não título.
    doc = DoclingDocument(name="horario")
    doc.origin = DocumentOrigin(mimetype="application/pdf", binary_hash=1, filename="x")
    doc.add_heading("SEGUNDA-FEIRA", level=1)
    doc.add_table(data=_tabela([["Hora", "Disciplina"], ["19:00", "IHC"]]))
    doc.add_heading("TERÇA-FEIRA", level=1)
    assert chunking.document_title(doc) == ""


def test_document_title_so_para_pdf():
    assert chunking.document_title(_cronograma_pdf(pdf=False)) == ""
