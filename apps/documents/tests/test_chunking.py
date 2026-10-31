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


# ---------------------------------------------------------------- calendário: o mês de cada linha

from docling_core.types.doc import DocItemLabel  # noqa: E402

from apps.documents import meses  # noqa: E402


def _caixa(topo, esquerda=300, largura=250, altura=12):
    return BoundingBox(l=esquerda, t=topo, r=esquerda + largura, b=topo + altura, coord_origin=CoordOrigin.TOPLEFT)


def _na_pagina(pagina, topo, esquerda=300, altura=12):
    return ProvenanceItem(page_no=pagina, bbox=_caixa(topo, esquerda, altura=altura), charspan=(0, 0))


def _tabela_posicionada(linhas, topo, *, cabecalho=True):
    """Tabela com a posição de cada célula, uma linha a cada 15 pontos a partir de `topo`."""
    cells = [
        TableCell(text=texto, bbox=_caixa(topo + 15 * r, 300 + 100 * c, 90),
                  start_row_offset_idx=r, end_row_offset_idx=r + 1,
                  start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=cabecalho and r == 0)
        for r, linha in enumerate(linhas) for c, texto in enumerate(linha)
    ]
    return TableData(num_rows=len(linhas), num_cols=len(linhas[0]), table_cells=cells)


def _mini_calendario(doc, pagina, topo, rotulo):
    """O mini-calendário à esquerda é uma figura; o rótulo do mês é texto DENTRO dela,
    alinhado com o cabeçalho DATAS da tabela (é assim que o Docling lê o PDF real)."""
    figura = doc.add_picture(prov=_na_pagina(pagina, topo - 20, esquerda=50, altura=120))
    doc.add_text(label=DocItemLabel.TEXT, text="Dom Seg Ter Qua Qui Sex Sáb", parent=figura,
                 prov=_na_pagina(pagina, topo + 15, esquerda=50))
    doc.add_text(label=DocItemLabel.TEXT, text=rotulo, parent=figura, prov=_na_pagina(pagina, topo, esquerda=50))


def _calendario_pdf():
    """Mesma forma do calendário acadêmico real: um bloco por mês (mini-calendário à
    esquerda, tabela `DATAS | ATIVIDADE` só com o dia à direita), uma tabela que junta dois
    meses com o rótulo do segundo numa célula, e na página 2 linhas que o Docling não
    reconheceu como tabela (dia e atividade em textos soltos), lidas DEPOIS do
    mini-calendário do mês seguinte."""
    doc = _pdf("calendario")
    doc.add_page(page_no=2, size=Size(width=600, height=800))
    doc.add_heading("CALENDÁRIO ACADÊMICO 2026", level=1, prov=_na_pagina(1, 20))

    _mini_calendario(doc, 1, 100, "Janeiro 2026")
    doc.add_table(data=_tabela_posicionada(
        [["DATAS", "ATIVIDADE"], ["02 a 31", "Rematrícula Online 2026/1"], ["5", "Fim de Férias Coletivas"]], 100,
    ), prov=_na_pagina(1, 95, altura=60))
    doc.add_text(label=DocItemLabel.TEXT, text="1 - Ano Novo", prov=_na_pagina(1, 160, esquerda=50))

    # Maio e Junho numa tabela só: o rótulo de junho é uma célula da coluna do mini-calendário.
    doc.add_table(data=_tabela_posicionada([
        ["Dom Seg Ter Qua Qui Sex Sáb Maio 2026", "DATAS", "ATIVIDADE"],
        ["3 4 5 6 7 8 9", "12 a 14", "Semana Acadêmica"],
        ["", "29 e 30", "Aplicação da Avaliação Regimental (A1)"],
        ["Junho 2026", "DATAS", "ATIVIDADE"],
        ["7 8 9 10 11 12 13", "01 a 30", "Recebimento dos pedidos Externos"],
    ], 400), prov=_na_pagina(1, 395, altura=80))

    # Página 2: o mini-calendário de julho vem antes, na ordem de leitura, das últimas
    # linhas de junho, que estão no topo da página (acima dele) como textos soltos.
    _mini_calendario(doc, 2, 300, "Julho 2026")
    doc.add_text(label=DocItemLabel.TEXT, text="9 - Feriado de Julho", prov=_na_pagina(2, 430, esquerda=50))
    for topo, dia, atividade in [(40, "10 a 30", "Rematrícula Online 2026/2"), (55, "13 e 14", "Aplicação da Avaliação Final (AF)")]:
        doc.add_text(label=DocItemLabel.TEXT, text=dia, prov=_na_pagina(2, topo))
        doc.add_text(label=DocItemLabel.TEXT, text=atividade, prov=_na_pagina(2, topo, esquerda=340))
    doc.add_table(data=_tabela_posicionada([["DATAS", "ATIVIDADE"], ["1", "Aplicação da Avaliação Parcial (A2)"]], 300),
                  prov=_na_pagina(2, 295, altura=30))
    return doc


@pytest.fixture
def calendario():
    return chunking.chunk_docling_document(_calendario_pdf(), max_tokens=300)


def _linhas(chunks):
    return [line for c in chunks for line in c.content.splitlines()]


def test_cada_linha_de_tabela_do_calendario_leva_o_mes(calendario):
    linhas = _linhas(calendario)
    assert "| Janeiro 2026 | 02 a 31 | Rematrícula Online 2026/1 |" in linhas
    assert "| Janeiro 2026 | 5 | Fim de Férias Coletivas |" in linhas
    assert "| Julho 2026 | 1 | Aplicação da Avaliação Parcial (A2) |" in linhas
    assert all(c.content.startswith("| MÊS |") for c in calendario if c.content.startswith("|"))


def test_tabela_que_junta_dois_meses_troca_o_mes_na_linha_do_rotulo(calendario):
    linhas = _linhas(calendario)
    assert any(l.startswith("| Maio 2026 |") and "| 29 e 30 |" in l for l in linhas)
    assert any(l.startswith("| Junho 2026 |") and "| 01 a 30 |" in l for l in linhas)
    assert not any(l.startswith("| Maio 2026 |") and "| 01 a 30 |" in l for l in linhas)


def test_texto_solto_no_topo_da_pagina_continua_o_mes_da_pagina_anterior(calendario):
    # Lido depois do mini-calendário de julho, mas está acima dele: é junho.
    [rematricula] = [c for c in calendario if "Rematrícula Online 2026/2" in c.content]
    assert rematricula.heading.endswith("Junho 2026")
    assert "10 a 30" in rematricula.content
    assert "Julho" not in rematricula.content


def test_chunk_de_texto_que_atravessa_meses_vira_um_por_mes(calendario):
    # A legenda de julho e as linhas de junho são lidas em sequência (mesmo chunk do
    # Docling); cada mês vira um chunk com o seu mês no heading.
    [legenda] = [c for c in calendario if "9 - Feriado de Julho" in c.content]
    assert legenda.heading == "CALENDÁRIO ACADÊMICO 2026 > Julho 2026"
    assert "Rematrícula" not in legenda.content
    [ano_novo] = [c for c in calendario if "1 - Ano Novo" in c.content]
    assert ano_novo.heading.endswith("Janeiro 2026")


def test_documento_com_um_so_mes_citado_nao_vira_calendario():
    doc = _pdf("cronograma")
    doc.add_text(label=DocItemLabel.TEXT, text="Março 2026", prov=_na_pagina(1, 50))
    doc.add_table(data=_tabela_posicionada([["Aula", "Data"], ["1", "05/03"]], 70), prov=_na_pagina(1, 65, altura=30))
    assert meses.prepare_document(doc) == {}
    chunks = chunking.chunk_docling_document(doc, max_tokens=300)
    assert any("| 1 | 05/03 |" in c.content for c in chunks)
    assert not any("MÊS" in c.content or "Março" in c.heading for c in chunks)


@pytest.mark.parametrize("texto,rotulo", [
    ("Janeiro 2026", "Janeiro 2026"),
    ("MARÇO DE 2026", "Março 2026"),
    ("Dom Seg Ter Qua Qui Sex Sáb Maio 2026", "Maio 2026"),
    ("Recesso de julho de 2026", ""),
    ("02 a 31", ""),
])
def test_rotulo_de_mes(texto, rotulo):
    assert meses.parse_label(texto) == rotulo
