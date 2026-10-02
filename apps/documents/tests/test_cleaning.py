"""Limpeza de boilerplate de página (rodapé, paginação, endereço, credenciamento)."""

import io

import pytest
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

from apps.documents import chunking, cleaning

PORTARIA_1 = (
    "Portaria Ministerial nº 655, de 12 de agosto de 2020, DOU nº 155, de 13 de agosto de 2020, "
    "seção 1, p. 54, prorrogado pela Portaria SERES/MEC nº 878, de 28 de"
)
PORTARIA_2 = "novembro de 2025, DOU nº 228, de 1 de dezembro de 2025, seção 1, p. 105-106."
ENDERECO = "Rua Silvério Manoel da Silva, 160 | 94940 243 Cachoeirinha RS | T 51 3396 1000"
RODAPE = ["www.cesuca.edu.br", ENDERECO, "Credenciamento Institucional", PORTARIA_1, PORTARIA_2]


@pytest.mark.parametrize(
    "linha",
    ["2 / 102", "1/9", "Página 3 de 10", "pág. 4/9", "www.cesuca.edu.br", "https://www.cesuca.edu.br/", ENDERECO, "Credenciamento Institucional",
     # rodapé de timbre de PDF escaneado (OCR) e linha de credenciamento de quadro
     "0 - Cachoeirinha / Rio Grande do Sul - RS - CEP: 94940-243", "~www.cesuca.edu.br(51)3396-1000",
     "Credenciado pela Portaria Ministerial nº 655 de 12/08/2020, DOU nº 155 de 13/08/2020, seção 1, p. 54."],
)
def test_linhas_de_rodape_sao_boilerplate(linha):
    assert cleaning.is_boilerplate_line(linha)


@pytest.mark.parametrize(
    "linha",
    [
        "Avaliação Regimental (A1) no valor de 0,0 a 5,0.",
        "A prova tem 2 / 3 das questões objetivas.",  # fração dentro de frase
        "O Cesuca fica em Cachoeirinha.",
        "Consulte www.cesuca.edu.br/matricula para o calendário completo.",
        "Em 2020, a Portaria Ministerial nº 655 autorizou o curso.",
        "A sede fica em Cachoeirinha / Rio Grande do Sul, junto à BR-290.",
    ],
)
def test_conteudo_real_nao_e_boilerplate_sozinho(linha):
    assert not cleaning.is_boilerplate_line(linha)
    assert cleaning.boilerplate_flags([linha]) == [False]


def test_bloco_de_credenciamento_inteiro_sai_junto_com_a_paginacao():
    textos = ["Texto da página.", *RODAPE, "1 / 102", "RECURSOS DISPONÍVEIS", "Conteúdo seguinte."]
    flags = cleaning.boilerplate_flags(textos)
    assert flags == [False, True, True, True, True, True, True, False, False]


def test_portaria_solta_fora_do_bloco_de_credenciamento_e_preservada():
    textos = ["Base legal", PORTARIA_1, PORTARIA_2, "Fim."]
    assert cleaning.boilerplate_flags(textos, repeated_footer=False) == [False] * 4


def test_bloco_de_credenciamento_nao_engole_o_conteudo_depois_dele():
    textos = ["Credenciamento Institucional", PORTARIA_2, "Texto real que continua.", "Mais texto."]
    assert cleaning.boilerplate_flags(textos) == [True, True, False, False]


def _paginas(n, rodape, extra_por_pagina=None):
    textos = []
    for page in range(1, n + 1):
        textos += [f"Conteúdo único da página {page} sobre um assunto diferente.", *(extra_por_pagina or {}).get(page, []), *rodape, f"{page} / {n}"]
    return textos


def test_rodape_repetido_sem_padrao_e_detectado_pela_heuristica():
    rodape = ["Instituto Fictício de Ensino - Setor de Registros"]
    textos = _paginas(8, rodape)

    flags = cleaning.boilerplate_flags(textos)

    assert [t for t, drop in zip(textos, flags) if drop and "Instituto" in t] == rodape * 8
    assert not any(drop for t, drop in zip(textos, flags) if t.startswith("Conteúdo único"))


def test_linha_repetida_em_poucas_paginas_nao_e_tratada_como_rodape():
    textos = _paginas(8, [], {1: ["Informação não disponível"], 2: ["Informação não disponível"]})
    flags = cleaning.boilerplate_flags(textos)
    assert [t for t, drop in zip(textos, flags) if drop and not cleaning.is_pagination(t)] == []


def test_heuristica_pode_ser_desligada():
    textos = _paginas(8, ["Instituto Fictício de Ensino - Setor de Registros"])
    flags = cleaning.boilerplate_flags(textos, repeated_footer=False)
    assert not any(drop for t, drop in zip(textos, flags) if "Instituto" in t)


def test_clean_text_para_txt():
    texto = "\n".join(["Linha boa.", "www.cesuca.edu.br", "3 / 10", "Outra linha boa."])
    assert cleaning.clean_text(texto) == "Linha boa.\nOutra linha boa."


def test_clean_heading_path_tira_segmentos_de_rodape():
    assert cleaning.clean_heading_path("Credenciamento Institucional") == ""
    assert cleaning.clean_heading_path("Plano > 2 / 102 > EMENTA") == "Plano > EMENTA"
    assert cleaning.clean_heading_path("Avaliação > Média") == "Avaliação > Média"


MD_PAGINADO = f"""
## EMENTA
Estudo dos conceitos de modelagem.

{RODAPE[0]}
{RODAPE[1]}
## Credenciamento Institucional
{PORTARIA_1}
{PORTARIA_2}
## 1 / 2

## AVALIAÇÃO
A Nota Final resulta da soma de A1 e A2.

{RODAPE[0]}
{RODAPE[1]}
## Credenciamento Institucional
{PORTARIA_1}
{PORTARIA_2}
## 2 / 2
"""


def test_chunks_do_markdown_do_docling_saem_sem_boilerplate():
    document = DocumentConverter().convert(DocumentStream(name="p.md", stream=io.BytesIO(MD_PAGINADO.encode()))).document

    chunks = chunking.chunk_docling_document(document)

    headings = {c.heading for c in chunks}
    assert headings == {"EMENTA", "AVALIAÇÃO"}
    texto = "\n".join(c.content for c in chunks)
    for ruido in ("cesuca.edu.br", "Portaria", "DOU", "Cachoeirinha", "/ 2"):
        assert ruido not in texto
    assert "Estudo dos conceitos de modelagem." in texto and "A Nota Final resulta" in texto


def test_chunk_text_limpa_rodape_de_txt():
    chunks = chunking.chunk_text("Parágrafo útil.\n\nwww.cesuca.edu.br\n1 / 3\n\nOutro parágrafo útil.")
    assert "cesuca" not in "".join(c.content for c in chunks)
    assert "Parágrafo útil." in chunks[0].content


def test_item_de_varias_linhas_do_ocr_tambem_e_limpo_no_chunk():
    md = (
        "## Art. 1\n\nO aluno deve respeitar o regimento.\n"
        "0 - Cachoeirinha / Rio Grande do Sul - RS - CEP: 94940-243\n"
        "~www.cesuca.edu.br(51)3396-1000\n"
        "Parágrafo único mantido.\n"
    )
    document = DocumentConverter().convert(DocumentStream(name="c.md", stream=io.BytesIO(md.encode()))).document

    chunks = chunking.chunk_docling_document(document)

    texto = "\n".join(c.content for c in chunks)
    assert "CEP" not in texto and "cesuca.edu.br" not in texto
    assert "O aluno deve respeitar o regimento." in texto and "Parágrafo único mantido." in texto
