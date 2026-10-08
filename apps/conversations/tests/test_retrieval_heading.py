"""Heading do contexto nunca carrega o título do documento (ele não pode chegar ao LLM)."""

import pytest

from apps.conversations import retrieval
from apps.conversations.retrieval import _clean_heading, format_context

from .test_retrieval import admin, ads, embed, ids, make_doc, vec  # noqa: F401 (fixtures)


@pytest.mark.parametrize(
    ("heading", "title", "expected"),
    [
        # título repetido como primeiro item do breadcrumb
        ("Código Disciplinar Cesuca > Avaliação > Média", "Código Disciplinar Cesuca", "Avaliação > Média"),
        # título de arquivo (extensão e separadores) e diferença de acento/caixa
        ("codigo disciplinar cesuca", "codigo_disciplinar_cesuca.pdf", ""),
        # segmento contido no título (2+ palavras)
        ("Código Disciplinar > Faltas", "Código Disciplinar do Cesuca 2025", "Faltas"),
        # título contido no segmento
        ("Código Disciplinar Cesuca - Capítulo 2 > Faltas", "Código Disciplinar Cesuca", "Faltas"),
        # seção legítima de uma palavra, mesmo contida no título, é preservada
        ("Avaliação > Média", "Regras de Avaliação do Curso", "Avaliação > Média"),
        # sem título: nada muda; sem heading: vazio
        ("Avaliação > Média", "", "Avaliação > Média"),
        ("", "Qualquer Título Aqui", ""),
    ],
)
def test_clean_heading(heading, title, expected):
    assert _clean_heading(heading, title) == expected


def test_busca_devolve_heading_sem_o_titulo_do_documento(admin, ads, embed):  # noqa: F811
    make_doc(
        admin,
        "Manual do Estudante ADS",
        [ads],
        [("Manual do Estudante ADS > Avaliação > Média", "A média mínima é sete.", vec(1.0))],
    )

    result = retrieval.search(admin, "média mínima", course_code="ads")

    assert [c.heading for c in result] == ["Avaliação > Média"]
    assert result[0].document_title == "Manual do Estudante ADS"  # só para limpeza, nunca exibido
    context = format_context([("T1", result[0])])
    assert context.startswith("[T1 · seção: Avaliação > Média]") and "Manual do Estudante" not in context
    assert ids(result)
