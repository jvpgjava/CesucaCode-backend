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
