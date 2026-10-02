from types import SimpleNamespace

import pytest

from apps.conversations import tools as tools_module
from apps.conversations.tests.agent_fakes import (
    FakeChunk,
    FakeRegistry,
    install_retrieval_stub,
    make_retrieval_stub,
)
from apps.conversations.tracing import TraceRecorder

TITULO = "Apostila-Secreta-de-Algoritmos.pdf"
USER = SimpleNamespace(role="cs_student", course=None)


def chunk(chunk_id, content="conteúdo do trecho", heading="Recursão", **kwargs):
    return FakeChunk(chunk_id=chunk_id, document_id=7, heading=heading, content=content, **kwargs)


def by_name(tools):
    return {tool.name: tool for tool in tools}


@pytest.fixture
def make(monkeypatch):
    def factory(search_results=None, neighbor_results=None, *, allow_web=False, recorder=None, course_code="cc"):
        stub = make_retrieval_stub(search_results, neighbor_results)
        install_retrieval_stub(monkeypatch, stub)
        registry = FakeRegistry()
        tools = tools_module.build_tools(
            user=USER, course_code=course_code, registry=registry, recorder=recorder, allow_web=allow_web
        )
        return by_name(tools), registry, stub

    return factory


def test_buscar_materiais_formata_trechos_com_ref_e_secao(make):
    tools, registry, stub = make([[chunk(1, "A recursão chama a si mesma."), chunk(2, "Caso base.", heading="")]])

    out = tools["buscar_materiais"].invoke({"consulta": "recursão"})

    assert out.startswith("[T1] (seção: Recursão)\nA recursão chama a si mesma.")
    assert "[T2]\nCaso base." in out
    assert registry.chunk_ids == [1, 2]
    assert stub.search_calls == [{"query": "recursão", "course_code": "cc", "top_k": 6}]


def test_buscar_materiais_nunca_expoe_titulo_de_documento(make):
    # Mesmo que o objeto do retrieval carregue um título, a saída não o traz.
    c = chunk(1)
    c.title = TITULO
    c.document_title = TITULO
    tools, _, _ = make([[c]])

    out = tools["buscar_materiais"].invoke({"consulta": "x"})

    assert TITULO not in out
    assert "Apostila" not in out


def test_buscar_materiais_trunca_trecho_longo(make):
    tools, _, _ = make([[chunk(1, "palavra " * 500)]])

    out = tools["buscar_materiais"].invoke({"consulta": "x"})

    body = out.split("\n", 1)[1]
    assert len(body) <= 905
    assert body.endswith("…")


def test_buscar_materiais_sem_resultado(make):
    tools, registry, _ = make([[]])

    out = tools["buscar_materiais"].invoke({"consulta": "x"})

    assert out == "Nenhum trecho relevante encontrado. Tente reformular a consulta com outros termos."
    assert registry.chunk_ids == []


def test_buscar_materiais_curso_do_modelo_e_ambos(make):
    tools, _, stub = make([[chunk(1)]])

    tools["buscar_materiais"].invoke({"consulta": "a", "curso": "ads"})
    tools["buscar_materiais"].invoke({"consulta": "b", "curso": "ambos"})
    tools["buscar_materiais"].invoke({"consulta": "c"})

    assert [c["course_code"] for c in stub.search_calls] == ["ads", None, "cc"]


def test_buscar_materiais_repetido_reaproveita_ref(make):
    tools, registry, _ = make([[chunk(1)]])  # a mesma busca devolve o mesmo trecho

    first = tools["buscar_materiais"].invoke({"consulta": "a"})
    second = tools["buscar_materiais"].invoke({"consulta": "b"})

    assert first.startswith("[T1]") and second.startswith("[T1]")
    assert registry.chunk_ids == [1]


def test_ler_contexto_le_vizinhos_e_registra_novos(make):
    vizinhos = [chunk(1, "antes", heading="A"), chunk(2, "centro", heading="A"), chunk(3, "depois", heading="A")]
    tools, registry, stub = make([[chunk(2, "centro", heading="A")]], vizinhos)
    tools["buscar_materiais"].invoke({"consulta": "x"})

    out = tools["ler_contexto"].invoke({"ref": "T1", "vizinhos": 5})

    assert stub.neighbor_calls == [{"chunk_id": 2, "window": 2}]  # janela limitada a 2
    assert registry.chunk_ids == [2, 1, 3]
    assert "antes" in out and "centro" in out and "depois" in out
    # o trecho já conhecido mantém a ref T1 mesmo `add` devolvendo só os novos
    assert "[T1] (seção: A)\ncentro" in out


def test_ler_contexto_trunca_total(make):
    vizinhos = [chunk(i, "x" * 1500) for i in range(1, 4)]
    tools, _, _ = make([[chunk(1)]], vizinhos)
    tools["buscar_materiais"].invoke({"consulta": "x"})

    out = tools["ler_contexto"].invoke({"ref": "T1"})

    assert len(out) <= 2600


def test_ler_contexto_ref_invalida(make):
    tools, _, stub = make([[chunk(1)]])

    for ref in ("T9", "abc", "../../etc", ""):
        out = tools["ler_contexto"].invoke({"ref": ref})
        assert out.startswith("Referência inválida")
        assert "buscar_materiais" in out
    assert stub.neighbor_calls == []


def test_ler_contexto_aceita_ref_com_colchetes_e_minuscula(make):
    tools, _, stub = make([[chunk(5)]], [chunk(5)])
    tools["buscar_materiais"].invoke({"consulta": "x"})

    out = tools["ler_contexto"].invoke({"ref": "[t1]", "vizinhos": 0})

    assert stub.neighbor_calls == [{"chunk_id": 5, "window": 0}]
    assert out.startswith("[T1]")


def test_web_so_existe_com_allow_web(make):
    sem, _, _ = make(allow_web=False)
    com, _, _ = make(allow_web=True)

    assert set(sem) == {"buscar_materiais", "ler_contexto"}
    assert set(com) == {"buscar_materiais", "ler_contexto", "pesquisar_web"}


def test_pesquisar_web_formata_como_referencia_externa(make, monkeypatch):
    monkeypatch.setattr(tools_module.web_search, "build_query", lambda user, q: f"Q:{q}")
    chamadas = []

    def fake_search(query, max_results):
        chamadas.append(query)
        return [{"title": "Grade", "body": "Disciplinas do curso"}]

    monkeypatch.setattr(tools_module.web_search, "search_web", fake_search)
    tools, _, _ = make(allow_web=True)

    out = tools["pesquisar_web"].invoke({"consulta": "grade"})

    assert chamadas == ["Q:grade"]
    assert "NÃO são da instituição" in out
    assert "[1] Grade — Disciplinas do curso" in out


def test_pesquisar_web_sem_resultado(make, monkeypatch):
    monkeypatch.setattr(tools_module.web_search, "build_query", lambda user, q: q)
    monkeypatch.setattr(tools_module.web_search, "search_web", lambda q, n: [])
    tools, _, _ = make(allow_web=True)

    assert tools["pesquisar_web"].invoke({"consulta": "x"}) == "Nenhum resultado externo encontrado."


def test_recorder_recebe_step_sem_conteudo(make):
    recorder = TraceRecorder()
    tools, _, _ = make([[chunk(1, "segredo do aluno")]], recorder=recorder)

    tools["buscar_materiais"].invoke({"consulta": "minha dúvida pessoal"})
    tools["ler_contexto"].invoke({"ref": "T7"})

    assert [(s["type"], s["name"], s["n_results"]) for s in recorder.steps] == [
        ("tool", "buscar_materiais", 1),
        ("tool", "ler_contexto", 0),
    ]
    assert "dúvida" not in str(recorder.steps)


def test_argumentos_invalidos_sao_rejeitados(make):
    tools, _, _ = make([[chunk(1)]])

    with pytest.raises(Exception):
        tools["buscar_materiais"].invoke({"consulta": "x", "curso": "medicina"})
