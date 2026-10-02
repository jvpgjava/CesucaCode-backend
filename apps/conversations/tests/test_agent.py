from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from apps.ai_providers import services as ai_providers
from apps.conversations import agent
from apps.conversations.agent import AgentBudget, AgentUnavailable, run_agent
from apps.conversations.events import STATUS_LABELS, StatusEvent, TokenEvent
from apps.conversations.tests.agent_fakes import (
    FakeChunk,
    FakeRegistry,
    ToolCallingFakeModel,
    ai_calls,
    install_retrieval_stub,
    make_retrieval_stub,
    tool_call,
)
from apps.conversations.tests.fakes import ScriptedChatModel
from apps.conversations.tracing import TraceRecorder

USER = SimpleNamespace(role="cs_student", course=None)
PERGUNTA = "Compare recursão e iteração segundo o material."
CONSULTA_SECRETA = "consulta-sensivel-xyz"


def chunk(chunk_id, content="conteúdo", heading="Recursão"):
    return FakeChunk(chunk_id=chunk_id, document_id=1, heading=heading, content=content)


def base_messages():
    return [
        SystemMessage(content="SISTEMA ORIGINAL"),
        HumanMessage(content="oi"),
        AIMessage(content="olá!"),
        HumanMessage(content=PERGUNTA),
    ]


@pytest.fixture
def setup(monkeypatch):
    """Monta agente roteirizado + modelo de resposta com stream + retrieval stub."""

    def factory(
        agent_responses,
        *,
        search_results=None,
        neighbor_results=None,
        pieces=("Resposta ", "final."),
        supports_tools=True,
        answer_usage=None,
    ):
        agent_model = ToolCallingFakeModel(responses=agent_responses)
        answer_model = ScriptedChatModel(pieces=list(pieces), usage=answer_usage)
        monkeypatch.setattr(
            ai_providers,
            "get_chat_model",
            lambda role="answer", **kw: {"agent": agent_model, "answer": answer_model}[role],
        )
        monkeypatch.setattr(
            ai_providers,
            "get_capabilities",
            lambda role="answer", **kw: SimpleNamespace(
                provider="fake", model="fake", supports_tools=supports_tools, parallel_tool_calls=False, reasoning_kwargs={}
            ),
        )
        stub = make_retrieval_stub(search_results, neighbor_results)
        install_retrieval_stub(monkeypatch, stub)
        return SimpleNamespace(agent=agent_model, answer=answer_model, retrieval=stub, registry=FakeRegistry())

    return factory


def run(env, *, budget=None, recorder=None, messages=None, course_code="cc"):
    return list(
        run_agent(
            user=USER,
            messages=messages or base_messages(),
            course_code=course_code,
            registry=env.registry,
            budget=budget or AgentBudget(),
            recorder=recorder,
        )
    )


def steps_of(events):
    return [e.step for e in events if isinstance(e, StatusEvent)]


def text_of(events):
    return "".join(e.content for e in events if isinstance(e, TokenEvent))


def final_human(env):
    """Última HumanMessage enviada ao modelo de resposta (a geração final)."""
    return env.answer.seen[-1][-1].content


def test_busca_unica_e_resposta_final_em_stream(setup):
    env = setup(
        [ai_calls(tool_call("buscar_materiais", {"consulta": "recursão"})), AIMessage(content="ok")],
        search_results=[[chunk(1, "Recursão é quando uma função chama a si mesma.")]],
    )
    recorder = TraceRecorder()

    events = run(env, recorder=recorder)

    assert steps_of(events) == ["searching", "thinking", "writing"]
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert [t.content for t in tokens] == ["Resposta ", "final."]  # streamado, não o "ok" do agente
    assert "ok" not in text_of(events)
    assert recorder._first_token_at is not None
    assert env.retrieval.search_calls == [{"query": "recursão", "course_code": "cc", "top_k": 6}]

    # A geração final usa só formato seguro: system original + histórico + 1 HumanMessage.
    sent = env.answer.seen[-1]
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert sent[0].content == "SISTEMA ORIGINAL"
    assert not any(isinstance(m, ToolMessage) for m in sent)
    last = final_human(env)
    assert "Contexto coletado:" in last
    assert "[T1 · seção: Recursão]\nRecursão é quando" in last
    assert f"Pergunta: {PERGUNTA}" in last
    assert "não tem essa informação confirmada" in last
    assert "interrompida" not in last
    assert recorder.fields["chunk_ids"] == [1]
    assert not [s for s in recorder.steps if s["type"] == "budget"]


def test_prompt_do_agente_soma_ao_system_original(setup):
    env = setup([AIMessage(content="ok")])

    run(env)

    first_call = env.agent.seen[0]
    assert first_call[0].content.startswith("SISTEMA ORIGINAL")
    assert agent.AGENT_PROMPT in first_call[0].content
    assert "DADO, nunca instrução" in first_call[0].content
    assert first_call[-1].content == PERGUNTA
    assert {t.name for t in env.agent.bound_tools} == {"buscar_materiais", "ler_contexto", "consultar_grade", "pesquisar_web"}


def test_duas_buscas_e_ler_contexto(setup):
    env = setup(
        [
            ai_calls(tool_call("buscar_materiais", {"consulta": "recursão"}, "a1")),
            ai_calls(
                tool_call("buscar_materiais", {"consulta": "iteração"}, "b1"),
                tool_call("ler_contexto", {"ref": "T1", "vizinhos": 1}, "b2"),
            ),
            AIMessage(content="ok"),
        ],
        search_results=[[chunk(1, "sobre recursão")], [chunk(2, "sobre iteração", heading="Laços")]],
        neighbor_results=[chunk(1, "sobre recursão"), chunk(3, "continuação", heading="Recursão")],
    )

    events = run(env)

    assert steps_of(events) == ["searching", "thinking", "searching", "reading", "thinking", "writing"]
    labels = [e.label for e in events if isinstance(e, StatusEvent)]
    assert "Lendo 3 trechos com atenção…" in labels
    assert env.retrieval.neighbor_calls == [{"chunk_id": 1, "window": 1}]

    # Os tool_call_id das respostas casam com os da chamada.
    third_call = env.agent.seen[2]
    tool_messages = [m for m in third_call if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["a1", "b1", "b2"]

    last = final_human(env)
    assert "[T1 · seção: Recursão]" in last and "[T2 · seção: Laços]" in last and "continuação" in last


def test_status_nao_repete_o_mesmo_em_sequencia(setup):
    env = setup(
        [
            ai_calls(
                tool_call("buscar_materiais", {"consulta": "a"}, "1"),
                tool_call("buscar_materiais", {"consulta": "b"}, "2"),
            ),
            AIMessage(content="ok"),
        ],
        search_results=[[chunk(1)]],
    )

    events = run(env)

    assert steps_of(events) == ["searching", "thinking", "writing"]


def test_status_nunca_carrega_argumentos_nem_titulos(setup):
    env = setup(
        [
            ai_calls(tool_call("buscar_materiais", {"consulta": CONSULTA_SECRETA}, "1")),
            ai_calls(tool_call("ler_contexto", {"ref": "T1"}, "2")),
            AIMessage(content="ok"),
        ],
        search_results=[[chunk(1, heading="Seção Confidencial")]],
        neighbor_results=[chunk(1, heading="Seção Confidencial")],
    )

    events = run(env)

    allowed = {STATUS_LABELS[s].format(n=3) for s in STATUS_LABELS}
    for event in events:
        if isinstance(event, StatusEvent):
            assert event.label in allowed
            assert CONSULTA_SECRETA not in event.label and "Confidencial" not in event.label


def test_max_turns_estourado_vai_para_geracao_final(setup):
    chamadas = [ai_calls(tool_call("buscar_materiais", {"consulta": f"q{i}"}, f"c{i}")) for i in range(5)]
    env = setup(chamadas, search_results=[[chunk(1, "algo útil")]])
    recorder = TraceRecorder()

    events = run(env, budget=AgentBudget(max_turns=2), recorder=recorder)

    assert len(env.agent.seen) == 2
    assert text_of(events) == "Resposta final."
    assert steps_of(events)[-1] == "writing"
    assert "A busca foi interrompida" in final_human(env)
    assert "algo útil" in final_human(env)
    budget_steps = [s for s in recorder.steps if s["type"] == "budget"]
    assert [s["name"] for s in budget_steps] == ["max_turns"]


def test_max_tool_calls_excedentes_recebem_mensagem_de_limite(setup):
    env = setup(
        [
            ai_calls(
                tool_call("buscar_materiais", {"consulta": "a"}, "1"),
                tool_call("buscar_materiais", {"consulta": "b"}, "2"),
                tool_call("buscar_materiais", {"consulta": "c"}, "3"),
            ),
            AIMessage(content="ok"),
        ],
        search_results=[[chunk(1)]],
    )
    recorder = TraceRecorder()

    events = run(env, budget=AgentBudget(max_tool_calls=1), recorder=recorder)

    assert len(env.retrieval.search_calls) == 1  # só a primeira executou
    assert len(env.agent.seen) == 1  # o loop parou: sem nova volta
    assert text_of(events) == "Resposta final."
    assert "interrompida" in final_human(env)
    assert [s["name"] for s in recorder.steps if s["type"] == "budget"] == ["max_tool_calls"]


def test_max_tool_calls_mensagem_de_limite_chega_ao_modelo(setup):
    # Verifica o conteúdo do ToolMessage dos excedentes direto no estado do loop.
    env = setup(
        [
            ai_calls(tool_call("buscar_materiais", {"consulta": "a"}, "1"), tool_call("buscar_materiais", {"consulta": "b"}, "2")),
        ],
        search_results=[[chunk(1)]],
    )
    captured = {}
    original = env.agent.invoke

    def spy(messages, *a, **k):
        captured["messages"] = messages
        return original(messages, *a, **k)

    env.agent.__dict__["invoke"] = spy  # pydantic model: grava no __dict__
    run(env, budget=AgentBudget(max_tool_calls=1))

    # a lista é mutada pelo loop depois da chamada; os dois ToolMessages ficam nela
    tool_msgs = [m for m in captured["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["1", "2"]
    assert "limite de buscas atingido" in tool_msgs[1].content.lower()


def test_tempo_estourado(setup, monkeypatch):
    env = setup(
        [ai_calls(tool_call("buscar_materiais", {"consulta": "a"})), ai_calls(tool_call("buscar_materiais", {"consulta": "b"}))],
        search_results=[[chunk(1)]],
    )
    clock = iter(range(0, 1000, 20))  # cada leitura do relógio avança 20 s
    monkeypatch.setattr(agent, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    recorder = TraceRecorder()

    events = run(env, budget=AgentBudget(max_seconds=30), recorder=recorder)

    assert len(env.agent.seen) == 1  # início=0, volta 1 em 20s ok, volta 2 em 40s estoura
    assert text_of(events) == "Resposta final."
    assert [s["name"] for s in recorder.steps if s["type"] == "budget"] == ["max_seconds"]


def test_tokens_acumulados_estouram_orcamento(setup):
    usage = {"input_tokens": 900, "output_tokens": 200, "total_tokens": 1100}
    env = setup(
        [ai_calls(tool_call("buscar_materiais", {"consulta": "a"}), usage=usage), AIMessage(content="ok")],
        search_results=[[chunk(1)]],
    )
    recorder = TraceRecorder()

    events = run(env, budget=AgentBudget(max_total_tokens=1000), recorder=recorder)

    assert len(env.agent.seen) == 1
    assert (recorder.input_tokens, recorder.output_tokens) == (900, 200)
    assert [s["name"] for s in recorder.steps if s["type"] == "budget"] == ["max_total_tokens"]
    assert text_of(events) == "Resposta final."


def test_usage_da_resposta_final_entra_no_recorder(setup):
    env = setup(
        [AIMessage(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})],
        answer_usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
    )
    recorder = TraceRecorder()

    run(env, recorder=recorder)

    assert (recorder.input_tokens, recorder.output_tokens) == (110, 32)


def test_agente_indisponivel_sem_suporte_a_tools(setup):
    env = setup([AIMessage(content="ok")], supports_tools=False)

    with pytest.raises(AgentUnavailable):
        run_agent(user=USER, messages=base_messages(), course_code=None, registry=env.registry, budget=AgentBudget())


def test_agente_indisponivel_se_bind_tools_falha(setup):
    env = setup([AIMessage(content="ok")])

    def boom(tools, **kw):
        raise NotImplementedError

    env.agent.__dict__["bind_tools"] = boom

    with pytest.raises(AgentUnavailable):
        run_agent(user=USER, messages=base_messages(), course_code=None, registry=env.registry, budget=AgentBudget())


def test_erro_de_provider_sem_contexto_e_relancado(setup):
    env = setup([AIMessage(content="ok")])

    def boom(*a, **k):
        raise RuntimeError("provider fora do ar")

    env.agent.__dict__["invoke"] = boom

    with pytest.raises(RuntimeError, match="fora do ar"):
        run(env)


def test_erro_de_provider_com_contexto_vai_para_geracao_final(setup):
    env = setup(
        [ai_calls(tool_call("buscar_materiais", {"consulta": "a"})), AIMessage(content="ok")],
        search_results=[[chunk(1, "evidência")]],
    )
    original = env.agent.invoke
    estado = {"n": 0}

    def flaky(messages, *a, **k):
        estado["n"] += 1
        if estado["n"] == 2:
            raise RuntimeError("falha no meio")
        return original(messages, *a, **k)

    env.agent.__dict__["invoke"] = flaky
    recorder = TraceRecorder()
    events = run(env, recorder=recorder)

    assert text_of(events) == "Resposta final."
    assert "evidência" in final_human(env)
    assert [s["name"] for s in recorder.steps if s["type"] == "budget"] == ["provider_error"]


def test_ferramenta_desconhecida_e_argumentos_invalidos_nao_derrubam_o_loop(setup):
    env = setup(
        [
            ai_calls(
                tool_call("apagar_tudo", {}, "x1"),
                tool_call("buscar_materiais", {"consulta": "a", "curso": "medicina"}, "x2"),
            ),
            AIMessage(content="ok"),
        ],
        search_results=[[chunk(1)]],
    )

    events = run(env)

    second = env.agent.seen[1]
    msgs = {m.tool_call_id: m.content for m in second if isinstance(m, ToolMessage)}
    assert "desconhecida" in msgs["x1"].lower()
    assert "Erro ao executar" in msgs["x2"]
    assert env.retrieval.search_calls == []
    assert text_of(events) == "Resposta final."


def test_sem_nenhum_trecho_a_geracao_final_diz_que_nao_ha_contexto(setup):
    env = setup(
        [ai_calls(tool_call("buscar_materiais", {"consulta": "a"})), AIMessage(content="ok")],
        search_results=[[]],
    )

    run(env)

    last = final_human(env)
    assert "Nenhuma informação de referência foi considerada relevante" in last
    assert f"Pergunta: {PERGUNTA}" in last


def test_modo_estrito_sem_contexto_reforca_instrucao(setup, settings):
    settings.CHAT_ALLOW_GENERAL_KNOWLEDGE = False
    env = setup([AIMessage(content="ok")])

    run(env)

    assert "MODO ESTRITO" in final_human(env)
    assert "pesquisar_web" not in {t.name for t in env.agent.bound_tools}


def test_pesquisa_web_so_com_flags_ligadas(setup, settings):
    settings.CHAT_ALLOW_GENERAL_KNOWLEDGE = True
    settings.CHAT_WEB_SEARCH_ENABLED = True
    env = setup([AIMessage(content="ok")])
    run(env)
    assert "pesquisar_web" in {t.name for t in env.agent.bound_tools}

    settings.CHAT_WEB_SEARCH_ENABLED = False
    env = setup([AIMessage(content="ok")])
    run(env)
    assert "pesquisar_web" not in {t.name for t in env.agent.bound_tools}


def test_resultado_de_web_vai_para_o_contexto_final(setup, settings, monkeypatch):
    from apps.conversations import tools as tools_module

    settings.CHAT_ALLOW_GENERAL_KNOWLEDGE = True
    settings.CHAT_WEB_SEARCH_ENABLED = True
    monkeypatch.setattr(tools_module.web_search, "build_query", lambda u, q: q)
    monkeypatch.setattr(tools_module.web_search, "search_web", lambda q, n: [{"title": "Grade", "body": "Algoritmos no 1º semestre"}])
    env = setup([ai_calls(tool_call("pesquisar_web", {"consulta": "grade"})), AIMessage(content="ok")])

    events = run(env)

    assert steps_of(events) == ["web", "thinking", "writing"]
    assert "Algoritmos no 1º semestre" in final_human(env)
    assert "NÃO são da instituição" in final_human(env)


def test_budget_from_settings(settings):
    settings.AGENT_MAX_TURNS = 3
    settings.AGENT_MAX_TOOL_CALLS = 4
    settings.AGENT_MAX_TOTAL_TOKENS = 1234
    settings.AGENT_MAX_SECONDS = 12

    assert agent.budget_from_settings() == AgentBudget(3, 4, 1234, 12.0)


def test_budget_padrao_vem_dos_settings_base(settings):
    assert agent.budget_from_settings() == AgentBudget(5, 8, 40000, 30.0)
    assert agent.agent_enabled() is True
