import json
import pytest

from apps.conversations import retrieval, services
from apps.conversations.events import DoneEvent, ErrorEvent, MetaEvent, StatusEvent, TokenEvent
from apps.conversations.models import Message, MessageTrace
from apps.conversations.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _variante_v0(settings):
    """Estes testes exercitam o envio/salvamento/guarda/regenerate com o pipeline na
    variante v0 (sem roteador LLM, suficiência nem follow-ups). O pipeline completo é
    coberto em test_pipeline.py."""
    settings.CHAT_ROUTER_ENABLED = False
    settings.CHAT_SUFFICIENCY_CHECK_ENABLED = False
    settings.CHAT_FOLLOWUPS_ENABLED = False


def _run(conversation, text, **kwargs):
    return list(services.send_message(conversation, text, **kwargs))


def test_fluxo_feliz_emite_eventos_na_ordem_e_salva_tudo(conversation, fake_llm):
    fake_llm("Uma ", "pilha ", "é LIFO.", usage={"input_tokens": 40, "output_tokens": 7, "total_tokens": 47})

    events = _run(conversation, "O que é uma pilha?")

    assert isinstance(events[0], StatusEvent) and events[0].step == "routing"
    assert isinstance(events[1], MetaEvent) and events[1].route == "direta"
    steps = [e.step for e in events if isinstance(e, StatusEvent)]
    assert steps == ["routing", "searching", "writing"]
    assert "".join(e.content for e in events if isinstance(e, TokenEvent)) == "Uma pilha é LIFO."
    assert isinstance(events[-1], DoneEvent)

    user_msg, answer = conversation.messages.all()
    assert user_msg.role == "user" and events[1].user_message_id == user_msg.id
    assert answer.content == "Uma pilha é LIFO." and events[-1].message_id == answer.id
    conversation.refresh_from_db()
    assert conversation.title == "O que é uma pilha?"


def test_trace_com_tokens_e_ttft(conversation, fake_llm, settings):
    fake_llm("oi", usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})

    _run(conversation, "Como funciona a avaliação?")

    trace = conversation.messages.get(role="assistant").trace
    assert (trace.route, trace.route_source) == ("direta", "fallback")
    assert (trace.input_tokens, trace.output_tokens) == (12, 3)
    assert trace.ttft_ms is not None
    assert trace.models == {"answer": settings.LLM_MODEL}
    assert [s["type"] for s in trace.steps] == ["routing", "retrieval", "llm"]
    assert trace.pipeline_version == settings.PIPELINE_VERSION


def test_usage_ausente_nao_quebra(conversation, fake_llm):
    fake_llm("oi")
    _run(conversation, "Oi")
    trace = conversation.messages.get(role="assistant").trace
    assert (trace.input_tokens, trace.output_tokens) == (0, 0)


def test_contexto_usa_refs_opacas_sem_titulo_do_documento(conversation, fake_llm, monkeypatch):
    model = fake_llm("ok")
    chunks = [
        RetrievedChunk(11, 1, "Avaliação", "A média é 7.", 0.5, 0.12, 0, document_title="Código Disciplinar Cesuca"),
        RetrievedChunk(12, 1, "", "Faltas: 25%.", 0.4, 0.2, 1, document_title="Código Disciplinar Cesuca"),
    ]
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: chunks)

    _run(conversation, "Como é a avaliação?")

    prompt = model.seen[0][-1].content
    assert "[T1 · seção: Avaliação]\nA média é 7." in prompt
    assert "[T2]\nFaltas: 25%." in prompt
    assert "Código Disciplinar" not in prompt and "Origem" not in prompt
    trace = conversation.messages.get(role="assistant").trace
    assert trace.chunk_ids == [11, 12] and trace.distances == [0.12, 0.2]


def test_guarda_redige_refs_vazadas_e_grava_flags(conversation, fake_llm):
    fake_llm("A média é 7 ", "[T1] ", "e a frequência mínima é 75%.")

    events = _run(conversation, "Qual a média?")

    # O stream já mostrou a ref (não dá para desfazer), mas o salvo sai limpo.
    assert "[T1]" in "".join(e.content for e in events if isinstance(e, TokenEvent))
    answer = conversation.messages.get(role="assistant")
    assert "[T1]" not in answer.content
    assert answer.content == "A média é 7 e a frequência mínima é 75%."
    assert "ref_leak" in answer.trace.guard_flags


def test_guarda_sinaliza_titulo_de_documento_acessivel(conversation, fake_llm, coordinator, student, course):
    from apps.documents.models import Document

    doc = Document.objects.create(title="Regimento Interno", uploaded_by=coordinator, status="ready")
    doc.courses.add(course)
    fake_llm("Segundo o regimento interno, é assim.")

    _run(conversation, "Regras?")

    flags = conversation.messages.get(role="assistant").trace.guard_flags
    assert "doc_title:regimento interno" in flags


def test_erro_do_llm_emite_error_amigavel_e_nao_emite_done(conversation, fake_llm):
    fake_llm(fail_with=RuntimeError("segredo interno do provedor"))

    events = _run(conversation, "Oi")

    assert isinstance(events[-1], ErrorEvent)
    assert "segredo" not in events[-1].message
    assert not any(isinstance(e, DoneEvent) for e in events)
    # A pergunta do usuário fica salva; sem resposta, não há mensagem do assistente.
    assert conversation.messages.filter(role="assistant").count() == 0
    assert conversation.messages.filter(role="user").count() == 1


def test_erro_no_meio_do_stream_salva_parcial_com_trace_de_erro(conversation, fake_llm):
    fake_llm("Começo da resposta", fail_with=RuntimeError("caiu"))

    events = _run(conversation, "Oi")

    assert isinstance(events[-1], ErrorEvent)
    answer = conversation.messages.get(role="assistant")
    assert answer.content == "Começo da resposta"
    assert "RuntimeError" in answer.trace.error


def test_recusa_do_provedor_vira_mensagem_padrao(conversation, fake_llm):
    fake_llm(fail_with=RuntimeError("blocked by content filter"))

    events = _run(conversation, "Oi")

    assert isinstance(events[-1], DoneEvent)
    assert conversation.messages.get(role="assistant").content == services.REFUSAL_MESSAGE


def test_desconexao_do_cliente_salva_texto_parcial(conversation, fake_llm):
    fake_llm("Uma ", "pilha ", "é LIFO.")

    gen = services.send_message(conversation, "O que é uma pilha?")
    for event in gen:
        if isinstance(event, TokenEvent) and event.content.startswith("pilha"):
            break
    gen.close()  # equivale ao cliente fechar a conexão (GeneratorExit)

    answer = conversation.messages.get(role="assistant")
    assert answer.content == "Uma pilha "
    assert answer.trace.guard_flags == []


def test_regenerate_apaga_o_par_anterior(conversation, fake_llm):
    fake_llm("Primeira resposta.")
    _run(conversation, "Pergunta 1")
    fake_llm("Segunda resposta.")
    _run(conversation, "Pergunta 2")
    fake_llm("Resposta nova.")

    events = _run(conversation, "Pergunta 2 reformulada", regenerate=True)

    contents = [(m.role, m.content) for m in conversation.messages.all()]
    assert contents == [
        ("user", "Pergunta 1"),
        ("assistant", "Primeira resposta."),
        ("user", "Pergunta 2 reformulada"),
        ("assistant", "Resposta nova."),
    ]
    assert isinstance(events[-1], DoneEvent)
    assert MessageTrace.objects.count() == 2  # o trace da resposta apagada vai junto (cascade)


def test_regenerate_apos_falha_apaga_so_a_pergunta_sem_resposta(conversation, fake_llm):
    fake_llm(fail_with=RuntimeError("x"))
    _run(conversation, "Pergunta")
    fake_llm("Agora foi.")

    _run(conversation, "Pergunta", regenerate=True)

    assert [m.content for m in conversation.messages.all()] == ["Pergunta", "Agora foi."]


def test_regenerate_em_conversa_vazia_funciona(conversation, fake_llm):
    fake_llm("ok")
    _run(conversation, "Oi", regenerate=True)
    assert conversation.messages.count() == 2


def test_historico_nao_duplica_a_pergunta_atual(conversation, fake_llm):
    model = fake_llm("ok")
    _run(conversation, "Primeira")
    _run(conversation, "Segunda")

    sent = model.seen[1]
    human = [m.content for m in sent if m.type == "human"]
    assert len(human) == 2  # "Primeira" do histórico + a pergunta atual com contexto
    assert human[0] == "Primeira" and human[1].endswith("Pergunta: Segunda")


# --- via HTTP -----------------------------------------------------------------


def _sse(response) -> list[tuple[str | None, dict]]:
    body = b"".join(response.streaming_content).decode()
    parsed = []
    for block in filter(None, body.split("\n\n")):
        lines = block.split("\n")
        name = lines.pop(0)[7:] if lines[0].startswith("event: ") else None
        parsed.append((name, json.loads(lines[0][6:])))
    return parsed


def test_endpoint_envia_sse_tipado(api, conversation, fake_llm):
    fake_llm("Olá", " mundo")

    response = api.post(f"/api/conversations/{conversation.id}/messages/send/", {"content": "Como funciona a avaliação?"}, format="json")

    assert response.status_code == 200 and response["Content-Type"].startswith("text/event-stream")
    events = _sse(response)
    names = [name for name, _ in events]
    assert names == ["status", "meta", "status", "status", None, None, "done"]
    assert events[0][1] == {"step": "routing", "label": "Entendendo sua pergunta"}
    assert events[1][1] == {"user_message_id": conversation.messages.get(role="user").id, "route": "direta"}
    assert events[2][1] == {"step": "searching", "label": "Buscando nos materiais do curso"}
    assert events[4][1] == {"content": "Olá"}
    assert events[-1][1] == {"message_id": conversation.messages.get(role="assistant").id}


def test_endpoint_regenerate_via_body(api, conversation, fake_llm):
    url = f"/api/conversations/{conversation.id}/messages/send/"
    fake_llm("A")
    _sse(api.post(url, {"content": "Q"}, format="json"))
    fake_llm("B")
    _sse(api.post(url, {"content": "Q2", "regenerate": True}, format="json"))

    assert [(m.role, m.content) for m in conversation.messages.all()] == [("user", "Q2"), ("assistant", "B")]
