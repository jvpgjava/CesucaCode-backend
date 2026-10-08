import pytest

from apps.conversations.models import Message, MessageTrace
from apps.conversations.tracing import TraceRecorder


@pytest.fixture
def assistant_message(conversation):
    return Message.objects.create(conversation=conversation, role=Message.Role.ASSISTANT, content="ok")


def test_recorder_grava_trace_completo(assistant_message):
    recorder = TraceRecorder()
    recorder.set(route="direta", intent="conteudo_tecnico", route_source="l1", models={"answer": "m"},
                 chunk_ids=[1, 2], distances=[0.1, 0.2])
    with recorder.step("retrieval", "vector_search") as meta:
        meta["n_chunks"] = 2
    recorder.add_usage({"input_tokens": 10, "output_tokens": 5})
    recorder.add_usage({"input_tokens": 3, "output_tokens": 1})
    recorder.add_usage(None)
    recorder.mark_first_token()

    trace = recorder.finish(assistant_message)

    trace = MessageTrace.objects.get(pk=trace.pk)
    assert trace.message == assistant_message
    assert assistant_message.trace == trace
    assert (trace.input_tokens, trace.output_tokens) == (13, 6)
    assert trace.route == "direta" and trace.route_source == "l1"
    assert trace.chunk_ids == [1, 2]
    assert trace.ttft_ms is not None and trace.latency_ms >= trace.ttft_ms
    assert trace.pipeline_version  # vem de settings.PIPELINE_VERSION
    assert trace.steps[0]["type"] == "retrieval" and trace.steps[0]["n_chunks"] == 2
    assert "ms" in trace.steps[0]


def test_step_registra_duracao_mesmo_com_excecao():
    recorder = TraceRecorder()
    with pytest.raises(RuntimeError):
        with recorder.step("web", "search"):
            raise RuntimeError("falhou")
    assert recorder.steps[0]["name"] == "search"


def test_ttft_fica_nulo_sem_token(assistant_message):
    trace = TraceRecorder().finish(assistant_message, error="boom")
    assert trace.ttft_ms is None
    assert trace.error == "boom"


def test_add_usage_tolera_lixo():
    recorder = TraceRecorder()
    recorder.add_usage({"input_tokens": None, "output_tokens": "abc"})
    assert recorder.input_tokens == 0


def test_set_rejeita_campo_desconhecido():
    with pytest.raises(TypeError):
        TraceRecorder().set(banana=1)


def test_finish_nao_levanta_se_gravar_falhar(assistant_message, monkeypatch, caplog):
    def boom(**kwargs):
        raise RuntimeError("banco fora")

    monkeypatch.setattr(MessageTrace.objects, "create", boom)
    assert TraceRecorder().finish(assistant_message) is None
    assert "Falha ao gravar o MessageTrace" in caplog.text


def test_finish_sem_mensagem_nao_faz_nada():
    assert TraceRecorder().finish(None) is None
