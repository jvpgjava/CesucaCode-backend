from types import SimpleNamespace

import pytest
from django.test import override_settings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from apps.ai_providers import capabilities, services
from apps.ai_providers.exceptions import ProviderConfigurationError

BASE = {
    "LLM_PROVIDER": "gemini",
    "LLM_MODEL": "gemini-2.5-flash",
    "LLM_TIMEOUT": 60,
    "LLM_MAX_RETRIES": 2,
    "LLM_STRUCTURED_METHOD": "",
    "LLM_THINKING_BUDGET": "",
    "GOOGLE_API_KEY": "fake",
    "OPENAI_API_KEY": "fake",
    "ANTHROPIC_API_KEY": "fake",
    "DEEPSEEK_API_KEY": "fake",
    "ABACUSAI_API_KEY": "fake",
    "OPENROUTER_API_KEY": "fake",
    "OPENROUTER_HTTP_REFERER": "",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    **{f"LLM_{r}_{k}": "" for r in ("ANSWER", "ROUTER", "AGENT", "JUDGE") for k in ("PROVIDER", "MODEL", "MAX_TOKENS", "TEMPERATURE")},
}


@pytest.fixture(autouse=True)
def base_settings():
    with override_settings(**BASE):
        yield


class Intent(BaseModel):
    label: str
    score: int = 0


# --- resolução de papéis ----------------------------------------------------------


def test_defaults_por_papel_e_fallback_para_llm_global():
    expected = {"answer": (1500, 0.3), "router": (256, 0.0), "agent": (1024, 0.2), "judge": (512, 0.0)}
    for role, (max_tokens, temperature) in expected.items():
        cfg = services.get_role_config(role)
        assert (cfg.provider, cfg.model) == ("gemini", "gemini-2.5-flash")
        assert (cfg.max_tokens, cfg.temperature) == (max_tokens, temperature)
        assert (cfg.timeout, cfg.max_retries) == (60.0, 2)


@override_settings(
    LLM_ROUTER_PROVIDER="openrouter",
    LLM_ROUTER_MODEL="google/gemini-2.5-flash-lite",
    LLM_ROUTER_MAX_TOKENS="128",
    LLM_ROUTER_TEMPERATURE="0.1",
)
def test_papel_com_env_proprio_nao_afeta_os_outros():
    router = services.get_role_config("router")
    assert (router.provider, router.model, router.max_tokens, router.temperature) == (
        "openrouter",
        "google/gemini-2.5-flash-lite",
        128,
        0.1,
    )
    assert services.get_role_config("answer").provider == "gemini"


@override_settings(LLM_ANSWER_MODEL="gemini-2.5-pro", LLM_ANSWER_MAX_TOKENS="lixo")
def test_valor_invalido_cai_no_padrao_e_model_pode_mudar_sem_provider():
    cfg = services.get_role_config("answer")
    assert (cfg.provider, cfg.model) == ("gemini", "gemini-2.5-pro")
    assert cfg.max_tokens == 1500


def test_overrides_explicitos_vencem_o_env():
    cfg = services.get_role_config("router", provider="claude", model="claude-x", max_tokens=10, temperature=0.9)
    assert (cfg.provider, cfg.model, cfg.max_tokens, cfg.temperature) == ("claude", "claude-x", 10, 0.9)


def test_papel_desconhecido_levanta_erro_de_configuracao():
    with pytest.raises(ProviderConfigurationError):
        services.get_role_config("inexistente")


def test_get_chat_model_sem_argumento_continua_funcionando():
    model = services.get_chat_model()
    assert model.model == "gemini-2.5-flash"
    assert model.max_output_tokens == 1500
    assert model.temperature == 0.3


# --- construção por provider -------------------------------------------------------


def test_gemini_recebe_parametros_e_thinking_budget():
    with override_settings(LLM_THINKING_BUDGET="0"):
        model = services.get_chat_model("router")
    assert (model.max_output_tokens, model.temperature, model.timeout, model.max_retries) == (256, 0.0, 60.0, 2)
    assert model.thinking_budget == 0
    assert services.get_chat_model("router").thinking_budget is None


@override_settings(LLM_PROVIDER="openrouter", LLM_MODEL="deepseek/deepseek-chat", OPENROUTER_HTTP_REFERER="https://sofia.example")
def test_openrouter_usa_base_url_headers_e_stream_usage():
    model = services.get_chat_model("agent")
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.default_headers == {"X-Title": "S.O.F.I.A", "HTTP-Referer": "https://sofia.example"}
    assert (model.max_tokens, model.temperature, model.request_timeout, model.max_retries) == (1024, 0.2, 60.0, 2)
    assert model.stream_usage is True


@override_settings(LLM_PROVIDER="openrouter", LLM_MODEL="x/y")
def test_openrouter_sem_referer_manda_so_x_title():
    assert services.get_chat_model().default_headers == {"X-Title": "S.O.F.I.A"}


@pytest.mark.parametrize(
    ("provider", "base"),
    [("deepseek", "https://api.deepseek.com/v1"), ("abacusai", "https://routellm.abacus.ai/v1")],
)
def test_providers_compativeis_com_openai(provider, base):
    model = services.get_chat_model("answer", provider=provider, model="m")
    assert model.openai_api_base == base
    assert model.max_tokens == 1500
    assert model.stream_usage is True


def test_claude_e_ollama_usam_nomes_proprios_de_parametro():
    claude = services.get_chat_model("answer", provider="claude", model="claude-sonnet-4-5")
    assert (claude.max_tokens, claude.default_request_timeout, claude.max_retries) == (1500, 60.0, 2)
    ollama = services.get_chat_model("router", provider="ollama", model="llama3.1")
    assert ollama.num_predict == 256
    assert ollama.client_kwargs == {"timeout": 60.0}


def test_provider_nao_suportado():
    with pytest.raises(ProviderConfigurationError):
        services.get_chat_model("answer", provider="nada", model="x")


# --- capacidades ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "method"),
    [
        ("gemini", "json_schema"),
        ("openai", "json_schema"),
        ("claude", "function_calling"),
        ("ollama", "json_schema"),
        ("deepseek", "function_calling"),
        ("abacusai", "function_calling"),
        ("openrouter", "function_calling"),
    ],
)
def test_metodo_estruturado_por_provider(provider, method):
    caps = capabilities.resolve_capabilities(provider, "qualquer")
    assert caps.structured_method == method
    assert caps.supports_tools is True


def test_override_por_prefixo_de_modelo():
    assert capabilities.resolve_capabilities("ollama", "DeepSeek-R1:8b").supports_tools is False
    assert capabilities.resolve_capabilities("ollama", "llama3.1").supports_tools is True


@override_settings(LLM_STRUCTURED_METHOD="json_mode")
def test_env_sobrescreve_metodo_estruturado():
    assert services.get_capabilities("router").structured_method == "json_mode"


@override_settings(LLM_STRUCTURED_METHOD="invalido")
def test_override_invalido_e_ignorado():
    assert services.get_capabilities("router").structured_method == "json_schema"


def test_reasoning_kwargs_so_para_gemini_e_so_com_env():
    assert capabilities.resolve_capabilities("gemini", "m").reasoning_kwargs == {}
    with override_settings(LLM_THINKING_BUDGET="512"):
        assert capabilities.resolve_capabilities("gemini", "m").reasoning_kwargs == {"thinking_budget": 512}
        assert capabilities.resolve_capabilities("openai", "m").reasoning_kwargs == {}
    with override_settings(LLM_THINKING_BUDGET="abc"):
        assert capabilities.resolve_capabilities("gemini", "m").reasoning_kwargs == {}


def test_get_capabilities_acompanha_o_papel():
    with override_settings(LLM_JUDGE_PROVIDER="claude", LLM_JUDGE_MODEL="claude-x"):
        caps = services.get_capabilities("judge")
        assert (caps.provider, caps.model, caps.structured_method) == ("claude", "claude-x", "function_calling")
        assert services.get_capabilities("answer").provider == "gemini"


# --- invoke_structured -------------------------------------------------------------


class FakeStructuredModel:
    """Imita `with_structured_output(..., include_raw=True)`: devolve, a cada chamada,
    o próximo item de `outputs` (dict raw/parsed/parsing_error ou uma exceção)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.method = None

    def with_structured_output(self, schema, method=None, include_raw=False):
        assert include_raw is True
        self.method = method

        def run(messages):
            self.calls.append(messages)
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(run)


def _usage_msg(i=10, o=5):
    return AIMessage(content="", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})


@pytest.fixture
def patch_model(monkeypatch):
    def _patch(fake):
        monkeypatch.setattr(services, "get_chat_model", lambda role="answer", **kw: fake)
        return fake

    return _patch


def test_invoke_structured_sucesso(patch_model):
    fake = patch_model(
        FakeStructuredModel([{"raw": _usage_msg(), "parsed": Intent(label="meta"), "parsing_error": None}])
    )
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")], "router")
    assert result.ok and result.value == Intent(label="meta") and result.error is None
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}
    assert fake.method == "json_schema"
    assert len(fake.calls) == 1


def test_invoke_structured_usa_metodo_do_registro(patch_model):
    fake = patch_model(FakeStructuredModel([{"raw": _usage_msg(), "parsed": Intent(label="a"), "parsing_error": None}]))
    with override_settings(LLM_ROUTER_PROVIDER="deepseek", LLM_ROUTER_MODEL="deepseek-chat"):
        services.invoke_structured(Intent, [HumanMessage(content="oi")], "router")
    assert fake.method == "function_calling"


def test_invoke_structured_retry_de_reparo_e_soma_usage(patch_model):
    fake = patch_model(
        FakeStructuredModel(
            [
                {"raw": _usage_msg(10, 5), "parsed": None, "parsing_error": ValueError("json inválido")},
                {"raw": _usage_msg(12, 6), "parsed": Intent(label="ok"), "parsing_error": None},
            ]
        )
    )
    original = [HumanMessage(content="oi")]
    result = services.invoke_structured(Intent, original, "router")
    assert result.ok and result.value.label == "ok"
    assert result.usage == {"input_tokens": 22, "output_tokens": 11}
    assert len(fake.calls) == 2
    repair = fake.calls[1][-1]
    assert isinstance(repair, HumanMessage) and "json inválido" in repair.content
    assert len(original) == 1  # não muta a lista do chamador


def test_invoke_structured_falha_duas_vezes_devolve_default(patch_model):
    bad = {"raw": _usage_msg(), "parsed": None, "parsing_error": ValueError("ruim")}
    fake = patch_model(FakeStructuredModel([bad, bad]))
    default = Intent(label="fallback")
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")], "router", default=default)
    assert not result.ok
    assert result.value is default
    assert "ruim" in result.error
    assert len(fake.calls) == 2


def test_invoke_structured_valida_dict_com_pydantic(patch_model):
    patch_model(FakeStructuredModel([{"raw": None, "parsed": {"label": "x", "score": "3"}, "parsing_error": None}]))
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")])
    assert result.ok and result.value == Intent(label="x", score=3)


def test_invoke_structured_excecao_do_provider_nao_propaga(patch_model):
    fake = patch_model(FakeStructuredModel([RuntimeError("503 indisponível")]))
    default = Intent(label="d")
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")], "router", default=default)
    assert not result.ok and result.value is default
    assert "503 indisponível" in result.error
    assert len(fake.calls) == 1  # exceção de provider não gasta o retry de reparo


def test_invoke_structured_erro_de_configuracao_nao_propaga():
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")], "router", provider="nada", model="x")
    assert not result.ok and result.value is None
    assert "não suportado" in result.error


class _FakeToolChat(GenericFakeChatModel):
    """Modelo fake que aceita bind_tools, para exercitar o with_structured_output real
    (método function_calling) de ponta a ponta, sem rede."""

    def bind_tools(self, tools, **kwargs):
        return self


@override_settings(LLM_STRUCTURED_METHOD="function_calling")
def test_invoke_structured_ponta_a_ponta_com_function_calling(monkeypatch):
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "Intent", "args": {"label": "duvida", "score": 2}, "id": "1"}],
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    fake = _FakeToolChat(messages=iter([msg]))
    monkeypatch.setattr(services, "get_chat_model", lambda role="answer", **kw: fake)
    result = services.invoke_structured(Intent, [HumanMessage(content="oi")], "router")
    assert result.ok, result.error
    assert result.value == Intent(label="duvida", score=2)
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}


# --- extract_usage -----------------------------------------------------------------


def test_extract_usage_de_usage_metadata():
    assert services.extract_usage(_usage_msg(3, 4)) == {"input_tokens": 3, "output_tokens": 4}


def test_extract_usage_tolerante():
    assert services.extract_usage(None) == {"input_tokens": 0, "output_tokens": 0}
    assert services.extract_usage(AIMessage(content="x")) == {"input_tokens": 0, "output_tokens": 0}
    assert services.extract_usage(SimpleNamespace(usage_metadata={"input_tokens": None})) == {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_extract_usage_fallback_response_metadata():
    msg = AIMessage(content="x", response_metadata={"token_usage": {"prompt_tokens": 8, "completion_tokens": 2}})
    assert services.extract_usage(msg) == {"input_tokens": 8, "output_tokens": 2}


# --- is_provider_refusal -----------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        Exception("Error code: 400 - Content Exists Risk"),
        Exception("The response was filtered due to content_filter"),
        Exception("Azure content violation detected"),
        Exception("Response was blocked: block_reason: SAFETY"),
        Exception("Candidate stopped, finish_reason: SAFETY"),
        Exception("Content blocked for safety reasons"),
        Exception("Your prompt was flagged as potentially violating our usage policy"),
    ],
)
def test_refusal_por_texto(exc):
    assert services.is_provider_refusal(exc)


def test_refusal_por_atributos_estruturados():
    exc = RuntimeError("falha")
    exc.finish_reason = "SAFETY"
    assert services.is_provider_refusal(exc)
    exc2 = RuntimeError("falha")
    exc2.body = {"error": {"code": "content_filter"}}
    assert services.is_provider_refusal(exc2)
    exc3 = RuntimeError("falha")
    exc3.finish_reason = "STOP"
    assert not services.is_provider_refusal(exc3)


def test_refusal_segue_a_causa_da_excecao():
    try:
        try:
            raise Exception("finish_reason: SAFETY")
        except Exception as inner:
            raise RuntimeError("erro do wrapper") from inner
    except RuntimeError as exc:
        assert services.is_provider_refusal(exc)


@pytest.mark.parametrize(
    "exc",
    [
        Exception("safety_identifier must be a string"),
        Exception("Invalid safety settings configuration"),
        Exception("503 Service Unavailable"),
        Exception("Rate limit exceeded"),
        TimeoutError("timed out"),
    ],
)
def test_nao_e_recusa(exc):
    assert not services.is_provider_refusal(exc)
