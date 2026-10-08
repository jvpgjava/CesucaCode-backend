import logging
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from django.conf import settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from .capabilities import ModelCapabilities, resolve_capabilities
from .exceptions import ProviderConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

Role = Literal["answer", "router", "agent", "judge"]
ROLES = ("answer", "router", "agent", "judge")

# Limites de saída e temperatura por papel. O roteador classifica (curto e
# determinístico); o agente decide passos; a resposta final é a única criativa.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "answer": {"max_tokens": 1500, "temperature": 0.3},
    "router": {"max_tokens": 256, "temperature": 0.0},
    "agent": {"max_tokens": 1024, "temperature": 0.2},
    "judge": {"max_tokens": 512, "temperature": 0.0},
}


@dataclass(frozen=True)
class RoleConfig:
    role: str
    provider: str
    model: str
    max_tokens: int
    temperature: float
    timeout: float
    max_retries: int


def _setting(name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _number(raw: Any, cast, default):
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning("Valor inválido %r em configuração de LLM; usando o padrão %r.", raw, default)
        return default


def get_role_config(role: Role = "answer", **overrides) -> RoleConfig:
    """Resolve provider/modelo/limites de um papel. Ordem: override explícito >
    LLM_{PAPEL}_* > LLM_PROVIDER/LLM_MODEL (provider/model) ou padrão do papel."""
    if role not in ROLE_DEFAULTS:
        raise ProviderConfigurationError(f"Papel de modelo '{role}' desconhecido (use {', '.join(ROLES)}).")
    prefix = f"LLM_{role.upper()}_"
    defaults = ROLE_DEFAULTS[role]

    def pick(key: str, fallback):
        value = overrides.get(key)
        return value if value not in (None, "") else fallback

    return RoleConfig(
        role=role,
        provider=pick("provider", _setting(prefix + "PROVIDER") or _setting("LLM_PROVIDER")),
        model=pick("model", _setting(prefix + "MODEL") or _setting("LLM_MODEL")),
        max_tokens=int(
            pick("max_tokens", _number(_setting(prefix + "MAX_TOKENS"), int, defaults["max_tokens"]))
        ),
        temperature=float(
            pick("temperature", _number(_setting(prefix + "TEMPERATURE"), float, defaults["temperature"]))
        ),
        timeout=float(pick("timeout", _number(getattr(settings, "LLM_TIMEOUT", 60), float, 60.0))),
        max_retries=int(pick("max_retries", _number(getattr(settings, "LLM_MAX_RETRIES", 2), int, 2))),
    )


_CONFIG_OVERRIDE_KEYS = ("provider", "model", "max_tokens", "temperature", "timeout", "max_retries")


def _split_overrides(overrides: dict) -> tuple[dict, dict]:
    config = {k: v for k, v in overrides.items() if k in _CONFIG_OVERRIDE_KEYS}
    extra = {k: v for k, v in overrides.items() if k not in _CONFIG_OVERRIDE_KEYS}
    return config, extra


def get_chat_model(role: Role = "answer", **overrides) -> BaseChatModel:
    """Modelo de chat do papel, já com max_tokens/temperature/timeout/max_retries.

    `overrides` aceita provider/model/max_tokens/temperature/timeout/max_retries;
    qualquer outra chave vai direto para o construtor da classe LangChain."""
    config_overrides, extra = _split_overrides(overrides)
    cfg = get_role_config(role, **config_overrides)
    caps = resolve_capabilities(cfg.provider, cfg.model)
    return _build_chat_model(cfg, caps, extra)


def get_capabilities(role: Role = "answer", **overrides) -> ModelCapabilities:
    config_overrides, _ = _split_overrides(overrides)
    cfg = get_role_config(role, **config_overrides)
    return resolve_capabilities(cfg.provider, cfg.model)


def _build_chat_model(cfg: RoleConfig, caps: ModelCapabilities, extra: dict) -> BaseChatModel:
    provider, model = cfg.provider, cfg.model

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            **{**caps.reasoning_kwargs, **extra},
        )

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            default_request_timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            **{**caps.reasoning_kwargs, **extra},
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # O cliente do Ollama não tem max_retries; o timeout vai para o httpx.
        return ChatOllama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            num_predict=cfg.max_tokens,
            temperature=cfg.temperature,
            client_kwargs={"timeout": cfg.timeout},
            **{**caps.reasoning_kwargs, **extra},
        )

    if provider in _OPENAI_COMPATIBLE:
        from langchain_openai import ChatOpenAI

        api_key_setting, base_url = _OPENAI_COMPATIBLE[provider]
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["openai_api_base"] = base_url
        if provider == "openrouter":
            headers = {"X-Title": "S.O.F.I.A"}
            referer = _setting("OPENROUTER_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer
            kwargs["default_headers"] = headers
        return ChatOpenAI(
            model=model,
            openai_api_key=getattr(settings, api_key_setting, ""),
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            # Sem isso o stream não traz usage e o trace de tokens fica zerado.
            stream_usage=True,
            **{**kwargs, **caps.reasoning_kwargs, **extra},
        )

    raise ProviderConfigurationError(f"Provider de chat '{provider}' não suportado.")


# provider -> (setting da chave, base_url; None = padrão da OpenAI)
_OPENAI_COMPATIBLE: dict[str, tuple[str, str | None]] = {
    "openai": ("OPENAI_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "abacusai": ("ABACUSAI_API_KEY", "https://routellm.abacus.ai/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}


# --- Uso de tokens e recusas do provider ---------------------------------------


def extract_usage(msg) -> dict:
    """{"input_tokens", "output_tokens"} a partir de `usage_metadata` (padrão do
    langchain-core), com fallback para `response_metadata` de providers que só
    preenchem ali. Tolerante a None/atributos ausentes: sempre devolve ints."""
    usage = getattr(msg, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        meta = getattr(msg, "response_metadata", None) or {}
        raw = meta.get("token_usage") or meta.get("usage") or {}
        input_tokens = raw.get("prompt_tokens", raw.get("input_tokens"))
        output_tokens = raw.get("completion_tokens", raw.get("output_tokens"))
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}


# finish/block reasons que significam "o provider se recusou a responder".
_REFUSAL_REASONS = {
    "safety",
    "content_filter",
    "blocklist",
    "prohibited_content",
    "spii",
    "image_safety",
    "refusal",
}
_REFUSAL_PHRASES = (
    "content violation",
    "content_filter",
    "content filter",
    "content exists risk",
    "block_reason",
    "flagged as potentially violating",
)


def _refusal_reason_attrs(exc: BaseException) -> bool:
    for attr in ("finish_reason", "block_reason", "stop_reason", "code"):
        value = getattr(exc, attr, None)
        if value is not None and str(getattr(value, "name", value)).lower() in _REFUSAL_REASONS:
            return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        if str(error.get("code") or "").lower() in _REFUSAL_REASONS:
            return True
    return False


def is_provider_refusal(exc: BaseException) -> bool:
    """A exceção é uma recusa/filtro de conteúdo do provider (e não uma falha de
    infraestrutura)? Prefere sinais estruturados (finish_reason/block_reason/code)
    e só então cai em marcadores de texto. "safety" sozinho NÃO basta — aparece em
    erros que nada têm a ver (ex.: "safety settings", "safety_identifier")."""
    seen = 0
    current: BaseException | None = exc
    while current is not None and seen < 3:  # segue __cause__/__context__ (wrappers do LangChain)
        if _refusal_reason_attrs(current):
            return True
        text = str(current).lower()
        if any(phrase in text for phrase in _REFUSAL_PHRASES):
            return True
        if "safety" in text and any(m in text for m in ("block", "finish_reason", "harm_category")):
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


# --- Saída estruturada ----------------------------------------------------------


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    ok: bool
    usage: dict
    error: str | None = None


def _add_usage(total: dict, extra: dict) -> None:
    total["input_tokens"] += extra["input_tokens"]
    total["output_tokens"] += extra["output_tokens"]


def _validate(schema, parsed):
    """Garante uma instância do schema (Pydantic) ou levanta ValueError."""
    if parsed is None:
        raise ValueError("o modelo não devolveu um objeto estruturado")
    if isinstance(schema, type):
        if isinstance(parsed, schema):
            return parsed
        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)
    return parsed


def invoke_structured(
    schema: type[T],
    messages: list[BaseMessage],
    role: Role = "router",
    *,
    default: T | None = None,
    **overrides,
) -> StructuredResult[T]:
    """Chama o modelo do papel pedindo saída estruturada. Usa o método do registro
    de capacidades (`with_structured_output(..., include_raw=True)`), valida com
    Pydantic e, se o parsing falhar, tenta UMA vez de novo com uma mensagem de reparo.

    Nunca levanta exceção de provider: em qualquer falha devolve ok=False,
    value=default e o erro em `error` — o chamador decide o fallback."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        caps = get_capabilities(role, **overrides)
        model = get_chat_model(role, **overrides)
        runnable = model.with_structured_output(schema, method=caps.structured_method, include_raw=True)

        attempt_messages = list(messages)
        error = "falha desconhecida"
        for attempt in range(2):
            result = runnable.invoke(attempt_messages)
            raw = result.get("raw") if isinstance(result, dict) else None
            if raw is not None:
                _add_usage(usage, extract_usage(raw))
            parsed = result.get("parsed") if isinstance(result, dict) else result
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            try:
                if parsing_error is not None:
                    raise ValueError(str(parsing_error))
                return StructuredResult(value=_validate(schema, parsed), ok=True, usage=usage)
            except Exception as exc:  # parsing/validação: vale o retry de reparo
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    attempt_messages = [
                        *messages,
                        HumanMessage(
                            content=(
                                "Sua resposta anterior não seguiu o formato exigido "
                                f"({error[:300]}). Responda de novo apenas com o objeto "
                                "estruturado válido, sem texto extra."
                            )
                        ),
                    ]
        return StructuredResult(value=default, ok=False, usage=usage, error=error)
    except Exception as exc:
        logger.warning("invoke_structured falhou (papel=%s): %s", role, exc)
        return StructuredResult(value=default, ok=False, usage=usage, error=f"{type(exc).__name__}: {exc}")


def get_embedding_model():
    provider = settings.EMBEDDING_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)

    raise ProviderConfigurationError(f"Provider de embedding '{provider}' não suportado.")


def embedding_text_for(kind: Literal["query", "document"], text: str) -> str:
    """Aplica o prefixo de tarefa que alguns modelos de embedding exigem.

    O EmbeddingGemma (via Ollama) foi treinado com prefixos distintos para consulta
    e documento; sem eles a qualidade da busca cai. Os outros modelos recebem o
    texto como está. ATENÇÃO: mudar isto (ou o modelo) altera os vetores gerados, então
    exige `manage.py reprocess_documents` — vetores com e sem prefixo não se comparam."""
    if settings.EMBEDDING_PROVIDER == "ollama" and "embeddinggemma" in settings.EMBEDDING_MODEL.lower():
        prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
        return prefix + text
    return text


def validate_embedding_dimensions(vector) -> None:
    """Falha com mensagem clara se o modelo devolver um vetor de tamanho diferente
    do da coluna `embedding` (EMBEDDING_DIMENSIONS). Sem isso, o erro aparece só
    como falha obscura de SQL ao gravar/buscar."""
    expected = settings.EMBEDDING_DIMENSIONS
    if len(vector) != expected:
        raise ProviderConfigurationError(
            f"O modelo de embedding '{settings.EMBEDDING_MODEL}' devolveu vetores de "
            f"{len(vector)} dimensões, mas EMBEDDING_DIMENSIONS={expected}. Ajuste o "
            "modelo ou a dimensão (trocar a dimensão exige nova migration e reprocessar os materiais)."
        )
