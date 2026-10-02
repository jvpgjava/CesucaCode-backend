"""Registro de capacidades por provider/modelo.

O que muda de um provider para outro e o resto do sistema precisa saber: qual método
de saída estruturada funciona, se há tool calling (e em paralelo) e como ligar o
"thinking". Fica em um lugar só para que trocar de provider seja só mexer no .env
(e rodar `manage.py test_ai_provider --check tools,structured,stream`)."""

from dataclasses import dataclass, field, replace
from typing import Literal

from django.conf import settings

StructuredMethod = Literal["json_schema", "function_calling", "json_mode"]
STRUCTURED_METHODS = ("json_schema", "function_calling", "json_mode")


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model: str
    structured_method: StructuredMethod
    supports_tools: bool
    parallel_tool_calls: bool
    # kwargs extras do construtor para ligar/limitar thinking quando o provider suporta.
    reasoning_kwargs: dict = field(default_factory=dict)


# Padrão por provider. DeepSeek, OpenRouter e Abacus (gateways) não garantem o
# `json_schema` estrito em todos os modelos; function_calling é o denominador comum.
# parallel_tool_calls é conservador (False) onde o suporte varia por modelo.
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "gemini": {"structured_method": "json_schema", "supports_tools": True, "parallel_tool_calls": True},
    "openai": {"structured_method": "json_schema", "supports_tools": True, "parallel_tool_calls": True},
    "claude": {"structured_method": "function_calling", "supports_tools": True, "parallel_tool_calls": True},
    "ollama": {"structured_method": "json_schema", "supports_tools": True, "parallel_tool_calls": False},
    "deepseek": {"structured_method": "function_calling", "supports_tools": True, "parallel_tool_calls": False},
    "abacusai": {"structured_method": "function_calling", "supports_tools": True, "parallel_tool_calls": False},
    "openrouter": {"structured_method": "function_calling", "supports_tools": True, "parallel_tool_calls": False},
}

# Exceções por prefixo de modelo (vence o maior prefixo que casar, sem diferenciar caixa).
_MODEL_OVERRIDES: list[tuple[str, str, dict]] = [
    # Os modelos de raciocínio do DeepSeek no Ollama não expõem tool calling.
    ("ollama", "deepseek-r1", {"supports_tools": False}),
]


def _thinking_budget() -> int | None:
    raw = str(getattr(settings, "LLM_THINKING_BUDGET", "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def reasoning_kwargs(provider: str) -> dict:
    """kwargs extras para controlar thinking. Hoje só o Gemini (`thinking_budget`,
    via LLM_THINKING_BUDGET; sem a variável, não setamos nada e vale o padrão do
    modelo). Nos demais, vazio."""
    if provider == "gemini":
        budget = _thinking_budget()
        if budget is not None:
            return {"thinking_budget": budget}
    return {}


def resolve_capabilities(provider: str, model: str) -> ModelCapabilities:
    base = _PROVIDER_DEFAULTS.get(provider)
    if base is None:
        # Provider desconhecido: o factory já levanta ProviderConfigurationError;
        # aqui só devolvemos o denominador comum para não quebrar quem consulta.
        base = {"structured_method": "function_calling", "supports_tools": False, "parallel_tool_calls": False}
    caps = ModelCapabilities(provider=provider, model=model, reasoning_kwargs=reasoning_kwargs(provider), **base)

    name = (model or "").lower()
    matches = [(prefix, over) for prov, prefix, over in _MODEL_OVERRIDES if prov == provider and name.startswith(prefix)]
    if matches:
        caps = replace(caps, **max(matches, key=lambda m: len(m[0]))[1])

    forced = str(getattr(settings, "LLM_STRUCTURED_METHOD", "") or "").strip().lower()
    if forced in STRUCTURED_METHODS:
        caps = replace(caps, structured_method=forced)
    return caps
