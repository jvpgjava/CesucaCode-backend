"""Eventos tipados do chat e sua serialização para SSE.

O pipeline (`services.send_message`) emite objetos destes tipos; só a camada de
transporte (`to_sse`) conhece o formato de fio. Os tokens saem no formato padrão
do SSE (sem `event:`), o que mantém a compatibilidade com clientes antigos que só
leem `data: {"content": ...}`.

Os rótulos de status vêm SEMPRE de `STATUS_LABELS`: nunca carregam argumentos de
ferramentas, títulos de documentos nem texto do aluno.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class StatusEvent:
    step: str
    label: str


@dataclass
class TokenEvent:
    content: str


@dataclass
class SuggestionsEvent:
    items: list[str] = field(default_factory=list)


@dataclass
class MetaEvent:
    user_message_id: int | None
    route: str | None = None


@dataclass
class DoneEvent:
    message_id: int | None


@dataclass
class ErrorEvent:
    message: str


STATUS_LABELS = {
    "routing": "Entendendo sua pergunta",
    "searching": "Buscando nos materiais do curso",
    "reading": "Lendo {n} trechos",
    "web": "Consultando referências externas",
    "thinking": "Analisando as informações",
    "writing": "Organizando a resposta",
    "checking": "Conferindo a resposta",
}

# Nome do evento SSE por tipo; None = evento padrão (sem linha `event:`).
_EVENT_NAMES = {
    StatusEvent: "status",
    TokenEvent: None,
    SuggestionsEvent: "suggestions",
    MetaEvent: "meta",
    DoneEvent: "done",
    ErrorEvent: "error",
}


def status(step: str, **fmt) -> StatusEvent:
    """Cria um StatusEvent com o rótulo fixo da etapa (`fmt` só preenche
    placeholders como `{n}`). Etapa desconhecida é erro de programação."""
    label = STATUS_LABELS[step].format(**fmt)
    return StatusEvent(step=step, label=label)


def to_sse(event) -> str:
    try:
        name = _EVENT_NAMES[type(event)]
    except KeyError:
        raise TypeError(f"Evento desconhecido: {type(event).__name__}") from None
    data = json.dumps(asdict(event), ensure_ascii=False)
    prefix = f"event: {name}\n" if name else ""
    return f"{prefix}data: {data}\n\n"
