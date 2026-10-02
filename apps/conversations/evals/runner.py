"""Execução do golden set contra o chat.

O chat é consumido SOMENTE por `services.send_message` (gerador de eventos) e o
`MessageTrace` da resposta fornece rota, tokens, latência e chunks. Cada caso roda
com usuário e conversa temporários, apagados no fim.
"""

import contextlib
import json
import secrets
import time
import uuid
from pathlib import Path

from django.conf import settings
from django.test import override_settings

from apps.accounts.models import Course, User
from apps.conversations import services
from apps.conversations.events import DoneEvent, ErrorEvent, MetaEvent, SuggestionsEvent, TokenEvent
from apps.conversations.models import Conversation, Message
from apps.documents.models import Document, DocumentChunk

from . import judge as judge_module
from .metrics import score_case

# persona -> (papel, código do curso do aluno, curso coordenado)
PERSONA_SPECS = {
    "student_cc": (User.Role.CS_STUDENT, "cc", None),
    "student_ads": (User.Role.CS_STUDENT, "ads", None),
    "admin": (User.Role.CS_ADMIN, None, None),
    "coordinator_cc": (User.Role.CS_COORDINATOR, None, "cc"),
}

# Flags que definem uma variante do pipeline. Algumas podem ainda não existir em
# settings (outra frente as adiciona): override_settings cria o atributo mesmo assim.
FLAG_NAMES = [
    "CHAT_ROUTER_ENABLED",
    "CHAT_AGENT_ENABLED",
    "RAG_HYBRID_ENABLED",
    "CHAT_SUFFICIENCY_CHECK_ENABLED",
    "CHAT_FOLLOWUPS_ENABLED",
    "RAG_RERANK_ENABLED",
    "PIPELINE_VERSION",
]
VARIANTS = {
    "v0": {
        "CHAT_ROUTER_ENABLED": False,
        "CHAT_AGENT_ENABLED": False,
        "RAG_HYBRID_ENABLED": False,
        "CHAT_SUFFICIENCY_CHECK_ENABLED": False,
        "CHAT_FOLLOWUPS_ENABLED": False,
        "PIPELINE_VERSION": "v0",
    },
    "v1": {
        "CHAT_ROUTER_ENABLED": True,
        "CHAT_AGENT_ENABLED": False,
        "RAG_HYBRID_ENABLED": True,
        "CHAT_SUFFICIENCY_CHECK_ENABLED": True,
        "CHAT_FOLLOWUPS_ENABLED": True,
        "PIPELINE_VERSION": "v1",
    },
    "v2": {
        "CHAT_ROUTER_ENABLED": True,
        "CHAT_AGENT_ENABLED": True,
        "RAG_HYBRID_ENABLED": True,
        "CHAT_SUFFICIENCY_CHECK_ENABLED": True,
        "CHAT_FOLLOWUPS_ENABLED": True,
        "PIPELINE_VERSION": "v2",
    },
    "current": {},
}


def effective_flags() -> dict:
    """Valores das flags no momento (None = a setting ainda não existe)."""
    return {name: getattr(settings, name, None) for name in FLAG_NAMES}


@contextlib.contextmanager
def variant_settings(variant: str):
    """Aplica as flags da variante enquanto o bloco roda (`current` não mexe em nada)."""
    overrides = VARIANTS[variant]
    if not overrides:
        yield
        return
    with override_settings(**overrides):
        yield


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def create_persona_user(persona: str) -> User:
    """Cria o usuário temporário da persona (senha aleatória, nunca exibida)."""
    role, course_code, coordinated = PERSONA_SPECS[persona]
    kwargs = {"role": role}
    if course_code:
        kwargs["course"] = Course.objects.get(code=course_code)
        kwargs["rgm"] = str(secrets.randbelow(10**9)).zfill(9)
    user = User.objects.create_user(
        email=f"{_unique('eval')}@example.invalid",
        password=secrets.token_urlsafe(24),
        full_name=f"Eval {persona}",
        **kwargs,
    )
    if coordinated:
        user.coordinated_courses.add(Course.objects.get(code=coordinated))
    return user


def document_titles() -> list[str]:
    """Títulos de todos os materiais (banco + manifest do seed) para detectar vazamento."""
    titles = set(Document.objects.values_list("title", flat=True))
    manifest = Path(__file__).resolve().parents[2] / "documents" / "seed_materials" / "manifest.json"
    with contextlib.suppress(OSError, ValueError):
        titles.update(entry["title"] for entry in json.loads(manifest.read_text(encoding="utf-8")))
    return sorted(titles)


def _read_trace(message: Message | None) -> dict:
    empty = {
        "route": None, "intent": None, "route_source": None, "pipeline_version": None, "course_code": None,
        "input_tokens": 0, "output_tokens": 0, "trace_latency_ms": None, "trace_ttft_ms": None,
        "chunk_ids": [], "guard_flags": [], "trace_error": "", "steps": [],
    }
    if message is None:
        return empty
    try:
        trace = message.trace
    except Exception:  # sem trace (ex.: pipeline que ainda não grava)
        return empty
    return {
        "route": trace.route or None,
        "intent": trace.intent or None,
        "route_source": trace.route_source or None,
        "pipeline_version": trace.pipeline_version or None,
        "course_code": trace.course_code or None,
        "input_tokens": trace.input_tokens or 0,
        "output_tokens": trace.output_tokens or 0,
        "trace_latency_ms": trace.latency_ms,
        "trace_ttft_ms": trace.ttft_ms,
        "chunk_ids": list(trace.chunk_ids or []),
        "guard_flags": list(trace.guard_flags or []),
        "trace_error": trace.error or "",
        "steps": [
            {"type": step.get("type"), "name": step.get("name"), "ms": step.get("ms")}
            for step in (trace.steps or [])
            if isinstance(step, dict)
        ],
    }


def _chunk_texts(chunk_ids: list[int]) -> list[str]:
    """Conteúdo dos chunks na ordem em que foram recuperados."""
    if not chunk_ids:
        return []
    by_id = {c.id: c.content for c in DocumentChunk.objects.filter(id__in=chunk_ids)}
    return [by_id[i] for i in chunk_ids if i in by_id]


def run_case(case, *, judge: bool = False, titles=(), send=None) -> dict:
    """Roda um caso e devolve o dicionário de resultado (respostas, trace e métricas)."""
    send = send or services.send_message
    case_dict = case.to_dict() if hasattr(case, "to_dict") else dict(case)

    user = create_persona_user(case_dict["persona"])
    response_parts: list[str] = []
    suggestions: list[str] = []
    event_route = None
    message_id = None
    error = None
    latency_s = ttft_s = None
    try:
        conversation = Conversation.objects.create(user=user, title="eval")
        for turn in case_dict["history"]:
            Message.objects.create(conversation=conversation, role=turn["role"], content=turn["content"])

        started = time.monotonic()
        stream = send(conversation, case_dict["question"])
        try:
            for event in stream:
                if isinstance(event, TokenEvent):
                    if ttft_s is None:
                        ttft_s = time.monotonic() - started
                    response_parts.append(event.content)
                elif isinstance(event, MetaEvent):
                    event_route = event.route
                elif isinstance(event, SuggestionsEvent):
                    suggestions = list(event.items)
                elif isinstance(event, DoneEvent):
                    message_id = event.message_id
                elif isinstance(event, ErrorEvent):
                    error = event.message
        except Exception as exc:  # falha no meio do stream: guarda o parcial e segue
            error = f"{type(exc).__name__}: {exc}"
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        latency_s = time.monotonic() - started

        message = None
        if message_id:
            message = Message.objects.filter(pk=message_id).select_related("trace").first()
        if message is None:
            message = (
                conversation.messages.filter(role=Message.Role.ASSISTANT).select_related("trace").order_by("-id").first()
            )
        trace = _read_trace(message)
        chunk_texts = _chunk_texts(trace["chunk_ids"])
    except Exception as exc:  # setup/leitura do trace: o caso conta como erro, a rodada continua
        error = error or f"{type(exc).__name__}: {exc}"
        trace = _read_trace(None)
        chunk_texts = []
    finally:
        user.delete()  # cascata: conversa, mensagens e traces

    response = "".join(response_parts)
    route = trace["route"] or event_route
    if ttft_s is None and trace["trace_ttft_ms"] is not None:
        ttft_s = trace["trace_ttft_ms"] / 1000
    if latency_s is None and trace["trace_latency_ms"] is not None:
        latency_s = trace["trace_latency_ms"] / 1000

    verdict, judge_usage, judge_error = None, {}, None
    if judge and response.strip() and not error:
        verdict, judge_usage, judge_error = judge_module.judge_case(case_dict, response, chunk_texts)

    scores = score_case(
        case_dict,
        response,
        chunk_texts=chunk_texts,
        route=route,
        intent=trace["intent"],
        document_titles=titles,
        judge=verdict,
        errored=bool(error),
    )
    return {
        "id": case_dict["id"],
        "tags": case_dict["tags"],
        "persona": case_dict["persona"],
        "expect": case_dict["expect"],
        "question": case_dict["question"],
        "response": response,
        "route": route,
        "intent": trace["intent"],
        "route_source": trace["route_source"],
        "pipeline_version": trace["pipeline_version"],
        "error": error or trace["trace_error"] or None,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "input_tokens": trace["input_tokens"],
        "output_tokens": trace["output_tokens"],
        "chunk_ids": trace["chunk_ids"],
        "n_chunks": len(trace["chunk_ids"]),
        "guard_flags": trace["guard_flags"],
        "suggestions": suggestions,
        "steps": trace["steps"],
        "judge": verdict,
        "judge_error": judge_error,
        "judge_usage": judge_usage,
        **scores,
    }


def run_cases(cases, *, judge: bool = False, send=None, progress=None) -> list[dict]:
    """Roda os casos em sequência. `progress(i, total, resultado)` é chamado a cada caso."""
    titles = document_titles()
    results = []
    for index, case in enumerate(cases, start=1):
        result = run_case(case, judge=judge, titles=titles, send=send)
        results.append(result)
        if progress:
            progress(index, len(cases), result)
    return results
