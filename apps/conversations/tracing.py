"""Registro de observabilidade de cada resposta (`MessageTrace`).

O `TraceRecorder` acompanha uma resposta do começo ao fim: tempo por etapa,
tokens, primeiro token, rota e metadados do retrieval. Ele nunca guarda texto do
aluno nem da resposta (só metadados), para respeitar a pseudonimização do TCLE/LGPD.

Gravar o trace é secundário: qualquer falha aqui só é logada, nunca derruba o chat.
"""

import logging
import time
from contextlib import contextmanager

from django.conf import settings
from django.db import transaction

from .models import MessageTrace

logger = logging.getLogger(__name__)

_SETTABLE = {
    "route", "intent", "route_source", "course_code", "models",
    "chunk_ids", "distances", "guard_flags", "pipeline_version",
}


class TraceRecorder:
    def __init__(self):
        self._start = time.monotonic()
        self._first_token_at: float | None = None
        self.steps: list[dict] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.fields: dict = {"pipeline_version": settings.PIPELINE_VERSION}

    @contextmanager
    def step(self, type: str, name: str, **meta):
        """Mede uma etapa. O dict devolvido pode receber metadados durante o
        bloco (ex.: `s["n_chunks"] = 3`); a duração é gravada ao sair, mesmo
        com exceção."""
        extra = dict(meta)
        begin = time.monotonic()
        try:
            yield extra
        finally:
            self.add_step(type, name, round((time.monotonic() - begin) * 1000), **extra)

    def add_step(self, type: str, name: str, ms: int, **meta):
        self.steps.append({"type": type, "name": name, "ms": int(ms), **meta})

    def add_usage(self, usage: dict | None):
        """Soma tokens de uma chamada ao LLM. Aceita None e valores ausentes."""
        if not usage:
            return
        try:
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
        except (TypeError, ValueError, AttributeError):
            logger.debug("usage inválido ignorado: %r", usage)

    def mark_first_token(self):
        if self._first_token_at is None:
            self._first_token_at = time.monotonic()

    def set(self, **fields):
        unknown = set(fields) - _SETTABLE
        if unknown:
            raise TypeError(f"Campos de trace desconhecidos: {sorted(unknown)}")
        self.fields.update(fields)

    def finish(self, message, error: str | None = None) -> MessageTrace | None:
        """Grava o MessageTrace da mensagem do assistente. Devolve None se não
        há mensagem ou se a gravação falhar (só loga)."""
        if message is None:
            return None
        end = time.monotonic()
        ttft = (
            round((self._first_token_at - self._start) * 1000)
            if self._first_token_at is not None
            else None
        )
        try:
            with transaction.atomic():
                return MessageTrace.objects.create(
                    message=message,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    latency_ms=round((end - self._start) * 1000),
                    ttft_ms=ttft,
                    steps=self.steps,
                    error=(error or "")[:2000],
                    **self.fields,
                )
        except Exception:
            logger.exception("Falha ao gravar o MessageTrace da mensagem %s", getattr(message, "pk", None))
            return None
