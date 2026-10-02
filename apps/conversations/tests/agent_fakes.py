"""Fakes compartilhados dos testes do agente e das ferramentas (sem rede/LLM)."""

import sys
from dataclasses import dataclass
from types import ModuleType

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Modelo de agente roteirizado: devolve as respostas em ordem e aceita
    `bind_tools` (devolve ele mesmo). `seen` guarda as mensagens de cada invoke."""

    seen: list = []
    bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai_calls(*calls, usage=None):
    return AIMessage(content="", tool_calls=list(calls), usage_metadata=usage)


@dataclass
class FakeChunk:
    chunk_id: int
    document_id: int
    heading: str
    content: str
    score: float = 0.9
    distance: float | None = 0.1
    order: int | None = None


class FakeRegistry:
    """Mesma API do RefRegistry do contrato; `add` devolve só os trechos NOVOS."""

    def __init__(self):
        self._by_ref = {}
        self.chunk_ids = []

    def add(self, chunks):
        added = []
        for chunk in chunks:
            if chunk.chunk_id in self.chunk_ids:
                continue
            ref = f"T{len(self.chunk_ids) + 1}"
            self.chunk_ids.append(chunk.chunk_id)
            self._by_ref[ref] = chunk
            added.append((ref, chunk))
        return added

    def get(self, ref):
        return self._by_ref.get(ref)


def fake_format_context(pairs):
    parts = []
    for ref, chunk in pairs:
        head = f"[{ref} · seção: {chunk.heading}]" if chunk.heading else f"[{ref}]"
        parts.append(f"{head}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def make_retrieval_stub(search_results=None, neighbor_results=None):
    """Módulo `retrieval` falso. `search_results`: lista de listas (uma por chamada
    de search, em ordem; a última se repete). `neighbor_results`: lista de trechos."""
    stub = ModuleType("apps.conversations.retrieval")
    stub.RetrievedChunk = FakeChunk
    stub.RefRegistry = FakeRegistry
    stub.format_context = fake_format_context
    stub.search_calls = []
    stub.neighbor_calls = []
    results = list(search_results or [])

    def search(user, query, *, course_code=None, top_k=6, curriculum=False):
        stub.search_calls.append({"query": query, "course_code": course_code, "top_k": top_k})
        index = min(len(stub.search_calls) - 1, len(results) - 1) if results else None
        return list(results[index]) if index is not None else []

    def neighbors(user, chunk_id, window=1):
        stub.neighbor_calls.append({"chunk_id": chunk_id, "window": window})
        return list(neighbor_results or [])

    stub.search = search
    stub.neighbors = neighbors
    return stub


def install_retrieval_stub(monkeypatch, stub):
    """`from . import retrieval` prefere o atributo do pacote ao sys.modules:
    troca os dois."""
    import apps.conversations as package

    monkeypatch.setitem(sys.modules, "apps.conversations.retrieval", stub)
    monkeypatch.setattr(package, "retrieval", stub, raising=False)
