from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedChatModel(BaseChatModel):
    """LLM fake que emite exatamente os pedaços informados (sem rede). Se `fail_with`
    for definido, a exceção é levantada depois dos pedaços (ou antes, se não houver)."""

    pieces: list[str] = []
    usage: dict | None = None
    fail_with: Exception | None = None
    seen: list = []  # mensagens recebidas em cada chamada de stream

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = "".join(self.pieces)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        last = len(self.pieces) - 1
        for i, piece in enumerate(self.pieces):
            usage = self.usage if (i == last and self.usage) else None
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, usage_metadata=usage))
        if self.fail_with is not None:
            raise self.fail_with
