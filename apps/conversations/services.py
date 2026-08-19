import logging
from pathlib import Path

from django.conf import settings
from pgvector.django import CosineDistance

from apps.ai_providers import services as ai_providers
from apps.documents.models import DocumentChunk
from apps.documents.views import get_documents_queryset

from .models import Conversation, Message

logger = logging.getLogger(__name__)

TOP_K_CHUNKS = 5


def get_system_prompt() -> str:
    path = Path(settings.SYSTEM_PROMPT_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"SYSTEM_PROMPT_PATH aponta para um arquivo que não existe: {path}"
        )
    return path.read_text(encoding="utf-8").strip()


def get_accessible_chunks_queryset(user):
    return DocumentChunk.objects.filter(
        document__in=get_documents_queryset(user),
        document__status="ready",
        embedding__isnull=False,
    ).select_related("document", "document__course")


def retrieve_context(user, query_text, top_k=TOP_K_CHUNKS):
    query_vector = ai_providers.get_embedding_model().embed_query(query_text)
    return list(
        get_accessible_chunks_queryset(user)
        .annotate(distance=CosineDistance("embedding", query_vector))
        .order_by("distance")[:top_k]
    )


def build_context_block(chunks) -> str:
    if not chunks:
        return ""
    parts = [f"[Fonte: {chunk.document.title}]\n{chunk.content}" for chunk in chunks]
    return "\n\n---\n\n".join(parts)


def build_messages(conversation: Conversation, user_text: str, context_block: str):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=get_system_prompt())]

    for past in conversation.messages.all():
        if past.role == Message.Role.USER:
            messages.append(HumanMessage(content=past.content))
        else:
            messages.append(AIMessage(content=past.content))

    if context_block:
        user_content = (
            f"Contexto dos materiais didáticos:\n\n{context_block}\n\n---\n\nPergunta: {user_text}"
        )
    else:
        user_content = user_text

    messages.append(HumanMessage(content=user_content))
    return messages


def send_message(conversation: Conversation, user_text: str):
    context_chunks = retrieve_context(conversation.user, user_text)
    context_block = build_context_block(context_chunks)
    messages = build_messages(conversation, user_text, context_block)

    Message.objects.create(conversation=conversation, role=Message.Role.USER, content=user_text)
    if not conversation.title:
        conversation.title = user_text[:80]
    conversation.save(update_fields=["title", "updated_at"])

    chat_model = ai_providers.get_chat_model()
    full_response = []
    try:
        for chunk in chat_model.stream(messages):
            piece = chunk.content
            if piece:
                full_response.append(piece)
                yield piece
    finally:
        if full_response:
            Message.objects.create(
                conversation=conversation,
                role=Message.Role.ASSISTANT,
                content="".join(full_response),
            )
