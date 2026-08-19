from drf_spectacular.utils import OpenApiResponse, extend_schema, extend_schema_view

from .views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationMessagesView,
    SendMessageView,
)

CONVERSATIONS = ["Conversas (Chat)"]

extend_schema_view(
    get=extend_schema(
        summary="Listar minhas conversas",
        description="Só as conversas do próprio usuário — conversas não são compartilhadas entre papéis.",
        tags=CONVERSATIONS,
    ),
    post=extend_schema(
        summary="Criar uma conversa nova",
        description="Cria uma conversa vazia. O título é preenchido sozinho a partir da primeira mensagem.",
        tags=CONVERSATIONS,
    ),
)(ConversationListCreateView)

extend_schema_view(
    get=extend_schema(summary="Ver uma conversa", tags=CONVERSATIONS),
    delete=extend_schema(summary="Excluir uma conversa", tags=CONVERSATIONS),
)(ConversationDetailView)

extend_schema_view(
    get=extend_schema(summary="Ver o histórico de mensagens de uma conversa", tags=CONVERSATIONS),
)(ConversationMessagesView)

extend_schema_view(
    post=extend_schema(
        summary="Enviar uma mensagem (streaming)",
        description=(
            "Busca os trechos de material didático mais relevantes (RAG, escopado pelo "
            "mesmo critério de permissão dos materiais), monta o prompt com o histórico da "
            "conversa e transmite a resposta do modelo em tempo real via "
            "[Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events) "
            "(`Content-Type: text/event-stream`), não como um JSON único. Cada evento vem no "
            "formato `data: {\"content\": \"...\"}`; ao final, um evento `event: done` fecha o "
            "stream, ou `event: error` se algo falhar. A mensagem do usuário e a resposta "
            "completa do assistente são salvas no banco."
        ),
        responses={200: OpenApiResponse(description="Stream de eventos (text/event-stream).")},
        tags=CONVERSATIONS,
    ),
)(SendMessageView)
