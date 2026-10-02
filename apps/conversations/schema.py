from drf_spectacular.utils import OpenApiResponse, extend_schema, extend_schema_view

from .views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationMessagesView,
    MessageFeedbackView,
    SendMessageView,
    SuggestionsView,
)
from .serializers import MessageFeedbackSerializer

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
    patch=extend_schema(
        summary="Renomear uma conversa",
        description="Atualiza só o título. Título vazio é aceito (volta a mostrar como "
        "'Nova conversa' no cliente).",
        tags=CONVERSATIONS,
    ),
    put=extend_schema(summary="Renomear uma conversa", tags=CONVERSATIONS),
    delete=extend_schema(summary="Excluir uma conversa", tags=CONVERSATIONS),
)(ConversationDetailView)

extend_schema_view(
    get=extend_schema(summary="Ver o histórico de mensagens de uma conversa", tags=CONVERSATIONS),
)(ConversationMessagesView)

extend_schema_view(
    post=extend_schema(
        summary="Enviar uma mensagem (streaming)",
        description=(
            "Classifica a intenção da mensagem, busca os trechos de material didático mais "
            "relevantes (RAG, escopado pelo mesmo critério de permissão dos materiais), "
            "monta o prompt com o histórico da "
            "conversa e transmite a resposta do modelo em tempo real via "
            "[Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events) "
            "(`Content-Type: text/event-stream`), não como um JSON único. Eventos: `meta` "
            "(`user_message_id` e `route`: meta, recusa, direta, composta, pedagogica ou "
            "clarificacao), `status` (`step`, `label` — rótulo fixo da etapa), "
            "tokens no formato padrão `data: {\"content\": \"...\"}`, `suggestions` (`items`), "
            "`done` (`message_id` da resposta salva) e `error` (`message`). Com "
            "`regenerate: true`, apaga a última resposta (e a pergunta que a gerou) e processa "
            "`content` como nova. A mensagem do usuário e a resposta do assistente são salvas "
            "no banco (a parcial, se o cliente desconectar). Limite por usuário: 20/min e "
            "300/dia (429 com `Retry-After`)."
        ),
        responses={200: OpenApiResponse(description="Stream de eventos (text/event-stream).")},
        tags=CONVERSATIONS,
    ),
)(SendMessageView)

extend_schema_view(
    get=extend_schema(
        summary="Sugestões de primeira mensagem",
        description=(
            "Perguntas prontas pro usuário clicar (grade curricular, disciplinas, "
            "materiais disponíveis). São fixas; a resposta sempre vem dos materiais enviados."
        ),
        tags=CONVERSATIONS,
    ),
)(SuggestionsView)

extend_schema_view(
    patch=extend_schema(
        summary="Avaliar uma resposta (👍/👎)",
        description=(
            "`feedback`: 1 = útil, -1 = não útil, null = remove a avaliação (`rating` segue aceito "
            "como alias). No 👎, `reason` (incorreta, incompleta, nao_entendeu, fora_do_curso, outro) "
            "e `comment` (até 500 caracteres) são opcionais; ao mudar para 1 ou null, são apagados. "
            "Só mensagens do assistente."
        ),
        request=MessageFeedbackSerializer,
        tags=CONVERSATIONS,
    ),
)(MessageFeedbackView)
