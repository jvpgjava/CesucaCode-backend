import logging
import math

from django.http import StreamingHttpResponse
from rest_framework import generics
from rest_framework.exceptions import Throttled
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

from . import services
from .events import ErrorEvent, to_sse
from .models import Conversation, Message
from .serializers import (
    ConversationSerializer,
    MessageFeedbackSerializer,
    MessageSerializer,
    SendMessageSerializer,
)

logger = logging.getLogger(__name__)


def get_conversations_queryset(user):
    return Conversation.objects.filter(user=user)


class ConversationListCreateView(generics.ListCreateAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return get_conversations_queryset(self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ConversationDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return get_conversations_queryset(self.request.user)


class ConversationMessagesView(generics.ListAPIView):
    serializer_class = MessageSerializer
    pagination_class = None

    def get_queryset(self):
        conversation = generics.get_object_or_404(
            get_conversations_queryset(self.request.user), pk=self.kwargs["pk"]
        )
        return conversation.messages.all()


class SuggestionsView(APIView):
    def get(self, request):
        return Response({"suggestions": services.build_suggestions()})


class MessageFeedbackView(APIView):
    def patch(self, request, pk, message_id):
        conversation = generics.get_object_or_404(get_conversations_queryset(request.user), pk=pk)
        message = generics.get_object_or_404(
            conversation.messages.filter(role=Message.Role.ASSISTANT), pk=message_id
        )
        serializer = MessageFeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        message.feedback = data["feedback"]
        if message.feedback == -1:
            # Motivo/comentário só mudam se vierem no corpo (permite avaliar primeiro
            # e detalhar depois sem apagar o que já foi enviado).
            if "reason" in data:
                message.feedback_reason = data["reason"]
            if "comment" in data:
                message.feedback_comment = data["comment"].strip()
        else:
            message.feedback_reason = ""
            message.feedback_comment = ""
        message.save(update_fields=["feedback", "feedback_reason", "feedback_comment", "updated_at"])
        return Response(MessageSerializer(message).data)


class ChatDailyThrottle(UserRateThrottle):
    """Teto diário de mensagens por usuário (taxa `chat_daily`)."""

    scope = "chat_daily"


class SendMessageView(APIView):
    # Dois limites por usuário: rajada por minuto (`chat`) e teto diário (`chat_daily`).
    throttle_classes = [ScopedRateThrottle, ChatDailyThrottle]
    throttle_scope = "chat"

    def throttled(self, request, wait):
        seconds = max(1, math.ceil(wait or 1))
        raise Throttled(
            wait=wait,
            detail=f"Você atingiu o limite de mensagens. Tente novamente em {seconds} segundos.",
        )

    def post(self, request, pk):
        conversation = generics.get_object_or_404(get_conversations_queryset(request.user), pk=pk)
        serializer = SendMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user_text = serializer.validated_data["content"]
        regenerate = serializer.validated_data["regenerate"]

        def event_stream():
            try:
                for event in services.send_message(conversation, user_text, regenerate=regenerate):
                    yield to_sse(event)
            except Exception:
                # O pipeline já converte falhas em ErrorEvent; isto cobre o inesperado.
                logger.exception("Falha ao gerar resposta para a conversa %s", conversation.id)
                yield to_sse(ErrorEvent(services.ERROR_MESSAGE))

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
