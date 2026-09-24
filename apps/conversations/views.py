import json
import logging

from django.http import StreamingHttpResponse
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services
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
        return Response({"suggestions": services.build_suggestions(request.user)})


class MessageFeedbackView(APIView):
    def patch(self, request, pk, message_id):
        conversation = generics.get_object_or_404(get_conversations_queryset(request.user), pk=pk)
        message = generics.get_object_or_404(
            conversation.messages.filter(role=Message.Role.ASSISTANT), pk=message_id
        )
        serializer = MessageFeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message.feedback = serializer.validated_data["rating"]
        message.save(update_fields=["feedback", "updated_at"])
        return Response(MessageSerializer(message).data)


class SendMessageView(APIView):
    def post(self, request, pk):
        conversation = generics.get_object_or_404(get_conversations_queryset(request.user), pk=pk)
        serializer = SendMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user_text = serializer.validated_data["content"]

        def event_stream():
            try:
                for piece in services.send_message(conversation, user_text):
                    yield f"data: {json.dumps({'content': piece})}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception:
                logger.exception("Falha ao gerar resposta para a conversa %s", conversation.id)
                error_payload = json.dumps({"error": "Falha ao gerar resposta. Tente novamente."})
                yield f"event: error\ndata: {error_payload}\n\n"

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
