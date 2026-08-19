import json
import logging

from django.http import StreamingHttpResponse
from rest_framework import generics
from rest_framework.views import APIView

from . import services
from .models import Conversation
from .serializers import ConversationSerializer, MessageSerializer, SendMessageSerializer

logger = logging.getLogger(__name__)


def get_conversations_queryset(user):
    return Conversation.objects.filter(user=user)


class ConversationListCreateView(generics.ListCreateAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return get_conversations_queryset(self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ConversationDetailView(generics.RetrieveDestroyAPIView):
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
