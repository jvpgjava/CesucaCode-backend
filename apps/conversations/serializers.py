from rest_framework import serializers

from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "feedback", "created_at"]
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ["id", "title", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class SendMessageSerializer(serializers.Serializer):
    content = serializers.CharField(min_length=1, max_length=8000)


class MessageFeedbackSerializer(serializers.Serializer):
    rating = serializers.ChoiceField(
        choices=[1, -1],
        allow_null=True,
        help_text="1 = útil, -1 = não útil, null = remove a avaliação.",
    )
