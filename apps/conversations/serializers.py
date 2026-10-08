from rest_framework import serializers

from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "feedback", "feedback_reason", "created_at"]
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ["id", "title", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class SendMessageSerializer(serializers.Serializer):
    content = serializers.CharField(min_length=1, max_length=8000)
    regenerate = serializers.BooleanField(
        required=False,
        default=False,
        help_text="true = apaga a última resposta (e a pergunta que a gerou) e processa `content` como nova.",
    )


class MessageFeedbackSerializer(serializers.Serializer):
    feedback = serializers.ChoiceField(
        choices=[1, -1],
        allow_null=True,
        help_text="1 = útil, -1 = não útil, null = remove a avaliação.",
    )
    reason = serializers.ChoiceField(
        choices=Message.FeedbackReason.choices,
        required=False,
        help_text="Motivo do 👎: incorreta, incompleta, nao_entendeu, fora_do_curso ou outro.",
    )
    comment = serializers.CharField(
        max_length=500, required=False, allow_blank=True, help_text="Comentário opcional do 👎 (até 500 caracteres)."
    )

    def to_internal_value(self, data):
        # `rating` era o nome antigo do campo; segue aceito como alias.
        if hasattr(data, "get") and "feedback" not in data and "rating" in data:
            data = {**dict(data.items()), "feedback": data["rating"]}
        return super().to_internal_value(data)

    def validate(self, attrs):
        if "feedback" not in attrs:
            raise serializers.ValidationError({"feedback": "Este campo é obrigatório (use null para remover)."})
        return attrs
