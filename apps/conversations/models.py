from django.db import models

from apps.core.models import TimeStampedModel


class Conversation(TimeStampedModel):
    user = models.ForeignKey(
        "accounts.User", on_delete=models.CASCADE, related_name="conversations"
    )
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Estado do roteador na conversa: `course` (curso resolvido: cc/ads, preenchido uma vez), "
            "`hint_level` (nível atual da escada de dicas, 1 a 3) e `last_intent`."
        ),
    )

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or f"Conversa #{self.pk}"


class Message(TimeStampedModel):
    class Role(models.TextChoices):
        USER = "user", "Usuário"
        ASSISTANT = "assistant", "Assistente"

    class FeedbackReason(models.TextChoices):
        INCORRECT = "incorreta", "Incorreta"
        INCOMPLETE = "incompleta", "Incompleta"
        NOT_UNDERSTOOD = "nao_entendeu", "Não entendeu"
        OUT_OF_COURSE = "fora_do_curso", "Fora do curso"
        OTHER = "outro", "Outro"

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=20, choices=Role.choices)
    content = models.TextField()
    feedback = models.SmallIntegerField(
        null=True,
        blank=True,
        choices=[(1, "Útil"), (-1, "Não útil")],
        help_text="Avaliação do usuário sobre a resposta (só em mensagens do assistente).",
    )

    feedback_reason = models.CharField(
        max_length=20,
        choices=FeedbackReason.choices,
        blank=True,
        help_text="Motivo do 👎 (só quando feedback = -1). Alimenta o golden set e os relatórios.",
    )
    feedback_comment = models.TextField(blank=True, help_text="Comentário opcional do 👎 (até 500 caracteres).")

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"


class MessageTrace(TimeStampedModel):
    """Observabilidade de uma resposta do assistente: rota, modelos, tokens,
    latência, etapas, trechos recuperados e flags da guarda de saída. Visível só
    no admin. Não guarda texto do aluno nem da resposta (pseudonimização/LGPD)."""

    class RouteSource(models.TextChoices):
        L0 = "l0", "Regras (L0)"
        L1 = "l1", "Modelo (L1)"
        FALLBACK = "fallback", "Fallback"
        LEGACY = "legacy", "Pipeline legado"

    message = models.OneToOneField(Message, on_delete=models.CASCADE, related_name="trace")
    pipeline_version = models.CharField(max_length=20, blank=True)
    route = models.CharField(max_length=40, blank=True)
    intent = models.CharField(max_length=40, blank=True)
    route_source = models.CharField(max_length=20, choices=RouteSource.choices, blank=True)
    course_code = models.CharField(max_length=20, blank=True)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(default=0)
    ttft_ms = models.PositiveIntegerField(null=True, blank=True, help_text="Tempo até o primeiro token.")
    steps = models.JSONField(default=list, blank=True, help_text="Lista de {type, name, ms, ...}.")
    chunk_ids = models.JSONField(default=list, blank=True)
    distances = models.JSONField(default=list, blank=True)
    guard_flags = models.JSONField(default=list, blank=True)
    error = models.TextField(blank=True)
    # Nome exigido pelo contrato; fica por último porque, no corpo da classe,
    # sombreia o módulo `models` (usado pelos campos acima).
    models = models.JSONField(default=dict, blank=True, help_text="Modelo usado por papel: {papel: modelo}.")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Trace da mensagem #{self.message_id} ({self.route or 'sem rota'})"
