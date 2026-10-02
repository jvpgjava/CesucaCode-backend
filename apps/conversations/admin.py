from django.contrib import admin

from .models import Conversation, Message, MessageTrace


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ["role", "content", "feedback", "feedback_reason", "created_at"]
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "created_at", "updated_at"]
    search_fields = ["title", "user__email"]
    readonly_fields = ["created_at", "updated_at"]
    inlines = [MessageInline]


@admin.register(MessageTrace)
class MessageTraceAdmin(admin.ModelAdmin):
    """Traces das respostas: somente leitura (são registro de observabilidade)."""

    list_display = [
        "message", "created_at", "pipeline_version", "route", "route_source",
        "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "has_flags", "has_error",
    ]
    list_filter = ["pipeline_version", "route", "route_source", "created_at"]
    search_fields = ["message__conversation__title", "error"]
    list_select_related = ["message"]

    @admin.display(boolean=True, description="Flags da guarda")
    def has_flags(self, obj):
        return bool(obj.guard_flags)

    @admin.display(boolean=True, description="Erro")
    def has_error(self, obj):
        return bool(obj.error)

    def get_readonly_fields(self, request, obj=None):
        return [field.name for field in self.model._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
