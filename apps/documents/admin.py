from django.contrib import admin

from .models import Document, DocumentChunk


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ["index", "content"]
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "course", "status", "uploaded_by", "created_at"]
    list_filter = ["status", "course"]
    search_fields = ["title"]
    readonly_fields = ["status", "processing_error", "created_at", "updated_at"]
    inlines = [DocumentChunkInline]
