from django.contrib import admin

from .models import Disciplina, Document, DocumentChunk


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ["index", "content"]
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "course_names", "status", "uploaded_by", "created_at"]
    list_filter = ["status", "courses"]
    search_fields = ["title"]
    readonly_fields = ["status", "processing_error", "created_at", "updated_at"]
    inlines = [DocumentChunkInline]

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("courses")

    @admin.display(description="Cursos")
    def course_names(self, obj):
        return ", ".join(course.code for course in obj.courses.all())


@admin.register(Disciplina)
class DisciplinaAdmin(admin.ModelAdmin):
    list_display = ["nome", "semestre", "carga_horaria", "periodo_letivo", "course", "document"]
    list_filter = ["course", "semestre"]
    search_fields = ["nome"]
