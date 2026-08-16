from rest_framework import serializers

from apps.accounts.models import Course, User
from apps.accounts.serializers import CourseSerializer

from . import extraction, services
from .models import Document, DocumentChunk

MAX_FILE_SIZE_MB = 20


class DocumentSerializer(serializers.ModelSerializer):
    course = CourseSerializer(read_only=True)
    uploaded_by_name = serializers.CharField(source="uploaded_by.full_name", read_only=True)
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id", "title", "course", "file", "uploaded_by_name",
            "status", "processing_error", "chunk_count", "created_at",
        ]
        read_only_fields = fields

    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentUploadSerializer(serializers.ModelSerializer):
    course = serializers.SlugRelatedField(slug_field="code", queryset=Course.objects.all())

    class Meta:
        model = Document
        fields = ["id", "title", "course", "file", "status", "processing_error"]
        read_only_fields = ["id", "status", "processing_error"]

    def validate_course(self, course):
        user = self.context["request"].user
        if user.role == User.Role.CS_COORDINATOR and not user.coordinated_courses.filter(id=course.id).exists():
            raise serializers.ValidationError("Você só pode adicionar materiais para cursos que coordena.")
        return course

    def validate_file(self, file):
        ext = extraction.get_extension(file.name)
        if ext not in extraction.SUPPORTED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Formato não suportado. Use: {', '.join(sorted(extraction.SUPPORTED_EXTENSIONS))}."
            )
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise serializers.ValidationError(f"Arquivo maior que {MAX_FILE_SIZE_MB}MB.")
        return file

    def create(self, validated_data):
        request = self.context["request"]
        return services.create_document(uploaded_by=request.user, **validated_data)


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ["id", "index", "content"]
        read_only_fields = fields
