from rest_framework import serializers

from apps.accounts.models import Course, User
from apps.accounts.serializers import CourseSerializer

from . import extraction, services
from .models import Document, DocumentChunk
from .permissions import can_delete_document, coordinated_course_ids

MAX_FILE_SIZE_MB = 20
# Campos de gestão que o estudante não vê no DocumentSerializer.
STUDENT_HIDDEN_FIELDS = ("file", "uploaded_by_name", "processing_error")


class DocumentSerializer(serializers.ModelSerializer):
    courses = CourseSerializer(many=True, read_only=True)
    uploaded_by_name = serializers.CharField(source="uploaded_by.full_name", read_only=True)
    chunk_count = serializers.SerializerMethodField()
    can_delete = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id", "title", "courses", "file", "uploaded_by_name",
            "status", "processing_error", "chunk_count", "can_delete", "created_at",
        ]
        read_only_fields = fields

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Estudante só precisa saber quais materiais existem: o arquivo, quem
        # enviou e o erro de processamento são de gestão (ficam só para
        # admin/coordenador).
        request = self.context.get("request")
        if request is None or request.user.role == User.Role.CS_STUDENT:
            for field in STUDENT_HIDDEN_FIELDS:
                data.pop(field, None)
        return data

    def get_chunk_count(self, obj):
        return obj.chunks.count()

    def get_can_delete(self, obj):
        return can_delete_document(self.context["request"].user, obj)


class DocumentUploadSerializer(serializers.ModelSerializer):
    courses = serializers.SlugRelatedField(
        slug_field="code", queryset=Course.objects.all(), many=True, allow_empty=False
    )

    class Meta:
        model = Document
        fields = ["id", "title", "courses", "file", "status", "processing_error"]
        read_only_fields = ["id", "status", "processing_error"]

    def validate_courses(self, courses):
        user = self.context["request"].user
        if user.role == User.Role.CS_COORDINATOR:
            allowed = coordinated_course_ids(user)
            if any(course.id not in allowed for course in courses):
                raise serializers.ValidationError(
                    "Você só pode adicionar materiais para cursos que coordena."
                )
        return courses

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


class DocumentUpdateSerializer(serializers.ModelSerializer):
    courses = serializers.SlugRelatedField(
        slug_field="code", queryset=Course.objects.all(), many=True, allow_empty=False
    )

    class Meta:
        model = Document
        fields = ["id", "title", "courses"]
        read_only_fields = ["id"]

    def validate_courses(self, courses):
        user = self.context["request"].user
        if user.role != User.Role.CS_COORDINATOR:
            return courses

        allowed = coordinated_course_ids(user)
        current = list(self.instance.courses.all())
        current_ids = {course.id for course in current}
        if any(course.id not in allowed and course.id not in current_ids for course in courses):
            raise serializers.ValidationError("Você só pode incluir cursos que coordena.")
        # Cursos que o coordenador não coordena não podem ser removidos por ele.
        kept = [course for course in current if course.id not in allowed]
        return kept + [course for course in courses if course.id not in {c.id for c in kept}]


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ["id", "index", "content", "heading"]
        read_only_fields = fields
