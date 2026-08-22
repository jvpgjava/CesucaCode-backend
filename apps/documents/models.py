import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField

from apps.core.models import TimeStampedModel


def document_upload_path(instance, filename):
    return f"documents/{instance.course.code}/{uuid.uuid4()}_{filename}"


class Document(TimeStampedModel):
    class Status(models.TextChoices):
        PROCESSING = "processing", "Processando"
        READY = "ready", "Pronto"
        FAILED = "failed", "Falhou"

    title = models.CharField(max_length=255)
    course = models.ForeignKey(
        "accounts.Course", on_delete=models.PROTECT, related_name="documents"
    )
    file = models.FileField(upload_to=document_upload_path)
    uploaded_by = models.ForeignKey(
        "accounts.User", on_delete=models.PROTECT, related_name="uploaded_documents"
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PROCESSING)
    processing_error = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class DocumentChunk(TimeStampedModel):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index = models.PositiveIntegerField()
    content = models.TextField()
    heading = models.CharField(
        max_length=500,
        blank=True,
        help_text="Caminho de seções do documento a que o chunk pertence (ex.: '5. Modelo ER > 5.1 Entidades'), quando o formato permite extrair isso.",
    )
    embedding = VectorField(dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True)

    class Meta:
        ordering = ["document_id", "index"]
        constraints = [
            models.UniqueConstraint(fields=["document", "index"], name="unique_chunk_index_per_document")
        ]

    def __str__(self):
        return f"{self.document.title} #{self.index}"
