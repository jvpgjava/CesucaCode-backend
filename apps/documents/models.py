import uuid

from django.conf import settings
from django.db import models
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from pgvector.django import HnswIndex, VectorField

from apps.core.models import TimeStampedModel


def document_upload_path(instance, filename):
    # Um material pode valer pra vários cursos (M2M, só definido depois do
    # create), então o caminho não carrega mais o curso.
    return f"documents/{uuid.uuid4()}_{filename}"


class Document(TimeStampedModel):
    class Status(models.TextChoices):
        PROCESSING = "processing", "Processando"
        READY = "ready", "Pronto"
        FAILED = "failed", "Falhou"

    title = models.CharField(max_length=255)
    courses = models.ManyToManyField(
        "accounts.Course",
        related_name="documents",
        help_text="Cursos para os quais o material vale (um ou mais).",
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
    # tsvector (config 'portuguese') de heading (peso A) + content (peso B), para a
    # busca textual da busca híbrida. Preenchido na ingestão (services.py).
    search_vector = SearchVectorField(null=True, blank=True, editable=False)

    class Meta:
        ordering = ["document_id", "index"]
        indexes = [
            # Busca vetorial aproximada por cosseno (pgvector aceita até 2000 dimensões
            # em índice HNSW; acima disso troque por halfvec).
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
            GinIndex(fields=["search_vector"], name="chunk_search_vector_gin"),
        ]
        constraints = [
            models.UniqueConstraint(fields=["document", "index"], name="unique_chunk_index_per_document")
        ]

    def __str__(self):
        return f"{self.document.title} #{self.index}"


class Disciplina(TimeStampedModel):
    """Disciplina extraída dos cabeçalhos de plano de ensino durante a ingestão.

    Alimenta a grade estruturada (tool `consultar_grade` e o bloco "Disciplinas
    cadastradas" da rota direta), que responde "quais disciplinas existem" sem depender
    de o retrieval trazer todos os planos. Uma linha por (curso, nome); havendo vários
    planos da mesma disciplina, vale o do período letivo mais recente. É apagada e
    refeita a cada (re)processamento do documento."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="disciplinas")
    course = models.ForeignKey(
        "accounts.Course",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="disciplinas",
        help_text="Curso do plano (resolvido pelos cursos do documento); vazio se ambíguo.",
    )
    nome = models.CharField(max_length=255)
    semestre = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Semestre curricular no curso.")
    carga_horaria = models.PositiveSmallIntegerField(null=True, blank=True, help_text="C/H semestral, em horas.")
    periodo_letivo = models.CharField(max_length=10, blank=True, help_text="Período do plano, ex.: 2025/1.")
    index_inicio = models.PositiveIntegerField(
        null=True, blank=True, help_text="Índice do primeiro chunk do plano dentro do documento."
    )

    class Meta:
        ordering = ["semestre", "nome"]
        verbose_name = "disciplina"
        verbose_name_plural = "disciplinas"

    def __str__(self):
        return f"{self.nome} ({self.semestre or '?'}º sem.)"
