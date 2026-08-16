from drf_spectacular.utils import extend_schema, extend_schema_view

from .views import (
    DocumentChunksView,
    DocumentDetailView,
    DocumentListView,
    DocumentReprocessView,
    DocumentUploadView,
)

DOCUMENTS = ["Materiais Didáticos"]

extend_schema_view(
    get=extend_schema(
        summary="Listar materiais didáticos",
        description=(
            "CSAdmin vê todos os materiais; CSCoordinator vê os dos cursos que "
            "coordena; CSStudent vê os do próprio curso."
        ),
        tags=DOCUMENTS,
    ),
)(DocumentListView)

extend_schema_view(
    post=extend_schema(
        summary="Enviar material didático",
        description=(
            "Envia um arquivo (PDF, DOCX, PPTX ou TXT), extrai o texto e divide "
            "em pedaços (chunks) prontos para indexação. Processamento é "
            "síncrono — a resposta já vem com o status final (`ready` ou "
            "`failed`, com o erro em `processing_error`). Restrito a CSAdmin "
            "(qualquer curso) e CSCoordinator (só dos cursos que coordena)."
        ),
        tags=DOCUMENTS,
    ),
)(DocumentUploadView)

extend_schema_view(
    get=extend_schema(summary="Ver um material didático", tags=DOCUMENTS),
    delete=extend_schema(summary="Remover um material didático", tags=DOCUMENTS),
)(DocumentDetailView)

extend_schema_view(
    get=extend_schema(
        summary="Ver os chunks extraídos de um material",
        description="Útil para conferir a qualidade da extração/divisão antes da Parte 3 (embeddings).",
        tags=DOCUMENTS,
    ),
)(DocumentChunksView)

extend_schema_view(
    post=extend_schema(
        summary="Reprocessar um material didático",
        description="Apaga os chunks existentes e refaz a extração/divisão a partir do arquivo já enviado.",
        request=None,
        tags=DOCUMENTS,
    ),
)(DocumentReprocessView)
