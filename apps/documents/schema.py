from drf_spectacular.utils import extend_schema, extend_schema_view

from .serializers import DocumentUpdateSerializer
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
            "Um material pode valer para vários cursos (`courses`). CSAdmin vê todos; "
            "CSCoordinator vê os que incluem algum curso que coordena; CSStudent vê os "
            "que incluem o próprio curso. `can_delete` indica se o usuário pode excluir."
        ),
        tags=DOCUMENTS,
    ),
)(DocumentListView)

extend_schema_view(
    post=extend_schema(
        summary="Enviar material didático",
        description=(
            "Envia um arquivo (PDF, DOCX, PPTX, MD ou TXT) para um ou mais cursos (`courses`, "
            "lista de códigos, ex.: `cc`, `ads`) e extrai/divide em pedaços "
            "(chunks) em background. A resposta já volta com `status: \"processing\"`; "
            "consulte `GET /{id}/` para acompanhar até virar `ready` ou `failed` "
            "(com o erro em `processing_error`). Restrito a CSAdmin (qualquer "
            "curso) e CSCoordinator (só para cursos que coordena)."
        ),
        tags=DOCUMENTS,
    ),
)(DocumentUploadView)

extend_schema_view(
    get=extend_schema(summary="Ver um material didático", tags=DOCUMENTS),
    patch=extend_schema(
        summary="Editar título/cursos de um material",
        description="Não reenvia o arquivo nem reprocessa os chunks — só metadados. "
        "Para reprocessar o conteúdo, use o endpoint de reprocessar. CSCoordinator "
        "edita se coordenar algum dos cursos do material, mas não remove os cursos "
        "que não coordena.",
        request=DocumentUpdateSerializer,
        tags=DOCUMENTS,
    ),
    delete=extend_schema(
        summary="Remover um material didático",
        description="CSAdmin sempre; CSCoordinator só se coordenar todos os cursos do "
        "material (`can_delete`). Num material compartilhado com curso que não coordena, "
        "ele vê e edita, mas não exclui (403).",
        tags=DOCUMENTS,
    ),
)(DocumentDetailView)

extend_schema_view(
    get=extend_schema(
        summary="Ver os chunks extraídos de um material",
        description="Útil para conferir a qualidade da extração/divisão usada na busca do chat (RAG).",
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
