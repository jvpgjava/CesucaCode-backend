from drf_spectacular.utils import (
    OpenApiResponse,
    extend_schema,
    extend_schema_view,
    inline_serializer,
)
from rest_framework import serializers
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    AccountListView,
    BulkImportStudentsView,
    ChangePasswordView,
    CourseListView,
    CreateCoordinatorView,
    CreateStudentView,
    LoginView,
    MeView,
    ResetPasswordView,
)

AUTH = ["Autenticação"]
ACCOUNTS = ["Contas (CSAdmin)"]

extend_schema_view(
    post=extend_schema(
        summary="Login",
        description=(
            "Login único para todos os papéis. `identifier` aceita e-mail "
            "(CSAdmin/CSCoordinator) ou RGM (CSStudent) — detectado pela "
            "presença de `@`. Se o identificador não existir, o erro aponta "
            "isso especificamente (dizendo se procurou por e-mail ou RGM); "
            "se existir mas a senha estiver errada, o erro é genérico (não "
            "confirma que o e-mail/RGM é válido)."
        ),
        tags=AUTH,
    ),
)(LoginView)

extend_schema_view(
    post=extend_schema(summary="Renovar access token", tags=AUTH),
)(TokenRefreshView)

extend_schema_view(
    get=extend_schema(summary="Ver meus dados", tags=AUTH),
)(MeView)

extend_schema_view(
    post=extend_schema(
        summary="Trocar minha senha",
        description="Sempre acessível, mesmo quando `must_change_password` for `true`.",
        tags=AUTH,
    ),
)(ChangePasswordView)

extend_schema_view(
    get=extend_schema(summary="Listar cursos", tags=AUTH),
)(CourseListView)

extend_schema_view(
    get=extend_schema(
        summary="Listar contas (alunos e coordenadores)",
        description=(
            "Lista todas as contas de CSStudent e CSCoordinator (CSAdmin não "
            "aparece aqui — não é criado/gerenciado por esta API). Aceita "
            "`?search=` (busca em nome, e-mail e RGM) e `?role=cs_student` ou "
            "`?role=cs_coordinator`. Restrito a CSAdmin."
        ),
        tags=ACCOUNTS,
    ),
)(AccountListView)

extend_schema_view(
    post=extend_schema(
        summary="Criar estudante",
        description=(
            "Cria uma conta de estudante. A senha é gerada aleatoriamente e "
            "enviada por e-mail — nunca aparece na resposta desta API. "
            "`email_sent` na resposta indica se o envio deu certo; se vier "
            "`false`, use 'Redefinir senha' para gerar e reenviar. Restrito "
            "a CSAdmin."
        ),
        tags=ACCOUNTS,
    ),
)(CreateStudentView)

extend_schema_view(
    post=extend_schema(
        summary="Criar coordenador",
        description=(
            "Cria uma conta de coordenador, associada a um ou mais cursos. "
            "Mesma política de senha do endpoint de criar estudante (aleatória, "
            "só por e-mail). Restrito a CSAdmin."
        ),
        tags=ACCOUNTS,
    ),
)(CreateCoordinatorView)

extend_schema_view(
    post=extend_schema(
        summary="Importar estudantes em massa (CSV)",
        description=(
            "Recebe um arquivo CSV (colunas: full_name, rgm, email, course "
            "e, opcionalmente, nickname) e cria um estudante por linha, cada "
            "um com senha aleatória própria enviada por e-mail. Não "
            "interrompe no primeiro erro — devolve quantas contas foram "
            "criadas, quantas falharam na validação (`errors`) e quantas "
            "tiveram falha só no envio do e-mail (`email_failures_count`). "
            "Restrito a CSAdmin."
        ),
        request={
            "multipart/form-data": inline_serializer(
                name="BulkImportStudentsRequest",
                fields={"file": serializers.FileField(help_text="Arquivo CSV")},
            ),
        },
        responses={
            201: OpenApiResponse(description="Todas as linhas foram importadas com sucesso."),
            207: OpenApiResponse(description="Importação parcial — veja `errors` na resposta."),
        },
        tags=ACCOUNTS,
    ),
)(BulkImportStudentsView)

extend_schema_view(
    post=extend_schema(
        summary="Redefinir senha de um usuário",
        description=(
            "Gera uma nova senha aleatória para o usuário, marca "
            "`must_change_password=True` e envia a nova senha por e-mail — "
            "nunca aparece na resposta desta API. Restrito a CSAdmin."
        ),
        request=None,
        tags=ACCOUNTS,
    ),
)(ResetPasswordView)
