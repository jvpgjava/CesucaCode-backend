from rest_framework.permissions import BasePermission

from apps.accounts.models import User


def coordinated_course_ids(user) -> set[int]:
    return set(user.coordinated_courses.values_list("id", flat=True))


def can_delete_document(user, document) -> bool:
    """Admin sempre. Coordenador só se coordenar TODOS os cursos do material —
    num material compartilhado com um curso que ele não coordena, ele vê (e
    pode editar), mas não exclui, pra não apagar algo de que o outro curso depende."""
    if user.role == User.Role.CS_ADMIN:
        return True
    if user.role != User.Role.CS_COORDINATOR:
        return False
    document_courses = {course.id for course in document.courses.all()}
    return bool(document_courses) and document_courses <= coordinated_course_ids(user)


class CanManageDocuments(BasePermission):
    message = "Apenas CSAdmin ou CSCoordinator podem gerenciar materiais didáticos."

    def has_permission(self, request, view):
        user = request.user
        return bool(
            user
            and user.is_authenticated
            and user.role in (User.Role.CS_ADMIN, User.Role.CS_COORDINATOR)
        )

    def has_object_permission(self, request, view, obj):
        user = request.user
        if user.role == User.Role.CS_ADMIN:
            return True
        if request.method == "DELETE":
            if can_delete_document(user, obj):
                return True
            self.message = (
                "Este material também vale para cursos que você não coordena — "
                "você pode vê-lo, mas só um CSAdmin pode excluí-lo."
            )
            return False
        # ver, editar e reprocessar: basta coordenar algum dos cursos do material
        # (get_documents_queryset já filtra assim; aqui é a defesa em profundidade).
        return obj.courses.filter(id__in=coordinated_course_ids(user)).exists()
