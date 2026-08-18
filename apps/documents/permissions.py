from rest_framework.permissions import BasePermission

from apps.accounts.models import User


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
        return user.coordinated_courses.filter(id=obj.course_id).exists()
