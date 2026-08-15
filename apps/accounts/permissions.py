from rest_framework.permissions import BasePermission

from .models import User


class IsCSAdmin(BasePermission):
    message = "Apenas administradores (CSAdmin) podem realizar esta ação."

    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and user.role == User.Role.CS_ADMIN)


class PasswordIsCurrent(BasePermission):
    message = "Sua senha precisa ser trocada antes de continuar. Use /api/auth/change-password/."

    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and not user.must_change_password)
