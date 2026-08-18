from django.urls import path
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

urlpatterns = [
    path("login/", LoginView.as_view(), name="auth-login"),
    path("login/refresh/", TokenRefreshView.as_view(), name="auth-login-refresh"),
    path("me/", MeView.as_view(), name="auth-me"),
    path("change-password/", ChangePasswordView.as_view(), name="auth-change-password"),
    path("courses/", CourseListView.as_view(), name="auth-courses"),
    path("accounts/", AccountListView.as_view(), name="auth-list-accounts"),
    path("accounts/students/", CreateStudentView.as_view(), name="auth-create-student"),
    path("accounts/students/import/", BulkImportStudentsView.as_view(), name="auth-import-students"),
    path("accounts/coordinators/", CreateCoordinatorView.as_view(), name="auth-create-coordinator"),
    path("accounts/<int:pk>/reset-password/", ResetPasswordView.as_view(), name="auth-reset-password"),
]
