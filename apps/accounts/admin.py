from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from .models import Course, User


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ["name", "code"]
    search_fields = ["name", "code"]


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    ordering = ["full_name"]
    list_display = ["email", "full_name", "role", "rgm", "course", "is_active"]
    list_filter = ["role", "course", "is_active"]
    search_fields = ["email", "full_name", "rgm"]
    filter_horizontal = ["coordinated_courses", "groups", "user_permissions"]

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (
            "Informações pessoais",
            {"fields": ("full_name", "nickname", "role", "rgm", "course", "coordinated_courses")},
        ),
        (
            "Permissões",
            {
                "fields": (
                    "is_active", "is_staff", "is_superuser",
                    "must_change_password", "groups", "user_permissions",
                )
            },
        ),
        ("Datas importantes", {"fields": ("last_login", "created_at", "updated_at")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "full_name", "role", "rgm", "course", "password1", "password2"),
            },
        ),
    )
    readonly_fields = ["created_at", "updated_at"]
