from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.core.exceptions import ValidationError
from django.db import models

from apps.core.models import TimeStampedModel

from .managers import UserManager
from .validators import rgm_validator


class Course(TimeStampedModel):
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        self.code = self.code.strip().lower()
        super().save(*args, **kwargs)


class User(AbstractBaseUser, PermissionsMixin, TimeStampedModel):
    class Role(models.TextChoices):
        CS_ADMIN = "cs_admin", "CSAdmin"
        CS_COORDINATOR = "cs_coordinator", "CSCoordinator"
        CS_STUDENT = "cs_student", "CSStudent"

    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=150)
    nickname = models.CharField(max_length=50, blank=True, help_text="Apelido, opcional.")
    rgm = models.CharField(
        max_length=20,
        unique=True,
        null=True,
        blank=True,
        validators=[rgm_validator],
        help_text="Obrigatório e exclusivo para estudantes (CSStudent).",
    )
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.CS_STUDENT)
    course = models.ForeignKey(
        Course,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="students",
        help_text="Curso do estudante. Não se aplica a CSAdmin/CSCoordinator.",
    )
    coordinated_courses = models.ManyToManyField(
        Course,
        blank=True,
        related_name="coordinators",
        help_text="Cursos coordenados por este usuário. Só se aplica a CSCoordinator.",
    )

    must_change_password = models.BooleanField(
        default=False,
        help_text="Se True, o usuário precisa trocar a senha antes de usar o resto da API.",
    )
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["full_name"]

    class Meta:
        ordering = ["full_name"]

    def __str__(self):
        return f"{self.full_name} <{self.email}>"

    def clean(self):
        super().clean()
        if self.role == self.Role.CS_STUDENT and not self.rgm:
            raise ValidationError({"rgm": "RGM é obrigatório para estudantes."})
        if self.role != self.Role.CS_STUDENT and self.rgm:
            raise ValidationError({"rgm": "RGM é exclusivo de estudantes."})

    @property
    def is_admin_role(self):
        return self.role == self.Role.CS_ADMIN

    @property
    def is_coordinator(self):
        return self.role == self.Role.CS_COORDINATOR

    @property
    def is_student(self):
        return self.role == self.Role.CS_STUDENT
