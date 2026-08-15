from django.contrib.auth import password_validation
from rest_framework import serializers
from rest_framework.validators import UniqueValidator

from . import services
from .models import Course, User
from .validators import rgm_validator


class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = ["id", "code", "name"]


class UserSerializer(serializers.ModelSerializer):
    course = CourseSerializer(read_only=True)
    coordinated_courses = CourseSerializer(many=True, read_only=True)

    class Meta:
        model = User
        fields = [
            "id", "email", "full_name", "nickname", "rgm", "role",
            "course", "coordinated_courses", "must_change_password", "created_at",
        ]
        read_only_fields = fields


class StudentCreateSerializer(serializers.ModelSerializer):
    course = serializers.SlugRelatedField(slug_field="code", queryset=Course.objects.all())
    rgm = serializers.CharField(
        max_length=20,
        validators=[
            rgm_validator,
            UniqueValidator(queryset=User.objects.all(), message="Já existe um usuário com esse RGM."),
        ],
    )
    email_sent = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "email", "full_name", "nickname", "rgm", "course", "email_sent"]
        read_only_fields = ["id"]

    def get_email_sent(self, obj):
        return getattr(obj, "_email_sent", None)

    def create(self, validated_data):
        return services.create_student(**validated_data)


class CoordinatorCreateSerializer(serializers.ModelSerializer):
    coordinated_courses = serializers.SlugRelatedField(
        slug_field="code", queryset=Course.objects.all(), many=True
    )
    email_sent = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "email", "full_name", "nickname", "coordinated_courses", "email_sent"]
        read_only_fields = ["id"]

    def get_email_sent(self, obj):
        return getattr(obj, "_email_sent", None)

    def create(self, validated_data):
        return services.create_coordinator(**validated_data)


class LoginSerializer(serializers.Serializer):
    identifier = serializers.CharField(help_text="E-mail ou RGM, dependendo do papel do usuário.")
    password = serializers.CharField(write_only=True, style={"input_type": "password"})

    def validate(self, attrs):
        identifier = attrs["identifier"].strip()
        is_email = "@" in identifier

        if is_email:
            user = User.objects.filter(email__iexact=identifier).first()
            not_found_message = "E-mail não encontrado."
        else:
            user = User.objects.filter(rgm=identifier).first()
            not_found_message = "RGM não encontrado."

        if user is None:
            raise serializers.ValidationError({"identifier": not_found_message})

        if not user.is_active:
            raise serializers.ValidationError(
                {"identifier": "Conta desativada. Procure a coordenação/administração."}
            )

        if not user.check_password(attrs["password"]):
            raise serializers.ValidationError(
                {"non_field_errors": ["E-mail/RGM ou senha incorretos."]}
            )

        attrs["user"] = user
        return attrs


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True, style={"input_type": "password"})
    new_password = serializers.CharField(write_only=True, style={"input_type": "password"})

    def validate_old_password(self, value):
        user = self.context["request"].user
        if not user.check_password(value):
            raise serializers.ValidationError("Senha atual incorreta.")
        return value

    def validate_new_password(self, value):
        password_validation.validate_password(value, user=self.context["request"].user)
        return value

    def save(self, **kwargs):
        user = self.context["request"].user
        user.set_password(self.validated_data["new_password"])
        user.must_change_password = False
        user.save(update_fields=["password", "must_change_password", "updated_at"])
        return user
