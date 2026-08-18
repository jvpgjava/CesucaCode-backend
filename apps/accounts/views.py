from rest_framework import generics, permissions, status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from . import services
from .models import Course, User
from .permissions import IsCSAdmin, PasswordIsCurrent
from .serializers import (
    ChangePasswordSerializer,
    CoordinatorCreateSerializer,
    CourseSerializer,
    LoginSerializer,
    StudentCreateSerializer,
    UserSerializer,
)


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]

        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "must_change_password": user.must_change_password,
            }
        )


class MeView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class ChangePasswordView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ChangePasswordSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"detail": "Senha alterada com sucesso."})


class CourseListView(generics.ListAPIView):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer


class CreateStudentView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = StudentCreateSerializer
    permission_classes = [IsCSAdmin, PasswordIsCurrent]


class CreateCoordinatorView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = CoordinatorCreateSerializer
    permission_classes = [IsCSAdmin, PasswordIsCurrent]


class BulkImportStudentsView(APIView):
    permission_classes = [IsCSAdmin, PasswordIsCurrent]
    parser_classes = [MultiPartParser]

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response(
                {"detail": "Envie um arquivo CSV no campo 'file'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = services.bulk_import_students(uploaded_file)
        except services.AccountCreationError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        email_failures = sum(1 for user in result.created if not getattr(user, "_email_sent", False))
        payload = {
            "created_count": len(result.created),
            "failed_count": len(result.errors),
            "email_failures_count": email_failures,
            "errors": result.errors,
        }
        response_status = status.HTTP_201_CREATED if not result.errors else status.HTTP_207_MULTI_STATUS
        return Response(payload, status=response_status)


class ResetPasswordView(APIView):
    permission_classes = [IsCSAdmin, PasswordIsCurrent]

    def post(self, request, pk):
        user = generics.get_object_or_404(User, pk=pk)
        email_sent = services.reset_password(user)
        detail = (
            f"Nova senha gerada para {user.email} e enviada por e-mail."
            if email_sent
            else (
                f"Nova senha gerada para {user.email}, mas o envio do e-mail "
                "falhou — chame este endpoint de novo para tentar reenviar."
            )
        )
        return Response({"detail": detail, "email_sent": email_sent})
