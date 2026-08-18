from django.conf import settings
from django.core.mail import send_mail

from .models import User


def send_temporary_password_email(user: User, password: str, *, is_reset: bool = False) -> None:
    if user.role == User.Role.CS_STUDENT:
        login_label, identifier = "RGM", user.rgm
    else:
        login_label, identifier = "e-mail", user.email

    if is_reset:
        subject = "Sua senha do CesucaCode foi redefinida"
        intro = "Sua senha no CesucaCode foi redefinida por um administrador."
    else:
        subject = "Bem-vindo(a) ao CesucaCode"
        intro = "Sua conta no CesucaCode foi criada."

    message = (
        f"Olá, {user.full_name}!\n\n"
        f"{intro}\n\n"
        f"Login ({login_label}): {identifier}\n"
        f"Senha temporária: {password}\n\n"
        "Essa senha é temporária — no primeiro login o sistema vai pedir "
        "para você trocá-la antes de liberar o resto do sistema.\n\n"
        "Se você não esperava este e-mail, procure a coordenação do seu curso."
    )

    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=False,
    )
