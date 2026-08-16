import csv
import io
import logging

from django.utils.crypto import get_random_string

from . import emails
from .models import User

logger = logging.getLogger(__name__)

REQUIRED_IMPORT_COLUMNS = {"full_name", "rgm", "email", "course"}


class AccountCreationError(Exception):
    pass


def generate_temporary_password() -> str:
    return get_random_string(20)


def _notify_password(user: User, password: str, *, is_reset: bool) -> bool:
    try:
        emails.send_temporary_password_email(user, password, is_reset=is_reset)
        return True
    except Exception:
        logger.exception("Falha ao enviar e-mail de senha para %s", user.email)
        return False


def create_student(*, email, full_name, rgm, course, nickname="") -> User:
    password = generate_temporary_password()
    user = User(
        email=email.strip().lower(),
        full_name=full_name,
        nickname=nickname,
        rgm=rgm,
        course=course,
        role=User.Role.CS_STUDENT,
        must_change_password=True,
    )
    user.set_password(password)
    user.save()

    user._email_sent = _notify_password(user, password, is_reset=False)
    return user


def create_coordinator(*, email, full_name, coordinated_courses, nickname="") -> User:
    password = generate_temporary_password()
    user = User(
        email=email.strip().lower(),
        full_name=full_name,
        nickname=nickname,
        role=User.Role.CS_COORDINATOR,
        must_change_password=True,
    )
    user.set_password(password)
    user.save()
    user.coordinated_courses.set(coordinated_courses)

    user._email_sent = _notify_password(user, password, is_reset=False)
    return user


def reset_password(user: User) -> bool:
    password = generate_temporary_password()
    user.set_password(password)
    user.must_change_password = True
    user.save(update_fields=["password", "must_change_password", "updated_at"])
    return _notify_password(user, password, is_reset=True)


class BulkImportResult:
    def __init__(self):
        self.created: list[User] = []
        self.errors: list[dict] = []


def bulk_import_students(csv_file) -> BulkImportResult:
    from .serializers import StudentCreateSerializer

    try:
        text_stream = io.TextIOWrapper(csv_file, encoding="utf-8-sig")
        reader = csv.DictReader(text_stream)
        fieldnames = reader.fieldnames or []
    except UnicodeDecodeError as exc:
        raise AccountCreationError("Não foi possível ler o arquivo. Envie um CSV em UTF-8.") from exc

    if not REQUIRED_IMPORT_COLUMNS.issubset(set(fieldnames)):
        raise AccountCreationError(
            "O CSV precisa ter as colunas: "
            f"{', '.join(sorted(REQUIRED_IMPORT_COLUMNS))} (nickname é opcional)."
        )

    result = BulkImportResult()
    for line_number, row in enumerate(reader, start=2):
        serializer = StudentCreateSerializer(
            data={
                "full_name": (row.get("full_name") or "").strip(),
                "rgm": (row.get("rgm") or "").strip(),
                "email": (row.get("email") or "").strip(),
                "course": (row.get("course") or "").strip().lower(),
                "nickname": (row.get("nickname") or "").strip(),
            }
        )
        if serializer.is_valid():
            result.created.append(serializer.save())
        else:
            result.errors.append(
                {"row": line_number, "rgm": row.get("rgm"), "errors": serializer.errors}
            )
    return result
