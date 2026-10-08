"""Grade estruturada: consulta das `Disciplina` visíveis a um usuário.

Fica separada de `views.py`/`services.py` para ser usada pela tool `consultar_grade` do
agente e pelo bloco "Disciplinas cadastradas" da rota direta (mesma função, mesma saída).
A visibilidade segue `get_documents_queryset` (documentos prontos que o usuário vê) e,
além disso, o curso da disciplina: um material compartilhado por CC e ADS não mostra ao
aluno de ADS as disciplinas do CC.
"""

from django.db.models import Q

from apps.accounts.models import User

from .disciplinas import normalize_key, period_key
from .models import Disciplina

GRADE_SOURCE_NOTE = (
    "Lista extraída dos planos de ensino disponíveis. O semestre e a carga horária podem variar "
    "conforme o período letivo do plano (vale o mais recente de cada disciplina): confirme com a coordenação."
)


def _visible_disciplinas(user):
    from .views import get_documents_queryset  # import tardio: views importa services

    queryset = Disciplina.objects.filter(
        document__in=get_documents_queryset(user), document__status="ready"
    ).select_related("course")
    if user.role == User.Role.CS_ADMIN:
        return queryset
    if user.role == User.Role.CS_COORDINATOR:
        allowed = user.coordinated_courses.all()
    else:
        allowed = [user.course_id] if user.course_id else []
    return queryset.filter(Q(course__isnull=True) | Q(course__in=allowed))


def list_disciplinas(user, curso: str | None = None, semestre: int | None = None) -> list[Disciplina]:
    """Disciplinas visíveis ao usuário, sem repetir (curso, nome) entre documentos (fica
    a de período mais recente), ordenadas por curso, semestre e nome.

    `curso` ("cc"/"ads") filtra pelo curso da disciplina (ou, se ela não tem curso, pelos
    cursos do documento); qualquer outro valor/None não filtra. `semestre` filtra o
    semestre curricular."""
    queryset = _visible_disciplinas(user)
    if curso in ("cc", "ads"):
        queryset = queryset.filter(
            Q(course__code=curso) | Q(course__isnull=True, document__courses__code=curso)
        ).distinct()
    if semestre is not None:
        queryset = queryset.filter(semestre=semestre)

    best: dict[tuple, Disciplina] = {}
    for item in queryset:
        key = (item.course_id, normalize_key(item.nome))
        current = best.get(key)
        if current is None or period_key(item.periodo_letivo) > period_key(current.periodo_letivo):
            best[key] = item
    return sorted(
        best.values(),
        key=lambda d: (d.course.name if d.course else "", d.semestre is None, d.semestre or 0, normalize_key(d.nome)),
    )


def format_grade(items: list[Disciplina]) -> str:
    """Texto para o LLM: disciplinas agrupadas por curso e semestre + aviso de fonte.
    Nunca cita nome de arquivo nem título de documento. Vazio se não houver itens."""
    if not items:
        return ""
    lines: list[str] = []
    current_course: object = object()
    current_semester: object = object()
    multiple_courses = len({d.course_id for d in items}) > 1
    for item in items:
        if multiple_courses and item.course_id != current_course:
            current_course, current_semester = item.course_id, object()
            lines.append(f"{item.course.name if item.course else 'Curso não identificado'}:")
        if item.semestre != current_semester:
            current_semester = item.semestre
            lines.append(f"{item.semestre}º semestre:" if item.semestre else "Semestre não informado:")
        workload = f" ({item.carga_horaria} h)" if item.carga_horaria else ""
        lines.append(f"- {item.nome}{workload}")
    lines.append(GRADE_SOURCE_NOTE)
    return "\n".join(lines)


def grade_text(user, curso: str | None = None, semestre: int | None = None) -> str:
    """`format_grade(list_disciplinas(...))`: texto pronto, ou "" se não houver dados."""
    return format_grade(list_disciplinas(user, curso, semestre))
