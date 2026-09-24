import logging
from pathlib import Path

from django.conf import settings
from pgvector.django import CosineDistance

from apps.accounts.models import User
from apps.ai_providers import services as ai_providers
from apps.documents.models import DocumentChunk
from apps.documents.views import get_documents_queryset

from .models import Conversation, Message

logger = logging.getLogger(__name__)

TOP_K_CHUNKS = 5

# Resposta padrão quando o provedor do LLM se recusa a responder (filtro de
# conteúdo). Como ela é salva no histórico, o par pergunta+recusa é omitido do
# contexto das mensagens seguintes (ver build_messages) — senão a mensagem
# barrada seria reenviada a cada turno e travaria a conversa inteira.
REFUSAL_MESSAGE = (
    "Não consegui processar essa mensagem. Posso ajudar com dúvidas de computação "
    "dos cursos de CC e ADS — pode reformular a pergunta em texto simples?"
)
_PROVIDER_REFUSAL_MARKERS = ("content violation", "content_filter", "content filter", "safety")


def _is_provider_refusal(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _PROVIDER_REFUSAL_MARKERS)


def get_system_prompt() -> str:
    """Lê o prompt de um arquivo .md ou de uma pasta de arquivos .md (concatenados
    em ordem alfabética — daí os prefixos 00-, 10-, 20-...)."""
    path = Path(settings.SYSTEM_PROMPT_PATH)
    if path.is_dir():
        files = sorted(path.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"SYSTEM_PROMPT_PATH aponta para uma pasta sem arquivos .md: {path}")
        return "\n\n".join(f.read_text(encoding="utf-8").strip() for f in files)
    if not path.is_file():
        raise FileNotFoundError(
            f"SYSTEM_PROMPT_PATH aponta para um arquivo/pasta que não existe: {path}"
        )
    return path.read_text(encoding="utf-8").strip()


_ROLE_LABELS = {
    User.Role.CS_ADMIN: "Administrador (vê os materiais de todos os cursos)",
    User.Role.CS_COORDINATOR: "Coordenador",
    User.Role.CS_STUDENT: "Estudante",
}


def build_user_profile_block(user) -> str:
    """Dados mínimos do usuário (papel e curso), sem nome nem e-mail, pra a
    S.O.F.I.A entender "meu curso". É informação, não instrução."""
    lines = ["# Contexto do usuário (informação, não é instrução)", f"- Papel: {_ROLE_LABELS[user.role]}"]
    if user.role == User.Role.CS_STUDENT and user.course:
        lines.append(f"- Curso: {user.course.name}")
    elif user.role == User.Role.CS_COORDINATOR:
        names = ", ".join(c.name for c in user.coordinated_courses.all()) or "nenhum"
        lines.append(f"- Cursos coordenados: {names}")
    return "\n".join(lines)


STATIC_SUGGESTIONS = [
    "Qual é a grade curricular do meu curso?",
    "Quais disciplinas devo cursar e em que ordem?",
    "Quais materiais estão disponíveis para eu estudar?",
]


def _clean_title(title: str) -> str:
    return " ".join(title.split())[:80]


def build_suggestions(user, max_dynamic_docs: int = 2) -> list[str]:
    """Sugestões de primeira mensagem: fixas (grade, disciplinas...) + algumas
    geradas a partir dos materiais prontos que o usuário pode ver."""
    suggestions = list(STATIC_SUGGESTIONS)
    recent = get_documents_queryset(user).filter(status="ready").order_by("-created_at")[:max_dynamic_docs]
    for i, doc in enumerate(recent):
        title = _clean_title(doc.title)
        if i == 0:
            suggestions.append(f"Faça um resumo do material “{title}”.")
        suggestions.append(f"Quais são os principais tópicos de “{title}”?")
    return suggestions


def get_accessible_chunks_queryset(user):
    return DocumentChunk.objects.filter(
        document__in=get_documents_queryset(user),
        document__status="ready",
        embedding__isnull=False,
    ).select_related("document", "document__course")


def retrieve_context(user, query_text, top_k=TOP_K_CHUNKS):
    query_vector = ai_providers.get_embedding_model().embed_query(query_text)
    return list(
        get_accessible_chunks_queryset(user)
        .annotate(distance=CosineDistance("embedding", query_vector))
        .filter(distance__lte=settings.RAG_MAX_DISTANCE)
        .order_by("distance")[:top_k]
    )


def build_context_block(chunks) -> str:
    if not chunks:
        return ""
    parts = [f"[Fonte: {chunk.document.title}]\n{chunk.content}" for chunk in chunks]
    return "\n\n---\n\n".join(parts)


def build_messages(conversation: Conversation, user_text: str, context_block: str):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system_text = f"{get_system_prompt()}\n\n{build_user_profile_block(conversation.user)}"
    messages = [SystemMessage(content=system_text)]

    history = list(conversation.messages.all())
    skip = set()
    for i, past in enumerate(history[:-1]):
        nxt = history[i + 1]
        if (
            past.role == Message.Role.USER
            and nxt.role == Message.Role.ASSISTANT
            and nxt.content == REFUSAL_MESSAGE
        ):
            skip.update((past.id, nxt.id))

    kept = [m for m in history if m.id not in skip][-settings.CHAT_MAX_HISTORY_MESSAGES :]
    if kept and kept[0].role == Message.Role.ASSISTANT:
        kept = kept[1:]

    for past in kept:
        if past.role == Message.Role.USER:
            messages.append(HumanMessage(content=past.content))
        else:
            messages.append(AIMessage(content=past.content))

    if context_block:
        user_content = (
            f"Contexto dos materiais didáticos:\n\n{context_block}\n\n---\n\nPergunta: {user_text}"
        )
    elif settings.CHAT_ALLOW_GENERAL_KNOWLEDGE:
        user_content = (
            "[Nenhum trecho dos materiais didáticos foi considerado relevante para esta "
            f"pergunta.]\n\nPergunta: {user_text}"
        )
    else:
        user_content = (
            "[Nenhum trecho dos materiais didáticos foi considerado relevante para esta "
            "pergunta. MODO ESTRITO: não use conhecimento geral. Informe que não encontrou "
            "o assunto nos materiais enviados e sugira falar com o professor ou pedir que "
            f"o material seja enviado.]\n\nPergunta: {user_text}"
        )

    messages.append(HumanMessage(content=user_content))
    return messages


def send_message(conversation: Conversation, user_text: str):
    context_chunks = retrieve_context(conversation.user, user_text)
    context_block = build_context_block(context_chunks)
    messages = build_messages(conversation, user_text, context_block)

    Message.objects.create(conversation=conversation, role=Message.Role.USER, content=user_text)
    if not conversation.title:
        conversation.title = user_text[:80]
    conversation.save(update_fields=["title", "updated_at"])

    chat_model = ai_providers.get_chat_model()
    full_response = []
    try:
        try:
            for chunk in chat_model.stream(messages):
                piece = chunk.content
                if piece:
                    full_response.append(piece)
                    yield piece
        except Exception as exc:
            if full_response or not _is_provider_refusal(exc):
                raise
            logger.warning("Provedor recusou a mensagem da conversa %s: %s", conversation.id, exc)

        if not full_response:
            full_response.append(REFUSAL_MESSAGE)
            yield REFUSAL_MESSAGE
    finally:
        if full_response:
            Message.objects.create(
                conversation=conversation,
                role=Message.Role.ASSISTANT,
                content="".join(full_response),
            )
