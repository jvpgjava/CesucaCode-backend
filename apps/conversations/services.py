import logging
from collections.abc import Iterator
from pathlib import Path

from django.conf import settings
from pgvector.django import CosineDistance

from apps.accounts.models import User
from apps.ai_providers import services as ai_providers
from apps.documents.models import DocumentChunk
from apps.documents.views import get_documents_queryset

from . import guard, web_search
from .events import DoneEvent, ErrorEvent, MetaEvent, TokenEvent, status
from .models import Conversation, Message
from .tracing import TraceRecorder

logger = logging.getLogger(__name__)

TOP_K_CHUNKS = 5
CURRICULUM_TOP_K_CHUNKS = 8
CURRICULUM_EXTRA_DISTANCE = 0.05
SHORT_FOLLOWUP_MAX_WORDS = 6

# Resposta padrão quando o provedor do LLM se recusa a responder (filtro de
# conteúdo). Como ela é salva no histórico, o par pergunta+recusa é omitido do
# contexto das mensagens seguintes (ver build_messages) — senão a mensagem
# barrada seria reenviada a cada turno e travaria a conversa inteira.
REFUSAL_MESSAGE = (
    "Não consegui processar essa mensagem. Posso ajudar com dúvidas de computação "
    "dos cursos de CC e ADS — pode reformular a pergunta em texto simples?"
)
ERROR_MESSAGE = "Falha ao gerar resposta. Tente novamente."


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
    "Com o que você pode me ajudar?",
]


def build_suggestions() -> list[str]:
    """Sugestões de primeira mensagem. São perguntas fixas; a resposta sempre
    vem das informações do curso (ou "não tenho essa informação", se não houver)."""
    return list(STATIC_SUGGESTIONS)


def get_accessible_chunks_queryset(user):
    return DocumentChunk.objects.filter(
        document__in=get_documents_queryset(user),
        document__status="ready",
        embedding__isnull=False,
    ).select_related("document")


def retrieve_context(user, query_text, top_k=TOP_K_CHUNKS, max_distance=None):
    max_distance = settings.RAG_MAX_DISTANCE if max_distance is None else max_distance
    query_vector = ai_providers.get_embedding_model().embed_query(query_text)
    ai_providers.validate_embedding_dimensions(query_vector)
    return list(
        get_accessible_chunks_queryset(user)
        .annotate(distance=CosineDistance("embedding", query_vector))
        .filter(distance__lte=max_distance)
        .order_by("distance")[:top_k]
    )


def build_search_text(conversation: Conversation, user_text: str) -> str:
    """Texto usado pra buscar no contexto (e detectar pergunta de grade).

    Respostas curtas de continuação ("CC", "sim", "e o 2º semestre?") não se
    parecem com nada nos materiais; sem a pergunta anterior a busca volta vazia
    e a resposta cai em conhecimento geral. Por isso, mensagens curtas são
    combinadas com a última pergunta do usuário na conversa."""
    if len(user_text.split()) > SHORT_FOLLOWUP_MAX_WORDS:
        return user_text
    previous = (
        conversation.messages.filter(role=Message.Role.USER).order_by("-id").values_list("content", flat=True).first()
    )
    return f"{previous} {user_text}" if previous else user_text


def build_context_block(chunks) -> str:
    """Monta o contexto com referências opacas: "[T1 · seção: <título da seção>]".
    O título do documento NUNCA entra no prompt (o modelo não tem o que vazar);
    a referência existe só pra ele se orientar e não deve ser citada."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        heading = " ".join((getattr(chunk, "heading", "") or "").split())
        ref = f"[T{i} · seção: {heading}]" if heading else f"[T{i}]"
        parts.append(f"{ref}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    conversation: Conversation,
    user_text: str,
    context_block: str,
    web_block: str = "",
    history: list[Message] | None = None,
):
    """`history` permite informar o histórico já lido (anterior à mensagem
    atual); sem ele, usa todas as mensagens salvas da conversa."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system_text = f"{get_system_prompt()}\n\n{build_user_profile_block(conversation.user)}"
    messages = [SystemMessage(content=system_text)]

    history = list(conversation.messages.all()) if history is None else history
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
        sections = [f"Contexto dos materiais didáticos:\n\n{context_block}"]
    elif settings.CHAT_ALLOW_GENERAL_KNOWLEDGE:
        sections = ["[Nenhuma informação de referência foi considerada relevante para esta pergunta.]"]
    else:
        sections = [
            "[Nenhuma informação de referência foi considerada relevante para esta "
            "pergunta. MODO ESTRITO: não use conhecimento geral. Diga que não tem essa "
            "informação confirmada e sugira falar com o professor ou a coordenação — sem "
            "mencionar materiais, arquivos ou base de dados.]"
        ]
    if web_block:
        sections.append(web_block)
    sections.append(f"Pergunta: {user_text}")

    messages.append(HumanMessage(content="\n\n---\n\n".join(sections)))
    return messages


def _text_of(content) -> str:
    """Texto de um chunk do LLM. Alguns providers devolvem uma lista de blocos
    (`{"type": "text", "text": ...}`) em vez de string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) and block.get("type") == "text" else block
            for block in content
            if isinstance(block, (str, dict))
        )
    return ""


def delete_last_exchange(conversation: Conversation) -> None:
    """Regenerar: apaga a última resposta do assistente e a pergunta que a gerou
    (só se estiverem no fim da conversa, nessa ordem). Se a conversa terminar numa
    pergunta sem resposta (falha anterior), apaga só ela."""
    last = conversation.messages.order_by("-id").first()
    if last is None:
        return
    to_delete = [last.id]
    if last.role == Message.Role.ASSISTANT:
        previous = conversation.messages.filter(id__lt=last.id).order_by("-id").first()
        if previous is not None and previous.role == Message.Role.USER:
            to_delete.append(previous.id)
    conversation.messages.filter(id__in=to_delete).delete()


def send_message(conversation: Conversation, user_text: str, *, regenerate: bool = False) -> Iterator:
    """Processa uma mensagem do usuário e emite eventos tipados (ver events.py):
    meta -> status... -> tokens... -> done (ou error).

    - A mensagem do usuário é salva antes de gerar a resposta.
    - A do assistente é salva no `finally`: também quando o cliente desconecta
      (GeneratorExit), caso em que fica o texto parcial.
    - A guarda de saída roda sobre o texto final. Como a resposta já foi
      transmitida por streaming, ela não desfaz o que o aluno viu: grava as flags
      no trace e salva a versão sem referências `[T#]`.
    """
    recorder = TraceRecorder()
    answer_model = getattr(settings, "LLM_ANSWER_MODEL", "") or settings.LLM_MODEL
    recorder.set(route="legacy", route_source="legacy", models={"answer": answer_model})

    if regenerate:
        delete_last_exchange(conversation)

    full_response: list[str] = []
    error: str | None = None
    failed = False
    saved: Message | None = None
    try:
        search_text = build_search_text(conversation, user_text)
        history = list(conversation.messages.all())

        user_message = Message.objects.create(conversation=conversation, role=Message.Role.USER, content=user_text)
        if not conversation.title:
            conversation.title = user_text[:80]
        conversation.save(update_fields=["title", "updated_at"])
        yield MetaEvent(user_message_id=user_message.id, route=None)

        yield status("searching")
        # Em pergunta de grade/disciplinas a resposta precisa do conjunto todo (a grade
        # se espalha por vários trechos, todos perto do limite de relevância), então
        # busca mais trechos e aceita uma distância um pouco maior.
        curriculum = web_search.is_curriculum_question(search_text)
        with recorder.step("retrieval", "vector_search", curriculum=curriculum) as step_meta:
            context_chunks = retrieve_context(
                conversation.user,
                search_text,
                top_k=CURRICULUM_TOP_K_CHUNKS if curriculum else TOP_K_CHUNKS,
                max_distance=settings.RAG_MAX_DISTANCE + (CURRICULUM_EXTRA_DISTANCE if curriculum else 0),
            )
            step_meta["n_chunks"] = len(context_chunks)
        recorder.set(
            chunk_ids=[chunk.id for chunk in context_chunks],
            distances=[round(float(chunk.distance), 4) for chunk in context_chunks],
        )
        context_block = build_context_block(context_chunks)

        # Referências externas (pesquisa na web) só em perguntas de grade/disciplinas
        # e não no modo estrito, que restringe a resposta ao que a instituição enviou.
        web_block = ""
        if settings.CHAT_ALLOW_GENERAL_KNOWLEDGE and settings.CHAT_WEB_SEARCH_ENABLED and curriculum:
            yield status("web")
            with recorder.step("web", "search") as step_meta:
                web_block = web_search.build_web_block(conversation.user, search_text)
                step_meta["used"] = bool(web_block)
        messages = build_messages(conversation, user_text, context_block, web_block, history=history)

        yield status("writing")
        chat_model = ai_providers.get_chat_model()
        try:
            for chunk in chat_model.stream(messages):
                if getattr(chunk, "usage_metadata", None):
                    recorder.add_usage(ai_providers.extract_usage(chunk))
                piece = _text_of(chunk.content)
                if piece:
                    recorder.mark_first_token()
                    full_response.append(piece)
                    yield TokenEvent(piece)
        except Exception as exc:
            if full_response or not ai_providers.is_provider_refusal(exc):
                raise
            logger.warning("Provedor recusou a mensagem da conversa %s: %s", conversation.id, exc)
            error = "provider_refusal"

        if not full_response:
            full_response.append(REFUSAL_MESSAGE)
            yield TokenEvent(REFUSAL_MESSAGE)
    except Exception as exc:
        logger.exception("Falha ao gerar resposta para a conversa %s", conversation.id)
        error = f"{type(exc).__name__}: {exc}"
        failed = True
        yield ErrorEvent(ERROR_MESSAGE)
    finally:
        # Roda também em GeneratorExit (cliente desconectou): salva o parcial.
        if full_response:
            text = "".join(full_response)
            try:
                titles = list(get_documents_queryset(conversation.user).values_list("title", flat=True))
                flags = guard.check_output(text, document_titles=titles)
                if flags:
                    logger.warning("Guarda de saída (conversa %s): %s", conversation.id, flags)
                    recorder.set(guard_flags=flags)
                    text = guard.redact(text, flags)
            except Exception:
                logger.exception("Falha na guarda de saída da conversa %s", conversation.id)
            saved = Message.objects.create(conversation=conversation, role=Message.Role.ASSISTANT, content=text)
            recorder.finish(saved, error=error)

    if saved is not None and not failed:
        yield DoneEvent(message_id=saved.id)
