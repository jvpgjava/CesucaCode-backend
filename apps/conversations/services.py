import logging
from collections.abc import Iterator
from pathlib import Path

from django.conf import settings

from apps.accounts.models import User

from . import retrieval
from .models import Conversation, Message

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
    """Mantido por compatibilidade; a implementação vive em retrieval.py."""
    return retrieval.get_accessible_chunks_queryset(user)


def retrieve_context(user, query_text, top_k=TOP_K_CHUNKS, max_distance=None):
    """Pipeline legacy: delega à busca híbrida (retrieval.search). Devolve
    `RetrievedChunk` (com `.id`, `.distance`, `.heading`, `.content`)."""
    return retrieval.search(user, query_text, top_k=top_k, max_distance=max_distance)


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
    return retrieval.format_context([(f"T{i}", chunk) for i, chunk in enumerate(chunks, 1)])


def history_messages(history: list[Message]) -> list:
    """Histórico (anterior à mensagem atual) no formato do LangChain, já filtrado:
    sem pares pergunta+recusa do provedor, limitado a `CHAT_MAX_HISTORY_MESSAGES` e
    começando sempre por uma mensagem do usuário."""
    from langchain_core.messages import AIMessage, HumanMessage

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
    return [
        HumanMessage(content=past.content) if past.role == Message.Role.USER else AIMessage(content=past.content)
        for past in kept
    ]


def compose_question(user_text: str, context_block: str, web_block: str = "", notes: tuple[str, ...] = ()) -> str:
    """Último turno do usuário: contexto (ou aviso de que não há) + referências
    externas + pergunta + `notes` (instruções extras do pipeline, ao final)."""
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
    sections.extend(notes)
    return "\n\n---\n\n".join(sections)


def build_messages(
    conversation: Conversation,
    user_text: str,
    context_block: str,
    web_block: str = "",
    history: list[Message] | None = None,
    *,
    system_text: str | None = None,
    notes: tuple[str, ...] = (),
):
    """Mensagens para o LLM: system (prompt + perfil) + histórico + pergunta com contexto.

    `history` permite informar o histórico já lido (anterior à mensagem atual); sem
    ele, usa todas as mensagens salvas. `system_text` substitui o prompt completo
    (o pipeline passa o prompt por rota, já com o bloco de perfil)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    if system_text is None:
        system_text = f"{get_system_prompt()}\n\n{build_user_profile_block(conversation.user)}"
    history = list(conversation.messages.all()) if history is None else history
    return [
        SystemMessage(content=system_text),
        *history_messages(history),
        HumanMessage(content=compose_question(user_text, context_block, web_block, notes)),
    ]


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
    status... -> meta -> status... -> tokens... -> suggestions -> done (ou error).
    A orquestração (roteamento, retrieval, agente, guarda, trace) vive em `pipeline`."""
    from . import pipeline  # import tardio: o pipeline usa helpers deste módulo

    return pipeline.run(conversation, user_text, regenerate=regenerate)
