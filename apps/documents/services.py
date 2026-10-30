import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from datetime import timedelta

from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.db.models import Value
from django.db import close_old_connections, transaction
from django.utils import timezone

from apps.ai_providers import services as ai_providers
from apps.ai_providers.exceptions import ProviderConfigurationError

from . import chunking, disciplinas, extraction
from .models import Disciplina, Document, DocumentChunk

logger = logging.getLogger(__name__)

# Extração com Docling é pesada em CPU (layout + OCR); processar em background
# pra não segurar o request de upload por minutos. Pool pequeno e fixo em vez
# de uma thread por upload — evita sobrecarregar a máquina se vários uploads
# chegarem juntos, sem precisar de um broker (Celery/Redis) pra um volume de
# uso tão baixo quanto o desse app.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="doc-processor")

# Mensagens das etapas, mostradas na tela de materiais enquanto o status é `processing`.
QUEUED_DETAIL = "Na fila para processamento…"
EXTRACTING_DETAIL = "Lendo o arquivo e identificando seções e tabelas…"
EMBEDDING_DETAIL = "Indexando o conteúdo: {done} de {total} trechos…"
QUOTA_WAIT_DETAIL = (
    "Aguardando o serviço externo de indexação (limite de uso por minuto atingido). "
    "O processamento continua sozinho, não é preciso fazer nada."
)
SAVING_DETAIL = "Salvando…"
# Faixas do progresso: a extração é rápida e a indexação (com as esperas) domina.
EXTRACTION_PROGRESS = 10
EMBEDDING_PROGRESS_END = 95


def _set_progress(document: Document, detail: str, progress: int | None = None, resume_at=None) -> None:
    """Grava etapa/progresso direto no banco (sem `save()` do objeto inteiro, que a
    thread de processamento ainda está usando). Renova `updated_at`: uma espera longa
    pela cota não pode parecer documento travado para `recover_stuck_documents`."""
    fields = {"processing_detail": detail[:255], "processing_resume_at": resume_at, "updated_at": timezone.now()}
    if progress is not None:
        fields["processing_progress"] = max(0, min(100, progress))
    Document.objects.filter(pk=document.pk).update(**fields)


def create_document(*, title, courses, file, uploaded_by) -> Document:
    document = Document.objects.create(
        title=title,
        file=file,
        uploaded_by=uploaded_by,
        status=Document.Status.PROCESSING,
        processing_detail=QUEUED_DETAIL,
    )
    document.courses.set(courses)
    _executor.submit(_process_document_in_background, document.id)
    return document


def reprocess_document(document: Document) -> Document:
    """Reenfileira o processamento. Os chunks e disciplinas atuais NÃO são apagados
    aqui: continuam respondendo no chat até a troca atômica pelos novos, no fim de um
    processamento bem-sucedido (ver `_process_document`). Se o reprocessamento falhar
    (ex.: cota do provider), o conteúdo anterior continua valendo."""
    document.status = Document.Status.PROCESSING
    document.processing_error = ""
    document.processing_detail = QUEUED_DETAIL
    document.processing_progress = 0
    document.processing_resume_at = None
    document.save(
        update_fields=[
            "status", "processing_error", "processing_detail", "processing_progress",
            "processing_resume_at", "updated_at",
        ]
    )  # fmt: skip
    _executor.submit(_process_document_in_background, document.id)
    return document


def recover_stuck_documents(minutes: int = 15) -> list[Document]:
    """Reenfileira documentos presos em `processing` há mais de `minutes` minutos.

    O pool de processamento vive na memória do processo: se o servidor reinicia no
    meio de uma ingestão, o documento fica em `processing` para sempre. Chamado no
    boot (`manage.py recover_documents`, via docker-entrypoint.py) e à mão."""
    limit = timezone.now() - timedelta(minutes=minutes)
    stuck = list(Document.objects.filter(status=Document.Status.PROCESSING, updated_at__lt=limit))
    for document in stuck:
        reprocess_document(document)
    return stuck


def _process_document_in_background(document_id: int) -> None:
    # _process_document já captura e trata erros da extração/embedding; o
    # try/except aqui fora é só pra garantir que nada escapa em silêncio pro
    # ThreadPoolExecutor (que descarta exceções de tasks cujo Future nunca é
    # inspecionado) — ex.: o documento ser apagado entre o upload e a task
    # rodar.
    try:
        document = Document.objects.get(id=document_id)
        _process_document(document)
    except Exception:
        logger.exception("Task de processamento em background falhou pro documento %s", document_id)
    finally:
        close_old_connections()


# PDF cuja extração sem OCR rende menos que isso (caracteres) é tratado como
# escaneado: tenta de novo com OCR antes de desistir.
OCR_MIN_CHARS = 200


def _total_chars(chunks: list[chunking.Chunk]) -> int:
    return sum(len(chunk.content) for chunk in chunks)


def _extract_chunks(document: Document) -> list[chunking.Chunk]:
    ext = extraction.get_extension(document.file.name)

    if ext == "txt":
        text = extraction.extract_txt(document.file)
        if not text.strip():
            raise extraction.UnsupportedFileTypeError(
                "Não foi possível extrair texto do arquivo (pode estar vazio)."
            )
        return chunking.chunk_text(text)

    data = document.file.read()
    chunks = chunking.chunk_docling_document(extraction.convert_document(data, document.file.name))

    if ext == "pdf" and _total_chars(chunks) < OCR_MIN_CHARS:
        # Sem texto nativo: provavelmente um PDF escaneado. OCR é lento, então só
        # roda aqui (conversor separado, criado sob demanda). Se o OCR não estiver
        # disponível, OcrUnavailableError sobe com uma mensagem clara.
        logger.info("Documento %s sem texto nativo; tentando OCR.", document.id)
        chunks = chunking.chunk_docling_document(
            extraction.convert_document(data, document.file.name, ocr=True)
        )
        if _total_chars(chunks) < OCR_MIN_CHARS:
            raise extraction.UnsupportedFileTypeError(
                "Não foi possível extrair texto do arquivo nem com OCR (pode estar vazio "
                "ou ilegível)."
            )

    if not chunks:
        raise extraction.UnsupportedFileTypeError("Não foi possível extrair texto do arquivo (pode estar vazio).")
    return chunks


def _course_label(document: Document) -> str:
    return ", ".join(course.name for course in document.courses.all())


def _embedding_input(chunk: chunking.Chunk, course_label: str = "", document_title: str = "") -> str:
    """Texto que vai ao modelo de embedding (o `content` salvo NÃO inclui o cabeçalho).

    Cabeçalho contextual determinístico para o chunk "saber" de onde veio: título do
    documento, curso(s), disciplina vigente (com semestre e C/H) e seção. O título do
    documento entra AQUI (e no search_vector) para a busca achar o material pelo tipo/
    nome ("horário", "plano de ensino"), mas nunca é salvo no `content`/`heading`: o LLM
    só vê esses dois (via `format_context`), então o título continua sem chegar a ele.
    Depois, aplica o prefixo de tarefa do modelo, se houver."""
    parts = []
    if document_title:
        parts.append(f"Documento: {document_title}")
    if course_label:
        parts.append(f"Documento do curso: {course_label}")
    if chunk.discipline is not None:
        parts.append(f"Disciplina: {chunk.discipline.label}")
    if chunk.heading:
        parts.append(f"Seção: {chunk.heading}")
    text = f"{' | '.join(parts)}\n\n{chunk.content}" if parts else chunk.content
    return ai_providers.embedding_text_for("document", text)


class _EmbeddingThrottle:
    """Espaça as chamadas de embedding para respeitar um teto de textos por minuto.

    Só atua se `EMBEDDING_MAX_REQUESTS_PER_MINUTE` > 0 (ex.: plano gratuito do
    Gemini, 100/min). Com 0 (padrão, pensado para produção) não espera nada. O
    estado é compartilhado entre as threads do pool, que processam documentos em
    paralelo contra a mesma cota."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self, n_texts: int) -> None:
        rpm = settings.EMBEDDING_MAX_REQUESTS_PER_MINUTE
        if rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + n_texts * 60.0 / rpm
        if delay:
            logger.info("Embedding: aguardando %.0fs para respeitar o limite configurado.", delay)
            time.sleep(delay)


_embedding_throttle = _EmbeddingThrottle()


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


_TRANSIENT_MARKERS = (
    "429", "resource_exhausted", "quota", "rate limit", "timeout", "timed out",
    "connection", "unavailable", "503", "502", "504", "temporarily",
)  # fmt: skip


def _is_transient_error(exc: Exception) -> bool:
    """Erros que valem nova tentativa: cota/limite por minuto, timeout, rede."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class DailyQuotaExceededError(Exception):
    """Cota DIÁRIA do provider de embedding esgotada: esperar minutos não resolve."""


_RETRY_IN_RE = re.compile(r"retry in\s+([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", re.IGNORECASE)
_DAILY_QUOTA_MARKERS = ("perday", "per day", "requests per day", "daily")
QUOTA_WAIT_FALLBACK_SECONDS = 30.0
QUOTA_WAIT_MARGIN_SECONDS = 2.0


def _is_daily_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return _is_quota_error(exc) and any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def _quota_retry_seconds(exc: Exception) -> float:
    """Quanto esperar antes de tentar de novo, pelo que o provider indicou no erro
    ("Please retry in 24.5s" / retryDelay '24s'); sem indicação, 30s."""
    text = str(exc)
    for pattern in (_RETRY_IN_RE, _RETRY_DELAY_RE):
        match = pattern.search(text)
        if match:
            return float(match.group(1)) + QUOTA_WAIT_MARGIN_SECONDS
    return QUOTA_WAIT_FALLBACK_SECONDS


OnWait = Callable[[float], None]


def _embed_batch_with_retry(model, batch: list[str], on_wait: OnWait | None = None) -> list[list[float]]:
    """`embed_documents` resiliente a limites do provider.

    - Cota por minuto esgotada (429): espera o tempo que o provider indica e tenta de
      novo, quantas vezes for preciso, até `EMBEDDING_QUOTA_MAX_WAIT_SECONDS` de espera
      somada. É o que permite indexar um material grande no plano gratuito.
    - Cota diária esgotada: `DailyQuotaExceededError` na hora.
    - Outros erros transitórios (timeout, rede, 5xx): backoff exponencial curto
      (`EMBEDDING_RETRY_ATTEMPTS`, `EMBEDDING_RETRY_BASE_SECONDS`).
    - Erros permanentes sobem na hora (tratados em _process_document).
    `on_wait(segundos)` é chamado antes de cada espera pela cota (aviso na tela)."""
    attempts = max(1, settings.EMBEDDING_RETRY_ATTEMPTS)
    transient_failures = 0
    waited = 0.0
    while True:
        try:
            return model.embed_documents(batch)
        except Exception as exc:
            if _is_quota_error(exc):
                if _is_daily_quota_error(exc):
                    raise DailyQuotaExceededError(str(exc)) from exc
                delay = _quota_retry_seconds(exc)
                if waited + delay > settings.EMBEDDING_QUOTA_MAX_WAIT_SECONDS:
                    raise
                logger.warning("Cota de embedding por minuto atingida; aguardando %.0fs.", delay)
                if on_wait is not None:
                    on_wait(delay)
                time.sleep(delay)
                waited += delay
                continue
            transient_failures += 1
            if transient_failures >= attempts or not _is_transient_error(exc):
                raise
            delay = settings.EMBEDDING_RETRY_BASE_SECONDS * (2 ** (transient_failures - 1))
            logger.warning(
                "Embedding falhou (%s); tentativa %d/%d, nova tentativa em %.0fs.",
                type(exc).__name__, transient_failures, attempts, delay,
            )  # fmt: skip
            time.sleep(delay)


def _embed_documents(
    texts: list[str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    on_wait: OnWait | None = None,
) -> list[list[float]]:
    """Embeda em lotes (`EMBEDDING_BATCH_SIZE`), espaçados se houver teto configurado,
    esperando a cota quando preciso (ver `_embed_batch_with_retry`). `on_progress(feitos,
    total)` é chamado antes do primeiro lote e depois de cada um."""
    model = ai_providers.get_embedding_model()
    size = settings.EMBEDDING_BATCH_SIZE
    embeddings: list[list[float]] = []
    if on_progress is not None:
        on_progress(0, len(texts))
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        _embedding_throttle.wait(len(batch))
        embeddings.extend(_embed_batch_with_retry(model, batch, on_wait=on_wait))
        if on_progress is not None:
            on_progress(len(embeddings), len(texts))
    if embeddings:
        ai_providers.validate_embedding_dimensions(embeddings[0])
    return embeddings


def update_search_vectors(document: Document) -> None:
    """Preenche o tsvector (busca textual) dos chunks do documento: seção (que já leva o
    nome da disciplina) e título do documento com peso A, conteúdo com peso B, na config
    'portuguese'. O título só existe aqui, no índice: nunca no texto mostrado ao LLM."""
    document.chunks.update(
        search_vector=SearchVector("heading", Value(document.title), weight="A", config="portuguese")
        + SearchVector("content", weight="B", config="portuguese")
    )


def sync_disciplinas(document: Document, chunks: list[chunking.Chunk]) -> int:
    """Refaz a tabela `Disciplina` do documento a partir dos planos detectados nos
    chunks (ver `disciplinas.build_disciplinas`). Devolve quantas linhas gravou."""
    document.disciplinas.all().delete()
    rows = disciplinas.build_disciplinas(chunks, list(document.courses.all()))
    Disciplina.objects.bulk_create([Disciplina(document=document, **row) for row in rows])
    return len(rows)


def _process_document(document: Document) -> None:
    try:
        _set_progress(document, EXTRACTING_DETAIL, 0)
        document.file.open("rb")
        try:
            chunks = _extract_chunks(document)
        finally:
            document.file.close()

        course_label = _course_label(document)
        span = EMBEDDING_PROGRESS_END - EXTRACTION_PROGRESS

        def on_progress(done: int, total: int) -> None:
            progress = EXTRACTION_PROGRESS + (span * done // total if total else span)
            _set_progress(document, EMBEDDING_DETAIL.format(done=done, total=total), progress)

        def on_wait(seconds: float) -> None:
            _set_progress(document, QUOTA_WAIT_DETAIL, resume_at=timezone.now() + timedelta(seconds=seconds))

        embeddings = _embed_documents(
            [_embedding_input(chunk, course_label, document.title) for chunk in chunks],
            on_progress=on_progress,
            on_wait=on_wait,
        )
        _set_progress(document, SAVING_DETAIL, EMBEDDING_PROGRESS_END)

        with transaction.atomic():
            # Troca atômica: o conteúdo anterior (reprocessamento) só sai junto com a
            # entrada do novo. Sem janela em que o material fica vazio no chat.
            document.chunks.all().delete()
            DocumentChunk.objects.bulk_create(
                [
                    DocumentChunk(
                        document=document,
                        index=i,
                        content=chunk.content,
                        heading=chunk.heading,
                        embedding=embedding,
                    )
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
            )
            update_search_vectors(document)
            sync_disciplinas(document, chunks)
            document.status = Document.Status.READY
            document.processing_error = ""
            document.processing_detail = ""
            document.processing_progress = 100
            document.processing_resume_at = None
            document.save(
                update_fields=[
                    "status", "processing_error", "processing_detail", "processing_progress",
                    "processing_resume_at", "updated_at",
                ]
            )  # fmt: skip
    except Exception as exc:
        if isinstance(
            exc,
            (extraction.UnsupportedFileTypeError, extraction.OcrUnavailableError, ProviderConfigurationError),
        ):
            document.processing_error = str(exc)
        elif isinstance(exc, DailyQuotaExceededError):
            logger.error("Cota diária do provider de embedding esgotada no documento %s: %s", document.id, exc)
            document.processing_error = (
                "O limite diário do serviço externo de indexação foi atingido. Tente "
                "Reprocessar amanhã (ou configure um plano/provider com mais cota)."
            )
        elif _is_quota_error(exc):
            logger.error("Cota do provider de embedding esgotada no documento %s: %s", document.id, exc)
            document.processing_error = (
                "O serviço externo de indexação ficou indisponível por limite de uso por "
                "tempo demais. Aguarde alguns minutos e clique em Reprocessar."
            )
        else:
            logger.exception("Falha ao processar o documento %s", document.id)
            document.processing_error = (
                "Falha ao processar o arquivo. Tente reenviar ou contate o suporte."
            )
        if document.chunks.exists():
            document.processing_error += " O conteúdo anterior deste material continua disponível no chat."
        document.status = Document.Status.FAILED
        document.processing_detail = ""
        document.processing_resume_at = None
        document.save(
            update_fields=["status", "processing_error", "processing_detail", "processing_resume_at", "updated_at"]
        )
