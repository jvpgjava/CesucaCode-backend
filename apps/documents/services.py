import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from datetime import timedelta

from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.db import close_old_connections, transaction
from django.utils import timezone

from apps.ai_providers import services as ai_providers
from apps.ai_providers.exceptions import ProviderConfigurationError

from . import chunking, extraction
from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)

# Extração com Docling é pesada em CPU (layout + OCR); processar em background
# pra não segurar o request de upload por minutos. Pool pequeno e fixo em vez
# de uma thread por upload — evita sobrecarregar a máquina se vários uploads
# chegarem juntos, sem precisar de um broker (Celery/Redis) pra um volume de
# uso tão baixo quanto o desse app.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="doc-processor")


def create_document(*, title, courses, file, uploaded_by) -> Document:
    document = Document.objects.create(
        title=title,
        file=file,
        uploaded_by=uploaded_by,
        status=Document.Status.PROCESSING,
    )
    document.courses.set(courses)
    _executor.submit(_process_document_in_background, document.id)
    return document


def reprocess_document(document: Document) -> Document:
    document.chunks.all().delete()
    document.status = Document.Status.PROCESSING
    document.processing_error = ""
    document.save(update_fields=["status", "processing_error", "updated_at"])
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


def _embedding_input(chunk: chunking.Chunk, course_label: str = "") -> str:
    """Texto que vai ao modelo de embedding (o `content` salvo NÃO inclui o cabeçalho).

    Cabeçalho contextual determinístico — curso(s) e seção —, para o chunk "saber"
    de onde veio. Nunca leva o título do arquivo (o título não pode vazar nem
    influenciar a busca). Depois, aplica o prefixo de tarefa do modelo, se houver."""
    parts = []
    if course_label:
        parts.append(f"Documento do curso: {course_label}")
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


def _embed_batch_with_retry(model, batch: list[str]) -> list[list[float]]:
    """`embed_documents` com backoff exponencial (EMBEDDING_RETRY_BASE_SECONDS, dobra a
    cada tentativa) só para erros transitórios. Erros permanentes sobem na hora; se
    as tentativas acabarem, sobe o último erro (tratado em _process_document)."""
    attempts = max(1, settings.EMBEDDING_RETRY_ATTEMPTS)
    for attempt in range(attempts):
        try:
            return model.embed_documents(batch)
        except Exception as exc:
            if attempt == attempts - 1 or not _is_transient_error(exc):
                raise
            delay = settings.EMBEDDING_RETRY_BASE_SECONDS * (2**attempt)
            logger.warning(
                "Embedding falhou (%s); tentativa %d/%d, nova tentativa em %.0fs.",
                type(exc).__name__, attempt + 1, attempts, delay,
            )  # fmt: skip
            time.sleep(delay)
    raise AssertionError("inalcançável")


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeda em lotes (`EMBEDDING_BATCH_SIZE`), espaçados se houver teto configurado
    e com retry para erros transitórios. Os demais erros do provider sobem normalmente."""
    model = ai_providers.get_embedding_model()
    size = settings.EMBEDDING_BATCH_SIZE
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        _embedding_throttle.wait(len(batch))
        embeddings.extend(_embed_batch_with_retry(model, batch))
    if embeddings:
        ai_providers.validate_embedding_dimensions(embeddings[0])
    return embeddings


def update_search_vectors(document: Document) -> None:
    """Preenche o tsvector (busca textual) dos chunks do documento: seção com peso A e
    conteúdo com peso B, na config 'portuguese'."""
    document.chunks.update(
        search_vector=SearchVector("heading", weight="A", config="portuguese")
        + SearchVector("content", weight="B", config="portuguese")
    )


def _process_document(document: Document) -> None:
    try:
        document.file.open("rb")
        try:
            chunks = _extract_chunks(document)
        finally:
            document.file.close()

        course_label = _course_label(document)
        embeddings = _embed_documents([_embedding_input(chunk, course_label) for chunk in chunks])

        with transaction.atomic():
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
            document.status = Document.Status.READY
            document.processing_error = ""
            document.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as exc:
        if isinstance(
            exc,
            (extraction.UnsupportedFileTypeError, extraction.OcrUnavailableError, ProviderConfigurationError),
        ):
            document.processing_error = str(exc)
        elif _is_quota_error(exc):
            logger.error("Cota do provider de embedding esgotada no documento %s: %s", document.id, exc)
            document.processing_error = (
                "A cota do provider de embedding foi esgotada. Aguarde a renovação "
                "da cota (ou use outro plano/provider) e clique em Reprocessar."
            )
        else:
            logger.exception("Falha ao processar o documento %s", document.id)
            document.processing_error = (
                "Falha ao processar o arquivo. Tente reenviar ou contate o suporte."
            )
        document.status = Document.Status.FAILED
        document.save(update_fields=["status", "processing_error", "updated_at"])
