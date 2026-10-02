import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.db import close_old_connections, transaction

from apps.ai_providers import services as ai_providers

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


def _extract_chunks(document: Document) -> list[chunking.Chunk]:
    ext = extraction.get_extension(document.file.name)

    if ext == "txt":
        text = extraction.extract_txt(document.file)
        if not text.strip():
            raise extraction.UnsupportedFileTypeError(
                "Não foi possível extrair texto do arquivo (pode estar vazio)."
            )
        return chunking.chunk_text(text)

    docling_document = extraction.convert_document(document.file, document.file.name)
    chunks = chunking.chunk_docling_document(docling_document)
    if not chunks:
        raise extraction.UnsupportedFileTypeError(
            "Não foi possível extrair texto do arquivo (pode estar vazio ou ser uma "
            "imagem escaneada sem OCR)."
        )
    return chunks


def _embedding_input(chunk: chunking.Chunk) -> str:
    if chunk.heading:
        return f"{chunk.heading}\n\n{chunk.content}"
    return chunk.content


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


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeda em lotes (`EMBEDDING_BATCH_SIZE`), espaçados se houver teto configurado.
    Erros do provider, inclusive de cota, sobem normalmente."""
    model = ai_providers.get_embedding_model()
    size = settings.EMBEDDING_BATCH_SIZE
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        _embedding_throttle.wait(len(batch))
        embeddings.extend(model.embed_documents(batch))
    return embeddings


def _process_document(document: Document) -> None:
    try:
        document.file.open("rb")
        try:
            chunks = _extract_chunks(document)
        finally:
            document.file.close()

        embeddings = _embed_documents([_embedding_input(chunk) for chunk in chunks])

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
            document.status = Document.Status.READY
            document.processing_error = ""
            document.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as exc:
        if isinstance(exc, extraction.UnsupportedFileTypeError):
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
