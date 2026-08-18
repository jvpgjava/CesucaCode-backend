import logging

from django.db import transaction

from apps.ai_providers import services as ai_providers

from . import chunking, extraction
from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)


def create_document(*, title, course, file, uploaded_by) -> Document:
    document = Document.objects.create(
        title=title,
        course=course,
        file=file,
        uploaded_by=uploaded_by,
        status=Document.Status.PROCESSING,
    )
    _process_document(document)
    return document


def reprocess_document(document: Document) -> Document:
    document.chunks.all().delete()
    _process_document(document)
    return document


def _process_document(document: Document) -> None:
    try:
        document.file.open("rb")
        try:
            text = extraction.extract_text(document.file, document.file.name)
        finally:
            document.file.close()

        if not text.strip():
            raise extraction.UnsupportedFileTypeError(
                "Não foi possível extrair texto do arquivo (pode estar vazio ou ser uma imagem escaneada)."
            )

        chunks = chunking.chunk_text(text)
        embeddings = ai_providers.get_embedding_model().embed_documents(chunks)

        with transaction.atomic():
            DocumentChunk.objects.bulk_create(
                [
                    DocumentChunk(document=document, index=i, content=content, embedding=embedding)
                    for i, (content, embedding) in enumerate(zip(chunks, embeddings))
                ]
            )
            document.status = Document.Status.READY
            document.processing_error = ""
            document.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as exc:
        if isinstance(exc, extraction.UnsupportedFileTypeError):
            document.processing_error = str(exc)
        else:
            logger.exception("Falha ao processar o documento %s", document.id)
            document.processing_error = (
                "Falha ao processar o arquivo. Tente reenviar ou contate o suporte."
            )
        document.status = Document.Status.FAILED
        document.save(update_fields=["status", "processing_error", "updated_at"])
