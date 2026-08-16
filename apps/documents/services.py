from django.db import transaction

from . import chunking, extraction
from .models import Document, DocumentChunk


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

        with transaction.atomic():
            DocumentChunk.objects.bulk_create(
                [
                    DocumentChunk(document=document, index=i, content=content)
                    for i, content in enumerate(chunks)
                ]
            )
            document.status = Document.Status.READY
            document.processing_error = ""
            document.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as exc:
        document.status = Document.Status.FAILED
        document.processing_error = str(exc)
        document.save(update_fields=["status", "processing_error", "updated_at"])
