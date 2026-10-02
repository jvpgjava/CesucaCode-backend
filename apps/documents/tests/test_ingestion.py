import io
from datetime import timedelta

import pytest
from django.conf import settings
from django.core.files.base import ContentFile
from django.contrib.postgres.search import SearchQuery
from django.core.management import call_command
from django.utils import timezone

from apps.accounts.models import Course, User
from apps.ai_providers import services as ai_providers
from apps.documents import chunking, extraction, services
from apps.documents.models import Document

DIMS = settings.EMBEDDING_DIMENSIONS


class FakeEmbeddings:
    """Modelo de embedding fake: levanta os erros da fila (um por chamada) e depois
    devolve vetores válidos."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.errors:
            raise self.errors.pop(0)
        return [[1.0] + [0.0] * (DIMS - 1) for _ in texts]


@pytest.fixture
def sleeps(monkeypatch):
    """Evita esperas reais e registra o backoff pedido."""
    registered: list[float] = []
    monkeypatch.setattr(services.time, "sleep", registered.append)
    monkeypatch.setattr(services._embedding_throttle, "wait", lambda n: None)
    return registered


@pytest.fixture
def admin(db):
    return User.objects.create_user(email="a@example.com", password="x", full_name="A", role=User.Role.CS_ADMIN)


@pytest.fixture(autouse=True)
def isolamento(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path
    # Sem prefixos de tarefa: os testes não dependem do provider do .env.
    settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL = "gemini", "gemini-embedding-001"


def make_document(admin, name="material.txt", content=b"Primeiro paragrafo.\n\nSegundo paragrafo.", **kwargs):
    document = Document.objects.create(
        title="Material secreto", file=ContentFile(content, name=name), uploaded_by=admin, **kwargs
    )
    document.courses.set(Course.objects.filter(code__in=["cc", "ads"]))
    return document


# --- retry de embedding ------------------------------------------------------------


def test_retry_com_backoff_em_erro_transitorio(monkeypatch, sleeps, settings):
    settings.EMBEDDING_RETRY_BASE_SECONDS = 5
    model = FakeEmbeddings(errors=[RuntimeError("429 RESOURCE_EXHAUSTED"), TimeoutError("timed out")])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)

    result = services._embed_documents(["a", "b"])

    assert len(result) == 2 and len(model.calls) == 3
    assert sleeps == [5, 10]  # backoff exponencial


def test_erro_permanente_nao_tenta_de_novo(monkeypatch, sleeps):
    model = FakeEmbeddings(errors=[ValueError("API key inválida")])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    with pytest.raises(ValueError):
        services._embed_documents(["a"])
    assert len(model.calls) == 1 and sleeps == []


def test_retry_esgota_tentativas(monkeypatch, sleeps, settings):
    settings.EMBEDDING_RETRY_ATTEMPTS = 3
    model = FakeEmbeddings(errors=[RuntimeError("429 quota")] * 5)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    with pytest.raises(RuntimeError):
        services._embed_documents(["a"])
    assert len(model.calls) == 3 and len(sleeps) == 2


def test_cota_esgotada_marca_falha_com_mensagem(admin, monkeypatch, sleeps):
    model = FakeEmbeddings(errors=[RuntimeError("429 quota")] * 5)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    document = make_document(admin)
    services._process_document(document)
    document.refresh_from_db()
    assert document.status == Document.Status.FAILED and "cota" in document.processing_error


# --- ingestão ponta a ponta (TXT) --------------------------------------------------


def test_ingestao_grava_chunks_search_vector_e_cabecalho_sem_titulo(admin, monkeypatch, sleeps):
    model = FakeEmbeddings()
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    document = make_document(admin)

    services._process_document(document)

    document.refresh_from_db()
    assert document.status == Document.Status.READY
    chunk = document.chunks.get()
    assert chunk.content.startswith("Primeiro paragrafo.")  # o content salvo não leva cabeçalho
    assert chunk.search_vector is not None
    assert document.chunks.filter(
        search_vector=SearchQuery("paragrafo", config="portuguese")
    ).exists()
    (text,) = model.calls[0]
    assert text.startswith("Documento do curso: ")
    assert "Ciência da Computação" in text and "Material secreto" not in text and "material.txt" not in text


def test_embedding_input_cabecalho_contextual():
    chunk = chunking.Chunk(content="corpo", heading="5. Modelo ER > 5.1 Entidades")
    assert services._embedding_input(chunk, "CC") == "Documento do curso: CC | Seção: 5. Modelo ER > 5.1 Entidades\n\ncorpo"
    assert services._embedding_input(chunking.Chunk(content="corpo"), "") == "corpo"


def test_prefixos_de_tarefa_so_para_embeddinggemma_no_ollama(settings):
    settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL = "ollama", "embeddinggemma"
    assert ai_providers.embedding_text_for("query", "oi") == "task: search result | query: oi"
    assert ai_providers.embedding_text_for("document", "oi") == "title: none | text: oi"
    settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL = "gemini", "gemini-embedding-001"
    assert ai_providers.embedding_text_for("query", "oi") == "oi"


# --- fallback de OCR ---------------------------------------------------------------


def fake_pipeline(monkeypatch, *, ocr_result):
    """convert_document devolve "plain"/"ocr"; o chunking de "plain" é vazio."""
    calls: list[bool] = []

    def convert(data, filename, *, ocr=False):
        calls.append(ocr)
        if ocr and isinstance(ocr_result, Exception):
            raise ocr_result
        return "ocr" if ocr else "plain"

    def chunk(doc):
        return [] if doc == "plain" else [chunking.Chunk(content="texto reconhecido " * 20, heading="Seção")]

    monkeypatch.setattr(extraction, "convert_document", convert)
    monkeypatch.setattr(chunking, "chunk_docling_document", chunk)
    return calls


def test_pdf_sem_texto_tenta_ocr(admin, monkeypatch, sleeps):
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: FakeEmbeddings())
    calls = fake_pipeline(monkeypatch, ocr_result=None)
    document = make_document(admin, name="scan.pdf", content=b"%PDF-fake")

    services._process_document(document)

    document.refresh_from_db()
    assert calls == [False, True]
    assert document.status == Document.Status.READY and document.chunks.count() == 1


def test_ocr_indisponivel_gera_mensagem_clara(admin, monkeypatch, sleeps):
    err = extraction.OcrUnavailableError("O PDF parece ser uma imagem escaneada e o OCR não está disponível")
    fake_pipeline(monkeypatch, ocr_result=err)
    document = make_document(admin, name="scan.pdf", content=b"%PDF-fake")

    services._process_document(document)

    document.refresh_from_db()
    assert document.status == Document.Status.FAILED
    assert "OCR" in document.processing_error and "escaneada" in document.processing_error


def test_pdf_com_texto_nao_aciona_ocr(admin, monkeypatch, sleeps):
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: FakeEmbeddings())
    calls: list[bool] = []
    monkeypatch.setattr(extraction, "convert_document", lambda d, f, *, ocr=False: calls.append(ocr) or "doc")
    monkeypatch.setattr(chunking, "chunk_docling_document", lambda d: [chunking.Chunk(content="x" * 500)])
    document = make_document(admin, name="ok.pdf", content=b"%PDF-fake")
    services._process_document(document)
    assert calls == [False]


# --- recover_documents -------------------------------------------------------------


def test_recover_reenfileira_so_os_presos(admin, monkeypatch):
    submitted: list[int] = []
    monkeypatch.setattr(services._executor, "submit", lambda fn, doc_id: submitted.append(doc_id))

    preso = make_document(admin, status=Document.Status.PROCESSING)
    recente = make_document(admin, status=Document.Status.PROCESSING)
    pronto = make_document(admin, status=Document.Status.READY)
    falhou = make_document(admin, status=Document.Status.FAILED)
    old = timezone.now() - timedelta(minutes=30)
    Document.objects.filter(id__in=[preso.id, pronto.id, falhou.id]).update(updated_at=old)

    call_command("recover_documents", "--no-wait", stdout=io.StringIO())

    assert submitted == [preso.id]
    preso.refresh_from_db()
    assert preso.status == Document.Status.PROCESSING and preso.updated_at > old
    recente.refresh_from_db()
    assert recente.status == Document.Status.PROCESSING


def test_recover_respeita_minutos(admin, monkeypatch):
    submitted: list[int] = []
    monkeypatch.setattr(services._executor, "submit", lambda fn, doc_id: submitted.append(doc_id))
    document = make_document(admin, status=Document.Status.PROCESSING)
    Document.objects.filter(id=document.id).update(updated_at=timezone.now() - timedelta(minutes=5))
    assert services.recover_stuck_documents(minutes=15) == []
    assert [d.id for d in services.recover_stuck_documents(minutes=3)] == [document.id]
    assert submitted == [document.id]
