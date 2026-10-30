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
from apps.documents import chunking, disciplinas, extraction, services
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


GEMINI_429 = (
    "Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 100. "
    "Please retry in 24.5s. 'retryDelay': '24s'"
)
GEMINI_429_DIARIO = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: embed_content_free_tier_requests, "
    "quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"
)


def test_retry_com_backoff_em_erro_transitorio(monkeypatch, sleeps, settings):
    settings.EMBEDDING_RETRY_BASE_SECONDS = 5
    model = FakeEmbeddings(errors=[TimeoutError("timed out"), ConnectionError("reset")])
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


def test_transitorio_esgota_tentativas(monkeypatch, sleeps, settings):
    settings.EMBEDDING_RETRY_ATTEMPTS = 3
    model = FakeEmbeddings(errors=[TimeoutError("timed out")] * 5)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    with pytest.raises(TimeoutError):
        services._embed_documents(["a"])
    assert len(model.calls) == 3 and len(sleeps) == 2


def test_cota_por_minuto_espera_o_tempo_indicado_e_continua(monkeypatch, sleeps, settings):
    # Mais esperas do que EMBEDDING_RETRY_ATTEMPTS: cota por minuto não conta como falha.
    settings.EMBEDDING_RETRY_ATTEMPTS = 2
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429)] * 4)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    waits = []

    result = services._embed_documents(["a"], on_wait=waits.append)

    assert len(result) == 1 and len(model.calls) == 5
    assert sleeps == [26.5] * 4 == waits  # "retry in 24.5s" + margem de 2s


def test_cota_sem_indicacao_de_tempo_espera_30s(monkeypatch, sleeps):
    model = FakeEmbeddings(errors=[RuntimeError("429 quota")])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    services._embed_documents(["a"])
    assert sleeps == [30.0]


def test_cota_por_minuto_desiste_no_teto_de_espera(monkeypatch, sleeps, settings):
    settings.EMBEDDING_QUOTA_MAX_WAIT_SECONDS = 60
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429)] * 10)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    with pytest.raises(RuntimeError):
        services._embed_documents(["a"])
    assert sleeps == [26.5, 26.5]  # a terceira passaria do teto de 60s


def test_cota_diaria_falha_na_hora(monkeypatch, sleeps):
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429_DIARIO)])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    with pytest.raises(services.DailyQuotaExceededError):
        services._embed_documents(["a"])
    assert sleeps == []


def test_cota_esgotada_por_tempo_demais_marca_falha_com_mensagem(admin, monkeypatch, sleeps, settings):
    settings.EMBEDDING_QUOTA_MAX_WAIT_SECONDS = 60
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429)] * 10)
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    document = make_document(admin)
    services._process_document(document)
    document.refresh_from_db()
    assert document.status == Document.Status.FAILED and "limite de uso" in document.processing_error
    assert document.processing_detail == "" and document.processing_resume_at is None


def test_cota_diaria_marca_falha_com_mensagem(admin, monkeypatch, sleeps):
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429_DIARIO)])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    document = make_document(admin)
    services._process_document(document)
    document.refresh_from_db()
    assert document.status == Document.Status.FAILED and "limite diário" in document.processing_error


def test_progresso_e_aviso_de_espera_durante_o_processamento(admin, monkeypatch, sleeps):
    """Enquanto espera a cota, o documento mostra a mensagem e quando retoma; no fim,
    o acompanhamento é limpo e o progresso vai a 100."""
    model = FakeEmbeddings(errors=[RuntimeError(GEMINI_429)])
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: model)
    document = make_document(admin)
    snapshots = []

    def sleep_and_snapshot(seconds):
        sleeps.append(seconds)
        snapshots.append(Document.objects.values(
            "processing_detail", "processing_progress", "processing_resume_at", "updated_at"
        ).get(pk=document.pk))

    monkeypatch.setattr(services.time, "sleep", sleep_and_snapshot)
    before = timezone.now()
    services._process_document(document)

    (during,) = snapshots
    assert during["processing_detail"] == services.QUOTA_WAIT_DETAIL
    assert during["processing_resume_at"] > before and during["updated_at"] >= before
    assert services.EXTRACTION_PROGRESS <= during["processing_progress"] < 100
    document.refresh_from_db()
    assert document.status == Document.Status.READY and document.processing_progress == 100
    assert document.processing_detail == "" and document.processing_resume_at is None


def test_reprocessar_reinicia_o_acompanhamento(admin, monkeypatch):
    monkeypatch.setattr(services._executor, "submit", lambda *a, **k: None)
    document = make_document(admin, processing_progress=100, processing_detail="x", status=Document.Status.FAILED)
    services.reprocess_document(document)
    document.refresh_from_db()
    assert document.status == Document.Status.PROCESSING and document.processing_progress == 0
    assert document.processing_detail == services.QUEUED_DETAIL


# --- ingestão ponta a ponta (TXT) --------------------------------------------------


def test_ingestao_grava_chunks_search_vector_e_cabecalho_com_titulo(admin, monkeypatch, sleeps):
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
    # O título do documento entra no embedding e no índice textual, nunca no conteúdo salvo.
    assert document.chunks.filter(search_vector=SearchQuery("secreto", config="portuguese")).exists()
    assert "Material secreto" not in chunk.content and "Material secreto" not in chunk.heading
    (text,) = model.calls[0]
    assert text.startswith("Documento: Material secreto | Documento do curso: ")
    assert "Ciência da Computação" in text and "material.txt" not in text


def test_embedding_input_cabecalho_contextual():
    chunk = chunking.Chunk(content="corpo", heading="5. Modelo ER > 5.1 Entidades")
    assert services._embedding_input(chunk, "CC") == "Documento do curso: CC | Seção: 5. Modelo ER > 5.1 Entidades\n\ncorpo"
    assert services._embedding_input(chunking.Chunk(content="corpo"), "") == "corpo"
    com_titulo = services._embedding_input(chunk, "CC", "Horário CC 2026/2 (Noturno)")
    assert com_titulo == (
        "Documento: Horário CC 2026/2 (Noturno) | Documento do curso: CC | "
        "Seção: 5. Modelo ER > 5.1 Entidades\n\ncorpo"
    )


def test_embedding_input_leva_a_disciplina_vigente():
    ctx = disciplinas.DisciplineContext(nome="Modelagem de Dados", semestre=6, carga_horaria=80, periodo_letivo="2025/1")
    chunk = chunking.Chunk(content="corpo", heading="Modelagem de Dados > EMENTA", discipline=ctx)
    text = services._embedding_input(chunk, "CC", "Planos de Ensino")
    assert "Disciplina: Modelagem de Dados (6º semestre, C/H 80 h, plano 2025/1)" in text
    assert "Seção: Modelagem de Dados > EMENTA" in text


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
