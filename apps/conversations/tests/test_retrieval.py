import pytest
from django.conf import settings

from apps.accounts.models import Course, User
from apps.ai_providers import services as ai_providers
from apps.conversations import retrieval, services
from apps.conversations.retrieval import RefRegistry, RetrievedChunk, rrf_fuse
from apps.documents import services as doc_services
from apps.documents.models import Document, DocumentChunk

DIMS = settings.EMBEDDING_DIMENSIONS


def vec(*head: float) -> list[float]:
    """Vetor de DIMS dimensões com os primeiros valores informados e o resto zero."""
    return [*head, *([0.0] * (DIMS - len(head)))]


class FakeEmbeddings:
    def __init__(self, query_vector):
        self.query_vector = query_vector

    def embed_query(self, text):
        return self.query_vector


@pytest.fixture
def embed(monkeypatch):
    """Define o vetor que o modelo de embedding (fake) devolve para a consulta."""

    def configure(query_vector):
        monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: FakeEmbeddings(query_vector))

    configure(vec(1.0))
    return configure


@pytest.fixture
def ads(db):
    return Course.objects.get(code="ads")


@pytest.fixture
def admin(db):
    return User.objects.create_user(
        email="admin@example.com", password="x", full_name="Admin", role=User.Role.CS_ADMIN
    )


def make_doc(admin, title, courses, chunks, status=Document.Status.READY):
    """`chunks`: lista de (heading, content, embedding)."""
    document = Document.objects.create(
        title=title, file="documents/x.txt", uploaded_by=admin, status=status
    )
    document.courses.set(courses)
    DocumentChunk.objects.bulk_create(
        [
            DocumentChunk(document=document, index=i, heading=h, content=c, embedding=e)
            for i, (h, c, e) in enumerate(chunks)
        ]
    )
    doc_services.update_search_vectors(document)
    return document


def ids(chunks):
    return [c.chunk_id for c in chunks]


# --- RRF (função pura) -------------------------------------------------------------


def test_rrf_soma_posicoes_dos_dois_rankings():
    scores = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    # Quem aparece nos dois rankings vence quem aparece em um só.
    assert sorted(scores, key=scores.get, reverse=True) == [1, 3, 2]


def test_rrf_vazio():
    assert rrf_fuse([[], []]) == {}


# --- search ------------------------------------------------------------------------


def test_hibrido_traz_acerto_textual_mesmo_com_vetor_distante(student, course, admin, embed):
    doc = make_doc(
        admin,
        "Plano",
        [course],
        [
            ("Avaliação", "A média das provas é sete.", vec(1.0)),  # só vetorial (distância 0)
            ("Ementa", "A ementa de Banco de Dados inclui modelagem.", vec(0.0, 1.0)),  # só textual
            ("Outro", "Texto sem relação alguma.", vec(0.0, 0.0, 1.0)),
        ],
    )
    result = retrieval.search(student, "ementa")
    by_id = {c.chunk_id: c for c in result}
    chunks = list(doc.chunks.order_by("index"))
    assert set(by_id) == {chunks[0].id, chunks[1].id}
    assert by_id[chunks[0].id].distance == pytest.approx(0.0, abs=1e-6)
    assert by_id[chunks[1].id].distance == pytest.approx(1.0, abs=1e-6)
    assert all(c.score > 0 for c in result)
    assert by_id[chunks[0].id].order == 0 and by_id[chunks[1].id].heading == "Ementa"


def test_chunk_nos_dois_rankings_fica_em_primeiro(student, course, admin, embed):
    doc = make_doc(
        admin,
        "Plano",
        [course],
        [
            ("A", "carga horária semanal", vec(0.9, 0.1)),
            ("B", "carga horária total da disciplina", vec(1.0)),
            ("C", "outro assunto", vec(1.0, 0.05)),
        ],
    )
    result = retrieval.search(student, "carga horária")
    chunks = list(doc.chunks.order_by("index"))
    # B e A casam no texto e no vetor; C só no vetor.
    assert ids(result)[-1] == chunks[2].id
    assert set(ids(result)[:2]) == {chunks[0].id, chunks[1].id}


def test_consulta_relaxada_respeita_margem_de_distancia(student, course, admin, embed):
    make_doc(
        admin,
        "Plano",
        [course],
        [
            ("Perto", "ementa resumida", vec(0.6, 0.8)),  # distância 0,4: dentro da margem (0,45)
            ("Longe", "ementa completa", vec(0.0, 1.0)),  # distância 1,0: fora
        ],
    )
    # Nenhum trecho tem "banco" -> consulta estrita vazia; a relaxada ("ementa or ...") casa.
    result = retrieval.search(student, "qual a ementa de banco de dados?")
    assert [c.heading for c in result] == ["Perto"]


def test_filtra_por_curso_e_permissoes(student, course, ads, admin, embed):
    cc_doc = make_doc(admin, "CC", [course], [("S", "regra de matrícula cc", vec(1.0))])
    ads_doc = make_doc(admin, "ADS", [ads], [("S", "regra de matrícula ads", vec(1.0))])
    both_doc = make_doc(admin, "Ambos", [course, ads], [("S", "regra de matrícula geral", vec(1.0))])
    make_doc(admin, "Preso", [course, ads], [("S", "matrícula em processamento", vec(1.0))], status="processing")

    def doc_ids(user, **kw):
        return {c.document_id for c in retrieval.search(user, "matrícula", **kw)}

    # Admin enxerga tudo; o filtro de curso restringe.
    assert doc_ids(admin) == {cc_doc.id, ads_doc.id, both_doc.id}
    assert doc_ids(admin, course_code="ads") == {ads_doc.id, both_doc.id}
    assert doc_ids(admin, course_code="cc") == {cc_doc.id, both_doc.id}
    for livre in ("ambos", "indefinido", None):
        assert doc_ids(admin, course_code=livre) == {cc_doc.id, ads_doc.id, both_doc.id}
    # Estudante de CC nunca vê material só de ADS, nem pedindo o filtro "ads".
    assert doc_ids(student) == {cc_doc.id, both_doc.id}
    assert doc_ids(student, course_code="ads") == {both_doc.id}


def test_flag_off_so_vetorial_com_corte_de_distancia(student, course, admin, embed, settings):
    settings.RAG_HYBRID_ENABLED = False
    make_doc(
        admin,
        "Plano",
        [course],
        [
            ("Perto", "texto qualquer", vec(1.0)),
            ("SoTexto", "ementa exata", vec(0.0, 1.0)),  # casaria no texto, mas está longe
        ],
    )
    result = retrieval.search(student, "ementa")
    assert [c.heading for c in result] == ["Perto"]
    assert result[0].distance == pytest.approx(0.0, abs=1e-6)


def test_corte_de_distancia_e_curriculum(student, course, admin, embed, settings):
    settings.RAG_HYBRID_ENABLED = False
    settings.RAG_MAX_DISTANCE = 0.30
    make_doc(admin, "Plano", [course], [("Borda", "x", vec(0.66, 0.751))])  # distância ~0,34
    assert retrieval.search(student, "pergunta") == []
    assert len(retrieval.search(student, "pergunta", curriculum=True)) == 1  # cutoff +0,05


def test_top_k_padrao_e_curriculum(student, course, admin, embed, settings):
    settings.RAG_HYBRID_ENABLED = False
    make_doc(admin, "Plano", [course], [(f"S{i}", f"conteúdo {i}", vec(1.0, i * 0.001)) for i in range(10)])
    assert len(retrieval.search(student, "pergunta")) == 6
    assert len(retrieval.search(student, "pergunta", curriculum=True)) == 8
    assert len(retrieval.search(student, "pergunta", top_k=3)) == 3


def test_falha_no_embedding_cai_para_so_textual(student, course, admin, monkeypatch):
    make_doc(admin, "Plano", [course], [("Ementa", "a ementa inclui modelagem", vec(1.0))])

    class Quebrado:
        def embed_query(self, text):
            raise RuntimeError("provider fora do ar")

    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: Quebrado())
    result = retrieval.search(student, "ementa")
    assert [c.heading for c in result] == ["Ementa"] and result[0].distance is None


def test_consulta_vazia(student):
    assert retrieval.search(student, "   ") == []


def test_retrieve_context_legacy_delega_ao_search(student, course, admin, embed):
    make_doc(admin, "Plano", [course], [("Avaliação", "média sete", vec(1.0))])
    (chunk,) = services.retrieve_context(student, "média")
    assert chunk.id == chunk.chunk_id and chunk.distance == pytest.approx(0.0, abs=1e-6)
    assert services.build_context_block([chunk]) == "[T1 · seção: Avaliação]\nmédia sete"


# --- neighbors ---------------------------------------------------------------------


def test_neighbors_janela_e_ordem(student, course, admin):
    doc = make_doc(admin, "Plano", [course], [(f"S{i}", f"c{i}", vec(1.0)) for i in range(6)])
    chunks = list(doc.chunks.order_by("index"))
    result = retrieval.neighbors(student, chunks[3].id, window=1)
    assert [c.order for c in result] == [2, 3, 4]
    assert all(c.distance is None for c in result)
    assert [c.order for c in retrieval.neighbors(student, chunks[0].id, window=2)] == [0, 1, 2]
    assert [c.order for c in retrieval.neighbors(student, chunks[3].id, window=0)] == [3]


def test_neighbors_respeita_permissoes_e_documento(student, ads, course, admin):
    outro = make_doc(admin, "ADS", [ads], [("S", "a", vec(1.0)), ("S", "b", vec(1.0))])
    proprio = make_doc(admin, "CC", [course], [("S", "c", vec(1.0)), ("S", "d", vec(1.0))])
    assert retrieval.neighbors(student, outro.chunks.first().id) == []
    assert {c.document_id for c in retrieval.neighbors(student, proprio.chunks.first().id)} == {proprio.id}


# --- RefRegistry / format_context --------------------------------------------------


def chunk(chunk_id, heading="", content="texto"):
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=1, heading=heading, content=content, score=0.1, distance=0.2, order=chunk_id
    )


def test_registry_refs_sequenciais_e_dedup():
    registry = RefRegistry()
    pairs = registry.add([chunk(10), chunk(11), chunk(10)])
    assert [ref for ref, _ in pairs] == ["T1", "T2"]
    again = registry.add([chunk(11), chunk(12)])
    assert [ref for ref, _ in again] == ["T2", "T3"]  # o 11 mantém a ref
    assert registry.chunk_ids == [10, 11, 12]
    assert registry.get("T3").chunk_id == 12 and registry.get("t1").chunk_id == 10
    assert registry.get("T9") is None


def test_format_context_sem_titulo_do_documento():
    pairs = [("T1", chunk(1, heading="  Avaliação >  Média ", content="A média é 7.")), ("T2", chunk(2, content="Faltas."))]
    text = retrieval.format_context(pairs)
    assert text == "[T1 · seção: Avaliação > Média]\nA média é 7.\n\n---\n\n[T2]\nFaltas."
    assert "Origem" not in text
    assert retrieval.format_context([]) == ""


def test_rerank_e_identidade():
    chunks = [chunk(1), chunk(2)]
    assert retrieval.rerank("q", chunks) is chunks
