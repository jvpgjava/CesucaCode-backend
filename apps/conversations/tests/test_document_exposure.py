import pytest

from apps.documents.models import Document

HIDDEN = {"file", "uploaded_by_name", "processing_error"}


@pytest.fixture
def document(coordinator, course):
    doc = Document.objects.create(
        title="Ementa CC", uploaded_by=coordinator, status="failed", processing_error="Traceback secreto"
    )
    doc.courses.add(course)
    return doc


def test_estudante_nao_ve_arquivo_autor_nem_erro(api, document):
    response = api.get("/api/documents/")
    assert response.status_code == 200
    item = response.data["results"][0]
    assert item["title"] == "Ementa CC"
    assert HIDDEN.isdisjoint(item)
    assert {"id", "courses", "status", "chunk_count", "can_delete", "created_at"} <= set(item)


@pytest.mark.parametrize("role_fixture", ["coordinator"])
def test_coordenador_continua_vendo_tudo(document, request, role_fixture):
    from rest_framework.test import APIClient

    client = APIClient()
    client.force_authenticate(request.getfixturevalue(role_fixture))
    item = client.get("/api/documents/").data["results"][0]
    assert HIDDEN <= set(item)
    assert item["processing_error"] == "Traceback secreto"
