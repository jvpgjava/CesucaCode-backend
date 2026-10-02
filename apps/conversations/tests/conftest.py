import pytest
from django.core.cache import cache
from rest_framework.test import APIClient

from apps.accounts.models import Course, User
from apps.ai_providers import services as ai_providers
from apps.conversations import services
from apps.conversations.models import Conversation

from .fakes import ScriptedChatModel


@pytest.fixture(autouse=True)
def _limpa_cache():
    """Os contadores do throttle ficam no cache: isola cada teste."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def course(db):
    return Course.objects.get(code="cc")


@pytest.fixture
def student(course):
    return User.objects.create_user(
        email="aluno@example.com", password="x", full_name="Aluno", rgm="123456", course=course
    )


@pytest.fixture
def coordinator(course):
    user = User.objects.create_user(
        email="coord@example.com", password="x", full_name="Coord", role=User.Role.CS_COORDINATOR
    )
    user.coordinated_courses.add(course)
    return user


@pytest.fixture
def conversation(student):
    return Conversation.objects.create(user=student)


@pytest.fixture
def api(student):
    client = APIClient()
    client.force_authenticate(student)
    return client


@pytest.fixture
def fake_llm(monkeypatch):
    """Troca o LLM e o retrieval por fakes (sem rede). Devolve uma função que
    define o que o modelo emite."""

    def configure(*pieces, **kwargs):
        model = ScriptedChatModel(pieces=list(pieces), **kwargs)
        monkeypatch.setattr(ai_providers, "get_chat_model", lambda *a, **k: model)
        return model

    monkeypatch.setattr(services, "retrieve_context", lambda *a, **k: [])
    return configure
