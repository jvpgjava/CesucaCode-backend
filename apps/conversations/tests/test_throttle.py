import pytest
from django.conf import settings
from rest_framework.throttling import ScopedRateThrottle

from apps.conversations.views import ChatDailyThrottle, SendMessageView

pytestmark = pytest.mark.django_db


def test_taxas_configuradas_em_settings():
    rates = settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]
    assert rates["chat"] and rates["chat_daily"]
    assert SendMessageView.throttle_scope == "chat"
    assert ScopedRateThrottle in SendMessageView.throttle_classes
    assert ChatDailyThrottle in SendMessageView.throttle_classes
    assert ChatDailyThrottle.scope == "chat_daily"


def _send(api, conversation):
    return api.post(f"/api/conversations/{conversation.id}/messages/send/", {"content": "Oi"}, format="json")


def test_excesso_por_minuto_devolve_429_em_portugues(api, conversation, fake_llm, monkeypatch):
    fake_llm("ok")
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "chat", "2/min")

    for _ in range(2):
        response = _send(api, conversation)
        assert response.status_code == 200
        b"".join(response.streaming_content)

    blocked = _send(api, conversation)
    assert blocked.status_code == 429
    assert "limite de mensagens" in blocked.data["detail"]
    assert blocked["Retry-After"]


def test_teto_diario(api, conversation, fake_llm, monkeypatch):
    fake_llm("ok")
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "chat_daily", "1/day")

    first = _send(api, conversation)
    assert first.status_code == 200
    b"".join(first.streaming_content)
    assert _send(api, conversation).status_code == 429


def test_limite_e_por_usuario(api, conversation, fake_llm, coordinator, monkeypatch):
    from rest_framework.test import APIClient

    from apps.conversations.models import Conversation

    fake_llm("ok")
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "chat", "1/min")
    b"".join(_send(api, conversation).streaming_content)
    assert _send(api, conversation).status_code == 429

    other = APIClient()
    other.force_authenticate(coordinator)
    other_conversation = Conversation.objects.create(user=coordinator)
    assert _send(other, other_conversation).status_code == 200
