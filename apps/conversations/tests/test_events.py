import json

import pytest

from apps.conversations.events import (
    STATUS_LABELS,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    SuggestionsEvent,
    TokenEvent,
    status,
    to_sse,
)


def _parse(sse: str):
    assert sse.endswith("\n\n")
    lines = sse.rstrip("\n").split("\n")
    name = None
    if lines[0].startswith("event: "):
        name = lines.pop(0)[len("event: "):]
    assert len(lines) == 1 and lines[0].startswith("data: ")
    return name, json.loads(lines[0][len("data: "):])


def test_token_usa_formato_padrao_sem_event():
    assert _parse(to_sse(TokenEvent("olá"))) == (None, {"content": "olá"})


def test_token_preserva_acentos_e_quebras_de_linha():
    sse = to_sse(TokenEvent("ação\nnova"))
    assert "ação" in sse  # ensure_ascii=False
    assert _parse(sse)[1]["content"] == "ação\nnova"


def test_meta_status_suggestions_done_error():
    assert _parse(to_sse(MetaEvent(user_message_id=7, route=None))) == (
        "meta", {"user_message_id": 7, "route": None})
    assert _parse(to_sse(SuggestionsEvent(["A?", "B?"]))) == ("suggestions", {"items": ["A?", "B?"]})
    assert _parse(to_sse(DoneEvent(message_id=9))) == ("done", {"message_id": 9})
    assert _parse(to_sse(ErrorEvent("Algo deu errado"))) == ("error", {"message": "Algo deu errado"})


def test_status_usa_rotulo_fixo_do_dicionario():
    event = status("searching")
    assert event.label == STATUS_LABELS["searching"]
    assert _parse(to_sse(event)) == ("status", {"step": "searching", "label": "Procurando nas informações do curso…"})


def test_status_formata_placeholder():
    assert status("reading", n=3).label == "Lendo 3 trechos com atenção…"


def test_status_etapa_desconhecida_falha():
    with pytest.raises(KeyError):
        status("inexistente")


def test_to_sse_rejeita_tipo_desconhecido():
    with pytest.raises(TypeError):
        to_sse({"content": "x"})
