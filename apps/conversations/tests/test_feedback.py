import pytest

from apps.conversations.models import Message


@pytest.fixture
def answer(conversation):
    return Message.objects.create(conversation=conversation, role=Message.Role.ASSISTANT, content="resposta")


def _url(conversation, message):
    return f"/api/conversations/{conversation.id}/messages/{message.id}/feedback/"


def test_feedback_negativo_com_motivo_e_comentario(api, conversation, answer):
    response = api.patch(
        _url(conversation, answer),
        {"feedback": -1, "reason": "incorreta", "comment": "  errou a data  "},
        format="json",
    )
    assert response.status_code == 200
    assert response.data["feedback"] == -1
    assert response.data["feedback_reason"] == "incorreta"
    answer.refresh_from_db()
    assert answer.feedback_comment == "errou a data"


@pytest.mark.parametrize("novo", [1, None])
def test_mudar_para_positivo_ou_nulo_limpa_motivo(api, conversation, answer, novo):
    api.patch(_url(conversation, answer), {"feedback": -1, "reason": "outro", "comment": "x"}, format="json")
    response = api.patch(_url(conversation, answer), {"feedback": novo, "reason": "outro"}, format="json")
    assert response.status_code == 200
    answer.refresh_from_db()
    assert answer.feedback == novo
    assert answer.feedback_reason == "" and answer.feedback_comment == ""


def test_motivo_invalido_e_rejeitado(api, conversation, answer):
    response = api.patch(_url(conversation, answer), {"feedback": -1, "reason": "chato"}, format="json")
    assert response.status_code == 400
    assert "reason" in response.data


def test_comentario_acima_de_500_e_rejeitado(api, conversation, answer):
    response = api.patch(_url(conversation, answer), {"feedback": -1, "comment": "a" * 501}, format="json")
    assert response.status_code == 400


def test_detalhar_depois_nao_apaga_o_que_ja_existe(api, conversation, answer):
    api.patch(_url(conversation, answer), {"feedback": -1, "reason": "incompleta"}, format="json")
    api.patch(_url(conversation, answer), {"feedback": -1, "comment": "faltou exemplo"}, format="json")
    answer.refresh_from_db()
    assert answer.feedback_reason == "incompleta"
    assert answer.feedback_comment == "faltou exemplo"


def test_rating_continua_aceito_como_alias(api, conversation, answer):
    assert api.patch(_url(conversation, answer), {"rating": 1}, format="json").status_code == 200
    answer.refresh_from_db()
    assert answer.feedback == 1


def test_sem_feedback_nem_rating_e_400(api, conversation, answer):
    assert api.patch(_url(conversation, answer), {"reason": "outro"}, format="json").status_code == 400
