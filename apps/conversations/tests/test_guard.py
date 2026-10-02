from apps.conversations import guard
from apps.conversations.guardrail_cases import INTERNAL_TERMS, LEAK_FRAGMENTS


def test_guardrail_cases_reexporta_do_guard():
    assert INTERNAL_TERMS is guard.INTERNAL_TERMS
    assert LEAK_FRAGMENTS is guard.LEAK_FRAGMENTS


def test_texto_limpo_nao_gera_flags():
    assert guard.check_output("A prova de Banco de Dados é em outubro.", document_titles=["Calendário Acadêmico 2025"]) == []


def test_detecta_termo_interno_e_fragmento_do_prompt():
    flags = guard.check_output("Nos materiais enviados não consta. MODO ESTRITO ativo.", document_titles=[])
    assert any(f.startswith("internal_term") for f in flags)
    assert any(f.startswith("leak_fragment") for f in flags)


def test_detecta_titulo_de_documento_sem_acento_e_sem_extensao():
    titles = ["Código_Disciplinar.pdf"]
    flags = guard.check_output("Conforme o codigo disciplinar, o aluno pode recorrer.", document_titles=titles)
    assert flags == ["doc_title:codigo disciplinar"]


def test_titulos_curtos_de_uma_palavra_sao_ignorados():
    # "Manual" e "Calendário" apareceriam em respostas legítimas.
    assert guard.check_output("Veja o manual e o calendário.", document_titles=["Manual", "Calendário"]) == []


def test_titulo_exige_fronteira_de_palavra():
    assert guard.check_output("O grade curricularismo é outro assunto.", document_titles=["Grade Curricular"]) == []


def test_detecta_ref_vazada():
    flags = guard.check_output("A recursão chama a si mesma [T1] e termina [T2 · seção: Base].", document_titles=[])
    assert "ref_leak" in flags


def test_redact_remove_refs_e_espacos():
    text = "A recursão chama a si mesma [T1] e termina [T2 · seção: Caso base]. Fim [T3, T4]."
    assert guard.redact(text) == "A recursão chama a si mesma e termina. Fim."


def test_redact_nao_mexe_em_codigo_com_tipos_genericos():
    code = "def f(x: Dict[T1, T2]) -> lista[T1]: ..."
    assert guard.check_output(code, document_titles=[]) == []
    assert guard.redact(code) == code


def test_redact_respeita_flags_sem_ref_leak():
    assert guard.redact("texto [T1]", flags=["internal_term:x"]) == "texto [T1]"
    assert guard.redact("texto [T1]", flags=["ref_leak"]) == "texto"
