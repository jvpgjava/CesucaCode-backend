"""Guarda de saída em blocos validados (`output_guard`)."""

import pytest

from apps.ai_providers import services as ai_providers
from apps.ai_providers.services import StructuredResult
from apps.conversations import output_guard, services
from apps.conversations.output_guard import GuardVerdict, OutputGuard, guarded, regex_flags, split_ready

from .fakes import ScriptedChatModel

FRASE_REAL = (
    "Sobre as aplicações, as datas do calendário estão sem o mês, então não consigo dizer com "
    "segurança quando cada aplicação começa. "
)


@pytest.fixture
def judge(monkeypatch, settings):
    """Simula o juiz: `verdicts` é a fila de respostas (True = expõe o interno)."""
    settings.CHAT_GUARD_JUDGE_ENABLED = True
    state = {"verdicts": [], "calls": []}

    def invoke_structured(schema, messages, role="router", *, default=None, **kw):
        assert schema is GuardVerdict
        state["calls"].append(messages[-1].content)
        verdict = state["verdicts"].pop(0) if state["verdicts"] else False
        value = None if verdict is None else GuardVerdict(expoe_interno=verdict)
        return StructuredResult(value=value or default, ok=value is not None, usage={"input_tokens": 3}, error=None)

    monkeypatch.setattr(ai_providers, "invoke_structured", invoke_structured)
    return state


@pytest.fixture
def rewriter(monkeypatch):
    """O modelo de reescrita (papel router) devolve `state['text']`."""
    state = {"text": "A A1 é em 30/09.", "calls": 0}

    def get_chat_model(role="answer", **kw):
        state["calls"] += 1
        return ScriptedChatModel(pieces=[state["text"]])

    monkeypatch.setattr(ai_providers, "get_chat_model", get_chat_model)
    return state


def run(pieces, question="quando começam as aplicações?"):
    return "".join(guarded(pieces, question))


# --- blocos --------------------------------------------------------------------------


def test_divide_em_frases_sem_cortar_abreviacoes_nem_numeros():
    texto = "O Prof. Silva explica a matéria de forma clara e objetiva. A prova vale 3.5 pontos e ocorre em 30/09. Fim"
    blocks, rest = split_ready(texto)
    assert blocks == ["O Prof. Silva explica a matéria de forma clara e objetiva. A prova vale 3.5 pontos e ocorre em 30/09. "]
    assert rest == "Fim"


def test_paragrafo_fecha_bloco_mesmo_curto():
    blocks, rest = split_ready("Oi!\n\nTudo bem")
    assert blocks == ["Oi!\n\n"] and rest == "Tudo bem"


def test_texto_liberado_e_identico_ao_gerado_quando_nada_e_reprovado():
    pieces = ["A média ", "é 6,0. ", "Quem fica abaixo faz a AF, que vale de 0 a 5 pontos ", "e substitui a menor nota.\n\n", "Bons estudos!"]
    assert run(pieces, "como funciona a avaliação?") == "".join(pieces)


# --- regex ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    [
        FRASE_REAL,
        "O cronograma que recebi não traz o horário.",
        "Nos meus dados não aparece isso.",
        "O calendário não especifica o mês.",
        "Pelo documento disponível, a prova é dia 30.",
        "Segundo os materiais enviados, a média é 6.",
    ],
)
def test_regex_pega_metacomentario(texto):
    assert regex_flags(texto)


@pytest.mark.parametrize(
    "texto",
    [
        "Na base binária, 10 vale 2.",
        "Nos dados de entrada, use um vetor.",
        "No documento HTML, a tag body vem depois do head.",
        "Não tenho essa informação por aqui. Vale confirmar com a coordenação.",
        "O formato da prova é online, com 10 questões.",
        "Esse algoritmo não indica erro quando a lista está vazia.",
    ],
)
def test_regex_nao_pega_texto_legitimo(texto):
    assert regex_flags(texto) == []


# --- correção ------------------------------------------------------------------------


def test_caso_real_vira_abstencao_padrao(rewriter):
    rewriter["text"] = "As datas estão sem o mês."  # a reescrita também falha
    out = run([FRASE_REAL])
    assert "sem o mês" not in out and "segurança" not in out
    assert out.startswith(services.ABSTENTION_PHRASE)


def test_bloco_reprovado_e_reescrito_mantendo_os_fatos(rewriter):
    rewriter["text"] = "A A1 de IHC é em 30/09. O horário você confirma com o professor."
    out = run(["A A1 de IHC é em 30/09, mas o cronograma que recebi não traz o horário da prova. "])
    assert out.startswith("A A1 de IHC é em 30/09. O horário você confirma com o professor.")
    assert "recebi" not in out


def test_bloco_reprovado_depois_de_conteudo_util_e_descartado(rewriter):
    rewriter["text"] = "O calendário que recebi não traz isso."  # reescrita falha de novo
    out = run(["A A2 de Mobile é em 23/11, valendo 3,0 ou 4,0 conforme a apresentação na mostra.\n\n", FRASE_REAL])
    assert out.startswith("A A2 de Mobile é em 23/11") and services.ABSTENTION_PHRASE not in out
    assert "sem o mês" not in out


def test_juiz_reprova_o_que_a_regex_nao_pega(judge, rewriter):
    judge["verdicts"] = [True]
    rewriter["text"] = services.ABSTENTION_PHRASE
    out = run(["Essa parte eu não tenho clara, porque ninguém me passou a planilha de aplicações deste semestre. "])
    assert out.startswith(services.ABSTENTION_PHRASE)
    assert len(judge["calls"]) == 1


def test_juiz_so_roda_no_primeiro_bloco_e_com_gatilho(judge):
    texto = [
        "A média para aprovação é 6,0 e vale para todas as disciplinas do curso.\n\n",
        "Recursão é quando uma função chama a si mesma até atingir um caso base.\n\n",
        "Não tenho essa informação por aqui sobre a data exata da rematrícula.\n\n",
    ]
    assert run(texto) == "".join(texto)
    assert len(judge["calls"]) == 2  # 1º bloco + o com gatilho ("não tenho")


def test_falha_do_juiz_vale_como_aprovado(judge):
    judge["verdicts"] = [None]
    texto = "A A2 de Mobile é em 23/11, na aula 16 do semestre, valendo 3,0 ou 4,0 pontos. "
    assert run([texto]) == texto


def test_guarda_desligada_nao_e_usada(settings):
    settings.CHAT_OUTPUT_GUARD_ENABLED = False
    assert output_guard.enabled() is False


def test_estatisticas(rewriter):
    rewriter["text"] = "O calendário que recebi não traz isso."
    g = OutputGuard("q")
    "".join([*g.feed(FRASE_REAL), *g.finish()])
    assert g.stats.blocks == 1 and g.stats.rewrites == 1 and g.stats.replaced == 1
    assert any(f.startswith("meta_comment") for f in g.stats.flags)


def test_reescrita_recebe_o_texto_ja_mostrado(monkeypatch):
    seen = {}

    def get_chat_model(role="answer", **kw):
        class M:
            def invoke(self, messages):
                seen["human"] = messages[-1].content
                from langchain_core.messages import AIMessage

                return AIMessage(content="O mês exato você confirma com a secretaria acadêmica.")

        return M()

    monkeypatch.setattr(ai_providers, "get_chat_model", get_chat_model)
    out = run(["A rematrícula vai do dia 01 ao dia 30 para veteranos regulares do curso.\n\n", "As datas estão sem o mês. "])
    assert "<anterior>\nA rematrícula vai do dia 01" in seen["human"]
    assert out.endswith("O mês exato você confirma com a secretaria acadêmica. ")


def test_abstencao_generica_no_meio_da_resposta_vira_corte(rewriter):
    rewriter["text"] = services.ABSTENTION_PHRASE + " A secretaria confirma."
    out = run(["A rematrícula vai do dia 01 ao dia 30 para veteranos regulares do curso.\n\n", "As datas estão sem o mês. "])
    assert out == "A rematrícula vai do dia 01 ao dia 30 para veteranos regulares do curso.\n\n"


def test_reescrita_vazia_corta_o_trecho(rewriter):
    rewriter["text"] = ""
    out = run(["A A2 de Mobile é em 23/11, na aula 16 do semestre, valendo 3,0 ou 4,0.\n\n", "O cronograma que recebi não traz o horário. "])
    assert out == "A A2 de Mobile é em 23/11, na aula 16 do semestre, valendo 3,0 ou 4,0.\n\n"


def test_reescrita_vazia_no_inicio_vira_abstencao(rewriter):
    rewriter["text"] = ""
    assert run([FRASE_REAL]).startswith(services.ABSTENTION_PHRASE)


@pytest.mark.parametrize(
    "texto",
    ["vai até o dia 30 ou 31 (aparece as duas datas).", "o fim aparece como dia 30 e como dia 31.", "As duas versões aparecem no prazo."],
)
def test_regex_pega_duas_versoes(texto):
    assert regex_flags(texto)


def test_regex_nao_pega_aparece_tecnico():
    assert regex_flags("O erro aparece quando a lista está vazia.") == []
