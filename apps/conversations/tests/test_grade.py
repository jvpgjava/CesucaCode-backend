"""Grade estruturada (`consultar_grade`, bloco "Disciplinas cadastradas" da rota direta)
e abstenção calibrada (fato institucional x conteúdo técnico). Sem LLM nem rede."""

import pytest
from django.core.files.base import ContentFile
from apps.accounts.models import Course, User
from apps.conversations import agent, pipeline, tools as tools_module
from apps.conversations.pipeline import Sufficiency
from apps.conversations.tests.agent_fakes import FakeRegistry
from apps.conversations.tests.test_pipeline import (  # noqa: F401 (fixtures e helpers)
    admin,
    admin_conversation,
    chunk,
    env,
    fake_agent,
    l1,
    run,
    text_of,
)
from apps.documents import grade
from apps.documents.models import Disciplina, Document


def make_document(admin, title, courses, *, status=Document.Status.READY):
    document = Document.objects.create(
        title=title, file=ContentFile(b"x", name="x.md"), uploaded_by=admin, status=status
    )
    document.courses.set(Course.objects.filter(code__in=courses))
    return document


def add(document, nome, semestre, ch, periodo, course_code=None, **kwargs):
    course = Course.objects.get(code=course_code) if course_code else None
    return Disciplina.objects.create(
        document=document, course=course, nome=nome, semestre=semestre, carga_horaria=ch, periodo_letivo=periodo, **kwargs
    )


@pytest.fixture
def grade_data(admin, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    planos_cc = make_document(admin, "Planos de Ensino de Ciência da Computação", ["cc"])
    add(planos_cc, "Modelagem de Dados", 6, 80, "2025/1", "cc")
    add(planos_cc, "Algoritmos e Programação", 1, 72, "2022/2", "cc")
    add(planos_cc, "Banco de Dados", 5, 72, "2024/2", "cc")
    # mesma disciplina em outro material, mais antiga: vale a mais recente (2024/2)
    outro = make_document(admin, "Planos antigos CC", ["cc"])
    add(outro, "Banco de Dados", 4, 60, "2020/1", "cc")
    planos_ads = make_document(admin, "Planos de Ensino de ADS", ["ads"])
    add(planos_ads, "Desenvolvimento Web", 3, 72, "2025/1", "ads")
    # material compartilhado: a disciplina é do CC e o aluno de ADS não deve vê-la
    compartilhado = make_document(admin, "Grade conjunta", ["cc", "ads"])
    add(compartilhado, "Teoria da Computação", 1, 72, "2022/2", "cc")
    # ainda processando: não aparece
    processando = make_document(admin, "Em processamento", ["cc"], status=Document.Status.PROCESSING)
    add(processando, "Disciplina Fantasma", 2, 40, "2025/1", "cc")


@pytest.fixture
def aluno_ads(db):
    return User.objects.create_user(
        email="ads@example.com", password="x", full_name="Aluno ADS", rgm="654321", course=Course.objects.get(code="ads")
    )


def nomes(items):
    return [d.nome for d in items]


# --- consulta e permissões --------------------------------------------------------


def test_aluno_cc_ve_so_disciplinas_do_proprio_curso_prontas_e_deduplicadas(grade_data, student):
    items = grade.list_disciplinas(student)

    assert nomes(items) == ["Algoritmos e Programação", "Teoria da Computação", "Banco de Dados", "Modelagem de Dados"]
    banco = next(d for d in items if d.nome == "Banco de Dados")
    assert (banco.semestre, banco.carga_horaria, banco.periodo_letivo) == (5, 72, "2024/2")
    assert "Disciplina Fantasma" not in nomes(items)  # documento não pronto
    assert "Desenvolvimento Web" not in nomes(items)  # documento de outro curso


def test_aluno_ads_nao_ve_disciplina_cc_de_material_compartilhado(grade_data, aluno_ads):
    assert nomes(grade.list_disciplinas(aluno_ads)) == ["Desenvolvimento Web"]


def test_admin_ve_tudo_e_filtros_por_curso_e_semestre(grade_data, admin):
    todos = grade.list_disciplinas(admin)
    assert "Desenvolvimento Web" in nomes(todos) and "Modelagem de Dados" in nomes(todos)
    assert nomes(grade.list_disciplinas(admin, curso="ads")) == ["Desenvolvimento Web"]
    assert nomes(grade.list_disciplinas(admin, curso="cc", semestre=1)) == [
        "Algoritmos e Programação",
        "Teoria da Computação",
    ]


def test_coordenador_so_ve_os_cursos_que_coordena(grade_data, coordinator):
    items = grade.list_disciplinas(coordinator)
    assert "Modelagem de Dados" in nomes(items) and "Desenvolvimento Web" not in nomes(items)


def test_disciplina_sem_curso_resolvido_usa_os_cursos_do_documento(admin, aluno_ads, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    doc = make_document(admin, "Grade ADS", ["ads"])
    add(doc, "Projeto Integrador", 2, 60, "2025/1", None)
    assert nomes(grade.list_disciplinas(aluno_ads, curso="ads")) == ["Projeto Integrador"]
    assert grade.list_disciplinas(aluno_ads, curso="cc") == []


def test_formato_agrupa_por_semestre_e_nao_cita_documento(grade_data, student):
    text = grade.grade_text(student)

    assert "1º semestre:\n- Algoritmos e Programação (72 h)\n- Teoria da Computação (72 h)" in text
    assert "5º semestre:\n- Banco de Dados (72 h)" in text
    assert "planos de ensino disponíveis" in text and "período letivo" in text
    assert "Planos de Ensino de Ciência da Computação" not in text and ".md" not in text
    assert grade.grade_text(student, semestre=11) == ""


# --- tool ---------------------------------------------------------------------------


def build(user, course_code="cc"):
    tools = tools_module.build_tools(user=user, course_code=course_code, registry=FakeRegistry(), allow_web=False)
    return {t.name: t for t in tools}


def test_tool_consultar_grade_usa_o_curso_da_conversa_e_respeita_permissoes(grade_data, student):
    out = build(student)["consultar_grade"].invoke({})

    assert "- Modelagem de Dados (80 h)" in out and "- Desenvolvimento Web" not in out
    assert "planos de ensino disponíveis" in out


def test_tool_consultar_grade_filtra_semestre_e_curso(grade_data, admin):
    ferramenta = build(admin, course_code=None)["consultar_grade"]

    assert "- Modelagem de Dados" in ferramenta.invoke({"curso": "cc", "semestre": 6})
    assert "Banco de Dados" not in ferramenta.invoke({"curso": "cc", "semestre": 6})
    out_ads = ferramenta.invoke({"curso": "ads"})
    assert "Desenvolvimento Web" in out_ads and "Modelagem" not in out_ads


def test_tool_consultar_grade_nunca_vaza_dados_de_outro_curso_mesmo_pedindo(grade_data, aluno_ads):
    out = build(aluno_ads, course_code="ads")["consultar_grade"].invoke({"curso": "cc"})

    assert "Modelagem de Dados" not in out and "Teoria da Computação" not in out
    assert out == tools_module.GRADE_EMPTY_MESSAGE


def test_tool_consultar_grade_sem_dados_orienta_a_buscar_nos_materiais(db, student):
    assert build(student)["consultar_grade"].invoke({}) == tools_module.GRADE_EMPTY_MESSAGE


def test_tool_rejeita_semestre_invalido(grade_data, student):
    with pytest.raises(Exception):
        build(student)["consultar_grade"].invoke({"semestre": 99})


def test_agent_prompt_menciona_a_tool():
    assert "consultar_grade" in agent.AGENT_PROMPT


# --- rota direta ----------------------------------------------------------------------


def test_rota_direta_de_grade_inclui_o_bloco_disciplinas_cadastradas(grade_data, conversation, env):
    env.l1 = l1("grade_disciplinas", query="disciplinas do curso")
    env.chunks = [chunk(content="Ementa de Modelagem.")]

    events = run(conversation, "Quais disciplinas existem no meu curso?")

    assert events[1].route == "direta"
    prompt = env.last_prompt
    assert "Disciplinas cadastradas:\n1º semestre:" in prompt
    assert "- Modelagem de Dados (80 h)" in prompt and "Desenvolvimento Web" not in prompt
    assert prompt.index("Disciplinas cadastradas:") < prompt.index("Ementa de Modelagem.")
    assert "Planos de Ensino de Ciência da Computação" not in prompt
    # a checagem de suficiência também enxerga o bloco
    (suficiencia,) = env.structured(Sufficiency)
    assert "Disciplinas cadastradas:" in suficiencia["messages"][-1].content


def test_rota_direta_de_grade_com_so_o_bloco_nao_depende_do_retrieval(grade_data, conversation, env):
    env.l1 = l1("grade_disciplinas")
    env.chunks = []

    run(conversation, "Quais disciplinas existem?")

    assert "- Algoritmos e Programação (72 h)" in env.last_prompt
    assert "Nenhuma informação de referência" not in env.last_prompt


def test_rota_direta_de_grade_sem_dados_segue_so_com_o_retrieval(db, conversation, env):
    env.l1 = l1("grade_disciplinas")
    env.chunks = [chunk(content="Trecho da grade.")]

    run(conversation, "Quais disciplinas existem?")

    assert "Disciplinas cadastradas:" not in env.last_prompt and "Trecho da grade." in env.last_prompt


def test_outras_intencoes_nao_recebem_o_bloco_de_grade(grade_data, conversation, env):
    env.l1 = l1("info_institucional")
    env.chunks = [chunk()]

    run(conversation, "Como funciona a avaliação?")

    assert "Disciplinas cadastradas:" not in env.last_prompt


def test_falha_na_grade_nao_derruba_o_turno(conversation, env, monkeypatch):
    env.l1 = l1("grade_disciplinas")
    env.chunks = [chunk(content="Trecho.")]
    monkeypatch.setattr(pipeline.tools, "grade_listing", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    events = run(conversation, "Quais disciplinas existem?")

    assert text_of(events) == "A média é 7."
    assert "Disciplinas cadastradas:" not in env.last_prompt


# --- abstenção calibrada ----------------------------------------------------------------


def test_conteudo_tecnico_direto_manda_explicar_com_conhecimento_geral(conversation, env):
    env.l1 = l1("conteudo_tecnico", query="normalização e formas normais")
    env.chunks = []

    run(conversation, "Explique normalização e formas normais")

    assert pipeline.TECHNICAL_NOTE in env.last_prompt
    assert "EXPLIQUE com conhecimento geral" in env.last_prompt


def test_conteudo_tecnico_em_modo_estrito_nao_ganha_a_nota(conversation, env, settings):
    settings.CHAT_ALLOW_GENERAL_KNOWLEDGE = False
    env.l1 = l1("conteudo_tecnico")
    env.chunks = []

    run(conversation, "Explique normalização")

    assert pipeline.TECHNICAL_NOTE not in env.last_prompt
    assert "MODO ESTRITO" in env.last_prompt


@pytest.mark.parametrize("intent", ["info_institucional", "grade_disciplinas", "exercicio_avaliativo"])
def test_outras_intencoes_nao_ganham_a_nota_tecnica(intent, conversation, env):
    env.l1 = l1(intent)
    env.chunks = [chunk()]

    run(conversation, "Qual o prazo?")

    assert pipeline.TECHNICAL_NOTE not in env.last_prompt


def test_info_institucional_mantem_a_nota_de_insuficiencia(conversation, env):
    env.l1 = l1("info_institucional")
    env.chunks = [chunk()]
    env.sufficiency = False

    run(conversation, "Quando é a colação de grau?")

    assert pipeline.INSUFFICIENT_NOTE in env.last_prompt


def test_suficiencia_so_roda_em_info_institucional_e_grade(conversation, env):
    env.l1 = l1("conteudo_tecnico")
    env.chunks = [chunk()]

    run(conversation, "Explique recursão")

    assert env.structured(Sufficiency) == []


def final_text(intent, *, allow_general=True, settings=None):
    settings.CHAT_ALLOW_GENERAL_KNOWLEDGE = allow_general
    messages = agent._final_messages(
        "SYS", [], "Pergunta?", FakeRegistry(), [], interrupted=False, intent=intent
    )
    return messages[-1].content


@pytest.mark.parametrize("intent", ["info_institucional", "grade_disciplinas"])
def test_agente_institucional_so_abstem_de_fato_institucional(intent, settings):
    text = final_text(intent, settings=settings)
    assert agent._INSTITUTIONAL_RULE in text
    assert agent._TECHNICAL_RULE not in text


@pytest.mark.parametrize("intent", ["conteudo_tecnico", None, "exercicio_avaliativo"])
def test_agente_tecnico_ou_misto_explica_com_conhecimento_geral(intent, settings):
    text = final_text(intent, settings=settings)
    assert agent._INSTITUTIONAL_RULE in text  # fato da instituição continua exigindo evidência
    assert agent._TECHNICAL_RULE in text and "Não se recuse a explicar" in text
    assert "não tem essa informação confirmada" in text


@pytest.mark.parametrize("intent", ["conteudo_tecnico", None, "info_institucional"])
def test_agente_modo_estrito_nao_enfraquece(intent, settings):
    text = final_text(intent, allow_general=False, settings=settings)
    assert agent._TECHNICAL_RULE not in text
    assert "MODO ESTRITO" in text  # sem trechos: nota estrita preservada


def test_composta_repassa_a_intencao_ao_agente(conversation, env, monkeypatch):
    env.l1 = l1("conteudo_tecnico", complexity="composta", query="normalização e índices")
    calls = fake_agent(monkeypatch, chunks=[chunk(7)])

    run(conversation, "Explique normalização e compare com índices")

    assert calls[0]["intent"] == "conteudo_tecnico"
