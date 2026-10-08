import pytest

from apps.conversations import prompts, services
from apps.conversations.prompts import ROUTE_MODULES, build_system_prompt, prompt_files

BASE_FILES = {
    "00-identidade.md", "10-escopo.md", "20-materiais-e-fontes.md", "22-como-falar-das-fontes.md",
    "25-anti-alucinacao.md", "40-seguranca.md", "50-idiomas-e-ofuscacao.md",
}


def names(intent):
    return {f.name for f in prompt_files(intent)}


def test_modulos_declarados_existem_na_pasta():
    assert set(ROUTE_MODULES) <= names(None)


@pytest.mark.parametrize(
    "intent",
    ["meta", "info_institucional", "grade_disciplinas", "conteudo_tecnico", "exercicio_avaliativo",
     "fora_escopo", "manipulacao"],
)
def test_base_entra_em_toda_intencao(intent):
    assert BASE_FILES <= names(intent)


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("grade_disciplinas", {"27-caminho-de-estudo.md"}),
        ("meta", {"29-meta.md"}),
        ("conteudo_tecnico", {"30-pedagogia.md"}),
        ("exercicio_avaliativo", {"30-pedagogia.md", "35-escada-de-dicas.md"}),
        ("info_institucional", set()),
        ("fora_escopo", set()),
        ("manipulacao", set()),
    ],
)
def test_modulos_por_rota(intent, expected):
    assert names(intent) - BASE_FILES == expected


def test_sem_intencao_inclui_todos_os_arquivos():
    assert names(None) == BASE_FILES | set(ROUTE_MODULES)


def test_sem_intencao_equivale_ao_get_system_prompt_legado():
    assert build_system_prompt(None) == services.get_system_prompt()


def test_caminho_de_estudo_so_na_grade():
    assert "# Grade, disciplinas e caminho de estudo" in build_system_prompt("grade_disciplinas")
    assert "# Grade, disciplinas e caminho de estudo" not in build_system_prompt("conteudo_tecnico")


def test_prompt_por_rota_e_menor_que_o_completo():
    full = len(build_system_prompt(None))
    for intent in ("info_institucional", "conteudo_tecnico", "meta"):
        assert len(build_system_prompt(intent)) < full


def test_ordem_estatica_primeiro_e_nivel_de_dica_por_ultimo():
    text = build_system_prompt("exercicio_avaliativo", hint_level=2)
    assert text.index("# Identidade") < text.index("# Postura pedagógica") < text.index("# Escada de dicas")
    assert text.split("\n\n")[-1].startswith("Nível 2 de 3")
    assert text.index("# Escada de dicas") < text.index("# Nível atual da escada de dicas")


@pytest.mark.parametrize(("level", "shown"), [(1, 1), (2, 2), (3, 3), (None, 1), (0, 1), (9, 3), ("x", 1)])
def test_nivel_da_escada(level, shown):
    text = build_system_prompt("exercicio_avaliativo", hint_level=level)
    assert f"Nível {shown} de 3" in text


def test_nivel_so_aparece_em_exercicio():
    assert "# Nível atual da escada" not in build_system_prompt("conteudo_tecnico", hint_level=2)
    assert "# Nível atual da escada" not in build_system_prompt(None, hint_level=2)


def test_escada_proibe_solucao_completa():
    text = build_system_prompt("exercicio_avaliativo")
    assert "nunca a solução completa" in text.lower()


def test_respeita_system_prompt_path_com_arquivo_unico(settings, tmp_path):
    arquivo = tmp_path / "prompt.md"
    arquivo.write_text("só isto", encoding="utf-8")
    settings.SYSTEM_PROMPT_PATH = str(arquivo)
    assert build_system_prompt("grade_disciplinas") == "só isto"


def test_respeita_system_prompt_path_com_pasta_propria(settings, tmp_path):
    (tmp_path / "00-base.md").write_text("base", encoding="utf-8")
    (tmp_path / "27-caminho-de-estudo.md").write_text("grade", encoding="utf-8")
    (tmp_path / "99-extra.md").write_text("extra", encoding="utf-8")  # fora de ROUTE_MODULES: é base
    settings.SYSTEM_PROMPT_PATH = str(tmp_path)
    assert build_system_prompt("conteudo_tecnico") == "base\n\nextra"
    assert build_system_prompt("grade_disciplinas") == "base\n\ngrade\n\nextra"


def test_erro_claro_quando_o_caminho_nao_existe(settings, tmp_path):
    settings.SYSTEM_PROMPT_PATH = str(tmp_path / "nao-existe")
    with pytest.raises(FileNotFoundError):
        build_system_prompt("meta")
    assert prompts.prompt_files  # módulo importável como apps.conversations.prompts
