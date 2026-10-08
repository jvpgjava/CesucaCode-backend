"""Montagem do system prompt por rota (intenção).

Os arquivos `.md` ficam em `SYSTEM_PROMPT_PATH` (pasta `prompts/sofia/`). Tudo o que
NÃO está em `ROUTE_MODULES` é **base** e entra sempre (identidade, escopo, fontes,
anti-alucinação, segurança, idiomas). Os arquivos listados em `ROUTE_MODULES` só
entram nas intenções indicadas: prompt menor, aderência maior.

Ordem: estático primeiro, na ordem alfabética dos arquivos (bom para o cache de
prompt do provedor); a parte dinâmica (nível da escada de dicas) vai por último,
e o chamador acrescenta depois dela o bloco de perfil do usuário.
"""

from pathlib import Path

from django.conf import settings

# Arquivo -> intenções em que ele entra. Fácil de auditar: o que não aparece aqui é base.
ROUTE_MODULES: dict[str, frozenset[str]] = {
    "27-caminho-de-estudo.md": frozenset({"grade_disciplinas"}),
    "29-meta.md": frozenset({"meta"}),
    "30-pedagogia.md": frozenset({"conteudo_tecnico", "exercicio_avaliativo"}),
    "35-escada-de-dicas.md": frozenset({"exercicio_avaliativo"}),
}

HINT_LEVELS = (1, 2, 3)
_HINT_LEVEL_NAMES = {
    1: "conceito e pergunta socrática",
    2: "dica direcionada",
    3: "esqueleto ou pseudocódigo parcial",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def prompt_files(intent: str | None) -> list[Path]:
    """Arquivos do prompt para a intenção, em ordem alfabética. `intent=None` devolve
    todos (comportamento legado)."""
    path = Path(settings.SYSTEM_PROMPT_PATH)
    if path.is_dir():
        files = sorted(path.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"SYSTEM_PROMPT_PATH aponta para uma pasta sem arquivos .md: {path}")
    elif path.is_file():
        return [path]  # arquivo único: não há como separar módulos
    else:
        raise FileNotFoundError(f"SYSTEM_PROMPT_PATH aponta para um arquivo/pasta que não existe: {path}")

    if intent is None:
        return files
    return [f for f in files if f.name not in ROUTE_MODULES or intent in ROUTE_MODULES[f.name]]


def clamp_hint_level(level: int | None) -> int:
    try:
        return min(max(int(level), HINT_LEVELS[0]), HINT_LEVELS[-1])
    except (TypeError, ValueError):
        return HINT_LEVELS[0]


def _hint_level_block(level: int | None) -> str:
    level = clamp_hint_level(level)
    return (
        "# Nível atual da escada de dicas\n\n"
        f"Nível {level} de {HINT_LEVELS[-1]} ({_HINT_LEVEL_NAMES[level]}). "
        "Responda neste degrau, sem pular para o seguinte."
    )


def build_system_prompt(intent: str | None, *, hint_level: int | None = None) -> str:
    """System prompt (sem o bloco de perfil do usuário) para a intenção dada.

    `intent=None` concatena todos os arquivos, como o `get_system_prompt()` antigo.
    Em `exercicio_avaliativo`, o nível atual da escada de dicas entra no fim."""
    parts = [_read(f) for f in prompt_files(intent)]
    if intent == "exercicio_avaliativo":
        parts.append(_hint_level_block(hint_level))
    return "\n\n".join(parts)
