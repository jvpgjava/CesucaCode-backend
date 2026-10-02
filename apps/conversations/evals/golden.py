"""Carregamento e validação do golden set (`golden.yaml`)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

GOLDEN_PATH = Path(__file__).with_name("golden.yaml")

PERSONAS = ("student_cc", "student_ads", "admin", "coordinator_cc")
EXPECTS = ("resposta", "sem_info", "recusa", "clarificacao")
ROUTES = ("direta", "composta", "meta", "recusa", "pedagogica", "clarificacao")
HISTORY_ROLES = ("user", "assistant")


class GoldenError(ValueError):
    """Golden set inválido; a mensagem lista todos os problemas encontrados."""


@dataclass
class GoldenCase:
    id: str
    question: str
    persona: str
    expect: str
    tags: list[str]
    history: list[dict] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    retrieval_expect: list[str] = field(default_factory=list)
    route_expect: str | None = None
    rubric: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "persona": self.persona,
            "expect": self.expect,
            "tags": list(self.tags),
            "history": [dict(turn) for turn in self.history],
            "must_include": list(self.must_include),
            "must_not_include": list(self.must_not_include),
            "retrieval_expect": list(self.retrieval_expect),
            "route_expect": self.route_expect,
            "rubric": self.rubric,
        }


_KNOWN_KEYS = {
    "id", "question", "persona", "history", "expect", "must_include", "must_not_include",
    "retrieval_expect", "route_expect", "rubric", "tags",
}


def _str_list(raw: dict, key: str, problems: list[str], where: str) -> list[str]:
    value = raw.get(key) or []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        problems.append(f"{where}: `{key}` deve ser uma lista de textos não vazios")
        return []
    return list(value)


def validate_cases(raw_cases) -> list[GoldenCase]:
    """Valida a lista crua (já lida do YAML) e devolve os casos tipados.
    Levanta `GoldenError` com TODOS os problemas, não só o primeiro."""
    if not isinstance(raw_cases, list) or not raw_cases:
        raise GoldenError("o golden set deve ser uma lista não vazia de casos")

    problems: list[str] = []
    cases: list[GoldenCase] = []
    seen: set[str] = set()

    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            problems.append(f"caso #{index + 1}: deve ser um mapa")
            continue
        case_id = raw.get("id")
        where = f"caso {case_id!r}" if isinstance(case_id, str) and case_id else f"caso #{index + 1}"

        if not isinstance(case_id, str) or not case_id.strip():
            problems.append(f"{where}: `id` obrigatório")
        elif case_id in seen:
            problems.append(f"{where}: `id` duplicado")
        else:
            seen.add(case_id)

        unknown = set(raw) - _KNOWN_KEYS
        if unknown:
            problems.append(f"{where}: campos desconhecidos {sorted(unknown)}")

        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{where}: `question` obrigatório")
        persona = raw.get("persona")
        if persona not in PERSONAS:
            problems.append(f"{where}: `persona` deve ser uma de {PERSONAS}")
        expect = raw.get("expect")
        if expect not in EXPECTS:
            problems.append(f"{where}: `expect` deve ser um de {EXPECTS}")
        route_expect = raw.get("route_expect")
        if route_expect is not None and route_expect not in ROUTES:
            problems.append(f"{where}: `route_expect` deve ser um de {ROUTES}")

        tags = _str_list(raw, "tags", problems, where)
        if not tags:
            problems.append(f"{where}: `tags` precisa ter ao menos uma tag")
        must_include = _str_list(raw, "must_include", problems, where)
        must_not_include = _str_list(raw, "must_not_include", problems, where)
        retrieval_expect = _str_list(raw, "retrieval_expect", problems, where)
        rubric = raw.get("rubric") or ""
        if not isinstance(rubric, str):
            problems.append(f"{where}: `rubric` deve ser texto")
            rubric = ""

        history = raw.get("history") or []
        if not isinstance(history, list):
            problems.append(f"{where}: `history` deve ser uma lista de turnos")
            history = []
        else:
            for turn in history:
                if (
                    not isinstance(turn, dict)
                    or turn.get("role") not in HISTORY_ROLES
                    or not isinstance(turn.get("content"), str)
                    or not turn["content"].strip()
                ):
                    problems.append(f"{where}: turno de `history` inválido (use role user|assistant e content)")
                    break

        # Coerência entre expectativa e fatos esperados.
        if expect == "resposta" and not must_include and not rubric:
            problems.append(f"{where}: `resposta` exige `must_include` ou `rubric`")
        if expect in ("sem_info", "recusa", "clarificacao") and must_include:
            problems.append(f"{where}: `{expect}` não deve ter `must_include`")

        if isinstance(case_id, str) and isinstance(question, str) and persona in PERSONAS and expect in EXPECTS:
            cases.append(
                GoldenCase(
                    id=case_id,
                    question=question,
                    persona=persona,
                    expect=expect,
                    tags=tags,
                    history=[dict(turn) for turn in history if isinstance(turn, dict)],
                    must_include=must_include,
                    must_not_include=must_not_include,
                    retrieval_expect=retrieval_expect,
                    route_expect=route_expect,
                    rubric=rubric,
                )
            )

    if problems:
        raise GoldenError("golden set inválido:\n- " + "\n- ".join(problems))
    return cases


def load_golden(path: str | Path | None = None) -> list[GoldenCase]:
    """Lê e valida o golden set (padrão: `golden.yaml` ao lado deste módulo)."""
    target = Path(path) if path else GOLDEN_PATH
    with open(target, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return validate_cases(raw)


def filter_cases(cases: list[GoldenCase], *, tags=None, ids=None, limit: int | None = None) -> list[GoldenCase]:
    """`tags`: mantém casos com QUALQUER das tags. `ids`: ids exatos ou prefixos terminados em `*`."""
    selected = cases
    if tags:
        wanted = set(tags)
        selected = [case for case in selected if wanted & set(case.tags)]
    if ids:
        def match(case_id: str) -> bool:
            return any(
                case_id.startswith(pattern[:-1]) if pattern.endswith("*") else case_id == pattern
                for pattern in ids
            )

        selected = [case for case in selected if match(case.id)]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected
