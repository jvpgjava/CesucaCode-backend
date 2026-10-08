"""Guarda de saída: confere o texto final da resposta antes de salvá-lo.

Detecta vazamentos do funcionamento interno (vocabulário de RAG, trechos do
prompt), títulos de documentos que o usuário enxerga e referências opacas
`[T1]` que deveriam ser só internas. Só as referências são redigidas (removê-las
é seguro); os demais casos viram flags no trace, para análise.

Limite importante: o texto é transmitido por streaming, então o aluno já viu a
resposta quando a guarda roda — ela NÃO desfaz o que foi mostrado. O que ela
garante é que a versão salva no histórico (e reenviada ao modelo nos próximos
turnos) saia sem referências, e que cada ocorrência fique registrada.
"""

import re
import unicodedata

# Vocabulário do funcionamento interno que a S.O.F.I.A nunca deve mostrar ao
# aluno (ver prompts/sofia/22-como-falar-das-fontes.md). Vale pra TODA resposta.
INTERNAL_TERMS = (
    r"materiais? (enviados?|did[aá]ticos?|carregados?)"
    r"|(nos?|dos?|pelos?|aos?) materiais?\b"
    r"|trechos? (d[eoa]s? )?(materiais?|documentos?|arquivos?)"
    r"|fragmentad"
    r"|base de (conhecimento|dados)"
    r"|contexto dos"
    r"|n[aã]o foi (enviad|carregad)"
    r"|(na|a|da) (grade|organiza[cç][aã]o|rela[cç][aã]o) (apresentada|exibida)"
    r"|est[aá] misturad|dispos[iç][aã]o (do|das|dos|das demais)"
    r"|consegui (identificar|confirmar)|informa[cç][oõ]es dispon[ií]veis"
    r"|(disciplinas|componentes) identificad[oa]s|(rela[cç][aã]o|lista) que consigo|que consigo confirmar"
    r"|n[aã]o consigo pesquisar|pesquisar (a|na) internet"
    r"|\(?fonte:"
    r"|pesquisei|pesquisa (na|feita na) (internet|web)|na internet, encontrei"
    r"|refer[eê]ncias? externas?"
    r"|https?://|www\."
)

# Trechos do próprio prompt que nunca devem aparecer na resposta (vazamento).
LEAK_FRAGMENTS = [
    "Como recusar",
    "Postura pedagógica",
    "Anti-alucinação",
    "MODO ESTRITO",
    "DAN ativada",
    "modo desenvolvedor ativado",
]

# Referência opaca de trecho: "[T1]", "[T2 · seção: Avaliação]", "[T1, T3]".
# O lookbehind evita tocar em código/tipos genéricos (`lista[T1]`, `Dict[T1, T2]`).
_REF_RE = re.compile(r"(?P<sp>[ \t]*)(?<![\w\])])\[\s*T\d+\b[^\]\n]{0,120}\]")

_INTERNAL_RE = re.compile(INTERNAL_TERMS, re.IGNORECASE)

FLAG_INTERNAL = "internal_term"
FLAG_LEAK = "leak_fragment"
FLAG_TITLE = "doc_title"
FLAG_REF = "ref_leak"


def _normalize(text: str) -> str:
    """Minúsculas, sem acentos, sem extensão de arquivo e com separadores de
    nome de arquivo (_ - .) virando espaço."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\.(pdf|docx?|pptx?|txt|md|html?)$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"[_\-.]+", " ", text.lower())
    return " ".join(text.split())


def _matchable_titles(document_titles) -> list[str]:
    """Títulos específicos o bastante para virar detecção: pelo menos 2 palavras E
    8 caracteres (normalizados). Títulos de uma palavra só ("Manual", "Calendário")
    apareceriam em qualquer resposta legítima e gerariam falso positivo; a flag é
    só um sinal para auditoria, não bloqueia nem redige nada."""
    result = []
    for title in document_titles or []:
        normalized = _normalize(title or "")
        if len(normalized) >= 8 and len(normalized.split()) >= 2:
            result.append(normalized)
    return result


def check_output(text: str, *, document_titles=()) -> list[str]:
    """Devolve a lista de flags (strings curtas) encontradas no texto final.
    Lista vazia = nada suspeito."""
    flags: list[str] = []
    if not text:
        return flags

    lowered = text.lower()
    for fragment in LEAK_FRAGMENTS:
        if fragment.lower() in lowered:
            flags.append(f"{FLAG_LEAK}:{fragment}")

    internal = _INTERNAL_RE.search(text)
    if internal:
        flags.append(f"{FLAG_INTERNAL}:{internal.group(0).strip()[:60]}")

    normalized_text = _normalize(text)
    for title in _matchable_titles(document_titles):
        if re.search(rf"(?<!\w){re.escape(title)}(?!\w)", normalized_text):
            flags.append(f"{FLAG_TITLE}:{title[:60]}")

    if _REF_RE.search(text):
        flags.append(FLAG_REF)
    return flags


def redact(text: str, flags=None) -> str:
    """Remove as referências opacas `[T#]` do texto. Se `flags` for informado e não
    contiver `ref_leak`, devolve o texto intacto. Os demais flags não alteram o
    texto: reescrever frases sem quebrar o sentido não é seguro de forma automática."""
    if flags is not None and FLAG_REF not in flags:
        return text
    return _REF_RE.sub("", text)
