"""Identificação de plano de ensino/disciplina e contexto "pegajoso" dos chunks.

Os planos de ensino chegam concatenados num único material, cada um começando com um
cabeçalho de identificação, no seed (exportação do Docling) assim, com o texto GRUDADO:

    Plano de Ensino - 2025/ 1º SEMESTRE
    Curso: CIÊNCIA DA COMPUTAÇÃODisciplina: MODELAGEM DE DADOS
    6º SEMESTREGraduaçãoC/H Semestral: 80

Sem tratamento, os chunks seguintes ficam só com o heading "EMENTA"/"AVALIAÇÃO" e o nome
da disciplina se perde. Aqui:

- `parse_identification` extrai período letivo, curso, disciplina, semestre e C/H de
  um trecho de texto (robusto ao texto grudado e a quebras de linha no nome);
- `prepare_document` percorre o DoclingDocument em ordem, acha esses cabeçalhos, faz do
  primeiro item de cada plano um cabeçalho de seção (assim o chunker NUNCA mistura o fim
  de um plano com o começo do seguinte) e devolve o contexto vigente de cada item;
- `Chunk.discipline` (ver chunking.py) leva o contexto adiante até a próxima
  identificação: o nome da disciplina vira o primeiro segmento do `heading` (que vai ao
  LLM, e isso é desejado: nome de disciplina não é nome de documento) e entra no
  cabeçalho contextual do embedding;
- `build_disciplinas` deduplica (nome + curso, período mais recente) para a tabela
  estruturada `Disciplina`.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from docling_core.types.doc import ContentLayer
from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TextItem

# "Plano de Ensino - 2025/ 1º SEMESTRE" -> período letivo (ano/semestre do calendário).
PLANO_RE = re.compile(
    r"Plano\s+de\s+Ensino\s*[-–—:]?\s*(?P<ano>\d{4})\s*/\s*(?P<sem>\d)\s*[ºo°]?\s*SEMESTRE",
    re.IGNORECASE,
)
# O item de identificação começa com "Curso: ...Disciplina:" ou "Disciplina:" (evita casar
# a palavra no meio de um parágrafo comum).
IDENT_LINE_RE = re.compile(r"^\s*(?:Curso\s*:[^\n]*?)?Disciplina\s*:", re.IGNORECASE)
CURSO_RE = re.compile(r"Curso\s*:\s*(?P<curso>[^\n]+?)\s*(?=Disciplina\s*:)", re.IGNORECASE)
# O nome termina no fim da linha ou onde o texto grudado recomeça (semestre, nível, C/H).
DISCIPLINA_RE = re.compile(
    r"Disciplina\s*:\s*(?P<nome>.+?)\s*(?=\d{1,2}\s*[ºo°]\s*SEMESTRE|Gradua[çc][ãa]o|C\s*/\s*H|\n|$)",
    re.IGNORECASE,
)
SEMESTRE_RE = re.compile(r"(?P<n>\d{1,2})\s*[ºo°]\s*SEMESTRE", re.IGNORECASE)
CARGA_RE = re.compile(r"C\s*/\s*H(?:\s*Semestral)?\s*:?\s*(?P<ch>\d{1,4})", re.IGNORECASE)

# Palavras que ficam em minúsculas no nome capitalizado (exceto na primeira posição).
_MINUSCULAS = frozenset({"a", "o", "as", "os", "e", "ou", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "para", "por", "com", "à", "ao"})
_ROMANOS = re.compile(r"^(?=[IVXLC]+$)(?:X{0,3})(?:IX|IV|V?I{0,3})$")
MAX_CONTINUATION_ITEMS = 3


@dataclass
class Identificacao:
    nome: str  # já capitalizado ("Modelagem de Dados")
    curso: str = ""  # texto original do cabeçalho ("CIÊNCIA DA COMPUTAÇÃO")
    semestre: int | None = None  # semestre curricular da disciplina no curso
    carga_horaria: int | None = None
    periodo_letivo: str = ""  # "2025/1"


@dataclass(eq=False)
class DisciplineContext(Identificacao):
    """Contexto vigente de um plano; comparado por identidade (um por plano)."""

    header_text: str = field(default="", repr=False)  # texto do cabeçalho de seção criado
    member_refs: set[str] = field(default_factory=set, repr=False)  # itens do bloco de identificação

    def summary(self) -> str:
        """Conteúdo legível do chunk de identificação (no lugar do texto grudado)."""
        lines = [f"Disciplina: {self.nome}"]
        if self.curso:
            lines.append(f"Curso: {format_nome(self.curso)}")
        if self.semestre:
            lines.append(f"Semestre do curso: {self.semestre}º")
        if self.carga_horaria:
            lines.append(f"Carga horária semestral: {self.carga_horaria} h")
        if self.periodo_letivo:
            lines.append(f"Período letivo do plano: {self.periodo_letivo}")
        return "\n".join(lines)

    @property
    def label(self) -> str:
        """"Modelagem de Dados (6º semestre, C/H 80 h, plano 2025/1)" — vai ao embedding."""
        details = []
        if self.semestre:
            details.append(f"{self.semestre}º semestre")
        if self.carga_horaria:
            details.append(f"C/H {self.carga_horaria} h")
        if self.periodo_letivo:
            details.append(f"plano {self.periodo_letivo}")
        return f"{self.nome} ({', '.join(details)})" if details else self.nome


def format_nome(raw: str) -> str:
    """Capitaliza o nome (planos vêm em CAIXA ALTA): "CÁLCULO DIFERENCIAL E INTEGRAL I"
    -> "Cálculo Diferencial e Integral I". Nome que já tem minúsculas é preservado."""
    nome = " ".join((raw or "").split())
    if not nome or nome != nome.upper():
        return nome

    def cap(word: str, first: bool) -> str:
        if "-" in word:
            return "-".join(cap(part, True) for part in word.split("-"))
        if _ROMANOS.match(word) and len(word) <= 4:
            return word
        lower = word.lower()
        if not first and lower in _MINUSCULAS:
            return lower
        # Preserva a pontuação inicial (aspas/parênteses) ao capitalizar.
        for i, ch in enumerate(lower):
            if ch.isalpha():
                return lower[:i] + ch.upper() + lower[i + 1 :]
        return lower

    words = nome.split(" ")
    return " ".join(cap(w, i == 0) for i, w in enumerate(words))


def parse_identification(text: str) -> Identificacao | None:
    """Extrai a identificação de um plano de `text` (None se não houver "Disciplina:").
    Funciona com o texto grudado ("...COMPUTAÇÃODisciplina: X" / "6º SEMESTREGraduação...")."""
    match = DISCIPLINA_RE.search(text or "")
    if not match or not match.group("nome").strip():
        return None
    after = text[match.end():]
    periodo = PLANO_RE.search(text)
    curso = CURSO_RE.search(text)
    semestre = SEMESTRE_RE.search(after)
    carga = CARGA_RE.search(after)
    return Identificacao(
        nome=format_nome(match.group("nome")),
        curso=" ".join(curso.group("curso").split()) if curso else "",
        semestre=int(semestre.group("n")) if semestre else None,
        carga_horaria=int(carga.group("ch")) if carga else None,
        periodo_letivo=f"{periodo.group('ano')}/{periodo.group('sem')}" if periodo else "",
    )


def period_key(periodo: str) -> tuple[int, int]:
    """"2025/1" -> (2025, 1), para comparar períodos; vazio/inválido vale o menor."""
    match = re.match(r"^\s*(\d{4})\s*/\s*(\d)\s*$", periodo or "")
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _text_items(document: DoclingDocument) -> list[TextItem | SectionHeaderItem]:
    return [item for item, _ in document.iterate_items() if isinstance(item, (TextItem, SectionHeaderItem))]


def _promote_to_header(document: DoclingDocument, item: TextItem) -> SectionHeaderItem:
    """Troca (no lugar) um item de texto por um cabeçalho de seção de nível 1, mantendo
    a ref: o chunker passa a tratá-lo como fronteira de seção."""
    header = SectionHeaderItem(
        self_ref=item.self_ref,
        parent=item.parent,
        children=item.children,
        content_layer=item.content_layer,
        prov=item.prov,
        orig=item.orig,
        text=item.text,
        level=1,
    )
    index = int(item.self_ref.rsplit("/", 1)[1])
    document.texts[index] = header
    return header


def prepare_document(document: DoclingDocument) -> dict[str, DisciplineContext]:
    """Detecta os cabeçalhos de plano/disciplina e prepara o documento para o chunker.

    Efeitos no documento (idempotentes o bastante para rodar uma vez por conversão):
    - linhas de continuação do nome (o nome longo quebra e a 2ª linha vira um "cabeçalho"
      solto, ex.: `## APLICAÇÕES`) são anexadas à linha "Disciplina:" e escondidas;
    - o primeiro item do bloco de identificação vira cabeçalho de seção.

    Devolve {self_ref do item -> contexto vigente}, para todo item de texto a partir da
    primeira identificação (itens antes dela não aparecem)."""
    items = _text_items(document)
    contexts: dict[str, DisciplineContext] = {}
    current: DisciplineContext | None = None
    index = 0
    while index < len(items):
        item = items[index]
        if IDENT_LINE_RE.search(item.text) and not isinstance(item, SectionHeaderItem):
            block_start = index
            # Cabeçalho do plano ("Plano de Ensino - 2025/ 1º SEMESTRE") logo antes.
            lookback = index - 1
            while lookback >= 0 and index - lookback <= 2:
                if PLANO_RE.search(items[lookback].text):
                    block_start = lookback
                    break
                lookback -= 1

            # Continuação do nome: itens entre "Disciplina:" e a linha do semestre/C/H.
            end = index
            for ahead in range(index + 1, min(index + 1 + MAX_CONTINUATION_ITEMS + 1, len(items))):
                candidate = items[ahead]
                if SEMESTRE_RE.search(candidate.text) or CARGA_RE.search(candidate.text):
                    end = ahead
                    break
            continuation = items[index + 1 : end]
            if end > index and all(len(c.text) <= 80 and not SEMESTRE_RE.search(c.text) for c in continuation):
                if continuation:
                    item.text = " ".join([item.text.strip(), *[c.text.strip() for c in continuation]])
                    for c in continuation:
                        c.content_layer = ContentLayer.FURNITURE
            else:
                end = index  # nada confiável: não engole itens

            group = [items[i] for i in range(block_start, end + 1)]
            ident = parse_identification("\n".join(g.text for g in group if g.content_layer == ContentLayer.BODY))
            if ident is not None:
                first = items[block_start]
                if isinstance(first, TextItem) and not isinstance(first, SectionHeaderItem):
                    first = _promote_to_header(document, first)
                    items[block_start] = first
                current = DisciplineContext(
                    **ident.__dict__,
                    header_text=first.text,
                    member_refs={g.self_ref for g in group},
                )
                for member in group:
                    contexts[member.self_ref] = current
                index = end + 1
                continue
        if current is not None:
            contexts[item.self_ref] = current
        index += 1
    return contexts


def contexts_for_text_chunks(chunks) -> None:
    """Variante para TXT (sem estrutura de itens): procura a identificação no próprio
    conteúdo de cada chunk e passa o contexto adiante, chunk a chunk. Menos precisa que
    `prepare_document` (um chunk pode misturar o fim de um plano com o início do
    próximo), mas cobre o caso sem Docling."""
    current: DisciplineContext | None = None
    for chunk in chunks:
        ident = parse_identification(chunk.content)
        if ident is not None:
            current = DisciplineContext(**ident.__dict__)
        if current is not None:
            chunk.discipline = current
            chunk.heading = " > ".join([current.nome, *([chunk.heading] if chunk.heading else [])])


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).lower().split())


def resolve_course(curso_text: str, courses: list):
    """Curso (objeto) a que o cabeçalho se refere: casa o nome do curso cadastrado com
    o texto do cabeçalho; sem texto/casamento, só decide se o material tem UM curso."""
    wanted = normalize_key(curso_text)
    if wanted:
        for course in courses:
            name = normalize_key(course.name)
            if name and (name in wanted or wanted in name):
                return course
    return courses[0] if len(courses) == 1 else None


def build_disciplinas(chunks, courses: list) -> list[dict]:
    """Linhas (dict) para a tabela `Disciplina`, a partir dos contextos dos chunks.

    Um registro por (curso, nome); havendo vários planos da mesma disciplina, vale o de
    período mais recente (semestre curricular e C/H podem mudar entre períodos).
    `index_inicio` = índice do primeiro chunk daquele plano."""
    first_index: dict[int, tuple[DisciplineContext, int]] = {}
    for i, chunk in enumerate(chunks):
        ctx = getattr(chunk, "discipline", None)
        if ctx is not None and id(ctx) not in first_index:
            first_index[id(ctx)] = (ctx, i)

    best: dict[tuple, dict] = {}
    for ctx, index in first_index.values():
        course = resolve_course(ctx.curso, courses)
        key = (course.id if course else None, normalize_key(ctx.nome))
        row = {
            "course": course,
            "nome": ctx.nome,
            "semestre": ctx.semestre,
            "carga_horaria": ctx.carga_horaria,
            "periodo_letivo": ctx.periodo_letivo,
            "index_inicio": index,
        }
        current = best.get(key)
        if current is None or period_key(row["periodo_letivo"]) > period_key(current["periodo_letivo"]):
            best[key] = row
    return list(best.values())
