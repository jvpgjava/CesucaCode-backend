"""Identificação de plano/disciplina, contexto pegajoso e tabela `Disciplina`."""

import io
from pathlib import Path

import pytest
from django.conf import settings
from django.core.files.base import ContentFile
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

from apps.accounts.models import Course, User
from apps.ai_providers import services as ai_providers
from apps.conversations.retrieval import _clean_heading
from apps.documents import chunking, disciplinas, extraction, services
from apps.documents.models import Disciplina, Document

SEED = Path(__file__).resolve().parents[1] / "seed_materials" / "plano-de-ensino-cc.md"
TITULO_SEED = "Planos de Ensino de Ciência da Computação"

# Texto REAL do seed (grudado, sem espaços entre os campos).
CABECALHO_GRUDADO = (
    "Plano de Ensino - 2025/ 1º SEMESTRE\n"
    "Curso: CIÊNCIA DA COMPUTAÇÃODisciplina: MODELAGEM DE DADOS\n"
    "6º SEMESTREGraduaçãoC/H Semestral: 80"
)


# --- parsing ----------------------------------------------------------------------


def test_parse_identification_do_texto_grudado_do_seed():
    ident = disciplinas.parse_identification(CABECALHO_GRUDADO)

    assert ident == disciplinas.Identificacao(
        nome="Modelagem de Dados",
        curso="CIÊNCIA DA COMPUTAÇÃO",
        semestre=6,
        carga_horaria=80,
        periodo_letivo="2025/1",
    )


def test_parse_identification_nao_confunde_semestre_do_periodo_com_o_do_curso():
    ident = disciplinas.parse_identification(
        "Plano de Ensino - 2023/ 2º SEMESTRE\nCurso: CIÊNCIA DA COMPUTAÇÃODisciplina: CÁLCULO NUMÉRICO\n"
        "2º SEMESTREGraduaçãoC/H Semestral: 80"
    )
    assert (ident.periodo_letivo, ident.semestre) == ("2023/2", 2)
    ident = disciplinas.parse_identification(CABECALHO_GRUDADO.replace("6º SEMESTRE", "4º SEMESTRE"))
    assert (ident.periodo_letivo, ident.semestre) == ("2025/1", 4)


@pytest.mark.parametrize(
    "texto",
    [
        "Curso: CIÊNCIA DA COMPUTAÇÃO\nDisciplina: BANCO DE DADOS\n5º SEMESTRE",  # linhas separadas
        "Curso: CIÊNCIA DA COMPUTAÇÃO Disciplina: BANCO DE DADOS 5º SEMESTRE Graduação C/H Semestral: 72",  # espaços
        "Curso: CIÊNCIA DA COMPUTAÇÃODisciplina: BANCO DE DADOS5º SEMESTREGraduaçãoC/H Semestral: 72",  # tudo colado
    ],
)
def test_parse_identification_robusta_a_formatos(texto):
    ident = disciplinas.parse_identification(texto)
    assert ident.nome == "Banco de Dados"
    assert ident.semestre == 5


def test_parse_identification_sem_disciplina_devolve_none():
    assert disciplinas.parse_identification("## EMENTA\nEstudo dos conceitos.") is None
    assert disciplinas.parse_identification("Disciplina:   \n") is None


@pytest.mark.parametrize(
    ("cru", "esperado"),
    [
        ("MODELAGEM DE DADOS", "Modelagem de Dados"),
        ("CÁLCULO DIFERENCIAL E INTEGRAL II", "Cálculo Diferencial e Integral II"),
        ("INTERAÇÃO HUMANO-COMPUTADOR", "Interação Humano-Computador"),
        ("INTERNET DAS COISAS: TECNOLOGIAS E APLICAÇÕES", "Internet das Coisas: Tecnologias e Aplicações"),
        ("PROGRAMAÇÃO ORIENTADA A OBJETOS", "Programação Orientada a Objetos"),
        ("Já Capitalizado de Propósito", "Já Capitalizado de Propósito"),
    ],
)
def test_format_nome(cru, esperado):
    assert disciplinas.format_nome(cru) == esperado


def test_label_para_o_embedding():
    ctx = disciplinas.DisciplineContext(nome="Modelagem de Dados", semestre=6, carga_horaria=80, periodo_letivo="2025/1")
    assert ctx.label == "Modelagem de Dados (6º semestre, C/H 80 h, plano 2025/1)"
    assert disciplinas.DisciplineContext(nome="Robótica").label == "Robótica"


# --- propagação no chunker --------------------------------------------------------

MD_DOIS_PLANOS = """
Plano de Ensino - 2025/ 1º SEMESTRE

Curso: CIÊNCIA DA COMPUTAÇÃODisciplina: MODELAGEM DE DADOS

6º SEMESTREGraduaçãoC/H Semestral: 80
## EMENTA
Conceitos de modelagem de dados e formas normais.
## BIBLIOGRAFIA BÁSICA
Elmasri e Navathe.

Plano de Ensino - 2023/ 1º SEMESTRE

Curso: CIÊNCIA DA COMPUTAÇÃODisciplina: INTERNET DAS COISAS: TECNOLOGIAS E
## APLICAÇÕES

2º SEMESTREGraduaçãoC/H Semestral: 72
## EMENTA
Sensores, atuadores e protocolos.
"""


@pytest.fixture(scope="module")
def chunks_dois_planos():
    document = DocumentConverter().convert(DocumentStream(name="p.md", stream=io.BytesIO(MD_DOIS_PLANOS.encode()))).document
    return chunking.chunk_docling_document(document)


def test_nome_da_disciplina_vira_primeiro_segmento_do_heading(chunks_dois_planos):
    headings = [c.heading for c in chunks_dois_planos]
    assert "Modelagem de Dados > EMENTA" in headings
    assert "Modelagem de Dados > BIBLIOGRAFIA BÁSICA" in headings
    # Nome longo quebrado em dois itens (a 2ª metade veio como "cabeçalho") é reunido.
    assert "Internet das Coisas: Tecnologias e Aplicações > EMENTA" in headings
    assert not any(h == "APLICAÇÕES" or "> APLICAÇÕES" in h for h in headings)


def test_contexto_nao_atravessa_a_fronteira_entre_planos(chunks_dois_planos):
    # O fim do plano 1 (bibliografia) NÃO se mistura com o cabeçalho do plano 2.
    bibliografia = next(c for c in chunks_dois_planos if c.heading.endswith("BIBLIOGRAFIA BÁSICA"))
    assert bibliografia.content == "Elmasri e Navathe."
    assert bibliografia.discipline.nome == "Modelagem de Dados"
    ementa_iot = next(c for c in chunks_dois_planos if "Sensores" in c.content)
    assert ementa_iot.discipline.nome == "Internet das Coisas: Tecnologias e Aplicações"
    assert (ementa_iot.discipline.semestre, ementa_iot.discipline.carga_horaria) == (2, 72)
    assert ementa_iot.discipline.periodo_letivo == "2023/1"


def test_chunk_de_identificacao_vira_texto_legivel(chunks_dois_planos):
    ident = chunks_dois_planos[0]
    assert ident.heading == "Modelagem de Dados"
    assert ident.content == (
        "Disciplina: Modelagem de Dados\nCurso: Ciência da Computação\nSemestre do curso: 6º\n"
        "Carga horária semestral: 80 h\nPeríodo letivo do plano: 2025/1"
    )


def test_txt_propaga_o_contexto_pelo_conteudo():
    texto = f"{CABECALHO_GRUDADO}\n\nEmenta: conceitos de modelagem.\n\nMais um parágrafo."
    result = chunking.chunk_text(texto, max_chars=130, overlap=0)
    assert result[0].discipline.nome == "Modelagem de Dados"
    assert all(c.discipline is result[0].discipline for c in result)
    assert all(c.heading.startswith("Modelagem de Dados") for c in result)


def test_documento_sem_plano_nao_ganha_contexto():
    document = DocumentConverter().convert(DocumentStream(name="m.md", stream=io.BytesIO(b"## Regras\nTexto simples."))).document
    (chunk,) = chunking.chunk_docling_document(document)
    assert chunk.discipline is None and chunk.heading == "Regras"


# --- seed real --------------------------------------------------------------------


@pytest.fixture(scope="module")
def seed_chunks():
    document = extraction.convert_document(SEED.read_bytes(), SEED.name)
    return chunking.chunk_docling_document(document)


def test_seed_nenhum_chunk_com_heading_ou_texto_de_rodape(seed_chunks):
    assert seed_chunks
    for c in seed_chunks:
        assert "Credenciamento" not in c.heading and "/ 102" not in c.heading
        assert "www.cesuca.edu.br" not in c.content and "Portaria Ministerial" not in c.content
        assert "Silvério Manoel da Silva" not in c.content and "94940" not in c.content


def test_seed_ementas_levam_o_nome_da_disciplina_no_heading(seed_chunks):
    ementas = [c for c in seed_chunks if c.heading.endswith("> EMENTA")]
    assert len(ementas) == 36
    modelagem = next(c for c in ementas if c.heading == "Modelagem de Dados > EMENTA")
    assert "modelos conceitual, lógico e físico" in modelagem.content
    avaliacoes = {c.heading for c in seed_chunks if c.heading.endswith("> AVALIAÇÃO")}
    assert "Banco de Dados > AVALIAÇÃO" in avaliacoes


@pytest.fixture(scope="module")
def seed_rows(seed_chunks):
    class FakeCourse:
        id, name = 1, "Ciência da Computação"

    return disciplinas.build_disciplinas(seed_chunks, [FakeCourse()])


def test_seed_extrai_as_36_disciplinas_com_semestres_plausiveis(seed_rows):
    assert len(seed_rows) == 36
    assert all(1 <= r["semestre"] <= 8 and 30 <= r["carga_horaria"] <= 120 for r in seed_rows)
    assert len({r["nome"] for r in seed_rows}) == 36
    por_nome = {r["nome"]: r for r in seed_rows}
    assert (por_nome["Modelagem de Dados"]["semestre"], por_nome["Modelagem de Dados"]["carga_horaria"]) == (6, 80)
    assert por_nome["Modelagem de Dados"]["periodo_letivo"] == "2025/1"
    assert "Internet das Coisas: Tecnologias e Aplicações" in por_nome  # nome quebrado em 2 linhas
    assert "Projeto e Desenvolvimento de Sistemas Móveis" in por_nome


def test_seed_nomes_de_disciplina_nao_sao_limpos_como_titulo_do_documento(seed_rows):
    for row in seed_rows:
        assert _clean_heading(f"{row['nome']} > EMENTA", TITULO_SEED) == f"{row['nome']} > EMENTA"


# --- dedupe e banco ---------------------------------------------------------------


def _ctx(nome, periodo, semestre, curso="CIÊNCIA DA COMPUTAÇÃO"):
    return disciplinas.DisciplineContext(
        nome=nome, curso=curso, semestre=semestre, carga_horaria=72, periodo_letivo=periodo
    )


def test_build_disciplinas_mantem_o_periodo_mais_recente_por_curso_e_nome():
    antigo, novo = _ctx("Banco de Dados", "2023/2", 4), _ctx("Banco de Dados", "2025/1", 5)
    outro_curso = _ctx("Banco de Dados", "2024/1", 3, curso="ANÁLISE E DESENVOLVIMENTO DE SISTEMAS")

    class C:
        def __init__(self, id, name):
            self.id, self.name = id, name

    courses = [C(1, "Ciência da Computação"), C(2, "Análise e Desenvolvimento de Sistemas")]
    chunks = [
        chunking.Chunk("a", "x", discipline=antigo),
        chunking.Chunk("b", "x", discipline=antigo),
        chunking.Chunk("c", "x", discipline=novo),
        chunking.Chunk("d", "x", discipline=outro_curso),
        chunking.Chunk("e", "sem plano"),
    ]

    rows = disciplinas.build_disciplinas(chunks, courses)

    assert len(rows) == 2
    cc = next(r for r in rows if r["course"].id == 1)
    ads = next(r for r in rows if r["course"].id == 2)
    assert (cc["periodo_letivo"], cc["semestre"], cc["index_inicio"]) == ("2025/1", 5, 2)
    assert (ads["periodo_letivo"], ads["index_inicio"]) == ("2024/1", 3)


def test_resolve_course_cai_no_unico_curso_do_documento():
    class C:
        def __init__(self, id, name):
            self.id, self.name = id, name

    unico = C(1, "Ciência da Computação")
    assert disciplinas.resolve_course("", [unico]) is unico
    assert disciplinas.resolve_course("", [unico, C(2, "Outro")]) is None


@pytest.fixture
def admin(db):
    return User.objects.create_user(email="a@example.com", password="x", full_name="A", role=User.Role.CS_ADMIN)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * (settings.EMBEDDING_DIMENSIONS - 1) for _ in texts]


def test_ingestao_do_seed_popula_chunks_e_tabela_disciplina(admin, monkeypatch, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL = "gemini", "gemini-embedding-001"
    monkeypatch.setattr(ai_providers, "get_embedding_model", lambda: FakeEmbeddings())
    document = Document.objects.create(
        title=TITULO_SEED, file=ContentFile(SEED.read_bytes(), name=SEED.name), uploaded_by=admin
    )
    document.courses.set(Course.objects.filter(code="cc"))

    services._process_document(document)

    document.refresh_from_db()
    assert document.status == Document.Status.READY
    assert not document.chunks.filter(heading__icontains="Credenciamento").exists()
    assert document.chunks.filter(heading="Modelagem de Dados > EMENTA").count() == 1
    rows = Disciplina.objects.filter(document=document)
    assert rows.count() == 36
    modelagem = rows.get(nome="Modelagem de Dados")
    assert (modelagem.semestre, modelagem.carga_horaria, modelagem.periodo_letivo) == (6, 80, "2025/1")
    assert modelagem.course.code == "cc"
    inicio = document.chunks.get(index=modelagem.index_inicio)
    assert inicio.heading == "Modelagem de Dados" and "Semestre do curso: 6º" in inicio.content

    # Reprocessar refaz a tabela (sem duplicar).
    document.chunks.all().delete()
    services._process_document(document)
    assert Disciplina.objects.filter(document=document).count() == 36
