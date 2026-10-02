# CesucaCode — Backend

Backend da IA acadêmica do Centro Universitário Cesuca para os cursos de
Ciência da Computação e Análise e Desenvolvimento de Sistemas: um assistente
de estudo/tutoria construído com RAG (Retrieval-Augmented Generation) sobre
conteúdos dos cursos.

Stack: **Python + Django + Django REST Framework + PostgreSQL (pgvector) +
LangChain**. 

---

## Como rodar o projeto (primeira vez)

Funciona igual em Windows, Mac ou Linux — todos os comandos abaixo são
`python`/`pip` puros, nenhum comando específico de sistema operacional.

**Pré-requisitos:** Python 3.11 ou 3.12 instalado, e PostgreSQL 14+ rodando
localmente (com a extensão [`pgvector`](https://github.com/pgvector/pgvector)
disponível para instalar — no Windows, o instalador oficial via
[EDB](https://www.postgresql.org/download/windows/) já traz isso pelo Stack
Builder; em outras plataformas, veja as instruções do próprio projeto
pgvector). Detalhes de versão em [Pré-requisitos](#pré-requisitos) abaixo.

### 1. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
```

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements/dev.txt
```

### 3. Configurar as variáveis de ambiente

```bash
python -c "import shutil; shutil.copy('.env.example', '.env')"
```

Gere uma `DJANGO_SECRET_KEY` real e cole no `.env` (no lugar de
`troque-esta-chave-por-uma-gerada-localmente`):

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

O `.env.example` já vem com uma `DATABASE_URL` padrão
(`postgres://cesucacode:cesucacode@localhost:5432/cesucacode`) — pode usar
como está, ou editar se preferir outro nome de usuário/senha/banco.

### 4. Criar o banco de dados

Este comando cria a role e o banco definidos na `DATABASE_URL` do `.env`
(se ainda não existirem) e já habilita a extensão `pgvector` — não precisa
abrir `psql`/pgAdmin na mão. Ele pede a senha de um superusuário do
Postgres (por padrão `postgres`) só para essa configuração inicial:

```bash
python manage.py bootstrap_db
```

Se o seu superusuário do Postgres não se chama `postgres`, use
`python manage.py bootstrap_db --superuser outro_nome`.

### 5. Rodar as migrações

```bash
python manage.py migrate
```

### 6. Criar o primeiro usuário CSAdmin

```bash
python manage.py createsuperuser
```

Pede e-mail, nome completo e senha, e sempre cria o usuário com papel
**CSAdmin** (veja [Papéis e contas](#papéis-e-contas) abaixo). O login é
por e-mail, não por "username". É o único jeito de criar um CSAdmin — não
existe endpoint de API para isso de propósito (evita escalonamento de
privilégio via API).

### 7. (Opcional) Carregar a base mínima de materiais

Para o chat já ter o que consultar numa instalação nova, carregue os materiais
de exemplo de `apps/documents/seed_materials/` (avaliação, manual das
disciplinas online, horário e planos de ensino de CC):

```bash
python manage.py seed_materials
```

Precisa do CSAdmin do passo 6 e do provider de embedding configurado no `.env`
(gera embeddings de verdade). É idempotente: pula o que já existe e reprocessa
o que tinha falhado. No plano gratuito do Gemini (100 embeddings/min) defina
`EMBEDDING_MAX_REQUESTS_PER_MINUTE=90` no `.env`: a ingestão passa a se espaçar
sozinha e o Plano de Ensino (~700 chunks) leva uns 8 minutos. Sem essa variável
(padrão, para produção) não há espera, e erro de cota do provider falha rápido. No Docker, `SEED_MATERIALS=1` no ambiente roda isso no startup (só faz
efeito se já existir um CSAdmin). A lista fica em `seed_materials/manifest.json`;
para incluir outro material, copie o arquivo para a pasta e adicione uma linha.
Cada `.md` tem seu PDF original em `seed_materials/originals/` (referência, não
ingerido; veja o `README.md` da pasta). Prefira `.md`/`.txt` a PDFs: processa em
segundos. PDF escaneado falha porque o OCR está desligado, então vira `.md`
antes (foi o caso do Código Disciplinar).

### 8. Subir o servidor

```bash
python manage.py runserver
```

A API sobe em `http://127.0.0.1:8000/`. O painel administrativo fica em
`http://127.0.0.1:8000/admin/`, e a documentação Swagger em
`http://127.0.0.1:8000/api/docs/` (veja [Documentação da API](#documentação-da-api-swagger)).

Para subir API, frontend e Postgres (com pgvector) juntos via Docker, use o
`docker compose up --build` na raiz do [CesucaCode-hub](https://github.com/jvpgjava/CesucaCode-hub).
O `Dockerfile` deste repositório é o da API; o Compose do hub orquestra os três serviços.

**Produção:** o `Dockerfile` usa `runserver` (desenvolvimento). Em produção use
`gunicorn config.wsgi:application -c gunicorn.conf.py`: o arquivo configura
workers `gthread` (8 threads) e timeout de 120 s, porque cada resposta do chat é
um stream longo que travaria um worker `sync` inteiro.

---

## Rodando no dia a dia (depois do setup inicial)

Já fez o setup uma vez (seção "Como rodar o projeto (primeira vez)" acima)?
Pra rodar de novo, ative o venv e use `python` normal — é o mesmo comando
em Windows, Mac ou Linux:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

```bash
python manage.py runserver
```

`(.venv)` aparece no início da linha do terminal quando está ativado —
é o sinal de que o `python`/`pip` que você vai chamar são os de dentro do
venv, com as dependências do projeto já instaladas.

Se preferir não ativar (ex.: rodando um comando avulso, ou dentro de um
script), dá pra chamar o executável do venv direto, sem ativar nada antes:

```bash
.venv\Scripts\python.exe manage.py runserver    # Windows
.venv/bin/python manage.py runserver            # Mac/Linux
```

Dá exatamente no mesmo resultado — ativar é só conveniência pra não repetir
o caminho toda hora.

**Importante:** o `runserver` **não** cria nem atualiza tabela nenhuma no
banco sozinho — quem faz isso é o `migrate`. Se você (ou eu, numa próxima
parte) mudou algum `models.py` desde a última vez, rode antes:

```bash
python manage.py migrate
```

Resumindo o ciclo de trabalho:
1. `python manage.py migrate` — só quando um model mudou
2. `python manage.py runserver` — sempre, pra subir a API

---

## Arquitetura do projeto

```
config/                 # settings, urls raiz, wsgi/asgi
  settings/
    base.py             # configuração compartilhada (lida o .env)
    dev.py               # desenvolvimento (DEBUG=True)
    prod.py              # produção (DEBUG=False, HTTPS, etc.)
apps/
  core/                  # base compartilhada (ex.: TimeStampedModel)
  accounts/              # usuários (CSAdmin/CSCoordinator/CSStudent), cursos e JWT
  documents/             # upload, extração, chunking e embeddings (pgvector)
  ai_providers/          # factory multi-provider de LLM/embeddings (LangChain)
  conversations/         # chat com RAG (streaming via Server-Sent Events)
requirements/
  base.txt               # dependências de produção
  dev.txt                 # base + ferramentas de desenvolvimento
  prod.txt                # base + servidor de produção (gunicorn, whitenoise)
```

**Padrão adotado:** cada app de domínio segue `models.py` (dados) →
`serializers.py` (validação/formato de entrada e saída da API) →
`views.py` (orquestração HTTP, fina) → `urls.py`. Regras de negócio mais
complexas (ex.: pipeline de RAG) ficam em módulos de serviço dedicados
dentro do app, não dentro das views. Todo modelo de negócio herda de
`apps.core.models.TimeStampedModel` (`created_at`/`updated_at`
automáticos; o `id` é sequencial, gerado pelo Django).

A documentação Swagger/OpenAPI de cada app fica num `schema.py` próprio
(ex.: `apps/accounts/schema.py`), carregado automaticamente pelo
`AppConfig.ready()` do app — as `views.py` nunca importam nada de
documentação, ficam só com a lógica HTTP.

---

## Pré-requisitos

1. **Python 3.11 ou 3.12** (recomendado). Este projeto foi validado também em
   Python 3.14, mas por ser uma versão muito recente algumas dependências
   (principalmente as de IA/ML, adicionadas nas próximas partes) podem ainda
   não ter builds prontos para ela. Se você tiver apenas o 3.14 instalado e
   encontrar erro ao instalar algum pacote, instale o 3.11/3.12 à parte
   (https://www.python.org/downloads/) e crie o ambiente virtual com ele
   (`py -3.12 -m venv .venv`).
2. **PostgreSQL 14+** com a extensão [`pgvector`](https://github.com/pgvector/pgvector)
   disponível. No Windows, o instalador oficial do PostgreSQL (via
   [EDB](https://www.postgresql.org/download/windows/)) já traz o
   `pgvector` disponível para instalar via Stack Builder; em outras
   plataformas, siga as instruções do próprio projeto pgvector.
3. Git (para clonar/versionar o repositório).

---

## Papéis e contas

Não existe cadastro público — quem cria conta é sempre um **CSAdmin**
(RGM é emitido pela instituição, não é algo que o próprio aluno escolhe).
Três papéis:

| Papel | Quem é | Identificador de login | Criado por |
|---|---|---|---|
| **CSAdmin** | Devs/TI — cria e gerencia contas, reseta senha | e-mail | `python manage.py createsuperuser` |
| **CSCoordinator** | Coordenador de um ou mais cursos | e-mail | CSAdmin |
| **CSStudent** | Estudante | **RGM** (e-mail também funciona) | CSAdmin (individual ou import CSV) |

O login é **um endpoint só** para os três papéis — `identifier` aceita
e-mail ou RGM, detectado automaticamente pela presença de `@`.

**Senha inicial:** toda conta criada por um CSAdmin (individual, import CSV,
ou reset de senha) recebe uma **senha aleatória gerada na hora** — nunca um
valor fixo repetido entre contas — enviada **só por e-mail**. Ela nunca
aparece em nenhuma resposta da API. A conta fica marcada com
`must_change_password: true`, e enquanto isso o usuário só consegue acessar
`/api/auth/me/` e `/api/auth/change-password/` — qualquer outro endpoint da
API retorna `403` até a senha ser trocada.

Em desenvolvimento local, os e-mails não são enviados de verdade — aparecem
no terminal onde o `runserver` está rodando (backend de e-mail "console",
padrão em `config.settings.dev`). Veja [Configurar envio de e-mail](#configurar-envio-de-e-mail)
para configurar SMTP de verdade.

As respostas de "criar estudante"/"criar coordenador" trazem um campo
`email_sent` (`true`/`false`) — se vier `false`, o envio falhou (SMTP fora
do ar, por exemplo) mas a conta foi criada normalmente; use o endpoint de
reset de senha para gerar e tentar reenviar.

Todas as rotas ficam sob `/api/auth/`:

| Método | Rota | Descrição | Quem pode |
|--------|------|-----------|-----------|
| POST | `/api/auth/login/` | Login (`identifier` = e-mail ou RGM + `password`) | Público |
| POST | `/api/auth/login/refresh/` | Renova o `access` token | Público |
| GET | `/api/auth/me/` | Dados do usuário autenticado | Qualquer autenticado |
| PATCH | `/api/auth/me/` | Edita o próprio `nickname` (autoatendimento; demais campos são geridos pelo CSAdmin) | Qualquer autenticado |
| POST | `/api/auth/change-password/` | Troca a própria senha | Qualquer autenticado |
| GET | `/api/auth/courses/` | Lista os cursos existentes | Qualquer autenticado (com senha em dia) |
| GET | `/api/auth/accounts/` | Lista alunos e coordenadores (`?search=`, `?role=`, paginado) | CSAdmin |
| GET | `/api/auth/accounts/{id}/` | Ver uma conta | CSAdmin |
| PATCH | `/api/auth/accounts/{id}/` | Edita `nickname` e/ou `is_active` (desativar bloqueia login sem apagar a conta) | CSAdmin |
| POST | `/api/auth/accounts/students/` | Cria um estudante | CSAdmin |
| POST | `/api/auth/accounts/students/import/` | Importa estudantes em massa via CSV | CSAdmin |
| POST | `/api/auth/accounts/coordinators/` | Cria um coordenador | CSAdmin |
| POST | `/api/auth/accounts/{id}/reset-password/` | Gera uma nova senha aleatória e envia por e-mail | CSAdmin |

`/api/auth/accounts/` e `/api/auth/accounts/{id}/` nunca alcançam contas
CSAdmin (essas só existem via `createsuperuser`, propositalmente fora da
API) — tentar editar uma dá `404`, não `403`, pra não confirmar que o id
existe.

Exemplo — CSAdmin lista contas (busca opcional por nome/e-mail/RGM, filtro opcional por papel):

```bash
curl "http://127.0.0.1:8000/api/auth/accounts/?search=maria&role=cs_student" \
  -H "Authorization: Bearer <token_do_csadmin>"
```

Exemplo — CSAdmin cria um estudante:

```bash
curl -X POST http://127.0.0.1:8000/api/auth/accounts/students/ \
  -H "Authorization: Bearer <token_do_csadmin>" -H "Content-Type: application/json" \
  -d '{"email":"aluno@cesuca.edu.br","full_name":"Nome do Aluno","rgm":"20260001","course":"cc"}'
```

Resposta (sem senha nenhuma — só a confirmação de que o e-mail foi enviado):

```json
{"id":7,"email":"aluno@cesuca.edu.br","full_name":"Nome do Aluno","nickname":"","rgm":"20260001","course":"cc","email_sent":true}
```

Exemplo — import em massa (CSV com colunas `full_name,rgm,email,course` e,
opcionalmente, `nickname`; `course` usa o código do curso, ex.: `cc`/`ads`):

```bash
curl -X POST http://127.0.0.1:8000/api/auth/accounts/students/import/ \
  -H "Authorization: Bearer <token_do_csadmin>" \
  -F "file=@alunos.csv"
```

A resposta traz quantas linhas foram criadas (`created_count`), quantas
falharam validação com o detalhe de cada uma (`errors`) e quantas falharam
só no envio do e-mail (`email_failures_count`) — não interrompe no primeiro
erro. Status `201` se tudo deu certo, `207` se parte falhou.

Exemplo — login de estudante (por RGM):

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"identifier":"20260001","password":"<senha>"}'
```

Exemplo — login de CSAdmin/CSCoordinator (por e-mail, mesma rota):

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"identifier":"coord.cc@cesuca.edu.br","password":"<senha>"}'
```

> **Nota de segurança:** o login avisa especificamente quando o e-mail/RGM
> não existe (dizendo qual dos dois procurou), e de forma genérica quando a
> senha está errada (identificador existe). Como RGM não é informação
> secreta (fica na carteirinha/crachá), isso é aceitável aqui, mas em teoria
> permite alguém testar quais e-mails/RGMs existem. Por isso o endpoint de
> login tem limite de 10 tentativas por minuto (`DEFAULT_THROTTLE_RATES` em
> `config/settings/base.py`).

---

## Configurar envio de e-mail

Usado para mandar a senha inicial/gerada de contas criadas pelo CSAdmin
(veja [Papéis e contas](#papéis-e-contas) acima).

**Desenvolvimento local:** não precisa configurar nada — o `.env.example`
já não define `EMAIL_BACKEND`, então o padrão (definido em
`config/settings/base.py`) entra em ação: o backend "console", que imprime
o e-mail inteiro no terminal onde o `runserver` está rodando em vez de
enviar de verdade. Ótimo para testar sem precisar de credenciais de SMTP.

**Produção:** `config/settings/prod.py` troca automaticamente para SMTP de
verdade e **exige** `EMAIL_HOST_USER`/`EMAIL_HOST_PASSWORD` no `.env` (o
servidor recusa subir sem isso). Configure no `.env`:

```bash
EMAIL_HOST=smtp.exemplo.com
EMAIL_PORT=587
EMAIL_HOST_USER=algum-usuario
EMAIL_HOST_PASSWORD=algum-segredo
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=CesucaCode <naoresponda@cesuca.edu.br>
```

Qualquer provedor SMTP funciona (só trocar os valores acima, nenhum código
muda) — ex.: SMTP institucional do Cesuca (Outlook/Exchange, geralmente
`smtp.office365.com`) ou um provedor de e-mail transacional. Se usar Gmail
para testes, lembre que o remetente (`DEFAULT_FROM_EMAIL`) precisa ser um
endereço `@gmail.com` de verdade — colocar um remetente `@cesuca.edu.br`
saindo por servidor do Gmail tende a cair em spam (falha de SPF/DKIM).

---

## Materiais didáticos (Documents)

Upload de material didático (PDF, DOCX, PPTX, MD ou TXT) com extração de texto,
divisão em pedaços (chunks) e geração de embedding pra cada chunk (ver seção
"Provedores de IA" abaixo) — a base da busca vetorial usada no chat com IA.

Para PDF/DOCX/PPTX/MD, a extração usa o [Docling](https://github.com/docling-project/docling)
em vez de leitura de texto ingênua: ele entende layout (cabeçalhos, seções,
tabelas, colunas). A divisão em chunks é feita de forma hierárquica a partir
dessa estrutura — cada chunk carrega o caminho de seções a que pertence
(campo `heading`, ex.: `"5. Modelo ER > 5.1 Entidades"`), que também é usado
como contexto extra na hora de gerar o embedding. **Markdown (`.md`)** passa
pelo Docling também: os títulos (`#`, `##`...) viram o `heading` de cada chunk,
como nos PDFs — por isso é melhor enviar `.md` do que renomear pra `.txt`. TXT
não tem estrutura pra aproveitar, então segue com divisão simples por parágrafo.

**OCR fica desligado por padrão** (`do_ocr=False`) — os materiais didáticos
são PDFs gerados digitalmente (têm texto real embutido), não escaneados, e
OCR é a parte mais cara do processamento (~150-240s → ~40-60s por PDF real
sem ele). Se algum material for realmente uma imagem escaneada sem texto, a
extração falha com uma mensagem clara (`processing_error`) em vez de
demorar minutos à toa; ligar OCR de volta é uma linha em
`apps/documents/extraction.py` (`PdfPipelineOptions(do_ocr=True)`), se algum
dia isso virar uma necessidade real. Tabelas usam o modo `FAST` do
TableFormer (em vez de `ACCURATE`) pelo mesmo motivo de custo.

Processamento é **assíncrono, em background**: o upload responde na hora com
`status: "processing"` e a extração roda num pool de threads do próprio
processo Django (`ThreadPoolExecutor`, 2 workers) — sem depender de um
broker externo (Celery/Redis), o que não se justifica pro volume de uso
desse app. `GET /api/documents/{id}/` reflete o status real (`processing` →
`ready`/`failed`) conforme o processamento avança; o frontend faz polling
desse endpoint enquanto o status é `processing`.

Isso é uma fila em memória do processo, não durável: se o servidor cair ou
reiniciar (ex.: autoreload do `runserver`) no meio do processamento, aquele
job se perde e o documento fica preso em `processing` — nesse caso, usar
`reprocess/` resolve. Se o volume de uploads crescer a ponto disso incomodar
(ou for rodar com múltiplos processos/workers, onde um pool em memória por
processo processa menos em paralelo do que parece), o próximo passo natural
é migrar para uma fila de verdade (Celery + Redis).

**Um material pode valer para mais de um curso.** No upload e na edição
informe `courses` (lista de códigos, ex.: `cc`, `ads`; ao menos um). Um material
marcado para CC e ADS aparece para os alunos, coordenadores e busca do chat
dos dois cursos — não precisa enviar duas vezes.

**Quem pode o quê:**

| Papel | Upload / editar | Excluir | Ver |
|---|---|---|---|
| **CSAdmin** | Qualquer curso | Qualquer material | Todos os materiais |
| **CSCoordinator** | Só para cursos que coordena; edita/reprocessa se coordenar *algum* curso do material, sem conseguir remover os cursos que não coordena | Só se coordenar **todos** os cursos do material (num material compartilhado com curso que não coordena, ele vê e edita, mas não exclui — `403`) | Materiais que incluam algum curso que coordena |
| **CSStudent** | Não pode | Não pode | Materiais que incluam o próprio curso |

A resposta de cada material traz `courses` (lista de cursos) e `can_delete`
(se o usuário logado pode excluí-lo).

Todas as rotas ficam sob `/api/documents/`:

| Método | Rota | Descrição | Quem pode |
|--------|------|-----------|-----------|
| GET | `/api/documents/` | Lista materiais (escopo por papel/cursos, paginado) | Qualquer autenticado |
| POST | `/api/documents/upload/` | Envia um arquivo para um ou mais cursos, extrai texto e divide em chunks | CSAdmin / CSCoordinator (dos cursos que coordena) |
| GET | `/api/documents/{id}/` | Detalhe de um material | CSAdmin / CSCoordinator (do curso) |
| PATCH | `/api/documents/{id}/` | Edita título/cursos (não reenvia o arquivo nem reprocessa) | CSAdmin / CSCoordinator (de algum curso do material) |
| DELETE | `/api/documents/{id}/` | Remove um material | CSAdmin / CSCoordinator (de todos os cursos do material) |
| GET | `/api/documents/{id}/chunks/` | Lista os pedaços de texto extraídos (cada um com `heading`) | CSAdmin / CSCoordinator (do curso) |
| POST | `/api/documents/{id}/reprocess/` | Apaga os chunks e refaz a extração/divisão | CSAdmin / CSCoordinator (do curso) |

Exemplo — CSAdmin envia um PDF válido para CC e ADS (repita o campo `courses`):

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload/ \
  -H "Authorization: Bearer <token>" \
  -F "title=Introdução a Algoritmos" -F "courses=cc" -F "courses=ads" -F "file=@aula1.pdf"
```

Resposta (imediata — processamento continua em background):

```json
{"id":1,"title":"Introdução a Algoritmos","courses":["cc","ads"],"file":"http://127.0.0.1:8000/media/documents/....pdf","status":"processing","processing_error":""}
```

Se a extração falhar (ex.: arquivo corrompido ou realmente sem conteúdo
extraível), `status` vem `"failed"` e `processing_error` traz o motivo; o
material fica salvo mesmo assim e pode ser corrigido/reenviado via
`reprocess/`.

Limites do upload: até 20MB, formatos `pdf`/`docx`/`pptx`/`txt` (validado
antes mesmo de tentar processar). Um `CSCoordinator` só consegue enviar
material para cursos que ele coordena — tentar para outro curso retorna
`400`.

---

## Provedores de IA (LLM e Embeddings)

A app `apps/ai_providers` esconde qual provider de IA está sendo usado atrás
de duas funções (`get_chat_model()` e `get_embedding_model()`) — o resto do
sistema nunca importa uma classe de provider específica, só chama essas
funções. A escolha de provider é feita por variável de ambiente, e o chat e
o embedding são **independentes**: dá pra usar um provider pra conversa e
outro pra gerar os vetores dos documentos.

**Chat** (`LLM_PROVIDER`): `gemini`, `openai`, `claude`, `ollama`,
`deepseek`, `abacusai` ou `openrouter`. `deepseek`, `abacusai` e `openrouter`
usam a API compatível com OpenAI de cada um (com `base_url` próprio), então
reaproveitam o mesmo pacote (`langchain-openai`). No OpenRouter o modelo vai
no formato `provedor/modelo` (ex.: `google/gemini-2.5-flash`) e a chave em
`OPENROUTER_API_KEY`.

**Papéis de modelo:** `get_chat_model(role)` aceita `answer` (resposta final,
padrão), `router` (classificador de intenção), `agent` (passos do loop
agêntico) e `judge` (avaliação). Cada papel pode ter seu próprio provider e
modelo, o que permite, por exemplo, um modelo barato só para o roteador:

```env
LLM_ROUTER_PROVIDER=openrouter
LLM_ROUTER_MODEL=google/gemini-2.5-flash-lite
# também: LLM_<PAPEL>_MAX_TOKENS e LLM_<PAPEL>_TEMPERATURE
```

O que não for definido cai em `LLM_PROVIDER`/`LLM_MODEL`. Padrões de
`max_tokens`/temperatura: answer 1500/0.3, router 256/0, agent 1024/0.2,
judge 512/0. Globais: `LLM_TIMEOUT` (60 s), `LLM_MAX_RETRIES` (2),
`LLM_STRUCTURED_METHOD` (força `json_schema`, `function_calling` ou
`json_mode`; vazio usa o registro de capacidades em
`apps/ai_providers/capabilities.py`) e `LLM_THINKING_BUDGET` (Gemini; vazio =
padrão do modelo). Atenção: no Gemini com thinking, `max_tokens` também conta
os tokens de raciocínio; se as respostas vierem cortadas, aumente o limite ou
reduza o `LLM_THINKING_BUDGET`.

`invoke_structured(Schema, messages, role)` pede saída estruturada com 1 retry
de reparo e nunca levanta exceção de provider (devolve `ok=False` e o valor
padrão).

**Embedding** (`EMBEDDING_PROVIDER`): `gemini` ou `ollama` — os dois
providers testados que têm uma API de embedding simples (texto entra, vetor
sai) compatível com o LangChain. A Abacus AI, por exemplo, só tem API de
chat (RouteLLM); não tem uma API de embedding equivalente, por isso não
está na lista de embedding.

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768

GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
ABACUSAI_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
```

Preencha só a chave do provider que for usar de fato.

> **Atenção com `EMBEDDING_DIMENSIONS`:** o vetor de cada chunk é salvo numa
> coluna `pgvector` de tamanho **fixo**, definido nessa variável e travado
> na migration (`apps/documents/migrations/0002_documentchunk_embedding.py`).
> Trocar de provider/modelo de embedding depois de já ter documentos
> processados exige gerar uma nova migration e reprocessar todos os
> materiais (`POST /api/documents/{id}/reprocess/`) — os embeddings antigos
> não são compatíveis com uma dimensão diferente.

### Alternativa local: EmbeddingGemma via Ollama

Além do Gemini (padrão), o embedding pode rodar local com o
[EmbeddingGemma](https://ollama.com/library/embeddinggemma) pelo Ollama: sem
chave, sem cota e sem GPU (modelo pequeno, roda em CPU). Gera vetores de
**768 dimensões**, a mesma do Gemini, então não exige migration.

```bash
ollama pull embeddinggemma
```

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_DIMENSIONS=768
OLLAMA_BASE_URL=http://localhost:11434
```

Ao trocar de modelo com materiais já processados:

1. `python manage.py test_ai_provider` — confirma o provider e a dimensão (se o
   modelo devolver outro tamanho, o erro agora diz isso claramente).
2. `python manage.py reprocess_documents` — refaz chunks e embeddings de todos
   os materiais. Vetores de modelos diferentes não são comparáveis, então
   misturar dá buscas sem sentido.
3. Reavalie `RAG_MAX_DISTANCE` (0.30 foi calibrado com o Gemini) e rode
   `python manage.py test_guardrails`.

O Gemini continua sendo o padrão e o que o time usa; o EmbeddingGemma é uma
opção em avaliação (ainda não validada como padrão de produção).

Pra testar se as chaves configuradas no `.env` estão funcionando, sem
precisar subir o servidor nem fazer upload de nada:

```bash
python manage.py test_ai_provider
```

Isso manda uma mensagem simples pro chat e gera um embedding de teste,
mostrando se cada provider respondeu certo ou qual foi o erro.

Para validar o contrato de um papel de modelo (rode sempre que trocar de
provider ou modelo; consome tokens reais):

```bash
python manage.py test_ai_provider --role router --check stream,tools,structured
```

`stream` confere o streaming e o `usage_metadata`, `tools` faz um tool calling
simples e `structured` testa a saída estruturada com o método do registro.

---

## Conversas (Chat com RAG)

Chat em tempo real com os materiais didáticos como contexto. Cada conversa é
só do usuário que criou — CSAdmin/CSCoordinator não veem conversas de
outras pessoas, e vice-versa.

**Como funciona:** ao enviar uma mensagem, o backend gera o embedding da
pergunta, busca os pedaços de material mais parecidos (limitados aos
materiais que aquele usuário tem permissão de ver — mesmo escopo por
curso/papel dos materiais didáticos), monta o prompt com esse contexto
mais o histórico da conversa, e manda pro provider de chat configurado
(`LLM_PROVIDER`). Se não achar nenhum material relevante, o modelo é
instruído a dizer isso em vez de inventar uma resposta.

**System prompt e guardrails:** ficam em arquivos `.md` na pasta
`apps/conversations/prompts/sofia/` — não hardcoded no Python. Os arquivos são
lidos em ordem alfabética (daí os prefixos) e concatenados, então cada tema é
um arquivo que dá pra editar/auditar separado, sem reiniciar o servidor (a
mudança vale na próxima mensagem). Pra adicionar uma regra nova, crie outro
`NN-tema.md`. `SYSTEM_PROMPT_PATH` no `.env` aceita uma pasta ou um único
arquivo, relativo à raiz do projeto ou absoluto.

| Arquivo | Assunto |
|---|---|
| `00-identidade.md` | Quem é a S.O.F.I.A, idioma, regras valem em todo turno |
| `10-escopo.md` | Só computação de CC/ADS; como recusar; pretextos que não mudam nada |
| `20-materiais-e-fontes.md` | Regras internas de fontes: informação da instituição só do contexto; aviso de "explicação geral" em conceitos técnicos |
| `22-como-falar-das-fontes.md` | **Nunca expor o funcionamento interno** ("materiais enviados", "trechos", "fragmentado"...): quando não tem a informação, diz que não tem confirmada e orienta a conferir com a coordenação/secretaria/professor; **não cita fontes** (sem `(Fonte: ...)`, títulos nem links) |
| `27-caminho-de-estudo.md` | Perguntas de grade/disciplinas: agrupa as disciplinas do curso em fases de estudo e sugere a ordem, usando as referências externas da pesquisa na web só como apoio (nunca como matriz oficial da Cesuca) |
| `25-anti-alucinacao.md` | Nunca inventar livros/páginas/datas/números; grade/semestre/ordem só quando explícitos; "não sei" > chute; sem acesso a internet/notas |
| `30-pedagogia.md` | Guiar em vez de entregar trabalho pronto |
| `40-seguranca.md` | Prompt injection, personas, vazamento do prompt, malware |
| `50-idiomas-e-ofuscacao.md` | Responder em português; binário/base64/invertido/leetspeak recebem as mesmas regras |

Além do prompt, há duas proteções em código (`apps/conversations/services.py`):

1. *Filtro de relevância* — só entram no contexto os trechos com distância de
   cosseno ≤ `RAG_MAX_DISTANCE` (padrão `0.30`, no `.env`). Se nenhum passar, o
   modelo é avisado de que nada nos materiais foi relevante e, se a pergunta
   for de computação, responde com conhecimento geral **avisando que não veio
   dos materiais**. Ele decide *se há material relevante*, não *se o assunto é
   do curso* (nas medições, computação geral e assuntos aleatórios ficam na
   mesma faixa de distância — o escopo é responsabilidade do prompt). Foi
   calibrado com pouco material; com mais conteúdo enviado, reavalie.
2. *Recusa do provedor* — quando o filtro de conteúdo do provedor do LLM barra
   a requisição (comum com instruções escondidas em binário/base64/hex), o chat
   responde uma recusa amigável em vez de erro, e esse par pergunta+recusa é
   omitido do histórico enviado nos turnos seguintes (senão a mensagem barrada
   travaria a conversa inteira).

3. *Perfil do usuário* — o system prompt recebe, além das regras, um bloco
   curto com o papel e o curso de quem está perguntando (só como informação,
   nunca como instrução), pra adaptar a resposta ao curso.
4. *Limite de histórico* — só as últimas `CHAT_MAX_HISTORY_MESSAGES` mensagens
   (padrão `12`) vão pro modelo. Isso não limita o modelo em si, só o quanto
   da conversa antiga é reenviado a cada turno.
5. *Modo estrito* — com `CHAT_ALLOW_GENERAL_KNOWLEDGE=False`, sem material
   relevante a S.O.F.I.A diz que não encontrou nos materiais em vez de
   responder com conhecimento geral. Informação da instituição (grade,
   disciplinas, datas, créditos, contatos) **sempre** vem só dos materiais,
   independente dessa opção.

6. *Pesquisa na web para grade e disciplinas* — em perguntas de grade
   curricular, disciplinas, ordem para cursar ou "por onde seguir" (detectadas
   por `web_search.is_curriculum_question`), o backend pesquisa na web (biblioteca
   `ddgs`, DuckDuckGo, **sem chave**) e anexa os resultados como "Referências
   externas" pro modelo sugerir um caminho de estudo. Regras: a consulta leva só
   a pergunta e o nome do curso (nenhum dado pessoal); só título e resumo dos
   resultados são usados (nenhuma página é baixada); o texto é tratado como
   dado, não como ordem; serve para *como estudar* as disciplinas, nunca para
   afirmar o que a Cesuca oferece (semestre, carga, créditos, pré-requisitos
   oficiais), que continua vindo só dos materiais. Se a busca falha ou demora
   mais que `CHAT_WEB_SEARCH_TIMEOUT` (8s), o chat responde sem ela. Ajustes:
   `CHAT_WEB_SEARCH_ENABLED` (padrão `True`), `CHAT_WEB_SEARCH_MAX_RESULTS`
   (`5`) e `CHAT_WEB_SEARCH_TIMEOUT` (`8`). Ignorada no modo estrito. Adiciona
   alguns segundos à primeira resposta dessas perguntas.

7. *Referências opacas* — o contexto enviado ao modelo identifica cada trecho
   como `[T1 · seção: <título da seção>]` (ou só `[T1]`); o título do documento
   nunca vai ao prompt, e o prompt manda não citar essas referências.
8. *Guarda de saída* (`apps/conversations/guard.py`) — ao fim de cada resposta,
   confere vazamentos: vocabulário interno (`INTERNAL_TERMS`), trechos do
   prompt (`LEAK_FRAGMENTS`), títulos de documentos que o usuário enxerga
   (com ≥ 2 palavras e ≥ 8 caracteres, sem acento/extensão — títulos de uma
   palavra só geram falso positivo) e referências `[T#]`. As flags vão para o
   trace e as referências são removidas do texto **salvo**. Como a resposta é
   transmitida por streaming, a guarda não desfaz o que o aluno já viu.
9. *Rastreamento (`MessageTrace`)* — cada resposta grava, no admin (somente
   leitura), versão do pipeline, rota, modelos, tokens, latência, tempo até o
   primeiro token, etapas, ids/distâncias dos trechos e flags da guarda. Não guarda
   texto do aluno nem da resposta.
10. *Limite de uso* — por usuário, `CHAT_THROTTLE_RATE` (padrão `20/min`) e
    `CHAT_DAILY_THROTTLE_RATE` (padrão `300/day`) no envio de mensagens; passou
    do limite, responde `429` com mensagem em português e `Retry-After`. O
    contador usa o cache do Django (em memória por padrão: com vários
    processos, configure um cache compartilhado).

Não há limite imposto ao modelo (tokens, temperatura etc.) além do teto de
histórico, que é janela de contexto.

O prompt reduz bastante, mas não elimina, a chance de burlar as regras com
pedidos elaborados — reveja os arquivos conforme surgirem casos novos.

**Suíte de regressão dos guardrails:** os casos ficam em
`apps/conversations/guardrail_cases.py` (recusas, pretextos, injeção,
vazamento de prompt, ofuscação, alucinação, perguntas legítimas...) e são
executados de verdade contra o LLM configurado:

```bash
python manage.py test_guardrails                       # todos os casos
python manage.py test_guardrails --only recusa         # só um tipo (recusa, resposta, sem_info, cautela, aviso_geral)
python manage.py test_guardrails --name "receita"      # casos cujo nome contém o texto
python manage.py test_guardrails --user email@x.com    # roda como esse usuário (padrão: um aluno com curso; o curso define o que ele vê)
python manage.py test_guardrails --retries 2           # tenta de novo casos que falharem (o LLM varia)
```

Rode depois de mexer nos arquivos de prompt ou trocar de modelo. Os casos
"sem_info" (data da prova, créditos, contatos...) assumem que essa informação não
está nos materiais enviados; se estiver, ajuste-os. Os de grade e disciplinas usam
"cautela" (não afirmar com falsa certeza), porque dependem de como a grade foi
enviada.

**Sugestões e feedback:** a tela inicial do chat mostra sugestões prontas
(grade curricular, disciplinas e "com o que você pode me ajudar" — perguntas fixas; a
resposta vem sempre dos materiais enviados). Cada resposta pode ser
avaliada com 👍/👎, e o chat exibe um aviso de que a IA pode errar.

Todas as rotas ficam sob `/api/conversations/`:

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/api/conversations/` | Lista as minhas conversas |
| POST | `/api/conversations/` | Cria uma conversa vazia (o título se preenche sozinho na primeira mensagem) |
| GET | `/api/conversations/{id}/` | Ver uma conversa |
| PATCH | `/api/conversations/{id}/` | Renomear (`title`) — vazio é aceito, volta a exibir como "Nova conversa" |
| DELETE | `/api/conversations/{id}/` | Excluir uma conversa |
| GET | `/api/conversations/{id}/messages/` | Histórico completo de mensagens |
| POST | `/api/conversations/{id}/messages/send/` | Enviar uma mensagem — resposta em streaming |
| GET | `/api/conversations/suggestions/` | Sugestões de perguntas prontas, de acordo com os materiais que eu posso ver |
| PATCH | `/api/conversations/{id}/messages/{message_id}/feedback/` | Avaliar uma resposta: `{"feedback": 1}` (👍), `{"feedback": -1, "reason": "incorreta", "comment": "..."}` (👎 com motivo opcional) ou `{"feedback": null}` (remove). `reason`: `incorreta`, `incompleta`, `nao_entendeu`, `fora_do_curso`, `outro`; `comment` até 500 caracteres; ao voltar para 👍/`null` os dois são apagados. `rating` segue aceito como alias de `feedback` |

O envio de mensagem **não devolve um JSON único** — a resposta vem em
tempo real via [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events)
(`Content-Type: text/event-stream`), pedaço por pedaço, do mesmo jeito que
ChatGPT/Claude mostram a resposta "sendo digitada". Corpo da requisição:
`{"content": "...", "regenerate": false}`. Eventos:

```
event: meta
data: {"user_message_id": 123, "route": null}

event: status
data: {"step": "searching", "label": "Buscando nos materiais do curso"}

data: {"content": "pedaço de texto"}

event: suggestions
data: {"items": ["Pergunta 1?", "Pergunta 2?"]}

event: done
data: {"message_id": 456}
```

(ou `event: error` com `{"message": "texto amigável"}` se algo falhar). Os
tokens usam o formato padrão (sem `event:`), compatível com clientes que só leem
`data: {"content": ...}`. Os rótulos de `status` vêm de um dicionário fixo
(`apps/conversations/events.py`) e nunca carregam argumentos internos.
`regenerate: true` apaga a última resposta e a pergunta que a gerou e processa
`content` como nova. A mensagem do usuário é salva antes de gerar e a resposta do
assistente ao final (se o cliente desconectar, fica o texto parcial) — não
precisa (nem dá pra) mandar a resposta de volta pra API depois.

Exemplo com curl (`-N` desativa o buffer, pra ver o streaming chegando aos
poucos):

```bash
curl -N -X POST http://127.0.0.1:8000/api/conversations/1/messages/send/ \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"content": "O que esse material fala sobre recursão?"}'
```

---

## Documentação da API (Swagger)

Com o servidor rodando (`python manage.py runserver`), a documentação
interativa fica disponível em:

- **Swagger UI:** http://127.0.0.1:8000/api/docs/
- **ReDoc:** http://127.0.0.1:8000/api/redoc/
- **Schema OpenAPI (JSON/YAML puro):** http://127.0.0.1:8000/api/schema/

Essas rotas são públicas (não exigem token), mesmo o resto da API exigindo
autenticação por padrão.

Para documentar um endpoint novo, **não edite `views.py`** — adicione (ou
crie) um `schema.py` no app correspondente usando `extend_schema_view` do
`drf-spectacular` e garanta que ele seja importado no `ready()` do
`AppConfig` daquele app (veja `apps/accounts/schema.py` e
`apps/accounts/apps.py` como referência).

### Collection do Postman

Em `docs-collections/CesucaCode.postman_collection.json` tem uma collection
pronta com todos os endpoints (autenticação, contas, cursos, documentos),
já com variáveis que se preenchem sozinhas (token, id do aluno/documento
criado) — é só importar no Postman e rodar **Login (CSAdmin)** primeiro.
Os arquivos de exemplo usados pelos testes (`sample_students.csv`,
`sample_material.txt`) ficam na mesma pasta.

---

## Testando se está tudo certo

```bash
python manage.py check      # valida configuração do projeto
python manage.py migrate    # aplica as migrações
python manage.py runserver  # sobe a API e confirma que responde
```

Se `python manage.py check` reclamar de `DJANGO_SECRET_KEY` ou
`DATABASE_URL`, o `.env` não foi criado/configurado corretamente (veja
"Como rodar o projeto (primeira vez)" → passo 3).

---

## Solução de problemas comuns

- **Erro ao instalar pacote com extensão nativa (ex.: build C/C++ falhando):**
  geralmente é incompatibilidade com uma versão de Python muito nova. Use
  Python 3.11/3.12 no ambiente virtual (veja Pré-requisitos).
- **`connection to server ... failed: FATAL: password authentication failed`:**
  a `DATABASE_URL` do `.env` não bate com a role/senha que existem de fato
  no PostgreSQL — rode `python manage.py bootstrap_db` de novo (é
  idempotente, não faz mal repetir) ou revise a `DATABASE_URL` no `.env`.
- **`bootstrap_db` falha ao conectar como superusuário:** confirme que o
  PostgreSQL está rodando e que a senha informada é a do usuário
  administrador de verdade (por padrão `postgres`) — não a senha da
  `DATABASE_URL` do projeto, que é de outra role (`cesucacode`).
- **`relation "..." does not exist` ou erros de migração:** rode
  `python manage.py migrate` novamente; se mudou modelos, gere a migração
  primeiro com `python manage.py makemigrations`.
- **CORS bloqueando o frontend:** confirme que a URL do frontend (ex.:
  `http://localhost:5173` do Vite) está listada em `CORS_ALLOWED_ORIGINS`
  no `.env`.
- **Não sei qual é a senha de uma conta recém-criada:** ela não fica em
  lugar nenhum além do e-mail enviado — nem no banco (fica só o hash), nem
  na resposta da API, nem em log. Em desenvolvimento local, olhe o
  terminal onde o `runserver` está rodando (o e-mail aparece impresso
  ali). Se realmente perdeu, use o endpoint de reset de senha para gerar
  uma nova.
- **`email_sent: false` na resposta ou e-mail não chega:** confira a
  configuração de SMTP (veja [Configurar envio de e-mail](#configurar-envio-de-e-mail))
  — em desenvolvimento isso é esperado nunca "chegar" de verdade, já que o
  backend padrão só imprime no console.