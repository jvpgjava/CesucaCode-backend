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

### 7. Subir o servidor

```bash
python manage.py runserver
```

A API sobe em `http://127.0.0.1:8000/`. O painel administrativo fica em
`http://127.0.0.1:8000/admin/`, e a documentação Swagger em
`http://127.0.0.1:8000/api/docs/` (veja [Documentação da API](#documentação-da-api-swagger)).

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
  documents/             # upload, extração de texto e chunking de material didático
  ai_providers/          # (em construção) abstração multi-provider de LLM/embeddings
  conversations/         # (em construção) chat com RAG
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
| POST | `/api/auth/change-password/` | Troca a própria senha | Qualquer autenticado |
| GET | `/api/auth/courses/` | Lista os cursos existentes | Qualquer autenticado (com senha em dia) |
| POST | `/api/auth/accounts/students/` | Cria um estudante | CSAdmin |
| POST | `/api/auth/accounts/students/import/` | Importa estudantes em massa via CSV | CSAdmin |
| POST | `/api/auth/accounts/coordinators/` | Cria um coordenador | CSAdmin |
| POST | `/api/auth/accounts/{id}/reset-password/` | Gera uma nova senha aleatória e envia por e-mail | CSAdmin |

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

Upload de material didático (PDF, DOCX, PPTX ou TXT) com extração de texto e
divisão em pedaços (chunks) — a base para a busca vetorial que entra na
Parte 3. Processamento é **síncrono**: a resposta do upload já vem com o
resultado final (`ready` ou `failed`).

**Quem pode o quê:**

| Papel | Upload / editar | Ver |
|---|---|---|
| **CSAdmin** | Qualquer curso | Todos os materiais |
| **CSCoordinator** | Só dos cursos que coordena | Só dos cursos que coordena |
| **CSStudent** | Não pode | Só do próprio curso |

Todas as rotas ficam sob `/api/documents/`:

| Método | Rota | Descrição | Quem pode |
|--------|------|-----------|-----------|
| GET | `/api/documents/` | Lista materiais (escopo por papel/curso) | Qualquer autenticado |
| POST | `/api/documents/upload/` | Envia um arquivo, extrai texto e divide em chunks | CSAdmin / CSCoordinator (do curso) |
| GET | `/api/documents/{id}/` | Detalhe de um material | CSAdmin / CSCoordinator (do curso) |
| DELETE | `/api/documents/{id}/` | Remove um material | CSAdmin / CSCoordinator (do curso) |
| GET | `/api/documents/{id}/chunks/` | Lista os pedaços de texto extraídos | CSAdmin / CSCoordinator (do curso) |
| POST | `/api/documents/{id}/reprocess/` | Apaga os chunks e refaz a extração/divisão | CSAdmin / CSCoordinator (do curso) |

Exemplo — CSAdmin envia um PDF:

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload/ \
  -H "Authorization: Bearer <token>" \
  -F "title=Introdução a Algoritmos" -F "course=cc" -F "file=@aula1.pdf"
```

Resposta:

```json
{"id":1,"title":"Introdução a Algoritmos","course":"cc","file":"http://127.0.0.1:8000/media/documents/cc/....pdf","status":"ready","processing_error":""}
```

Se a extração falhar (ex.: arquivo corrompido, PDF sem texto — como um
scaneado sem OCR), `status` vem `"failed"` e `processing_error` traz o
motivo; o material fica salvo mesmo assim e pode ser corrigido/reenviado
via `reprocess/`.

Limites do upload: até 20MB, formatos `pdf`/`docx`/`pptx`/`txt` (validado
antes mesmo de tentar processar). Um `CSCoordinator` só consegue enviar
material para cursos que ele coordena — tentar para outro curso retorna
`400`.

> **Nota:** não existe fila assíncrona (Celery/RQ) — o processamento
> acontece na mesma requisição do upload. Para arquivos muito grandes isso
> pode demorar; se isso virar um problema real, mover para processamento
> em background é o próximo passo natural, mas não foi necessário até
> agora.

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