from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("DJANGO_SECRET_KEY")
DEBUG = env.bool("DJANGO_DEBUG", default=False)
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=[])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "drf_spectacular",
    "corsheaders",
    "apps.core",
    "apps.accounts",
    "apps.documents",
    "apps.ai_providers",
    "apps.conversations",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": env.db("DATABASE_URL"),
}

AUTH_USER_MODEL = "accounts.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
        "apps.accounts.permissions.PasswordIsCurrent",
    ),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_RATES": {
        "login": "10/min",
        # Envio de mensagens do chat, por usuário: limite por minuto (contra
        # rajadas/scripts) e por dia (contra custo descontrolado de LLM).
        "chat": env("CHAT_THROTTLE_RATE", default="20/min"),
        "chat_daily": env("CHAT_DAILY_THROTTLE_RATE", default="300/day"),
    },
}

EMAIL_BACKEND = env("EMAIL_BACKEND", default="django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = env("EMAIL_HOST", default="localhost")
EMAIL_PORT = env.int("EMAIL_PORT", default=587)
EMAIL_HOST_USER = env("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", default="")
EMAIL_USE_TLS = env.bool("EMAIL_USE_TLS", default=True)
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", default="CesucaCode <naoresponda@cesuca.edu.br>")

SPECTACULAR_SETTINGS = {
    "TITLE": "CesucaCode API",
    "DESCRIPTION": "IA acadêmica dos cursos de Tecnologia do Centro Universitário Cesuca",
    "VERSION": "1.0.1",
    "SERVE_INCLUDE_SCHEMA": False,
    "SORT_OPERATIONS": False,
    "SERVE_PERMISSIONS": ["rest_framework.permissions.AllowAny"],
}

from datetime import timedelta  # noqa: E402

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=2),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

CORS_ALLOWED_ORIGINS = env.list("CORS_ALLOWED_ORIGINS", default=[])

LLM_PROVIDER = env("LLM_PROVIDER", default="gemini")
LLM_MODEL = env("LLM_MODEL", default="gemini-2.5-flash")
EMBEDDING_PROVIDER = env("EMBEDDING_PROVIDER", default="gemini")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", default="gemini-embedding-001")
EMBEDDING_DIMENSIONS = env.int("EMBEDDING_DIMENSIONS", default=768)
# Textos por chamada de embedding na ingestão de materiais.
EMBEDDING_BATCH_SIZE = env.int("EMBEDDING_BATCH_SIZE", default=100)
# Teto de textos embeddados por minuto (0 = sem limite, o padrão para produção).
# Use ~90 no plano gratuito do Gemini (100/min); sem isso, materiais grandes
# estouram a cota e falham.
EMBEDDING_MAX_REQUESTS_PER_MINUTE = env.int("EMBEDDING_MAX_REQUESTS_PER_MINUTE", default=0)

GOOGLE_API_KEY = env("GOOGLE_API_KEY", default="")
OPENAI_API_KEY = env("OPENAI_API_KEY", default="")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY", default="")
DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY", default="")
ABACUSAI_API_KEY = env("ABACUSAI_API_KEY", default="")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY", default="")
# Opcional: o OpenRouter usa esse cabeçalho para atribuir o tráfego ao app (ranking deles).
OPENROUTER_HTTP_REFERER = env("OPENROUTER_HTTP_REFERER", default="")

# Papéis de modelo (apps/ai_providers): LLM_{PAPEL}_PROVIDER/MODEL/MAX_TOKENS/TEMPERATURE.
# Vazio = cai em LLM_PROVIDER/LLM_MODEL (provider/model) ou no padrão do papel
# (max_tokens/temperature; ver ai_providers.services.ROLE_DEFAULTS).
for _role in ("ANSWER", "ROUTER", "AGENT", "JUDGE"):
    globals()[f"LLM_{_role}_PROVIDER"] = env(f"LLM_{_role}_PROVIDER", default="")
    globals()[f"LLM_{_role}_MODEL"] = env(f"LLM_{_role}_MODEL", default="")
    globals()[f"LLM_{_role}_MAX_TOKENS"] = env(f"LLM_{_role}_MAX_TOKENS", default="")
    globals()[f"LLM_{_role}_TEMPERATURE"] = env(f"LLM_{_role}_TEMPERATURE", default="")
LLM_TIMEOUT = env.float("LLM_TIMEOUT", default=60)  # segundos
LLM_MAX_RETRIES = env.int("LLM_MAX_RETRIES", default=2)
# Sobrescreve o método de saída estruturada do registro de capacidades:
# json_schema | function_calling | json_mode (vazio = usa o registro).
LLM_STRUCTURED_METHOD = env("LLM_STRUCTURED_METHOD", default="")
# Gemini: orçamento de thinking em tokens (0 desliga; vazio = padrão do modelo).
LLM_THINKING_BUDGET = env("LLM_THINKING_BUDGET", default="")
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", default="http://localhost:11434")

_system_prompt_path = Path(
    env("SYSTEM_PROMPT_PATH", default="apps/conversations/prompts/sofia")
)
SYSTEM_PROMPT_PATH = str(
    _system_prompt_path if _system_prompt_path.is_absolute() else BASE_DIR / _system_prompt_path
)

# Distância de cosseno máxima (0 = idêntico, ~1 = sem relação) pra um trecho dos
# materiais ser considerado relevante e entrar no contexto do chat. Acima disso,
# o trecho é descartado em vez de virar "contexto" de uma pergunta que não tem
# nada a ver com ele.
RAG_MAX_DISTANCE = env.float("RAG_MAX_DISTANCE", default=0.30)

# Busca híbrida: vetorial (pgvector) + textual (tsvector 'portuguese'), fundidas por
# RRF. Desligada, volta ao comportamento antigo (só vetorial, com corte de distância).
RAG_HYBRID_ENABLED = env.bool("RAG_HYBRID_ENABLED", default=True)
# Gancho de reranking (retrieval.rerank). Ainda é a identidade: só ganha efeito
# quando um reranker for implementado e o eval mostrar ganho.
RAG_RERANK_ENABLED = env.bool("RAG_RERANK_ENABLED", default=False)
# Tamanho máximo de cada chunk, em "tokens" aproximados (palavras + pontuação; ver
# documents/chunking.py). Mudar isto exige `manage.py reprocess_documents`.
RAG_CHUNK_MAX_TOKENS = env.int("RAG_CHUNK_MAX_TOKENS", default=300)
# Tentativas e espera base (s, dobra a cada tentativa) para erros transitórios
# (429/cota por minuto, timeout, conexão) ao gerar embeddings na ingestão.
EMBEDDING_RETRY_ATTEMPTS = env.int("EMBEDDING_RETRY_ATTEMPTS", default=3)
EMBEDDING_RETRY_BASE_SECONDS = env.float("EMBEDDING_RETRY_BASE_SECONDS", default=5.0)

# Quantas mensagens anteriores da conversa vão pro modelo a cada turno. Sem teto,
# conversas longas ficam cada vez mais caras/lentas e estouram o contexto.
CHAT_MAX_HISTORY_MESSAGES = env.int("CHAT_MAX_HISTORY_MESSAGES", default=12)

# Se nenhum trecho dos materiais for relevante: True = pode explicar conceitos
# gerais de computação (avisando que não veio dos materiais); False = modo
# estrito, responde só que não encontrou nos materiais enviados.
CHAT_ALLOW_GENERAL_KNOWLEDGE = env.bool("CHAT_ALLOW_GENERAL_KNOWLEDGE", default=True)

# Pesquisa na web (DuckDuckGo, sem chave) só em perguntas de grade curricular e
# disciplinas, pra sugerir um caminho de estudo. Falhou ou desligado: o chat
# segue normalmente sem as referências externas. Ignorada no modo estrito.
CHAT_WEB_SEARCH_ENABLED = env.bool("CHAT_WEB_SEARCH_ENABLED", default=True)
CHAT_WEB_SEARCH_MAX_RESULTS = env.int("CHAT_WEB_SEARCH_MAX_RESULTS", default=5)
CHAT_WEB_SEARCH_TIMEOUT = env.int("CHAT_WEB_SEARCH_TIMEOUT", default=8)

# Versão do pipeline do chat, gravada em cada MessageTrace para comparar
# resultados entre versões (v0 = baseline anterior ao harness agêntico).
PIPELINE_VERSION = env("PIPELINE_VERSION", default="v2")

# Loop agêntico (rota "composta", apps/conversations/agent.py). Desligado, as
# perguntas compostas seguem o RAG simples. O orçamento limita custo e latência:
# ao estourar qualquer teto, o agente para e responde com o que já coletou.
CHAT_AGENT_ENABLED = env.bool("CHAT_AGENT_ENABLED", default=True)
AGENT_MAX_TURNS = env.int("AGENT_MAX_TURNS", default=5)  # chamadas ao modelo com ferramentas
AGENT_MAX_TOOL_CALLS = env.int("AGENT_MAX_TOOL_CALLS", default=8)
AGENT_MAX_TOTAL_TOKENS = env.int("AGENT_MAX_TOTAL_TOKENS", default=40000)  # entrada + saída acumuladas
AGENT_MAX_SECONDS = env.float("AGENT_MAX_SECONDS", default=30)

# Roteador de intenção (L0 determinístico + L1 com LLM pequeno). Desligado, só o L0
# roda e o resto cai no fallback (heurística da v0: regex de grade + pergunta anterior).
CHAT_ROUTER_ENABLED = env.bool("CHAT_ROUTER_ENABLED", default=True)
