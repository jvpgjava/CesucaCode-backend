from .dev import *  # noqa: F403

# Os testes criam um banco próprio (test_<nome>) e precisam da extensão pgvector
# nele. O usuário da aplicação normalmente não tem CREATEDB nem pode criar
# extensões, então TEST_DATABASE_URL permite apontar para um usuário com essas
# permissões (ex.: o superusuário local do Postgres). Sem ela, usa DATABASE_URL.
if env("TEST_DATABASE_URL", default=""):  # noqa: F405
    DATABASES = {"default": env.db("TEST_DATABASE_URL")}  # noqa: F405

# Senhas com hash rápido deixam a criação de usuários nos testes instantânea.
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
