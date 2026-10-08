"""Aguarda o Postgres, garante a extensão pgvector, aplica migrações e reenfileira
materiais presos em processamento."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psycopg


def _dsn() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL não está definida.")
    return url.replace("postgres://", "postgresql://", 1)


def wait_for_db(dsn: str, attempts: int = 60) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    sys.exit(f"PostgreSQL não ficou pronto a tempo: {last_error}")


def ensure_pgvector(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Informe o comando a executar após o entrypoint.")

    dsn = _dsn()
    wait_for_db(dsn)
    ensure_pgvector(dsn)

    import django
    from django.core.management import call_command

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    django.setup()
    call_command("migrate", interactive=False, verbosity=1)

    # Opcional: SEED_MATERIALS=1 carrega a base mínima de materiais do RAG. Precisa
    # de um CSAdmin e do provider de embedding configurado; se faltar algo, só avisa.
    if os.environ.get("SEED_MATERIALS", "").lower() in ("1", "true", "yes"):
        try:
            call_command("seed_materials")
        except Exception as exc:
            print(f"seed_materials não concluído: {exc}", file=sys.stderr)

    # Materiais presos em `processing` (o pool de ingestão vive na memória do processo
    # e se perde num restart) são reenfileirados. Roda em um subprocesso à parte para
    # não atrasar o boot do servidor: o execvp abaixo substitui este processo, e o
    # processamento (lento) precisa de um processo que continue vivo.
    try:
        subprocess.Popen(
            [sys.executable, "manage.py", "recover_documents"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except Exception as exc:
        print(f"recover_documents não iniciado: {exc}", file=sys.stderr)

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
