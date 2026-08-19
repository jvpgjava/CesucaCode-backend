"""Aguarda o Postgres, garante a extensão pgvector e aplica migrações."""

from __future__ import annotations

import os
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

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
