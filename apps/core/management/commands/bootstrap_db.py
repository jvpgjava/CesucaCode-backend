import getpass

import psycopg
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from psycopg import sql


class Command(BaseCommand):
    help = (
        "Cria a role/banco do PostgreSQL definidos em DATABASE_URL (idempotente) "
        "e habilita a extensão pgvector. Pede a senha de um superusuário do "
        "Postgres (padrão: 'postgres') só para essa configuração inicial."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--superuser",
            default="postgres",
            help="Usuário administrador do PostgreSQL usado para criar a role/banco (padrão: postgres).",
        )
        parser.add_argument(
            "--superuser-password",
            default=None,
            help="Senha do superusuário. Se omitida, é pedida interativamente (mais seguro que passar na linha de comando).",
        )

    def handle(self, *args, **options):
        db = settings.DATABASES["default"]
        target_db = db["NAME"]
        target_user = db["USER"]
        target_password = db["PASSWORD"]
        host = db["HOST"] or "localhost"
        port = db["PORT"] or "5432"
        superuser = options["superuser"]

        if not target_db or not target_user:
            raise CommandError(
                "DATABASE_URL não está configurada corretamente no .env "
                "(faltando nome do banco ou usuário)."
            )
        if not target_password:
            raise CommandError(
                "DATABASE_URL precisa incluir uma senha para criar a role/banco "
                "(ex.: postgres://usuario:senha@host:porta/banco)."
            )

        superuser_password = options["superuser_password"] or getpass.getpass(
            f"Senha do usuário '{superuser}' no PostgreSQL ({host}:{port}): "
        )

        self.stdout.write(f"Conectando em {host}:{port} como '{superuser}'...")
        try:
            admin_conn = psycopg.connect(
                host=host, port=port, user=superuser, password=superuser_password,
                dbname="postgres", autocommit=True,
            )
        except psycopg.OperationalError as exc:
            raise CommandError(f"Não consegui conectar como '{superuser}': {exc}") from exc

        with admin_conn, admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [target_user])
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(
                        sql.Identifier(target_user), sql.Literal(target_password)
                    )
                )
                self.stdout.write(self.style.SUCCESS(f"Role '{target_user}' criada."))
            else:
                self.stdout.write(f"Role '{target_user}' já existia, ok.")

            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", [target_db])
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {}").format(
                        sql.Identifier(target_db), sql.Identifier(target_user)
                    )
                )
                self.stdout.write(self.style.SUCCESS(f"Banco '{target_db}' criado."))
            else:
                self.stdout.write(f"Banco '{target_db}' já existia, ok.")
        admin_conn.close()

        target_conn = psycopg.connect(
            host=host, port=port, user=superuser, password=superuser_password,
            dbname=target_db, autocommit=True,
        )
        with target_conn, target_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON SCHEMA public TO {}").format(sql.Identifier(target_user))
            )
        target_conn.close()

        self.stdout.write(self.style.SUCCESS(
            "Banco pronto! Agora rode: python manage.py migrate"
        ))
