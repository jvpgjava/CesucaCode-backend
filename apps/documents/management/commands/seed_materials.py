import json
import time
from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from apps.accounts.models import Course, User
from apps.documents import services
from apps.documents.models import Document

SEED_DIR = Path(__file__).resolve().parents[2] / "seed_materials"


class Command(BaseCommand):
    help = (
        "Carrega os materiais didáticos mínimos (apps/documents/seed_materials/) para o "
        "RAG funcionar numa instalação nova. Idempotente: pula o que já existe (pelo "
        "título). Gera embeddings de verdade, então precisa do provider de embedding "
        "configurado no .env. Requer um CSAdmin (python manage.py createsuperuser)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--admin-email", help="CSAdmin dono dos materiais (padrão: o primeiro).")
        parser.add_argument("--timeout", type=int, default=1800, help="Segundos aguardando o processamento (padrão: 1800; no plano gratuito do Gemini o Plano de Ensino leva ~8 min).")

    def handle(self, *args, **opts):
        admin = (
            User.objects.filter(role=User.Role.CS_ADMIN, email=opts["admin_email"]).first()
            if opts["admin_email"]
            else User.objects.filter(role=User.Role.CS_ADMIN).order_by("id").first()
        )
        if admin is None:
            raise CommandError("Nenhum CSAdmin encontrado. Rode `python manage.py createsuperuser` primeiro.")

        manifest = json.loads((SEED_DIR / "manifest.json").read_text(encoding="utf-8"))
        pending = []
        for item in manifest:
            existing = Document.objects.filter(title=item["title"]).first()
            if existing and existing.status == Document.Status.FAILED:
                services.reprocess_document(existing)
                self.stdout.write(f"reprocessando (tinha falhado): {item['title']}")
                pending.append(existing)
                continue
            if existing:
                self.stdout.write(f"já existe, pulando: {item['title']}")
                continue
            courses = list(Course.objects.filter(code__in=item["courses"]))
            if len(courses) != len(item["courses"]):
                raise CommandError(f"Cursos {item['courses']} não existem no banco (rode `migrate`).")
            path = SEED_DIR / item["file"]
            with path.open("rb") as fh:
                document = services.create_document(
                    title=item["title"], courses=courses, file=File(fh, name=path.name), uploaded_by=admin
                )
            self.stdout.write(f"enviado: {item['title']} (processando...)")
            pending.append(document)

        deadline = time.monotonic() + opts["timeout"]
        failed = 0
        for document in pending:
            while document.status == Document.Status.PROCESSING and time.monotonic() < deadline:
                time.sleep(2)
                document.refresh_from_db(fields=["status", "processing_error"])
            if document.status == Document.Status.READY:
                self.stdout.write(self.style.SUCCESS(f"pronto: {document.title} ({document.chunks.count()} chunks)"))
            else:
                failed += 1
                reason = document.processing_error or "tempo esgotado"
                self.stdout.write(self.style.ERROR(f"{document.status}: {document.title} — {reason}"))

        if failed:
            raise CommandError(f"{failed} material(is) não ficaram prontos. Veja o .env (provider de embedding) e use Reprocessar.")
        self.stdout.write(self.style.SUCCESS("Base mínima de materiais pronta."))
