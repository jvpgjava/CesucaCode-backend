import time

from django.core.management.base import BaseCommand

from apps.documents import services
from apps.documents.models import Document


class Command(BaseCommand):
    help = (
        "Reenfileira materiais presos em `processing` há mais de N minutos (ex.: o "
        "servidor reiniciou no meio da ingestão). Rodado no boot pelo "
        "docker-entrypoint.py. Gera embeddings de verdade (tem custo/cota)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--minutes", type=int, default=15, help="Considera preso quem está em processing há mais que isso (padrão: 15).")
        parser.add_argument("--timeout", type=int, default=3600, help="Segundos aguardando o processamento (padrão: 3600).")
        parser.add_argument("--no-wait", action="store_true", help="Só enfileira, sem aguardar (o pool é do processo: ele só termina de processar se o processo continuar vivo).")

    def handle(self, *args, **opts):
        stuck = services.recover_stuck_documents(minutes=opts["minutes"])
        if not stuck:
            self.stdout.write("Nenhum material preso em processamento.")
            return
        for document in stuck:
            self.stdout.write(f"reenfileirado: {document.title}")
        if opts["no_wait"]:
            return

        deadline = time.monotonic() + opts["timeout"]
        for document in stuck:
            while document.status == Document.Status.PROCESSING and time.monotonic() < deadline:
                time.sleep(2)
                document.refresh_from_db(fields=["status", "processing_error"])
            if document.status == Document.Status.READY:
                self.stdout.write(self.style.SUCCESS(f"pronto: {document.title}"))
            else:
                self.stdout.write(self.style.ERROR(f"{document.status}: {document.title} — {document.processing_error or 'tempo esgotado'}"))
