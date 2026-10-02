import time

from django.core.management.base import BaseCommand, CommandError

from apps.documents import services
from apps.documents.models import Document


class Command(BaseCommand):
    help = (
        "Reprocessa TODOS os materiais (refaz chunks e embeddings). Use depois de "
        "trocar o provider/modelo de embedding: vetores de modelos diferentes não "
        "são comparáveis. Gera embeddings de verdade (tem custo/cota)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--timeout", type=int, default=3600, help="Segundos aguardando o processamento (padrão: 3600).")

    def handle(self, *args, **opts):
        documents = list(Document.objects.all())
        if not documents:
            self.stdout.write("Nenhum material para reprocessar.")
            return
        for document in documents:
            services.reprocess_document(document)
            self.stdout.write(f"reprocessando: {document.title}")

        deadline = time.monotonic() + opts["timeout"]
        failed = 0
        for document in documents:
            while document.status == Document.Status.PROCESSING and time.monotonic() < deadline:
                time.sleep(2)
                document.refresh_from_db(fields=["status", "processing_error"])
            if document.status == Document.Status.READY:
                self.stdout.write(self.style.SUCCESS(f"pronto: {document.title} ({document.chunks.count()} chunks)"))
            else:
                failed += 1
                self.stdout.write(self.style.ERROR(f"{document.status}: {document.title} — {document.processing_error or 'tempo esgotado'}"))
        if failed:
            raise CommandError(f"{failed} material(is) não ficaram prontos.")
        self.stdout.write(self.style.SUCCESS("Todos os materiais reprocessados."))
