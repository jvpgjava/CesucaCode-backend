from django.core.management.base import BaseCommand

from apps.ai_providers import services


class Command(BaseCommand):
    help = "Testa a conexão com os providers de IA (chat e embedding) configurados no .env."

    def handle(self, *args, **options):
        self.stdout.write("Testando provider de chat...")
        try:
            chat = services.get_chat_model()
            response = chat.invoke("Responda apenas 'ok'.")
            self.stdout.write(self.style.SUCCESS(f"Chat OK: {response.content!r}"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Chat falhou: {exc}"))

        self.stdout.write("Testando provider de embedding...")
        try:
            embeddings = services.get_embedding_model()
            vector = embeddings.embed_query("teste de conexão")
            self.stdout.write(self.style.SUCCESS(f"Embedding OK: dimensão {len(vector)}"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Embedding falhou: {exc}"))
