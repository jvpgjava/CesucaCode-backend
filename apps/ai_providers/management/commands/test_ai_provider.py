from django.core.management.base import BaseCommand, CommandError
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from apps.ai_providers import services

CHECKS = ("stream", "tools", "structured")


@tool
def somar(a: int, b: int) -> int:
    """Soma dois números inteiros."""
    return a + b


class _Cidade(BaseModel):
    nome: str
    pais: str


class Command(BaseCommand):
    help = (
        "Testa a conexão com os providers de IA (chat e embedding) configurados no .env. "
        "Com --check, roda o teste de contrato (stream/tools/structured) de um papel de modelo."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--role",
            choices=services.ROLES,
            default="answer",
            help="Papel de modelo a testar (padrão: answer).",
        )
        parser.add_argument(
            "--check",
            default="",
            help=(
                "Lista separada por vírgula de verificações de contrato: "
                f"{', '.join(CHECKS)}. Sem ela, faz o teste simples de chat + embedding."
            ),
        )

    def handle(self, *args, **options):
        role = options["role"]
        checks = [c.strip() for c in options["check"].split(",") if c.strip()]
        unknown = [c for c in checks if c not in CHECKS]
        if unknown:
            raise CommandError(f"Verificação(ões) desconhecida(s): {', '.join(unknown)}. Use: {', '.join(CHECKS)}.")

        if checks:
            self._run_checks(role, checks)
            return

        self.stdout.write(f"Testando provider de chat (papel {role})...")
        try:
            chat = services.get_chat_model(role)
            response = chat.invoke("Responda apenas 'ok'.")
            self.stdout.write(self.style.SUCCESS(f"Chat OK: {response.content!r}"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Chat falhou: {exc}"))

        self.stdout.write("Testando provider de embedding...")
        try:
            embeddings = services.get_embedding_model()
            vector = embeddings.embed_query("teste de conexão")
            services.validate_embedding_dimensions(vector)
            self.stdout.write(self.style.SUCCESS(f"Embedding OK: dimensão {len(vector)}"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Embedding falhou: {exc}"))

    # --- teste de contrato -----------------------------------------------------

    def _run_checks(self, role: str, checks: list[str]) -> None:
        try:
            caps = services.get_capabilities(role)
        except Exception as exc:
            raise CommandError(f"Configuração inválida para o papel {role}: {exc}") from exc
        self.stdout.write(
            f"Papel {role}: {caps.provider}/{caps.model} (structured={caps.structured_method}, "
            f"tools={caps.supports_tools}, parallel_tools={caps.parallel_tool_calls})"
        )

        failures = 0
        for name in checks:
            try:
                detail = getattr(self, f"_check_{name}")(role, caps)
                self.stdout.write(self.style.SUCCESS(f"[ok] {name}: {detail}"))
            except Exception as exc:
                failures += 1
                self.stdout.write(self.style.ERROR(f"[falhou] {name}: {exc}"))
        if failures:
            raise CommandError(f"{failures} verificação(ões) falharam para o papel {role}.")

    def _check_stream(self, role, caps) -> str:
        model = services.get_chat_model(role)
        chunks, text, final = 0, "", None
        for chunk in model.stream("Conte de 1 a 5, separado por vírgulas."):
            chunks += 1
            text += chunk.text if hasattr(chunk, "text") else str(chunk.content)
            final = chunk if final is None else final + chunk
        if not chunks or not text.strip():
            raise RuntimeError("o stream não devolveu texto")
        usage = services.extract_usage(final)
        if not (usage["input_tokens"] or usage["output_tokens"]):
            raise RuntimeError(f"stream OK ({chunks} pedaços), mas sem usage_metadata (tokens não serão contabilizados)")
        return f"{chunks} pedaços, usage={usage}"

    def _check_tools(self, role, caps) -> str:
        if not caps.supports_tools:
            raise RuntimeError("o registro de capacidades diz que este modelo não suporta tools")
        model = services.get_chat_model(role).bind_tools([somar])
        response = model.invoke("Use a ferramenta somar para calcular 17 + 25.")
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            raise RuntimeError(f"nenhuma tool_call na resposta: {str(response.content)[:120]!r}")
        return f"tool_calls={[(c['name'], c['args']) for c in calls]}"

    def _check_structured(self, role, caps) -> str:
        result = services.invoke_structured(
            _Cidade, [HumanMessage(content="Qual a capital do Brasil? Informe nome e país.")], role
        )
        if not result.ok:
            raise RuntimeError(result.error)
        return f"{result.value!r} (usage={result.usage})"
