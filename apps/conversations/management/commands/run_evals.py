import argparse
import platform
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.conversations.evals import golden, report, runner
from apps.conversations.evals.metrics import acceptance, build_summary


class Command(BaseCommand):
    help = (
        "Roda o golden set contra o chat (send_message) e gera relatório JSON/CSV com acerto, "
        "retrieval hit@k, vazamento, correção/fidelidade (juiz), latência e tokens. "
        "FAZ CHAMADAS REAIS ao LLM e ao embedding (tem custo e leva minutos). "
        "Com --compare só compara dois relatórios já gerados, sem LLM."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--variant", choices=sorted(runner.VARIANTS), default="current",
            help="Flags do pipeline: v0 (baseline), v1 (router+híbrido+suficiência+follow-ups), "
                 "v2 (tudo, com agente) ou current (usa as settings atuais). Padrão: current.",
        )
        parser.add_argument("--tag", action="append", help="Só casos com esta tag (repetível).")
        parser.add_argument("--id", action="append", dest="ids",
                            help="Só este id (repetível; aceita prefixo com '*', ex.: aval-*).")
        parser.add_argument("--limit", type=int, help="Roda só os N primeiros casos (depois dos filtros).")
        parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False,
                            help="Liga o LLM-judge (papel 'judge'): correção, fidelidade e abstenção. Padrão: --no-judge.")
        parser.add_argument("--out", default=str(report.DEFAULT_OUT_DIR),
                            help="Diretório de saída (padrão: apps/conversations/evals/results/).")
        parser.add_argument("--golden", help="Caminho de outro golden.yaml (padrão: o do repositório).")
        parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                            help="Compara dois relatórios (B - A) e sai, sem rodar o chat.")
        parser.add_argument("--compare-tags", action="store_true", help="Com --compare, inclui o detalhe por tag.")
        parser.add_argument("--strict", action="store_true",
                            help="Sai com erro se algum limiar de aceite (vazamento=0, recusa/abstenção>=90%%, "
                                 "fidelidade>=0,85, p95 direta<=6s) falhar.")

    def handle(self, *args, **opts):
        if opts["compare"]:
            return self._compare(*opts["compare"], include_tags=opts["compare_tags"])

        try:
            cases = golden.load_golden(opts["golden"])
        except (OSError, golden.GoldenError) as exc:
            raise CommandError(str(exc)) from exc
        cases = golden.filter_cases(cases, tags=opts["tag"], ids=opts["ids"], limit=opts["limit"])
        if not cases:
            raise CommandError("Nenhum caso selecionado pelos filtros.")

        variant = opts["variant"]
        use_judge = opts["judge"]
        self.stdout.write(
            f"Rodando {len(cases)} caso(s) | variante={variant} | juiz={'sim' if use_judge else 'não'}\n"
        )

        started = datetime.now()
        with runner.variant_settings(variant):
            flags = runner.effective_flags()
            results = runner.run_cases(cases, judge=use_judge, progress=self._progress)

        summary = build_summary(results)
        data = {
            "meta": {
                "variant": variant,
                "started_at": started.isoformat(timespec="seconds"),
                "duration_s": round((datetime.now() - started).total_seconds(), 1),
                "judge": use_judge,
                "n_cases": len(results),
                "filters": {"tag": opts["tag"], "id": opts["ids"], "limit": opts["limit"]},
                "flags": flags,
                "models": {
                    "answer": getattr(settings, "LLM_ANSWER_MODEL", "") or settings.LLM_MODEL,
                    "router": getattr(settings, "LLM_ROUTER_MODEL", ""),
                    "agent": getattr(settings, "LLM_AGENT_MODEL", ""),
                    "judge": getattr(settings, "LLM_JUDGE_MODEL", ""),
                },
                "python": platform.python_version(),
            },
            "summary": summary,
            "acceptance": acceptance(summary),
            "cases": results,
        }
        json_path, csv_path = report.write_report(data, opts["out"], variant, started.strftime("%Y%m%d-%H%M%S"))

        self.stdout.write("")
        self.stdout.write(report.format_summary_table(summary, title=f"Resumo — {variant}"))
        self.stdout.write("")
        self.stdout.write(report.format_routes(summary))
        self.stdout.write("")
        self.stdout.write(report.format_acceptance(summary))
        self.stdout.write("")
        self.stdout.write(f"JSON: {json_path}")
        self.stdout.write(f"CSV:  {csv_path}")

        failed = [r["id"] for r in results if not r["ok"]]
        if failed:
            self.stdout.write(self.style.WARNING(f"{len(failed)} caso(s) fora da expectativa: {', '.join(failed)}"))
        if opts["strict"]:
            reprovados = [c["criterion"] for c in data["acceptance"] if c["passed"] is False]
            if reprovados:
                raise CommandError("Limiares de aceite não atendidos: " + "; ".join(reprovados))

    def _progress(self, index: int, total: int, result: dict) -> None:
        mark = self.style.SUCCESS("ok   ") if result["ok"] else self.style.ERROR("FALHA")
        seconds = "-" if result["latency_s"] is None else f"{result['latency_s']:.1f}s"
        extra = f" erro={result['error'][:60]!r}" if result.get("error") else ""
        self.stdout.write(
            f"[{index:>3}/{total}] {mark} {result['id']:<32} {seconds:>6} rota={result.get('route') or '-'}{extra}"
        )

    def _compare(self, path_a: str, path_b: str, *, include_tags: bool) -> None:
        try:
            a, b = report.load_report(path_a), report.load_report(path_b)
        except (OSError, ValueError) as exc:
            raise CommandError(f"Não consegui ler os relatórios: {exc}") from exc
        if "summary" not in a or "summary" not in b:
            raise CommandError("Arquivo não parece um relatório do run_evals (sem 'summary').")
        self.stdout.write(report.format_comparison(a, b, include_tags=include_tags))
