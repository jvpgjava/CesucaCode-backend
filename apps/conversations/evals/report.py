"""Relatório da rodada: JSON completo, CSV por caso, tabelas de texto e comparação."""

import csv
import json
from pathlib import Path

from .metrics import acceptance, compare_summaries

DEFAULT_OUT_DIR = Path(__file__).with_name("results")

CSV_COLUMNS = [
    "id", "tags", "persona", "expect", "ok", "ok_heuristic", "route", "intent", "route_ok",
    "must_include_frac", "retrieval_hit", "retrieval_recall", "leak", "judge_correctness",
    "judge_faithfulness", "judge_abstained", "latency_s", "ttft_s", "input_tokens", "output_tokens",
    "n_chunks", "error",
]


def report_basename(variant: str, timestamp: str) -> str:
    return f"{variant}-{timestamp}"


def write_report(report: dict, out_dir: Path | str, variant: str, timestamp: str) -> tuple[Path, Path]:
    """Grava `<variante>-<timestamp>.json` e `.csv` em `out_dir`. Devolve os dois caminhos."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = report_basename(variant, timestamp)
    json_path, csv_path = out / f"{base}.json", out / f"{base}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in report["cases"]:
            judge = case.get("judge") or {}
            row = {column: case.get(column) for column in CSV_COLUMNS}
            row["tags"] = ";".join(case["tags"])
            row["judge_correctness"] = judge.get("correctness")
            row["judge_faithfulness"] = judge.get("faithfulness")
            row["judge_abstained"] = judge.get("abstained")
            writer.writerow(row)
    return json_path, csv_path


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- formatação

def _pct(value) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _num(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_summary_table(summary: dict, *, title: str = "") -> str:
    """Tabela-resumo: uma linha para o geral e uma por tag."""
    header = ["", "n", "acerto", "must_inc", "hit@k", "vazam.", "corr.", "fiel.", "p50 s", "p95 s", "ttft50", "tok in", "tok out"]
    rows = []

    def line(name: str, s: dict) -> list[str]:
        return [
            name, str(s["n"]), _pct(s["ok_rate"]), _pct(s["must_include_mean"]), _pct(s["retrieval_hit_rate"]),
            _pct(s["leak_rate"]), _num(s["correctness_mean"]), _num(s["faithfulness_mean"]),
            _num(s["latency_p50"], 1), _num(s["latency_p95"], 1), _num(s["ttft_p50"], 1),
            _num(s["input_tokens_mean"], 0), _num(s["output_tokens_mean"], 0),
        ]

    rows.append(line("GERAL", summary["overall"]))
    for expect, s in summary["by_expect"].items():
        rows.append(line(f"expect:{expect}", s))
    for tag, s in summary["by_tag"].items():
        rows.append(line(f"tag:{tag}", s))
    return _render(header, rows, title)


def _render(header: list[str], rows: list[list[str]], title: str = "") -> str:
    table = [header] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(header))]
    lines = [title] if title else []
    for index, row in enumerate(table):
        lines.append("  ".join(str(cell).ljust(widths[i]) if i == 0 else str(cell).rjust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_routes(summary: dict) -> str:
    header = ["rota", "n", "acerto", "p50 s", "p95 s"]
    rows = [
        [name, str(s["n"]), _pct(s["ok_rate"]), _num(s["latency_p50"], 1), _num(s["latency_p95"], 1)]
        for name, s in summary["by_route"].items()
    ]
    return _render(header, rows, "Rotas e latência")


def format_acceptance(summary: dict) -> str:
    lines = ["Limiares de aceite"]
    for check in acceptance(summary):
        mark = {True: "OK  ", False: "FALHA", None: "n/d "}[check["passed"]]
        value = "-" if check["value"] is None else check["value"]
        lines.append(f"  [{mark}] {check['criterion']} (valor: {value})")
    return "\n".join(lines)


def _fmt_metric(value, kind: str) -> str:
    if value is None:
        return "-"
    return _pct(value) if kind == "pct" else _num(value)


def _fmt_delta(delta, kind: str) -> str:
    if delta is None:
        return "-"
    if kind == "pct":
        return f"{delta * 100:+.0f} pp"
    return f"{delta:+.2f}"


def format_comparison(report_a: dict, report_b: dict, *, include_tags: bool = False) -> str:
    """Deltas lado a lado (B - A) do resumo geral e por expectativa (e por tag, se pedido)."""
    rows = compare_summaries(report_a["summary"], report_b["summary"])
    if not include_tags:
        rows = [row for row in rows if not row["scope"].startswith("tag:")]
    label_a = report_a["meta"].get("variant", "A")
    label_b = report_b["meta"].get("variant", "B")
    if label_a == label_b:
        label_a, label_b = f"{label_a} (A)", f"{label_b} (B)"
    header = ["escopo", "métrica", label_a, label_b, "delta"]
    table = [
        [row["scope"], row["label"], _fmt_metric(row["a"], row["kind"]), _fmt_metric(row["b"], row["kind"]),
         _fmt_delta(row["delta"], row["kind"])]
        for row in rows
    ]
    return _render(header, table, f"Comparação: {label_a} -> {label_b}")
