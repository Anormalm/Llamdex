from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def load_result_rows(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "rows" in raw:
            return list(raw["rows"])
        raise ValueError(f"Unsupported JSON payload in {path}")
    if suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported result format: {path}")


def normalize_result_row(row: Dict, source_path: str) -> Dict:
    out = dict(row)
    out["source_path"] = source_path
    out.setdefault("status", "ok")
    out.setdefault("task", out.get("task_name", out.get("dataset", "unknown")))
    out.setdefault("baseline_name", out.get("model_type", out.get("baseline_mode", "unknown")))
    out.setdefault("benchmark_family", out.get("benchmark_family", "unspecified"))
    metric_name = out.get("metric_name")
    if not metric_name:
        if "accuracy" in out:
            metric_name = "accuracy"
        elif "mae" in out:
            metric_name = "mae"
        elif "metric" in out:
            metric_name = "metric"
        else:
            metric_name = "unknown"
    out["metric_name"] = metric_name
    if "metric" not in out:
        value = out.get(metric_name, 0.0)
        try:
            out["metric"] = float(value)
        except Exception:
            out["metric"] = 0.0
    return out


def summarize_rows(rows: Sequence[Dict]) -> Dict:
    total = len(rows)
    ok_rows = [r for r in rows if str(r.get("status", "ok")).lower() == "ok"]
    error_rows = [r for r in rows if str(r.get("status", "ok")).lower() != "ok"]
    family_counts: Dict[str, int] = {}
    for row in rows:
        fam = str(row.get("benchmark_family", "unspecified"))
        family_counts[fam] = family_counts.get(fam, 0) + 1
    best_by_task: Dict[str, Dict] = {}
    for row in ok_rows:
        task = str(row.get("task", "unknown"))
        current = best_by_task.get(task)
        metric_name = str(row.get("metric_name", "metric"))
        lower_is_better = metric_name in {"mae", "calibration_error", "latency", "inference_latency"}
        metric_value = float(row.get("metric", 0.0))
        if current is None:
            best_by_task[task] = row
            continue
        prev = float(current.get("metric", 0.0))
        if lower_is_better:
            if metric_value < prev:
                best_by_task[task] = row
        else:
            if metric_value > prev:
                best_by_task[task] = row
    return {
        "total_rows": total,
        "ok_rows": len(ok_rows),
        "error_rows": len(error_rows),
        "family_counts": family_counts,
        "best_by_task": best_by_task,
    }


def render_markdown_report(rows: Sequence[Dict], summary: Dict, title: str = "Benchmark Report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- Total rows: {summary['total_rows']}")
    lines.append(f"- OK rows: {summary['ok_rows']}")
    lines.append(f"- Error rows: {summary['error_rows']}")
    if summary.get("family_counts"):
        lines.append(f"- Families: {summary['family_counts']}")
    lines.append("")
    lines.append("## Best By Task")
    if summary["best_by_task"]:
        for task, row in sorted(summary["best_by_task"].items()):
            lines.append(
                f"- `{task}`: `{row.get('baseline_name')}` "
                f"{row.get('metric_name')}={float(row.get('metric', 0.0)):.4f} "
                f"from `{row.get('source_path')}`"
            )
    else:
        lines.append("- No successful rows.")
    lines.append("")
    lines.append("## Errors")
    error_rows = [r for r in rows if str(r.get("status", "ok")).lower() != "ok"]
    if error_rows:
        for row in error_rows[:20]:
            lines.append(
                f"- `{row.get('baseline_name')}` on `{row.get('task')}` from `{row.get('source_path')}`: "
                f"{row.get('error', 'unknown error')}"
            )
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"
