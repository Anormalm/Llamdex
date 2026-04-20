import argparse
import csv
import json
import os
from typing import Callable, Dict, Iterable, List, Optional


def _read_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _task_metric_values(rows: List[Dict[str, str]], task: str, metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if str(row.get("task", "")).strip() != task:
            continue
        value = _to_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def _reduce_optional(values: Iterable[Optional[float]], reducer: Callable[[List[float]], float]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return reducer(present)


def _best(rows: List[Dict[str, str]], task: str, metric: str) -> Optional[float]:
    return _reduce_optional(_task_metric_values(rows, task, metric), max)


def _best_min(rows: List[Dict[str, str]], task: str, metric: str) -> Optional[float]:
    return _reduce_optional(_task_metric_values(rows, task, metric), min)


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def build_review(
    task_rows: List[Dict[str, str]],
    grounded_rows: List[Dict[str, str]],
    api_rows: List[Dict[str, str]],
    args,
) -> Dict:
    single_acc = _best(task_rows, "finegrained", "accuracy")
    yesno_acc = _best(task_rows, "strict_yesno", "accuracy")
    population_mae = _best_min(task_rows, "population", "mae")
    grounded_f1 = _best(task_rows, "grounded_generation", "f1")

    grounded_consistency = None
    grounded_fmt = None
    grounded_nonempty = None
    if grounded_rows:
        grounded_consistency = _reduce_optional((_to_float(r.get("rationale_consistency")) for r in grounded_rows), max)
        grounded_fmt = _reduce_optional((_to_float(r.get("format_compliance")) for r in grounded_rows), max)
        grounded_nonempty = _reduce_optional((_to_float(r.get("rationale_nonempty_rate")) for r in grounded_rows), max)

    api_statuses = [str(r.get("status", "")).strip().lower() for r in api_rows if "status" in r]
    if api_statuses:
        api_ok = any(s == "ok" for s in api_statuses)
    else:
        api_ok = len(api_rows) > 0

    checks = [
        {
            "name": "single_image_accuracy",
            "value": single_acc,
            "threshold": args.min_single_image_acc,
            "operator": ">=",
            "pass": (single_acc is not None and single_acc >= args.min_single_image_acc),
        },
        {
            "name": "yesno_accuracy",
            "value": yesno_acc,
            "threshold": args.min_yesno_acc,
            "operator": ">=",
            "pass": (yesno_acc is not None and yesno_acc >= args.min_yesno_acc),
        },
        {
            "name": "population_mae",
            "value": population_mae,
            "threshold": args.max_population_mae,
            "operator": "<=",
            "pass": (population_mae is not None and population_mae <= args.max_population_mae),
        },
        {
            "name": "grounded_f1",
            "value": grounded_f1,
            "threshold": args.min_grounded_f1,
            "operator": ">=",
            "pass": (grounded_f1 is not None and grounded_f1 >= args.min_grounded_f1),
        },
        {
            "name": "rationale_consistency",
            "value": grounded_consistency,
            "threshold": args.min_rationale_consistency,
            "operator": ">=",
            "pass": (grounded_consistency is not None and grounded_consistency >= args.min_rationale_consistency),
        },
        {
            "name": "api_baseline_available",
            "value": 1.0 if api_ok else 0.0,
            "threshold": 1.0,
            "operator": "==",
            "pass": api_ok,
        },
    ]

    decision = "GO" if all(c["pass"] for c in checks) else "NO-GO"
    blockers = [c["name"] for c in checks if not c["pass"]]

    return {
        "decision": decision,
        "checks": checks,
        "blockers": blockers,
        "context": {
            "grounded_format_compliance": grounded_fmt,
            "grounded_nonempty_rate": grounded_nonempty,
            "task_rows": len(task_rows),
            "grounded_rows": len(grounded_rows),
            "api_rows": len(api_rows),
        },
    }


def _render_md(review: Dict) -> str:
    lines = []
    lines.append("# Pilot Go/No-Go Review")
    lines.append("")
    lines.append(f"**Decision:** `{review['decision']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | value | threshold | status |")
    lines.append("|---|---:|---:|---|")
    for c in review["checks"]:
        value = "n/a" if c["value"] is None else f"{c['value']:.4f}"
        threshold = f"{c['operator']} {c['threshold']:.4f}" if isinstance(c["threshold"], float) else f"{c['operator']} {c['threshold']}"
        status = "PASS" if c["pass"] else "FAIL"
        lines.append(f"| {c['name']} | {value} | {threshold} | {status} |")

    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if review["blockers"]:
        for b in review["blockers"]:
            lines.append(f"- {b}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- grounded_format_compliance: {review['context']['grounded_format_compliance']}")
    lines.append(f"- grounded_nonempty_rate: {review['context']['grounded_nonempty_rate']}")
    lines.append(f"- task_rows: {review['context']['task_rows']}")
    lines.append(f"- grounded_rows: {review['context']['grounded_rows']}")
    lines.append(f"- api_rows: {review['context']['api_rows']}")
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Generate pilot Go/No-Go decision from benchmark artifacts.")
    p.add_argument("--task_matrix_csv", type=str, default="/disk1/lfhu/runs/task_matrix_dtd_dinov2_full.remote.csv")
    p.add_argument("--grounded_csv", type=str, default="/disk1/lfhu/runs/urgent_grounded_rationale.csv")
    p.add_argument("--api_summary_csv", type=str, default="/disk1/lfhu/runs/api_baselines.remote.summary.csv")
    p.add_argument("--out_md", type=str, default="runs/go_no_go_review.md")
    p.add_argument("--out_json", type=str, default="runs/go_no_go_review.json")

    p.add_argument("--min_single_image_acc", type=float, default=0.70)
    p.add_argument("--min_yesno_acc", type=float, default=0.90)
    p.add_argument("--max_population_mae", type=float, default=0.05)
    p.add_argument("--min_grounded_f1", type=float, default=0.75)
    p.add_argument("--min_rationale_consistency", type=float, default=0.20)
    return p.parse_args()


def main():
    a = parse_args()
    task_rows = _read_csv(a.task_matrix_csv)
    grounded_rows = _read_csv(a.grounded_csv)
    api_rows = _read_csv(a.api_summary_csv)

    review = build_review(task_rows, grounded_rows, api_rows, a)

    _ensure_parent_dir(a.out_md)
    _ensure_parent_dir(a.out_json)

    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(review))
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2)

    print(f"Decision: {review['decision']}")
    print(f"MD: {a.out_md}")
    print(f"JSON: {a.out_json}")


if __name__ == "__main__":
    main()
