import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


METRIC_FIELDS = [
    "accuracy",
    "f1",
    "mae",
    "calibration_error",
    "rationale_consistency",
    "format_compliance",
    "rationale_nonempty_rate",
    "answer_code_accuracy",
    "rationale_faithful_rate",
]


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _std(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0 if vals else None
    mu = sum(vals) / len(vals)
    return float((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)


def _row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")).strip(),
        str(row.get("task", "")).strip(),
        str(row.get("baseline_mode", "")).strip(),
        str(row.get("fusion_policy", "")).strip(),
    )


def _status_counts(rows: List[Dict[str, str]]) -> Counter:
    return Counter(str(r.get("status", "")).strip().lower() or "unknown" for r in rows)


def _aggregate_rows(path: str, rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("seed", "")).strip().lower() == "aggregate":
            continue
        grouped[_row_key(row)].append(row)

    out: List[Dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        dataset, task, baseline_mode, fusion_policy = key
        status = _status_counts(group)
        summary: Dict[str, object] = {
            "source_file": path,
            "dataset": dataset,
            "task": task,
            "baseline_mode": baseline_mode,
            "fusion_policy": fusion_policy,
            "rows": len(group),
            "ok_rows": int(status.get("ok", 0)),
            "error_rows": int(status.get("error", 0)),
            "mixed_rows": int(status.get("mixed", 0)),
            "seeds": len({str(r.get("seed", "")).strip() for r in group if str(r.get("seed", "")).strip()}),
        }
        for field in METRIC_FIELDS:
            vals = [_to_float(r.get(field)) for r in group]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[f"{field}_mean"] = _mean(vals)
                summary[f"{field}_std"] = _std(vals)
        summary["triage"] = _triage(summary)
        summary["reasons"] = "; ".join(_reasons(summary))
        out.append(summary)
    return out


def _reasons(row: Dict[str, object]) -> List[str]:
    reasons: List[str] = []
    if int(row.get("error_rows", 0)) > 0 or int(row.get("mixed_rows", 0)) > 0:
        reasons.append("has error/mixed rows")
    if int(row.get("ok_rows", 0)) == 0:
        reasons.append("no ok rows")
    if int(row.get("seeds", 0)) < 3:
        reasons.append("fewer than 3 seeds")

    task = str(row.get("task", ""))
    acc = row.get("accuracy_mean")
    f1 = row.get("f1_mean")
    mae = row.get("mae_mean")
    fmt = row.get("format_compliance_mean")
    nonempty = row.get("rationale_nonempty_rate_mean")

    if isinstance(acc, float) and isinstance(f1, float):
        if acc >= 0.95 and f1 <= 0.10:
            reasons.append("high accuracy with near-zero F1")
        if acc <= 0.02 and task in {"finegrained", "grounded_generation", "vqa"}:
            reasons.append("near-zero accuracy")
    if task == "population" and isinstance(mae, float) and mae > 0.15:
        reasons.append("population MAE too high")
    if task == "grounded_generation":
        if isinstance(fmt, float) and fmt < 0.95:
            reasons.append("format compliance below 0.95")
        if isinstance(nonempty, float) and nonempty < 0.50:
            reasons.append("rationale nonempty below 0.50")
    return reasons


def _triage(row: Dict[str, object]) -> str:
    reasons = _reasons(row)
    fatal = {"has error/mixed rows", "no ok rows", "high accuracy with near-zero F1", "near-zero accuracy"}
    if any(r in fatal for r in reasons):
        return "drop_or_fix"
    if reasons:
        return "needs_rerun_or_caveat"
    return "usable"


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = Counter(str(r.get("triage", "")) for r in rows)
    lines = ["# Task-Matrix Result Audit", ""]
    lines.append("## Summary")
    lines.append("")
    for key in ["usable", "needs_rerun_or_caveat", "drop_or_fix"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| triage | dataset | task | baseline | acc | f1 | mae | reasons | source |")
    lines.append("|---|---|---|---|---:|---:|---:|---|---|")
    for row in rows:
        acc = row.get("accuracy_mean")
        f1 = row.get("f1_mean")
        mae = row.get("mae_mean")
        lines.append(
            "| {triage} | {dataset} | {task} | {baseline} | {acc} | {f1} | {mae} | {reasons} | {source} |".format(
                triage=row.get("triage", ""),
                dataset=row.get("dataset", ""),
                task=row.get("task", ""),
                baseline=row.get("baseline_mode", ""),
                acc="" if acc is None else f"{float(acc):.3f}",
                f1="" if f1 is None else f"{float(f1):.3f}",
                mae="" if mae is None else f"{float(mae):.3f}",
                reasons=str(row.get("reasons", "")).replace("|", "/"),
                source=os.path.basename(str(row.get("source_file", ""))),
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Audit task-matrix CSV artifacts for publication-readiness triage.")
    p.add_argument("inputs", nargs="*", help="CSV files or glob patterns.")
    p.add_argument("--out_csv", default="runs/task_matrix_result_audit.csv")
    p.add_argument("--out_json", default="runs/task_matrix_result_audit.json")
    p.add_argument("--out_md", default="runs/task_matrix_result_audit.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    patterns = args.inputs or ["/disk1/lfhu/runs/task_matrix_*.csv"]
    paths: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    paths = sorted(dict.fromkeys(p for p in paths if os.path.exists(p)))

    rows: List[Dict[str, object]] = []
    for path in paths:
        try:
            rows.extend(_aggregate_rows(path, _read_csv(path)))
        except Exception as exc:
            rows.append({"source_file": path, "triage": "drop_or_fix", "reasons": f"read error: {exc}"})

    _write_csv(args.out_csv, rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _write_md(args.out_md, rows)
    counts = Counter(str(r.get("triage", "")) for r in rows)
    print(f"audited={len(rows)} usable={counts.get('usable', 0)} needs={counts.get('needs_rerun_or_caveat', 0)} drop_or_fix={counts.get('drop_or_fix', 0)}")
    print(f"csv={args.out_csv}")
    print(f"md={args.out_md}")


if __name__ == "__main__":
    main()
