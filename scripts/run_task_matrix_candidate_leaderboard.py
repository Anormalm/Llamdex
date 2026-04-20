import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import load_task_matrix_config, run_task_matrix


def _candidate_name(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".remote.json", ".json"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def _metric_direction(metric_name: str) -> str:
    return "min" if str(metric_name).strip().lower() == "mae" else "max"


def _metric_value(row: Dict) -> float:
    metric_name = str(row.get("metric_name") or "").strip().lower()
    value = row.get("metric")
    if isinstance(value, (int, float)):
        return float(value)
    if metric_name and isinstance(row.get(metric_name), (int, float)):
        return float(row[metric_name])
    if metric_name and row.get(metric_name) is not None:
        try:
            return float(row[metric_name])
        except Exception:
            return 0.0
    try:
        return float(row.get("metric", 0.0))
    except Exception:
        return 0.0


def _usable_rows(rows: List[Dict], aggregate_only: bool) -> List[Dict]:
    out = []
    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        if aggregate_only and row.get("seed") not in {"aggregate", None}:
            continue
        out.append(row)
    return out


def _rank_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (str(row.get("task", "")), str(row.get("dataset", "")))
        grouped.setdefault(key, []).append(row)

    ranked: List[Dict] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        metric_name = str(group[0].get("metric_name") or "accuracy")
        reverse = _metric_direction(metric_name) != "min"
        group.sort(key=_metric_value, reverse=reverse)
        for idx, row in enumerate(group, start=1):
            ranked_row = dict(row)
            ranked_row["leaderboard_rank"] = idx
            ranked_row["leaderboard_metric"] = _metric_value(row)
            ranked_row["leaderboard_direction"] = _metric_direction(metric_name)
            ranked.append(ranked_row)
    return ranked


def _write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _render_md(rows: List[Dict], source_configs: List[str]) -> str:
    winners = [row for row in rows if int(row.get("leaderboard_rank", 999999)) == 1]
    lines = []
    lines.append("# Task-Matrix Candidate Leaderboard")
    lines.append("")
    lines.append("## Source Configs")
    for path in source_configs:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Winners")
    lines.append("| task | dataset | candidate | baseline | fusion_policy | metric | value | direction |")
    lines.append("|---|---|---|---|---|---|---:|---|")
    for row in winners:
        lines.append(
            f"| {row.get('task', '')} | {row.get('dataset', '')} | {row.get('candidate_name', '')} | {row.get('baseline_mode', '')} | {row.get('fusion_policy', '')} | {row.get('metric_name', '')} | {row.get('leaderboard_metric', 0.0):.4f} | {row.get('leaderboard_direction', '')} |"
        )
    lines.append("")
    lines.append("## Full Ranking")
    lines.append("| rank | task | dataset | candidate | baseline | metric | value | status |")
    lines.append("|---:|---|---|---|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.get('leaderboard_rank', '')} | {row.get('task', '')} | {row.get('dataset', '')} | {row.get('candidate_name', '')} | {row.get('baseline_mode', '')} | {row.get('metric_name', '')} | {row.get('leaderboard_metric', 0.0):.4f} | {row.get('status', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple task-matrix configs and emit a ranked leaderboard.")
    p.add_argument("--configs", nargs="+", required=True, help="Task-matrix config JSON files to compare.")
    p.add_argument("--aggregate_only", action="store_true", help="Rank only aggregate-seed rows when present.")
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/task_matrix_candidate_leaderboard.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/task_matrix_candidate_leaderboard.json")
    p.add_argument("--out_md", type=str, default="/disk1/lfhu/runs/task_matrix_candidate_leaderboard.md")
    return p.parse_args()


def main():
    args = parse_args()
    merged_rows: List[Dict] = []
    for path in args.configs:
        cfg = load_task_matrix_config(path)
        rows = run_task_matrix(cfg)
        for row in _usable_rows(rows, aggregate_only=bool(args.aggregate_only)):
            tagged = dict(row)
            tagged["candidate_config"] = path
            tagged["candidate_name"] = _candidate_name(path)
            merged_rows.append(tagged)

    ranked_rows = _rank_rows(merged_rows)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    _write_csv(args.out_csv, ranked_rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": ranked_rows, "source_configs": args.configs}, f, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(ranked_rows, args.configs))

    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")
    print(f"MD: {args.out_md}")
    print(f"Rows: {len(ranked_rows)}")


if __name__ == "__main__":
    main()
