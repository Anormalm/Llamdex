import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_rows(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
    return []


def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _canon(row: Dict, source: str, default_family: str, default_name: str) -> Dict:
    metric_name = str(row.get("metric_name") or ("mae" if "mae" in row else "accuracy"))
    metric = _as_float(
        row.get(
            "metric",
            row.get(
                "metric_mean",
                row.get(
                    "metric_value",
                    row.get(
                        metric_name,
                        row.get(f"{metric_name}_mean", 0.0),
                    ),
                ),
            ),
        )
    )
    return {
        "source": source,
        "benchmark_family": row.get("benchmark_family", default_family),
        "baseline_name": row.get("baseline_name", row.get("baseline_mode", row.get("model_type", default_name))),
        "task": row.get("task", row.get("task_family", "unknown")),
        "dataset": row.get("dataset", "unknown"),
        "metric_name": metric_name,
        "metric": metric,
        "status": row.get("status", "ok"),
        "seed": row.get("seed", ""),
        "notes": row.get("notes", row.get("baseline_note", "")),
    }


def merge(args):
    out: List[Dict] = []

    for r in _load_rows(args.task_matrix_json):
        out.append(_canon(r, "task_matrix", "llamdex", "router_parallel"))

    for r in _load_rows(args.local_baseline_csv):
        out.append(_canon(r, "local_baseline", "architecture", "local_arch"))

    for r in _load_rows(args.api_summary_csv):
        out.append(_canon(r, "api_summary", "api", "api_suite"))

    for r in _load_rows(args.policy_ablation_csv):
        out.append(_canon(r, "policy_ablation", "llamdex", "policy_ablation"))

    for r in _load_rows(args.lora_template_csv):
        out.append(_canon(r, "lora_template", "architecture", "lora_finetune"))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = ["source", "benchmark_family", "baseline_name", "task", "dataset", "metric_name", "metric", "status", "seed", "notes"]
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)

    summary = {
        "rows": len(out),
        "out_csv": args.out_csv,
        "task_matrix_rows": len(_load_rows(args.task_matrix_json)),
        "local_rows": len(_load_rows(args.local_baseline_csv)),
        "api_rows": len(_load_rows(args.api_summary_csv)),
        "policy_rows": len(_load_rows(args.policy_ablation_csv)),
        "lora_rows": len(_load_rows(args.lora_template_csv)),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Merge benchmark artifacts into a single slide-ready CSV.")
    p.add_argument("--task_matrix_json", default="/disk1/lfhu/runs/task_matrix_parity.dtd_oxford.remote.json")
    p.add_argument("--local_baseline_csv", default="/disk1/lfhu/runs/local_arch_baselines.remote.oxford_pet.label_code.csv")
    p.add_argument("--api_summary_csv", default="/disk1/lfhu/runs/api_baselines.remote.dtd.summary.csv")
    p.add_argument("--policy_ablation_csv", default="/disk1/lfhu/runs/policy_ablation.csv")
    p.add_argument("--lora_template_csv", default="/home/anormalm/Llamdex/conf/lora_baseline_results.template.csv")
    p.add_argument("--out_csv", default="/disk1/lfhu/runs/slide_results_merged.csv")
    p.add_argument("--out_json", default="/disk1/lfhu/runs/slide_results_merged.summary.json")
    return p.parse_args()


if __name__ == "__main__":
    merge(parse_args())
