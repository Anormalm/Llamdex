import argparse
import csv
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Policy ablation: pre_attn_overwrite vs post_attn_router_parallel.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--run_root", type=str, default="/disk1/lfhu/runs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--max_train_samples", type=int, default=256)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def _task_specs(a):
    return [
        TaskSpec(name="finegrained", train_steps=a.train_steps, max_train_samples=a.max_train_samples, max_eval_samples=a.max_eval_samples, batch_size=a.batch_size),
        TaskSpec(name="strict_yesno", train_steps=a.train_steps, max_train_samples=a.max_train_samples, max_eval_samples=a.max_eval_samples, batch_size=a.batch_size),
        TaskSpec(name="population", train_steps=a.train_steps, max_train_samples=a.max_train_samples, max_eval_samples=a.max_eval_samples, batch_size=a.batch_size),
        TaskSpec(name="grounded_generation", train_steps=max(1, a.train_steps // 2), max_train_samples=a.max_train_samples, max_eval_samples=a.max_eval_samples, batch_size=a.batch_size),
    ]


def _write_rows(rows, out_csv, out_json):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _score_task(row):
    if row.get("task") == "population":
        return -float(row.get("mae", 1.0))
    return float(row.get("accuracy", row.get("metric", 0.0)))


def _build_status_report(rows, out_md):
    by = {}
    for r in rows:
        by.setdefault((r.get("task"), r.get("fusion_policy")), r)
    tasks = sorted(set(r.get("task") for r in rows if r.get("task")))
    better = 0
    lines = []
    lines.append(f"# Policy Ablation Status ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    lines.append("")
    lines.append("| task | pre_attn_overwrite | post_attn_router_parallel | winner |")
    lines.append("|---|---:|---:|---|")
    for t in tasks:
        r_pre = by.get((t, "pre_attn_overwrite"), {})
        r_post = by.get((t, "post_attn_router_parallel"), {})
        s_pre = _score_task(r_pre) if r_pre else float("-inf")
        s_post = _score_task(r_post) if r_post else float("-inf")
        win = "post" if s_post >= s_pre else "pre"
        if win == "post":
            better += 1
        lines.append(f"| {t} | {s_pre:.4f} | {s_post:.4f} | {win} |")
    lines.append("")
    lines.append(f"- post_attn >= pre_attn on {better} tasks.")
    lines.append("- llm_only/api_sota rows are included only when providers/keys are available.")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return better


def main():
    a = parse_args()
    all_rows = []
    for pol in ("pre_attn_overwrite", "post_attn_router_parallel"):
        cfg = TaskMatrixConfig(
            server_models_path=a.server_models_path,
            model_name=a.model_name,
            data_root=a.data_root,
            seed=a.seed,
            device=a.device,
            learning_rate=2e-4,
            evidence_dim=128,
            num_tokens=4,
            layer_idx=0,
            alpha=1.0,
            expert_type="tabular",
            expert_output_dim=32,
            fusion_policy=pol,
            tasks=_task_specs(a),
            out_csv=os.path.join(a.run_root, f"_tmp_policy_{pol}.csv"),
            out_json=os.path.join(a.run_root, f"_tmp_policy_{pol}.json"),
        )
        rows = run_task_matrix(cfg)
        all_rows.extend(rows)

    all_rows.append(
        {
            "task": "llm_only",
            "fusion_policy": "n/a",
            "status": "available_not_run",
            "metric": 0.0,
            "note": "Use src/multimodal/baselines/suite.py for full llm_only baseline sweeps.",
        }
    )

    # Optional API SOTA row when key exists and API suite is available.
    if os.getenv("OPENAI_API_KEY"):
        all_rows.append(
            {
                "task": "api_sota",
                "fusion_policy": "n/a",
                "status": "available_not_run",
                "metric": 0.0,
                "note": "OPENAI_API_KEY present; run scripts/run_sota_baselines_urgent.py separately for full results.",
            }
        )

    out_csv = os.path.join(a.run_root, "policy_ablation.csv")
    out_json = os.path.join(a.run_root, "policy_ablation.json")
    out_md = os.path.join(a.run_root, "status_report.md")
    _write_rows(all_rows, out_csv, out_json)
    better = _build_status_report(all_rows, out_md)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    print(f"post_attn >= pre_attn tasks: {better}")


if __name__ == "__main__":
    main()
