import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="URGENT: NL output tuning sweep for grounded answer+rationale.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--expert_type", type=str, default="clip")
    p.add_argument("--expert_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--expert_output_dim", type=int, default=512)
    p.add_argument("--train_steps", type=int, default=60)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_max_new_tokens", type=int, default=24)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    p.add_argument("--top_ps", type=float, nargs="+", default=[0.9, 0.95])
    p.add_argument("--rep_penalties", type=float, nargs="+", default=[1.0, 1.1])
    p.add_argument("--do_samples", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    p.add_argument("--out_csv", type=str, default="runs/nl_output_tuning.csv")
    p.add_argument("--out_json", type=str, default="runs/nl_output_tuning.json")
    p.add_argument("--save_config", type=str, default="runs/nl_output_tuning.config.json")
    return p.parse_args()


def _write_csv(rows, out_csv):
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


def main():
    a = parse_args()
    grid = list(itertools.product(a.temperatures, a.top_ps, a.rep_penalties, a.do_samples))
    all_rows = []
    out_dir = os.path.dirname(a.out_csv) or "."
    for i, (temp, top_p, rep, do_sample) in enumerate(grid):
        task = TaskSpec(
            name="grounded_generation",
            train_steps=a.train_steps,
            max_train_samples=a.max_train_samples,
            max_eval_samples=a.max_eval_samples,
            batch_size=a.batch_size,
            constrained_decoding=True,
            gen_max_new_tokens=a.gen_max_new_tokens,
            gen_temperature=float(temp),
            gen_top_p=float(top_p),
            gen_repetition_penalty=float(rep),
            gen_do_sample=bool(do_sample),
        )
        cfg = TaskMatrixConfig(
            server_models_path=a.server_models_path,
            model_name=a.model_name,
            data_root=a.data_root,
            seed=a.seed + i,
            device=a.device,
            learning_rate=a.learning_rate,
            evidence_dim=a.evidence_dim,
            num_tokens=a.num_tokens,
            layer_idx=a.layer_idx,
            alpha=a.alpha,
            expert_type=a.expert_type,
            expert_model_id=a.expert_model_id,
            expert_output_dim=a.expert_output_dim,
            tasks=[task],
            out_csv=os.path.join(out_dir, "_tmp_nl_tuning.csv"),
            out_json=os.path.join(out_dir, "_tmp_nl_tuning.json"),
        )
        rows = run_task_matrix(cfg)
        row = dict(rows[0]) if rows else {"status": "error"}
        row.update(
            {
                "temperature": float(temp),
                "top_p": float(top_p),
                "repetition_penalty": float(rep),
                "do_sample": int(do_sample),
            }
        )
        all_rows.append(row)
        print(
            f"[nl_tuning] {i+1}/{len(grid)} temp={temp} top_p={top_p} rep={rep} sample={do_sample} "
            f"acc={row.get('accuracy')} fmt={row.get('format_compliance')} rat={row.get('rationale_consistency')}"
        )

    _write_csv(all_rows, a.out_csv)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    with open(a.save_config, "w", encoding="utf-8") as f:
        json.dump(
            {
                "server_models_path": a.server_models_path,
                "model_name": a.model_name,
                "data_root": a.data_root,
                "device": a.device,
                "seed": a.seed,
                "learning_rate": a.learning_rate,
                "evidence_dim": a.evidence_dim,
                "num_tokens": a.num_tokens,
                "layer_idx": a.layer_idx,
                "alpha": a.alpha,
                "expert_type": a.expert_type,
                "expert_model_id": a.expert_model_id,
                "expert_output_dim": a.expert_output_dim,
                "train_steps": a.train_steps,
                "max_train_samples": a.max_train_samples,
                "max_eval_samples": a.max_eval_samples,
                "batch_size": a.batch_size,
                "gen_max_new_tokens": a.gen_max_new_tokens,
                "temperatures": a.temperatures,
                "top_ps": a.top_ps,
                "rep_penalties": a.rep_penalties,
                "do_samples": a.do_samples,
                "out_csv": a.out_csv,
                "out_json": a.out_json,
            },
            f,
            indent=2,
        )
    print(f"Saved: {a.out_csv}")
    print(f"Saved: {a.out_json}")
    print(f"Config: {a.save_config}")


if __name__ == "__main__":
    main()
