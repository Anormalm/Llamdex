import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run Plan 1 ablations for num_tokens and injection layer")
    p.add_argument("--output_csv", type=str, default="runs/plan1_ablation_summary.csv")
    p.add_argument("--mistral_models_path", type=str, default="model/llm")
    p.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--tokens_grid", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16, 24])
    p.add_argument("--base_run_dir", type=str, default="runs/plan1_ablation")
    p.add_argument("--train_samples", type=int, default=1024)
    p.add_argument("--eval_samples", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    Path(os.path.dirname(args.output_csv) or ".").mkdir(parents=True, exist_ok=True)

    rows = []
    for t in args.tokens_grid:
        for k in args.layers:
            run_dir = os.path.join(args.base_run_dir, f"tokens{t}_layer{k}")
            train_cmd = [
                sys.executable,
                "scripts/train_plan1.py",
                "--mistral_models_path",
                args.mistral_models_path,
                "--model_name",
                args.model_name,
                "--run_dir",
                run_dir,
                "--task_family",
                "single_image",
                "--evidence_source",
                "vision",
                "--qa_type",
                "label",
                "--num_tokens",
                str(t),
                "--layer",
                str(k),
                "--max_train_samples",
                str(args.train_samples),
                "--max_eval_samples",
                str(args.eval_samples),
                "--device",
                args.device,
            ]
            subprocess.run(train_cmd, check=True)

            eval_cmd = [
                sys.executable,
                "scripts/eval_plan1.py",
                "--mistral_models_path",
                args.mistral_models_path,
                "--model_name",
                args.model_name,
                "--task_family",
                "single_image",
                "--evidence_source",
                "vision",
                "--qa_type",
                "label",
                "--num_tokens",
                str(t),
                "--layer",
                str(k),
                "--max_eval_samples",
                str(args.eval_samples),
                "--baseline",
                "injection",
                "--device",
                args.device,
            ]
            best_ckpt = os.path.join(run_dir, "best_connectors.pt")
            last_ckpt = os.path.join(run_dir, "last_connectors.pt")
            connectors_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
            eval_cmd.extend(["--connectors_path", connectors_path])
            proc = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            rows.append({"num_tokens": t, "layer": k, "raw_eval_output": proc.stdout.strip()})

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_tokens", "layer", "raw_eval_output"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
