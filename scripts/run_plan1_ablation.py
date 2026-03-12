import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from transformers import AutoConfig


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
    p.add_argument("--inject_locations", type=str, nargs="+", default=["post_attn", "pre_ffn", "post_ffn"])
    p.add_argument("--use_adapters", type=int, default=0, choices=[0, 1])
    p.add_argument("--adapter_bottlenecks", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--tune_layernorm_grid", type=int, nargs="+", default=[0])
    return p.parse_args()


def main():
    args = parse_args()
    Path(os.path.dirname(args.output_csv) or ".").mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=args.mistral_models_path)
    max_layers = int(getattr(cfg, "num_hidden_layers", 0))
    valid_layers = [x for x in args.layers if 0 <= int(x) < max_layers]
    if not valid_layers:
        raise RuntimeError(f"No valid layers in --layers for model {args.model_name}; num_hidden_layers={max_layers}")
    if len(valid_layers) != len(args.layers):
        print(f"[ablation] skipping invalid layers; valid={valid_layers} num_hidden_layers={max_layers}", flush=True)

    rows = []
    best_row = None
    adapter_grid = args.adapter_bottlenecks if args.use_adapters else [64]
    for t in args.tokens_grid:
        for k in valid_layers:
            for loc in args.inject_locations:
                for bneck in adapter_grid:
                    for tune_ln in args.tune_layernorm_grid:
                        run_dir = os.path.join(
                            args.base_run_dir,
                            f"tokens{t}_layer{k}_{loc}_adapt{args.use_adapters}_b{bneck}_ln{tune_ln}",
                        )
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
                            "--inject_location",
                            loc,
                            "--use_adapters",
                            str(args.use_adapters),
                            "--adapter_bottleneck",
                            str(bneck),
                            "--tune_layernorm",
                            str(tune_ln),
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
                            "--inject_location",
                            loc,
                            "--use_adapters",
                            str(args.use_adapters),
                            "--adapter_bottleneck",
                            str(bneck),
                            "--tune_layernorm",
                            str(tune_ln),
                        ]
                        best_ckpt = os.path.join(run_dir, "best_connectors.pt")
                        last_ckpt = os.path.join(run_dir, "last_connectors.pt")
                        connectors_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
                        eval_cmd.extend(["--connectors_path", connectors_path])
                        proc = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
                        out = (proc.stdout or "").strip()
                        m = re.search(r"accuracy['\"]?\s*:\s*([0-9]*\.?[0-9]+)", out)
                        acc = float(m.group(1)) if m else -1.0
                        rows.append(
                            {
                                "num_tokens": t,
                                "layer": k,
                                "inject_location": loc,
                                "use_adapters": args.use_adapters,
                                "adapter_bottleneck": bneck,
                                "tune_layernorm": tune_ln,
                                "accuracy": acc,
                                "raw_eval_output": out,
                            }
                        )
                        if best_row is None or acc > best_row["accuracy"]:
                            best_row = dict(rows[-1])

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_tokens",
                "layer",
                "inject_location",
                "use_adapters",
                "adapter_bottleneck",
                "tune_layernorm",
                "accuracy",
                "raw_eval_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if best_row is not None:
        best_path = os.path.splitext(args.output_csv)[0] + ".best.csv"
        with open(best_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
            writer.writeheader()
            writer.writerow(best_row)
        print(f"[ablation] best={best_row}")
        print(f"[ablation] best_csv={best_path}")


if __name__ == "__main__":
    main()
