import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Server-side run using uploaded client expert bundle.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--load_bundle_dir", type=str, required=True, help="Path to uploaded bundle directory.")
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/server_bundle_eval.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/server_bundle_eval.json")
    p.add_argument(
        "--baseline_modes",
        type=str,
        nargs="+",
        default=["overwrite", "router_parallel", "llm_only", "text_prompt_only"],
        help="Comparable task-matrix baselines. expert_only is intentionally unsupported.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
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
        load_bundle_dir=a.load_bundle_dir,
        baseline_modes=a.baseline_modes,
        tasks=[
            TaskSpec(name="finegrained", train_steps=1, max_train_samples=32, max_eval_samples=64, batch_size=8),
            TaskSpec(name="strict_yesno", train_steps=1, max_train_samples=32, max_eval_samples=64, batch_size=8),
            TaskSpec(name="population", train_steps=1, max_train_samples=32, max_eval_samples=64, batch_size=8),
            TaskSpec(name="grounded_generation", train_steps=1, max_train_samples=32, max_eval_samples=64, batch_size=8),
        ],
        out_csv=a.out_csv,
        out_json=a.out_json,
    )
    rows = run_task_matrix(cfg)
    print(rows)
    print(f"CSV: {a.out_csv}")
    print(f"JSON: {a.out_json}")


if __name__ == "__main__":
    main()
