import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Train connector/router on public data and export uploadable expert bundle.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--run_root", type=str, default="/disk1/lfhu/runs")
    p.add_argument("--fusion_policy", type=str, default="post_attn_router_parallel", choices=["pre_attn_overwrite", "post_attn_router_parallel"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_steps", type=int, default=40)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    bundle_root = os.path.join(a.run_root, "bundles")
    out_csv = os.path.join(a.run_root, f"public_bundle_train_{a.fusion_policy}.csv")
    out_json = os.path.join(a.run_root, f"public_bundle_train_{a.fusion_policy}.json")
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
        fusion_policy=a.fusion_policy,
        save_bundle_dir=bundle_root,
        label_schema={"task": "finegrained_pet_codes"},
        normalization_stats={"note": "expert encoder handles normalization internally"},
        tasks=[
            TaskSpec(
                name="finegrained",
                train_steps=a.train_steps,
                max_train_samples=a.max_train_samples,
                max_eval_samples=a.max_eval_samples,
                batch_size=a.batch_size,
            )
        ],
        out_csv=out_csv,
        out_json=out_json,
    )
    rows = run_task_matrix(cfg)
    print(rows)
    print(f"Bundle root: {bundle_root}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
