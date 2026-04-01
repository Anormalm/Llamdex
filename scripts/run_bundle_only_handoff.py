import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description="Privacy-preserving bundle-only workflow: export frozen bundle, then reload it for inference."
    )
    p.add_argument("--mode", choices=["export", "reload", "both"], default="both")
    p.add_argument(
        "--policy",
        choices=["post_attn_router_parallel", "pre_ffn_router_parallel"],
        default="post_attn_router_parallel",
        help="Injection policy to train/export and then reload.",
    )
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="hf-internal-testing/tiny-random-MistralForCausalLM")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument("--bundle_root", type=str, default="/tmp/llamdex_bundle_only_handoff")
    p.add_argument("--evidence_dim", type=int, default=64)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--expert_type", type=str, default="tabular")
    p.add_argument("--expert_model_id", type=str, default=None)
    p.add_argument("--expert_model_path", type=str, default=None)
    p.add_argument("--expert_output_dim", type=int, default=32)
    p.add_argument("--use_runtime_detector", action="store_true", default=False)
    p.add_argument("--task", type=str, default="finegrained", choices=["finegrained", "strict_yesno", "population", "grounded_generation"])
    p.add_argument("--image_dataset_name", type=str, default="dtd")
    p.add_argument("--train_steps", type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=8)
    p.add_argument("--max_eval_samples", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def _task_spec(a) -> TaskSpec:
    return TaskSpec(
        name=a.task,
        image_dataset_name=a.image_dataset_name,
        train_steps=a.train_steps,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        batch_size=a.batch_size,
        constrained_decoding=True,
    )


def _base_cfg(a) -> dict:
    return dict(
        server_models_path=a.server_models_path,
        model_name=a.model_name,
        data_root=a.data_root,
        seed=a.seed,
        device=a.device,
        local_files_only=bool(a.local_files_only),
        learning_rate=a.learning_rate,
        evidence_dim=a.evidence_dim,
        num_tokens=a.num_tokens,
        layer_idx=a.layer_idx,
        alpha=a.alpha,
        expert_type=a.expert_type,
        expert_model_id=a.expert_model_id,
        expert_model_path=a.expert_model_path,
        expert_output_dim=a.expert_output_dim,
        use_runtime_detector=bool(a.use_runtime_detector),
        fusion_policy=a.policy,
        baseline_modes=["router_parallel"],
        tasks=[_task_spec(a)],
    )


def _bundle_dir(a) -> str:
    return os.path.join(a.bundle_root, a.policy, f"{a.task}_router_parallel")


def run_export(a):
    save_root = os.path.join(a.bundle_root, a.policy)
    cfg = TaskMatrixConfig(
        **_base_cfg(a),
        save_bundle_dir=save_root,
        out_csv=f"/tmp/bundle_only_export.{a.policy}.csv",
        out_json=f"/tmp/bundle_only_export.{a.policy}.json",
    )
    rows = run_task_matrix(cfg)
    print(f"[export] rows={len(rows)} bundle_dir={_bundle_dir(a)}")
    return rows


def run_reload(a):
    cfg = TaskMatrixConfig(
        **_base_cfg(a),
        load_bundle_dir=_bundle_dir(a),
        freeze_loaded_semantic_expert=True,
        freeze_loaded_fusion_policy=False,
        out_csv=f"/tmp/bundle_only_reload.{a.policy}.csv",
        out_json=f"/tmp/bundle_only_reload.{a.policy}.json",
    )
    rows = run_task_matrix(cfg)
    print(f"[reload] rows={len(rows)} bundle_dir={_bundle_dir(a)}")
    return rows


def main():
    a = parse_args()
    os.makedirs(os.path.join(a.bundle_root, a.policy), exist_ok=True)

    if a.mode in {"export", "both"}:
        run_export(a)
    if a.mode in {"reload", "both"}:
        run_reload(a)

    print("Privacy note: this workflow exports and reloads frozen bundles only; it does not transfer per-example z.")


if __name__ == "__main__":
    main()
