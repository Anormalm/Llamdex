import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="URGENT: run grounded answer+rationale task.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_dataset_name", type=str, default="oxford_pet")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--lr_schedule", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--expert_type", type=str, default="clip")
    p.add_argument("--expert_model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--expert_model_path", type=str, default=None)
    p.add_argument("--expert_output_dim", type=int, default=512)
    p.add_argument("--use_runtime_detector", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--fusion_policy", type=str, default="post_attn_router_parallel")
    p.add_argument("--baseline_modes", type=str, nargs="+", default=["router_parallel"])
    p.add_argument("--save_bundle_dir", type=str, default=None)
    p.add_argument("--load_bundle_dir", type=str, default=None)
    p.add_argument("--freeze_loaded_semantic_expert", type=int, default=1, choices=[0, 1])
    p.add_argument("--freeze_loaded_fusion_policy", type=int, default=0, choices=[0, 1])
    p.add_argument("--train_steps", type=int, default=80)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--repeat_seeds", type=int, nargs="*", default=[])
    p.add_argument("--gen_max_new_tokens", type=int, default=24)
    p.add_argument("--gen_temperature", type=float, default=0.7)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument("--gen_repetition_penalty", type=float, default=1.1)
    p.add_argument("--gen_do_sample", type=int, default=0, choices=[0, 1])
    p.add_argument("--min_rationale_chars", type=int, default=12)
    p.add_argument("--min_rationale_keyword_score", type=float, default=0.6)
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/urgent_grounded_rationale.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/urgent_grounded_rationale.json")
    p.add_argument("--save_config", type=str, default="/disk1/lfhu/runs/urgent_grounded_rationale.config.json")
    return p.parse_args()


def main():
    a = parse_args()
    task = TaskSpec(
        name="grounded_generation",
        image_dataset_name=a.image_dataset_name,
        train_steps=a.train_steps,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        batch_size=a.batch_size,
        constrained_decoding=True,
        gen_max_new_tokens=a.gen_max_new_tokens,
        gen_temperature=a.gen_temperature,
        gen_top_p=a.gen_top_p,
        gen_repetition_penalty=a.gen_repetition_penalty,
        gen_do_sample=bool(a.gen_do_sample),
        min_rationale_chars=a.min_rationale_chars,
        min_rationale_keyword_score=a.min_rationale_keyword_score,
        strict_answer_code=True,
    )
    cfg = TaskMatrixConfig(
        server_models_path=a.server_models_path,
        model_name=a.model_name,
        data_root=a.data_root,
        seed=a.seed,
        device=a.device,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        grad_clip_norm=a.grad_clip_norm,
        lr_schedule=a.lr_schedule,
        lr_warmup_steps=a.lr_warmup_steps,
        evidence_dim=a.evidence_dim,
        num_tokens=a.num_tokens,
        layer_idx=a.layer_idx,
        alpha=a.alpha,
        expert_type=a.expert_type,
        expert_model_id=a.expert_model_id,
        expert_model_path=a.expert_model_path,
        expert_output_dim=a.expert_output_dim,
        use_runtime_detector=a.use_runtime_detector,
        local_files_only=a.local_files_only,
        fusion_policy=a.fusion_policy,
        baseline_modes=a.baseline_modes,
        save_bundle_dir=a.save_bundle_dir,
        load_bundle_dir=a.load_bundle_dir,
        freeze_loaded_semantic_expert=bool(a.freeze_loaded_semantic_expert),
        freeze_loaded_fusion_policy=bool(a.freeze_loaded_fusion_policy),
        tasks=[task],
        repeat_seeds=a.repeat_seeds,
        aggregate_seed_metrics=True,
        out_csv=a.out_csv,
        out_json=a.out_json,
    )
    os.makedirs(os.path.dirname(a.save_config) or ".", exist_ok=True)
    with open(a.save_config, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    rows = run_task_matrix(cfg)
    print(f"Completed {len(rows)} grounded task rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")
    print(f"Config: {a.save_config}")


if __name__ == "__main__":
    main()
