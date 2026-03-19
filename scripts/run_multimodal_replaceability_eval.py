import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.framework.runner import AblationConfig, run_ablation_grid, run_single_experiment


def _ints(values):
    return [int(v) for v in values]


def parse_args():
    p = argparse.ArgumentParser(description="Research-grade multimodal replaceability evaluation for prefix injection.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--dataset", type=str, default="dtd", choices=["cifar10", "cifar100", "dtd"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--task", type=str, default="single", choices=["single", "population"])
    p.add_argument("--qa_type", type=str, default="label_code")
    p.add_argument("--population_output_mode", type=str, default="integer", choices=["integer", "letter"])
    p.add_argument("--population_group_size", type=int, default=8)
    p.add_argument(
        "--expert_type",
        type=str,
        default="clip",
        choices=["clip", "siglip", "groundingdino_sam2", "xgboost", "ft_transformer", "tabpfn", "tabular"],
    )
    p.add_argument("--expert_model_id", type=str, default=None)
    p.add_argument("--expert_model_path", type=str, default=None)
    p.add_argument("--expert_output_dim", type=int, default=512)
    p.add_argument("--use_runtime_detector", type=int, default=0, choices=[0, 1])
    p.add_argument("--k", type=int, default=4, help="Prefix token count.")
    p.add_argument("--layer_idx", type=int, default=0, help="Injection layer index.")
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=1024)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--train_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ablation", type=int, default=1, choices=[0, 1])
    p.add_argument("--k_grid", nargs="+", default=["1", "4", "8", "16"])
    p.add_argument("--layer_grid", nargs="+", default=["0"])
    p.add_argument("--evidence_dim_grid", nargs="+", default=["128", "256", "512"])
    p.add_argument("--out_csv", type=str, default="runs/multimodal_replaceability_results.csv")
    p.add_argument("--out_json", type=str, default="runs/multimodal_replaceability_results.json")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = AblationConfig(
        server_models_path=a.server_models_path,
        model_name=a.model_name,
        dataset=a.dataset,
        data_root=a.data_root,
        task=a.task,
        qa_type=a.qa_type,
        population_output_mode=a.population_output_mode,
        population_group_size=a.population_group_size,
        expert_type=a.expert_type,
        expert_model_id=a.expert_model_id,
        expert_model_path=a.expert_model_path,
        expert_output_dim=a.expert_output_dim,
        use_runtime_detector=bool(a.use_runtime_detector),
        k=a.k,
        layer_idx=a.layer_idx,
        evidence_dim=a.evidence_dim,
        alpha=a.alpha,
        batch_size=a.batch_size,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        train_steps=a.train_steps,
        learning_rate=a.learning_rate,
        seed=a.seed,
        device=a.device,
        out_csv=a.out_csv,
        out_json=a.out_json,
    )

    if a.ablation:
        rows = run_ablation_grid(
            base_cfg=cfg,
            k_grid=_ints(a.k_grid),
            layer_grid=_ints(a.layer_grid),
            evidence_dim_grid=_ints(a.evidence_dim_grid),
        )
        print(f"Saved ablation rows: {len(rows)}")
        print(f"CSV: {a.out_csv}")
        print(f"JSON: {a.out_json}")
    else:
        row = run_single_experiment(cfg)
        print(row)


if __name__ == "__main__":
    main()
