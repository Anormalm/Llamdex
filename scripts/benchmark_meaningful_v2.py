import argparse
import csv
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.eval.plan1_eval import EvalPlan1Args, evaluate_plan1


def run_case(args: EvalPlan1Args):
    t0 = time.time()
    metrics = evaluate_plan1(args)
    dt = time.time() - t0
    return metrics, dt


def main():
    p = argparse.ArgumentParser(description="Benchmark meaningful Plan-1 tasks (v2)")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--dataset_name", type=str, default="dtd", choices=["cifar10", "cifar100", "dtd"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--expert_checkpoint", type=str, default="runs/experts/dtd_resnet18_best.pt")
    p.add_argument("--connectors_yesno_set2", type=str, default="runs/plan1_dtd_yesno_set2_vision_v2/best_connectors.pt")
    p.add_argument("--connectors_label_code", type=str, default="runs/plan1_dtd_label_code_vision_v2/best_connectors.pt")
    p.add_argument("--connectors_population", type=str, default="runs/plan1_dtd_population_vision_v1/best_connectors.pt")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_eval_samples", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_csv", type=str, default="runs/benchmark_meaningful_v2_dtd.csv")
    p.add_argument("--out_json", type=str, default="runs/benchmark_meaningful_v2_dtd.json")
    p.add_argument("--use_adapters", type=int, default=1, choices=[0, 1])
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--adapter_activation", type=str, default="gelu", choices=["gelu", "relu"])
    p.add_argument("--tune_layernorm", type=int, default=0, choices=[0, 1])
    p.add_argument("--inject_location", type=str, default="post_attn", choices=["layer_input", "post_attn", "pre_ffn", "post_ffn"])
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)

    task_specs = [
        {
            "task_name": "single_yesno_set2_vision",
            "task_family": "single_image",
            "evidence_source": "vision",
            "qa_type": "yesno_set2",
            "population_output_mode": "integer",
            "population_group_size": 8,
            "num_tokens": 4,
            "layer_to_add": 0,
            "evidence_dim": 256,
            "alpha": 1.0,
            "expert_kind": "classifier",
            "expert_output_mode": "logits",
            "baselines": ["vision_only", "llm_only", "text_prompt", "injection"],
            "constrain_llm_only_outputs": True,
            "connectors_path": a.connectors_yesno_set2,
        },
        {
            "task_name": "single_label_code_vision",
            "task_family": "single_image",
            "evidence_source": "vision",
            "qa_type": "label_code",
            "population_output_mode": "integer",
            "population_group_size": 8,
            "num_tokens": 4,
            "layer_to_add": 0,
            "evidence_dim": 256,
            "alpha": 1.0,
            "expert_kind": "classifier",
            "expert_output_mode": "logits",
            "baselines": ["vision_only", "llm_only", "text_prompt", "injection"],
            "constrain_llm_only_outputs": True,
            "connectors_path": a.connectors_label_code,
        },
        {
            "task_name": "population_fraction_vision",
            "task_family": "population",
            "evidence_source": "vision",
            "qa_type": "label",
            "population_output_mode": "integer",
            "population_group_size": 8,
            "num_tokens": 4,
            "layer_to_add": 0,
            "evidence_dim": 256,
            "alpha": 1.0,
            "expert_kind": "classifier",
            "expert_output_mode": "logits",
            "baselines": ["vision_only", "llm_only", "text_prompt", "injection"],
            "constrain_llm_only_outputs": True,
            "connectors_path": a.connectors_population,
        },
    ]

    rows = []
    for spec in task_specs:
        for baseline in spec["baselines"]:
            eval_args = EvalPlan1Args(
                server_models_path=a.server_models_path,
                model_name=a.model_name,
                connectors_path=spec["connectors_path"] if baseline == "injection" else None,
                dataset_name=a.dataset_name,
                data_root=a.data_root,
                task_family=spec["task_family"],
                evidence_source=spec["evidence_source"],
                qa_type=spec["qa_type"],
                population_output_mode=spec["population_output_mode"],
                population_group_size=spec["population_group_size"],
                population_sigma=0.0,
                num_tokens=spec["num_tokens"],
                layer_to_add=spec["layer_to_add"],
                evidence_dim=spec["evidence_dim"],
                alpha=spec["alpha"],
                expert_kind=spec["expert_kind"],
                expert_output_mode=spec["expert_output_mode"],
                expert_checkpoint=a.expert_checkpoint,
                text_encoder_model_id="distilroberta-base",
                batch_size=a.batch_size,
                max_eval_samples=a.max_eval_samples,
                baseline=baseline,
                device=a.device,
                constrain_llm_only_outputs=spec["constrain_llm_only_outputs"],
                use_adapters=bool(a.use_adapters),
                adapter_bottleneck=a.adapter_bottleneck,
                adapter_dropout=a.adapter_dropout,
                adapter_activation=a.adapter_activation,
                tune_layernorm=bool(a.tune_layernorm),
                inject_location=a.inject_location,
            )
            metrics, dt = run_case(eval_args)
            row = {"task_name": spec["task_name"], "baseline": baseline, "wall_time_s": round(dt, 2)}
            row.update(metrics)
            rows.append(row)
            print(row)

    all_cols = []
    for r in rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        w.writerows(rows)

    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {a.out_csv}")
    print(f"Saved: {a.out_json}")


if __name__ == "__main__":
    main()
