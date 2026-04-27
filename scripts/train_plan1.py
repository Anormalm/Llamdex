import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def _as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool-like value, got: {v}")


def parse_args():
    p = argparse.ArgumentParser(description="Plan 1: train semantic evidence injection connectors")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="dtd", choices=["cifar10", "cifar100", "dtd", "oxford_pet", "hospital_text"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--hospital_train_file", type=str, default=None)
    p.add_argument("--hospital_eval_file", type=str, default=None)
    p.add_argument("--task_family", type=str, default="single_image", choices=["single_image", "population"])
    p.add_argument("--evidence_source", type=str, default="text", choices=["vision", "text", "diffusion_future"])
    p.add_argument("--description_file", type=str, default=None,
                   help="External text descriptions (json/jsonl/csv with index->description) for text evidence mode.")
    p.add_argument("--qa_type", type=str, default="label", choices=["label", "yesno", "index", "label_code", "yesno_set2"])
    p.add_argument("--population_output_mode", type=str, default="integer", choices=["integer", "letter"])
    p.add_argument("--population_group_size", type=int, default=16)
    p.add_argument("--population_sigma", type=float, default=0.0)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--expert_kind", type=str, default="classifier", choices=["classifier", "embedding"])
    p.add_argument("--expert_output_mode", type=str, default="logits", choices=["logits", "embedding"])
    p.add_argument("--expert_checkpoint", type=str, default=None)
    p.add_argument("--expert_init_weights", type=str, default="imagenet", choices=["imagenet", "random"])
    p.add_argument("--text_encoder_model_id", type=str, default="distilroberta-base")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--eval_every_steps", type=int, default=100)
    p.add_argument("--answer_only_loss", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_adapters", type=int, default=1, choices=[0, 1])
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--adapter_activation", type=str, default="gelu", choices=["gelu", "relu"])
    p.add_argument("--tune_layernorm", type=int, default=0, choices=[0, 1])
    p.add_argument("--inject_location", type=str, default="layer_input", choices=["layer_input"])
    p.add_argument("--fusion_policy", type=str, default="post_attn_router_parallel", choices=["post_attn_router_parallel"])
    p.add_argument("--use_pre_router", type=_as_bool, default=False)
    p.add_argument("--pre_router_mode", type=str, default="global_feature", choices=["global", "feature", "global_feature"])
    p.add_argument("--pre_router_hidden_dim", type=int, default=128)
    p.add_argument("--task_conditioning", type=_as_bool, default=False)
    p.add_argument("--pre_router_task_embedding_dim", type=int, default=16)
    p.add_argument("--pre_router_max_task_ids", type=int, default=32)
    p.add_argument("--load_in_4bit", type=int, default=0, choices=[0, 1])
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_use_double_quant", type=int, default=1, choices=[0, 1])
    p.add_argument("--bnb_4bit_cpu_offload", type=int, default=0, choices=[0, 1])
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    from src.multimodal.trainers.plan1_trainer import TrainPlan1Args, train_plan1

    args = TrainPlan1Args(
        server_models_path=a.server_models_path,
        model_name=a.model_name,
        run_dir=a.run_dir,
        dataset_name=a.dataset_name,
        data_root=a.data_root,
        hospital_train_file=a.hospital_train_file,
        hospital_eval_file=a.hospital_eval_file,
        task_family=a.task_family,
        evidence_source=a.evidence_source,
        description_file=a.description_file,
        qa_type=a.qa_type,
        population_output_mode=a.population_output_mode,
        population_group_size=a.population_group_size,
        population_sigma=a.population_sigma,
        num_tokens=a.num_tokens,
        layer_to_add=a.layer,
        evidence_dim=a.evidence_dim,
        alpha=a.alpha,
        expert_kind=a.expert_kind,
        expert_output_mode=a.expert_output_mode,
        expert_checkpoint=a.expert_checkpoint,
        expert_init_weights=a.expert_init_weights,
        text_encoder_model_id=a.text_encoder_model_id,
        batch_size=a.batch_size,
        num_epochs=a.num_epochs,
        learning_rate=a.learning_rate,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        eval_every_steps=a.eval_every_steps,
        answer_only_loss=bool(a.answer_only_loss),
        use_adapters=bool(a.use_adapters),
        adapter_bottleneck=a.adapter_bottleneck,
        adapter_dropout=a.adapter_dropout,
        adapter_activation=a.adapter_activation,
        tune_layernorm=bool(a.tune_layernorm),
        inject_location=a.inject_location,
        fusion_policy=a.fusion_policy,
        use_pre_router=a.use_pre_router,
        pre_router_mode=a.pre_router_mode,
        pre_router_hidden_dim=a.pre_router_hidden_dim,
        task_conditioning=a.task_conditioning,
        pre_router_task_embedding_dim=a.pre_router_task_embedding_dim,
        pre_router_max_task_ids=a.pre_router_max_task_ids,
        load_in_4bit=bool(a.load_in_4bit),
        bnb_4bit_compute_dtype=a.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=a.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(a.bnb_4bit_use_double_quant),
        bnb_4bit_cpu_offload=bool(a.bnb_4bit_cpu_offload),
        device_map=a.device_map,
        seed=a.seed,
        device=a.device,
    )
    metrics = train_plan1(args)
    print(metrics)
