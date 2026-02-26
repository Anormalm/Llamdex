import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.trainers.plan1_trainer import TrainPlan1Args, train_plan1


def parse_args():
    p = argparse.ArgumentParser(description="Plan 1: train semantic evidence injection connectors")
    p.add_argument("--mistral_models_path", type=str, default="model/llm")
    p.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="dtd", choices=["cifar10", "cifar100", "dtd"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--task_family", type=str, default="single_image", choices=["single_image", "population"])
    p.add_argument("--evidence_source", type=str, default="vision", choices=["vision", "text", "diffusion_future"])
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    args = TrainPlan1Args(
        mistral_models_path=a.mistral_models_path,
        model_name=a.model_name,
        run_dir=a.run_dir,
        dataset_name=a.dataset_name,
        data_root=a.data_root,
        task_family=a.task_family,
        evidence_source=a.evidence_source,
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
        seed=a.seed,
        device=a.device,
    )
    metrics = train_plan1(args)
    print(metrics)
