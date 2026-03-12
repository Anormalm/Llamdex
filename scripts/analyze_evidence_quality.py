import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.data.cifar_qa import CIFARSingleImageQADataset, collate_single_image
from src.multimodal.evidence import TextEvidenceBuilder, VisionEvidenceBuilder
from src.multimodal.experts import build_vision_expert


def parse_args():
    p = argparse.ArgumentParser(description="Analyze evidence vector quality (z separability) with linear probe.")
    p.add_argument("--mistral_models_path", type=str, default="runs/hf_cache_tiny")
    p.add_argument("--model_name", type=str, default="hf-internal-testing/tiny-random-MistralForCausalLM")
    p.add_argument("--dataset_name", type=str, default="dtd", choices=["cifar10", "cifar100", "dtd", "oxford_pet"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--evidence_source", type=str, default="vision", choices=["vision", "text"])
    p.add_argument("--qa_type", type=str, default="label_code")
    p.add_argument("--description_file", type=str, default=None)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--expert_kind", type=str, default="classifier", choices=["classifier", "embedding"])
    p.add_argument("--expert_output_mode", type=str, default="logits", choices=["logits", "embedding"])
    p.add_argument("--expert_checkpoint", type=str, default=None)
    p.add_argument("--expert_init_weights", type=str, default="imagenet", choices=["imagenet", "random"])
    p.add_argument("--text_encoder_model_id", type=str, default="distilroberta-base")
    p.add_argument("--connectors_path", type=str, default=None, help="Optional checkpoint to load evidence_builder weights.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train_samples", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_csv", type=str, default="runs/evidence_quality_report.csv")
    p.add_argument("--out_json", type=str, default="runs/evidence_quality_report.json")
    return p.parse_args()


def _num_classes(dataset_name: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "dtd": 47, "oxford_pet": 37}[dataset_name]


def _safe_load(module: torch.nn.Module, state_dict: dict):
    cur = module.state_dict()
    keep = {}
    for k, v in state_dict.items():
        if k in cur and hasattr(v, "shape") and hasattr(cur[k], "shape") and tuple(v.shape) == tuple(cur[k].shape):
            keep[k] = v
    module.load_state_dict(keep, strict=False)


def _build_builder(args, device):
    num_classes = _num_classes(args.dataset_name)
    if args.evidence_source == "vision":
        expert = build_vision_expert(
            expert_kind=args.expert_kind,
            checkpoint_path=args.expert_checkpoint,
            num_classes=num_classes,
            init_weights=args.expert_init_weights,
        ).to(device)
        builder = VisionEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            vision_expert=expert,
            expert_output_mode=args.expert_output_mode,
            num_classes=num_classes,
            embedding_dim=512,
            use_embedding_adapter=False,
        )
    else:
        expert = None
        builder = TextEvidenceBuilder(
            evidence_dim=args.evidence_dim,
            text_encoder_model_id=args.text_encoder_model_id,
            cache_dir=args.mistral_models_path,
        )
    builder = builder.to(device)

    if args.connectors_path and os.path.exists(args.connectors_path):
        st = torch.load(args.connectors_path, map_location="cpu")
        if "evidence_builder" in st:
            _safe_load(builder, st["evidence_builder"])
    return builder


def _collect_z(builder, loader, device, evidence_source):
    zs = []
    ys = []
    builder.eval()
    with torch.no_grad():
        for batch in loader:
            if evidence_source == "vision":
                z = builder(batch["images"].to(device))
            else:
                z = builder(
                    {
                        "input_ids": batch["desc_input_ids"].to(device),
                        "attention_mask": batch["desc_attention_mask"].to(device),
                    }
                )
            zs.append(z.detach().float().cpu().numpy())
            ys.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def _class_centroid_margin(z, y):
    classes = np.unique(y)
    centroids = []
    for c in classes:
        centroids.append(z[y == c].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    d = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(-1))
    n = d.shape[0]
    if n <= 1:
        return 0.0
    vals = d[np.triu_indices(n, k=1)]
    return float(vals.mean())


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.mistral_models_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token

    train_ds = CIFARSingleImageQADataset(
        root=args.data_root,
        tokenizer=tok,
        train=True,
        dataset_name=args.dataset_name,
        mode=args.evidence_source,
        qa_type=args.qa_type,
        max_samples=args.max_train_samples,
        description_file=args.description_file,
    )
    eval_ds = CIFARSingleImageQADataset(
        root=args.data_root,
        tokenizer=tok,
        train=False,
        dataset_name=args.dataset_name,
        mode=args.evidence_source,
        qa_type=args.qa_type,
        max_samples=args.max_eval_samples,
        description_file=args.description_file,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_image)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_image)
    builder = _build_builder(args, device)
    z_train, y_train = _collect_z(builder, train_loader, device, args.evidence_source)
    z_eval, y_eval = _collect_z(builder, eval_loader, device, args.evidence_source)

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=1)
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_eval)
    acc = float(accuracy_score(y_eval, y_pred))
    margin = _class_centroid_margin(z_train, y_train)

    row = {
        "dataset": args.dataset_name,
        "evidence_source": args.evidence_source,
        "evidence_dim": int(args.evidence_dim),
        "probe_acc": acc,
        "centroid_margin": margin,
        "train_n": int(len(y_train)),
        "eval_n": int(len(y_eval)),
    }
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(row)


if __name__ == "__main__":
    main()
