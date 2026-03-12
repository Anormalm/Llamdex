from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.eval.plan1_eval import EvalPlan1Args, evaluate_plan1
from src.multimodal.experts import build_vision_expert
from src.multimodal.tasks.parsing import parse_yes_no_answer, population_bin_to_fraction_midpoint, population_fraction_to_bin
from src.multimodal.task_matrix.metrics import macro_f1_from_ints, mean_absolute_error
from src.multimodal.utils.repro import set_seed


@dataclass
class FrozenVLMConfig:
    model_type: str  # llava_next|qwen25_vl|internvl
    model_id: str
    enabled: bool = True


@dataclass
class BaselineSuiteConfig:
    mistral_models_path: str = "model/llm"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    dataset_name: str = "dtd"
    data_root: str = "./data"
    task_family: str = "single_image"  # single_image|population
    qa_type: str = "label_code"
    population_output_mode: str = "integer"
    population_group_size: int = 8
    batch_size: int = 8
    max_eval_samples: int = 256
    device: str = "cuda"
    seed: int = 42
    # expert configs
    expert_checkpoint: Optional[str] = None
    expert_init_weights: str = "imagenet"
    expert_kind: str = "classifier"
    expert_output_mode: str = "logits"
    # two-stage caption
    caption_model_id: str = "Salesforce/blip-image-captioning-base"
    enable_llm_only: bool = True
    enable_expert_only: bool = True
    enable_two_stage: bool = True
    enable_rag: bool = True
    # rag
    rag_db_size: int = 256
    # frozen VLM
    frozen_vlms: List[FrozenVLMConfig] = field(default_factory=list)
    # output
    out_csv: str = "runs/baseline_suite_results.csv"
    out_json: str = "runs/baseline_suite_results.json"
    # optional injection row for direct comparison
    include_injection: bool = False
    connectors_path: Optional[str] = None
    num_tokens: int = 4
    layer_to_add: int = 0
    evidence_dim: int = 256
    alpha: float = 1.0
    use_adapters: bool = False
    adapter_bottleneck: int = 64
    adapter_dropout: float = 0.0
    adapter_activation: str = "gelu"
    tune_layernorm: bool = False
    inject_location: str = "post_attn"
    enforce_checkpoint_compat: bool = True


def _device(name: str):
    return torch.device(name if torch.cuda.is_available() and name.startswith("cuda") else "cpu")


def _num_classes(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "dtd":
        return 47
    if dataset_name == "oxford_pet":
        return 37
    raise ValueError(dataset_name)


def _single_token_ids(tokenizer, text: str):
    ids = set()
    for v in (text, text.lower(), text.upper()):
        for prefix in (" ", ""):
            t = tokenizer.encode(prefix + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(int(t[0]))
    return ids


def _allowed_token_ids(task_family: str, qa_mode: str, tokenizer, dataset) -> List[int]:
    ids = set()
    if task_family == "single_image":
        if qa_mode in {"yesno", "yesno_set2"}:
            ids.update(_single_token_ids(tokenizer, "Yes"))
            ids.update(_single_token_ids(tokenizer, "No"))
        elif qa_mode == "label_code" and getattr(dataset, "code_to_token_id", None) is not None:
            ids.update(int(v) for v in dataset.code_to_token_id.values())
        elif qa_mode == "index":
            for i in range(len(dataset.class_names)):
                ids.update(_single_token_ids(tokenizer, str(i)))
        else:
            for i in range(min(26, len(dataset.class_names))):
                ids.update(_single_token_ids(tokenizer, chr(ord("A") + i)))
    else:
        for i in range(11):
            ids.update(_single_token_ids(tokenizer, str(i)))
    return sorted(ids)


def _build_eval_loader(cfg: BaselineSuiteConfig, tokenizer):
    if cfg.task_family == "single_image":
        ds = CIFARSingleImageQADataset(
            root=cfg.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=cfg.dataset_name,
            mode="vision",
            qa_type=cfg.qa_type,
            max_samples=cfg.max_eval_samples,
            seed=cfg.seed,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_single_image)
    ds = CIFARPopulationDataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=False,
        dataset_name=cfg.dataset_name,
        group_size=cfg.population_group_size,
        output_mode=cfg.population_output_mode,
        max_groups=cfg.max_eval_samples,
        seed=cfg.seed,
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_population)


def _build_llm(cfg: BaselineSuiteConfig, dev: torch.device):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.mistral_models_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        cfg.model_name,
        cache_dir=cfg.mistral_models_path,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        tokenizer=tok,
    )
    model.num_tokens = 0
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(dev).to(torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model.eval()
    return model, tok


def _count_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _predict_llm_next_token(model, tokens, mask, allowed_ids: List[int]):
    out = model(tokens, attention_mask=mask, expert_inputs=None, use_cache=False)
    logits = out.logits[:, -1, :]
    if allowed_ids:
        aid = torch.tensor(allowed_ids, device=logits.device, dtype=torch.long)
        local = logits.index_select(dim=-1, index=aid).argmax(dim=-1)
        pred = aid[local]
    else:
        pred = logits.argmax(dim=-1)
    return pred


def _metric_from_preds(
    cfg: BaselineSuiteConfig,
    batch: Dict,
    pred_ids: List[int],
    allowed_ids: List[int],
) -> Tuple[int, int, List[int], List[int], List[float], List[float]]:
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    pop_true = []
    pop_pred = []
    if cfg.task_family == "single_image":
        gt_ids = batch["target_token_id"].cpu().tolist()
        lookup = {int(t): i for i, t in enumerate(allowed_ids)} if allowed_ids else {}
        for p, g in zip(pred_ids, gt_ids):
            correct += int(int(p) == int(g))
            total += 1
            if lookup:
                y_pred.append(lookup.get(int(p), 0))
                y_true.append(lookup.get(int(g), 0))
    else:
        for p in pred_ids:
            # expected integer token baseline
            try:
                b = int(str(p))
            except Exception:
                b = 0
            pop_pred.append(population_bin_to_fraction_midpoint(b, mode="integer"))
        gtf = batch["fraction"].cpu().tolist()
        pop_true.extend(float(x) for x in gtf)
        gtbin = [population_fraction_to_bin(x, mode="integer") for x in gtf]
        predbin = [population_fraction_to_bin(x, mode="integer") for x in pop_pred]
        for p, g in zip(predbin, gtbin):
            correct += int(int(p) == int(g))
            total += 1
    return correct, total, y_true, y_pred, pop_true, pop_pred


def _evaluate_llm_only(cfg: BaselineSuiteConfig, model, tokenizer, loader, dev):
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_token_ids(cfg.task_family, qa_mode, tokenizer, ds)
    t0 = time.time()
    correct = total = 0
    all_true = []
    all_pred = []
    pop_true = []
    pop_pred = []
    with torch.no_grad():
        for batch in loader:
            tok = batch["prompt_tokens"].to(dev)
            m = batch["prompt_mask"].to(dev)
            pred = _predict_llm_next_token(model, tok, m, allowed).cpu().tolist()
            # For yes/no tasks, score by semantic parsing instead of raw token-id equality.
            # This avoids false zeros when tokenizer-specific single-token ids differ.
            if cfg.task_family == "single_image" and qa_mode in {"yesno", "yesno_set2"}:
                gt_yesno = [1 if str(a).lower().startswith("yes") else 0 for a in batch["answer_text"]]
                pred_yesno = []
                for p in pred:
                    txt = tokenizer.decode([int(p)], skip_special_tokens=True)
                    parsed = parse_yes_no_answer(txt)
                    if parsed is None:
                        # Fallback: keep deterministic behavior if decoding is empty/ambiguous.
                        parsed = 0
                    pred_yesno.append(int(parsed))
                correct += sum(int(a == b) for a, b in zip(pred_yesno, gt_yesno))
                total += len(gt_yesno)
                all_true.extend(gt_yesno)
                all_pred.extend(pred_yesno)
            else:
                c, t, yt, yp, pt, pp = _metric_from_preds(cfg, batch, pred, allowed)
                correct += c
                total += t
                all_true.extend(yt)
                all_pred.extend(yp)
                pop_true.extend(pt)
                pop_pred.extend(pp)
    latency = (time.time() - t0) / max(1, total)
    metrics = {"accuracy": correct / max(1, total)}
    if cfg.task_family == "single_image" and all_true:
        metrics["f1"] = macro_f1_from_ints(all_true, all_pred, num_classes=max(1, len(set(all_true))))
    if cfg.task_family == "population":
        metrics["mae"] = mean_absolute_error(pop_true, pop_pred)
    return metrics, latency


def _evaluate_expert_only(cfg: BaselineSuiteConfig, loader, dev):
    num_classes = _num_classes(cfg.dataset_name)
    expert = build_vision_expert("classifier", cfg.expert_checkpoint, num_classes, init_weights=cfg.expert_init_weights).to(dev)
    t0 = time.time()
    correct = total = 0
    pop_true = []
    pop_pred = []
    with torch.no_grad():
        for batch in loader:
            if cfg.task_family == "single_image":
                logits = expert(batch["images"].to(dev)).logits.float()
                pred_cls = logits.argmax(dim=-1).cpu().tolist()
                gts = batch["labels"].cpu().tolist()
                correct += sum(int(a == b) for a, b in zip(pred_cls, gts))
                total += len(gts)
            else:
                images = batch["images"].to(dev)
                b, n = images.shape[:2]
                flat = images.view(b * n, *images.shape[2:])
                logits = expert(flat).logits.float().view(b, n, -1)
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                tgt = batch["target_class"].to(dev)
                frac = probs[torch.arange(b, device=dev), tgt].cpu().tolist()
                gt = batch["fraction"].cpu().tolist()
                pop_pred.extend(frac)
                pop_true.extend(gt)
                pbin = [population_fraction_to_bin(x, mode="integer") for x in frac]
                gbin = [population_fraction_to_bin(x, mode="integer") for x in gt]
                correct += sum(int(a == b) for a, b in zip(pbin, gbin))
                total += b
    latency = (time.time() - t0) / max(1, total)
    metrics = {"accuracy": correct / max(1, total)}
    if cfg.task_family == "population":
        metrics["mae"] = mean_absolute_error(pop_true, pop_pred)
    return metrics, latency


def _build_captioner(model_id: str, device: torch.device):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, proc


def _two_stage_caption_eval(cfg: BaselineSuiteConfig, model, tokenizer, loader, dev):
    captioner, proc = _build_captioner(cfg.caption_model_id, dev)
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_token_ids(cfg.task_family, qa_mode, tokenizer, ds)
    t0 = time.time()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"]
            prompts = []
            for i in range(imgs.size(0)):
                pil = ((imgs[i].cpu() - imgs[i].cpu().min()) / (imgs[i].cpu().max() - imgs[i].cpu().min() + 1e-6))
                inputs = proc(images=pil, return_tensors="pt").to(dev)
                out = captioner.generate(**inputs, max_new_tokens=20)
                cap = proc.decode(out[0], skip_special_tokens=True)
                q = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
                msgs = [{"role": "system", "content": "Follow output format strictly."}, {"role": "user", "content": f"{q}\nImage caption: {cap}"}]
                prompts.append(tokenizer.apply_chat_template(msgs, return_tensors="pt").squeeze(0))
            max_len = max(x.size(0) for x in prompts)
            tok = torch.full((len(prompts), max_len), tokenizer.pad_token_id, dtype=torch.long, device=dev)
            m = torch.zeros((len(prompts), max_len), dtype=torch.long, device=dev)
            for i, t in enumerate(prompts):
                tok[i, max_len - t.size(0) :] = t.to(dev)
                m[i, max_len - t.size(0) :] = 1
            pred = _predict_llm_next_token(model, tok, m, allowed).cpu().tolist()
            gt = batch["target_token_id"].cpu().tolist()
            for p, g in zip(pred, gt):
                correct += int(int(p) == int(g))
                total += 1
    latency = (time.time() - t0) / max(1, total)
    return {"accuracy": correct / max(1, total)}, latency


def _vlm_generate_answer(model_type: str, model_id: str, prompt: str, image, device: torch.device) -> str:
    import transformers as tr

    proc = tr.AutoProcessor.from_pretrained(model_id)
    model_cls = getattr(tr, "AutoModelForImageTextToText", None) or getattr(tr, "AutoModelForVision2Seq", None)
    if model_cls is None:
        model_cls = tr.AutoModelForCausalLM
    model = model_cls.from_pretrained(model_id).to(device)
    model.eval()
    inputs = proc(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16)
    return proc.decode(out[0], skip_special_tokens=True)


def _postparse_to_allowed(text: str, allowed_token_ids: List[int], tokenizer) -> int:
    # Standardized post-constraint: choose first token from decoded text that maps into allowed ids.
    toks = tokenizer.encode(text, add_special_tokens=False)
    for t in toks:
        if int(t) in set(allowed_token_ids):
            return int(t)
    return int(allowed_token_ids[0]) if allowed_token_ids else (int(toks[0]) if toks else 0)


def _evaluate_frozen_vlm(cfg: BaselineSuiteConfig, loader, tokenizer, dev, model_type: str, model_id: str):
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_token_ids(cfg.task_family, qa_mode, tokenizer, ds)
    t0 = time.time()
    correct = total = 0
    for batch in loader:
        for i in range(batch["images"].size(0)):
            prompt = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
            img = batch["images"][i]
            x = (img - img.min()) / (img.max() - img.min() + 1e-6)
            text = _vlm_generate_answer(model_type, model_id, prompt, x, dev)
            pred_id = _postparse_to_allowed(text, allowed, tokenizer)
            gt_id = int(batch["target_token_id"][i].item())
            correct += int(pred_id == gt_id)
            total += 1
    latency = (time.time() - t0) / max(1, total)
    return {"accuracy": correct / max(1, total)}, latency


def _build_rag_db(cfg: BaselineSuiteConfig, tokenizer, dev):
    num_classes = _num_classes(cfg.dataset_name)
    expert = build_vision_expert("classifier", cfg.expert_checkpoint, num_classes, init_weights=cfg.expert_init_weights).to(dev)
    ds = CIFARSingleImageQADataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=True,
        dataset_name=cfg.dataset_name,
        mode="vision",
        qa_type=cfg.qa_type,
        max_samples=cfg.rag_db_size,
        seed=cfg.seed,
    )
    docs = []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds[i]
            img = row["image"].unsqueeze(0).to(dev)
            logits = expert(img).logits.float()
            probs = torch.softmax(logits, dim=-1)[0]
            top = torch.topk(probs, k=min(3, probs.numel()))
            triples = [f"class_{int(idx)}={float(val):.3f}" for val, idx in zip(top.values, top.indices)]
            docs.append(" ; ".join(triples))
    return docs


def _retrieve_doc(question: str, docs: List[str]) -> str:
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    best = ""
    score = -1
    for d in docs:
        d_words = set(re.findall(r"[a-z0-9]+", d.lower()))
        s = len(q_words & d_words)
        if s > score:
            score = s
            best = d
    return best if best else docs[0]


def _evaluate_rag(cfg: BaselineSuiteConfig, model, tokenizer, loader, dev):
    docs = _build_rag_db(cfg, tokenizer, dev)
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_token_ids(cfg.task_family, qa_mode, tokenizer, ds)
    t0 = time.time()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            prompts = []
            for i in range(batch["prompt_tokens"].size(0)):
                q = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
                ctx = _retrieve_doc(q, docs)
                msgs = [{"role": "system", "content": "Use retrieved evidence and answer with one token."}, {"role": "user", "content": f"{q}\nRetrieved: {ctx}"}]
                prompts.append(tokenizer.apply_chat_template(msgs, return_tensors="pt").squeeze(0))
            max_len = max(x.size(0) for x in prompts)
            tok = torch.full((len(prompts), max_len), tokenizer.pad_token_id, dtype=torch.long, device=dev)
            m = torch.zeros((len(prompts), max_len), dtype=torch.long, device=dev)
            for i, t in enumerate(prompts):
                tok[i, max_len - t.size(0) :] = t.to(dev)
                m[i, max_len - t.size(0) :] = 1
            pred = _predict_llm_next_token(model, tok, m, allowed).cpu().tolist()
            gt = batch["target_token_id"].cpu().tolist()
            for p, g in zip(pred, gt):
                correct += int(int(p) == int(g))
                total += 1
    latency = (time.time() - t0) / max(1, total)
    return {"accuracy": correct / max(1, total)}, latency


def _save(rows: List[Dict], out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_baseline_suite(cfg: BaselineSuiteConfig) -> List[Dict]:
    set_seed(cfg.seed)
    dev = _device(cfg.device)
    model, tokenizer = _build_llm(cfg, dev)
    loader = _build_eval_loader(cfg, tokenizer)

    rows: List[Dict] = []

    if cfg.include_injection:
        t0 = time.time()
        inj_metrics = evaluate_plan1(
            EvalPlan1Args(
                mistral_models_path=cfg.mistral_models_path,
                model_name=cfg.model_name,
                connectors_path=cfg.connectors_path,
                dataset_name=cfg.dataset_name,
                data_root=cfg.data_root,
                task_family=cfg.task_family,
                evidence_source="vision",
                qa_type=cfg.qa_type,
                population_output_mode=cfg.population_output_mode,
                population_group_size=cfg.population_group_size,
                num_tokens=cfg.num_tokens,
                layer_to_add=cfg.layer_to_add,
                evidence_dim=cfg.evidence_dim,
                alpha=cfg.alpha,
                use_adapters=bool(cfg.use_adapters),
                adapter_bottleneck=int(cfg.adapter_bottleneck),
                adapter_dropout=float(cfg.adapter_dropout),
                adapter_activation=str(cfg.adapter_activation),
                tune_layernorm=bool(cfg.tune_layernorm),
                inject_location=str(cfg.inject_location),
                enforce_checkpoint_compat=bool(cfg.enforce_checkpoint_compat),
                expert_kind=str(cfg.expert_kind),
                expert_output_mode=str(cfg.expert_output_mode),
                expert_checkpoint=cfg.expert_checkpoint,
                expert_init_weights=cfg.expert_init_weights,
                batch_size=cfg.batch_size,
                max_eval_samples=cfg.max_eval_samples,
                baseline="injection",
                device=cfg.device,
                constrain_llm_only_outputs=True,
            )
        )
        latency = (time.time() - t0) / max(1, cfg.max_eval_samples)
        n_trainable = 0
        if cfg.connectors_path and os.path.exists(cfg.connectors_path):
            st = torch.load(cfg.connectors_path, map_location="cpu")
            for k in ("evidence_builder", "projector"):
                if k in st and isinstance(st[k], dict):
                    n_trainable += int(sum(v.numel() for v in st[k].values() if torch.is_tensor(v)))
        rows.append(
            {
                "model_type": "injection",
                "dataset": cfg.dataset_name,
                "metric": float(inj_metrics.get("accuracy", inj_metrics.get("mae", 0.0))),
                "metric_name": "accuracy" if "accuracy" in inj_metrics else "mae",
                "params_trainable": n_trainable,
                "inference_latency": float(latency),
                **inj_metrics,
                "status": "ok",
            }
        )

    # 1) LLM-only
    if cfg.enable_llm_only:
        met, lat = _evaluate_llm_only(cfg, model, tokenizer, loader, dev)
        rows.append(
            {
                "model_type": "llm_only",
                "dataset": cfg.dataset_name,
                "metric": float(met.get("accuracy", met.get("mae", 0.0))),
                "metric_name": "accuracy" if "accuracy" in met else "mae",
                "params_trainable": _count_trainable(model),
                "inference_latency": float(lat),
                **met,
                "status": "ok",
            }
        )

    # 2) Expert-only
    if cfg.enable_expert_only:
        try:
            met, lat = _evaluate_expert_only(cfg, loader, dev)
            rows.append(
                {
                    "model_type": "expert_only",
                    "dataset": cfg.dataset_name,
                    "metric": float(met.get("accuracy", met.get("mae", 0.0))),
                    "metric_name": "accuracy" if "accuracy" in met else "mae",
                    "params_trainable": 0,
                    "inference_latency": float(lat),
                    **met,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"model_type": "expert_only", "dataset": cfg.dataset_name, "metric": 0.0, "params_trainable": 0, "inference_latency": 0.0, "status": "error", "error": str(exc)})

    # 3) Two-stage caption -> LLM
    if cfg.enable_two_stage:
        try:
            met, lat = _two_stage_caption_eval(cfg, model, tokenizer, loader, dev)
            rows.append(
                {
                    "model_type": "two_stage_caption_llm",
                    "dataset": cfg.dataset_name,
                    "metric": float(met.get("accuracy", 0.0)),
                    "metric_name": "accuracy",
                    "params_trainable": _count_trainable(model),
                    "inference_latency": float(lat),
                    **met,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"model_type": "two_stage_caption_llm", "dataset": cfg.dataset_name, "metric": 0.0, "params_trainable": _count_trainable(model), "inference_latency": 0.0, "status": "error", "error": str(exc)})

    # 4) Frozen SOTA VLMs
    vlms = cfg.frozen_vlms or [
        FrozenVLMConfig(model_type="llava_next", model_id="llava-hf/llava-v1.6-mistral-7b-hf", enabled=True),
        FrozenVLMConfig(model_type="qwen25_vl", model_id="Qwen/Qwen2.5-VL-7B-Instruct", enabled=True),
        FrozenVLMConfig(model_type="internvl", model_id="OpenGVLab/InternVL2_5-8B", enabled=False),
    ]
    for v in vlms:
        if not v.enabled:
            continue
        try:
            met, lat = _evaluate_frozen_vlm(cfg, loader, tokenizer, dev, v.model_type, v.model_id)
            rows.append(
                {
                    "model_type": f"frozen_vlm::{v.model_type}",
                    "dataset": cfg.dataset_name,
                    "metric": float(met.get("accuracy", 0.0)),
                    "metric_name": "accuracy",
                    "params_trainable": 0,
                    "inference_latency": float(lat),
                    **met,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"model_type": f"frozen_vlm::{v.model_type}", "dataset": cfg.dataset_name, "metric": 0.0, "params_trainable": 0, "inference_latency": 0.0, "status": "error", "error": str(exc)})

    # 5) Optional RAG baseline
    if cfg.enable_rag:
        try:
            met, lat = _evaluate_rag(cfg, model, tokenizer, loader, dev)
            rows.append(
                {
                    "model_type": "rag_structured",
                    "dataset": cfg.dataset_name,
                    "metric": float(met.get("accuracy", 0.0)),
                    "metric_name": "accuracy",
                    "params_trainable": _count_trainable(model),
                    "inference_latency": float(lat),
                    **met,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"model_type": "rag_structured", "dataset": cfg.dataset_name, "metric": 0.0, "params_trainable": _count_trainable(model), "inference_latency": 0.0, "status": "error", "error": str(exc)})

    _save(rows, cfg.out_csv, cfg.out_json)
    return rows


def load_baseline_suite_config(path: str) -> BaselineSuiteConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    fv = [FrozenVLMConfig(**x) for x in raw.pop("frozen_vlms", [])]
    cfg = BaselineSuiteConfig(**raw)
    if fv:
        cfg.frozen_vlms = fv
    return cfg
