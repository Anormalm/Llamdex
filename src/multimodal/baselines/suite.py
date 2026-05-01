from __future__ import annotations

import csv
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer

from src.model.DomainQwenModel import DomainQwenForCausalLM
from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    _chat_template_to_ids,
    collate_population,
    collate_single_image,
)
from src.multimodal.eval.plan1_eval import EvalPlan1Args, evaluate_plan1
from src.multimodal.experts import build_vision_expert
from src.multimodal.framework.expert_encoders import CLIPSigLIPExpertEncoder, DINOv2ExpertEncoder
from src.multimodal.tasks.parsing import population_bin_to_fraction_midpoint, population_fraction_to_bin
from src.multimodal.task_matrix.metrics import macro_f1_from_ints, mean_absolute_error
from src.multimodal.utils.repro import set_seed


@dataclass
class FrozenVLMConfig:
    model_type: str  # qwen25_vl|llava_onevision|llama32_vision|internvl
    model_id: str
    enabled: bool = True
    trust_remote_code: bool = False


@dataclass
class BaselineSuiteConfig:
    server_models_path: str = "/disk1/lfhu/hf_cache"
    model_name: str = "Qwen/Qwen3.5-9B"
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
    enable_two_stage: bool = True
    enable_rag: bool = True
    enable_linear_probe: bool = True
    # rag
    rag_db_size: int = 256
    # meaningful adapted baselines
    linear_probe_encoders: List[str] = field(default_factory=lambda: ["dinov2", "clip"])
    linear_probe_max_train_samples: int = 2048
    linear_probe_max_iter: int = 1000
    # prediction audit
    audit_predictions: bool = True
    audit_out_jsonl: Optional[str] = None
    audit_max_rows_per_model: int = 512
    # frozen VLM
    enable_frozen_vlms: bool = True
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


def _load_torch_weights(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _num_classes(dataset_name: str) -> int:
    known = {
        "cifar10": 10,
        "cifar100": 100,
        "dtd": 47,
        "oxford_pet": 37,
        "eurosat": 10,
        "food101": 101,
        "fgvc_aircraft": 100,
        "resisc45": 45,
    }
    if dataset_name in known:
        return known[dataset_name]
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


def _yes_no_token_sets(tokenizer):
    yes_ids = set(_single_token_ids(tokenizer, "Yes"))
    no_ids = set(_single_token_ids(tokenizer, "No"))
    return yes_ids, no_ids


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


def _audit_path(cfg: BaselineSuiteConfig) -> str:
    if cfg.audit_out_jsonl:
        return cfg.audit_out_jsonl
    root, _ = os.path.splitext(cfg.out_json)
    return f"{root}.audit.jsonl"


def _append_audit_rows(cfg: BaselineSuiteConfig, rows: List[Dict[str, Any]]) -> None:
    if not cfg.audit_predictions or not rows:
        return
    path = _audit_path(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows[: max(0, int(cfg.audit_max_rows_per_model))]:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _class_lookup(ds) -> Tuple[List[str], Dict[int, int], Dict[str, int], Dict[str, int]]:
    names = list(getattr(ds, "class_names", []) or [])
    token_to_label: Dict[int, int] = {}
    code_to_label: Dict[str, int] = {}
    if getattr(ds, "idx_to_code", None):
        code_to_label = {str(code).strip(): int(idx) for idx, code in ds.idx_to_code.items()}
        token_to_label = {
            int(ds.code_to_token_id[str(code)]): int(idx)
            for idx, code in ds.idx_to_code.items()
            if getattr(ds, "code_to_token_id", None) and str(code) in ds.code_to_token_id
        }
    norm_name_to_label: Dict[str, int] = {}
    for idx, name in enumerate(names):
        variants = {
            str(name),
            str(name).replace("_", " "),
            str(name).replace("-", " "),
            str(name).replace("_", "-"),
        }
        for v in variants:
            norm_name_to_label[_normalize_label_text(v)] = int(idx)
    return names, token_to_label, code_to_label, norm_name_to_label


def _normalize_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def _parse_label_prediction(text: str, ds, tokenizer, allowed_token_ids: Optional[List[int]] = None) -> Tuple[int, str, bool]:
    names, token_to_label, code_to_label, norm_name_to_label = _class_lookup(ds)
    raw = str(text or "").strip()
    stripped = raw.strip().strip("`'\".，,;:")
    if stripped in code_to_label:
        return code_to_label[stripped], stripped, True
    if "=" in stripped:
        left, right = stripped.split("=", 1)
        left = left.strip().strip("`'\".，,;:")
        right = right.strip().strip("`'\".，,;:")
        if left in code_to_label:
            return code_to_label[left], stripped, True
        norm_right = _normalize_label_text(right)
        if norm_right in norm_name_to_label:
            return norm_name_to_label[norm_right], stripped, True
    if stripped in code_to_label:
        return code_to_label[stripped], stripped, True
    if len(stripped) != 1:
        # Multi-character answers should be interpreted as labels/phrases, not
        # as arbitrary code tokens embedded inside the generated text.
        toks = []
    else:
        toks = tokenizer.encode(stripped, add_special_tokens=False) if tokenizer is not None else []
    allowed = set(int(x) for x in (allowed_token_ids or []))
    for tid in toks:
        if (not allowed or int(tid) in allowed) and int(tid) in token_to_label:
            return token_to_label[int(tid)], stripped, True
    norm = _normalize_label_text(stripped)
    if norm in norm_name_to_label:
        return norm_name_to_label[norm], stripped, True
    padded = f" {norm} "
    # Prefer longer labels first to avoid matching "dotted" inside a longer phrase incorrectly.
    for key, idx in sorted(norm_name_to_label.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key and f" {key} " in padded:
            return idx, names[idx] if idx < len(names) else key, True
    return -1, stripped, False


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).strip()
    except TypeError:
        return str(tokenizer.decode([int(token_id)])).strip()


def _build_llm(cfg: BaselineSuiteConfig, dev: torch.device):
    tok = _build_tokenizer(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    model = DomainQwenForCausalLM.from_pretrained_qwen(
        cfg.model_name,
        cache_dir=cfg.server_models_path,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        tokenizer=tok,
    )
    model.num_tokens = 0
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(dev).to(torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model.eval()
    return model, tok


def _build_tokenizer(cfg: BaselineSuiteConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.server_models_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    return tok


def _count_trainable(model) -> int:
    if model is None:
        return 0
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
    yes_ids, no_ids = _yes_no_token_sets(tokenizer) if qa_mode in {"yesno", "yesno_set2"} else (set(), set())
    t0 = time.time()
    correct = total = 0
    all_true = []
    all_pred = []
    pop_true = []
    pop_pred = []
    parse_success = 0
    audit = []
    with torch.no_grad():
        for batch in loader:
            tok = batch["prompt_tokens"].to(dev)
            m = batch["prompt_mask"].to(dev)
            pred = _predict_llm_next_token(model, tok, m, allowed).cpu().tolist()
            if cfg.task_family == "single_image" and qa_mode in {"yesno", "yesno_set2"}:
                gt_yesno = [1 if str(a).lower().startswith("yes") else 0 for a in batch["answer_text"]]
                pred_yesno = []
                for p in pred:
                    if int(p) in yes_ids:
                        pred_yesno.append(1)
                    elif int(p) in no_ids:
                        pred_yesno.append(0)
                    else:
                        pred_yesno.append(-1)
                correct += sum(int(a == b) for a, b in zip(pred_yesno, gt_yesno))
                total += len(gt_yesno)
                all_true.extend(gt_yesno)
                all_pred.extend(pred_yesno)
            else:
                if cfg.task_family == "single_image":
                    for i, (p, g) in enumerate(zip(pred, batch["labels"].cpu().tolist())):
                        raw_pred = _decode_token(tokenizer, int(p))
                        pred_label, normalized, ok = _parse_label_prediction(raw_pred, ds, tokenizer, allowed)
                        parse_success += int(ok)
                        audit.append(
                            {
                                "model_type": "llm_only",
                                "dataset": cfg.dataset_name,
                                "task": cfg.task_family,
                                "question": batch.get("question_text", [""] * len(pred))[i],
                                "raw_prediction": raw_pred,
                                "normalized_prediction": normalized,
                                "pred_label": int(pred_label),
                                "target_label": int(g),
                                "target_name": ds.class_names[int(g)] if hasattr(ds, "class_names") else "",
                                "correct": bool(int(pred_label) == int(g)),
                                "parse_success": bool(ok),
                            }
                        )
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
        metrics["parse_success_rate"] = parse_success / max(1, total)
        metrics["format_compliance"] = metrics["parse_success_rate"]
    if cfg.task_family == "population":
        metrics["mae"] = mean_absolute_error(pop_true, pop_pred)
    _append_audit_rows(cfg, audit)
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
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_success = 0
    audit = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"]
            prompts = []
            captions = []
            for i in range(imgs.size(0)):
                norm = (imgs[i].cpu() - imgs[i].cpu().min()) / (imgs[i].cpu().max() - imgs[i].cpu().min() + 1e-6)
                pil = to_pil_image(norm)
                inputs = proc(images=pil, return_tensors="pt").to(dev)
                out = captioner.generate(**inputs, max_new_tokens=20)
                cap = proc.decode(out[0], skip_special_tokens=True)
                captions.append(cap)
                q = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
                msgs = [{"role": "system", "content": "Follow output format strictly."}, {"role": "user", "content": f"{q}\nImage caption: {cap}"}]
                prompts.append(_chat_template_to_ids(tokenizer, msgs))
            max_len = max(x.size(0) for x in prompts)
            tok = torch.full((len(prompts), max_len), tokenizer.pad_token_id, dtype=torch.long, device=dev)
            m = torch.zeros((len(prompts), max_len), dtype=torch.long, device=dev)
            for i, t in enumerate(prompts):
                tok[i, max_len - t.size(0) :] = t.to(dev)
                m[i, max_len - t.size(0) :] = 1
            pred = _predict_llm_next_token(model, tok, m, allowed).cpu().tolist()
            gt = batch["target_token_id"].cpu().tolist()
            labels = batch.get("labels")
            label_list = labels.cpu().tolist() if labels is not None else [None] * len(pred)
            for i, (p, g) in enumerate(zip(pred, gt)):
                correct += int(int(p) == int(g))
                total += 1
                if label_list[i] is not None:
                    raw_pred = _decode_token(tokenizer, int(p))
                    pred_label, normalized, ok = _parse_label_prediction(raw_pred, ds, tokenizer, allowed)
                    parse_success += int(ok)
                    y_true.append(int(label_list[i]))
                    y_pred.append(int(pred_label) if pred_label >= 0 else 0)
                    audit.append(
                        {
                            "model_type": "two_stage_caption_llm",
                            "dataset": cfg.dataset_name,
                            "task": cfg.task_family,
                            "question": batch.get("question_text", [""] * len(pred))[i],
                            "caption": captions[i],
                            "raw_prediction": raw_pred,
                            "normalized_prediction": normalized,
                            "pred_label": int(pred_label),
                            "target_label": int(label_list[i]),
                            "target_name": ds.class_names[int(label_list[i])] if hasattr(ds, "class_names") else "",
                            "correct": bool(int(pred_label) == int(label_list[i])),
                            "parse_success": bool(ok),
                        }
                    )
    latency = (time.time() - t0) / max(1, total)
    metrics = {"accuracy": correct / max(1, total)}
    if y_true:
        metrics["f1"] = macro_f1_from_ints(y_true, y_pred, num_classes=len(getattr(ds, "class_names", [])) or max(1, len(set(y_true))))
        metrics["parse_success_rate"] = parse_success / max(1, len(y_true))
        metrics["format_compliance"] = metrics["parse_success_rate"]
    _append_audit_rows(cfg, audit)
    return metrics, latency


def _frozen_vlm_dtype(device: torch.device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _load_frozen_vlm(cfg: BaselineSuiteConfig, spec: FrozenVLMConfig, device: torch.device):
    import transformers as tr

    proc = tr.AutoProcessor.from_pretrained(
        spec.model_id,
        cache_dir=cfg.server_models_path,
        trust_remote_code=bool(spec.trust_remote_code),
    )
    model_cls = getattr(tr, "AutoModelForImageTextToText", None) or getattr(tr, "AutoModelForVision2Seq", None)
    if model_cls is None:
        model_cls = tr.AutoModelForCausalLM
    model = model_cls.from_pretrained(
        spec.model_id,
        cache_dir=cfg.server_models_path,
        torch_dtype=_frozen_vlm_dtype(device),
        low_cpu_mem_usage=True,
        trust_remote_code=bool(spec.trust_remote_code),
    ).to(device)
    model.eval()
    return model, proc


def _vlm_generate_answer(
    prompt: str,
    image,
    *,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 16,
) -> str:
    constrained = prompt
    # VLMs answer more reliably when they can choose natural labels, while the
    # evaluator still maps labels/codes back to the canonical target.
    constrained += "\nIf labels are listed as CODE=label, reply with either the exact CODE or exact label."
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{constrained}\nReturn only the answer. No explanation."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    else:
        inputs = processor(text=f"{constrained}\nReturn only the answer. No explanation.", images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = 0
    if "input_ids" in inputs and inputs["input_ids"].ndim == 2:
        input_len = int(inputs["input_ids"].shape[1])
    gen_ids = out[0][input_len:] if input_len > 0 else out[0]
    if hasattr(processor, "decode"):
        return str(processor.decode(gen_ids, skip_special_tokens=True))
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return str(tok.decode(gen_ids, skip_special_tokens=True))
    return ""


def _postparse_to_allowed(text: str, allowed_token_ids: List[int], tokenizer) -> int:
    # Standardized post-constraint: choose first token from decoded text that maps into allowed ids.
    toks = tokenizer.encode(text, add_special_tokens=False)
    for t in toks:
        if int(t) in set(allowed_token_ids):
            return int(t)
    return int(allowed_token_ids[0]) if allowed_token_ids else (int(toks[0]) if toks else 0)


def _evaluate_frozen_vlm(
    cfg: BaselineSuiteConfig,
    loader,
    tokenizer,
    dev,
    spec: FrozenVLMConfig,
    model=None,
    processor=None,
):
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_token_ids(cfg.task_family, qa_mode, tokenizer, ds)
    if model is None or processor is None:
        model, processor = _load_frozen_vlm(cfg, spec, dev)
    t0 = time.time()
    correct = total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_success = 0
    audit = []
    for batch in loader:
        for i in range(batch["images"].size(0)):
            prompt = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
            img = batch["images"][i]
            x = (img - img.min()) / (img.max() - img.min() + 1e-6)
            text = _vlm_generate_answer(prompt, x, model=model, processor=processor, device=dev)
            if cfg.task_family == "single_image" and "labels" in batch:
                pred_label, normalized, ok = _parse_label_prediction(text, ds, tokenizer, allowed)
                gt_label = int(batch["labels"][i].item())
                correct += int(pred_label == gt_label)
                y_true.append(gt_label)
                y_pred.append(int(pred_label) if pred_label >= 0 else 0)
                parse_success += int(ok)
                audit.append(
                    {
                        "model_type": f"frozen_vlm::{spec.model_type}",
                        "dataset": cfg.dataset_name,
                        "task": cfg.task_family,
                        "question": prompt,
                        "raw_prediction": text,
                        "normalized_prediction": normalized,
                        "pred_label": int(pred_label),
                        "target_label": gt_label,
                        "target_name": ds.class_names[gt_label] if hasattr(ds, "class_names") else "",
                        "correct": bool(pred_label == gt_label),
                        "parse_success": bool(ok),
                    }
                )
            else:
                pred_id = _postparse_to_allowed(text, allowed, tokenizer)
                gt_id = int(batch["target_token_id"][i].item())
                correct += int(pred_id == gt_id)
            total += 1
    latency = (time.time() - t0) / max(1, total)
    metrics = {"accuracy": correct / max(1, total)}
    if y_true:
        metrics["f1"] = macro_f1_from_ints(y_true, y_pred, num_classes=len(getattr(ds, "class_names", [])) or max(1, len(set(y_true))))
        metrics["parse_success_rate"] = parse_success / max(1, len(y_true))
        metrics["format_compliance"] = metrics["parse_success_rate"]
    _append_audit_rows(cfg, audit)
    return metrics, latency


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
            img = row["images"].unsqueeze(0).to(dev)
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
                prompts.append(_chat_template_to_ids(tokenizer, msgs))
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


def _build_probe_encoder(name: str, cfg: BaselineSuiteConfig, dev: torch.device):
    n = str(name or "").strip().lower()
    if n == "dinov2":
        enc = DINOv2ExpertEncoder(model_id="facebook/dinov2-base", cache_dir=cfg.server_models_path)
    elif n == "clip":
        enc = CLIPSigLIPExpertEncoder(expert_type="clip", model_id="openai/clip-vit-base-patch32", cache_dir=cfg.server_models_path)
    elif n == "siglip":
        enc = CLIPSigLIPExpertEncoder(expert_type="siglip", model_id="google/siglip-base-patch16-224", cache_dir=cfg.server_models_path)
    else:
        raise ValueError(f"Unsupported linear_probe_encoder: {name}")
    enc = enc.to(dev)
    enc.eval()
    return enc


def _probe_dataset(cfg: BaselineSuiteConfig, tokenizer, *, train: bool, max_samples: Optional[int]):
    return CIFARSingleImageQADataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=train,
        dataset_name=cfg.dataset_name,
        mode="vision",
        qa_type=cfg.qa_type,
        max_samples=max_samples,
        seed=cfg.seed,
    )


def _extract_probe_features(encoder, dataset, batch_size: int, dev: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_single_image)
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            z = encoder.encode({"images": batch["images"].to(dev)}).detach().float().cpu().numpy()
            feats.append(z)
            labels.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _evaluate_linear_probe(cfg: BaselineSuiteConfig, tokenizer, dev: torch.device, encoder_name: str) -> Tuple[Dict[str, float], float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if cfg.task_family != "single_image":
        raise ValueError("linear_probe baseline currently supports task_family=single_image only.")
    train_ds = _probe_dataset(cfg, tokenizer, train=True, max_samples=cfg.linear_probe_max_train_samples)
    eval_ds = _probe_dataset(cfg, tokenizer, train=False, max_samples=cfg.max_eval_samples)
    encoder = _build_probe_encoder(encoder_name, cfg, dev)
    t0 = time.time()
    x_train, y_train = _extract_probe_features(encoder, train_ds, cfg.batch_size, dev)
    x_eval, y_eval = _extract_probe_features(encoder, eval_ds, cfg.batch_size, dev)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=int(cfg.linear_probe_max_iter), class_weight="balanced", solver="lbfgs", multi_class="auto"),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_eval)
    latency = (time.time() - t0) / max(1, len(y_eval))
    n_classes = len(getattr(eval_ds, "class_names", [])) or int(max(y_eval.max(), pred.max()) + 1)
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1": float(macro_f1_from_ints([int(x) for x in y_eval], [int(x) for x in pred], num_classes=n_classes)),
        "parse_success_rate": 1.0,
        "format_compliance": 1.0,
        "train_samples": int(len(y_train)),
    }, latency


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
    if cfg.audit_predictions:
        audit_path = _audit_path(cfg)
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        with open(audit_path, "w", encoding="utf-8"):
            pass
    needs_llm = bool(cfg.include_injection or cfg.enable_llm_only or cfg.enable_two_stage or cfg.enable_rag)
    if needs_llm:
        model, tokenizer = _build_llm(cfg, dev)
    else:
        model = None
        tokenizer = _build_tokenizer(cfg)
    loader = _build_eval_loader(cfg, tokenizer)

    rows: List[Dict] = []

    if cfg.include_injection:
        t0 = time.time()
        inj_metrics = evaluate_plan1(
            EvalPlan1Args(
                server_models_path=cfg.server_models_path,
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
            st = _load_torch_weights(cfg.connectors_path, map_location="cpu")
            for k in ("evidence_builder", "projector"):
                if k in st and isinstance(st[k], dict):
                    n_trainable += int(sum(v.numel() for v in st[k].values() if torch.is_tensor(v)))
        rows.append(
            {
                "benchmark_family": "local_baseline",
                "model_type": "injection",
                "task": cfg.task_family,
                "backbone": cfg.model_name,
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
                "benchmark_family": "local_baseline",
                "model_type": "llm_only",
                "task": cfg.task_family,
                "backbone": cfg.model_name,
                "dataset": cfg.dataset_name,
                "metric": float(met.get("accuracy", met.get("mae", 0.0))),
                "metric_name": "accuracy" if "accuracy" in met else "mae",
                "params_trainable": _count_trainable(model),
                "inference_latency": float(lat),
                **met,
                "status": "ok",
            }
        )

    # 2) Two-stage caption -> LLM
    if cfg.enable_two_stage:
        try:
            met, lat = _two_stage_caption_eval(cfg, model, tokenizer, loader, dev)
            rows.append(
                {
                    "benchmark_family": "local_baseline",
                    "model_type": "two_stage_caption_llm",
                    "task": cfg.task_family,
                    "backbone": cfg.model_name,
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
            rows.append({"benchmark_family": "local_baseline", "model_type": "two_stage_caption_llm", "task": cfg.task_family, "backbone": cfg.model_name, "dataset": cfg.dataset_name, "metric": 0.0, "metric_name": "accuracy", "params_trainable": _count_trainable(model), "inference_latency": 0.0, "status": "error", "error": str(exc) or repr(exc), "traceback": traceback.format_exc(limit=5)})

    # 3) Lightly adapted representation baselines. These are the most meaningful
    # classical comparison for DTD-style classification: frozen SOTA features,
    # trained linear classifier.
    if cfg.enable_linear_probe and cfg.task_family == "single_image":
        for enc_name in cfg.linear_probe_encoders:
            try:
                met, lat = _evaluate_linear_probe(cfg, tokenizer, dev, enc_name)
                rows.append(
                    {
                        "benchmark_family": "local_baseline",
                        "model_type": f"linear_probe::{enc_name}",
                        "task": cfg.task_family,
                        "backbone": cfg.model_name,
                        "dataset": cfg.dataset_name,
                        "metric": float(met.get("accuracy", 0.0)),
                        "metric_name": "accuracy",
                        "params_trainable": int((_num_classes(cfg.dataset_name) + 1) * 768) if enc_name == "dinov2" else 0,
                        "inference_latency": float(lat),
                        **met,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "benchmark_family": "local_baseline",
                        "model_type": f"linear_probe::{enc_name}",
                        "task": cfg.task_family,
                        "backbone": cfg.model_name,
                        "dataset": cfg.dataset_name,
                        "metric": 0.0,
                        "metric_name": "accuracy",
                        "params_trainable": 0,
                        "inference_latency": 0.0,
                        "status": "error",
                        "error": str(exc) or repr(exc),
                        "traceback": traceback.format_exc(limit=5),
                    }
                )
            finally:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    # 4) Frozen SOTA VLMs
    if cfg.enable_frozen_vlms:
        vlms = cfg.frozen_vlms or [
            FrozenVLMConfig(model_type="qwen25_vl", model_id="Qwen/Qwen2.5-VL-7B-Instruct", enabled=True),
            FrozenVLMConfig(model_type="llava_onevision", model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf", enabled=False),
            FrozenVLMConfig(model_type="llama32_vision", model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", enabled=False),
            FrozenVLMConfig(model_type="internvl", model_id="OpenGVLab/InternVL2_5-8B", enabled=False, trust_remote_code=True),
        ]
        for v in vlms:
            if not v.enabled:
                continue
            try:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                vlm_model, vlm_processor = _load_frozen_vlm(cfg, v, dev)
                met, lat = _evaluate_frozen_vlm(cfg, loader, tokenizer, dev, v, model=vlm_model, processor=vlm_processor)
                rows.append(
                    {
                        "benchmark_family": "local_baseline",
                        "model_type": f"frozen_vlm::{v.model_type}",
                        "task": cfg.task_family,
                        "backbone": cfg.model_name,
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
                rows.append({"benchmark_family": "local_baseline", "model_type": f"frozen_vlm::{v.model_type}", "task": cfg.task_family, "backbone": cfg.model_name, "dataset": cfg.dataset_name, "metric": 0.0, "metric_name": "accuracy", "params_trainable": 0, "inference_latency": 0.0, "status": "error", "error": str(exc) or repr(exc), "traceback": traceback.format_exc(limit=5)})

    # 5) Optional RAG baseline
    if cfg.enable_rag:
        try:
            met, lat = _evaluate_rag(cfg, model, tokenizer, loader, dev)
            rows.append(
                {
                    "benchmark_family": "local_baseline",
                    "model_type": "rag_structured",
                    "task": cfg.task_family,
                    "backbone": cfg.model_name,
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
            rows.append({"benchmark_family": "local_baseline", "model_type": "rag_structured", "task": cfg.task_family, "backbone": cfg.model_name, "dataset": cfg.dataset_name, "metric": 0.0, "metric_name": "accuracy", "params_trainable": _count_trainable(model), "inference_latency": 0.0, "status": "error", "error": str(exc) or repr(exc), "traceback": traceback.format_exc(limit=5)})

    _save(rows, cfg.out_csv, cfg.out_json)
    return rows


def load_baseline_suite_config(path: str) -> BaselineSuiteConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("enable_expert_only", None)
    fv: List[FrozenVLMConfig] = []
    for item in raw.pop("frozen_vlms", []):
        data: Dict[str, Any] = dict(item)
        data.setdefault("trust_remote_code", bool(data.get("model_type", "") in {"internvl"}))
        fv.append(FrozenVLMConfig(**data))
    cfg = BaselineSuiteConfig(**raw)
    if fv:
        cfg.frozen_vlms = fv
    return cfg
