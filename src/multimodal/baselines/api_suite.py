from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.multimodal.data.cifar_qa import (
    CIFARPopulationDataset,
    CIFARSingleImageQADataset,
    collate_population,
    collate_single_image,
)
from src.multimodal.tasks.parsing import population_fraction_to_bin
from src.multimodal.task_matrix.metrics import macro_f1_from_ints, mean_absolute_error
from src.multimodal.utils.repro import set_seed


@dataclass
class APIModelSpec:
    name: str
    model_id: str
    enabled: bool = True
    is_vision: bool = False
    provider: str = "openai_compatible"
    api_base_url: Optional[str] = None
    api_chat_path: Optional[str] = None
    api_key_env: Optional[str] = None


@dataclass
class APIBaselineSuiteConfig:
    # OpenAI-compatible API settings
    api_base_url: str = "https://api.openai.com/v1"
    api_chat_path: str = "/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 120
    api_max_retries: int = 5
    api_retry_base_s: float = 1.0
    api_retry_max_s: float = 16.0
    api_retry_jitter_s: float = 0.25

    # Task/data
    dataset_name: str = "dtd"
    dataset_names: List[str] = field(default_factory=list)
    data_root: str = "./data"
    task_family: str = "single_image"  # single_image|population
    qa_type: str = "label_code"
    population_output_mode: str = "integer"
    population_group_size: int = 8
    batch_size: int = 1
    max_eval_samples: int = 64
    seed: int = 42
    eval_repeats: int = 1
    repeat_seed_stride: int = 1000
    verbose: bool = True
    log_every_samples: int = 16

    # Baseline toggles
    enable_llm_only: bool = True
    enable_two_stage: bool = True
    enable_vlm_direct: bool = True
    enable_rag: bool = True

    # Two-stage
    caption_prompt: str = "Describe this image in one short sentence."

    # RAG settings
    rag_db_size: int = 128

    # SOTA model slots (editable in config)
    llm_models: List[APIModelSpec] = field(
        default_factory=lambda: [
            APIModelSpec(name="gpt-4.1", model_id="gpt-4.1", enabled=True, is_vision=False),
            APIModelSpec(name="qwen3-235b", model_id="Qwen/Qwen3-235B-A22B-Instruct", enabled=False, is_vision=False),
        ]
    )
    vlm_models: List[APIModelSpec] = field(
        default_factory=lambda: [
            APIModelSpec(name="gpt-4o", model_id="gpt-4o", enabled=True, is_vision=True),
            APIModelSpec(name="qwen2.5-vl-72b", model_id="Qwen/Qwen2.5-VL-72B-Instruct", enabled=False, is_vision=True),
        ]
    )

    # Output
    out_csv: str = "runs/api_baseline_suite_results.csv"
    out_json: str = "runs/api_baseline_suite_results.json"
    out_summary_csv: str = "runs/api_baseline_suite_results.summary.csv"
    out_summary_json: str = "runs/api_baseline_suite_results.summary.json"
    hf_cache_dir: str = "runs/hf_cache_tiny"


def _prepare_image_for_api(img_t: torch.Tensor) -> str:
    x = img_t.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor CxHxW, got {tuple(x.shape)}")
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(x)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_text_from_response(resp_json: Dict) -> str:
    # OpenAI-compatible chat/completions
    if isinstance(resp_json, dict):
        choices = resp_json.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    chunks = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            chunks.append(str(c.get("text", "")))
                    return " ".join(chunks).strip()
    return ""


def _api_chat(
    cfg: APIBaselineSuiteConfig,
    model: APIModelSpec,
    messages: List[Dict],
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    key_env = model.api_key_env or cfg.api_key_env
    key = os.getenv(key_env, "")
    if not key:
        raise RuntimeError(f"Missing API key env var: {key_env}")
    api_base_url = (model.api_base_url or cfg.api_base_url).rstrip("/")
    api_chat_path = model.api_chat_path or cfg.api_chat_path
    url = api_base_url + api_chat_path
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model.model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = ""
    for attempt in range(cfg.api_max_retries + 1):
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=cfg.timeout_s)
        if r.status_code < 400:
            return _extract_text_from_response(r.json())

        last_err = f"API {r.status_code}: {r.text[:400]}"
        should_retry = r.status_code in {408, 409, 429, 500, 502, 503, 504}
        if not should_retry or attempt >= cfg.api_max_retries:
            break
        backoff = min(cfg.api_retry_max_s, cfg.api_retry_base_s * (2**attempt))
        if cfg.api_retry_jitter_s > 0:
            backoff += random.random() * cfg.api_retry_jitter_s
        time.sleep(backoff)
    raise RuntimeError(last_err or "Unknown API error")


def _single_token_ids(tokenizer, text: str):
    ids = set()
    for v in (text, text.lower(), text.upper()):
        for p in (" ", ""):
            t = tokenizer.encode(p + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(int(t[0]))
    return ids


def _allowed_ids(task_family: str, qa_mode: str, tokenizer, ds) -> List[int]:
    ids = set()
    if task_family == "single_image":
        if qa_mode in {"yesno", "yesno_set2"}:
            ids.update(_single_token_ids(tokenizer, "Yes"))
            ids.update(_single_token_ids(tokenizer, "No"))
        elif qa_mode == "label_code" and getattr(ds, "code_to_token_id", None) is not None:
            ids.update(int(v) for v in ds.code_to_token_id.values())
        elif qa_mode == "index":
            for i in range(len(ds.class_names)):
                ids.update(_single_token_ids(tokenizer, str(i)))
        else:
            for i in range(min(26, len(ds.class_names))):
                ids.update(_single_token_ids(tokenizer, chr(ord("A") + i)))
    else:
        for i in range(11):
            ids.update(_single_token_ids(tokenizer, str(i)))
    return sorted(ids)


def _decode_constrained_token(text: str, tokenizer, allowed_ids: List[int]) -> int:
    toks = tokenizer.encode(text, add_special_tokens=False)
    allowed = set(int(x) for x in allowed_ids)
    for t in toks:
        if int(t) in allowed:
            return int(t)
    return int(allowed_ids[0]) if allowed_ids else (int(toks[0]) if toks else 0)


def _strict_answer_instruction(task_family: str, qa_mode: str, ds) -> str:
    if task_family == "single_image":
        if qa_mode == "label_code" and getattr(ds, "class_codes", None) is not None and getattr(ds, "class_names", None) is not None:
            pairs = ", ".join([f"{ds.class_codes[i]}={ds.class_names[i]}" for i in range(len(ds.class_names))])
            return (
                "Return ONLY one class code token, nothing else.\n"
                f"Valid codes and classes: {pairs}\n"
                "Example valid output: A"
            )
        if qa_mode in {"yesno", "yesno_set2"}:
            return "Return ONLY one token: Yes or No."
        return "Return ONLY one token."
    return "Return ONLY one integer token from 0 to 10."


def _build_loader(cfg: APIBaselineSuiteConfig, tokenizer, dataset_name: str, seed: int):
    if cfg.task_family == "single_image":
        ds = CIFARSingleImageQADataset(
            root=cfg.data_root,
            tokenizer=tokenizer,
            train=False,
            dataset_name=dataset_name,
            mode="vision",
            qa_type=cfg.qa_type,
            max_samples=cfg.max_eval_samples,
            seed=seed,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_single_image)
    ds = CIFARPopulationDataset(
        root=cfg.data_root,
        tokenizer=tokenizer,
        train=False,
        dataset_name=dataset_name,
        group_size=cfg.population_group_size,
        output_mode=cfg.population_output_mode,
        max_groups=cfg.max_eval_samples,
        seed=seed,
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_population)


def _retrieve_structured(question: str, docs: List[str]) -> str:
    qw = set(re.findall(r"[a-z0-9]+", question.lower()))
    best = docs[0] if docs else ""
    best_s = -1
    for d in docs:
        dw = set(re.findall(r"[a-z0-9]+", d.lower()))
        s = len(qw & dw)
        if s > best_s:
            best_s = s
            best = d
    return best


def _build_rag_docs(loader, limit: int) -> List[str]:
    docs = []
    for batch in loader:
        if len(docs) >= limit:
            break
        bs = batch["target_token_id"].size(0)
        for i in range(bs):
            q = batch["question_text"][i] if "question_text" in batch else "population query"
            a = batch["answer_text"][i] if "answer_text" in batch else "0"
            docs.append(f"Q: {q} ; A: {a}")
            if len(docs) >= limit:
                break
    return docs


def _eval_api_model(
    cfg: APIBaselineSuiteConfig,
    model: APIModelSpec,
    loader,
    tokenizer,
    mode: str,  # llm_only|vlm_direct|two_stage|rag
) -> Tuple[Dict[str, float], float]:
    ds = loader.dataset
    qa_mode = getattr(ds, "effective_qa_type", cfg.qa_type)
    allowed = _allowed_ids(cfg.task_family, qa_mode, tokenizer, ds)
    yes_ids = set(_single_token_ids(tokenizer, "Yes")) if qa_mode in {"yesno", "yesno_set2"} else set()
    no_ids = set(_single_token_ids(tokenizer, "No")) if qa_mode in {"yesno", "yesno_set2"} else set()

    rag_docs = _build_rag_docs(loader, cfg.rag_db_size) if mode == "rag" else []
    t0 = time.time()
    strict_correct = semantic_correct = total = 0
    y_true: List[int] = []
    y_pred_strict: List[int] = []
    y_pred_semantic: List[int] = []
    pop_true: List[float] = []
    pop_pred: List[float] = []

    for batch in loader:
        bs = batch["target_token_id"].size(0)
        for i in range(bs):
            question = batch["question_text"][i] if "question_text" in batch else "Answer with one token."
            img = batch["images"][i]
            if isinstance(img, torch.Tensor) and img.ndim == 4:
                # Population batches provide a group of images; send the first image as visual context.
                img = img[0]
            img_b64 = _prepare_image_for_api(img)

            answer_rule = _strict_answer_instruction(cfg.task_family, qa_mode, ds)
            if mode == "llm_only":
                msg = [
                    {"role": "system", "content": "You are an evaluator. Follow output format exactly."},
                    {"role": "user", "content": f"{question}\n{answer_rule}"},
                ]
                text = _api_chat(cfg, model, msg)
            elif mode == "vlm_direct":
                msg = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{question}\n{answer_rule}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }
                ]
                text = _api_chat(cfg, model, msg)
            elif mode == "two_stage":
                caption_msg = [
                    {
                        "role": "system",
                        "content": "Describe visual evidence only. One short sentence.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": cfg.caption_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }
                ]
                caption = _api_chat(cfg, model, caption_msg, max_tokens=32)
                msg = [
                    {"role": "system", "content": "You are an evaluator. Follow output format exactly."},
                    {"role": "user", "content": f"{question}\nImage caption: {caption}\n{answer_rule}"},
                ]
                text = _api_chat(cfg, model, msg)
            elif mode == "rag":
                ctx = _retrieve_structured(question, rag_docs)
                msg = [
                    {"role": "system", "content": "You are an evaluator. Use retrieved context and follow output format exactly."},
                    {"role": "user", "content": f"{question}\nRetrieved context: {ctx}\n{answer_rule}"},
                ]
                text = _api_chat(cfg, model, msg)
            else:
                raise ValueError(mode)

            strict_pred_id = _decode_constrained_token(text, tokenizer, allowed)
            gt_id = int(batch["target_token_id"][i].item())
            total += 1

            if cfg.task_family == "single_image" and qa_mode in {"yesno", "yesno_set2"}:
                gt_bin = 1 if str(batch["answer_text"][i]).lower().startswith("yes") else 0
                if int(strict_pred_id) in yes_ids:
                    strict_bin = 1
                elif int(strict_pred_id) in no_ids:
                    strict_bin = 0
                else:
                    strict_bin = -1
                strict_correct += int(int(strict_bin) == int(gt_bin))
                semantic_correct += int(int(strict_bin) == int(gt_bin))
                y_true.append(int(gt_bin))
                y_pred_strict.append(int(strict_bin))
                y_pred_semantic.append(int(strict_bin))
            elif cfg.task_family == "single_image":
                strict_correct += int(strict_pred_id == gt_id)
                semantic_correct += int(strict_pred_id == gt_id)
                lut = {int(t): idx for idx, t in enumerate(allowed)} if allowed else {}
                y_true.append(lut.get(gt_id, 0))
                y_pred_strict.append(lut.get(strict_pred_id, 0))
                y_pred_semantic.append(lut.get(strict_pred_id, 0))
            else:
                strict_correct += int(strict_pred_id == gt_id)
                semantic_correct += int(strict_pred_id == gt_id)
                try:
                    pb = int(tokenizer.decode([strict_pred_id]).strip())
                except Exception:
                    pb = 0
                pf = min(1.0, max(0.0, pb / 10.0))
                pop_pred.append(pf)
                pop_true.append(float(batch["fraction"][i].item()))
            if cfg.verbose and total % max(1, int(cfg.log_every_samples)) == 0:
                print(f"[api_suite] progress mode={mode} model={model.name} samples={total}", flush=True)

    latency = (time.time() - t0) / max(1, total)
    metrics = {
        "accuracy": semantic_correct / max(1, total),
        "strict_accuracy": strict_correct / max(1, total),
    }
    if cfg.task_family == "single_image":
        # Use the constrained label space size (not observed-label cardinality),
        # because label_code indices can be sparse within a small eval slice.
        ncls = max(1, len(allowed))
        metrics["f1"] = macro_f1_from_ints(y_true, y_pred_semantic, num_classes=ncls)
        metrics["strict_f1"] = macro_f1_from_ints(y_true, y_pred_strict, num_classes=ncls)
    else:
        metrics["mae"] = mean_absolute_error(pop_true, pop_pred)
    return metrics, latency


def _save(rows: List[Dict], out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    keys: List[str] = []
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


def _mean_ci95(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    mean_v = float(sum(vals) / len(vals))
    if len(vals) == 1:
        return mean_v, 0.0, 0.0
    var = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
    std = var**0.5
    ci95 = 1.96 * std / (len(vals) ** 0.5)
    return float(mean_v), float(std), float(ci95)


def _aggregate_summary(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        key = (str(r.get("model_type", "")), str(r.get("dataset", "")), str(r.get("metric_name", "")))
        grouped.setdefault(key, []).append(r)

    out: List[Dict] = []
    for (model_type, dataset, metric_name), gr in grouped.items():
        ok = [x for x in gr if str(x.get("status", "")) == "ok"]
        fail = [x for x in gr if str(x.get("status", "")) != "ok"]
        metric_vals = [float(x.get("metric", 0.0)) for x in ok]
        latency_vals = [float(x.get("inference_latency", 0.0)) for x in ok]
        metric_mean, metric_std, metric_ci95 = _mean_ci95(metric_vals)
        lat_mean, _, lat_ci95 = _mean_ci95(latency_vals)
        row = {
            "model_type": model_type,
            "dataset": dataset,
            "metric_name": metric_name,
            "metric_mean": metric_mean,
            "metric_std": metric_std,
            "metric_ci95": metric_ci95,
            "latency_mean": lat_mean,
            "latency_ci95": lat_ci95,
            "ok_repeats": len(ok),
            "failed_repeats": len(fail),
            "failure_rate": float(len(fail) / max(1, len(gr))),
        }
        if ok:
            for k in ("accuracy", "strict_accuracy", "f1", "strict_f1", "mae"):
                vals = [float(x[k]) for x in ok if k in x and x[k] is not None]
                if vals:
                    m, _, c = _mean_ci95(vals)
                    row[f"{k}_mean"] = m
                    row[f"{k}_ci95"] = c
        out.append(row)
    return out


def _eval_with_repeats(
    cfg: APIBaselineSuiteConfig,
    model: APIModelSpec,
    mode: str,
    dataset_name: str,
    tokenizer,
    model_type: str,
) -> List[Dict]:
    rows: List[Dict] = []
    repeats = max(1, int(cfg.eval_repeats))
    for rep in range(repeats):
        rep_seed = int(cfg.seed + rep * cfg.repeat_seed_stride)
        if cfg.verbose:
            print(f"[api_suite] start dataset={dataset_name} model={model.name} mode={mode} repeat={rep+1}/{repeats} seed={rep_seed}", flush=True)
        set_seed(rep_seed)
        loader = _build_loader(cfg, tokenizer, dataset_name=dataset_name, seed=rep_seed)
        try:
            met, lat = _eval_api_model(cfg, model, loader, tokenizer, mode=mode)
            if cfg.verbose:
                print(f"[api_suite] done dataset={dataset_name} model={model.name} mode={mode} repeat={rep+1}/{repeats} metric={met.get('accuracy', met.get('mae', 0.0)):.4f}", flush=True)
            rows.append(
                {
                    "model_type": model_type,
                    "provider": model.provider,
                    "model_id": model.model_id,
                    "dataset": dataset_name,
                    "repeat_idx": rep,
                    "seed": rep_seed,
                    "metric": float(met.get("mae", met.get("accuracy", 0.0))) if cfg.task_family == "population" else float(met.get("accuracy", met.get("mae", 0.0))),
                    "metric_name": "mae" if cfg.task_family == "population" else "accuracy",
                    "params_trainable": 0,
                    "inference_latency": float(lat),
                    **met,
                    "status": "ok",
                }
            )
        except Exception as exc:
            if cfg.verbose:
                print(f"[api_suite] error dataset={dataset_name} model={model.name} mode={mode} repeat={rep+1}/{repeats} err={exc}", flush=True)
            rows.append(
                {
                    "model_type": model_type,
                    "provider": model.provider,
                    "model_id": model.model_id,
                    "dataset": dataset_name,
                    "repeat_idx": rep,
                    "seed": rep_seed,
                    "metric": 0.0,
                    "metric_name": "accuracy" if cfg.task_family == "single_image" else "mae",
                    "params_trainable": 0,
                    "inference_latency": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return rows


def run_api_baseline_suite(cfg: APIBaselineSuiteConfig) -> List[Dict]:
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B",
        cache_dir=cfg.hf_cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    datasets_to_run = list(cfg.dataset_names) if cfg.dataset_names else [cfg.dataset_name]
    rows: List[Dict] = []
    for dataset_name in datasets_to_run:
        if cfg.verbose:
            print(f"[api_suite] dataset={dataset_name} begin", flush=True)
        for m in cfg.llm_models:
            if not m.enabled:
                continue
            if cfg.enable_llm_only:
                rows.extend(_eval_with_repeats(cfg, m, mode="llm_only", dataset_name=dataset_name, tokenizer=tokenizer, model_type=f"api_llm_only::{m.name}"))
            if cfg.enable_rag:
                rows.extend(_eval_with_repeats(cfg, m, mode="rag", dataset_name=dataset_name, tokenizer=tokenizer, model_type=f"api_rag::{m.name}"))

        for v in cfg.vlm_models:
            if not v.enabled:
                continue
            if cfg.enable_vlm_direct:
                rows.extend(_eval_with_repeats(cfg, v, mode="vlm_direct", dataset_name=dataset_name, tokenizer=tokenizer, model_type=f"api_frozen_vlm::{v.name}"))
            if cfg.enable_two_stage:
                rows.extend(_eval_with_repeats(cfg, v, mode="two_stage", dataset_name=dataset_name, tokenizer=tokenizer, model_type=f"api_two_stage::{v.name}"))
        if cfg.verbose:
            print(f"[api_suite] dataset={dataset_name} end", flush=True)

    summary = _aggregate_summary(rows)
    _save(rows, cfg.out_csv, cfg.out_json)
    _save(summary, cfg.out_summary_csv, cfg.out_summary_json)
    return rows


def load_api_baseline_config(path: str) -> APIBaselineSuiteConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    llms = [APIModelSpec(**x) for x in raw.pop("llm_models", [])]
    vlms = [APIModelSpec(**x) for x in raw.pop("vlm_models", [])]
    cfg = APIBaselineSuiteConfig(**raw)
    if llms:
        cfg.llm_models = llms
    if vlms:
        cfg.vlm_models = vlms
    return cfg
