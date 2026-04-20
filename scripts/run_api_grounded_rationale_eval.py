import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.multimodal.baselines.api_suite import APIBaselineSuiteConfig, APIModelSpec, _api_chat, _prepare_image_for_api
from src.multimodal.task_matrix.datasets import GroundedGenerationDataset, collate_task_batch
from src.multimodal.task_matrix.metrics import keyword_consistency_score
from src.multimodal.utils.repro import set_seed


def _extract_answer_rationale(text: str):
    t = str(text)
    ans_matches = re.findall(r"(?im)\banswer\s*:\s*([^\n\.]+)", t)
    rat_matches = re.findall(r"(?ims)\brationale\s*:\s*(.+?)(?=\banswer\s*:|$)", t)
    answer_text = ans_matches[-1].strip() if ans_matches else ""
    rationale_text = rat_matches[-1].strip() if rat_matches else ""
    if not answer_text:
        m = re.search(r"([A-Za-z0-9!@#$%^&*()\[\]{}<>?/|])", t)
        answer_text = m.group(1) if m else ""
    if not rationale_text:
        rationale_text = t.strip()
    return answer_text, rationale_text


def _normalize_code_text(s: str) -> str:
    m = re.search(r"[A-Za-z0-9!@#$%^&*()\\[\\]{}<>?/|]", str(s))
    return m.group(0) if m else ""


def _eval_one_model(cfg: APIBaselineSuiteConfig, model: APIModelSpec, loader, tokenizer, max_new_tokens: int) -> Dict:
    total = 0
    fmt_ok = 0
    rationale_nonempty = 0
    faithful = 0
    halluc = 0
    answer_valid = 0
    answer_correct = 0
    kw_scores: List[float] = []

    for batch in loader:
        tokens = batch["prompt_tokens"]
        targets = batch["target_token_id"]
        images = batch["images"]
        kws_all = batch.get("rationale_keywords", [[] for _ in range(tokens.size(0))])

        for i in range(tokens.size(0)):
            prompt_text = tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True)
            img_b64 = _prepare_image_for_api(images[i])
            messages = [
                {"role": "system", "content": "Follow format strictly. Output: Answer: <code>. Rationale: <short sentence>."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ]
            out_text = _api_chat(cfg, model, messages, max_tokens=max_new_tokens, temperature=0.0)
            ans_text, rat_text = _extract_answer_rationale(out_text)
            if ans_text and rat_text:
                fmt_ok += 1
            if len(rat_text) >= 12:
                rationale_nonempty += 1

            pred_code = _normalize_code_text(ans_text)
            true_code = _normalize_code_text(tokenizer.decode([int(targets[i].item())], skip_special_tokens=True))
            if pred_code:
                answer_valid += 1
            if pred_code and pred_code == true_code:
                answer_correct += 1

            kw_score = keyword_consistency_score(rat_text or out_text, kws_all[i])
            kw_scores.append(float(kw_score))
            if kw_score <= 0.0:
                halluc += 1
            if kw_score >= 0.6 and pred_code == true_code and len(rat_text) >= 12:
                faithful += 1
            total += 1

    return {
        "model_type": f"api_frozen_vlm::{model.name}",
        "dataset": "dtd",
        "samples": total,
        "rationale_faithful_rate": faithful / max(1, total),
        "rationale_hallucination_rate": halluc / max(1, total),
        "rationale_consistency": (sum(kw_scores) / max(1, len(kw_scores))) if kw_scores else 0.0,
        "format_compliance": fmt_ok / max(1, total),
        "rationale_nonempty_rate": rationale_nonempty / max(1, total),
        "answer_code_valid_rate": answer_valid / max(1, total),
        "answer_code_accuracy": answer_correct / max(1, total),
        "status": "ok",
    }


def parse_args():
    p = argparse.ArgumentParser(description="API grounded-rationale evaluator")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--data_root", type=str, default="/disk1/lfhu/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_samples", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--out_csv", type=str, default="/disk1/lfhu/runs/api_grounded_generation.remote.csv")
    p.add_argument("--out_json", type=str, default="/disk1/lfhu/runs/api_grounded_generation.remote.json")
    p.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-4o-mini"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.server_models_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    eval_ds = GroundedGenerationDataset(
        root=args.data_root,
        tokenizer=tokenizer,
        train=False,
        max_samples=args.max_eval_samples,
        dataset_name="dtd",
    )
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_task_batch)

    cfg = APIBaselineSuiteConfig()
    rows = []
    for m in args.models:
        spec = APIModelSpec(name=m, model_id=m, enabled=True, is_vision=True, provider="openai_compatible")
        print(f"[api_grounded] running model={m}")
        rows.append(_eval_one_model(cfg, spec, eval_loader, tokenizer, args.max_new_tokens))

    # upper model by faithful rate (tie-break lower hallucination)
    upper = sorted(rows, key=lambda r: (-float(r["rationale_faithful_rate"]), float(r["rationale_hallucination_rate"])))[0]

    out = {
        "rows": rows,
        "upper": upper,
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        keys = list(rows[0].keys()) if rows else ["status"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("CSV:", args.out_csv)
    print("JSON:", args.out_json)
    print("UPPER faithful=", upper["rationale_faithful_rate"], "halluc=", upper["rationale_hallucination_rate"], "model=", upper["model_type"])


if __name__ == "__main__":
    main()
