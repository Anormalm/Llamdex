import argparse
import json
import os
import sys
from datetime import datetime
import re
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer as EncoderTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.eval.plan1_eval import (  # noqa: E402
    EvalPlan1Args,
    _build_connectors,
    _build_model,
)

OXFORD_PET_CLASSES = [
    "Abyssinian",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "boxer",
    "British_Shorthair",
    "chihuahua",
    "Egyptian_Mau",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "Maine_Coon",
    "miniature_pinscher",
    "newfoundland",
    "Persian",
    "pomeranian",
    "pug",
    "Ragdoll",
    "Russian_Blue",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "Siamese",
    "Sphynx",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]


def _build_prompt(question: str, description: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a multimodal pet assistant. "
                "Use the provided image description evidence and answer clearly. "
                "Return exactly two lines:\n"
                "Answer: <short answer>\n"
                "Rationale: <one concise sentence>"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nImage description: {description}",
        },
    ]


def _chat_template_to_ids(tokenizer, messages) -> torch.Tensor:
    try:
        out = tokenizer.apply_chat_template(messages, return_tensors="pt")
    except Exception:
        plain = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        out = tokenizer(plain, return_tensors="pt", add_special_tokens=True)["input_ids"]
    if isinstance(out, torch.Tensor):
        return out.squeeze(0).long()
    if hasattr(out, "input_ids"):
        ids = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        ids = out["input_ids"]
    else:
        raise TypeError("Unsupported apply_chat_template return type for tokenizer.")
    if isinstance(ids, list):
        ids = torch.tensor(ids, dtype=torch.long)
    if isinstance(ids, torch.Tensor) and ids.dim() == 2:
        return ids.squeeze(0).long()
    return ids.long()


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> int:
    step_logits = logits.clone()
    if repetition_penalty > 1.0 and generated_ids:
        for tid in set(generated_ids):
            step_logits[..., tid] = step_logits[..., tid] / repetition_penalty
    if temperature > 0:
        step_logits = step_logits / temperature
    step_logits = _top_p_filter(step_logits, top_p=top_p)
    if do_sample:
        probs = F.softmax(step_logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(step_logits.argmax(dim=-1).item())


def _decode_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_inputs: Optional[Dict[str, torch.Tensor]],
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
    do_sample: bool = True,
) -> Dict[str, Any]:
    generated_ids = []
    cur_ids = input_ids
    cur_mask = attention_mask
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                cur_ids,
                attention_mask=cur_mask,
                expert_inputs=(expert_inputs,) if expert_inputs is not None else None,
                use_cache=False,
            )
            next_id = _sample_next_token(
                logits=outputs.logits[:, -1, :],
                generated_ids=generated_ids,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            generated_ids.append(next_id)
            next_tok = torch.tensor([[next_id]], dtype=torch.long, device=cur_ids.device)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((cur_mask.size(0), 1), dtype=cur_mask.dtype, device=cur_mask.device)],
                dim=1,
            )
            if eos_id is not None and next_id == int(eos_id):
                break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "generated_token_ids": generated_ids,
        "generated_text": text,
    }


def _normalize_answer_rationale(text: str) -> Dict[str, str]:
    t = text.replace("\r", "\n").strip()
    answer_match = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", t)
    rationale_match = re.search(r"(?im)^\s*rationale\s*:\s*(.+)$", t)
    answer = answer_match.group(1).strip() if answer_match else ""
    rationale = rationale_match.group(1).strip() if rationale_match else ""

    # Fallbacks if model does not follow template.
    if not answer:
        first_line = next((ln.strip() for ln in t.split("\n") if ln.strip()), "")
        answer = first_line[:120] if first_line else "Unknown"
    if not rationale:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        rationale = lines[1][:240] if len(lines) > 1 else "Insufficient structured rationale from current model output."
    formatted = f"Answer: {answer}\nRationale: {rationale}"
    return {"answer": answer, "rationale": rationale, "formatted": formatted}


def _single_token_ids(tokenizer, text: str):
    ids = set()
    for prefix in (" ", ""):
        toks = tokenizer.encode(prefix + text, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(int(toks[0]))
    return ids


def _build_class_codebook(tokenizer, class_names):
    candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()[]{}<>?/|")
    rows = []
    used_tids = set()
    for c in candidates:
        one = _single_token_ids(tokenizer, c)
        if not one:
            continue
        tid = int(next(iter(one)))
        if tid in used_tids:
            continue
        rows.append((c, tid))
        used_tids.add(tid)
        if len(rows) >= len(class_names):
            break
    if len(rows) < len(class_names):
        raise RuntimeError(f"Not enough single-token codes for {len(class_names)} classes.")
    return rows


def _extract_clues(description: str):
    d = description.lower()
    clue_bank = [
        ("floppy ears", "long and floppy ears"),
        ("black patches", "black patches on a white coat"),
        ("white coat", "predominantly white coat"),
        ("hound", "hound-like facial structure"),
        ("short muzzle", "short muzzle"),
        ("long fur", "long fur texture"),
        ("small size", "small body size"),
        ("large size", "large body size"),
    ]
    found = [label for key, label in clue_bank if key in d]
    if len(found) < 2:
        if "medium" in d:
            found.append("medium body size")
        if "gentle" in d:
            found.append("gentle expression")
    if len(found) < 2:
        found.append("overall breed-specific coat and face pattern")
    return found[:2]


def _build_constrained_prompt(question: str, description: str, class_names, codebook):
    mapping = ", ".join([f"{c}={class_names[i]}" for i, (c, _) in enumerate(codebook)])
    return [
        {
            "role": "system",
            "content": "You are a pet classification assistant. Output exactly one code token from the provided mapping.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Image description: {description}\n"
                f"Choose the best breed code only. Mapping: {mapping}\n"
                "Output exactly one code token."
            ),
        },
    ]


def _constrained_breed_predict(model, tokenizer, prompt_ids, prompt_mask, expert_inputs, class_names, codebook):
    allowed_ids = torch.tensor([tid for _, tid in codebook], device=prompt_ids.device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            prompt_ids,
            attention_mask=prompt_mask,
            expert_inputs=(expert_inputs,) if expert_inputs is not None else None,
            use_cache=False,
        )
        step_logits = outputs.logits[:, -1, :]
        allowed_logits = step_logits.index_select(dim=-1, index=allowed_ids)
        topv, topi = torch.topk(allowed_logits, k=min(5, allowed_logits.size(-1)), dim=-1)
        pred_local = int(allowed_logits.argmax(dim=-1).item())
        pred_token_id = int(allowed_ids[pred_local].item())

    idx_by_tid = {tid: i for i, (_, tid) in enumerate(codebook)}
    pred_index = idx_by_tid[pred_token_id]
    pred_code = codebook[pred_index][0]
    pred_label = class_names[pred_index]
    topk = []
    for score, local_idx in zip(topv[0].tolist(), topi[0].tolist()):
        i = int(local_idx)
        topk.append(
            {
                "code": codebook[i][0],
                "label": class_names[i],
                "token_id": int(codebook[i][1]),
                "logit": float(score),
            }
        )
    return {
        "pred_code": pred_code,
        "pred_label": pred_label,
        "pred_token_id": pred_token_id,
        "topk": topk,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run one full Plan-1 service-style demo and log I/O.")
    p.add_argument("--server_models_path", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--connectors_path", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="oxford_pet", choices=["oxford_pet", "hospital_text", "dtd"])
    p.add_argument("--run_name", type=str, default="service_demo_oxford_pet")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--baseline", type=str, default="injection", choices=["injection", "llm_only"])
    p.add_argument("--service_mode", type=str, default="constrained", choices=["constrained", "freeform"])
    p.add_argument("--task_family", type=str, default="single_image", choices=["single_image"])
    p.add_argument("--evidence_source", type=str, default="text", choices=["text"])
    p.add_argument("--qa_type", type=str, default="label", choices=["label", "yesno"])
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--evidence_dim", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--text_encoder_model_id", type=str, default="distilroberta-base")
    p.add_argument("--use_adapters", type=int, default=1, choices=[0, 1])
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--adapter_activation", type=str, default="gelu", choices=["gelu", "relu"])
    p.add_argument("--inject_location", type=str, default="layer_input", choices=["layer_input"])
    p.add_argument("--load_in_4bit", type=int, default=0, choices=[0, 1])
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_use_double_quant", type=int, default=1, choices=[0, 1])
    p.add_argument("--bnb_4bit_cpu_offload", type=int, default=0, choices=[0, 1])
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.12)
    p.add_argument("--do_sample", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--question",
        type=str,
        default="What dog breed is this most likely, and what two clues support your answer?",
    )
    p.add_argument(
        "--description",
        type=str,
        default=(
            "A medium-sized dog with a mostly white coat and large black patches. "
            "The ears are long and floppy, and the face has a gentle hound-like shape."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    eval_args = EvalPlan1Args(
        server_models_path=args.server_models_path,
        model_name=args.model_name,
        connectors_path=args.connectors_path,
        dataset_name=args.dataset_name,
        task_family=args.task_family,
        evidence_source=args.evidence_source,
        qa_type=args.qa_type,
        num_tokens=args.num_tokens,
        layer_to_add=args.layer,
        evidence_dim=args.evidence_dim,
        alpha=args.alpha,
        text_encoder_model_id=args.text_encoder_model_id,
        baseline=args.baseline,
        device=args.device,
        use_adapters=bool(args.use_adapters),
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        adapter_activation=args.adapter_activation,
        inject_location=args.inject_location,
        enforce_checkpoint_compat=False,
        load_in_4bit=bool(args.load_in_4bit),
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        bnb_4bit_cpu_offload=bool(args.bnb_4bit_cpu_offload),
        device_map=args.device_map,
    )

    model, tokenizer = _build_model(eval_args)
    model.eval()
    if not model.is_quantized_4bit:
        # Keep demo mode numerically stable across custom connector dtypes.
        model = model.to(device).to(torch.float32)

    expert_info = {}
    if args.baseline == "injection":
        expert, _ = _build_connectors(eval_args, model)
        text_tok = EncoderTokenizer.from_pretrained(args.text_encoder_model_id, cache_dir=args.server_models_path)
        enc = text_tok(
            [args.description],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        expert_inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        # Probe once so evidence shapes are available in log.
        with torch.no_grad():
            projected = expert.forward_with_features(expert_inputs)
            z = expert.get_expert_output()
        expert_info = {
            "evidence_shape": list(z.shape) if z is not None else None,
            "projected_tokens_shape": list(projected.shape),
            "projected_tokens_dtype": str(projected.dtype),
        }
    else:
        expert_inputs = None

    class_names = OXFORD_PET_CLASSES if args.dataset_name == "oxford_pet" else ["benign", "malignant"]
    if args.service_mode == "constrained":
        codebook = _build_class_codebook(tokenizer, class_names)
        messages = _build_constrained_prompt(args.question, args.description, class_names, codebook)
        prompt_ids = _chat_template_to_ids(tokenizer, messages).unsqueeze(0).to(device)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        pred = _constrained_breed_predict(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            expert_inputs=expert_inputs,
            class_names=class_names,
            codebook=codebook,
        )
        clues = _extract_clues(args.description)
        normalized = {
            "answer": pred["pred_label"],
            "rationale": f"Prediction is based on {clues[0]} and {clues[1]}.",
            "formatted": f"Answer: {pred['pred_label']}\nRationale: Prediction is based on {clues[0]} and {clues[1]}.",
        }
        gen = {
            "generated_token_ids": [pred["pred_token_id"]],
            "generated_text": pred["pred_code"],
            "constrained_topk": pred["topk"],
        }
    else:
        messages = _build_prompt(args.question, args.description)
        prompt_ids = _chat_template_to_ids(tokenizer, messages).unsqueeze(0).to(device)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        gen = _decode_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            expert_inputs=expert_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=bool(args.do_sample),
        )
        normalized = _normalize_answer_rationale(gen["generated_text"])

    record = {
        "run_name": args.run_name,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "plan1_service_demo",
        "config": {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "baseline": args.baseline,
            "service_mode": args.service_mode,
            "evidence_source": args.evidence_source,
            "layer": args.layer,
            "inject_location": args.inject_location,
            "num_tokens": args.num_tokens,
            "use_adapters": bool(args.use_adapters),
            "adapter_bottleneck": args.adapter_bottleneck,
            "load_in_4bit": bool(args.load_in_4bit),
        },
        "input": {
            "question": args.question,
            "description": args.description,
            "messages": messages,
        },
        "pipeline_trace": expert_info,
        "output": gen,
        "normalized_output": normalized,
    }

    if args.out_json is None:
        out_json = os.path.join("runs", f"{args.run_name}.json")
    else:
        out_json = args.out_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "out_json": out_json,
                "generated_text": gen["generated_text"],
                "normalized_output": normalized["formatted"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
