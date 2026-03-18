from __future__ import annotations

import json
import random
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset


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


def _single_token_id(tokenizer, text: str) -> int:
    for t in (text, text.lower(), text.upper()):
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + t, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0])


class HospitalTextQADataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        qa_type: str = "label",
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.qa_type = qa_type
        self.rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if max_samples is not None and len(self.rows) > int(max_samples):
            rng = random.Random(seed)
            idxs = list(range(len(self.rows)))
            rng.shuffle(idxs)
            self.rows = [self.rows[i] for i in idxs[: int(max_samples)]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        question = str(row.get("question", "Classify the case."))
        answer = str(row.get("answer", row.get("label", "0")))
        desc = str(row.get("description", row.get("text", "")))
        target_tid = _single_token_id(self.tokenizer, answer)
        messages = [
            {"role": "system", "content": "Answer with one short token."},
            {"role": "user", "content": question},
        ]
        prompt_ids = _chat_template_to_ids(self.tokenizer, messages)
        return {
            "prompt_tokens": prompt_ids,
            "target_token_id": int(target_tid),
            "question_text": question,
            "answer_text": answer,
            "description": desc,
            "labels": int(row.get("label_id", 0)),
        }


def collate_hospital_text(batch: Sequence[Dict]):
    max_len = max(int(x["prompt_tokens"].size(0)) for x in batch)
    tokens = torch.full((len(batch), max_len), 0, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        t = item["prompt_tokens"]
        sl = int(t.size(0))
        tokens[i, max_len - sl :] = t
        mask[i, max_len - sl :] = 1
    return {
        "prompt_tokens": tokens,
        "prompt_mask": mask,
        "target_token_id": torch.tensor([int(x["target_token_id"]) for x in batch], dtype=torch.long),
        "question_text": [x["question_text"] for x in batch],
        "answer_text": [x["answer_text"] for x in batch],
        "description": [x["description"] for x in batch],
        "labels": torch.tensor([int(x["labels"]) for x in batch], dtype=torch.long),
    }

