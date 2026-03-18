from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _token_id_for_word(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return int(ids[-1])


def _chat_template_tokens(tokenizer, msgs) -> torch.Tensor:
    out = tokenizer.apply_chat_template(msgs, return_tensors="pt")
    if isinstance(out, torch.Tensor):
        ids = out
    elif hasattr(out, "input_ids"):
        ids = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        ids = out["input_ids"]
    else:
        raw_ids = tokenizer.apply_chat_template(msgs, tokenize=True)
        ids = torch.tensor(raw_ids, dtype=torch.long).unsqueeze(0)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.squeeze(0).long()


def _build_codebook(tokenizer, n: int):
    candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()[]{}<>?/|")
    codes = []
    code_to_tid = {}
    used = set()
    for c in candidates:
        ids = tokenizer.encode(" " + c, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = int(ids[0])
        if tid in used:
            continue
        used.add(tid)
        codes.append(c)
        code_to_tid[c] = tid
        if len(codes) >= n:
            break
    if len(codes) < n:
        raise RuntimeError(f"Could not build enough single-token codes for n={n}.")
    return codes, code_to_tid


@dataclass
class VQASubsetSpec:
    hf_dataset_name: str = "Graphcore/gqa-lxmert"
    split: str = "validation[:1024]"
    image_field: str = "image"
    question_field: str = "question"
    answer_field: str = "answer"


class VQASubsetDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        spec: VQASubsetSpec,
        max_samples: Optional[int] = 512,
        seed: int = 42,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.ds = load_dataset(spec.hf_dataset_name, split=spec.split)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.image_field = spec.image_field
        self.question_field = spec.question_field
        self.answer_field = spec.answer_field
        self.tx = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        answers = []
        for r in self.ds:
            a = r[self.answer_field]
            if isinstance(a, list):
                a = a[0]
            answers.append(str(a).strip().lower())
        uniq = sorted(set(answers))
        self.answer_vocab = uniq
        self.codes, self.code_to_tid = _build_codebook(tokenizer, len(self.answer_vocab))
        self.ans_to_idx = {a: i for i, a in enumerate(self.answer_vocab)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row[self.image_field]
        img = self.tx(img.convert("RGB"))
        q = str(row[self.question_field]).strip()
        a = row[self.answer_field]
        if isinstance(a, list):
            a = a[0]
        ans = str(a).strip().lower()
        ans_idx = self.ans_to_idx.get(ans, 0)
        code = self.codes[ans_idx]
        mapping = ", ".join([f"{self.codes[i]}={w}" for i, w in enumerate(self.answer_vocab[:50])])
        prompt = f"Question: {q} Answer with one code only. Codes: {mapping}."
        msgs = [{"role": "system", "content": "Answer with one code only."}, {"role": "user", "content": prompt}]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        mask = (tok != self.tokenizer.pad_token_id).long()
        return {
            "images": img,
            "prompt_tokens": tok,
            "prompt_mask": mask,
            "target_token_id": torch.tensor(self.code_to_tid[code], dtype=torch.long),
            "answer_idx": torch.tensor(ans_idx, dtype=torch.long),
            "task_name": "vqa",
            "allowed_token_ids": list(self.code_to_tid.values()),
        }


class FineGrainedPetDataset(Dataset):
    def __init__(self, root: str, tokenizer, train: bool, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        split = "trainval" if train else "test"
        self.ds = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        )
        self.class_names = list(self.ds.classes)
        self.codes, self.code_to_tid = _build_codebook(tokenizer, len(self.class_names))
        self.indices = list(range(len(self.ds)))
        if max_samples is not None:
            self.indices = self.indices[: max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image, label = self.ds[i]
        mapping = ", ".join([f"{self.codes[i]}={name}" for i, name in enumerate(self.class_names)])
        prompt = f"Classify this pet image. Answer with one code only. Codes: {mapping}."
        msgs = [{"role": "system", "content": "Answer with one code only."}, {"role": "user", "content": prompt}]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        mask = (tok != self.tokenizer.pad_token_id).long()
        code = self.codes[int(label)]
        return {
            "images": image,
            "prompt_tokens": tok,
            "prompt_mask": mask,
            "target_token_id": torch.tensor(self.code_to_tid[code], dtype=torch.long),
            "label_idx": torch.tensor(int(label), dtype=torch.long),
            "task_name": "finegrained",
            "allowed_token_ids": list(self.code_to_tid.values()),
            "rationale_keywords": self.class_names[int(label)].replace("_", " ").split(),
        }


class PopulationBagDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool,
        group_size: int = 8,
        max_groups: Optional[int] = 256,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.rng = random.Random(seed)
        split = "trainval" if train else "test"
        self.ds = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        )
        self.class_names = list(self.ds.classes)
        self.n = len(self.ds) // group_size if max_groups is None else max_groups

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = [self.rng.randrange(len(self.ds)) for _ in range(self.group_size)]
        batch = [self.ds[i] for i in ids]
        images = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
        target = self.rng.randrange(len(self.class_names))
        frac = (labels == target).float().mean().item()
        bin_id = int(round(frac * 10.0))
        prompt = (
            f"In this bag of images, what fraction are class {self.class_names[target]}? "
            "Answer with one integer 0 to 10."
        )
        msgs = [{"role": "system", "content": "Answer with one integer only."}, {"role": "user", "content": prompt}]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        mask = (tok != self.tokenizer.pad_token_id).long()
        return {
            "images": images,
            "prompt_tokens": tok,
            "prompt_mask": mask,
            "target_token_id": torch.tensor(_token_id_for_word(self.tokenizer, str(bin_id)), dtype=torch.long),
            "fraction": torch.tensor(frac, dtype=torch.float32),
            "bin_id": torch.tensor(bin_id, dtype=torch.long),
            "target_class": torch.tensor(target, dtype=torch.long),
            "task_name": "population",
            "allowed_token_ids": [_token_id_for_word(self.tokenizer, str(i)) for i in range(11)],
        }


class GroundedGenerationDataset(FineGrainedPetDataset):
    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        label_words = row["rationale_keywords"]
        prompt = (
            f"Identify the pet breed. First output one code token. Then output one short rationale sentence.\n"
            f"Format: Answer: <code>. Rationale: <short sentence>."
        )
        msgs = [{"role": "system", "content": "Follow format strictly."}, {"role": "user", "content": prompt}]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        row["prompt_tokens"] = tok
        row["prompt_mask"] = (tok != self.tokenizer.pad_token_id).long()
        row["task_name"] = "grounded_generation"
        row["rationale_target"] = " ".join(label_words)
        return row


def collate_task_batch(batch: List[Dict]) -> Dict:
    max_len = max(x["prompt_tokens"].size(0) for x in batch)
    pad_id = 0
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        l = row["prompt_tokens"].size(0)
        tokens[i, max_len - l :] = row["prompt_tokens"]
        mask[i, max_len - l :] = row["prompt_mask"]

    out = {
        "prompt_tokens": tokens,
        "prompt_mask": mask,
        "target_token_id": torch.stack([x["target_token_id"] for x in batch], dim=0),
        "task_name": batch[0]["task_name"],
        "allowed_token_ids": batch[0]["allowed_token_ids"],
    }
    if batch[0]["task_name"] == "population":
        out["images"] = torch.stack([x["images"] for x in batch], dim=0)
        out["fraction"] = torch.stack([x["fraction"] for x in batch], dim=0)
        out["bin_id"] = torch.stack([x["bin_id"] for x in batch], dim=0)
        out["target_class"] = torch.stack([x["target_class"] for x in batch], dim=0)
    else:
        out["images"] = torch.stack([x["images"] for x in batch], dim=0)
    if "label_idx" in batch[0]:
        out["label_idx"] = torch.stack([x["label_idx"] for x in batch], dim=0)
    if "answer_idx" in batch[0]:
        out["answer_idx"] = torch.stack([x["answer_idx"] for x in batch], dim=0)
    if "rationale_keywords" in batch[0]:
        out["rationale_keywords"] = [x["rationale_keywords"] for x in batch]
    if "rationale_target" in batch[0]:
        out["rationale_target"] = [x["rationale_target"] for x in batch]
    return out
