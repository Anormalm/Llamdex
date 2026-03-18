from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, DTD, OxfordIIITPet

from src.multimodal.tasks.prompts import build_population_prompt, class_names_for_dataset


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


def _single_token_ids(tokenizer, text: str) -> List[int]:
    ids = set()
    variants = [text, text.lower(), text.upper()]
    for v in variants:
        for prefix in (" ", ""):
            toks = tokenizer.encode(prefix + v, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return sorted(ids)


def _build_codebook(tokenizer, n_classes: int):
    candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()[]{}<>?/|")
    code_to_token_id = {}
    idx_to_code = {}
    c = 0
    for ch in candidates:
        one = _single_token_ids(tokenizer, ch)
        if not one:
            continue
        code_to_token_id[ch] = int(one[0])
        idx_to_code[c] = ch
        c += 1
        if c >= n_classes:
            break
    if len(idx_to_code) < n_classes:
        raise ValueError(f"Unable to build label_code map for {n_classes} classes with tokenizer.")
    return idx_to_code, code_to_token_id


def _vision_tfms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )


@dataclass
class _VisionRow:
    image: torch.Tensor
    label: int


def _load_rows(root: str, dataset_name: str, train: bool) -> List[_VisionRow]:
    tfm = _vision_tfms()
    rows: List[_VisionRow] = []
    if dataset_name == "cifar10":
        ds = CIFAR10(root=root, train=train, download=True, transform=tfm)
        for img, y in ds:
            rows.append(_VisionRow(image=img, label=int(y)))
        return rows
    if dataset_name == "cifar100":
        ds = CIFAR100(root=root, train=train, download=True, transform=tfm)
        for img, y in ds:
            rows.append(_VisionRow(image=img, label=int(y)))
        return rows
    if dataset_name == "dtd":
        split = "train" if train else "test"
        ds = DTD(root=root, split=split, download=True, transform=tfm)
        for img, y in ds:
            rows.append(_VisionRow(image=img, label=int(y)))
        return rows
    if dataset_name == "oxford_pet":
        split = "trainval" if train else "test"
        ds = OxfordIIITPet(root=root, split=split, target_types="category", download=True, transform=tfm)
        for img, y in ds:
            rows.append(_VisionRow(image=img, label=int(y)))
        return rows
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


class CIFARSingleImageQADataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool = True,
        dataset_name: str = "dtd",
        mode: str = "vision",
        qa_type: str = "label",
        seed: int = 42,
        max_samples: Optional[int] = None,
        description_file: Optional[str] = None,
    ):
        del description_file
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.mode = mode
        self.qa_type = qa_type
        self.effective_qa_type = qa_type
        self.rows = _load_rows(root=root, dataset_name=dataset_name, train=train)
        self.class_names = class_names_for_dataset(dataset_name)
        if max_samples is not None:
            rng = random.Random(seed)
            idxs = list(range(len(self.rows)))
            rng.shuffle(idxs)
            idxs = idxs[: max(0, int(max_samples))]
            self.rows = [self.rows[i] for i in idxs]
        self.idx_to_code = None
        self.code_to_token_id = None
        if qa_type == "label_code":
            self.idx_to_code, self.code_to_token_id = _build_codebook(tokenizer, len(self.class_names))

    def __len__(self):
        return len(self.rows)

    def _build_question_answer(self, label: int):
        if self.qa_type == "label_code":
            mapping = ", ".join([f"{self.idx_to_code[i]}={self.class_names[i]}" for i in range(len(self.class_names))])
            question = f"Classify this image. Answer with one code only. Codes: {mapping}."
            answer = self.idx_to_code[int(label)]
            target_token_id = int(self.code_to_token_id[answer])
            return question, answer, target_token_id
        if self.qa_type == "index":
            question = "Return the class index only."
            answer = str(int(label))
            ids = _single_token_ids(self.tokenizer, answer)
            target_token_id = int(ids[0]) if ids else int(self.tokenizer.encode(answer, add_special_tokens=False)[0])
            return question, answer, target_token_id
        if self.qa_type in {"yesno", "yesno_set2"}:
            # Deterministic binary question; "Yes" iff asked class equals label.
            asked = int((label + 1) % len(self.class_names))
            if self.qa_type == "yesno_set2":
                asked2 = int((label + 2) % len(self.class_names))
                question = (
                    f"Is this either {self.class_names[asked]} or {self.class_names[asked2]}? "
                    "Respond with exactly one token: Yes or No."
                )
                is_yes = int(label in {asked, asked2})
            else:
                question = f"Is this a {self.class_names[asked]}? Respond with exactly one token: Yes or No."
                is_yes = int(label == asked)
            answer = "Yes" if is_yes else "No"
            tid = _single_token_ids(self.tokenizer, answer)
            target_token_id = int(tid[0]) if tid else int(self.tokenizer.encode(answer, add_special_tokens=False)[0])
            return question, answer, target_token_id
        # default label mode
        code = chr(ord("A") + int(label)) if int(label) < 26 else str(int(label))
        question = "Classify this image. Answer with one label code only."
        answer = code
        tid = _single_token_ids(self.tokenizer, answer)
        target_token_id = int(tid[0]) if tid else int(self.tokenizer.encode(answer, add_special_tokens=False)[0])
        return question, answer, target_token_id

    def __getitem__(self, idx):
        row = self.rows[idx]
        question, answer, target_token_id = self._build_question_answer(row.label)
        messages = [
            {"role": "system", "content": "You are a precise classifier. Follow output format strictly."},
            {"role": "user", "content": question},
        ]
        prompt_ids = _chat_template_to_ids(self.tokenizer, messages)
        return {
            "images": row.image,
            "labels": int(row.label),
            "question_text": question,
            "answer_text": answer,
            "description": self.class_names[int(row.label)],
            "prompt_tokens": prompt_ids,
            "target_token_id": int(target_token_id),
        }


class CIFARPopulationDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool = True,
        dataset_name: str = "dtd",
        group_size: int = 8,
        output_mode: str = "integer",
        seed: int = 42,
        max_groups: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.group_size = int(group_size)
        self.output_mode = output_mode
        self.rows = _load_rows(root=root, dataset_name=dataset_name, train=train)
        self.class_names = class_names_for_dataset(dataset_name)
        rng = random.Random(seed)
        self.groups = []
        n = len(self.rows)
        max_groups = max_groups if max_groups is not None else max(1, n // max(1, self.group_size))
        for _ in range(int(max_groups)):
            idxs = [rng.randrange(n) for _ in range(self.group_size)]
            target_class = rng.randrange(len(self.class_names))
            labels = [self.rows[i].label for i in idxs]
            frac = float(sum(1 for y in labels if y == target_class) / float(self.group_size))
            self.groups.append((idxs, target_class, frac))

    def __len__(self):
        return len(self.groups)

    def _answer_token(self, frac: float):
        if self.output_mode == "integer":
            val = str(int(round(frac * 10.0)))
            tok = _single_token_ids(self.tokenizer, val)
            tid = int(tok[0]) if tok else int(self.tokenizer.encode(val, add_special_tokens=False)[0])
            return val, tid
        bin_idx = min(9, max(0, int(frac * 10.0)))
        val = chr(ord("A") + bin_idx)
        tok = _single_token_ids(self.tokenizer, val)
        tid = int(tok[0]) if tok else int(self.tokenizer.encode(val, add_special_tokens=False)[0])
        return val, tid

    def __getitem__(self, idx):
        idxs, target_class, frac = self.groups[idx]
        imgs = torch.stack([self.rows[i].image for i in idxs], dim=0)
        question = build_population_prompt(self.class_names[int(target_class)], output_mode=self.output_mode)
        answer, target_tid = self._answer_token(frac)
        messages = [
            {"role": "system", "content": "Use one-token answers only."},
            {"role": "user", "content": question},
        ]
        prompt_ids = _chat_template_to_ids(self.tokenizer, messages)
        return {
            "images": imgs,
            "target_class": int(target_class),
            "fraction": float(frac),
            "question_text": question,
            "answer_text": answer,
            "prompt_tokens": prompt_ids,
            "target_token_id": int(target_tid),
        }


def _pad_prompt_tokens(seqs: Sequence[torch.Tensor], pad_token_id: int):
    max_len = max(int(s.size(0)) for s in seqs)
    bsz = len(seqs)
    tokens = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        sl = int(s.size(0))
        tokens[i, max_len - sl :] = s
        mask[i, max_len - sl :] = 1
    return tokens, mask


def collate_single_image(batch: Sequence[Dict]):
    pad_id = 0
    seqs = [x["prompt_tokens"] for x in batch]
    tokens, mask = _pad_prompt_tokens(seqs, pad_id)
    return {
        "images": torch.stack([x["images"] for x in batch], dim=0),
        "labels": torch.tensor([int(x["labels"]) for x in batch], dtype=torch.long),
        "prompt_tokens": tokens,
        "prompt_mask": mask,
        "target_token_id": torch.tensor([int(x["target_token_id"]) for x in batch], dtype=torch.long),
        "question_text": [x["question_text"] for x in batch],
        "answer_text": [x["answer_text"] for x in batch],
        "description": [x["description"] for x in batch],
    }


def collate_population(batch: Sequence[Dict]):
    pad_id = 0
    seqs = [x["prompt_tokens"] for x in batch]
    tokens, mask = _pad_prompt_tokens(seqs, pad_id)
    return {
        "images": torch.stack([x["images"] for x in batch], dim=0),
        "target_class": torch.tensor([int(x["target_class"]) for x in batch], dtype=torch.long),
        "fraction": torch.tensor([float(x["fraction"]) for x in batch], dtype=torch.float32),
        "prompt_tokens": tokens,
        "prompt_mask": mask,
        "target_token_id": torch.tensor([int(x["target_token_id"]) for x in batch], dtype=torch.long),
        "question_text": [x["question_text"] for x in batch],
        "answer_text": [x["answer_text"] for x in batch],
    }

