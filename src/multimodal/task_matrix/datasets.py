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
    try:
        out = tokenizer.apply_chat_template(msgs, return_tensors="pt")
    except Exception:
        plain = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
        out = tokenizer(plain, return_tensors="pt", add_special_tokens=True)["input_ids"]
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
        return ids.long()
    if ids.ndim == 2:
        # Keep logic robust across tokenizer/template variants.
        return ids[0].long()
    raise ValueError(f"Unexpected token shape from chat template: {tuple(ids.shape)}")


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


def _build_image_dataset(root: str, dataset_name: str, train: bool):
    tx = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    name = str(dataset_name).strip().lower()
    if name == "oxford_pet":
        split = "trainval" if train else "test"
        ds = datasets.OxfordIIITPet(root=root, split=split, target_types="category", download=True, transform=tx)
        return ds, list(ds.classes), "pet breed"
    if name == "dtd":
        split = "train" if train else "test"
        ds = datasets.DTD(root=root, split=split, download=True, transform=tx)
        return ds, list(ds.classes), "texture class"
    if name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tx)
        return ds, list(ds.classes), "object class"
    raise ValueError(f"Unsupported image dataset for task matrix: {dataset_name}")


def _dataset_prompt_tokens(dataset_name: str) -> Dict[str, str]:
    name = str(dataset_name).strip().lower()
    if name == "dtd":
        return {
            "schema": "DTD schema",
            "class_var": "texture_class",
            "evidence_phrase": "texture-grounded evidence sentence",
            "finegrained": "Classify this texture image. Return `label_id` only from DTD schema. No explanation.",
            "strict_yesno": "Question: Is this image `<texture_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this image bag, estimate fraction of `<texture_class>`. Return one integer in `[0..10]` only.",
        }
    if name == "oxford_pet":
        return {
            "schema": "Oxford-IIIT Pet schema",
            "class_var": "breed_name",
            "evidence_phrase": "breed-grounded evidence sentence",
            "finegrained": "Classify this pet image by breed. Return `label_id` only from Oxford-IIIT Pet schema. No explanation.",
            "strict_yesno": "Question: Is this image `<breed_name>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this image bag, estimate fraction of `<breed_name>`. Return one integer in `[0..10]` only.",
        }
    if name == "cifar10":
        return {
            "schema": "CIFAR-10 schema",
            "class_var": "cifar_class",
            "evidence_phrase": "object-grounded evidence sentence",
            "finegrained": "Classify this image into CIFAR-10 classes. Return `label_id` only. No explanation.",
            "strict_yesno": "Question: Is this image `<cifar_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this image bag, estimate fraction of `<cifar_class>`. Return one integer in `[0..10]` only.",
        }
    return {
        "schema": "label schema",
        "class_var": "class_name",
        "evidence_phrase": "image-grounded evidence sentence",
        "finegrained": "Classify this image. Return `label_id` only. No explanation.",
        "strict_yesno": "Question: Is this image `<class_name>`? Return exactly one token: `Yes` or `No`.",
        "population": "In this image bag, estimate fraction of `<class_name>`. Return one integer in `[0..10]` only.",
    }


class FineGrainedPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool,
        max_samples: Optional[int] = None,
        dataset_name: str = "oxford_pet",
        prompt_template_style: str = "legacy",
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.prompt_template_style = str(prompt_template_style).strip().lower()
        self.ds, self.class_names, self.task_label = _build_image_dataset(root=root, dataset_name=dataset_name, train=train)
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
        if self.prompt_template_style == "compact":
            p = _dataset_prompt_tokens(self.dataset_name)
            prompt = f"{p['finegrained']}\nLabel IDs: {mapping}"
            msgs = [{"role": "system", "content": "Return only one label_id token from the label schema."}, {"role": "user", "content": prompt}]
        else:
            prompt = f"Classify this {self.task_label}. Answer with one code only. Codes: {mapping}."
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
            "code_labels": list(self.class_names),
            "rationale_keywords": self.class_names[int(label)].replace("_", " ").split(),
        }


class PopulationBagDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool,
        dataset_name: str = "oxford_pet",
        group_size: int = 8,
        max_groups: Optional[int] = 256,
        seed: int = 42,
        prompt_template_style: str = "legacy",
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.prompt_template_style = str(prompt_template_style).strip().lower()
        self.group_size = group_size
        self.rng = random.Random(seed)
        self.ds, self.class_names, self.task_label = _build_image_dataset(root=root, dataset_name=dataset_name, train=train)
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
        if self.prompt_template_style == "compact":
            p = _dataset_prompt_tokens(self.dataset_name)
            prompt = p["population"].replace(f"<{p['class_var']}>", self.class_names[target])
            msgs = [{"role": "system", "content": "Return one integer only in [0..10]."}, {"role": "user", "content": prompt}]
        else:
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
        mapping = ", ".join([f"{self.codes[i]}={name}" for i, name in enumerate(self.class_names)])
        if self.prompt_template_style == "compact":
            p = _dataset_prompt_tokens(self.dataset_name)
            prompt = (
                "Return JSON only: "
                '{ "answer": "<label_id>", "rationale": "<'
                + p["evidence_phrase"]
                + '>" }\n'
                f"Label IDs: {mapping}"
            )
            msgs = [
                {"role": "system", "content": "Return valid JSON only with keys answer and rationale."},
                {"role": "user", "content": prompt},
            ]
        else:
            prompt = (
                f"Identify the correct {self.task_label}. Answer with one code from the mapping. "
                "Then write one short rationale sentence that includes words from the chosen class name.\n"
                f"Codes: {mapping}\n"
                "Format: Answer: <code>. Rationale: <short sentence with class words>."
            )
            msgs = [
                {"role": "system", "content": "Follow format strictly. Include class words in the rationale."},
                {"role": "user", "content": prompt},
            ]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        row["prompt_tokens"] = tok
        row["prompt_mask"] = (tok != self.tokenizer.pad_token_id).long()
        row["task_name"] = "grounded_generation"
        row["rationale_target"] = " ".join(label_words)
        return row


class StrictYesNoPetDataset(FineGrainedPetDataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        train: bool,
        max_samples: Optional[int] = None,
        seed: int = 42,
        dataset_name: str = "oxford_pet",
        prompt_template_style: str = "legacy",
    ):
        super().__init__(
            root=root,
            tokenizer=tokenizer,
            train=train,
            max_samples=max_samples,
            dataset_name=dataset_name,
            prompt_template_style=prompt_template_style,
        )
        self.rng = random.Random(seed)
        self.yes_id = _token_id_for_word(tokenizer, "Yes")
        self.no_id = _token_id_for_word(tokenizer, "No")

    def __getitem__(self, idx):
        i = self.indices[idx]
        image, label = self.ds[i]
        target_cls = self.rng.randrange(len(self.class_names))
        answer_is_yes = int(label) == int(target_cls)
        if self.prompt_template_style == "compact":
            p = _dataset_prompt_tokens(self.dataset_name)
            prompt = p["strict_yesno"].replace(f"<{p['class_var']}>", self.class_names[target_cls])
            msgs = [
                {"role": "system", "content": "Return exactly one token: Yes or No."},
                {"role": "user", "content": prompt},
            ]
        else:
            prompt = (
                f"Question: Is this image a {self.class_names[target_cls]}? "
                "Answer strictly with one token: Yes or No."
            )
            msgs = [
                {"role": "system", "content": "Strict output format: one token only, Yes or No."},
                {"role": "user", "content": prompt},
            ]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        mask = (tok != self.tokenizer.pad_token_id).long()
        return {
            "images": image,
            "prompt_tokens": tok,
            "prompt_mask": mask,
            "target_token_id": torch.tensor(self.yes_id if answer_is_yes else self.no_id, dtype=torch.long),
            "label_idx": torch.tensor(1 if answer_is_yes else 0, dtype=torch.long),
            "task_name": "strict_yesno",
            "allowed_token_ids": [self.yes_id, self.no_id],
        }


def collate_task_batch(batch: List[Dict]) -> Dict:
    max_len = max(x["prompt_tokens"].size(0) for x in batch)
    pad_id = 0
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        seq_len = row["prompt_tokens"].size(0)
        tokens[i, max_len - seq_len :] = row["prompt_tokens"]
        mask[i, max_len - seq_len :] = row["prompt_mask"]

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
    if "code_labels" in batch[0]:
        out["code_labels"] = list(batch[0]["code_labels"])
    return out
