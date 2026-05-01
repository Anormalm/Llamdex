from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset
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
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    candidates = (
        alphabet
        + list("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()[]{}<>?/|")
        + [f"{a}{b}" for a in alphabet for b in "0123456789"]
        + [f"{a}{b}" for a in alphabet for b in alphabet]
        + [f"{a}{b}{c}" for a in alphabet for b in alphabet for c in "0123456789"]
    )
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
    second_image_field: str = ""
    question_field: str = "question"
    answer_field: str = "answer"
    context_field: str = ""


class VQASubsetDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        spec: VQASubsetSpec,
        max_samples: Optional[int] = 512,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.local_base_dir = None
        if str(spec.hf_dataset_name).startswith("local:"):
            path = str(spec.hf_dataset_name)[len("local:") :]
            self.local_base_dir = str(Path(path).expanduser().parent)
            self.ds = _read_manifest_rows(path)
            if max_samples is not None:
                self.ds = self.ds[: max_samples]
        else:
            from datasets import load_dataset

            self.ds = load_dataset(spec.hf_dataset_name, split=spec.split)
            if max_samples is not None:
                self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.image_field = spec.image_field
        self.second_image_field = spec.second_image_field
        self.question_field = spec.question_field
        self.answer_field = spec.answer_field
        self.context_field = spec.context_field
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
        img = _load_manifest_image(row[self.image_field], base_dir=self.local_base_dir)
        if self.second_image_field and row.get(self.second_image_field):
            second = _load_manifest_image(row[self.second_image_field], base_dir=self.local_base_dir)
            img = _concat_images_horizontally(img, second)
        img = self.tx(img.convert("RGB"))
        q = str(row[self.question_field]).strip()
        a = row[self.answer_field]
        if isinstance(a, list):
            a = a[0]
        ans = str(a).strip().lower()
        ans_idx = self.ans_to_idx.get(ans, 0)
        code = self.codes[ans_idx]
        mapping = ", ".join([f"{self.codes[i]}={w}" for i, w in enumerate(self.answer_vocab[:50])])
        context = ""
        if self.context_field and row.get(self.context_field):
            context = f"Context: {str(row[self.context_field]).strip()}\n"
        prompt = f"{context}Question: {q} Answer with one code only. Codes: {mapping}."
        msgs = [{"role": "system", "content": "Answer with one code only."}, {"role": "user", "content": prompt}]
        tok = _chat_template_tokens(self.tokenizer, msgs)
        mask = (tok != self.tokenizer.pad_token_id).long()
        out = {
            "images": img,
            "prompt_tokens": tok,
            "prompt_mask": mask,
            "target_token_id": torch.tensor(self.code_to_tid[code], dtype=torch.long),
            "answer_idx": torch.tensor(ans_idx, dtype=torch.long),
            "task_name": "vqa",
            "allowed_token_ids": list(self.code_to_tid.values()),
        }
        vector = row.get("tabular", row.get("semantic_vector"))
        if vector is not None:
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except json.JSONDecodeError:
                    vector = [float(x) for x in vector.split(",") if x.strip()]
            out["tabular"] = torch.tensor(vector, dtype=torch.float32)
        return out


def _read_manifest_rows(path: str) -> List[Dict]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Private manifest not found: {p}")
    if p.suffix.lower() == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("rows") or payload.get("data") or []
        if not isinstance(payload, list):
            raise ValueError(f"JSON manifest must contain a list of rows: {p}")
        return [dict(r) for r in payload]
    with open(p, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _load_manifest_image(value, base_dir: Optional[str] = None) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir:
        path = Path(base_dir).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"Image path in private manifest not found: {path}")
    return Image.open(path).convert("RGB")


def _concat_images_horizontally(left: Image.Image, right: Image.Image) -> Image.Image:
    left = ImageOps.contain(left.convert("RGB"), (224, 224))
    right = ImageOps.contain(right.convert("RGB"), (224, 224))
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), color=(0, 0, 0))
    canvas.paste(left, (0, (canvas.height - left.height) // 2))
    canvas.paste(right, (left.width, (canvas.height - right.height) // 2))
    return canvas


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        transform,
        image_field: str = "image_path",
        label_field: str = "label",
        max_samples: Optional[int] = None,
    ):
        self.rows = _read_manifest_rows(manifest_path)
        self.base_dir = str(Path(manifest_path).expanduser().parent)
        if max_samples is not None:
            self.rows = self.rows[: max_samples]
        self.image_field = image_field
        self.label_field = label_field
        self.transform = transform
        labels = [str(r[label_field]).strip() for r in self.rows]
        self.classes = sorted(set(labels))
        if not self.classes:
            raise ValueError(f"Manifest has no labels in field {label_field!r}: {manifest_path}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = _load_manifest_image(row[self.image_field], base_dir=self.base_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[str(row[self.label_field]).strip()]
        return img, label


class HFImageClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        transform,
        cache_dir: Optional[str] = None,
        image_field: str = "image",
        label_field: str = "label",
    ):
        from datasets import ClassLabel, load_dataset

        self.ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.transform = transform
        self.image_field = image_field
        self.label_field = label_field
        feature = self.ds.features[label_field]
        if isinstance(feature, ClassLabel):
            self.classes = [str(name).replace("_", " ") for name in feature.names]
            self._label_to_idx = None
        else:
            labels = sorted({str(row[label_field]).strip() for row in self.ds})
            self.classes = [label.replace("_", " ") for label in labels]
            self._label_to_idx = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        image = row[self.image_field]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        raw_label = row[self.label_field]
        if self._label_to_idx is None:
            label = int(raw_label)
        else:
            label = self._label_to_idx[str(raw_label).strip()]
        return image, label


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
    if name == "eurosat":
        base = datasets.EuroSAT(root=root, download=True, transform=tx)
        indices = list(range(len(base)))
        split = int(0.8 * len(indices))
        selected = indices[:split] if train else indices[split:]
        return Subset(base, selected), list(base.classes), "land cover class"
    if name == "food101":
        split = "train" if train else "test"
        ds = datasets.Food101(root=root, split=split, download=True, transform=tx)
        return ds, [c.replace("_", " ") for c in ds.classes], "food dish class"
    if name == "fgvc_aircraft":
        split = "trainval" if train else "test"
        ds = datasets.FGVCAircraft(root=root, split=split, annotation_level="variant", download=True, transform=tx)
        return ds, list(ds.classes), "aircraft variant"
    if name == "resisc45":
        split = "train" if train else "test"
        ds = HFImageClassificationDataset("timm/resisc45", split=split, transform=tx, cache_dir=root)
        return ds, list(ds.classes), "remote-sensing scene class"
    if name in {"private_manifest", "mimic_cxr_jpg", "chexpert_plus", "private_doc_ocr_synth", "privacy_risk"}:
        split = "train" if train else "eval"
        manifest_path = os.path.join(root, name, f"{split}.csv")
        if not os.path.isfile(manifest_path):
            alt = os.path.join(root, name, f"{split}.jsonl")
            manifest_path = alt if os.path.isfile(alt) else manifest_path
        ds = ManifestImageDataset(manifest_path=manifest_path, transform=tx)
        task_label = {
            "mimic_cxr_jpg": "chest X-ray label",
            "chexpert_plus": "chest X-ray label",
            "private_doc_ocr_synth": "document class",
            "privacy_risk": "privacy risk class",
        }.get(name, "private image label")
        return ds, list(ds.classes), task_label
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
    if name == "eurosat":
        return {
            "schema": "EuroSAT land-cover schema",
            "class_var": "land_cover_class",
            "evidence_phrase": "land-cover evidence sentence",
            "finegrained": "Classify this satellite image by land-cover type. Return `label_id` only from EuroSAT schema. No explanation.",
            "strict_yesno": "Question: Is this satellite image `<land_cover_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this satellite image bag, estimate fraction of `<land_cover_class>`. Return one integer in `[0..10]` only.",
        }
    if name == "food101":
        return {
            "schema": "Food-101 dish schema",
            "class_var": "dish_class",
            "evidence_phrase": "dish-grounded evidence sentence",
            "finegrained": "Classify this food image by dish category. Return `label_id` only from Food-101 schema. No explanation.",
            "strict_yesno": "Question: Is this food image `<dish_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this food image bag, estimate fraction of `<dish_class>`. Return one integer in `[0..10]` only.",
        }
    if name == "fgvc_aircraft":
        return {
            "schema": "FGVC-Aircraft variant schema",
            "class_var": "aircraft_variant",
            "evidence_phrase": "aircraft-variant evidence sentence",
            "finegrained": "Classify this aircraft image by variant. Return `label_id` only from FGVC-Aircraft schema. No explanation.",
            "strict_yesno": "Question: Is this aircraft image `<aircraft_variant>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this aircraft image bag, estimate fraction of `<aircraft_variant>`. Return one integer in `[0..10]` only.",
        }
    if name == "resisc45":
        return {
            "schema": "RESISC45 remote-sensing scene schema",
            "class_var": "scene_class",
            "evidence_phrase": "remote-sensing scene evidence sentence",
            "finegrained": "Classify this remote-sensing image by scene type. Return `label_id` only from RESISC45 schema. No explanation.",
            "strict_yesno": "Question: Is this remote-sensing image `<scene_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this remote-sensing image bag, estimate fraction of `<scene_class>`. Return one integer in `[0..10]` only.",
        }
    if name in {"mimic_cxr_jpg", "chexpert_plus"}:
        return {
            "schema": "chest radiograph label schema",
            "class_var": "finding_label",
            "evidence_phrase": "radiograph-grounded evidence sentence",
            "finegrained": "Classify this chest radiograph finding. Return `label_id` only from the local clinical label schema. No explanation.",
            "strict_yesno": "Question: Does this chest radiograph show `<finding_label>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this chest radiograph bag, estimate fraction showing `<finding_label>`. Return one integer in `[0..10]` only.",
        }
    if name == "private_doc_ocr_synth":
        return {
            "schema": "private document schema",
            "class_var": "document_class",
            "evidence_phrase": "document-grounded evidence sentence",
            "finegrained": "Classify this private document image. Return `label_id` only from the local document schema. No explanation.",
            "strict_yesno": "Question: Is this document `<document_class>`? Return exactly one token: `Yes` or `No`.",
            "population": "In this document image bag, estimate fraction of `<document_class>`. Return one integer in `[0..10]` only.",
        }
    if name == "privacy_risk":
        return {
            "schema": "privacy risk schema",
            "class_var": "privacy_risk_class",
            "evidence_phrase": "privacy-risk evidence sentence",
            "finegrained": "Classify this image by privacy risk severity. Return `label_id` only from the local privacy schema. No explanation.",
            "strict_yesno": "Question: Does this image contain `<privacy_risk_class>` privacy risk? Return exactly one token: `Yes` or `No`.",
            "population": "In this image bag, estimate fraction with `<privacy_risk_class>` privacy risk. Return one integer in `[0..10]` only.",
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
            "true_class": torch.tensor(int(label), dtype=torch.long),
            "target_class": torch.tensor(int(target_cls), dtype=torch.long),
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
    if "true_class" in batch[0]:
        out["true_class"] = torch.stack([x["true_class"] for x in batch], dim=0)
    if "target_class" in batch[0]:
        out["target_class"] = torch.stack([x["target_class"] for x in batch], dim=0)
    if "answer_idx" in batch[0]:
        out["answer_idx"] = torch.stack([x["answer_idx"] for x in batch], dim=0)
    if "tabular" in batch[0]:
        out["tabular"] = torch.stack([x["tabular"] for x in batch], dim=0)
    if "rationale_keywords" in batch[0]:
        out["rationale_keywords"] = [x["rationale_keywords"] for x in batch]
    if "rationale_target" in batch[0]:
        out["rationale_target"] = [x["rationale_target"] for x in batch]
    if "code_labels" in batch[0]:
        out["code_labels"] = list(batch[0]["code_labels"])
    return out
