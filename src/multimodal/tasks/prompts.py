from __future__ import annotations

from typing import List


def class_names_for_dataset(dataset_name: str) -> List[str]:
    if dataset_name == "cifar10":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if dataset_name == "cifar100":
        return [f"class_{i}" for i in range(100)]
    if dataset_name == "dtd":
        return [f"class_{i}" for i in range(47)]
    if dataset_name == "oxford_pet":
        return [f"class_{i}" for i in range(37)]
    if dataset_name == "hospital_text":
        return ["benign", "malignant"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_label_prompt(question_style: str = "label", class_names: List[str] | None = None) -> str:
    if question_style == "label":
        if class_names is None:
            return "Classify this image. Answer with one label only."
        if len(class_names) > 26:
            raise ValueError("Label-code prompt currently supports up to 26 classes.")
        mapping = ", ".join([f"{chr(ord('A') + i)}={name}" for i, name in enumerate(class_names)])
        return f"Classify this image. Answer with one label code only. Codes: {mapping}."
    if question_style == "index":
        return "Return the class index only."
    raise ValueError(f"Unsupported question_style: {question_style}")


def build_yes_no_prompt(class_name: str) -> str:
    return (
        f"Is this a {class_name}? "
        "Respond with exactly one token: Yes or No. "
        "Do not output punctuation or any other words."
    )


def build_population_prompt(target_class: str, output_mode: str = "integer") -> str:
    if output_mode == "integer":
        return f"In this batch of images, what fraction are {target_class}? Answer with a single integer from 0 to 10."
    if output_mode == "letter":
        return (
            f"In this batch of images, what fraction are {target_class}? "
            "Answer with one letter A-J, where A=0-10%, B=10-20%, ..., J=90-100%."
        )
    raise ValueError(f"Unsupported output_mode: {output_mode}")
