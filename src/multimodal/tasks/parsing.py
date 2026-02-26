from __future__ import annotations

import re


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text.strip().lower())


def parse_label_answer(text: str, class_names):
    cleaned = _clean(text)
    if len(cleaned) == 1 and cleaned.isalpha():
        idx = ord(cleaned.upper()) - ord("A")
        if 0 <= idx < len(class_names):
            return idx
    for i, c in enumerate(class_names):
        if cleaned == _clean(c):
            return i
    if cleaned.isdigit():
        return int(cleaned)
    return None


def parse_yes_no_answer(text: str):
    cleaned = _clean(text)
    if cleaned.startswith("yes"):
        return 1
    if cleaned.startswith("no"):
        return 0
    return None


def parse_population_answer(text: str, mode: str = "integer"):
    cleaned = _clean(text).upper()
    if mode == "integer":
        m = re.search(r"\d+", cleaned)
        return int(m.group()) if m else None
    if mode == "letter":
        if len(cleaned) >= 1 and cleaned[0] in "ABCDEFGHIJ":
            return ord(cleaned[0]) - ord("A")
        return None
    raise ValueError(mode)


def population_bin_to_fraction_midpoint(bin_id: int, mode: str = "integer") -> float:
    if mode == "integer":
        bin_id = max(0, min(10, int(bin_id)))
        return bin_id / 10.0
    bin_id = max(0, min(9, int(bin_id)))
    return (bin_id + 0.5) / 10.0


def population_fraction_to_bin(fraction: float, mode: str = "integer") -> int:
    fraction = max(0.0, min(1.0, float(fraction)))
    if mode == "integer":
        return int(round(fraction * 10.0))
    return min(9, int(fraction * 10.0))
