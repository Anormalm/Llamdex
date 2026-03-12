from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch


def macro_f1_from_ints(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    if len(y_true) == 0:
        return 0.0
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / max(1, len(f1s)))


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.array(list(y_true), dtype=np.float32)
    yp = np.array(list(y_pred), dtype=np.float32)
    if yt.size == 0:
        return 0.0
    return float(np.abs(yt - yp).mean())


def expected_calibration_error_from_bins(
    y_true_fraction: torch.Tensor,
    y_pred_fraction: torch.Tensor,
    n_bins: int = 10,
) -> float:
    if y_true_fraction.numel() == 0:
        return 0.0
    y_true = y_true_fraction.clamp(0, 1).float()
    y_pred = y_pred_fraction.clamp(0, 1).float()
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=y_true.device)
    ece = torch.tensor(0.0, device=y_true.device)
    n = y_true.numel()
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        if mask.any():
            conf = y_pred[mask].mean()
            acc = y_true[mask].mean()
            ece = ece + (mask.float().mean()) * torch.abs(acc - conf)
    return float(ece.item())


def keyword_consistency_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.lower()
    hit = sum(1 for k in keywords if k.lower() in t)
    return float(hit / len(keywords))

