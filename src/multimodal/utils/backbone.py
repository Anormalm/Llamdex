from __future__ import annotations

import os
import importlib.util
from typing import List, Tuple

import torch


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        prop = torch.cuda.get_device_properties(0)
        return float(prop.total_memory) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _exists_in_hf_cache(cache_dir: str, model_id: str) -> bool:
    safe = model_id.replace("/", "--")
    roots = []
    if cache_dir:
        roots.append(cache_dir)
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    for root in roots:
        p = os.path.join(root, f"models--{safe}", "snapshots")
        if os.path.isdir(p):
            try:
                snaps = [os.path.join(p, d) for d in os.listdir(p)]
                for s in snaps:
                    if os.path.isfile(os.path.join(s, "config.json")):
                        return True
            except Exception:
                return True
    return False


def resolve_backbone(model_name: str, cache_dir: str) -> Tuple[str, str]:
    """
    Resolve runtime backbone.

    Returns: (resolved_model_name, reason)
    """
    def _qwen_runtime_supported() -> bool:
        # Some older local wrappers depend on HF model internals.
        # Gate Qwen selection on local transformers support to avoid hard runtime failures.
        return importlib.util.find_spec("transformers.models.qwen2") is not None or importlib.util.find_spec(
            "transformers.models.qwen3"
        ) is not None

    def _safe_user_selected(name: str) -> Tuple[str, str]:
        if "qwen" not in name.lower():
            return name, "user-specified"
        if _qwen_runtime_supported():
            return name, "user-specified (qwen-supported)"
        return (
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "qwen requested but local transformers lacks qwen2/qwen3 support; fell back to a lightweight local model",
        )

    if model_name != "auto":
        return _safe_user_selected(model_name)

    mem = _gpu_mem_gb()
    # Ordered from stronger to lighter, with conservative VRAM guards.
    candidates: List[Tuple[str, float]] = [
        ("Qwen/Qwen3.5-9B", 16.0),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 2.0),
    ]
    for mid, min_mem in candidates:
        if "qwen" in mid.lower() and not _qwen_runtime_supported():
            continue
        if mem >= min_mem and _exists_in_hf_cache(cache_dir, mid):
            return mid, f"auto-selected (gpu_mem={mem:.1f}GB, cache-hit)"
    # Final fallback: non-random small model if cache miss handled by HF.
    return "HuggingFaceTB/SmolLM2-360M-Instruct", f"auto-selected fallback (gpu_mem={mem:.1f}GB)"
