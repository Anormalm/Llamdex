from __future__ import annotations

import os
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
    if model_name != "auto":
        return model_name, "user-specified"

    mem = _gpu_mem_gb()
    # Ordered from stronger to lighter, with conservative VRAM guards.
    candidates: List[Tuple[str, float]] = [
        ("mistralai/Mistral-7B-Instruct-v0.3", 13.0),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 2.0),
        ("hf-internal-testing/tiny-random-MistralForCausalLM", 0.0),
    ]
    for mid, min_mem in candidates:
        if mem >= min_mem and _exists_in_hf_cache(cache_dir, mid):
            return mid, f"auto-selected (gpu_mem={mem:.1f}GB, cache-hit)"
    # Final fallback: non-random small model if cache miss handled by HF.
    return "HuggingFaceTB/SmolLM2-360M-Instruct", f"auto-selected fallback (gpu_mem={mem:.1f}GB)"
