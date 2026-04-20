import argparse
import json
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


PRESET_MODEL_IDS: Dict[str, List[str]] = {
    # Strong open-weight VLMs commonly used as local baselines.
    "vlm_sota_2026q2": [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "OpenGVLab/InternVL2_5-8B",
        "allenai/Molmo-7B-D-0924",
        "mistralai/Pixtral-12B-2409",
        # Heavy models kept in the same pack for optional download.
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf",
        "OpenGVLab/InternVL2_5-38B",
        "OpenGVLab/InternVL2_5-78B",
        "allenai/Molmo-72B-0924",
    ],
    # Practical pack for single-GPU / faster iteration.
    "vlm_budget_2026q2": [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "OpenGVLab/InternVL2_5-8B",
        "allenai/Molmo-7B-D-0924",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Download baseline HF models into local cache.")
    p.add_argument("--cache_dir", type=str, default="/disk1/lfhu/hf_cache")
    p.add_argument("--baseline_config", type=str, default=None, help="Optional baseline suite JSON config.")
    p.add_argument("--include_backbone", action="store_true", help="Also download config.model_name from baseline config.")
    p.add_argument("--include_captioner", action="store_true", help="Also download config.caption_model_id from baseline config.")
    p.add_argument(
        "--include_disabled_from_config",
        action="store_true",
        help="Include frozen_vlms entries with enabled=false from baseline_config.",
    )
    p.add_argument("--preset", action="append", default=[], help="Preset pack name. Repeatable. Use --list_presets to inspect.")
    p.add_argument("--list_presets", action="store_true", help="Print available preset packs and exit.")
    p.add_argument("--model_id", action="append", default=[], help="Extra model id to download (can be repeated).")
    p.add_argument("--allow_patterns", action="append", default=None, help="Optional snapshot allow pattern (repeatable).")
    p.add_argument("--dry_run", action="store_true", help="Print resolved models and exit without downloading.")
    p.add_argument("--continue_on_error", action="store_true", help="Continue if a model fails to download.")
    return p.parse_args()


def _collect_from_baseline_config(
    path: str,
    include_backbone: bool,
    include_captioner: bool,
    include_disabled_from_config: bool,
):
    model_ids = []
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for item in cfg.get("frozen_vlms", []):
        if not include_disabled_from_config and not bool(item.get("enabled", True)):
            continue
        mid = str(item.get("model_id", "")).strip()
        if mid:
            model_ids.append(mid)
    if include_backbone:
        mid = str(cfg.get("model_name", "")).strip()
        if mid:
            model_ids.append(mid)
    if include_captioner:
        mid = str(cfg.get("caption_model_id", "")).strip()
        if mid:
            model_ids.append(mid)
    # Preserve order while deduping.
    out = []
    seen = set()
    for m in model_ids:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def main():
    a = parse_args()
    if a.list_presets:
        print("Available presets:")
        for name, mids in PRESET_MODEL_IDS.items():
            print(f"- {name} ({len(mids)} models)")
            for mid in mids:
                print(f"  - {mid}")
        return

    model_ids = list(a.model_id)
    for preset_name in a.preset:
        mids = PRESET_MODEL_IDS.get(preset_name)
        if mids is None:
            raise SystemExit(f"Unknown preset '{preset_name}'. Use --list_presets.")
        model_ids.extend(mids)
    if a.baseline_config:
        model_ids.extend(
            _collect_from_baseline_config(
                a.baseline_config,
                include_backbone=bool(a.include_backbone),
                include_captioner=bool(a.include_captioner),
                include_disabled_from_config=bool(a.include_disabled_from_config),
            )
        )
    model_ids = [m for i, m in enumerate(model_ids) if m and m not in model_ids[:i]]
    if not model_ids:
        raise SystemExit("No model ids provided. Use --model_id and/or --baseline_config.")

    print("Resolved model ids:")
    for mid in model_ids:
        print(f"- {mid}")
    if a.dry_run:
        return

    cache_dir = Path(a.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(model_ids)} model(s) into cache_dir={cache_dir}")
    for idx, mid in enumerate(model_ids, start=1):
        print(f"[{idx}/{len(model_ids)}] {mid}")
        try:
            snapshot_download(
                repo_id=mid,
                cache_dir=str(cache_dir),
                allow_patterns=a.allow_patterns,
                local_files_only=False,
                resume_download=True,
            )
        except Exception as exc:
            msg = f"Failed downloading {mid}: {exc}"
            if not a.continue_on_error:
                raise RuntimeError(msg) from exc
            print(msg)
    print("Download complete.")


if __name__ == "__main__":
    main()
