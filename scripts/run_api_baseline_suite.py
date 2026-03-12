import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.baselines.api_suite import load_api_baseline_config, run_api_baseline_suite


def parse_args():
    p = argparse.ArgumentParser(description="API-backed SOTA baseline benchmark runner (OpenAI-compatible endpoint).")
    p.add_argument("--config", type=str, required=True, help="Path to API baseline config JSON.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_api_baseline_config(a.config)
    rows = run_api_baseline_suite(cfg)
    print(f"Completed {len(rows)} rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")
    print(f"Summary CSV: {cfg.out_summary_csv}")
    print(f"Summary JSON: {cfg.out_summary_json}")


if __name__ == "__main__":
    main()
