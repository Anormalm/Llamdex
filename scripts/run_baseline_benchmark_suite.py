import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.baselines.suite import load_baseline_suite_config, run_baseline_suite


def parse_args():
    p = argparse.ArgumentParser(description="Baseline benchmarking suite for multimodal prefix-injection framework.")
    p.add_argument("--config", type=str, required=True, help="Path to baseline suite config JSON.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_baseline_suite_config(a.config)
    rows = run_baseline_suite(cfg)
    print(f"Completed {len(rows)} baseline rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")


if __name__ == "__main__":
    main()

