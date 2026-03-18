import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.benchmark.unified import load_unified_benchmark_config, run_unified_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Unified benchmark driver for Llamdex, architecture baselines, and API baselines.")
    p.add_argument("--config", type=str, required=True, help="Path to unified benchmark config JSON.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_unified_benchmark_config(a.config)
    rows = run_unified_benchmark(cfg)
    print(f"Completed {len(rows)} unified benchmark rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")


if __name__ == "__main__":
    main()
