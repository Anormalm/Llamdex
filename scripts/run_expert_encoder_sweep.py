import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.benchmark.expert_sweep import load_expert_sweep_config, run_expert_sweep


def parse_args():
    p = argparse.ArgumentParser(description="Run clip/siglip/dinov2 expert sweep on a shared task-matrix protocol.")
    p.add_argument("--config", type=str, required=True, help="Path to expert sweep config JSON.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_expert_sweep_config(a.config)
    rows = run_expert_sweep(cfg)
    print(f"Completed {len(rows)} expert sweep rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")


if __name__ == "__main__":
    main()
