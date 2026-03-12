import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.scaling.runner import load_backbone_scaling_config, run_backbone_scaling_suite


def parse_args():
    p = argparse.ArgumentParser(description="Backbone-scaling suite for prefix-injection stability.")
    p.add_argument("--config", type=str, required=True, help="Path to backbone scaling JSON config.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_backbone_scaling_config(a.config)
    out = run_backbone_scaling_suite(cfg)
    print("Saved outputs:")
    for k, v in out.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

