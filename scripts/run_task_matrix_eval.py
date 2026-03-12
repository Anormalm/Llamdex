import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.task_matrix.runner import load_task_matrix_config, run_task_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Unified task-matrix runner for multimodal prefix-injection generalization.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_task_matrix_config(a.config)
    rows = run_task_matrix(cfg)
    print(f"Completed {len(rows)} tasks.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")


if __name__ == "__main__":
    main()

