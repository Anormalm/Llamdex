import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="URGENT: run expanded SOTA API baseline pack.")
    p.add_argument("--config", type=str, default="conf/api_baseline_sota.urgent.json")
    return p.parse_args()


def main():
    a = parse_args()
    with open(a.config, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    try:
        from src.multimodal.baselines.api_suite import load_api_baseline_config, run_api_baseline_suite

        cfg = load_api_baseline_config(a.config)
        rows = run_api_baseline_suite(cfg)
        print(f"Completed {len(rows)} rows.")
        print(f"CSV: {cfg.out_csv}")
        print(f"JSON: {cfg.out_json}")
        print(f"Summary CSV: {cfg.out_summary_csv}")
        print(f"Summary JSON: {cfg.out_summary_json}")
    except Exception as exc:
        out_csv = raw_cfg.get("out_csv", "runs/api_baseline_sota_urgent.csv")
        out_json = raw_cfg.get("out_json", "runs/api_baseline_sota_urgent.json")
        out_summary_csv = raw_cfg.get("out_summary_csv", "runs/api_baseline_sota_urgent.summary.csv")
        out_summary_json = raw_cfg.get("out_summary_json", "runs/api_baseline_sota_urgent.summary.json")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        row = {
            "model_type": "api_suite",
            "dataset": ",".join(raw_cfg.get("dataset_names", [])) or raw_cfg.get("dataset_name", ""),
            "metric": 0.0,
            "metric_name": "accuracy",
            "inference_latency": 0.0,
            "status": "error",
            "error": str(exc),
        }
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([row], f, indent=2)
        with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump([row], f, indent=2)
        print(f"API suite unavailable, wrote error rows. reason={exc}")
        print(f"CSV: {out_csv}")
        print(f"JSON: {out_json}")
        print(f"Summary CSV: {out_summary_csv}")
        print(f"Summary JSON: {out_summary_json}")


if __name__ == "__main__":
    main()
