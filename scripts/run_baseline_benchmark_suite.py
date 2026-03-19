import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.baselines.suite import load_baseline_suite_config, run_baseline_suite
from src.multimodal.utils.reporting import normalize_result_row, render_markdown_report, summarize_rows


def parse_args():
    p = argparse.ArgumentParser(description="Baseline benchmarking suite for multimodal prefix-injection framework.")
    p.add_argument("--config", type=str, required=True, help="Path to baseline suite config JSON.")
    p.add_argument("--report_md", type=str, default=None, help="Optional markdown summary output.")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_baseline_suite_config(a.config)
    rows = run_baseline_suite(cfg)
    if a.report_md:
        report_rows = [normalize_result_row(row, source_path=cfg.out_json) for row in rows]
        summary = summarize_rows(report_rows)
        os.makedirs(os.path.dirname(a.report_md) or ".", exist_ok=True)
        with open(a.report_md, "w", encoding="utf-8") as f:
            f.write(render_markdown_report(report_rows, summary, title="Local Baseline Benchmark Report"))
    print(f"Completed {len(rows)} baseline rows.")
    print(f"CSV: {cfg.out_csv}")
    print(f"JSON: {cfg.out_json}")
    if a.report_md:
        print(f"Report: {a.report_md}")


if __name__ == "__main__":
    main()
