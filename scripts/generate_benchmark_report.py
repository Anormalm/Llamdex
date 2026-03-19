import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.utils.reporting import load_result_rows, normalize_result_row, render_markdown_report, summarize_rows


def parse_args():
    p = argparse.ArgumentParser(description="Generate a compact markdown/json benchmark report from result artifacts.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input result files (.json/.csv).")
    p.add_argument("--out_md", type=str, default="runs/benchmark_report.md")
    p.add_argument("--out_json", type=str, default="runs/benchmark_report.summary.json")
    p.add_argument("--title", type=str, default="Benchmark Report")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in args.inputs:
        for row in load_result_rows(path):
            rows.append(normalize_result_row(row, source_path=path))
    summary = summarize_rows(rows)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(rows, summary, title=args.title))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
