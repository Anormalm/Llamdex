import argparse
import csv
import json
from pathlib import Path

from src.multimodal.data import (
    detect_pii_risks,
    evaluate_description_quality,
    load_jsonl,
    load_text_privacy_policy,
    validate_text_privacy_record,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run sanitization QA on description/question records and emit structured reports.")
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--privacy_policy", type=str, default="conf/text_privacy_mode.v1.json")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    rows = load_jsonl(a.input_jsonl)
    policy = load_text_privacy_policy(a.privacy_policy)

    out_rows = []
    for idx, row in enumerate(rows, start=1):
        desc = str(row.get("description", ""))
        question = str(row.get("question", ""))
        quality = evaluate_description_quality(desc, policy=policy)
        pii = sorted(set(detect_pii_risks(desc, policy=policy) + detect_pii_risks(question, policy=policy)))
        privacy = validate_text_privacy_record(row, policy=policy)
        issues = sorted(set(quality + pii + privacy))
        out_rows.append(
            {
                "line": idx,
                "record_id": row.get("record_id", ""),
                "split": row.get("split", ""),
                "quality_issue_count": len(quality),
                "pii_risk_count": len(pii),
                "privacy_error_count": len(privacy),
                "total_issue_count": len(issues),
                "status": "ok" if not issues else "error",
                "issues": " | ".join(issues),
            }
        )

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = list(out_rows[0].keys()) if out_rows else [
        "line",
        "record_id",
        "split",
        "quality_issue_count",
        "pii_risk_count",
        "privacy_error_count",
        "total_issue_count",
        "status",
        "issues",
    ]
    with Path(a.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)

    summary = {
        "records": len(out_rows),
        "error_records": sum(1 for r in out_rows if r["status"] != "ok"),
        "quality_issue_records": sum(1 for r in out_rows if int(r["quality_issue_count"]) > 0),
        "pii_risk_records": sum(1 for r in out_rows if int(r["pii_risk_count"]) > 0),
        "privacy_error_records": sum(1 for r in out_rows if int(r["privacy_error_count"]) > 0),
        "input_jsonl": a.input_jsonl,
        "privacy_policy": a.privacy_policy,
        "out_csv": a.out_csv,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if summary["error_records"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
