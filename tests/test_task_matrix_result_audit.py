import csv
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_task_matrix_results_flags_degenerate_accuracy_f1(tmp_path):
    csv_path = tmp_path / "task_matrix.csv"
    out_csv = tmp_path / "audit.csv"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    _write_csv(
        csv_path,
        [
            {
                "dataset": "food101",
                "task": "finegrained",
                "baseline_mode": "router_parallel",
                "fusion_policy": "pre_ffn_router_parallel",
                "status": "ok",
                "seed": "42",
                "accuracy": "1.0",
                "f1": "0.0",
            },
            {
                "dataset": "dtd",
                "task": "population",
                "baseline_mode": "router_parallel",
                "fusion_policy": "pre_ffn_router_parallel",
                "status": "ok",
                "seed": "42",
                "mae": "0.02",
                "calibration_error": "0.02",
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_task_matrix_results.py",
            str(csv_path),
            "--out_csv",
            str(out_csv),
            "--out_json",
            str(out_json),
            "--out_md",
            str(out_md),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    rows = list(csv.DictReader(open(out_csv, encoding="utf-8", newline="")))
    by_dataset = {row["dataset"]: row for row in rows}
    assert by_dataset["food101"]["triage"] == "drop_or_fix"
    assert "high accuracy with near-zero F1" in by_dataset["food101"]["reasons"]
    assert by_dataset["dtd"]["triage"] == "needs_rerun_or_caveat"
    assert "fewer than 3 seeds" in by_dataset["dtd"]["reasons"]
    assert out_md.exists()
