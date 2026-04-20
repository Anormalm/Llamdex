import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def test_generate_go_no_go_review_flags_rationale_blocker(tmp_path):
    task_csv = tmp_path / "task.csv"
    grounded_csv = tmp_path / "grounded.csv"
    api_csv = tmp_path / "api.csv"
    out_md = tmp_path / "review.md"
    out_json = tmp_path / "review.json"

    _write_csv(
        task_csv,
        [
            {"task": "finegrained", "accuracy": "0.80"},
            {"task": "strict_yesno", "accuracy": "0.95"},
            {"task": "population", "mae": "0.02"},
            {"task": "grounded_generation", "f1": "0.82"},
        ],
    )
    _write_csv(
        grounded_csv,
        [
            {
                "task": "grounded_generation",
                "rationale_consistency": "0.00",
                "format_compliance": "1.0",
                "rationale_nonempty_rate": "1.0",
            }
        ],
    )
    _write_csv(api_csv, [{"status": "ok"}])

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_go_no_go_review.py",
            "--task_matrix_csv",
            str(task_csv),
            "--grounded_csv",
            str(grounded_csv),
            "--api_summary_csv",
            str(api_csv),
            "--out_md",
            str(out_md),
            "--out_json",
            str(out_json),
            "--min_rationale_consistency",
            "0.2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["decision"] == "NO-GO"
    assert "rationale_consistency" in payload["blockers"]
