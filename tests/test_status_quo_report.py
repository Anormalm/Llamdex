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


def test_generate_status_quo_report_includes_required_sections(tmp_path):
    task_csv = tmp_path / "task.csv"
    grounded_csv = tmp_path / "grounded.csv"
    review_json = tmp_path / "review.json"
    commands_txt = tmp_path / "commands.txt"
    out_md = tmp_path / "status.md"
    out_json = tmp_path / "status.json"

    _write_csv(
        task_csv,
        [
            {
                "task": "grounded_generation",
                "dataset": "dtd",
                "baseline_mode": "router_parallel",
                "accuracy": "1.0",
                "f1": "1.0",
                "status": "ok",
            }
        ],
    )
    _write_csv(
        grounded_csv,
        [
            {
                "task": "grounded_generation",
                "dataset": "dtd",
                "rationale_consistency": "1.0",
                "format_compliance": "1.0",
                "rationale_nonempty_rate": "1.0",
                "status": "ok",
            }
        ],
    )
    review_json.write_text(json.dumps({"decision": "GO"}), encoding="utf-8")
    commands_txt.write_text("python -u scripts/run_task_matrix_eval.py --config conf/example.json", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_status_quo_report.py",
            "--repo_root",
            str(Path(__file__).resolve().parents[1]),
            "--report_date",
            "2026-03-31",
            "--task_matrix_csv",
            str(task_csv),
            "--grounded_csv",
            str(grounded_csv),
            "--go_no_go_json",
            str(review_json),
            "--commands_file",
            str(commands_txt),
            "--branch",
            "feature/test",
            "--commit",
            "abc1234",
            "--python_version",
            "3.10.12",
            "--library_versions",
            "2.10.0 / 5.3.0 / 1.13.0",
            "--timestamp",
            "2026-03-31 +0800",
            "--artifact",
            "/disk1/lfhu/runs/task.csv",
            "--failure",
            "Historical blocker was low rationale consistency; fixed in latest run.",
            "--next_step",
            "Regenerate unified benchmark outputs.",
            "--out_md",
            str(out_md),
            "--out_json",
            str(out_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    md = out_md.read_text(encoding="utf-8")
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert "# Remote Status Quo (2026-03-31)" in md
    assert "## Commit and Branch" in md
    assert "## Environment" in md
    assert "## Commands Executed" in md
    assert "## Result Table" in md
    assert "## Failures / Root Cause / Fix" in md
    assert "## Next 72-Hour Plan" in md
    assert "pilot_gate" in md
    assert payload["branch"] == "feature/test"
    assert payload["commit"] == "abc1234"
    assert payload["results"][-1]["value"] == "GO"
