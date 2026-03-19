import json
import subprocess
import sys
from pathlib import Path


def test_generate_benchmark_report_summarizes_best_rows(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            [
                {"model_type": "llm_only", "dataset": "dtd", "task": "single_image", "metric_name": "accuracy", "metric": 0.2, "status": "ok"},
                {"model_type": "router_parallel", "dataset": "dtd", "task": "single_image", "metric_name": "accuracy", "metric": 0.8, "status": "ok"},
                {"model_type": "rag_structured", "dataset": "dtd", "task": "single_image", "metric_name": "accuracy", "metric": 0.0, "status": "error", "error": "boom"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "report.md"
    out_json = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_benchmark_report.py",
            "--inputs",
            str(result_path),
            "--out_md",
            str(out_md),
            "--out_json",
            str(out_json),
            "--title",
            "Test Report",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    md = out_md.read_text(encoding="utf-8")
    assert "# Test Report" in md
    assert "`single_image`: `router_parallel` accuracy=0.8000" in md
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["summary"]["ok_rows"] == 2
