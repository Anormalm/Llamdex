import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_task_matrix_candidate_leaderboard_ranks_max_and_min_metrics(tmp_path):
    cfg_a = tmp_path / "candidate_a.remote.json"
    cfg_b = tmp_path / "candidate_b.remote.json"
    out_csv = tmp_path / "leaderboard.csv"
    out_json = tmp_path / "leaderboard.json"
    out_md = tmp_path / "leaderboard.md"

    _write_json(
        cfg_a,
        {
            "out_csv": str(tmp_path / "a_rows.csv"),
            "out_json": str(tmp_path / "a_rows.json"),
            "tasks": [{"name": "finegrained"}, {"name": "population"}],
        },
    )
    _write_json(
        cfg_b,
        {
            "out_csv": str(tmp_path / "b_rows.csv"),
            "out_json": str(tmp_path / "b_rows.json"),
            "tasks": [{"name": "finegrained"}, {"name": "population"}],
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    stub_module = stub_dir / "sitecustomize.py"
    stub_module.write_text(
        """
import os
import src.multimodal.task_matrix.runner as runner

def _fake_run_task_matrix(cfg):
    name = os.path.basename(str(cfg.out_csv))
    if name == "a_rows.csv":
        return [
            {"task": "finegrained", "dataset": "dtd", "metric_name": "accuracy", "metric": 0.81, "status": "ok", "baseline_mode": "router_parallel", "fusion_policy": "pre_ffn_router_parallel", "seed": "aggregate"},
            {"task": "population", "dataset": "oxford_pet", "metric_name": "mae", "metric": 0.08, "status": "ok", "baseline_mode": "router_parallel", "fusion_policy": "pre_ffn_router_parallel", "seed": "aggregate"},
        ]
    return [
        {"task": "finegrained", "dataset": "dtd", "metric_name": "accuracy", "metric": 0.77, "status": "ok", "baseline_mode": "router_parallel", "fusion_policy": "post_attn_router_parallel", "seed": "aggregate"},
        {"task": "population", "dataset": "oxford_pet", "metric_name": "mae", "metric": 0.03, "status": "ok", "baseline_mode": "router_parallel", "fusion_policy": "post_attn_router_parallel", "seed": "aggregate"},
    ]

runner.run_task_matrix = _fake_run_task_matrix
""".strip(),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{stub_dir}:{repo_root}:{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_task_matrix_candidate_leaderboard.py",
            "--configs",
            str(cfg_a),
            str(cfg_b),
            "--aggregate_only",
            "--out_csv",
            str(out_csv),
            "--out_json",
            str(out_json),
            "--out_md",
            str(out_md),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    rows = _read_csv(out_csv)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")

    finegrained_rows = [r for r in rows if r["task"] == "finegrained"]
    population_rows = [r for r in rows if r["task"] == "population"]

    assert finegrained_rows[0]["candidate_name"] == "candidate_a"
    assert finegrained_rows[0]["leaderboard_rank"] == "1"
    assert population_rows[0]["candidate_name"] == "candidate_b"
    assert population_rows[0]["leaderboard_rank"] == "1"
    assert payload["rows"][0]["candidate_name"] in {"candidate_a", "candidate_b"}
    assert "| finegrained | dtd | candidate_a |" in md
    assert "| population | oxford_pet | candidate_b |" in md
