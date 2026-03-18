from src.multimodal.benchmark.unified import UnifiedBenchmarkConfig, run_unified_benchmark


def test_unified_benchmark_merges_sources(monkeypatch, tmp_path):
    import src.multimodal.benchmark.unified as unified

    monkeypatch.setattr(unified, "load_task_matrix_config", lambda path: {"kind": "task_matrix", "path": path})
    monkeypatch.setattr(unified, "load_baseline_suite_config", lambda path: {"kind": "local_suite", "path": path})
    monkeypatch.setattr(unified, "load_api_baseline_config", lambda path: {"kind": "api_suite", "path": path})
    monkeypatch.setattr(
        unified,
        "run_task_matrix",
        lambda cfg: [{"task": "finegrained", "baseline_mode": "overwrite", "backbone": "Qwen/Qwen3.5-9B", "fusion_policy": "pre_attn_overwrite", "metric": 0.7, "status": "ok"}],
    )
    monkeypatch.setattr(
        unified,
        "run_baseline_suite",
        lambda cfg: [{"model_type": "two_stage_caption_llm", "dataset": "oxford_pet", "metric": 0.4, "status": "ok"}],
    )
    monkeypatch.setattr(
        unified,
        "run_api_baseline_suite",
        lambda cfg: [{"model_type": "api_llm_only::gpt-4.1", "task_family": "single_image", "dataset": "oxford_pet", "model_id": "gpt-4.1", "metric": 0.5, "status": "ok"}],
    )

    cfg = UnifiedBenchmarkConfig(
        task_matrix_config="task.json",
        local_baseline_config="local.json",
        api_baseline_config="api.json",
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
    )
    rows = run_unified_benchmark(cfg)
    assert len(rows) == 3
    assert rows[0]["benchmark_family"] == "llamdex"
    assert rows[1]["benchmark_family"] == "architecture"
    assert rows[2]["benchmark_family"] == "api"
