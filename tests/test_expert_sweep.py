from src.multimodal.benchmark.expert_sweep import ExpertSweepConfig, ExpertSweepSpec, run_expert_sweep


def test_expert_sweep_overrides_expert_fields(monkeypatch, tmp_path):
    import src.multimodal.benchmark.expert_sweep as sweep
    from src.multimodal.task_matrix.runner import TaskMatrixConfig

    monkeypatch.setattr(
        sweep,
        "_load_task_matrix_config",
        lambda path: TaskMatrixConfig(
            out_csv=str(tmp_path / "base.csv"),
            out_json=str(tmp_path / "base.json"),
            device="cpu",
        ),
    )
    monkeypatch.setattr(
        sweep,
        "run_task_matrix",
        lambda cfg: [{"expert_type": cfg.expert_type, "expert_model_id": cfg.expert_model_id, "metric": 0.5, "status": "ok"}],
    )

    cfg = ExpertSweepConfig(
        base_task_matrix_config="ignored.json",
        experts=[
            ExpertSweepSpec(name="clip", expert_type="clip", expert_model_id="openai/clip-vit-base-patch32", expert_output_dim=512),
            ExpertSweepSpec(name="dinov2", expert_type="dinov2", expert_model_id="facebook/dinov2-base", expert_output_dim=768),
        ],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
    )
    rows = run_expert_sweep(cfg)
    assert len(rows) == 2
    assert rows[0]["expert_sweep_name"] == "clip"
    assert rows[1]["expert_type"] == "dinov2"
