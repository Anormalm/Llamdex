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


def test_expert_sweep_saves_partial_rows_when_later_expert_fails(monkeypatch, tmp_path):
    import json
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

    def fake_run(cfg):
        if cfg.expert_type == "clip":
            return [{"expert_type": cfg.expert_type, "metric": 0.5, "status": "ok"}]
        raise RuntimeError("boom")

    monkeypatch.setattr(sweep, "run_task_matrix", fake_run)

    cfg = ExpertSweepConfig(
        base_task_matrix_config="ignored.json",
        experts=[
            ExpertSweepSpec(name="clip", expert_type="clip"),
            ExpertSweepSpec(name="siglip", expert_type="siglip"),
        ],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
    )
    rows = run_expert_sweep(cfg)
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "error"
    saved = json.loads((tmp_path / "rows.json").read_text())
    assert len(saved) == 2
    assert saved[0]["expert_sweep_name"] == "clip"
    assert saved[1]["expert_sweep_name"] == "siglip"


def test_expert_sweep_supports_base_config_alias_and_model_path(tmp_path):
    import json
    from src.multimodal.benchmark.expert_sweep import load_expert_sweep_config

    cfg_path = tmp_path / "expert_sweep.json"
    cfg_path.write_text(
        json.dumps(
            {
                "base_config": "conf/task_matrix_router_parallel.dtd_resnet18.json",
                "experts": [
                    {
                        "name": "resnet18_classifier",
                        "expert_type": "resnet18_classifier",
                        "expert_model_path": "/tmp/expert.pt",
                        "expert_output_dim": 47,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_expert_sweep_config(str(cfg_path))
    assert cfg.base_task_matrix_config == "conf/task_matrix_router_parallel.dtd_resnet18.json"
    assert cfg.experts[0].expert_model_path == "/tmp/expert.pt"
