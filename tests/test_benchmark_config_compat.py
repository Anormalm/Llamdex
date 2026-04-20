import json


def test_baseline_suite_loader_ignores_deprecated_enable_expert_only(tmp_path):
    from src.multimodal.baselines.suite import load_baseline_suite_config

    cfg_path = tmp_path / "baseline.json"
    cfg_path.write_text(
        json.dumps(
            {
                "server_models_path": "/tmp/hf_cache",
                "model_name": "Qwen/Qwen3.5-9B",
                "enable_expert_only": True,
                "enable_llm_only": True,
                "enable_two_stage": False,
                "enable_rag": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_baseline_suite_config(str(cfg_path))
    assert cfg.enable_llm_only is True
    assert cfg.server_models_path == "/tmp/hf_cache"


def test_task_matrix_loader_reads_server_models_path(tmp_path):
    from src.multimodal.task_matrix.runner import load_task_matrix_config

    cfg_path = tmp_path / "task_matrix.json"
    cfg_path.write_text(
        json.dumps(
            {
                "server_models_path": "/tmp/hf_cache",
                "model_name": "Qwen/Qwen3.5-9B",
                "tasks": [{"name": "finegrained"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_task_matrix_config(str(cfg_path))
    assert cfg.server_models_path == "/tmp/hf_cache"


def test_task_matrix_loader_defaults_to_router_parallel(tmp_path):
    from src.multimodal.task_matrix.runner import load_task_matrix_config

    cfg_path = tmp_path / "task_matrix.json"
    cfg_path.write_text(
        json.dumps(
            {
                "server_models_path": "/tmp/hf_cache",
                "model_name": "Qwen/Qwen3.5-9B",
                "tasks": [{"name": "finegrained", "image_dataset_name": "dtd"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_task_matrix_config(str(cfg_path))
    assert cfg.baseline_modes == ["router_parallel"]
    assert cfg.expert_type == "siglip"
    assert cfg.upload_scope == "expert_only"
    assert cfg.tasks[0].image_dataset_name == "dtd"


def test_remote_bootstrap_bundle_config_uses_expert_only_upload():
    from src.multimodal.task_matrix.runner import load_task_matrix_config

    cfg = load_task_matrix_config("conf/task_matrix_bootstrap_bundle.pre_ffn.remote.json")
    assert cfg.save_bundle_dir == "/disk1/lfhu/runs/bundles_preffn_uploaded_qwen35"
    assert cfg.upload_scope == "expert_only"
