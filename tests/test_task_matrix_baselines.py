import torch

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


class _DummyRuntime:
    def __init__(self, policy_name: str):
        self.inject_via_layer = policy_name == "pre_attn_overwrite"
        self.semantic_expert = torch.nn.Linear(1, 1)
        self.fusion_policy = type("Policy", (), {"policy_name": policy_name, "to": lambda self, *args, **kwargs: self})()


class _DummyModel:
    def __init__(self):
        self.config = type("Cfg", (), {"hidden_size": 8})()

    def to(self, *args, **kwargs):
        return self


class _DummyHookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _DummyMLP:
    def __init__(self):
        self.last_hook = None

    def register_forward_hook(self, hook):
        self.last_hook = hook
        return _DummyHookHandle()


class _DummyLayer:
    def __init__(self):
        self.mlp = _DummyMLP()


class _DummyHookableModel(_DummyModel):
    def __init__(self):
        super().__init__()
        self.model = type("Inner", (), {"layers": [_DummyLayer()]})()


def test_run_task_matrix_emits_rows_for_unified_baselines(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    attach_calls = []
    train_calls = []
    eval_calls = []

    def attach_connector(cfg, model, baseline_mode):
        attach_calls.append(baseline_mode)
        if baseline_mode in {"llm_only", "text_prompt_only"}:
            return None
        return _DummyRuntime(runner._fusion_policy_for_mode(cfg, baseline_mode))

    def train_connector(model, runtime, train_loader, task, cfg, device, baseline_mode):
        train_calls.append((baseline_mode, id(runtime) if runtime is not None else None))

    def eval_task(model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode):
        eval_calls.append((baseline_mode, id(runtime) if runtime is not None else None))
        return {"accuracy": 0.5 if runtime is None else 0.75}

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", attach_connector)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", train_connector)
    monkeypatch.setattr(runner, "_eval_task", eval_task)

    cfg = TaskMatrixConfig(
        baseline_modes=[
            "llm_only",
            "text_prompt_only",
            "overwrite",
            "router_parallel",
            "shuffled_router_parallel",
            "zeroed_router_parallel",
        ],
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert [r["baseline_mode"] for r in rows] == [
        "llm_only",
        "text_prompt_only",
        "overwrite",
        "router_parallel",
        "shuffled_router_parallel",
        "zeroed_router_parallel",
    ]
    assert rows[0]["k"] == 0
    assert rows[2]["fusion_policy"] == "pre_attn_overwrite"
    assert rows[3]["fusion_policy"] == "pre_ffn_router_parallel"
    assert rows[4]["fusion_policy"] == "pre_ffn_router_parallel"
    assert rows[5]["fusion_policy"] == "pre_ffn_router_parallel"
    assert "shuffled" in rows[4]["baseline_note"]
    assert "zeroed" in rows[5]["baseline_note"]
    assert "baseline_note" in rows[1]
    assert attach_calls == ["llm_only", "text_prompt_only", "overwrite", "router_parallel"]
    assert [m for m, _ in train_calls] == ["llm_only", "text_prompt_only", "overwrite", "router_parallel"]
    router_runtime_ids = {runtime_id for mode, runtime_id in eval_calls if mode in {"router_parallel", "shuffled_router_parallel", "zeroed_router_parallel"}}
    assert len(router_runtime_ids) == 1


def test_run_task_matrix_accepts_expert_only_alias(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_tokenizer", lambda cfg: object())
    monkeypatch.setattr(runner, "_attach_direct_expert", lambda cfg: _DummyRuntime("expert_direct"))
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["expert_only"],
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert rows[0]["baseline_mode"] == "expert_direct"
    assert rows[0]["fusion_policy"] == "expert_direct"


def test_run_task_matrix_repeat_seeds_emits_aggregate_row(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", lambda cfg, model, baseline_mode: None)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["llm_only"],
        repeat_seeds=[43],
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert len(rows) == 3
    assert rows[0]["seed"] == 42
    assert rows[1]["seed"] == 43
    agg = rows[2]
    assert agg["seed"] == "aggregate"
    assert agg["metric"] == 0.5
    assert agg["accuracy"] == 0.5
    assert agg["metric_mean"] == 0.5


def test_run_task_matrix_router_parallel_respects_pre_ffn_policy(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", lambda cfg, model, baseline_mode: None)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["router_parallel"],
        fusion_policy="pre_ffn_router_parallel",
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert rows[0]["fusion_policy"] == "pre_ffn_router_parallel"


def test_run_task_matrix_router_parallel_respects_post_attn_layers_policy(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", lambda cfg, model, baseline_mode: None)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["router_parallel"],
        fusion_policy="post_attn_router_layers",
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert rows[0]["fusion_policy"] == "post_attn_router_layers"


def test_run_task_matrix_router_parallel_hybrid_routes_by_task(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", lambda cfg, model, baseline_mode: None)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["router_parallel"],
        fusion_policy="hybrid_task_routed",
        tasks=[
            TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1),
            TaskSpec(name="grounded_generation", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1),
        ],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert rows[0]["task"] == "finegrained"
    assert rows[0]["fusion_policy"] == "pre_ffn_router_parallel"
    assert rows[1]["task"] == "grounded_generation"
    assert rows[1]["fusion_policy"] == "post_attn_router_layers"


def test_run_task_matrix_vqa_rows_report_hf_dataset_name(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(runner, "_attach_connector", lambda cfg, model, baseline_mode: None)
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["llm_only"],
        tasks=[
            TaskSpec(
                name="vqa",
                vqa_hf_dataset_name="Graphcore/gqa-lxmert",
                max_train_samples=1,
                max_eval_samples=1,
                batch_size=1,
                train_steps=0,
            )
        ],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert rows[0]["dataset"] == "Graphcore/gqa-lxmert"


def test_attach_connector_loaded_pre_ffn_registers_hook_and_freezes_semantic_expert(monkeypatch):
    import src.multimodal.task_matrix.runner as runner

    class _LoadedPolicy(torch.nn.Module):
        policy_name = "pre_ffn_router_parallel"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def register_to_model(self, model, layer_idx):
            self.calls.append((model, layer_idx))

    loaded_expert = torch.nn.Linear(1, 1)
    loaded_policy = _LoadedPolicy()
    monkeypatch.setattr(
        runner,
        "load_expert_bundle",
        lambda *args, **kwargs: {
            "semantic_expert": loaded_expert,
            "fusion_policy": loaded_policy,
            "manifest": type("Manifest", (), {"fusion_policy": "pre_ffn_router_parallel"})(),
        },
    )

    cfg = runner.TaskMatrixConfig(
        baseline_modes=["router_parallel"],
        fusion_policy="pre_ffn_router_parallel",
        load_bundle_dir="/tmp/bundle",
        freeze_loaded_semantic_expert=True,
        freeze_loaded_fusion_policy=False,
        layer_idx=0,
        device="cpu",
    )
    model = _DummyHookableModel()
    runtime = runner._attach_connector(cfg, model, "router_parallel")

    assert runtime is not None
    assert runtime.inject_via_layer is False
    assert loaded_policy.calls == [(model, 0)]
    assert all(not p.requires_grad for p in loaded_expert.parameters())
    assert any(p.requires_grad for p in loaded_policy.parameters())


def test_build_model_tokenizer_respects_local_files_only_flag(monkeypatch):
    import src.multimodal.task_matrix.runner as runner

    seen = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    fake_config = type("Cfg", (), {})()
    fake_generation_config = type("GenCfg", (), {})()

    class _Tok:
        pad_token = None
        unk_token = "<unk>"
        pad_token_id = 0

    class _LoadedModel:
        def __init__(self):
            self.config = type("Cfg", (), {})()
            self.generation_config = type("GenCfg", (), {"pad_token_id": None})()

        def parameters(self):
            return []

    def _fake_tok_from_pretrained(model_name, **kwargs):
        seen["tokenizer"] = {"model_name": model_name, **kwargs}
        return _Tok()

    def _fake_model_from_pretrained(model_name, **kwargs):
        seen["model"] = {"model_name": model_name, **kwargs}
        return _LoadedModel()

    def _fake_config_from_pretrained(model_name, **kwargs):
        seen["config"] = {"model_name": model_name, **kwargs}
        return fake_config

    monkeypatch.setattr(runner.AutoConfig, "from_pretrained", _fake_config_from_pretrained)
    monkeypatch.setattr(runner.GenerationConfig, "from_model_config", lambda cfg: fake_generation_config)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", _fake_tok_from_pretrained)
    monkeypatch.setattr(runner.DomainQwenForCausalLM, "from_pretrained_qwen", _fake_model_from_pretrained)

    cfg = runner.TaskMatrixConfig(model_name="stub/model", server_models_path="/tmp/cache", local_files_only=True)
    _, tokenizer = runner._build_model_tokenizer(cfg)

    assert tokenizer.pad_token == "<unk>"
    assert runner.os.environ["HF_HUB_OFFLINE"] == "1"
    assert runner.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert seen["config"]["local_files_only"] is True
    assert seen["tokenizer"]["local_files_only"] is True
    assert seen["model"]["local_files_only"] is True
    assert seen["model"]["config"] is fake_config
    assert seen["model"]["generation_config"] is fake_generation_config
    assert seen["model"]["use_safetensors"] is False


def test_build_model_tokenizer_uses_offline_env_as_local_files_only(monkeypatch):
    import src.multimodal.task_matrix.runner as runner

    seen = {}
    fake_config = type("Cfg", (), {})()
    fake_generation_config = type("GenCfg", (), {})()

    class _Tok:
        pad_token = "<pad>"
        unk_token = "<unk>"
        pad_token_id = 0

    class _LoadedModel:
        def __init__(self):
            self.config = type("Cfg", (), {})()
            self.generation_config = type("GenCfg", (), {"pad_token_id": None})()

        def parameters(self):
            return []

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def _fake_tok_from_pretrained(model_name, **kwargs):
        seen["tokenizer"] = {"model_name": model_name, **kwargs}
        return _Tok()

    def _fake_model_from_pretrained(model_name, **kwargs):
        seen["model"] = {"model_name": model_name, **kwargs}
        return _LoadedModel()

    def _fake_config_from_pretrained(model_name, **kwargs):
        seen["config"] = {"model_name": model_name, **kwargs}
        return fake_config

    monkeypatch.setattr(runner.AutoConfig, "from_pretrained", _fake_config_from_pretrained)
    monkeypatch.setattr(runner.GenerationConfig, "from_model_config", lambda cfg: fake_generation_config)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", _fake_tok_from_pretrained)
    monkeypatch.setattr(runner.DomainQwenForCausalLM, "from_pretrained_qwen", _fake_model_from_pretrained)

    runner._build_model_tokenizer(runner.TaskMatrixConfig(model_name="stub/model", server_models_path="/tmp/cache"))

    assert seen["config"]["local_files_only"] is True
    assert seen["tokenizer"]["local_files_only"] is True
    assert seen["model"]["local_files_only"] is True
    assert seen["model"]["generation_config"] is fake_generation_config
    assert seen["model"]["use_safetensors"] is False


def test_suppress_hf_auto_conversion_noise_patches_transformers_symbols():
    import src.multimodal.task_matrix.runner as runner
    import transformers.modeling_utils as modeling_utils
    import transformers.safetensors_conversion as safetensors_conversion

    original_modeling = modeling_utils.auto_conversion
    original_safetensors = safetensors_conversion.auto_conversion
    with runner._suppress_hf_auto_conversion_noise(True):
        assert modeling_utils.auto_conversion is not original_modeling
        assert safetensors_conversion.auto_conversion is not original_safetensors
        assert modeling_utils.auto_conversion("stub/model") == (None, None, None)
        assert safetensors_conversion.auto_conversion("stub/model") == (None, None, None)

    assert modeling_utils.auto_conversion is original_modeling
    assert safetensors_conversion.auto_conversion is original_safetensors


def test_save_rows_supports_filename_only_outputs(monkeypatch, tmp_path):
    import src.multimodal.task_matrix.runner as runner

    monkeypatch.chdir(tmp_path)
    runner._save_rows([{"task": "finegrained", "metric": 1.0}], "rows.csv", "rows.json")
    assert (tmp_path / "rows.csv").exists()
    assert (tmp_path / "rows.json").exists()


def test_build_lr_scheduler_warmup_then_cosine_decay():
    import src.multimodal.task_matrix.runner as runner

    param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    cfg = runner.TaskMatrixConfig(lr_schedule="cosine", lr_warmup_steps=2)
    sched = runner._build_lr_scheduler(opt, cfg, total_steps=6)

    vals = []
    for _ in range(6):
        opt.step()
        sched.step()
        vals.append(opt.param_groups[0]["lr"])

    assert vals[0] >= 0.5
    assert vals[1] >= vals[0]
    assert vals[-1] < vals[2]


def test_task_ids_for_batch_handles_collated_task_name_lists():
    import src.multimodal.task_matrix.runner as runner

    cfg = runner.TaskMatrixConfig(task_conditioning=True, pre_router_max_task_ids=32)
    batch = {
        "prompt_tokens": torch.zeros((3, 5), dtype=torch.long),
        "task_name": ["finegrained", "finegrained", "finegrained"],
    }
    task_ids = runner._task_ids_for_batch(cfg, batch, device=torch.device("cpu"))
    expected = runner.task_name_to_id("finegrained", max_task_ids=32)
    assert task_ids is not None
    assert task_ids.shape == (3,)
    assert torch.all(task_ids == expected)


def test_annotate_joint_scores_applies_latency_format_and_instability_penalties():
    import src.multimodal.task_matrix.runner as runner

    cfg = runner.TaskMatrixConfig(
        joint_score_latency_weight=0.1,
        joint_score_format_weight=0.2,
        joint_score_instability_weight=0.3,
    )
    rows = [
        {
            "task": "grounded_generation",
            "metric_name": "accuracy",
            "accuracy": 0.8,
            "format_compliance": 0.75,
            "latency_ms_per_sample": 200.0,
            "metric_std": 0.1,
        }
    ]
    out = runner._annotate_joint_scores(rows, cfg)
    row = out[0]
    assert row["quality_score"] == 0.8
    assert row["format_fail_rate"] == 0.25
    assert row["latency_s_per_sample"] == 0.2
    assert row["instability_penalty"] == 0.1
    # 0.8 - (0.1*0.2) - (0.2*0.25) - (0.3*0.1) = 0.70
    assert abs(row["joint_score"] - 0.7) < 1e-9
