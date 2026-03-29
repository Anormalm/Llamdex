import torch

from src.multimodal.task_matrix.runner import TaskMatrixConfig, TaskSpec, run_task_matrix


class _DummyRuntime:
    def __init__(self, policy_name: str):
        self.inject_via_layer = policy_name == "pre_attn_overwrite"
        self.semantic_expert = torch.nn.Linear(1, 1)
        self.fusion_policy = type("Policy", (), {"policy_name": policy_name, "to": lambda self, device: self})()


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

    monkeypatch.setattr(runner, "_build_model_tokenizer", lambda cfg: (_DummyModel(), object()))
    monkeypatch.setattr(
        runner,
        "_attach_connector",
        lambda cfg, model, baseline_mode: None
        if baseline_mode in {"llm_only", "text_prompt_only"}
        else _DummyRuntime(runner._fusion_policy_for_mode(baseline_mode)),
    )
    monkeypatch.setattr(runner, "_build_task_loaders", lambda task, tokenizer, data_root, seed: (["train"], ["eval"]))
    monkeypatch.setattr(runner, "_train_connector", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_eval_task",
        lambda model, runtime, tokenizer, eval_loader, task, cfg, device, baseline_mode: {"accuracy": 0.5 if runtime is None else 0.75},
    )

    cfg = TaskMatrixConfig(
        baseline_modes=["llm_only", "text_prompt_only", "overwrite", "router_parallel"],
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    rows = run_task_matrix(cfg)
    assert [r["baseline_mode"] for r in rows] == ["llm_only", "text_prompt_only", "overwrite", "router_parallel"]
    assert rows[0]["k"] == 0
    assert rows[2]["fusion_policy"] == "pre_attn_overwrite"
    assert rows[3]["fusion_policy"] == "post_attn_router_parallel"
    assert "baseline_note" in rows[1]


def test_run_task_matrix_rejects_expert_only(tmp_path):
    cfg = TaskMatrixConfig(
        baseline_modes=["expert_only"],
        tasks=[TaskSpec(name="finegrained", train_steps=0, max_train_samples=1, max_eval_samples=1, batch_size=1)],
        out_csv=str(tmp_path / "rows.csv"),
        out_json=str(tmp_path / "rows.json"),
        device="cpu",
    )
    try:
        run_task_matrix(cfg)
    except ValueError as exc:
        assert "expert_only baseline has been removed" in str(exc)
    else:
        raise AssertionError("Expected expert_only to be rejected.")


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
