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
