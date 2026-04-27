import torch

import src.multimodal.task_matrix.runner as runner


class _DummyModel:
    def eval(self):
        return self


class _DummyRuntime:
    def __init__(self):
        self.inject_via_layer = False
        self.fusion_policy = object()


class _DummyTokenizer:
    def decode(self, ids):
        if list(ids) == [2]:
            return " A"
        return ""


def test_grounded_generation_uses_unconstrained_decode_for_rationale(monkeypatch):
    captured = {"allowed": "unset", "prefix_text": ""}

    def _fake_forward(*args, **kwargs):
        return torch.tensor([[0.0, 0.0, 10.0, 0.0]], dtype=torch.float32)

    def _fake_expert_input(*args, **kwargs):
        return torch.zeros(1, 3, 224, 224)

    def _fake_generate(*args, **kwargs):
        captured["allowed"] = kwargs.get("allowed_token_ids")
        captured["prefix_text"] = kwargs.get("prefix_text")
        return "Answer: A. Rationale: braided texture pattern."

    monkeypatch.setattr(runner, "_forward_logits_for_batch", _fake_forward)
    monkeypatch.setattr(runner, "_expert_input_from_batch", _fake_expert_input)
    monkeypatch.setattr(runner, "_generate_rationale", _fake_generate)

    batch = {
        "prompt_tokens": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "target_token_id": torch.tensor([2], dtype=torch.long),
        "label_idx": torch.tensor([0], dtype=torch.long),
        "task_name": "grounded_generation",
        "allowed_token_ids": [2, 3],
        "code_labels": ["braided texture", "striped coat"],
        "images": torch.zeros(1, 3, 224, 224),
        "rationale_keywords": [["braided", "texture"]],
    }

    task = runner.TaskSpec(name="grounded_generation", constrained_decoding=True, max_eval_samples=1)
    cfg = runner.TaskMatrixConfig(tasks=[task])

    metrics = runner._eval_task(
        model=_DummyModel(),
        runtime=_DummyRuntime(),
        tokenizer=_DummyTokenizer(),
        eval_loader=[batch],
        task=task,
        cfg=cfg,
        device=torch.device("cpu"),
        baseline_mode="router_parallel",
    )

    assert captured["allowed"] is None
    assert captured["prefix_text"] == "Answer: A. Rationale: braided texture"
    assert metrics["rationale_consistency"] == 1.0
    assert metrics["format_compliance"] == 1.0
    assert metrics["answer_code_valid_rate"] == 1.0
    assert metrics["answer_code_accuracy"] == 1.0
    assert metrics["rationale_hallucination_rate"] == 0.0
    assert metrics["rationale_faithful_rate"] == 1.0


def test_extract_answer_rationale_prefers_latest_match():
    text = (
        "Format: Answer: <code>. Rationale: <short sentence with class words>. "
        "Answer: B. Rationale: braided woven texture."
    )
    ans, rat = runner._extract_answer_rationale(text)
    assert ans == "B"
    assert rat == "braided woven texture."


def test_extract_answer_rationale_parses_json_payload():
    text = 'prefix {"answer":"C","rationale":"woven rough texture"} suffix'
    ans, rat = runner._extract_answer_rationale(text)
    assert ans == "C"
    assert rat == "woven rough texture"


def test_canonicalize_grounded_output_falls_back_to_predicted_code():
    text = "braided woven texture pattern."
    out = runner._canonicalize_grounded_output(text, "A")
    assert out == "Answer: A. Rationale: braided woven texture pattern."


def test_predicted_label_text_uses_predicted_code_lookup():
    batch = {"allowed_token_ids": [11, 22], "code_labels": ["braided weave", "striped coat"]}
    assert runner._predicted_label_text(batch, 11) == "braided weave"
    assert runner._predicted_label_text(batch, 99) == ""
