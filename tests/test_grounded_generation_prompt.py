import torch

import src.multimodal.task_matrix.datasets as ds_mod


class _DummyTokenizer:
    pad_token_id = 0


def test_grounded_generation_prompt_includes_mapping_and_class_word_instruction(monkeypatch):
    captured = {}

    def _fake_chat_template_tokens(tokenizer, msgs):
        captured["msgs"] = msgs
        return torch.tensor([1, 2, 3], dtype=torch.long)

    def _fake_base_getitem(self, idx):
        return {
            "images": torch.zeros(3, 224, 224),
            "prompt_tokens": torch.tensor([1], dtype=torch.long),
            "prompt_mask": torch.tensor([1], dtype=torch.long),
            "target_token_id": torch.tensor(1, dtype=torch.long),
            "label_idx": torch.tensor(0, dtype=torch.long),
            "task_name": "finegrained",
            "allowed_token_ids": [1, 2],
            "rationale_keywords": ["braided", "woven"],
        }

    monkeypatch.setattr(ds_mod, "_chat_template_tokens", _fake_chat_template_tokens)
    monkeypatch.setattr(ds_mod.FineGrainedPetDataset, "__getitem__", _fake_base_getitem)

    ds = object.__new__(ds_mod.GroundedGenerationDataset)
    ds.tokenizer = _DummyTokenizer()
    ds.task_label = "texture class"
    ds.codes = ["A", "B"]
    ds.class_names = ["braided", "striped"]

    row = ds_mod.GroundedGenerationDataset.__getitem__(ds, 0)

    assert row["task_name"] == "grounded_generation"
    assert row["rationale_target"] == "braided woven"
    assert "codes: a=braided, b=striped" in captured["msgs"][1]["content"].lower()
    assert "include class words" in captured["msgs"][0]["content"].lower()
