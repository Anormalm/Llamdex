import pandas as pd

from src.dataset.utils import TextDataset, get_collate_fn


class _DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, return_tensors="pt"):
        text = " ".join([m["content"] for m in messages])
        import torch

        ids = [min(255, ord(c)) for c in text[:32]]
        if not ids:
            ids = [1]
        return torch.tensor([ids], dtype=torch.long)


def test_tabular_dataset_and_collate_still_work():
    dataset_json = {
        "X": {
            "age": {"type": "int", "range": [0, 100]},
            "income": {"type": "int", "range": [0, 100000]},
        },
        "y": {"name": "answer", "type": "category"},
    }
    df = pd.DataFrame(
        [
            {
                "formatted_text": "age: 30 income: 50000 Answer:",
                "prompt_to_llm": "unused",
                "answer": "A",
                "age": 30,
                "income": 50000,
            },
            {
                "formatted_text": "age: 40 income: 70000 Answer:",
                "prompt_to_llm": "unused",
                "answer": "B",
                "age": 40,
                "income": 70000,
            },
        ]
    )
    ds = TextDataset(
        data=df,
        answer_column="answer",
        tokenizer=_DummyTokenizer(),
        dataset_json=dataset_json,
        add_features=False,
    )
    b0 = ds[0]
    b1 = ds[1]
    collate = get_collate_fn(pad_token_id=0)
    out = collate([b0, b1])
    assert "tokens" in out and "attention_mask" in out and "features" in out and "labels" in out
    assert out["tokens"].shape[0] == 2
