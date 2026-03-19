import torch


def test_build_rag_db_uses_images_key(monkeypatch):
    import src.multimodal.baselines.suite as suite

    class DummyExpert:
        def to(self, dev):
            return self

        def __call__(self, img):
            return type("Out", (), {"logits": torch.tensor([[3.0, 1.0, 0.5]])})()

    class DummyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"images": torch.zeros(3, 224, 224), "question_text": "q", "target_token_id": 1}

    cfg = suite.BaselineSuiteConfig(dataset_name="cifar10", data_root="/tmp", seed=42)

    monkeypatch.setattr(suite, "build_vision_expert", lambda *args, **kwargs: DummyExpert())
    monkeypatch.setattr(suite, "CIFARSingleImageQADataset", lambda **kwargs: DummyDataset())

    docs = suite._build_rag_db(cfg, tokenizer=None, dev=torch.device("cpu"))
    assert len(docs) == 1
    assert "class_" in docs[0]
