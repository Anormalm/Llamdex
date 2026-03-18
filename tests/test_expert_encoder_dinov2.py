import torch

from src.multimodal.framework.expert_encoders import DINOv2ExpertEncoder, ExpertEncoderSpec, build_expert_encoder


class _DummyDINOModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 12):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        pooled = torch.ones((b, self.config.hidden_size), dtype=pixel_values.dtype, device=pixel_values.device)
        return type("Out", (), {"pooler_output": pooled, "last_hidden_state": pooled.unsqueeze(1)})()


def test_build_expert_encoder_supports_dinov2(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: _DummyDINOModel(hidden_size=12))
    enc = build_expert_encoder(
        ExpertEncoderSpec(
            expert_type="dinov2",
            model_id="facebook/dinov2-base",
            output_dim=10,
            cache_dir="/disk1/lfhu/hf_cache",
        )
    )
    assert isinstance(enc, DINOv2ExpertEncoder)
    images = torch.randn(2, 3, 32, 32)
    out = enc.encode({"images": images})
    assert out.shape == (2, 10)
