from types import SimpleNamespace

import torch

from src.model.DomainQwenModel import DomainQwenDecoderLayer, DomainQwenForCausalLM


class _RecordingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, *args, **kwargs):
        self.seen = hidden_states.detach().clone()
        return hidden_states + 1.0


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size=3, vocab_size=5):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, to_dict=lambda: {"hidden_size": hidden_size})
        self.generation_config = SimpleNamespace(pad_token_id=None)
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([_RecordingLayer()]))
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
        _ = (attention_mask, output_hidden_states, return_dict, kwargs)
        batch, seq_len = input_ids.shape
        hidden = torch.zeros(batch, seq_len, self.config.hidden_size, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class _TokenExpert(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = torch.nn.Parameter(tokens.clone())
        self.encoder_trainable = None
        self.decoder_trainable = None

    def forward_with_features(self, x):
        batch = x.shape[0]
        return self.tokens.unsqueeze(0).expand(batch, -1, -1)

    def setup_encoder(self, requires_grad):
        self.encoder_trainable = requires_grad

    def setup_decoder(self, requires_grad):
        self.decoder_trainable = requires_grad


def test_domain_qwen_legacy_add_expert_overwrites_reserved_slots():
    model = DomainQwenForCausalLM(_FakeBackbone(hidden_size=3))
    expert_tokens = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    expert = _TokenExpert(expert_tokens)

    model.add_expert_(expert, layer_id=0, map_to_expert_emb=None)
    assert isinstance(model.model.layers[0], DomainQwenDecoderLayer)

    input_ids = torch.ones(1, 4, dtype=torch.long)
    model(input_ids=input_ids, expert_inputs=(torch.zeros(1, 3),), return_dict=True)

    seen = model.model.layers[0].layer.seen
    assert torch.equal(seen[:, :2, :], torch.zeros(1, 2, 3))
    assert torch.equal(seen[:, 2:, :], expert_tokens.unsqueeze(0))
    assert model.model.layers[0]._active_expert_inputs is None


def test_domain_qwen_setup_encoder_decoder_delegates_to_attached_expert():
    model = DomainQwenForCausalLM(_FakeBackbone(hidden_size=3))
    expert = _TokenExpert(torch.zeros(2, 3))
    model.add_expert_(expert, layer_id=0, map_to_expert_emb=None)

    model.setup_encoder(0, 0, True)
    model.setup_decoder(0, 0, False)

    assert expert.encoder_trainable is True
    assert expert.decoder_trainable is False
