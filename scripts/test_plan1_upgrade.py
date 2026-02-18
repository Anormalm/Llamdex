"""
Forward-pass validation for the Plan-1 IMG upgrade.

Checks:
1) Vision evidence -> token projection
2) Text evidence -> token projection
3) Population aggregation -> token projection
"""

import os
import sys
import random

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.vision.vision_domain_expert import VisionDomainExpert


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_test():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 4096
    num_tokens = 10
    batch_size = 2
    evidence_dim = 512

    # 1) Vision evidence path
    vision_expert = VisionDomainExpert(
        embed_size=embed_size,
        num_tokens=num_tokens,
        evidence_source="vision",
        evidence_dim=evidence_dim,
        task="single",
        vision_model="resnet18",
        vision_output="logits",
    ).to(device)
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    out_vision = vision_expert.forward_with_features(images)
    assert out_vision.shape == (batch_size, num_tokens, embed_size)
    print(f"[OK] vision single: {tuple(out_vision.shape)}")

    # 2) Text evidence path
    text_expert = VisionDomainExpert(
        embed_size=embed_size,
        num_tokens=num_tokens,
        evidence_source="text",
        evidence_dim=evidence_dim,
        task="single",
        text_encoder_model="distilroberta-base",
    ).to(device)
    text_payload = {"texts": ["This is a CIFAR-10 image of a truck.", "This is a CIFAR-10 image of an airplane."]}
    out_text = text_expert.forward_with_features(text_payload)
    assert out_text.shape == (batch_size, num_tokens, embed_size)
    print(f"[OK] text single: {tuple(out_text.shape)}")

    # 3) Population aggregation path
    pop_expert = VisionDomainExpert(
        embed_size=embed_size,
        num_tokens=num_tokens,
        evidence_source="vision",
        evidence_dim=evidence_dim,
        task="population",
        vision_model="resnet18",
        vision_output="logits",
        num_classes=10,
    ).to(device)
    pop_images = torch.randn(1, 8, 3, 32, 32, device=device)
    out_pop = pop_expert.forward_with_features({"images": pop_images})
    assert out_pop.shape == (1, num_tokens, embed_size)
    print(f"[OK] population: {tuple(out_pop.shape)}")

    print("Plan-1 upgrade forward-pass checks passed.")


if __name__ == "__main__":
    run_test()
