from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch

from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, build_expert_encoder
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert, build_fusion_policy


def save_expert_bundle(
    bundle_dir: str,
    *,
    encoder_spec: Dict[str, Any],
    semantic_expert: SemanticEvidenceDomainExpert,
    fusion_policy,
    normalization_stats: Optional[Dict[str, Any]] = None,
    label_schema: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(bundle_dir, exist_ok=True)
    meta = dict(metadata or {})
    meta["encoder_spec"] = dict(encoder_spec)
    meta["evidence_dim"] = int(semantic_expert.evidence_builder.evidence_dim)
    meta["expert_output_dim"] = int(getattr(semantic_expert.evidence_builder, "output_dim", meta["evidence_dim"]))
    meta["projector_hidden_size"] = int(semantic_expert.projector.hidden_size)
    meta["projector_num_tokens"] = int(semantic_expert.projector.num_tokens)
    meta["projector_alpha"] = float(semantic_expert.projector.alpha)
    meta["fusion_policy"] = str(getattr(fusion_policy, "policy_name", "pre_attn_overwrite"))

    payload = {
        "encoder_state_dict": semantic_expert.evidence_builder.expert_encoder.state_dict(),
        "builder_state_dict": semantic_expert.evidence_builder.state_dict(),
        "projector_state_dict": semantic_expert.projector.state_dict(),
        "fusion_policy_state_dict": fusion_policy.state_dict() if fusion_policy is not None else {},
        "normalization_stats": normalization_stats or {},
        "label_schema": label_schema or {},
        "metadata": meta,
    }
    weights_path = os.path.join(bundle_dir, "bundle.pt")
    torch.save(payload, weights_path)
    with open(os.path.join(bundle_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return bundle_dir


def load_expert_bundle(bundle_dir: str):
    payload = torch.load(os.path.join(bundle_dir, "bundle.pt"), map_location="cpu")
    meta = payload.get("metadata", {})
    enc_spec = ExpertEncoderSpec(**meta["encoder_spec"])
    encoder = build_expert_encoder(enc_spec)
    encoder.load_state_dict(payload.get("encoder_state_dict", {}), strict=False)
    for p in encoder.parameters():
        p.requires_grad = False

    evidence_dim = int(meta["evidence_dim"])
    output_dim = int(meta.get("expert_output_dim", evidence_dim))
    builder = EncoderEvidenceBuilder(expert_encoder=encoder, evidence_dim=evidence_dim, output_dim=output_dim)
    builder.load_state_dict(payload.get("builder_state_dict", {}), strict=False)

    projector = EvidenceProjector(
        evidence_dim=evidence_dim,
        hidden_size=int(meta["projector_hidden_size"]),
        num_tokens=int(meta["projector_num_tokens"]),
        alpha=float(meta.get("projector_alpha", 1.0)),
    )
    projector.load_state_dict(payload.get("projector_state_dict", {}), strict=False)

    semantic_expert = SemanticEvidenceDomainExpert(builder, projector)
    policy = build_fusion_policy(meta.get("fusion_policy", "pre_attn_overwrite"), hidden_size=projector.hidden_size)
    policy.load_state_dict(payload.get("fusion_policy_state_dict", {}), strict=False)
    return {
        "semantic_expert": semantic_expert,
        "fusion_policy": policy,
        "normalization_stats": payload.get("normalization_stats", {}),
        "label_schema": payload.get("label_schema", {}),
        "metadata": meta,
    }

