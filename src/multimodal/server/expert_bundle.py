from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import ExpertEncoderSpec, build_expert_encoder
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert, build_fusion_policy


BUNDLE_SCHEMA_VERSION = "1.0"
BUNDLE_WEIGHTS_FILE = "bundle.pt"
BUNDLE_MANIFEST_FILE = "bundle_meta.json"
UPLOAD_SCOPE_FULL = "full_bundle"
UPLOAD_SCOPE_EXPERT_ONLY = "expert_only"


class BundleValidationError(ValueError):
    pass


@dataclass
class BundleCompatibilitySpec:
    hidden_size: Optional[int] = None
    evidence_dim: Optional[int] = None
    num_tokens: Optional[int] = None
    expert_output_dim: Optional[int] = None
    fusion_policy: Optional[str] = None
    model_name: Optional[str] = None
    layer_idx: Optional[int] = None
    expected_feature_format: Optional[str] = None
    class_count: Optional[int] = None


@dataclass
class BundleManifest:
    schema_version: str = BUNDLE_SCHEMA_VERSION
    bundle_id: str = ""
    encoder_spec: Dict[str, Any] = field(default_factory=dict)
    evidence_dim: int = 0
    expert_output_dim: int = 0
    projector_hidden_size: int = 0
    projector_num_tokens: int = 0
    projector_alpha: float = 1.0
    fusion_policy: str = "pre_attn_overwrite"
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    label_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload_sha256: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, raw: Dict[str, Any]) -> "BundleManifest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: raw[k] for k in raw.keys() & known}
        manifest = cls(**payload)
        # Preserve older metadata layout.
        if not manifest.preprocessing_config and "normalization_stats" in raw:
            manifest.preprocessing_config = raw.get("normalization_stats") or {}
        return manifest


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_feature_format(encoder_spec: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    fmt = metadata.get("feature_format")
    if fmt:
        return str(fmt)
    expert_type = str(encoder_spec.get("expert_type", "")).lower()
    if expert_type in {"xgboost", "ft_transformer", "tabpfn", "tabular"}:
        return "tabular"
    if expert_type in {"clip", "siglip", "groundingdino_sam2"}:
        return "image_tensor"
    return "unknown"


def _normalize_manifest(
    *,
    encoder_spec: Dict[str, Any],
    semantic_expert: SemanticEvidenceDomainExpert,
    fusion_policy,
    preprocessing_config: Optional[Dict[str, Any]],
    label_schema: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> BundleManifest:
    meta = dict(metadata or {})
    evidence_builder = semantic_expert.evidence_builder
    if not hasattr(evidence_builder, "expert_encoder"):
        raise BundleValidationError("Only EncoderEvidenceBuilder-style bundles with a frozen expert encoder can be exported.")
    output_dim = int(getattr(evidence_builder, "output_dim", semantic_expert.evidence_builder.evidence_dim))
    return BundleManifest(
        encoder_spec=dict(encoder_spec),
        evidence_dim=int(evidence_builder.evidence_dim),
        expert_output_dim=output_dim,
        projector_hidden_size=int(semantic_expert.projector.hidden_size),
        projector_num_tokens=int(semantic_expert.projector.num_tokens),
        projector_alpha=float(semantic_expert.projector.alpha),
        fusion_policy=str(getattr(fusion_policy, "policy_name", "pre_attn_overwrite")),
        preprocessing_config=dict(preprocessing_config or {}),
        label_schema=dict(label_schema or {}),
        metadata=meta,
    )


def validate_bundle_manifest(
    manifest: BundleManifest,
    *,
    compatibility: Optional[BundleCompatibilitySpec] = None,
) -> None:
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"Unsupported bundle schema_version={manifest.schema_version!r}; expected {BUNDLE_SCHEMA_VERSION!r}."
        )
    if not manifest.encoder_spec:
        raise BundleValidationError("Bundle manifest is missing encoder_spec.")
    if manifest.evidence_dim <= 0:
        raise BundleValidationError(f"Invalid evidence_dim={manifest.evidence_dim}.")
    if manifest.projector_hidden_size <= 0:
        raise BundleValidationError(f"Invalid projector_hidden_size={manifest.projector_hidden_size}.")
    if manifest.projector_num_tokens <= 0:
        raise BundleValidationError(f"Invalid projector_num_tokens={manifest.projector_num_tokens}.")
    if manifest.expert_output_dim <= 0:
        raise BundleValidationError(f"Invalid expert_output_dim={manifest.expert_output_dim}.")

    if compatibility is None:
        return
    if compatibility.hidden_size is not None and int(compatibility.hidden_size) != int(manifest.projector_hidden_size):
        raise BundleValidationError(
            f"Bundle hidden size mismatch: bundle={manifest.projector_hidden_size}, expected={compatibility.hidden_size}."
        )
    if compatibility.evidence_dim is not None and int(compatibility.evidence_dim) != int(manifest.evidence_dim):
        raise BundleValidationError(
            f"Bundle evidence_dim mismatch: bundle={manifest.evidence_dim}, expected={compatibility.evidence_dim}."
        )
    if compatibility.num_tokens is not None and int(compatibility.num_tokens) != int(manifest.projector_num_tokens):
        raise BundleValidationError(
            f"Bundle num_tokens mismatch: bundle={manifest.projector_num_tokens}, expected={compatibility.num_tokens}."
        )
    if compatibility.expert_output_dim is not None and int(compatibility.expert_output_dim) != int(manifest.expert_output_dim):
        raise BundleValidationError(
            f"Bundle expert_output_dim mismatch: bundle={manifest.expert_output_dim}, expected={compatibility.expert_output_dim}."
        )
    if compatibility.fusion_policy is not None and str(compatibility.fusion_policy) != str(manifest.fusion_policy):
        raise BundleValidationError(
            f"Bundle fusion_policy mismatch: bundle={manifest.fusion_policy}, expected={compatibility.fusion_policy}."
        )
    if compatibility.model_name is not None and manifest.metadata.get("model_name") not in {None, compatibility.model_name}:
        raise BundleValidationError(
            f"Bundle model_name mismatch: bundle={manifest.metadata.get('model_name')}, expected={compatibility.model_name}."
        )
    if compatibility.layer_idx is not None and manifest.metadata.get("layer_idx") not in {None, compatibility.layer_idx}:
        raise BundleValidationError(
            f"Bundle layer_idx mismatch: bundle={manifest.metadata.get('layer_idx')}, expected={compatibility.layer_idx}."
        )
    if compatibility.expected_feature_format is not None:
        feature_format = _infer_feature_format(manifest.encoder_spec, manifest.metadata)
        if feature_format != compatibility.expected_feature_format:
            raise BundleValidationError(
                f"Bundle feature_format mismatch: bundle={feature_format}, expected={compatibility.expected_feature_format}."
            )
    if compatibility.class_count is not None and manifest.label_schema:
        classes = manifest.label_schema.get("classes")
        if isinstance(classes, list) and len(classes) != int(compatibility.class_count):
            raise BundleValidationError(
                f"Bundle class_count mismatch: bundle={len(classes)}, expected={compatibility.class_count}."
            )


def save_expert_bundle(
    bundle_dir: str,
    *,
    encoder_spec: Dict[str, Any],
    semantic_expert: SemanticEvidenceDomainExpert,
    fusion_policy,
    normalization_stats: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    label_schema: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_scope: str = UPLOAD_SCOPE_FULL,
) -> str:
    os.makedirs(bundle_dir, exist_ok=True)
    upload_scope = str(upload_scope or UPLOAD_SCOPE_FULL).strip().lower()
    if upload_scope not in {UPLOAD_SCOPE_FULL, UPLOAD_SCOPE_EXPERT_ONLY}:
        raise BundleValidationError(f"Unsupported upload_scope={upload_scope!r}.")
    manifest = _normalize_manifest(
        encoder_spec=encoder_spec,
        semantic_expert=semantic_expert,
        fusion_policy=fusion_policy,
        preprocessing_config=preprocessing_config if preprocessing_config is not None else normalization_stats,
        label_schema=label_schema,
        metadata={**dict(metadata or {}), "upload_scope": upload_scope},
    )
    payload = {
        "encoder_state_dict": semantic_expert.evidence_builder.expert_encoder.state_dict(),
        "preprocessing_config": manifest.preprocessing_config,
        "normalization_stats": manifest.preprocessing_config,
        "label_schema": manifest.label_schema,
        "metadata": {
            **manifest.metadata,
            "schema_version": manifest.schema_version,
            "bundle_id": manifest.bundle_id,
        },
    }
    if upload_scope == UPLOAD_SCOPE_FULL:
        payload["builder_state_dict"] = semantic_expert.evidence_builder.state_dict()
        payload["projector_state_dict"] = semantic_expert.projector.state_dict()
        payload["fusion_policy_state_dict"] = fusion_policy.state_dict() if fusion_policy is not None else {}
    weights_path = os.path.join(bundle_dir, BUNDLE_WEIGHTS_FILE)
    torch.save(payload, weights_path)
    manifest.payload_sha256 = _sha256_file(weights_path)
    manifest.bundle_id = manifest.bundle_id or manifest.payload_sha256[:16]
    payload["metadata"]["bundle_id"] = manifest.bundle_id
    torch.save(payload, weights_path)
    manifest.payload_sha256 = _sha256_file(weights_path)
    with open(os.path.join(bundle_dir, BUNDLE_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest.to_json_dict(), f, indent=2)
    return bundle_dir


def read_bundle_manifest(bundle_dir: str) -> BundleManifest:
    manifest_path = os.path.join(bundle_dir, BUNDLE_MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        raise BundleValidationError(f"Bundle manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    manifest = BundleManifest.from_json_dict(raw)
    validate_bundle_manifest(manifest)
    return manifest


def load_expert_bundle(
    bundle_dir: str,
    *,
    compatibility: Optional[BundleCompatibilitySpec] = None,
    verify_checksum: bool = True,
):
    weights_path = os.path.join(bundle_dir, BUNDLE_WEIGHTS_FILE)
    if not os.path.isfile(weights_path):
        raise BundleValidationError(f"Bundle weights not found: {weights_path}")
    manifest = read_bundle_manifest(bundle_dir)
    validate_bundle_manifest(manifest, compatibility=compatibility)
    if verify_checksum and manifest.payload_sha256:
        actual = _sha256_file(weights_path)
        if actual != manifest.payload_sha256:
            raise BundleValidationError(
                f"Bundle checksum mismatch for {weights_path}: expected={manifest.payload_sha256}, actual={actual}."
            )

    payload = torch.load(weights_path, map_location="cpu")
    upload_scope = str(payload.get("metadata", {}).get("upload_scope") or manifest.metadata.get("upload_scope") or UPLOAD_SCOPE_FULL).strip().lower()
    enc_spec = ExpertEncoderSpec(**manifest.encoder_spec)
    encoder = build_expert_encoder(enc_spec)
    encoder.load_state_dict(payload.get("encoder_state_dict", {}), strict=False)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    builder = EncoderEvidenceBuilder(
        expert_encoder=encoder,
        evidence_dim=int(manifest.evidence_dim),
        output_dim=int(manifest.expert_output_dim),
    )
    if upload_scope == UPLOAD_SCOPE_FULL:
        builder.load_state_dict(payload.get("builder_state_dict", {}), strict=False)
    builder.eval()

    projector = EvidenceProjector(
        evidence_dim=int(manifest.evidence_dim),
        hidden_size=int(manifest.projector_hidden_size),
        num_tokens=int(manifest.projector_num_tokens),
        alpha=float(manifest.projector_alpha),
    )
    if upload_scope == UPLOAD_SCOPE_FULL:
        projector.load_state_dict(payload.get("projector_state_dict", {}), strict=False)
    projector.eval()

    semantic_expert = SemanticEvidenceDomainExpert(builder, projector)
    semantic_expert.eval()

    policy = build_fusion_policy(manifest.fusion_policy, hidden_size=projector.hidden_size)
    if upload_scope == UPLOAD_SCOPE_FULL:
        policy.load_state_dict(payload.get("fusion_policy_state_dict", {}), strict=False)
    policy.eval()
    for module in (semantic_expert.evidence_builder.expert_encoder,):
        for p in module.parameters():
            p.requires_grad = False

    return {
        "semantic_expert": semantic_expert,
        "fusion_policy": policy,
        "preprocessing_config": payload.get("preprocessing_config", payload.get("normalization_stats", {})),
        "normalization_stats": payload.get("normalization_stats", payload.get("preprocessing_config", {})),
        "label_schema": payload.get("label_schema", manifest.label_schema),
        "metadata": payload.get("metadata", manifest.metadata),
        "manifest": manifest,
    }
