import torch

from src.multimodal.framework.builders import EncoderEvidenceBuilder
from src.multimodal.framework.expert_encoders import TabularExpertEncoder
from src.multimodal.injection import EvidenceProjector, SemanticEvidenceDomainExpert, build_fusion_policy
from src.multimodal.server import (
    BundleCompatibilitySpec,
    BundleServiceRuntime,
    BundleValidationError,
    ServiceRequest,
    build_service_response,
    load_expert_bundle,
    read_bundle_manifest,
    save_expert_bundle,
)


def _build_semantic_expert():
    encoder = TabularExpertEncoder(model_kind="xgboost", model_path=None, output_dim=6)
    builder = EncoderEvidenceBuilder(expert_encoder=encoder, evidence_dim=4, output_dim=6)
    projector = EvidenceProjector(evidence_dim=4, hidden_size=8, num_tokens=2, alpha=1.0)
    return SemanticEvidenceDomainExpert(builder, projector)


def test_bundle_roundtrip_and_manifest_validation(tmp_path):
    semantic_expert = _build_semantic_expert()
    fusion_policy = build_fusion_policy("post_attn_router_parallel", hidden_size=8)
    bundle_dir = tmp_path / "bundle"
    save_expert_bundle(
        str(bundle_dir),
        encoder_spec={
            "expert_type": "tabular",
            "model_id": None,
            "model_path": None,
            "output_dim": 6,
            "cache_dir": "/disk1/lfhu/hf_cache",
            "use_runtime_detector": False,
        },
        semantic_expert=semantic_expert,
        fusion_policy=fusion_policy,
        preprocessing_config={"normalization": "client_bundle"},
        label_schema={"classes": ["No", "Yes"]},
        metadata={"model_name": "Qwen/Qwen3.5-9B", "layer_idx": 3, "feature_format": "tabular"},
    )

    manifest = read_bundle_manifest(str(bundle_dir))
    assert manifest.schema_version == "1.0"
    assert manifest.fusion_policy == "post_attn_router_parallel"
    assert manifest.preprocessing_config["normalization"] == "client_bundle"
    assert manifest.label_schema["classes"] == ["No", "Yes"]

    loaded = load_expert_bundle(
        str(bundle_dir),
        compatibility=BundleCompatibilitySpec(
            hidden_size=8,
            evidence_dim=4,
            num_tokens=2,
            expert_output_dim=6,
            model_name="Qwen/Qwen3.5-9B",
            layer_idx=3,
            expected_feature_format="tabular",
            class_count=2,
        ),
    )
    expert = loaded["semantic_expert"]
    sample = {"tabular": torch.randn(2, 6)}
    tokens = expert.forward_with_features(sample)
    assert tokens.shape == (2, 2, 8)
    assert loaded["manifest"].bundle_id


def test_bundle_validation_rejects_dimension_mismatch(tmp_path):
    semantic_expert = _build_semantic_expert()
    fusion_policy = build_fusion_policy("pre_attn_overwrite", hidden_size=8)
    bundle_dir = tmp_path / "bundle"
    save_expert_bundle(
        str(bundle_dir),
        encoder_spec={
            "expert_type": "tabular",
            "model_id": None,
            "model_path": None,
            "output_dim": 6,
            "cache_dir": "/disk1/lfhu/hf_cache",
            "use_runtime_detector": False,
        },
        semantic_expert=semantic_expert,
        fusion_policy=fusion_policy,
    )

    try:
        load_expert_bundle(str(bundle_dir), compatibility=BundleCompatibilitySpec(hidden_size=99))
    except BundleValidationError as exc:
        assert "hidden size mismatch" in str(exc)
    else:
        raise AssertionError("Expected bundle compatibility validation to fail.")


def test_expert_only_bundle_reconstructs_fresh_connector_modules(tmp_path):
    semantic_expert = _build_semantic_expert()
    fusion_policy = build_fusion_policy("post_attn_router_parallel", hidden_size=8)
    bundle_dir = tmp_path / "expert_only_bundle"
    save_expert_bundle(
        str(bundle_dir),
        encoder_spec={
            "expert_type": "tabular",
            "model_id": None,
            "model_path": None,
            "output_dim": 6,
            "cache_dir": "/disk1/lfhu/hf_cache",
            "use_runtime_detector": False,
        },
        semantic_expert=semantic_expert,
        fusion_policy=fusion_policy,
        upload_scope="expert_only",
        metadata={"model_name": "Qwen/Qwen3.5-9B", "layer_idx": 3, "feature_format": "tabular"},
    )

    loaded = load_expert_bundle(
        str(bundle_dir),
        compatibility=BundleCompatibilitySpec(
            hidden_size=8,
            evidence_dim=4,
            num_tokens=2,
            expert_output_dim=6,
            model_name="Qwen/Qwen3.5-9B",
            layer_idx=3,
            expected_feature_format="tabular",
        ),
    )
    assert loaded["metadata"]["upload_scope"] == "expert_only"
    sample = {"tabular": torch.randn(2, 6)}
    tokens = loaded["semantic_expert"].forward_with_features(sample)
    assert tokens.shape == (2, 2, 8)
    assert loaded["fusion_policy"].policy_name == "post_attn_router_parallel"
    assert all(not p.requires_grad for p in loaded["semantic_expert"].evidence_builder.expert_encoder.parameters())


def test_service_response_contract_exposes_trace_bundle_and_audit():
    runtime = BundleServiceRuntime(
        semantic_expert=_build_semantic_expert(),
        fusion_policy=build_fusion_policy("pre_attn_overwrite", hidden_size=8),
        inject_via_layer=True,
        manifest={"fusion_policy": "pre_attn_overwrite"},
        bundle_id="bundle-123",
        preprocessing_config={"normalization": "client_bundle"},
        label_schema={"classes": ["A", "B"]},
        metadata={"feature_format": "tabular"},
    )
    request = ServiceRequest(
        sanitized_input={"tabular": [[0.1] * 6]},
        task_type="strict_yesno",
        bundle_id="bundle-123",
        service_mode="freeform",
        trace_id="trace-1",
    )
    response = build_service_response(
        request,
        bundle_runtime=runtime,
        prediction="Yes",
        confidence=0.87,
        rationale="The sanitized feature pattern matches the positive class.",
        latency_ms=12.5,
        layer_idx=2,
        fusion_policy="pre_attn_overwrite",
        calibration={"ece": 0.03},
    )
    assert response.trace_id == "trace-1"
    assert response.bundle_id == "bundle-123"
    assert response.prediction == "Yes"
    assert response.confidence == 0.87
    assert response.audit["fusion_policy"] == "pre_attn_overwrite"
    assert "Rationale:" in response.output_text
