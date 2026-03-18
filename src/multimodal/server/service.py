from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.multimodal.injection import build_fusion_policy
from src.multimodal.server.expert_bundle import BundleCompatibilitySpec, BundleValidationError, load_expert_bundle


@dataclass
class ServiceRequest:
    sanitized_input: Dict[str, Any]
    task_type: str
    bundle_id: Optional[str] = None
    service_mode: str = "strict"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass
class ServiceResponse:
    trace_id: str
    task_type: str
    mode: str
    prediction: str
    confidence: Optional[float]
    calibration: Dict[str, Any]
    rationale: Optional[str]
    bundle_id: str
    output_text: str
    latency_ms: float
    audit: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BundleServiceRuntime:
    semantic_expert: nn.Module
    fusion_policy: nn.Module
    inject_via_layer: bool
    manifest: Dict[str, Any]
    bundle_id: str
    preprocessing_config: Dict[str, Any]
    label_schema: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_bundle_dir(
        cls,
        bundle_dir: str,
        *,
        hidden_size: int,
        model_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
        feature_format: Optional[str] = None,
    ) -> "BundleServiceRuntime":
        compatibility = BundleCompatibilitySpec(
            hidden_size=hidden_size,
            model_name=model_name,
            layer_idx=layer_idx,
            expected_feature_format=feature_format,
        )
        loaded = load_expert_bundle(bundle_dir, compatibility=compatibility)
        manifest = loaded["manifest"].to_json_dict()
        metadata = dict(loaded["metadata"])
        bundle_id = metadata.get("bundle_id") or manifest.get("bundle_id") or uuid.uuid4().hex[:16]
        return cls(
            semantic_expert=loaded["semantic_expert"],
            fusion_policy=loaded["fusion_policy"],
            inject_via_layer=False,
            manifest=manifest,
            bundle_id=bundle_id,
            preprocessing_config=dict(loaded["preprocessing_config"]),
            label_schema=dict(loaded["label_schema"]),
            metadata=metadata,
        )


def attach_bundle_runtime(model, runtime: BundleServiceRuntime, *, layer_idx: int, fusion_policy_name: Optional[str] = None):
    policy_name = fusion_policy_name or runtime.manifest.get("fusion_policy", "pre_attn_overwrite")
    if runtime.fusion_policy is None or getattr(runtime.fusion_policy, "policy_name", None) != policy_name:
        runtime.fusion_policy = build_fusion_policy(policy_name, hidden_size=runtime.semantic_expert.projector.hidden_size)
    runtime.inject_via_layer = False
    if policy_name == "pre_attn_overwrite":
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "add_expert_"):
            layer.add_expert_(runtime.semantic_expert, map_to_expert_emb=None)
            runtime.inject_via_layer = True
        else:
            if not hasattr(model, "_external_experts"):
                model._external_experts = nn.ModuleList()
            model._external_experts.append(runtime.semantic_expert)
    return runtime


def build_service_response(
    request: ServiceRequest,
    *,
    bundle_runtime: BundleServiceRuntime,
    prediction: str,
    confidence: Optional[float],
    rationale: Optional[str],
    latency_ms: float,
    layer_idx: int,
    fusion_policy: str,
    calibration: Optional[Dict[str, Any]] = None,
) -> ServiceResponse:
    mode = request.service_mode
    if mode == "strict":
        output_text = prediction
    else:
        output_text = f"Answer: {prediction}"
        if rationale:
            output_text += f"\nRationale: {rationale}"
    audit = {
        "bundle_id": bundle_runtime.bundle_id,
        "task_type": request.task_type,
        "layer_idx": int(layer_idx),
        "fusion_policy": str(fusion_policy),
        "feature_format": bundle_runtime.metadata.get("feature_format"),
        "request_keys": sorted(request.sanitized_input.keys()),
    }
    return ServiceResponse(
        trace_id=request.trace_id,
        task_type=request.task_type,
        mode=mode,
        prediction=prediction,
        confidence=confidence,
        calibration=dict(calibration or {}),
        rationale=rationale if mode != "strict" else None,
        bundle_id=bundle_runtime.bundle_id,
        output_text=output_text,
        latency_ms=float(latency_ms),
        audit=audit,
    )


def make_audit_log(
    request: ServiceRequest,
    response: ServiceResponse,
    *,
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "trace_id": response.trace_id,
        "bundle_id": response.bundle_id,
        "task_type": response.task_type,
        "mode": response.mode,
        "latency_ms": response.latency_ms,
        "status": status,
        "error": error,
        "request_keys": sorted(request.sanitized_input.keys()),
        "audit": response.audit,
    }


def timed_service_response(*args, **kwargs) -> ServiceResponse:
    started = time.perf_counter()
    response = build_service_response(*args, **kwargs)
    response.latency_ms = (time.perf_counter() - started) * 1000.0
    return response
