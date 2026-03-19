from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from src.multimodal.experts import build_vision_expert


class ExpertEncoder(nn.Module):
    expert_type: str
    output_dim: int

    def encode(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError


def _prepare_images_01(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    x = images.float()
    if x.ndim == 5:
        b, n = x.shape[:2]
        x = x.view(b * n, *x.shape[2:])
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    # Robustly map arbitrary normalized tensors back to [0,1] per sample.
    mins = x.amin(dim=(1, 2, 3), keepdim=True)
    maxs = x.amax(dim=(1, 2, 3), keepdim=True)
    x = (x - mins) / (maxs - mins + 1e-6)
    return x.clamp(0.0, 1.0)


class CLIPSigLIPExpertEncoder(ExpertEncoder):
    """
    Frozen ViT family expert encoder.
    expert_type: "clip" | "siglip"
    """

    def __init__(
        self,
        expert_type: str = "clip",
        model_id: Optional[str] = None,
        cache_dir: str = "model/llm",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        if expert_type not in {"clip", "siglip"}:
            raise ValueError(f"Unsupported ViT expert_type={expert_type}")
        self.expert_type = expert_type
        self.model_id = model_id or (
            "openai/clip-vit-base-patch32" if expert_type == "clip" else "google/siglip-base-patch16-224"
        )

        if expert_type == "clip":
            from transformers import CLIPVisionModelWithProjection

            self.model = CLIPVisionModelWithProjection.from_pretrained(self.model_id, cache_dir=cache_dir)
            self.output_dim = int(self.model.config.projection_dim)
            self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:
            from transformers import SiglipVisionModel

            self.model = SiglipVisionModel.from_pretrained(self.model_id, cache_dir=cache_dir)
            hidden = int(self.model.config.hidden_size)
            self.output_dim = int(hidden)
            self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, batch: Any) -> torch.Tensor:
        images = batch["images"] if isinstance(batch, dict) and "images" in batch else batch
        x = _prepare_images_01(images)
        x = (x.to(self.mean.device) - self.mean) / self.std
        with torch.no_grad():
            if self.expert_type == "clip":
                out = self.model(pixel_values=x)
                emb = out.image_embeds
            else:
                out = self.model(pixel_values=x)
                emb = out.pooler_output
        if images.ndim == 5:
            b, n = images.shape[:2]
            emb = emb.view(b, n, -1).mean(dim=1)
        return emb.float()


class DINOv2ExpertEncoder(ExpertEncoder):
    """
    Frozen DINOv2 vision encoder.
    Uses CLS-style pooled output when available, otherwise falls back to the first token.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: str = "model/llm",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.expert_type = "dinov2"
        self.model_id = model_id or "facebook/dinov2-base"

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.model_id, cache_dir=cache_dir)
        hidden = int(getattr(self.model.config, "hidden_size", output_dim or 768))
        self.output_dim = int(output_dim or hidden)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.adapter = nn.Identity() if self.output_dim == hidden else nn.Linear(hidden, self.output_dim, bias=False)

        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, batch: Any) -> torch.Tensor:
        images = batch["images"] if isinstance(batch, dict) and "images" in batch else batch
        x = _prepare_images_01(images)
        x = (x.to(self.mean.device) - self.mean) / self.std
        with torch.no_grad():
            out = self.model(pixel_values=x)
            emb = getattr(out, "pooler_output", None)
            if emb is None:
                emb = out.last_hidden_state[:, 0, :]
        emb = self.adapter(emb.float())
        if images.ndim == 5:
            b, n = images.shape[:2]
            emb = emb.view(b, n, -1).mean(dim=1)
        return emb.float()


class ResNet18ClassifierCheckpointExpertEncoder(ExpertEncoder):
    def __init__(
        self,
        model_path: str,
        output_dim: int,
        init_weights: str = "imagenet",
    ):
        super().__init__()
        if not model_path:
            raise ValueError("resnet18_classifier expert requires model_path checkpoint.")
        self.expert_type = "resnet18_classifier"
        self.output_dim = int(output_dim)
        self.model = build_vision_expert(
            expert_kind="classifier",
            checkpoint_path=model_path,
            num_classes=self.output_dim,
            init_weights=init_weights,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, batch: Any) -> torch.Tensor:
        images = batch["images"] if isinstance(batch, dict) and "images" in batch else batch
        x = _prepare_images_01(images)
        param = next(self.model.parameters())
        x = (x.to(device=self.mean.device, dtype=param.dtype) - self.mean.to(dtype=param.dtype)) / self.std.to(dtype=param.dtype)
        with torch.no_grad():
            out = self.model(x)
            emb = out.logits
        if images.ndim == 5:
            b, n = images.shape[:2]
            emb = emb.view(b, n, -1).mean(dim=1)
        return emb.float()


class GroundingDinoSAM2ExpertEncoder(ExpertEncoder):
    """
    Structured semantic encoder from detection+segmentation outputs.
    Supports:
    1) Precomputed structured inputs in batch (recommended for fast evaluation).
    2) Optional GroundingDINO runtime if dependencies are installed.
    """

    def __init__(
        self,
        output_dim: int = 32,
        use_runtime_detector: bool = False,
        detector_model_id: str = "IDEA-Research/grounding-dino-base",
        cache_dir: str = "model/llm",
    ):
        super().__init__()
        self.expert_type = "groundingdino_sam2"
        self.output_dim = int(output_dim)
        self.use_runtime_detector = bool(use_runtime_detector)
        self.detector = None
        self.processor = None

        if self.use_runtime_detector:
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(detector_model_id, cache_dir=cache_dir)
                self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model_id, cache_dir=cache_dir)
                for p in self.detector.parameters():
                    p.requires_grad = False
            except Exception as exc:
                raise RuntimeError(
                    "GroundingDINO runtime path requested but dependencies/model loading failed. "
                    "Either install dependencies or pass precomputed structured detection+segmentation features."
                ) from exc

        self.stats_head = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, self.output_dim),
        )

    def _summarize_precomputed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = batch.get("det_scores")
        labels = batch.get("det_labels")
        mask_areas = batch.get("mask_areas")
        if scores is None or labels is None or mask_areas is None:
            raise ValueError(
                "For groundingdino_sam2 precomputed path, batch must include det_scores, det_labels, mask_areas."
            )
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
            labels = labels.unsqueeze(0)
            mask_areas = mask_areas.unsqueeze(0)
        one = torch.ones_like(scores)
        valid = (scores >= 0).float()
        n_obj = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_score = (scores * valid).sum(dim=1, keepdim=True) / n_obj
        max_score = (scores * valid + (1.0 - valid) * (-1e9)).max(dim=1, keepdim=True).values
        mean_area = (mask_areas * valid).sum(dim=1, keepdim=True) / n_obj
        std_area = torch.sqrt((((mask_areas - mean_area) ** 2) * valid).sum(dim=1, keepdim=True) / n_obj)
        label_hist = []
        for c in range(8):
            label_hist.append((((labels == c).float()) * valid).sum(dim=1, keepdim=True) / n_obj)
        feats = torch.cat([n_obj / (scores.shape[1] + 1e-6), mean_score, max_score, mean_area, std_area] + label_hist, dim=1)
        pad = torch.zeros(feats.size(0), 16 - feats.size(1), device=feats.device, dtype=feats.dtype)
        return torch.cat([feats, pad], dim=1)

    def _runtime_detect(self, images: torch.Tensor) -> torch.Tensor:
        # Runtime mode expects slow detector path and returns coarse structured stats.
        if self.processor is None or self.detector is None:
            raise RuntimeError("Runtime detector is not initialized.")
        x = _prepare_images_01(images, size=800)
        pil = [TVF.to_pil_image(img.cpu()) for img in x]
        inputs = self.processor(images=pil, text=["object ."] * len(pil), return_tensors="pt")
        inputs = {k: v.to(next(self.detector.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.detector(**inputs)
        logits = outputs.logits.sigmoid().amax(dim=-1)
        # Coarse fallback stats if SAM2 masks are not available in runtime path.
        n = torch.full((logits.size(0), 1), float(logits.size(1)), device=logits.device)
        mean_score = logits.mean(dim=1, keepdim=True)
        max_score = logits.max(dim=1, keepdim=True).values
        zeros = torch.zeros_like(mean_score)
        feats = torch.cat([n / max(1.0, float(logits.size(1))), mean_score, max_score, zeros, zeros], dim=1)
        pad = torch.zeros(feats.size(0), 11, device=feats.device, dtype=feats.dtype)
        return torch.cat([feats, pad], dim=1)

    def _image_structured_fallback(self, images: torch.Tensor) -> torch.Tensor:
        x = _prepare_images_01(images, size=224)
        if images.ndim == 5:
            b, n = images.shape[:2]
            x = x.view(b, n, *x.shape[1:]).mean(dim=1)
        mean_rgb = x.mean(dim=(2, 3))
        std_rgb = x.std(dim=(2, 3))
        edge_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(2, 3))
        edge_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(2, 3))
        energy = (x * x).mean(dim=(2, 3))
        feats = torch.cat([mean_rgb, std_rgb, edge_h, edge_w, energy], dim=1)  # (B, 15)
        pad = torch.zeros(feats.size(0), 1, device=feats.device, dtype=feats.dtype)
        return torch.cat([feats, pad], dim=1)

    def encode(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict) and "semantic_vector" in batch:
            v = batch["semantic_vector"].float()
            if v.ndim == 1:
                v = v.unsqueeze(0)
            if v.shape[1] != self.output_dim:
                head = nn.Linear(v.shape[1], self.output_dim, bias=False).to(v.device, v.dtype)
                with torch.no_grad():
                    v = head(v)
            return v

        if isinstance(batch, dict) and {"det_scores", "det_labels", "mask_areas"} <= set(batch.keys()):
            feats = self._summarize_precomputed(batch)
            return self.stats_head(feats)

        if self.use_runtime_detector:
            images = batch["images"] if isinstance(batch, dict) else batch
            feats = self._runtime_detect(images)
            return self.stats_head(feats)

        images = batch["images"] if isinstance(batch, dict) and "images" in batch else batch
        if torch.is_tensor(images):
            feats = self._image_structured_fallback(images)
            return self.stats_head(feats)
        raise ValueError("groundingdino_sam2 encoder input is invalid. Provide image tensor or structured tensors.")


class TabularExpertEncoder(ExpertEncoder):
    """
    Frozen tabular experts:
      - xgboost (predict_proba)
      - ft_transformer / tabpfn (torch module returning logits/probs)
    """

    def __init__(
        self,
        model_kind: str = "xgboost",
        model_path: Optional[str] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.expert_type = "tabular"
        self.model_kind = model_kind
        self.model = None
        if model_path is not None:
            if model_kind == "xgboost":
                import joblib

                self.model = joblib.load(model_path)
            else:
                self.model = torch.load(model_path, map_location="cpu")
                if hasattr(self.model, "eval"):
                    self.model.eval()
                    for p in self.model.parameters():
                        p.requires_grad = False
        self.output_dim = int(output_dim or 32)

    def encode(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            x = batch.get("tabular", None)
            if x is None:
                raise ValueError("Tabular encoder expects batch['tabular'].")
        else:
            x = batch
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if self.model is None:
            # Fallback: identity projection from raw features.
            return x[:, : self.output_dim] if x.shape[1] >= self.output_dim else F.pad(x, (0, self.output_dim - x.shape[1]))

        if self.model_kind == "xgboost":
            prob = self.model.predict_proba(x.cpu().numpy())
            y = torch.tensor(prob, dtype=torch.float32, device=x.device)
        else:
            with torch.no_grad():
                y = self.model(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y.ndim == 1:
                    y = y.unsqueeze(0)
        if y.shape[1] == self.output_dim:
            return y.float()
        if y.shape[1] > self.output_dim:
            return y[:, : self.output_dim].float()
        return F.pad(y.float(), (0, self.output_dim - y.shape[1]))


@dataclass
class ExpertEncoderSpec:
    expert_type: str
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    output_dim: int = 512
    cache_dir: str = "model/llm"
    use_runtime_detector: bool = False


def build_expert_encoder(spec: ExpertEncoderSpec) -> ExpertEncoder:
    t = spec.expert_type.lower()
    if t in {"resnet18_classifier", "classifier_checkpoint", "vision_classifier_checkpoint"}:
        return ResNet18ClassifierCheckpointExpertEncoder(
            model_path=spec.model_path,
            output_dim=spec.output_dim,
        )
    if t in {"clip", "siglip"}:
        return CLIPSigLIPExpertEncoder(
            expert_type=t,
            model_id=spec.model_id,
            cache_dir=spec.cache_dir,
            output_dim=spec.output_dim,
        )
    if t in {"dinov2", "dino", "dinov2_vit"}:
        return DINOv2ExpertEncoder(
            model_id=spec.model_id,
            cache_dir=spec.cache_dir,
            output_dim=spec.output_dim,
        )
    if t in {"groundingdino_sam2", "groundingdino+sam2", "groundingdino_sam"}:
        return GroundingDinoSAM2ExpertEncoder(
            output_dim=spec.output_dim,
            use_runtime_detector=spec.use_runtime_detector,
            cache_dir=spec.cache_dir,
        )
    if t in {"xgboost", "ft_transformer", "tabpfn", "tabular"}:
        return TabularExpertEncoder(model_kind=t if t != "tabular" else "xgboost", model_path=spec.model_path, output_dim=spec.output_dim)
    raise ValueError(f"Unsupported expert_type: {spec.expert_type}")
