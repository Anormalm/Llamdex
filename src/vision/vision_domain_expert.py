"""
Vision DomainExpert wrapper that integrates vision expert with Llamdex injection mechanism.
This follows the DomainExpert interface so it can be used with existing DomainMistralModel.
"""

import torch
import torch.nn as nn
from typing import Optional
from .vision_expert import VisionExpert
from .vision_decoder import EvidenceProjector
from .evidence_builder import (
    EvidenceBuilder,
    PopulationEvidenceBuilder,
    TextEvidenceBuilder,
    VisionEvidenceBuilder,
)


class VisionDomainExpert(nn.Module):
    """
    DomainExpert wrapper for semantic evidence injection.
    Compatible with DomainExpert interface for integration with DomainMistralModel.
    
    Args:
        embed_size: LLM hidden size
        num_tokens: Number of tokens to inject
        vision_model: Vision model name (default: 'resnet18')
        num_classes: Number of classes (default: 10 for CIFAR-10)
        vision_output: 'logits' or 'embedding' (default: 'logits')
        ffn_hidden_size: Hidden size for decoder FFN (default: 2048, -1 for Linear)
        alpha: Scaling factor for injected embeddings (default: 1.0)
        dropout: Dropout rate (default: 0.0)
        evidence_source: 'vision' or 'text'
        evidence_dim: Evidence vector dimension z
        task: 'single', 'yesno', or 'population'
        text_encoder_model: HuggingFace model id for text evidence encoder
        mistral_models_path: cache path used to load text encoder
        evidence_builder: Optional custom evidence builder
    """
    
    def __init__(
        self,
        embed_size: int,
        num_tokens: int,
        vision_model: str = 'resnet18',
        num_classes: int = 10,
        vision_output: str = 'logits',
        ffn_hidden_size: int = 2048,
        alpha: float = 1.0,
        dropout: float = 0.0,
        evidence_source: str = 'vision',
        evidence_dim: Optional[int] = None,
        task: str = 'single',
        text_encoder_model: str = "distilroberta-base",
        mistral_models_path: str = "model/llm",
        evidence_builder: Optional[EvidenceBuilder] = None,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_tokens = num_tokens
        self.vision_model_name = vision_model
        self.num_classes = num_classes
        self.vision_output = vision_output
        self.evidence_source = evidence_source
        self.task = task
        self.text_encoder_model = text_encoder_model
        self.mistral_models_path = mistral_models_path
        
        if evidence_builder is not None:
            self.evidence_builder = evidence_builder
            self.vision_expert = getattr(evidence_builder, "vision_expert", None)
        else:
            if evidence_source == "text":
                if evidence_dim is None:
                    evidence_dim = 512
                self.vision_expert = None
                self.evidence_builder = TextEvidenceBuilder(
                    evidence_dim=evidence_dim,
                    text_encoder_model=text_encoder_model,
                    cache_dir=mistral_models_path,
                )
            else:
                vision_expert = VisionExpert(
                    model_name=vision_model,
                    num_classes=num_classes,
                    output_mode=vision_output,
                    pretrained=True,
                )
                self.vision_expert = vision_expert
                if evidence_dim is None:
                    evidence_dim = vision_expert.get_output_dim()
                if task == "population":
                    self.evidence_builder = PopulationEvidenceBuilder(
                        vision_expert=vision_expert,
                        evidence_dim=evidence_dim,
                        num_classes=num_classes,
                    )
                else:
                    self.evidence_builder = VisionEvidenceBuilder(
                        vision_expert=vision_expert,
                        evidence_dim=evidence_dim,
                    )

        self.evidence_dim = evidence_dim if evidence_dim is not None else self.evidence_builder.evidence_dim
        
        # Trainable evidence projector
        self.projector = EvidenceProjector(
            input_dim=self.evidence_dim,
            hidden_size=embed_size,
            num_tokens=num_tokens,
            ffn_hidden_size=ffn_hidden_size,
            alpha=alpha,
            dropout=dropout,
        )
        # Backward compatibility alias
        self.decoder = self.projector
        
        # Store intermediate results (for compatibility with DomainExpert interface)
        self.intermediate_result = None
        self.output = None
        self.last_evidence = None
    
    def forward_with_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with raw features (images).
        This is called by DomainMistralDecoderLayer when expert_inputs is provided.
        
        Args:
            x: Image tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Token embeddings of shape (batch_size, num_tokens, embed_size)
        """
        # Build evidence vector z from selected evidence source
        z = self.evidence_builder.build(x)
        self.last_evidence = z
        self.intermediate_result = z.detach().clone()
        self.output = z.detach().clone()
        
        # Project evidence to token embeddings (this is trainable)
        token_embeddings = self.projector(z)  # (batch_size, num_tokens, embed_size)
        
        return token_embeddings
    
    def get_expert_input(self, scale=True):
        """Get intermediate result (for compatibility)."""
        return self.intermediate_result
    
    def get_expert_output(self):
        """Get expert output (for compatibility)."""
        return self.output
    
    def setup_encoder(self, requires_grad: bool):
        """Vision expert is always frozen, so this is a no-op."""
        pass
    
    def setup_decoder(self, requires_grad: bool):
        """Set evidence projector parameters to trainable or frozen."""
        for param in self.projector.parameters():
            param.requires_grad = requires_grad
    
    def clone(self):
        """Clone the expert (for multi-layer injection)."""
        # Get ffn_hidden_size from projector
        if hasattr(self.projector.ffn, 'linear1'):
            ffn_hidden_size = self.projector.ffn.linear1.out_features
        else:
            # Linear layer case
            ffn_hidden_size = -1
        
        return VisionDomainExpert(
            embed_size=self.embed_size,
            num_tokens=self.num_tokens,
            vision_model=self.vision_model_name,
            num_classes=self.num_classes,
            vision_output=self.vision_output,
            ffn_hidden_size=ffn_hidden_size,
            alpha=self.projector.alpha,
            dropout=0.0,  # Dropout not stored, use default
            evidence_source=self.evidence_source,
            evidence_dim=self.evidence_dim,
            task=self.task,
            text_encoder_model=self.text_encoder_model,
            mistral_models_path=self.mistral_models_path,
        )
