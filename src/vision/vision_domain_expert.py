"""
Vision DomainExpert wrapper that integrates vision expert with Llamdex injection mechanism.
This follows the DomainExpert interface so it can be used with existing DomainMistralModel.
"""

import torch
import torch.nn as nn
from typing import Optional
from .vision_expert import VisionExpert
from .vision_decoder import VisionToLlamdexDecoder


class VisionDomainExpert(nn.Module):
    """
    Vision DomainExpert that wraps a frozen vision expert and trainable decoder.
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
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_tokens = num_tokens
        self.vision_model_name = vision_model
        self.num_classes = num_classes
        self.vision_output = vision_output
        
        # Frozen vision expert
        self.vision_expert = VisionExpert(
            model_name=vision_model,
            num_classes=num_classes,
            output_mode=vision_output,
            pretrained=True,
        )
        
        # Trainable decoder
        expert_output_dim = self.vision_expert.get_output_dim()
        self.decoder = VisionToLlamdexDecoder(
            input_dim=expert_output_dim,
            hidden_size=embed_size,
            num_tokens=num_tokens,
            ffn_hidden_size=ffn_hidden_size,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Store intermediate results (for compatibility with DomainExpert interface)
        self.intermediate_result = None
        self.output = None
    
    def forward_with_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with raw features (images).
        This is called by DomainMistralDecoderLayer when expert_inputs is provided.
        
        Args:
            x: Image tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Token embeddings of shape (batch_size, num_tokens, embed_size)
        """
        # Store intermediate result (vision expert output)
        with torch.no_grad():
            vision_output = self.vision_expert(x)  # (batch_size, output_dim)
        
        self.intermediate_result = vision_output.clone()
        self.output = vision_output.clone()
        
        # Decode to token embeddings (this is trainable)
        token_embeddings = self.decoder(vision_output)  # (batch_size, num_tokens, embed_size)
        
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
        """Set decoder parameters to trainable or frozen."""
        for param in self.decoder.parameters():
            param.requires_grad = requires_grad
    
    def clone(self):
        """Clone the expert (for multi-layer injection)."""
        # Get ffn_hidden_size from decoder
        if hasattr(self.decoder.ffn, 'linear1'):
            ffn_hidden_size = self.decoder.ffn.linear1.out_features
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
            alpha=self.decoder.alpha,
            dropout=0.0,  # Dropout not stored, use default
        )
