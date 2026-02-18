"""
Trainable projector that maps evidence vectors to token embeddings for injection.
"""

import torch
import torch.nn as nn
from ..model.util import SwiGLU


class EvidenceProjector(nn.Module):
    """
    Projector that maps generic evidence vectors z to token embeddings.
    
    Architecture:
    - FFN (SwiGLU or Linear) -> (num_tokens * hidden_size) -> reshape
    - LayerNorm over hidden_size
    - Optional alpha scaling before LayerNorm
    
    Args:
        input_dim: Input dimension (expert output size)
        hidden_size: LLM hidden size (embed_size)
        num_tokens: Number of tokens to inject
        ffn_hidden_size: Hidden size for FFN (if -1, use Linear instead of SwiGLU)
        alpha: Scaling factor for injected embeddings (default: 1.0)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_tokens: int,
        ffn_hidden_size: int = 2048,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.alpha = alpha
        
        # FFN: input_dim -> ffn_hidden_size -> (num_tokens * hidden_size)
        if ffn_hidden_size != -1:
            self.ffn = SwiGLU(input_dim, ffn_hidden_size, num_tokens * hidden_size, dropout=dropout)
        else:
            self.ffn = nn.Linear(input_dim, num_tokens * hidden_size, bias=True)
        
        # LayerNorm over hidden_size dimension
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map evidence vector z to token embeddings.
        
        Args:
            x: Evidence tensor z of shape (batch_size, input_dim)
            
        Returns:
            Token embeddings of shape (batch_size, num_tokens, hidden_size)
        """
        # FFN: (batch_size, input_dim) -> (batch_size, num_tokens * hidden_size)
        x = self.ffn(x)
        
        # Reshape: (batch_size, num_tokens * hidden_size) -> (batch_size, num_tokens, hidden_size)
        x = x.view(x.size(0), self.num_tokens, self.hidden_size)
        
        # Apply alpha scaling before LayerNorm
        x = self.alpha * x
        
        # LayerNorm over hidden_size dimension
        # LayerNorm expects (..., hidden_size), so we apply it to the last dimension
        x = self.layernorm(x)
        
        return x


# Backward compatibility alias for existing imports/usages.
VisionToLlamdexDecoder = EvidenceProjector
