"""
Frozen vision expert wrapper for image classification.
Supports ResNet-18 (and optionally other torchvision models) with logits or embedding output modes.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal


class VisionExpert(nn.Module):
    """
    Frozen vision expert that processes images and outputs either logits or embeddings.
    
    Args:
        model_name: Name of the vision model (default: 'resnet18')
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        output_mode: 'logits' or 'embedding' (default: 'logits')
        pretrained: Whether to use pretrained weights (default: True)
    """
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 10,
        output_mode: Literal['logits', 'embedding'] = 'logits',
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.output_mode = output_mode
        
        # Load pretrained model
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            # Modify first conv layer for CIFAR-10 (32x32 images)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
            # Replace final FC layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only 'resnet18' is supported for now.")
        
        # If embedding mode, we'll extract features before the final FC layer
        if output_mode == 'embedding':
            # Store the FC layer separately for potential use
            if hasattr(model, 'fc'):
                self.embedding_dim = model.fc.in_features
                self.classifier = model.fc
                # Remove FC from backbone
                model.fc = nn.Identity()
            else:
                self.embedding_dim = 512
                self.classifier = None
        else:
            self.embedding_dim = num_classes
            self.classifier = None
        
        self.backbone = model
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision expert.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, H, W)
            
        Returns:
            If output_mode == 'logits': (batch_size, num_classes)
            If output_mode == 'embedding': (batch_size, embedding_dim)
        """
        if self.output_mode == 'embedding':
            # Extract features before final FC
            features = self.backbone(x)  # (batch_size, embedding_dim)
            return features
        else:
            # Return logits
            logits = self.backbone(x)  # (batch_size, num_classes)
            return logits
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the expert."""
        if self.output_mode == 'logits':
            return self.num_classes
        else:
            return self.embedding_dim
