"""
Convolutional Model for N-gram Detection

Implements 1-D convolution with multiple filters to capture local morphology
and word-order patterns: Y_i = σ(sum_{j=0}^{k-1} W_j · X_{i·s+j} + b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .base import BaseModel


class ConvolutionalModel(BaseModel):
    """
    Convolutional model for n-gram pattern detection.
    
    Uses multiple 1D convolutional filters with different kernel sizes
    to capture local patterns at various scales.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
        kernel_sizes: List[int] = [2, 3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.1,
        pooling: str = "max",  # "max", "mean", or "adaptive"
        **kwargs
    ):
        """
        Initialize convolutional model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            output_dim: Output dimension
            kernel_sizes: List of kernel sizes for different n-gram patterns
            num_filters: Number of filters per kernel size
            dropout: Dropout rate
            pooling: Pooling strategy ("max", "mean", or "adaptive")
        """
        super().__init__(vocab_size, embed_dim, output_dim, **kwargs)
        
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.pooling = pooling
        
        # Create convolutional layers for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0  # No padding, let sequence length vary
            )
            for k in kernel_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total feature dimension
        total_features = len(kernel_sizes) * num_filters
        
        # Output projection
        if self.output_dim != total_features:
            self.projection = nn.Linear(total_features, self.output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through convolutional layers.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Convolved and pooled representation of shape (batch_size, output_dim)
        """
        # Get embeddings: (batch_size, seq_len, embed_dim)
        embeddings = self.get_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        
        # Transpose for 1D convolution: (batch_size, embed_dim, seq_len)
        x = embeddings.transpose(1, 2)
        
        conv_outputs = []
        
        for conv in self.convs:
            # Apply convolution: (batch_size, num_filters, conv_seq_len)
            conv_out = F.relu(conv(x))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Create mask for convolved sequence
                kernel_size = conv.kernel_size[0]
                conv_mask = self._create_conv_mask(attention_mask, kernel_size)
                conv_out = conv_out * conv_mask.unsqueeze(1)
            
            # Pool across sequence dimension
            if self.pooling == "max":
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            elif self.pooling == "mean":
                if attention_mask is not None:
                    # Masked mean pooling
                    conv_mask = self._create_conv_mask(attention_mask, conv.kernel_size[0])
                    valid_length = conv_mask.sum(dim=1, keepdim=True).float()
                    pooled = (conv_out * conv_mask.unsqueeze(1)).sum(dim=2, keepdim=True) / (valid_length.unsqueeze(1) + 1e-8)
                else:
                    pooled = F.avg_pool1d(conv_out, kernel_size=conv_out.size(2))
            elif self.pooling == "adaptive":
                pooled = F.adaptive_max_pool1d(conv_out, 1)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
            
            # Remove the sequence dimension: (batch_size, num_filters)
            pooled = pooled.squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs: (batch_size, total_features)
        features = torch.cat(conv_outputs, dim=1)
        
        # Apply projection
        output = self.projection(features)
        
        return output
    
    def _create_conv_mask(self, attention_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Create attention mask for convolved sequence.
        
        Args:
            attention_mask: Original attention mask (batch_size, seq_len)
            kernel_size: Convolution kernel size
            
        Returns:
            Mask for convolved sequence (batch_size, conv_seq_len)
        """
        batch_size, seq_len = attention_mask.shape
        conv_seq_len = seq_len - kernel_size + 1
        
        if conv_seq_len <= 0:
            return torch.zeros(batch_size, 1, device=attention_mask.device)
        
        # A convolved position is valid if all positions in the kernel are valid
        conv_mask = torch.zeros(batch_size, conv_seq_len, device=attention_mask.device)
        
        for i in range(conv_seq_len):
            # Check if all positions in the kernel window are valid
            kernel_window = attention_mask[:, i:i+kernel_size]
            conv_mask[:, i] = (kernel_window.sum(dim=1) == kernel_size).float()
        
        return conv_mask
    
    def get_model_config(self) -> dict:
        """Get model configuration."""
        config = super().get_model_config()
        config.update({
            "kernel_sizes": self.kernel_sizes,
            "num_filters": self.num_filters,
            "pooling": self.pooling,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
        })
        return config 