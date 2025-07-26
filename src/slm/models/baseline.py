"""
Mean Pooling Baseline Model

Implements the simplest baseline: Y = (1/L) * sum(X_t) for t=1 to L
This serves as a sanity-check comparator for more complex architectures.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseModel


class MeanPoolingModel(BaseModel):
    """
    Mean pooling baseline model.
    
    Simply averages all token embeddings in the sequence:
    Y = (1/L) * sum_{t=1}^L X_t
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize mean pooling model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            output_dim: Output dimension (defaults to embed_dim if None)
            dropout: Dropout rate
        """
        super().__init__(vocab_size, embed_dim, output_dim, **kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
        # Optional projection layer if output_dim differs from embed_dim
        if self.output_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass: mean pool over sequence length.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Pooled representation of shape (batch_size, output_dim)
        """
        # Get embeddings: (batch_size, seq_len, embed_dim)
        embeddings = self.get_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        
        if attention_mask is not None:
            # Apply attention mask to ignore padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            embeddings = embeddings * mask
            
            # Compute mean only over non-masked positions
            seq_lens = attention_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            pooled = embeddings.sum(dim=1) / seq_lens  # (batch_size, embed_dim)
        else:
            # Simple mean over all positions
            pooled = embeddings.mean(dim=1)  # (batch_size, embed_dim)
        
        # Apply projection if needed
        output = self.projection(pooled)  # (batch_size, output_dim)
        
        return output
    
    def get_model_config(self) -> dict:
        """Get model configuration."""
        config = super().get_model_config()
        config.update({
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
        })
        return config 