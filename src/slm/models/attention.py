"""
Multi-Head Attention Model

Implements standard Transformer multi-head attention:
MultiHead(X) = [head_1; ...; head_h] W^O
head_i = softmax(QK^T / sqrt(d_k)) V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .base import BaseModel


class MultiHeadAttentionModel(BaseModel):
    """
    Multi-head attention model using standard Transformer attention.
    
    Implements the attention mechanism with multiple heads for capturing
    different types of relationships in the sequence.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_mode: str = "mean",  # "mean", "cls", or "all"
        use_positional_encoding: bool = True,
        **kwargs
    ):
        """
        Initialize multi-head attention model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension (must be divisible by num_heads)
            output_dim: Output dimension (defaults to embed_dim if None)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            output_mode: How to extract final representation
            use_positional_encoding: Whether to add positional encoding
        """
        super().__init__(vocab_size, embed_dim, output_dim, **kwargs)
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.output_mode = output_mode
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # CLS token for classification tasks (if using "cls" output mode)
        if output_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Output projection
        if output_mode == "all":
            self.projection = nn.Identity()
            self.output_dim = embed_dim
        else:
            if self.output_dim != embed_dim:
                self.projection = nn.Linear(embed_dim, self.output_dim)
            else:
                self.projection = nn.Identity()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention layers.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Attention output based on output_mode
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings: (batch_size, seq_len, embed_dim)
        x = self.get_embeddings(input_ids)
        
        # Add CLS token if using "cls" output mode
        if self.output_mode == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # Update attention mask for CLS token
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        # Extract final representation based on output_mode
        if self.output_mode == "cls":
            # Take the CLS token representation
            output = self.projection(x[:, 0, :])
        elif self.output_mode == "mean":
            if attention_mask is not None:
                # Masked mean pooling (excluding CLS token if present)
                start_idx = 1 if self.output_mode == "cls" else 0
                mask = attention_mask[:, start_idx:].unsqueeze(-1).float()
                masked_x = x[:, start_idx:, :] * mask
                seq_lens = attention_mask[:, start_idx:].sum(dim=1, keepdim=True).float()
                mean_x = masked_x.sum(dim=1) / (seq_lens + 1e-8)
            else:
                mean_x = x.mean(dim=1)
            output = self.projection(mean_x)
        elif self.output_mode == "all":
            # Return all representations
            output = x
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        
        return output
    
    def get_model_config(self) -> dict:
        """Get model configuration."""
        config = super().get_model_config()
        config.update({
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "output_mode": self.output_mode,
            "use_positional_encoding": self.use_positional_encoding,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
        })
        return config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Single transformer layer with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(
            x, x, x, 
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff_network(x)
        x = self.norm2(x + ff_output)
        
        return x 