"""
Base model interface for Structured Language Modeling.

Defines the common interface that all model architectures must implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all SLM models.
    
    All models must implement forward pass and provide model-specific
    configuration and output processing methods.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize base model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            output_dim: Output dimension (defaults to embed_dim if None)
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim
        
        # Common embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Model-specific parameters will be set by subclasses
        self.model_type = self.__class__.__name__
        
    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model output tensor
        """
        pass
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for input."""
        return self.embedding(input_ids)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "output_dim": self.output_dim,
        }
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_info(self) -> Dict[str, int]:
        """Get detailed parameter information."""
        param_info = {}
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                param_info[name] = param_count
                total += param_count
        param_info["total"] = total
        return param_info 