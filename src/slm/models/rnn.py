"""
RNN and LSTM Models

Implements standard RNN and LSTM architectures with the mathematical
formulations specified in the README.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseModel


class RNNModel(BaseModel):
    """
    Standard RNN model.
    
    Implements: h_t = σ(W_hh * h_{t-1} + W_xh * x_t + b_h)
    Output: Y = h_L or (1/L) * sum_t h_t
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_mode: str = "last",  # "last", "mean", or "all"
        activation: str = "tanh",  # "tanh" or "relu"
        **kwargs
    ):
        """
        Initialize RNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension (defaults to hidden_dim if None)
            num_layers: Number of RNN layers
            dropout: Dropout rate
            output_mode: How to extract final representation ("last", "mean", "all")
            activation: Activation function ("tanh" or "relu")
        """
        super().__init__(vocab_size, embed_dim, output_dim or hidden_dim, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_mode = output_mode
        
        # RNN layer
        if activation == "tanh":
            nonlinearity = "tanh"
        elif activation == "relu":
            nonlinearity = "relu"
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity=nonlinearity
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        if output_mode == "all":
            # If returning all hidden states, no projection needed
            self.projection = nn.Identity()
            self.output_dim = hidden_dim
        else:
            # Project to desired output dimension
            if self.output_dim != hidden_dim:
                self.projection = nn.Linear(hidden_dim, self.output_dim)
            else:
                self.projection = nn.Identity()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through RNN.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            RNN output based on output_mode
        """
        # Get embeddings: (batch_size, seq_len, embed_dim)
        embeddings = self.get_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        
        # RNN forward pass: (batch_size, seq_len, hidden_dim)
        rnn_out, hidden = self.rnn(embeddings)
        
        if self.output_mode == "last":
            if attention_mask is not None:
                # Get the last non-masked position for each sequence
                seq_lens = attention_mask.sum(dim=1) - 1  # 0-indexed
                batch_size = input_ids.size(0)
                last_outputs = rnn_out[torch.arange(batch_size), seq_lens]
            else:
                # Simply take the last position
                last_outputs = rnn_out[:, -1, :]
            output = self.projection(last_outputs)
            
        elif self.output_mode == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                masked_out = rnn_out * mask
                seq_lens = attention_mask.sum(dim=1, keepdim=True).float()
                mean_out = masked_out.sum(dim=1) / seq_lens
            else:
                mean_out = rnn_out.mean(dim=1)
            output = self.projection(mean_out)
            
        elif self.output_mode == "all":
            # Return all hidden states
            output = rnn_out
            
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        
        return output
    
    def get_model_config(self) -> dict:
        """Get model configuration."""
        config = super().get_model_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_mode": self.output_mode,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
        })
        return config


class LSTMModel(BaseModel):
    """
    LSTM model for long-range dependencies.
    
    Implements the full LSTM equations with forget, input, and output gates:
    f_t = σ(W_f[h_{t-1}, x_t] + b_f)
    i_t = σ(W_i[h_{t-1}, x_t] + b_i)  
    o_t = σ(W_o[h_{t-1}, x_t] + b_o)
    C̃_t = tanh(W_C[h_{t-1}, x_t] + b_C)
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_mode: str = "last",  # "last", "mean", or "all"
        bidirectional: bool = False,
        **kwargs
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension (defaults to hidden_dim if None)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_mode: How to extract final representation ("last", "mean", "all")
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__(vocab_size, embed_dim, output_dim or hidden_dim, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate actual output dimension from LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output projection
        if output_mode == "all":
            # If returning all hidden states, output dimension is the LSTM output dimension
            self.projection = nn.Identity()
            self.output_dim = lstm_output_dim
        else:
            # Project to desired output dimension
            if self.output_dim != lstm_output_dim:
                self.projection = nn.Linear(lstm_output_dim, self.output_dim)
            else:
                self.projection = nn.Identity()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            LSTM output based on output_mode
        """
        # Get embeddings: (batch_size, seq_len, embed_dim)
        embeddings = self.get_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        
        # Handle variable length sequences with packing if attention_mask is provided
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).cpu()
            # Pack the sequence
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, seq_lengths, batch_first=True, enforce_sorted=False
            )
            # LSTM forward pass
            packed_output, (hidden, cell) = self.lstm(packed_embeddings)
            # Unpack the sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            # Standard forward pass
            lstm_out, (hidden, cell) = self.lstm(embeddings)
        
        if self.output_mode == "last":
            if attention_mask is not None:
                # Get the last non-masked position for each sequence
                seq_lens = attention_mask.sum(dim=1) - 1  # 0-indexed
                batch_size = input_ids.size(0)
                last_outputs = lstm_out[torch.arange(batch_size), seq_lens]
            else:
                # Simply take the last position
                last_outputs = lstm_out[:, -1, :]
            output = self.projection(last_outputs)
            
        elif self.output_mode == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                masked_out = lstm_out * mask
                seq_lens = attention_mask.sum(dim=1, keepdim=True).float()
                mean_out = masked_out.sum(dim=1) / seq_lens
            else:
                mean_out = lstm_out.mean(dim=1)
            output = self.projection(mean_out)
            
        elif self.output_mode == "all":
            # Return all hidden states
            output = lstm_out
            
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        
        return output
    
    def get_model_config(self) -> dict:
        """Get model configuration."""
        config = super().get_model_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_mode": self.output_mode,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
        })
        return config 