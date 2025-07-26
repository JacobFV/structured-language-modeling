"""
Loss functions for different task types.
"""

import torch
import torch.nn as nn
from typing import Callable


def get_loss_function(task_type: str) -> Callable:
    """
    Get loss function for a specific task type.
    
    Args:
        task_type: Type of task ("classification", "language_modeling", etc.)
    
    Returns:
        Loss function
    """
    if task_type == "classification":
        return nn.CrossEntropyLoss()
    elif task_type == "language_modeling":
        return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    elif task_type == "regression":
        return nn.MSELoss()
    elif task_type == "sequence_labeling":
        return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    else:
        # Default to cross-entropy
        return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean() 