"""
Optimizer and scheduler utilities.
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, OneCycleLR
)
from typing import Optional
import math


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer by name.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    num_training_steps: Optional[int] = None,
    num_warmup_steps: Optional[int] = None,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler by name.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None
    
    elif scheduler_name.lower() == "step":
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.9)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name.lower() == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name.lower() == "cosine":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for cosine scheduler")
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    elif scheduler_name.lower() == "plateau":
        patience = kwargs.get("patience", 10)
        factor = kwargs.get("factor", 0.5)
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    
    elif scheduler_name.lower() == "onecycle":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for onecycle scheduler")
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]["lr"])
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_training_steps)
    
    elif scheduler_name.lower() == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer, 
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps or 0
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ] 