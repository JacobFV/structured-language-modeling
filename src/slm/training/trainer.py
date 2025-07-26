"""
Main trainer class for SLM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import wandb
from tqdm import tqdm
import os
from pathlib import Path

from ..models.base import BaseModel
from ..evaluation.metrics import compute_metrics
from .losses import get_loss_function
from .optimizers import get_optimizer, get_scheduler


class Trainer:
    """
    Main trainer class for SLM models.
    
    Handles training, validation, and checkpointing for all model types.
    """
    
    def __init__(
        self,
        model: BaseModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        task_type: str = "classification",
        **kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device to use
            output_dir: Output directory for checkpoints
            save_steps: Steps between checkpoint saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            max_grad_norm: Maximum gradient norm for clipping
            use_wandb: Whether to use wandb logging
            task_type: Type of task ("classification", "language_modeling", etc.)
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.task_type = task_type
        
        # Setup optimizer, scheduler, and loss
        self.optimizer = optimizer or get_optimizer(model, "adamw", lr=5e-4)
        self.scheduler = scheduler
        self.loss_fn = loss_fn or get_loss_function(task_type)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("-inf") if task_type in ["classification"] else float("inf")
        
        # Setup logging
        if use_wandb and wandb.run is None:
            wandb.init(project="slm-experiments")
            wandb.watch(model)
    
    def train(
        self,
        num_epochs: int = 3,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.model.train()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.training_step(batch)
                epoch_loss += loss
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self.log_metrics({
                        "train/loss": loss,
                        "train/epoch": epoch,
                        "train/learning_rate": self.get_lr(),
                    })
                
                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    self.evaluate()
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            
            # End of epoch evaluation
            if self.val_dataloader:
                self.evaluate()
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            self.log_metrics({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch,
            })
        
        # Final evaluation on test set
        if self.test_dataloader:
            test_metrics = self.evaluate(self.test_dataloader, "test")
            print(f"Test results: {test_metrics}")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        
        # Compute loss
        if self.task_type == "language_modeling":
            # For language modeling, shift labels
            labels = batch["input_ids"][:, 1:].contiguous()
            outputs = outputs[:, :-1, :].contiguous()
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            # For classification and other tasks
            labels = batch["labels"]
            loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()
    
    def evaluate(
        self, 
        dataloader: Optional[DataLoader] = None, 
        prefix: str = "eval"
    ) -> Dict[str, float]:
        """Evaluate the model."""
        if dataloader is None:
            dataloader = self.val_dataloader
        
        if dataloader is None:
            return {}
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating ({prefix})"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                
                # Compute loss
                if self.task_type == "language_modeling":
                    labels = batch["input_ids"][:, 1:].contiguous()
                    outputs_for_loss = outputs[:, :-1, :].contiguous()
                    loss = self.loss_fn(
                        outputs_for_loss.view(-1, outputs_for_loss.size(-1)), 
                        labels.view(-1)
                    )
                else:
                    labels = batch["labels"]
                    loss = self.loss_fn(outputs, labels)
                    
                    # Collect predictions and labels
                    if outputs.dim() > 1:
                        predictions = torch.argmax(outputs, dim=-1)
                    else:
                        predictions = outputs
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = {f"{prefix}/loss": avg_loss}
        
        if self.task_type != "language_modeling":
            task_metrics = compute_metrics(
                predictions=all_predictions,
                labels=all_labels,
                task_type=self.task_type
            )
            metrics.update({f"{prefix}/{k}": v for k, v in task_metrics.items()})
        else:
            # For language modeling, compute perplexity
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            metrics[f"{prefix}/perplexity"] = perplexity
        
        self.log_metrics(metrics)
        
        # Update best metric
        primary_metric = metrics.get(f"{prefix}/accuracy", metrics.get(f"{prefix}/perplexity", avg_loss))
        if self.task_type == "language_modeling":
            is_better = primary_metric < self.best_metric
        else:
            is_better = primary_metric > self.best_metric
        
        if is_better:
            self.best_metric = primary_metric
            self.save_checkpoint(is_best=True)
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "model_config": self.model.get_model_config(),
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb and console."""
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Console logging
        if self.global_step % self.logging_steps == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Step {self.global_step}: {metrics_str}")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"] 