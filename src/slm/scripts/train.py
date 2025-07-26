#!/usr/bin/env python3
"""
Main training script for Structured Language Modeling.

This script trains models on various datasets with different architectures.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path
import sys

from ..utils.config import get_default_config, load_config, save_config
from ..utils.model_factory import create_model
from ..utils.logging import setup_logging
from ..data.datasets import get_dataset, get_dataloader, get_task_info
from ..data.tokenizers import get_tokenizer
from ..training.trainer import Trainer
from ..training.optimizers import get_optimizer, get_scheduler


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(device_config: str) -> str:
    """Setup compute device."""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"Using device: {device}")
    return device


def main():
    parser = argparse.ArgumentParser(description="Train SLM models")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, help="Model type to train")
    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.model:
        config.model.type = args.model
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.wandb:
        config.logging.use_wandb = True
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed
    
    # Setup
    set_seed(config.seed)
    device = setup_device(config.device)
    logger = setup_logging(config.logging.log_level)
    
    # Save config
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    
    logger.info(f"Starting training with config: {config}")
    
    # Setup data
    logger.info(f"Loading dataset: {config.data.dataset_name}")
    
    # Get tokenizer
    tokenizer = get_tokenizer(
        config.data.tokenizer_name, 
        cache_dir=config.data.cache_dir
    )
    
    # Load datasets
    train_dataset = get_dataset(
        config.data.dataset_name,
        split="train",
        cache_dir=config.data.cache_dir
    )
    
    # Try to load validation dataset
    try:
        val_dataset = get_dataset(
            config.data.dataset_name,
            split="validation",
            cache_dir=config.data.cache_dir
        )
    except:
        # Some datasets don't have validation split
        logger.warning(f"No validation split found for {config.data.dataset_name}")
        val_dataset = None
    
    # Update dataset with tokenizer
    train_dataset.tokenizer = tokenizer
    if val_dataset:
        val_dataset.tokenizer = tokenizer
    
    # Create data loaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True
    )
    
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False
    ) if val_dataset else None
    
    # Get task info
    task_info = get_task_info(config.data.dataset_name)
    task_type = task_info["task_type"]
    num_classes = task_info.get("num_classes")
    
    # Update model config with task-specific info
    if num_classes:
        config.model.output_dim = num_classes
    
    # Create model
    logger.info(f"Creating model: {config.model.type}")
    model = create_model(config.model)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Setup training
    optimizer = get_optimizer(
        model,
        config.training.optimizer,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Calculate training steps for scheduler
    num_training_steps = len(train_dataloader) * config.training.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_scheduler(
        optimizer,
        config.training.scheduler,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config.training.output_dir,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_wandb=config.logging.use_wandb,
        task_type=task_type,
    )
    
    # Initialize wandb if enabled
    if config.logging.use_wandb:
        import wandb
        wandb.init(
            project=config.logging.project,
            config=dict(config),
            name=f"{config.model.type}_{config.data.dataset_name}",
        )
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from_checkpoint=args.resume,
    )
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "training_config": dict(config),
    }, final_model_path)
    
    logger.info(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main() 