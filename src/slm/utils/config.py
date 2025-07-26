"""
Configuration utilities for SLM.
"""

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, save_path)


def get_default_config() -> DictConfig:
    """Get default configuration for SLM experiments."""
    default_config = {
        "model": {
            "type": "MeanPoolingModel",
            "vocab_size": 30522,  # BERT vocab size
            "embed_dim": 256,
            "output_dim": 128,
            "dropout": 0.1,
        },
        "data": {
            "dataset_name": "sst2",
            "tokenizer_name": "bert-base-uncased",
            "max_length": 128,
            "batch_size": 32,
            "cache_dir": "./cache",
        },
        "training": {
            "num_epochs": 3,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "warmup_cosine",
            "max_grad_norm": 1.0,
            "eval_steps": 500,
            "save_steps": 1000,
            "logging_steps": 100,
            "output_dir": "./outputs",
        },
        "evaluation": {
            "datasets": ["sst2", "imdb", "ag_news", "cola"],
            "batch_size": 32,
            "max_samples": null,
        },
        "logging": {
            "use_wandb": false,
            "project": "slm-experiments",
            "log_level": "INFO",
        },
        "device": "auto",  # auto, cpu, cuda
        "seed": 42,
    }
    
    return OmegaConf.create(default_config)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config) 