"""
Model factory for creating models from configuration.
"""

from typing import Dict, Any, Type
from omegaconf import DictConfig

from ..models.base import BaseModel
from ..models.baseline import MeanPoolingModel
from ..models.conv import ConvolutionalModel
from ..models.rnn import RNNModel, LSTMModel
from ..models.attention import MultiHeadAttentionModel
from ..models.regex import RegexModel, DifferentiableRegexModel
from ..models.cfg import CFGModel, PCFGModel


# Registry of available models
MODEL_REGISTRY = {
    "MeanPoolingModel": MeanPoolingModel,
    "ConvolutionalModel": ConvolutionalModel,
    "RNNModel": RNNModel,
    "LSTMModel": LSTMModel,
    "MultiHeadAttentionModel": MultiHeadAttentionModel,
    "RegexModel": RegexModel,
    "DifferentiableRegexModel": DifferentiableRegexModel,
    "CFGModel": CFGModel,
    "PCFGModel": PCFGModel,
}


def get_model_class(model_type: str) -> Type[BaseModel]:
    """
    Get model class by name.
    
    Args:
        model_type: Name of the model type
        
    Returns:
        Model class
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type]


def create_model(model_config: DictConfig, **kwargs) -> BaseModel:
    """
    Create model from configuration.
    
    Args:
        model_config: Model configuration
        **kwargs: Additional arguments to override config
        
    Returns:
        Instantiated model
    """
    model_type = model_config.get("type", "MeanPoolingModel")
    model_class = get_model_class(model_type)
    
    # Extract model parameters
    model_params = dict(model_config)
    model_params.pop("type", None)  # Remove type from params
    
    # Override with any additional kwargs
    model_params.update(kwargs)
    
    # Create model
    model = model_class(**model_params)
    
    return model


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: DictConfig = None,
    **kwargs
) -> BaseModel:
    """
    Create model and load from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration (if None, will use config from checkpoint)
        **kwargs: Additional arguments
        
    Returns:
        Model loaded from checkpoint
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Use model config from checkpoint if not provided
    if model_config is None:
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            raise ValueError("No model config found in checkpoint and none provided")
    
    # Create model
    model = create_model(model_config, **kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def list_available_models() -> Dict[str, Type[BaseModel]]:
    """List all available models."""
    return MODEL_REGISTRY.copy()


def register_model(name: str, model_class: Type[BaseModel]):
    """
    Register a new model type.
    
    Args:
        name: Name for the model
        model_class: Model class
    """
    if not issubclass(model_class, BaseModel):
        raise ValueError("Model class must inherit from BaseModel")
    
    MODEL_REGISTRY[name] = model_class 