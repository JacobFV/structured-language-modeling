#!/usr/bin/env python3
"""
Model comparison script for Structured Language Modeling.

This script compares multiple model architectures on the evaluation suite.
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from slm.utils.config import get_default_config
from slm.utils.model_factory import create_model, list_available_models
from slm.utils.logging import setup_logging
from slm.evaluation.evaluator import Evaluator
from slm.data.datasets import AVAILABLE_DATASETS


def create_models_for_comparison(
    model_types: list,
    vocab_size: int = 30522,
    embed_dim: int = 256,
    output_dim: int = 128,
) -> dict:
    """Create models for comparison."""
    models = {}
    
    config = get_default_config()
    
    for model_type in model_types:
        print(f"Creating {model_type}...")
        
        # Model-specific configurations
        model_config = config.model.copy()
        model_config.type = model_type
        model_config.vocab_size = vocab_size
        model_config.embed_dim = embed_dim
        model_config.output_dim = output_dim
        
        # Adjust config for specific models
        if model_type in ["RNNModel", "LSTMModel"]:
            model_config.hidden_dim = embed_dim
        elif model_type == "ConvolutionalModel":
            model_config.num_filters = 100
            model_config.kernel_sizes = [2, 3, 4, 5]
        elif model_type == "MultiHeadAttentionModel":
            model_config.num_heads = 8
            model_config.num_layers = 2  # Smaller for fair comparison
            model_config.ff_dim = embed_dim * 2
        
        try:
            model = create_model(model_config)
            models[model_type] = model
            print(f"✓ {model_type}: {model.count_parameters():,} parameters")
        except Exception as e:
            print(f"✗ Failed to create {model_type}: {e}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Compare SLM models")
    parser.add_argument("--models", nargs="+", 
                       help="Model types to compare")
    parser.add_argument("--datasets", nargs="+", 
                       help="Datasets to evaluate on")
    parser.add_argument("--output-dir", type=str, default="./comparison_results",
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size (smaller for memory)")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Max samples per dataset for quick comparison")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--embed-dim", type=int, default=128, 
                       help="Embedding dimension")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        available_models = list_available_models()
        print("Available models:")
        for name, model_class in available_models.items():
            print(f"  {name}: {model_class.__doc__ or 'No description'}")
        return
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Determine models to compare
    if args.models:
        model_types = args.models
    else:
        # Default comparison: all simple models
        model_types = [
            "MeanPoolingModel",
            "ConvolutionalModel", 
            "RNNModel",
            "LSTMModel",
            "MultiHeadAttentionModel",
        ]
    
    # Validate model types
    available_models = list_available_models()
    invalid_models = [m for m in model_types if m not in available_models]
    if invalid_models:
        logger.warning(f"Invalid models (will be skipped): {invalid_models}")
        model_types = [m for m in model_types if m in available_models]
    
    logger.info(f"Comparing models: {model_types}")
    
    # Create models
    models = create_models_for_comparison(
        model_types,
        embed_dim=args.embed_dim,
        output_dim=128,  # Fixed for classification
    )
    
    if not models:
        logger.error("No valid models created!")
        return
    
    # Determine datasets
    if args.datasets:
        datasets = args.datasets
    else:
        # Quick comparison datasets
        datasets = ["sst2", "imdb", "ag_news", "cola"]
    
    # Validate datasets
    invalid_datasets = [d for d in datasets if d not in AVAILABLE_DATASETS]
    if invalid_datasets:
        logger.warning(f"Invalid datasets (will be skipped): {invalid_datasets}")
        datasets = [d for d in datasets if d in AVAILABLE_DATASETS]
    
    logger.info(f"Evaluating on datasets: {datasets}")
    
    # Create evaluator (we'll use the first model as placeholder)
    first_model = next(iter(models.values()))
    evaluator = Evaluator(
        model=first_model,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cache_dir="./cache",
    )
    
    # Run comparison
    logger.info("Starting model comparison...")
    results = evaluator.compare_models(
        models=models,
        dataset_names=datasets,
        output_dir=args.output_dir,
    )
    
    # Print comparison summary
    summary = results["summary"]
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Print table header
    dataset_names = list(summary.keys())
    model_names = list(models.keys())
    
    if dataset_names:
        # Print header
        print(f"{'Dataset':<15s}", end="")
        for model_name in model_names:
            print(f"{model_name:<15s}", end="")
        print()
        print("-" * (15 + 15 * len(model_names)))
        
        # Print results
        for dataset in dataset_names:
            print(f"{dataset:<15s}", end="")
            for model_name in model_names:
                result = summary[dataset].get(model_name, "N/A")
                if isinstance(result, float):
                    print(f"{result:<15.3f}", end="")
                else:
                    print(f"{str(result):<15s}", end="")
            print()
    
    # Print parameter counts
    print(f"\nModel Parameter Counts:")
    print("-" * 30)
    for model_name, model in models.items():
        param_count = model.count_parameters()
        print(f"{model_name:<20s}: {param_count:>8,} params")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 