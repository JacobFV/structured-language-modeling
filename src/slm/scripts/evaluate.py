#!/usr/bin/env python3
"""
Main evaluation script for Structured Language Modeling.

This script evaluates trained models on the full evaluation suite.
"""

import argparse
import torch
from pathlib import Path

from ..utils.config import get_default_config, load_config
from ..utils.model_factory import create_model, create_model_from_checkpoint
from ..utils.logging import setup_logging
from ..evaluation.evaluator import Evaluator
from ..data.datasets import AVAILABLE_DATASETS


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLM models")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--datasets", nargs="+", help="Datasets to evaluate on")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset (for quick testing)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Load model
    if args.model_path:
        logger.info(f"Loading model from: {args.model_path}")
        
        if args.config:
            config = load_config(args.config)
            model = create_model_from_checkpoint(args.model_path, config.model)
        else:
            model = create_model_from_checkpoint(args.model_path)
    else:
        # Create a default model for testing
        logger.info("No model path provided, creating default model for testing")
        config = get_default_config()
        model = create_model(config.model)
    
    logger.info(f"Model type: {model.model_type}")
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Setup evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )
    
    # Determine datasets to evaluate
    if args.datasets:
        datasets = args.datasets
    else:
        # Default subset for quick evaluation
        datasets = [
            "sst2", "imdb", "ag_news", "cola", "snli",
            "ptb_lm", "wikitext2_lm"
        ]
    
    # Validate datasets
    invalid_datasets = [d for d in datasets if d not in AVAILABLE_DATASETS]
    if invalid_datasets:
        logger.warning(f"Invalid datasets (will be skipped): {invalid_datasets}")
        datasets = [d for d in datasets if d in AVAILABLE_DATASETS]
    
    logger.info(f"Evaluating on datasets: {datasets}")
    
    # Run evaluation
    results = evaluator.evaluate_suite(
        dataset_names=datasets,
        save_results=True,
        output_dir=args.output_dir,
    )
    
    # Print summary
    summary = results["summary"]
    logger.info(f"Evaluation completed!")
    logger.info(f"Successful evaluations: {summary['successful_evaluations']}")
    logger.info(f"Failed evaluations: {summary['failed_evaluations']}")
    
    # Print results by category
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for category, category_results in summary["results_by_category"].items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        for dataset, result in category_results.items():
            if "accuracy" in result:
                metric_value = f"{result['accuracy']:.3f}"
                metric_name = "Accuracy"
            elif "perplexity" in result:
                metric_value = f"{result['perplexity']:.2f}"
                metric_name = "Perplexity"
            elif "f1" in result:
                metric_value = f"{result['f1']:.3f}"
                metric_name = "F1"
            else:
                metric_value = "N/A"
                metric_name = "Score"
            
            print(f"{dataset:15s}: {metric_name:10s} = {metric_value}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 