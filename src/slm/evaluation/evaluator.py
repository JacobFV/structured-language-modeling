"""
Evaluator class for comprehensive model evaluation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import time
from pathlib import Path
import json

from ..models.base import BaseModel
from ..data.datasets import get_dataset, get_dataloader, get_task_info, AVAILABLE_DATASETS
from .metrics import compute_metrics, TASK_METRICS


class Evaluator:
    """
    Comprehensive evaluator for the SLM evaluation suite.
    
    Implements the 20-task evaluation framework described in the README.
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
            batch_size: Batch size for evaluation
            max_samples: Maximum samples per dataset (for quick testing)
            cache_dir: Directory to cache datasets
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        # Evaluation results
        self.results = {}
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        split: str = "validation",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            dataset_name: Name of dataset to evaluate on
            split: Dataset split to use
            **kwargs: Additional dataset-specific arguments
        
        Returns:
            Dictionary of evaluation results
        """
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Evaluating on {dataset_name} ({split})...")
        
        # Load dataset
        try:
            dataset = get_dataset(
                dataset_name, 
                split=split, 
                cache_dir=self.cache_dir,
                max_samples=self.max_samples,
                **kwargs
            )
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return {"error": str(e)}
        
        # Create dataloader
        dataloader = get_dataloader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Get task info
        task_info = get_task_info(dataset_name)
        task_type = task_info.get("task_type", "classification")
        
        # Evaluate
        start_time = time.time()
        results = self._evaluate_dataloader(dataloader, task_type)
        eval_time = time.time() - start_time
        
        # Add metadata
        results.update({
            "dataset": dataset_name,
            "split": split,
            "task_type": task_type,
            "num_samples": len(dataset),
            "eval_time": eval_time,
            "model_type": self.model.model_type,
        })
        
        return results
    
    def _evaluate_dataloader(
        self, 
        dataloader: DataLoader, 
        task_type: str
    ) -> Dict[str, Any]:
        """Evaluate model on a dataloader."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                try:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                    )
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    return {"error": str(e)}
                
                # Handle different output types
                if task_type == "language_modeling":
                    # For language modeling, compute perplexity
                    if "labels" in batch:
                        labels = batch["labels"]
                    else:
                        labels = batch["input_ids"][:, 1:].contiguous()
                        outputs = outputs[:, :-1, :].contiguous()
                    
                    # Compute loss for perplexity calculation
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
                    loss = loss_fn(
                        outputs.view(-1, outputs.size(-1)), 
                        labels.view(-1)
                    )
                    total_loss += loss.item()
                
                else:
                    # For classification and other tasks
                    if "labels" in batch:
                        labels = batch["labels"]
                        
                        # Get predictions
                        if outputs.dim() > 1 and outputs.size(-1) > 1:
                            predictions = torch.argmax(outputs, dim=-1)
                        else:
                            predictions = outputs.squeeze()
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                num_batches += 1
        
        # Compute metrics
        if task_type == "language_modeling":
            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            return {
                "loss": avg_loss,
                "perplexity": perplexity,
            }
        else:
            if len(all_predictions) == 0:
                return {"error": "No predictions collected"}
            
            metrics = compute_metrics(
                predictions=all_predictions,
                labels=all_labels,
                task_type=task_type
            )
            return metrics
    
    def evaluate_suite(
        self,
        dataset_names: Optional[List[str]] = None,
        splits: Dict[str, str] = None,
        save_results: bool = True,
        output_dir: str = "./evaluation_results",
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple datasets from the evaluation suite.
        
        Args:
            dataset_names: List of datasets to evaluate (None for all)
            splits: Dictionary mapping dataset names to splits
            save_results: Whether to save results to file
            output_dir: Output directory for results
        
        Returns:
            Dictionary of all evaluation results
        """
        if dataset_names is None:
            # Use a subset of key datasets for quick evaluation
            dataset_names = [
                "sst2", "imdb", "ag_news", "cola", "snli",
                "ptb_lm", "wikitext2_lm"
            ]
        
        if splits is None:
            splits = {}
        
        all_results = {}
        summary = {
            "model_type": self.model.model_type,
            "total_datasets": len(dataset_names),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "results_by_category": {},
        }
        
        # Group datasets by category
        from ..data.datasets import TASK_CATEGORIES
        category_map = {}
        for category, datasets in TASK_CATEGORIES.items():
            for dataset in datasets:
                category_map[dataset] = category
        
        for dataset_name in dataset_names:
            split = splits.get(dataset_name, "validation")
            
            try:
                # Handle split mappings for different datasets
                if dataset_name in ["sst2", "cola"] and split == "validation":
                    split = "validation"
                elif dataset_name in ["imdb", "ag_news"] and split == "validation":
                    split = "test"  # Some datasets don't have validation split
                
                results = self.evaluate_dataset(dataset_name, split)
                
                if "error" not in results:
                    all_results[dataset_name] = results
                    summary["successful_evaluations"] += 1
                    
                    # Add to category summary
                    category = category_map.get(dataset_name, "unknown")
                    if category not in summary["results_by_category"]:
                        summary["results_by_category"][category] = {}
                    summary["results_by_category"][category][dataset_name] = results
                    
                    print(f"✓ {dataset_name}: {results.get('accuracy', results.get('perplexity', 'N/A'))}")
                else:
                    print(f"✗ {dataset_name}: {results['error']}")
                    summary["failed_evaluations"] += 1
                    
            except Exception as e:
                print(f"✗ {dataset_name}: {str(e)}")
                summary["failed_evaluations"] += 1
        
        # Save results
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_file = output_path / f"evaluation_results_{self.model.model_type}.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            # Save summary
            summary_file = output_path / f"evaluation_summary_{self.model.model_type}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"Results saved to {output_path}")
        
        return {
            "detailed_results": all_results,
            "summary": summary,
        }
    
    def compare_models(
        self,
        models: Dict[str, BaseModel],
        dataset_names: Optional[List[str]] = None,
        output_dir: str = "./comparison_results",
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the evaluation suite.
        
        Args:
            models: Dictionary mapping model names to model instances
            dataset_names: List of datasets to evaluate
            output_dir: Output directory for results
        
        Returns:
            Comparison results
        """
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Temporarily set model
            original_model = self.model
            self.model = model.to(self.device)
            
            try:
                results = self.evaluate_suite(
                    dataset_names=dataset_names,
                    save_results=False
                )
                all_results[model_name] = results["detailed_results"]
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
            finally:
                # Restore original model
                self.model = original_model
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(all_results)
        
        # Save comparison results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        comparison_file = output_path / "model_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump({
                "detailed_results": all_results,
                "summary": comparison_summary
            }, f, indent=2)
        
        print(f"Comparison results saved to {comparison_file}")
        return {
            "detailed_results": all_results,
            "summary": comparison_summary
        }
    
    def _create_comparison_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary table for model comparison."""
        summary = {}
        
        # Get all datasets that were evaluated
        all_datasets = set()
        for model_results in all_results.values():
            if isinstance(model_results, dict) and "error" not in model_results:
                all_datasets.update(model_results.keys())
        
        # Create summary for each dataset
        for dataset in all_datasets:
            summary[dataset] = {}
            for model_name, model_results in all_results.items():
                if isinstance(model_results, dict) and dataset in model_results:
                    result = model_results[dataset]
                    # Extract primary metric
                    if "accuracy" in result:
                        summary[dataset][model_name] = result["accuracy"]
                    elif "perplexity" in result:
                        summary[dataset][model_name] = result["perplexity"]
                    elif "f1" in result:
                        summary[dataset][model_name] = result["f1"]
                    else:
                        summary[dataset][model_name] = "N/A"
        
        return summary 