"""
Metrics for different evaluation tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, mean_squared_error, 
    classification_report
)
from typing import List, Dict, Any, Optional, Union
import torch


def compute_metrics(
    predictions: Union[List, np.ndarray, torch.Tensor],
    labels: Union[List, np.ndarray, torch.Tensor],
    task_type: str = "classification",
    **kwargs
) -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        predictions: Model predictions
        labels: True labels
        task_type: Type of task
        **kwargs: Additional task-specific arguments
    
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if task_type == "classification":
        return compute_classification_metrics(predictions, labels)
    elif task_type == "regression":
        return compute_regression_metrics(predictions, labels)
    elif task_type == "language_modeling":
        return compute_lm_metrics(predictions, labels)
    elif task_type == "sequence_labeling":
        return compute_sequence_labeling_metrics(predictions, labels)
    else:
        # Default to classification
        return compute_classification_metrics(predictions, labels)


def compute_classification_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute classification metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    # Add Matthews correlation coefficient for binary classification
    if len(np.unique(labels)) == 2:
        mcc = matthews_corrcoef(labels, predictions)
        metrics["matthews_correlation"] = mcc
    
    return metrics


def compute_regression_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))
    
    # R-squared
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def compute_lm_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute language modeling metrics."""
    # For language modeling, predictions are usually logits
    # and we compute perplexity from loss
    
    # If predictions are class indices, compute accuracy
    if predictions.dtype in [np.int32, np.int64]:
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    else:
        # For continuous predictions, use MSE as proxy
        mse = mean_squared_error(labels, predictions)
        return {"mse": mse}


def compute_sequence_labeling_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute sequence labeling metrics (NER, POS tagging, etc.)."""
    # Flatten sequences if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Filter out padding tokens (assuming 0 is padding)
    mask = labels != 0
    predictions = predictions[mask]
    labels = labels[mask]
    
    return compute_classification_metrics(predictions, labels)


def compute_bleu_score(
    predictions: List[str], 
    references: List[List[str]]
) -> float:
    """
    Compute BLEU score for machine translation.
    
    This is a simplified implementation. For production use,
    consider using nltk.translate.bleu_score or sacrebleu.
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu
        bleu = corpus_bleu(references, predictions)
        return bleu
    except ImportError:
        print("NLTK not available for BLEU score computation")
        return 0.0


def compute_exact_match(
    predictions: List[str], 
    references: List[str]
) -> float:
    """Compute exact match accuracy."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    exact_matches = sum(
        pred.strip().lower() == ref.strip().lower() 
        for pred, ref in zip(predictions, references)
    )
    return exact_matches / len(predictions)


def compute_f1_token(
    predictions: List[str], 
    references: List[str]
) -> float:
    """Compute token-level F1 score (useful for QA tasks)."""
    def _f1_score(prediction: str, reference: str) -> float:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        common = set(pred_tokens) & set(ref_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    f1_scores = [
        _f1_score(pred, ref) 
        for pred, ref in zip(predictions, references)
    ]
    return np.mean(f1_scores)


def get_metric_fn(metric_name: str):
    """Get metric function by name."""
    metric_functions = {
        "accuracy": accuracy_score,
        "matthews_correlation": matthews_corrcoef,
        "bleu": compute_bleu_score,
        "exact_match": compute_exact_match,
        "f1_token": compute_f1_token,
    }
    
    if metric_name in metric_functions:
        return metric_functions[metric_name]
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


# Task-specific metric configurations
TASK_METRICS = {
    "ptb_lm": ["perplexity"],
    "wikitext2_lm": ["perplexity"],
    "sst2": ["accuracy", "f1"],
    "imdb": ["accuracy", "f1"],
    "ag_news": ["accuracy", "f1"],
    "cola": ["matthews_correlation"],
    "snli": ["accuracy"],
    "multinli": ["accuracy"],
    "squad1": ["exact_match", "f1_token"],
    "squad2": ["exact_match", "f1_token"],
    "iwslt14_en_de": ["bleu"],
    "wmt14_en_de": ["bleu"],
} 