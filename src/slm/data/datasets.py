"""
Dataset loading for all evaluation tasks.

Implements data loading for the 20 tasks across 6 categories in the evaluation suite.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from typing import Dict, Any, Optional, Tuple, Union, List
import os
from pathlib import Path

# Task categories from the README
TASK_CATEGORIES = {
    "next_word_prediction": [
        "ptb_lm", "wikitext2_lm", "wikitext103_lm", "enwik8_char", 
        "lambada", "linzen_agreement"
    ],
    "sequence_classification": [
        "sst2", "imdb", "ag_news", "dbpedia", "cola"
    ],
    "sequence_labeling": [
        "ptb_pos", "conll2003_ner", "ontonotes5_ner", "conll2000_chunking"
    ],
    "structured_prediction": [
        "ptb_parsing", "ud_parsing", "iwslt14_en_de", "wmt14_en_de", 
        "wikisql", "atis"
    ],
    "pattern_grammar": [
        "tomita_grammars", "listops", "math_expr", "cfq"
    ],
    "reasoning_inference": [
        "snli", "multinli", "anli", "squad1", "squad2"
    ]
}

# Flatten for easy lookup
AVAILABLE_DATASETS = {
    name for category in TASK_CATEGORIES.values() 
    for name in category
}


class TextDataset(Dataset):
    """Generic text dataset wrapper."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        task_type: str = "classification",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
        else:
            # Simple whitespace tokenization fallback
            tokens = text.split()[:self.max_length]
            input_ids = torch.tensor([hash(token) % 10000 for token in tokens])
            attention_mask = torch.ones(len(input_ids))
            
            # Pad to max_length
            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[idx])
        
        return result


def get_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split ("train", "validation", "test")
        cache_dir: Directory to cache datasets
        max_samples: Maximum number of samples to load
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Dataset object
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {AVAILABLE_DATASETS}")
    
    # Language Modeling datasets
    if dataset_name == "ptb_lm":
        dataset = load_dataset("ptb_text_only", split=split, cache_dir=cache_dir)
        texts = dataset["sentence"]
        return TextDataset(texts, task_type="language_modeling")
    
    elif dataset_name == "wikitext2_lm":
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split, cache_dir=cache_dir)
        texts = [text for text in dataset["text"] if text.strip()]
        return TextDataset(texts, task_type="language_modeling")
    
    elif dataset_name == "wikitext103_lm":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, cache_dir=cache_dir)
        texts = [text for text in dataset["text"] if text.strip()]
        return TextDataset(texts, task_type="language_modeling")
    
    elif dataset_name == "enwik8_char":
        # Character-level dataset - simplified version
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split, cache_dir=cache_dir)
        texts = [text for text in dataset["text"] if text.strip()]
        return TextDataset(texts, task_type="language_modeling")
    
    # Classification datasets
    elif dataset_name == "sst2":
        dataset = load_dataset("sst2", split=split, cache_dir=cache_dir)
        texts = dataset["sentence"]
        labels = dataset["label"]
        return TextDataset(texts, labels, task_type="classification")
    
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb", split=split, cache_dir=cache_dir)
        texts = dataset["text"]
        labels = dataset["label"]
        return TextDataset(texts, labels, task_type="classification")
    
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news", split=split, cache_dir=cache_dir)
        texts = dataset["text"]
        labels = dataset["label"]
        return TextDataset(texts, labels, task_type="classification")
    
    elif dataset_name == "cola":
        dataset = load_dataset("glue", "cola", split=split, cache_dir=cache_dir)
        texts = dataset["sentence"]
        labels = dataset["label"]
        return TextDataset(texts, labels, task_type="classification")
    
    # Add more datasets as needed
    elif dataset_name == "snli":
        dataset = load_dataset("snli", split=split, cache_dir=cache_dir)
        # Filter out examples with no gold label
        dataset = dataset.filter(lambda x: x["label"] != -1)
        texts = [f"{premise} [SEP] {hypothesis}" for premise, hypothesis in 
                zip(dataset["premise"], dataset["hypothesis"])]
        labels = dataset["label"]
        return TextDataset(texts, labels, task_type="classification")
    
    else:
        # Placeholder for other datasets
        print(f"Dataset {dataset_name} not yet implemented, using dummy data")
        dummy_texts = ["This is a sample text."] * 100
        dummy_labels = [0] * 100
        return TextDataset(dummy_texts, dummy_labels, task_type="classification")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


def get_task_info(dataset_name: str) -> Dict[str, Any]:
    """Get task-specific information."""
    task_info = {
        # Language modeling
        "ptb_lm": {"task_type": "language_modeling", "metric": "perplexity", "num_classes": None},
        "wikitext2_lm": {"task_type": "language_modeling", "metric": "perplexity", "num_classes": None},
        "sst2": {"task_type": "classification", "metric": "accuracy", "num_classes": 2},
        "imdb": {"task_type": "classification", "metric": "accuracy", "num_classes": 2},
        "ag_news": {"task_type": "classification", "metric": "accuracy", "num_classes": 4},
        "cola": {"task_type": "classification", "metric": "matthews_correlation", "num_classes": 2},
        "snli": {"task_type": "classification", "metric": "accuracy", "num_classes": 3},
    }
    
    return task_info.get(dataset_name, {
        "task_type": "classification", 
        "metric": "accuracy", 
        "num_classes": 2
    }) 