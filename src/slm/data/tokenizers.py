"""
Tokenizer utilities for the SLM framework.
"""

from transformers import AutoTokenizer
from typing import Optional, Dict, Any


def get_tokenizer(
    tokenizer_name: str = "bert-base-uncased",
    cache_dir: Optional[str] = None,
    **kwargs
) -> AutoTokenizer:
    """
    Get a tokenizer by name.
    
    Args:
        tokenizer_name: Name or path of the tokenizer
        cache_dir: Directory to cache the tokenizer
        **kwargs: Additional tokenizer arguments
    
    Returns:
        Tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        **kwargs
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def build_vocab_from_dataset(texts, vocab_size: int = 10000) -> Dict[str, int]:
    """
    Build a simple vocabulary from a list of texts.
    
    Args:
        texts: List of text strings
        vocab_size: Maximum vocabulary size
    
    Returns:
        Dictionary mapping tokens to indices
    """
    from collections import Counter
    
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Create vocabulary with most frequent tokens
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    
    for token, count in token_counts.most_common(vocab_size - 4):
        vocab[token] = len(vocab)
    
    return vocab 