# Structured Language Modeling

A research framework for implementing language models with **progressive linguistic priors**, from simple baselines to complex regex and context-free grammar structures.

## 🎯 Motivation

Suppose we could write an algorithm that directly **synthesizes** a language model with an *a-priori* understanding of syntactic and semantic structure, instead of rediscovering those regularities from data. Such a model would:

1. **Remove much of the heavy empirical training loop** (fewer GPU hours)
2. **Enforce language-first inductive biases** in its latent state  
3. **Enable qualitatively different kinds of systematic reasoning** than "enslopified" next-word predictors

The project goal is to embed as many *general linguistic priors* as possible **inside the architecture itself**. We start with a non-prior baseline (mean pooling) and then progressively graft in convolution, recurrent nets, attention, *regular-expression* banks and finally *context-free-grammar* (CFG) structure.

## 🏗️ Architecture Overview

The framework implements a progression of increasingly sophisticated linguistic priors:

| Model | Architecture | Linguistic Prior | Use Case |
|-------|-------------|------------------|----------|
| **Mean Pooling** | `Y = (1/L) * Σ X_t` | None (baseline) | Sanity check |
| **Convolutional** | 1D CNN with multiple kernels | N-gram patterns | Local morphology |
| **RNN/LSTM** | Recurrent networks | Sequential dependencies | Temporal structure |
| **Multi-Head Attention** | Transformer layers | Long-range relationships | Global context |
| **Regex** | Differentiable finite automata | Pattern matching | Structured text |
| **CFG** | Probabilistic context-free grammars | Hierarchical syntax | Compositional structure |

## 📦 Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/JacobFV/structured-language-modeling.git
cd structured-language-modeling

# Install with uv
uv sync

# For development
uv sync --extra dev

# For full features (including spaCy)
uv sync --extra full
```

### Using pip

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Training a Model

```bash
# Train a convolutional model on sentiment analysis
uv run slm-train --model ConvolutionalModel --dataset sst2 --epochs 3

# Train with custom configuration
uv run slm-train --config configs/default.yaml --model LSTMModel --dataset imdb

# Train with Weights & Biases logging
uv run slm-train --model MultiHeadAttentionModel --dataset cola --wandb
```

### Evaluating Models

```bash
# Evaluate a trained model
uv run slm-evaluate --model-path ./outputs/final_model.pt

# Quick evaluation on subset of datasets
uv run slm-evaluate --datasets sst2 imdb --max-samples 1000

# Evaluate on specific datasets
uv run slm-evaluate --datasets ptb_lm wikitext2_lm
```

### Comparing Architectures

```bash
# Compare multiple model architectures
uv run slm-compare --models MeanPoolingModel ConvolutionalModel LSTMModel

# Quick comparison on smaller datasets
uv run slm-compare --max-samples 500 --datasets sst2 cola

# List available models
uv run slm-compare --list-models
```

## 📊 Evaluation Suite

The framework implements a comprehensive **20-task evaluation suite** across 6 categories:

### A. Next-Word Prediction
- **PTB, WikiText-2/103**: Language modeling (perplexity)
- **enwik8**: Character-level modeling
- **LAMBADA**: Discourse coherence  
- **Linzen Agreement**: Subject-verb dependencies

### B. Sequence Classification
- **SST-2, IMDB**: Sentiment analysis
- **AG News, DBPedia**: Topic classification
- **CoLA**: Acceptability judgments

### C. Sequence Labeling  
- **PTB POS**: Part-of-speech tagging
- **CoNLL-2003 NER**: Named entity recognition
- **CoNLL-2000**: Chunking

### D. Structured Prediction
- **PTB Parsing**: Constituency parsing
- **UD Parsing**: Dependency parsing  
- **IWSLT/WMT Translation**: Machine translation
- **WikiSQL, ATIS**: Semantic parsing

### E. Pattern/Grammar Tasks
- **Tomita Grammars**: Finite-state recognition
- **ListOps**: Hierarchical reasoning
- **Math Expressions**: Arithmetic parsing
- **CFQ**: Compositional generalization

### F. Reasoning/Inference
- **SNLI, MultiNLI**: Natural language inference
- **SQuAD**: Reading comprehension

## 🧪 Model Architectures

### Basic Models (Ready to Use)

```python
from slm.utils.model_factory import create_model
from slm.utils.config import get_default_config

config = get_default_config()

# Mean pooling baseline
config.model.type = "MeanPoolingModel"
model = create_model(config.model)

# CNN for n-gram patterns  
config.model.type = "ConvolutionalModel"
config.model.num_filters = 100
config.model.kernel_sizes = [2, 3, 4, 5]
model = create_model(config.model)

# RNN/LSTM for sequential processing
config.model.type = "LSTMModel" 
config.model.hidden_dim = 256
config.model.bidirectional = True
model = create_model(config.model)

# Transformer with multi-head attention
config.model.type = "MultiHeadAttentionModel"
config.model.num_heads = 8
config.model.num_layers = 6
model = create_model(config.model)
```

## 📁 Project Structure

```
structured-language-modeling/
├── src/slm/
│   ├── models/           # Model architectures
│   │   ├── base.py       # Abstract base model
│   │   ├── baseline.py   # Mean pooling
│   │   ├── conv.py       # CNN models
│   │   ├── rnn.py        # RNN/LSTM models
│   │   ├── attention.py  # Transformer models
│   │   ├── regex.py      # Regex models (placeholder)
│   │   └── cfg.py        # CFG models (placeholder)
│   ├── data/             # Data loading and processing
│   ├── training/         # Training infrastructure
│   ├── evaluation/       # Evaluation framework
│   ├── utils/            # Utilities and configuration
│   └── scripts/          # CLI entry points
├── configs/              # Configuration files
├── scripts/              # Legacy CLI scripts
└── pyproject.toml        # Project configuration
```

## ⚙️ Configuration

The framework uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. See `configs/default.yaml` for the default configuration.

### Key Configuration Sections

```yaml
model:
  type: "ConvolutionalModel"  # Model architecture
  embed_dim: 256              # Embedding dimension
  output_dim: 128             # Output dimension

data:
  dataset_name: "sst2"        # Dataset to use
  batch_size: 32              # Training batch size
  max_length: 128             # Maximum sequence length

training:
  num_epochs: 3               # Training epochs
  learning_rate: 5e-4         # Learning rate
  optimizer: "adamw"          # Optimizer type
  scheduler: "warmup_cosine"  # LR scheduler

evaluation:
  datasets: ["sst2", "imdb"] # Evaluation datasets
  max_samples: null          # Max samples (null = all)
```

## 🧬 Research Framework

This framework is designed for systematic research on linguistic priors in neural architectures. Key research directions:

### Encoder Swap Experiments
Keep downstream tasks identical, swap the encoder architecture, compare performance under identical training conditions.

### Sample Efficiency Analysis  
Measure performance gains from built-in priors across different training data sizes (1%, 10%, 100%).

### Symbolic Prior Ablations
Compare discrete vs. relaxed vs. removed symbolic components (regex/CFG) to quantify the contribution of each linguistic prior.

### Generalization Stress Tests
- Zero-shot splits in CFQ
- Length extrapolation in ListOps  
- Long-distance dependencies in Linzen agreement

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/JacobFV/structured-language-modeling.git
cd structured-language-modeling
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`uv run pytest`)
6. Format your code (`uv run black . && uv run isort .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{structured-language-modeling,
  title={Structured Language Modeling: Progressive Linguistic Priors for Neural Architectures},
  author={Jacob Valdez},
  year={2024},
  url={https://github.com/JacobFV/structured-language-modeling}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Built on top of [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/)
- Evaluation datasets from [Hugging Face Datasets](https://huggingface.co/datasets)
- Configuration management with [OmegaConf](https://omegaconf.readthedocs.io/)
- Dependency management with [uv](https://github.com/astral-sh/uv)

---

**Framework Status**: 🚧 **Research** - Core architectures implemented, symbolic models (Regex/CFG) are research prototypes

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/JacobFV/structured-language-modeling).