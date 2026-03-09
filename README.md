# nanoGPT

A minimal GPT language model built from scratch with PyTorch, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

## Overview

This project implements a small-scale GPT (Generative Pre-trained Transformer) model designed for learning and experimentation. It generates coherent short stories after training on a 5% subset of the TinyStories dataset.

## Architecture

| Component | Detail |
|-----------|--------|
| Layers | 6 transformer blocks |
| Attention Heads | 6 per layer |
| Embedding Dimension | 384 |
| Context Length | 256 tokens |
| Vocabulary | 50,304 (GPT-2 tokenizer) |
| Dropout | 0.2 |

**Key features:**
- **Flash Attention** via PyTorch's `scaled_dot_product_attention` for fast, memory-efficient self-attention
- **Weight Tying** between token embeddings and the output head to reduce parameters
- **Pre-norm architecture** with LayerNorm before attention and MLP blocks

## Training

The training loop includes several modern optimization techniques:

- **Cosine Learning Rate Schedule** with linear warmup (100 steps) and decay to 10% of peak LR
- **Gradient Accumulation** (4 micro-steps) to simulate larger effective batch sizes
- **Mixed Precision Training** (bfloat16/float16) for speed on compatible hardware
- **Gradient Clipping** (max norm 1.0) for training stability
- **AdamW Optimizer** with decoupled weight decay (0.1) and beta values (0.9, 0.95)

## Project Structure

```
nano-gpt/
├── data.py      # Downloads TinyStories, tokenizes with GPT-2 encoder, saves as binary
├── model.py     # GPT model: CausalSelfAttention, Block, and GPT classes
├── train.py     # Training loop with LR scheduling and gradient accumulation
├── .gitignore   # Excludes binary data and common artifacts
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install torch numpy tiktoken datasets
```

### 1. Prepare the Data

```bash
python data.py
```

This downloads a 5% subset of TinyStories, tokenizes it using the GPT-2 tokenizer, and saves `train.bin` and `val.bin` to the project directory.

### 2. Train the Model

```bash
python train.py
```

Training runs for 5,000 iterations by default. A CUDA-capable GPU is recommended but not required (it will fall back to CPU).

## Configuration

Model hyperparameters can be adjusted in `model.py` (GPTConfig class) and training settings at the top of `train.py`:

```python
# model.py
block_size = 256      # context window
n_layer = 6           # transformer blocks
n_head = 6            # attention heads
n_embd = 384          # embedding dimension

# train.py
batch_size = 4
max_iters = 5000
learning_rate = 6e-4
grad_accum_steps = 4
```

## License

This project is open source and available for educational purposes.