# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An educational Python project that implements a Transformer-based LLM code generation demo system from scratch. It simulates every step of a large language model — tokenization, embedding, Transformer encoder-decoder inference, autoregressive code generation, post-processing, and KV caching. All code is in Chinese-documented with Chinese comments.

## Commands

```bash
# Install dependencies
pip install torch numpy matplotlib seaborn

# Run the interactive demo (6 modes: basic, sampling, cache, multi-sample, visualization, interactive)
python scripts/main.py

# Run all tests (8 module tests + math verification)
python scripts/tests/test_all.py

# Verify math foundation documentation
python scripts/tests/verify_math.py

# Minimal API usage
python -c "from scripts.pipeline import CodeGenerationPipeline; p = CodeGenerationPipeline(preset='small'); print(p.generate('public class Hello')['processed_code'])"
```

Tests use raw `try/except` assertions (no pytest/unittest framework) and output pass/fail for each module.

## Architecture

The system is modular with clear separation of concerns, following this data flow:

```
User prompt → GenerationCache (check) → SimpleTokenizer (encode)
  → TransformerModel (encoder-decoder w/ KV cache) → Temperature sampling
  → SimplePostProcessor (format + imports) → GenerationCache (store) → output
```

### Module Map

| Layer | File | Responsibility |
|-------|------|----------------|
| **Entry** | `scripts/main.py` | Interactive CLI with 6 demo modes |
| **Pipeline** | `scripts/pipeline.py` | End-to-end orchestration: cache → tokenize → generate → post-process → cache update |
| **Tokenizer** | `scripts/core/tokenizer.py` | Regex-based Java code tokenizer; builds vocab with caching |
| **Attention** | `scripts/core/attention.py` | Multi-head attention (Q/K/V projections, scaled dot-product, masking) + sinusoidal positional encoding |
| **Transformer** | `scripts/core/transformer.py` | Full encoder-decoder with step-by-step inference, KV cache integration |
| **Generator** | `scripts/generation/code_generator.py` | Autoregressive generation (temperature, top-k, top-p sampling) + post-processor (bracket matching, formatting) |
| **Result Cache** | `scripts/optimization/cache.py` | OrderedDict/LRU-based cache with exact (MD5) and semantic matching |
| **KV Cache** | `scripts/optimization/kv_cache.py` | Pre-allocated cache tensors for O(n) decoder inference |
| **Config** | `scripts/config/presets.py` | Three presets (`tiny`/`small`/`medium`) via `get_preset_config()` |
| **Logger** | `scripts/utils/logger.py` | `DualOutputLogger` — mirrors stdout to timestamped log files in `scripts/logs/` |
| **Visualizer** | `scripts/utils/visualizer.py` | Matplotlib/seaborn plots for attention weights, generation traces, probability distributions |
| **Training** | `scripts/optional/training/trainer.py` | Optional training loop: CrossEntropyLoss, AdamW, warmup-linear scheduler, checkpointing |

### Preset System

Three configurations hiding all underlying parameters:

| Preset | Vocab | d_model | Heads | Encoder/Decoder Layers | Cache | ~Parameters |
|--------|-------|---------|-------|----------------------|-------|-------------|
| tiny | 500 | 64 | 4 | 1/1 | 20 | 50K |
| small | 1000 | 128 | 8 | 2/2 | 50 | 200K |
| medium | 2000 | 256 | 8 | 4/4 | 100 | 800K |

### Key Design Patterns

- **Preset-first configuration**: Users pick a preset string; internals are abstracted via `get_preset_config()`.
- **Step-by-step inference**: `TransformerModel` exposes a `step()` method for one decoder token at a time, enabling KV cache injection between steps.
- **Cache-aspect oriented**: Generation is always routed through `GenerationCache` (LRU) which delegates to the model on miss and stores results on hit. Semantic matching uses token-level overlap, not just exact string match.
- **Device-agnostic**: Pipeline accepts `device='cuda'` for GPU acceleration.
- **Logging context manager**: `logging_context()` wraps any block with simultaneous console + file logging; log files are date-stamped in `scripts/logs/`.

### Key Conventions

- All docstrings and comments are in **Chinese** — this is a teaching project for Chinese-speaking learners.
- No build system required — run directly with `python`. No `venv` in VCS.
- No CI/CD or linter configuration present.
- Testing uses simple scripts (no pytest/unittest framework).
- Code is single-file-per-concept — each module is self-contained and readable as a standalone reference.
