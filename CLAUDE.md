# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VtT ("Vision to Text") is the official implementation of *"Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning"* (CVPR 2026). It introduces a method for Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL) that uses a Mamba-based cross-modal bridge to align vision and text representations from a frozen CLIP model, with a novel "absorb token" mechanism to inject learned cross-modal features back into the text encoder.

## Running Experiments

```bash
# EuroSAT 1-shot
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset EuroSAT

# ISIC 5-shot (harder domain, higher LRs needed)
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset ISIC --lr 4e-4 --mamba_lr 1e-3
```

Key arguments:
- `--dataset`: `ISIC`, `EuroSAT`, `CropDisease`, `ChestX`
- `--shot`: 1 or 5
- `--encoder`: `vision`, `text`, or `both` (which CLIP encoder gets LoRA)
- `--r`, `--alpha`: LoRA rank and scaling
- `--lr`, `--mamba_lr`: Learning rates for CLIP/LoRA and Mamba bridge respectively
- `--beta`: MAE loss weight (default 7)
- `--grad_steps`: Gradient similarity window for ProGrad (default 50)

## Installation

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm causal-conv1d
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

## Architecture

### Data Flow

1. **Support/query images** → CLIP ViT-B/16 vision encoder → patch embeddings from all transformer layers
2. **Class name text** → CLIP text encoder → text embeddings from all transformer layers
3. **Interleave** vision and text patches into a 5×5 spatial grid
4. **Mamba bridge** (`Mamba_models/mamba_ffn_neck.py` via `Mamba_tool/mamba_blocks.py`) processes the interleaved grid for cross-modal alignment
5. **Absorb token** — the Mamba output is injected at position 5 of the CLIP text encoder token sequence (hardcoded in `clip/model.py:~401`) and the encoder re-runs to produce `mae_embeddings`
6. **Dual classification heads** — standard CLIP cosine similarity + MAE cosine similarity, combined at 0.5/0.5

### Key Files

| File | Role |
|------|------|
| `main.py` | Entry point; loads CLIP ViT-B/16, constructs dataset managers, calls `run_lora()` |
| `VtT.py` | Core training/eval logic: `Mamba_Net`, `run_lora()`, `fsl_test()`, ProGrad gradient projection |
| `run_utils.py` | Argument parser (`get_arguments()`) and `set_random_seed()` |
| `utils.py` | `cls_acc()`, `clip_classifier()`, `pre_load_features()` |
| `clip/model.py` | Modified CLIP — adds `more_token` (inject custom token into vision encoder) and `inject` (absorb token into text encoder at position 5) parameters to `encode_image()` and `encode_text()` |
| `loralib/` | LoRA layer wrappers and `apply_lora()` / `save_lora()` / `load_lora()` utilities |
| `Mamba_tool/mamba_blocks.py` | VSSM (Vision State-Space Model) blocks used by the Mamba bridge |
| `Mamba_models/mamba_ffn_neck.py` | `MambaNeck` — wraps VSSM blocks with FFN layers (512-dim, 5×5 grid, 2 layers) |
| `fslcd_datasets_aug2/` | Dataset managers for ISIC, EuroSAT, CropDisease, ChestX — each exposes `SetDataManager` |

### Training Optimization: ProGrad

`VtT.py` lines ~48–99 implement **Projected Gradient** descent to handle conflicting gradients between the cross-entropy loss (standard CLIP) and the MAE loss (cross-modal bridge). If the cosine similarity between the two gradient vectors is negative, the MAE gradient is projected onto the CE gradient's normal plane. Beta can drop to −1 to disable the MAE loss entirely when gradients consistently conflict.

### LoRA Application

`loralib/utils.py:apply_lora()` wraps Q/K/V (and optionally O) projections in CLIP's transformer attention layers. Target positions can be `top3`, `mid`, `bottom`, `half-up`, or `all`. The scaling factor uses `lora_alpha / sqrt(r)` (non-standard — see `loralib/layers.py:42`).

## Dataset Paths

Dataset paths are **hardcoded** in `fslcd_datasets_aug2/` (e.g., `/home/zzy/fsl_CD_dataset/`). Update these paths for your environment before running.

## Modified CLIP vs. Upstream CLIP

`clip/model.py` is a fork of OpenAI CLIP with two additions that are central to VtT:
- `encode_image(..., more_token=None, ret_all=False)` — `more_token` replaces the CLS token; `ret_all` returns intermediate layer outputs
- `encode_text(..., inject=None, ret_all=False)` — `inject` replaces position 5 with the Mamba-produced absorb token; `ret_all` returns all-layer embeddings

Do not upgrade or replace this with the upstream CLIP package — the modifications are load-bearing.
