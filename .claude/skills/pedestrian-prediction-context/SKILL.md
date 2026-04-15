---
name: pedestrian-prediction-context
description: Project context for a pedestrian behavior prediction system using multimodal deep learning (ViT + Motion Encoder + Cross-Attention + Ensemble). Use this skill whenever working on code in this project — including adding features to model components, debugging errors, writing training or evaluation code, modifying data pipelines, or touching any file in the models/, scripts/, or config.py. Also trigger when the user mentions pedestrian prediction, crossing prediction, ViT, motion encoder, cross-attention, ensemble model, LMDB dataset, or any of the project's file names. Even if the user just says "fix this bug" or "add a feature," if the context involves this project, use this skill.
---

# Pedestrian Prediction Project Context

You are working on an undergraduate thesis project that predicts pedestrian behavior (crossing, looking, actions) from video sequences using a multimodal deep learning pipeline.

## Core Principle: Minimal, Surgical Changes

- **Write the least code necessary.** 5 lines over 20. No bloated, convoluted code.
- **Preserve what works.** Understand existing code before changing it. Small edits over rewrites.
- **Touch only what's needed.** Don't "improve" unrelated code unless asked.
- **Explain before you act.** Briefly state what you're changing and why.

## Architecture

Four components in a pipeline:

1. **ViT_Hierarchical** (`models/Vision_Transformer.py`) — Visual feature extraction via ViT with hierarchical windowed attention. Takes tight + context crops.
2. **MotionEncoder** (`models/Motion_Encoder.py`) — Temporal motion pattern encoding from frame sequences.
3. **CrossAttentionModule** (`models/Cross_Attention_Module.py`) — Multimodal fusion of visual and motion features. Frame-level cross-attention with logsumexp pooling.
4. **EnsembleModel** (`models/Unified_Module.py`) — Top-level model wiring all components. Produces predictions for three tasks.

**Data flows like this:**
```
Video frames → ViT (visual features) ─┐
                                       ├→ CrossAttention → EnsembleModel → predictions
Motion sequences → MotionEncoder ──────┘
```

The model predicts three things simultaneously:
- **Action**: walking / standing
- **Look**: looking at traffic or not
- **Crossing**: crossing behavior classification

## Key Files

See `references/project-details.md` for full file organization, config patterns, data loading, and training loop structure.

- `models/*.py` — Architecture (most feature work happens here)
- `config.py` — Hyperparameters and configuration
- `train.py` — Training loop (AMP, gradient scaling, memory management)
- `test.py` — Evaluation with comprehensive metrics
- `scripts/lmdb_dataset.py` — LMDB data loading

## Coding Rules

**Style**: PascalCase classes, snake_case functions/variables, UPPER_SNAKE_CASE constants. Underscore prefix for private methods. Imports: stdlib → third-party → local.

**PyTorch**: `.to(device)` for placement. `torch.no_grad()` for inference. `model.eval()`/`model.train()` correctly. Mixed precision via `torch.cuda.amp.autocast()`. Gradient scaling.

**Memory**: `gc.collect()` + `torch.cuda.empty_cache()` after chunks. `pin_memory=True`. Process large data in chunks.

**Quality**: Type hints on signatures. Docstrings on classes/complex functions. Comments for math/complex logic. Functions under 50 lines. Lines under 120 chars.

**Verify** after modifying model components:
```bash
python -c "from models.ComponentName import ClassName; print('OK')"
```

## Adding Features

1. Read the existing code first
2. Check `config.py` for related configuration
3. Follow existing patterns (see `references/project-details.md`)
4. One feature per change — keep it focused
5. Ensure compatibility with existing training loop and data pipeline
6. Type hints + brief docstring

## Debugging

1. Check tensor shapes and dtypes first (most common issue)
2. Verify device placement — CPU/CUDA mixing causes silent failures
3. Isolate: data loading vs forward pass vs loss computation
4. Use `.shape`, `.device`, `.dtype` for inspection
5. Fix first, refactor later — don't combine the two
