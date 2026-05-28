# Pedestrian Behavior Prediction

Undergraduate thesis project: multimodal deep learning for pedestrian behavior prediction on the
[PIE dataset](https://github.com/aras62/PIE). The model jointly predicts three binary tasks from
short video-frame sequences:

- **actions** — walking vs. standing
- **looks** — looking at traffic vs. not
- **crosses** — crossing vs. not crossing

## Architecture

```
Video frames → ViT_Hierarchical (context crop)   ──┐
                                                    ├→ CrossAttentionModule → EnsembleModel → {actions, looks, crosses}
Motion seqs  → MotionEncoder (tight crop + motion) ──┘
```

| Component | File | Role |
|---|---|---|
| `ViT_Hierarchical` | [models/Vision_Transformer.py](models/Vision_Transformer.py) | Hierarchical windowed-attention ViT over context crops → `[B, T, D]` |
| `MotionEncoder` | [models/Motion_Encoder.py](models/Motion_Encoder.py) | Temporal ConvNet-GRU-Attention over tight crops + motion cues → `[B, T, D]` |
| `CrossAttentionModule` | [models/Cross_Attention_Module.py](models/Cross_Attention_Module.py) | Frame-level cross-attention fusion, logsumexp pooling, per-task logits |
| `EnsembleModel` | [models/Unified_Module.py](models/Unified_Module.py) | Wires components together, applies LayerNorm before fusion |
| Ablation variants | [models/AblationModels.py](models/AblationModels.py) | `MotionOnlyModel`, `VisualOnlyModel`, `VanillaConcatModel` |

All modules share a unified `d_model = 128` via [`get_unified_dim_model()`](config.py).

Model output dict keys: `actions`, `looks`, `crosses_frame` (frame-level crosses is
full-model only).

## Setup

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # verify CUDA
```

## Training

```bash
# Single-phase training on the full dataset
python train.py

# Two-phase training: balanced subset → full fine-tune → decoupled classifier heads
python train_two_phase.py
```

Both pipelines use AMP + gradient scaling, per-chunk memory cleanup, and class-imbalance handling
(see [class_imbalance_strategies.py](class_imbalance_strategies.py) and
[imbalance_config.py](imbalance_config.py)).

## Evaluation

```bash
python test.py                              # full model
python test.py --model_type motion_only     # ablations
python test.py --model_type visual_only
python test.py --model_type vanilla_concat
```

Reported per task: Accuracy, F1, AUC, Precision, Recall. Latency, FLOPs, and FPS are also logged.

## Inference

```bash
python main.py            # video inference (uses ultralytics + tracked crops)
python label_count.py     # sanity-check LMDB label distribution
```

## Data Pipeline

- **Storage**: LMDB — each sample stores `<key>_meta` (pickled) plus image frames.
- **Dataset**: [`LMDBChunkDataset`](scripts/lmdb_dataset.py).
- **Collation / utils**: [`scripts/train_utils.py`](scripts/train_utils.py) provides `collate_fn`,
  `EarlyStopping`, `remap_cross_labels`, `gather_chunks`, `wait_for_memory`, `mp_async_load`.
- **Normalization**: ImageNet stats (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
- **Crosses labels**: call `remap_cross_labels()` on every batch before loss — the raw PIE crosses
  label space is remapped to `{0, 1}`.

## Project Structure

```
├── models/                        # model components and ablation variants
│   ├── Vision_Transformer.py
│   ├── Motion_Encoder.py
│   ├── Cross_Attention_Module.py
│   ├── Unified_Module.py
│   └── AblationModels.py
├── scripts/                       # data + training utilities
│   ├── lmdb_dataset.py
│   ├── train_utils.py
│   ├── model_utils.py             # get_model(), model_forward() dispatcher
│   ├── preprocess_data_lmdb.py
│   ├── generate_sequences.py
│   ├── augment_sequences.py
│   ├── balance_sequences.py
│   ├── split_balance_sequences_all.py
│   └── pedestrian_detection.py
├── PIE/                           # PIE dataset toolkit (pie_data.py)
├── config.py                      # d_model + ViT/MotionEncoder kwargs
├── train.py                       # standard training
├── train_two_phase.py             # balanced → full → decoupled training
├── test.py                        # evaluation (full + ablation)
├── main.py                        # video inference
├── label_count.py                 # LMDB label distribution check
├── class_imbalance_strategies.py  # focal loss, reweighting, samplers
├── imbalance_config.py            # imbalance preset configurations
├── visualize_gt.py                # ground-truth visualizer
├── visualize_comparison.py        # prediction vs. GT comparison
└── requirements.txt
```

## Further Reading

- [CLAUDE.md](CLAUDE.md) — developer reference: architecture invariants, training patterns, data
  conventions, debugging checklist.
- [GUIDELINE.md](GUIDELINE.md) — project guidelines.
