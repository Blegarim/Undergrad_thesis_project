# Project Details Reference

This file contains the detailed implementation patterns, configuration, and conventions for the pedestrian prediction project. Update this file as the project evolves.

## File Organization

```
models/           # Model architecture definitions
├── __init__.py
├── Vision_Transformer.py    # ViT_Hierarchical class
├── Motion_Encoder.py        # MotionEncoder class
├── Cross_Attention_Module.py # CrossAttentionModule class
└── Unified_Module.py        # EnsembleModel (top-level model)

scripts/          # Data processing and utilities
├── __init__.py
├── lmdb_dataset.py          # LMDB dataset + DataLoader setup
├── pedestrian_detection.py  # Detection utilities
└── preprocess_*.py          # Various preprocessing scripts

config.py         # All configuration parameters
train.py          # Training script
test.py           # Testing/evaluation script
main.py           # Inference on video
requirements.txt  # Dependencies
```

## Model Initialization Pattern

This is how the ensemble model gets assembled — follow this pattern when modifying initialization:

```python
model = EnsembleModel(
    motion_enc=MotionEncoder(**motion_enc_args),
    vit=ViT_Hierarchical(**vit_args),
    cross_attention=CrossAttentionModule(
        d_model=embedding_dim,
        num_heads=4,
        num_classes_dict=num_classes_dict,
        use_frame_crosses=True,
        frame_pool="logsumexp",
    )
).to(device)
```

## Data Loading Pattern

```python
dataset = LMDBChunkDataset(
    lmdb_path,
    transform_tight=base_transforms,
    transform_context=base_transforms
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=use_pin_memory
)
```

Image normalization uses ImageNet standards:
```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

## Training Loop Pattern

```python
model.train()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.amp.autocast(enabled=use_amp):
    outputs = model(images_tight, images_context, motions)
    loss = compute_loss(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Key details: uses automatic mixed precision (AMP), gradient scaling, and `set_to_none=True` for optimizer zero_grad (more efficient than default).

## Error Handling Patterns

```python
# Path validation
assert os.path.exists(model_path), f"Model not found: {model_path}"

# CUDA memory management
if device.type == "cuda":
    torch.cuda.empty_cache()
```

## Evaluation Metrics

The project tracks these metrics — when writing evaluation code, include all of them:
- Accuracy
- F1 score
- AUC (area under ROC curve)
- Precision
- Recall

Results get logged to CSV files. Computational metrics (FLOPs, latency, FPS) are also tracked.

## Performance Settings

These should be present in training/inference code:
- `torch.backends.cudnn.benchmark = True` (fixed input sizes)
- `torch.set_float32_matmul_precision("high")` (CUDA optimization)
- Mixed precision via `torch.cuda.amp`
- `pin_memory=True` in DataLoader
- Multiprocessing for data loading with proper cleanup

## Updating This File

When the project changes (new model components, new config parameters, changed file structure), update the relevant sections above. The SKILL.md file contains the stable principles that rarely change — this file contains the specifics that evolve with the project.
