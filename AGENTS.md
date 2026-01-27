# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this repository.

## Project Overview

This is an undergraduate thesis project focused on pedestrian behavior prediction using a multimodal deep learning approach. The project combines:
- Vision Transformer (ViT) for visual feature extraction
- Motion Encoder for temporal motion patterns
- Cross-Attention Module for multimodal fusion
- Ensemble Model for final predictions

The system predicts pedestrian actions (crossing/not crossing), looks, and crossing behaviors from video sequences.

## Build/Environment Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# The project uses PyTorch with CUDA support
# Verify CUDA availability: python -c "import torch; print(torch.cuda.is_available())"
```

### Running Tests
```bash
# Run full test suite
python test.py

# Run training
python train.py

# Run inference on video
python main.py

# Check label distribution
python label_count.py
```

### Single Test Commands
```bash
# Test specific model components (modify test.py accordingly):
python -c "from models.Vision_Transformer import ViT_Hierarchical; print('ViT import OK')"
python -c "from models.Motion_Encoder import MotionEncoder; print('Motion Encoder import OK')"
python -c "from models.Cross_Attention_Module import CrossAttentionModule; print('Cross Attention import OK')"
python -c "from models.Unified_Module import EnsembleModel; print('Ensemble Model import OK')"
```

## Code Style Guidelines

### Import Organization
- Standard library imports first (os, sys, time, etc.)
- Third-party imports next (torch, torchvision, numpy, sklearn, etc.)
- Local imports last (from models.*, from scripts.*, from config)
- Group related imports together
- Use absolute imports for local modules

```python
# Standard library
import os
import time
import gc

# Third-party
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

# Local imports
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from config import vit_args_config
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `MotionEncoder`, `CrossAttentionModule`)
- **Functions/Methods**: snake_case (e.g., `window_partition`, `compute_flops`)
- **Variables**: snake_case (e.g., `batch_size`, `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `EMBEDDING_DIM`, `NUM_EPOCHS`)
- **Private methods**: prefix with underscore (e.g., `_inverse_class_weights`)

### Type Hints
- Use type hints for function signatures and important variables
- Import typing from `typing` module when needed
- Focus on return types and parameter types for complex functions

```python
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate model performance and return metrics dictionary."""
    pass
```

### Error Handling
- Use specific exceptions when possible
- Include descriptive error messages
- Use assertions for validating inputs and configurations
- Handle CUDA memory errors gracefully with try-catch blocks

```python
assert os.path.exists(model_path), f"Model not found: {model_path}"
if device.type == "cuda":
    torch.cuda.empty_cache()
```

### Documentation
- Use docstrings for classes and complex functions
- Include parameter descriptions and return types
- Add inline comments for complex logic or mathematical operations
- Use TODO comments for future improvements

### Code Structure
- Keep functions focused and under 50 lines when possible
- Use helper functions for complex operations
- Group related functionality together
- Maintain consistent indentation (4 spaces)
- Line length under 120 characters preferred

### PyTorch Specific Guidelines
- Use `.to(device)` for tensor/model device placement
- Use `torch.no_grad()` context for inference
- Use `model.eval()` and `model.train()` appropriately
- Handle mixed precision with `torch.cuda.amp.autocast()`
- Use `torch.set_float32_matmul_precision("high")` for CUDA optimizations

### Data Processing
- Use transforms from `torchvision.transforms` for image preprocessing
- Normalize images using ImageNet standards: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- Use LMDB for efficient data loading with large datasets
- Implement proper collate functions for batching

### Configuration Management
- Keep configuration parameters in `config.py`
- Use configuration functions for model arguments
- Maintain consistent parameter naming across files
- Use descriptive variable names for hyperparameters

### Testing and Validation
- Include comprehensive metrics (accuracy, F1, AUC, precision, recall)
- Use separate test/validation splits
- Log results to CSV files for analysis
- Include computational metrics (FLOPs, latency, FPS)

### Memory Management
- Use `gc.collect()` and `torch.cuda.empty_cache()` after processing chunks
- Monitor memory usage with `psutil`
- Use multiprocessing for data loading with proper cleanup
- Handle large datasets in chunks to avoid memory overflow

## File Organization

```
models/           # Model architecture definitions
├── __init__.py
├── Vision_Transformer.py
├── Motion_Encoder.py
├── Cross_Attention_Module.py
└── Unified_Module.py

scripts/          # Data processing and utilities
├── __init__.py
├── lmdb_dataset.py
├── pedestrian_detection.py
└── preprocess_*.py

config.py         # Configuration parameters
train.py          # Training script
test.py           # Testing/evaluation script
main.py           # Inference on video
requirements.txt  # Dependencies
```

## Common Patterns

### Model Initialization
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

### Data Loading
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

### Training Loop Pattern
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

## Performance Considerations

- Use mixed precision training (`torch.cuda.amp`) for CUDA
- Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes
- Use `pin_memory=True` for faster data transfer to GPU
- Implement proper multiprocessing for data loading
- Monitor and manage GPU/CPU memory usage
- Use gradient scaling for numerical stability

## Debugging Tips

- Check tensor shapes frequently with `.shape`
- Use `torch.sum(tensor).item()` for debugging scalar values
- Verify device placement with `.device`
- Use `print(f"Variable: {variable.shape}, dtype: {variable.dtype}")` for debugging
- Monitor GPU memory with `torch.cuda.memory_summary()` when needed