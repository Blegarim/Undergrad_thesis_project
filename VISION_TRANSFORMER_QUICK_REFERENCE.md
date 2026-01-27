# Vision Transformer - Quick Reference Guide

## File Location
📍 `/models/Vision_Transformer.py`

## Architecture Overview

### Key Components
1. **Stem**: Conv2d(3→36, kernel=7, stride=4) → Initial feature extraction (4x downsampling)
2. **4 Hierarchical Stages**: Progressive resolution reduction with adaptive capacity
3. **Window Attention**: Efficient local attention within fixed windows
4. **Global Avg Pool**: Collapses spatial dimensions
5. **Frame Projection**: Linear(36→128) for cross-attention compatibility

### Configuration (from config.py)
```python
vit_args_config(
    in_channels=3,
    stage_dims=[36, 36, 288, 36],      # Features per stage (4 stages)
    layer_nums=[2, 4, 5, 7],           # Transformer blocks per stage
    head_nums=[2, 2, 16, 2],           # Attention heads per stage
    window_size=[8, 4, 2, None],       # Window size per stage (None=global)
    mlp_ratio=[4, 4, 4, 4],            # MLP expansion ratio
    d_model=128,                       # Final embedding dimension
    drop_path=0.15,                    # Progressive stochastic depth
    attn_dropout=0.15,
    proj_dropout=0.15,
    dropout=0.15
)
```

**Note:** These are the production configs. ViT_Hierarchical class defaults are different.

## Data Flow Checklist

### Input → Output
```
✓ Input:  [B, T, 3, 384, 384]     B=batch, T=time steps (context crop: 128*3.0)
  ↓
✓ Stem:   [B*T, 36, 96, 96]       4x downsampling, 36 channels
  ↓
✓ Stage1: [B*T, 36, 96, 96]       8×8 windows, 2 layers
  ↓
✓ Stage2: [B*T, 36, 48, 48]       4×4 windows, 4 layers (after downsampling)
  ↓
✓ Stage3: [B*T, 288, 24, 24]      2×2 windows, 5 layers (peak capacity, after downsampling)
  ↓
✓ Stage4: [B*T, 36, 12, 12]       Global window, 7 layers (after downsampling)
  ↓
✓ Pool:   [B*T, 36]               Global average pooling
  ↓
✓ Norm:   [B*T, 36]               LayerNorm
  ↓
✓ Reshape:[B, T, 36]              Unflatten batch
  ↓
✓ Proj:   [B, T, 128]             Linear(36→128)
  ↓
✓ Output: [B, T, 128]             ← Ready for Cross-Attention
```

## Window Attention Mechanics

### Window Partition
```python
window_partition(x, window_size)
  Input:  [B, H, W, C]  e.g., [2, 32, 32, 128]
  Output: [num_windows*B, window_size², C]  e.g., [128, 16, 128]
  
  Where num_windows = (H/window_size) × (W/window_size)
                    = (32/4) × (32/4) = 64 windows
```

### Attention per Window
```python
WindowAttention(dim, window_size, num_heads)
  • Each window = independent computation
  • Complexity: O(window_size²) NOT O(total_tokens²)
  • Relative position bias: Learned per-head weights
  • Key advantage: 61× more efficient than global attention
```

### Window Reverse
```python
window_reverse(windows, window_size, H, W)
  Input:  [num_windows*B, window_size², C]  e.g., [128, 16, 128]
  Output: [B, H, W, C]  e.g., [2, 32, 32, 128]
  
  Reverses the partitioning, fusing attended features back to spatial map
```

## Stage Design Pattern

Each stage contains:
```
Stage = {
  'down_sample': Conv2d (or Identity for stage 0)
  'block': ModuleList of N WindowTransformerBlocks
}

WindowTransformerBlock structure:
  1. LayerNorm(dim)
  2. WindowAttention(dim, num_heads, window_size)
  3. DropPath (stochastic depth)
  4. Residual connection
  5. LayerNorm(dim)
  6. MLP(dim → dim*mlp_ratio → dim)
  7. DropPath (stochastic depth)
  8. Residual connection
```

## Key Hyperparameters

| Parameter | Stage1 | Stage2 | Stage3 | Effect |
|-----------|--------|--------|--------|--------|
| **dim** | 64 | 128 | 256 | Feature capacity |
| **window_size** | 8 | 4 | global | Attention scope (8×8=64, 4×4=16, 16×16=256 tokens) |
| **num_heads** | 2 | 4 | 8 | Attention parallelism |
| **layer_nums** | 2 | 4 | 6 | Transformer depth |
| **mlp_ratio** | 4 | 4 | 4 | FFN hidden dim = dim×mlp_ratio |

## Stochastic Depth (DropPath)

```python
# Progressive schedule across all blocks
dpr = torch.linspace(0, drop_path_rate, total_blocks)

Example for drop_path=0.1:
  Block 1 (Stage1): dpr ≈ 0.01   (low dropout)
  ...
  Block 12 (Stage3): dpr ≈ 0.10  (high dropout)
  
Why? Later blocks have more parameters, benefit from regularization
```

## Relative Position Bias

```python
# Only for fixed (non-global) windows
relative_position_bias_table: 
  Shape: [(2*Wh-1) * (2*Ww-1), num_heads]
  
Example for 4×4 window:
  (2*4-1) * (2*4-1) = 7×7 = 49 entries per head
  
Purpose: Encode spatial locality - tokens closer together get higher bias
```

## Integration with Motion Encoder & Cross-Attention

```python
# From main forward pipeline:
ViT Output [B, T, 128]
           ↓
        Key/Value
           ↓
Cross-Attention Module ← fused with motion features (Query)
           ↓
        Output: Actions, Looks, Crosses predictions
```

## Computational Complexity

### FLOPs Estimate (Per Frame, 256×256 input)
```
Stem:     ~13M FLOPs
Stage 1:  ~50M FLOPs
Stage 2:  ~100M FLOPs
Stage 3:  ~200M FLOPs
Pool/Proj: <1M FLOPs
─────────────────────
Total:    ~360M FLOPs per frame
          ~7.2B FLOPs per 20-frame sequence

For comparison:
  Dense attention stage 3: ~1.6B FLOPs (4.4× slower)
```

### Memory Usage (Batch Size=1, Seq Len=20)
```
Activations: ~800MB (peak at Stage 2/3)
Parameters:  ~40MB
Optimizer:   ~80MB (Adam state)
───────────────────────
Total:       ~900MB GPU memory
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Window not divisible by H/W | Bad config | Ensure (H%window_size==0) and (W%window_size==0) |
| NaN in attention | Numerical instability | Check relative_position_index initialization |
| OOM on large batches | Too many tokens | Reduce batch size or frame sequence length |
| Slow inference | Global attention bottleneck | Move to lower resolution or use smaller windows |

## Testing

```bash
# Quick test (from main block)
python models/Vision_Transformer.py

# With FLOPs analysis
x = torch.randn(1, 10, 3, 480, 480)
vit = ViT_Hierarchical(...)
out = vit(x)  # [1, 10, 128]
flops = FlopCountAnalysis(vit, x)
```

## Debug Shapes

Adding prints to forward pass:
```python
print(f"[Stem] -> {x.shape}")              # [B*T, 64, 64, 64]
print(f"[Stage1] -> {x.shape}")            # [B*T, 64, 64, 64]
print(f"[Stage2] -> {x.shape}")            # [B*T, 128, 32, 32]
print(f"[Stage3] -> {x.shape}")            # [B*T, 256, 16, 16]
print(f"[Global Avg Pool] -> {x.shape}")   # [B*T, 256]
print(f"[Reshape] -> {x.shape}")           # [B, T, 256]
print(f"[Projection] -> {x.shape}")        # [B, T, 128]
```

## Advanced: Modifying Architecture

### Change final embedding dimension
```python
d_model=256  # Instead of 128
# frame_proj automatically adapts: Linear(256→256) = Identity
```

### Use all global attention
```python
window_size=["global", "global", "global"]
# More expressive but slower - use only for small inputs
```

### Add more stages
```python
stage_dims=[64, 128, 256, 512],
layer_nums=[2, 4, 6, 8],
head_nums=[2, 4, 8, 16],
window_size=[8, 4, 2, "global"]
# Deeper but slower - consider computational budget
```

---

**Generated for**: Pedestrian Behavior Prediction with Hierarchical ViT  
**Last Updated**: Jan 22, 2026
