# Vision Transformer Documentation - Complete Summary

## 📄 Documents Created

### 1. **VISION_TRANSFORMER_DATA_FLOW.md** (571 lines, 32KB)
Comprehensive visual diagrams showing data transformation through the hierarchical ViT pipeline.

**Contents:**
- **Diagram 1**: End-to-end pipeline showing all 3 stages with spatial resolution tracking
- **Diagram 2**: Deep dive into a single Window Transformer block with:
  - LayerNorm → Window Attention mechanics
  - Window Partition operation with visual representation
  - Attention computation per window (Q/K/V, scaled dot-product, relative position bias)
  - Window Reverse to reconstruct spatial features
  - MLP feedforward applied per spatial position
  - Residual connections and DropPath
- **Summary Diagrams**: ASCII art showing both high-level and detailed views
- **Computing Resources**: Complexity analysis and efficiency metrics
- **Architectural Insights**: Why hierarchical design, window attention, global pooling

### 2. **VISION_TRANSFORMER_QUICK_REFERENCE.md** (240 lines, 6.8KB)
Quick lookup guide for implementation details and troubleshooting.

**Contents:**
- Architecture overview and configuration
- Data flow checklist with input/output shapes
- Window attention mechanics (partition, attention, reverse)
- Stage design pattern
- Hyperparameters table
- Stochastic depth and relative position bias explanation
- Integration with downstream modules (Cross-Attention)
- FLOPs and memory usage estimates
- Common issues and solutions
- Testing instructions
- Advanced modifications

---

## 🎯 Key Visualizations

### Diagram 1: Overall Pipeline
```
Input [B, T, 3, 384, 384]  (context crop: 128*3.0)
   ↓ Stem (Conv 3→36, stride=4)
[B*T, 36, 96, 96]
   ↓ Stage 1 (8×8 windows, 2 layers)
[B*T, 36, 96, 96]  ← 144 windows
   ↓ Stage 2 (4×4 windows, 4 layers, downsample)
[B*T, 36, 48, 48]  ← 144 windows
   ↓ Stage 3 (2×2 windows, 5 layers, downsample)
[B*T, 288, 24, 24]  ← 144 windows
   ↓ Stage 4 (global window, 7 layers, downsample)
[B*T, 36, 12, 12]  ← 1 window with 144 tokens
   ↓ Global Avg Pool
[B*T, 36]
   ↓ LayerNorm + Projection
Output [B, T, 128]
```

### Diagram 2: Window Attention Block Deep Dive
```
Input [B*T, 36, 48, 48]  (Stage 2 example)
   ↓
[1] Reshape for spatial ops: [B*T, 48, 48, 36]
   ↓
[2] Window Partition (4×4): [144, 16, 36]  (144 windows of 16 tokens, 36 channels)
   ↓
[3] Window Attention: Within each 4×4 window, self-attention
     - Q/K/V projection
     - Scaled dot-product with 2 heads
     - Relative position bias (49×2 learned params)
     - Output: [144, 16, 36]
   ↓
[4] Window Reverse: [B*T, 48, 48, 36]
   ↓
[5] Residual + DropPath
   ↓
[6] MLP (applied per spatial position): [B*T, 48, 48, 36]
   ↓
[7] Residual + DropPath
   ↓
Output [B*T, 36, 48, 48]
```

---

## 📊 Data Dimension Tracking

| Stage | Input Shape | Output Shape | Window | Tokens/Win | Layers | Heads |
|-------|---|---|---|---|---|---|
| Stem | [B*T, 3, 384, 384] | [B*T, 36, 96, 96] | — | — | — | — |
| Stage1 | [B*T, 36, 96, 96] | [B*T, 36, 96, 96] | 8×8 | 64 | 2 | 2 |
| Stage2 | [B*T, 36, 48, 48] | [B*T, 36, 48, 48] | 4×4 | 16 | 4 | 2 |
| Stage3 | [B*T, 36, 24, 24] → [B*T, 288, 24, 24] | [B*T, 288, 24, 24] | 2×2 | 4 | 5 | 16 |
| Stage4 | [B*T, 288, 12, 12] → [B*T, 36, 12, 12] | [B*T, 36, 12, 12] | global | 144 | 7 | 2 |
| Pool | [B*T, 36, 12, 12] | [B*T, 36] | — | — | — | — |
| Proj | [B, T, 36] | [B, T, 128] | — | — | — | — |

---

## ⚡ Key Efficiency Metrics

### Attention Complexity Comparison
```
Stage 2 (32×32, 4×4 windows):
   Global attention:   O(1024²) = ~1M operations
   Window attention:   O(64×16²) = ~16K operations
   Speedup: 61×
   Memory savings: 61×

Stage 3 (16×16, 2×2 windows):
   Global attention:   O(256²) = ~65K operations
   Window attention:   O(64×4²) = ~1K operations
   Speedup: 65×
   
Stage 4 (8×8, global):
   Single global window anyway, so no savings
   But: only happens at very low resolution (8×8 = 64 tokens)
```

### Computational Budget
```
Per Frame (256×256 input):
  Total: ~360M FLOPs
  
Per Sequence (20 frames):
  Total: ~7.2B FLOPs
  
vs Dense Attention:
  Stage3 would be 1.6B → 4.4× slower
```

### Memory Usage
```
Activations: ~800MB (peak)
Parameters:  ~40MB
Total GPU:   ~900MB for B=1, T=20
```

---

## 🔧 Implementation Details

### Window Partition Logic
```python
Input: [B, H, W, C]  e.g., [2, 32, 32, 128]
1. view(B, H//ws, ws, W//ws, ws, C)     # Group into windows
2. permute(0, 1, 3, 2, 4, 5)            # Rearrange
3. view(-1, ws*ws, C)                   # Flatten windows
Output: [num_windows*B, ws², C]  e.g., [128, 16, 128]
```

### Window Attention per Head
```
For each window independently:
1. Q/K/V = linear(x) → [N_tokens, head_dim]
2. scale = 1/√(head_dim)
3. attention = softmax((Q_scaled @ K^T))
4. attention += relative_position_bias
5. output = attention @ V
6. concatenate across heads
```

### Relative Position Bias
```
# Only for fixed windows (not global)
# Shape: [(2*window_h-1) * (2*window_w-1), num_heads]
# Example for 4×4 window: 7×7 = 49 entries per head

# Encodes: relative position between any two tokens
#          learned weights per attention head
#          enables position awareness within window
```

---

## 🎓 Learning Resources

### Understanding Window Partitioning
- View the detailed ASCII diagram in VISION_TRANSFORMER_DATA_FLOW.md
- Shows 32×32 grid tiled into 8×8=64 4×4 windows
- Each window becomes a sequence of 16 tokens

### Understanding Attention Mechanism
- Follows standard multi-head self-attention
- Unique: relative position bias (learnable per head)
- Unique: windowed scope (not global)
- See step [3] in Diagram 2

### Understanding Hierarchical Design
- Stage 1: 64 windows → local detail extraction
- Stage 2: 64 windows → medium-range dependencies
- Stage 3: 1 global window → full-image context
- Progressive downsampling balances efficiency + expressiveness

---

## 🐛 Debugging Guide

### Common Issues

**Issue**: `Window not divisible by H/W`
- **Cause**: Input shape not multiple of window size
- **Fix**: Ensure (H % window_size == 0) and (W % window_size == 0)

**Issue**: NaN in attention
- **Cause**: Relative position index not initialized
- **Fix**: Block calls `init_relative_position_bias(H_s, W_s)` in forward

**Issue**: Out of memory
- **Cause**: Batch size too large or sequence too long
- **Fix**: Reduce batch_size or sequence_length

**Issue**: Slow inference
- **Cause**: Stage 3 global attention is the bottleneck
- **Fix**: Use smaller input resolution or skip Stage 3

### Debug Shapes
```python
print(f"Input:           {x.shape}")  # [B, T, 3, 384, 384]
print(f"Stem:            {x.shape}")  # [B*T, 36, 96, 96]
print(f"Stage 1:         {x.shape}")  # [B*T, 36, 96, 96]
print(f"Stage 2:         {x.shape}")  # [B*T, 36, 48, 48]
print(f"Stage 3:         {x.shape}")  # [B*T, 288, 24, 24]
print(f"Stage 4:         {x.shape}")  # [B*T, 36, 12, 12]
print(f"Global Pool:     {x.shape}")  # [B*T, 36]
print(f"Final:           {x.shape}")  # [B, T, 128]
```

---

## 📚 Document Navigation

### For Architecture Overview
→ Start with **VISION_TRANSFORMER_QUICK_REFERENCE.md** "Architecture Overview" section

### For Data Flow Understanding
→ Read **VISION_TRANSFORMER_DATA_FLOW.md** "Diagram 1: Overall Pipeline"

### For Detailed Implementation
→ Study **VISION_TRANSFORMER_DATA_FLOW.md** "Diagram 2: Single Stage Deep Dive"

### For Quick Lookup
→ Use **VISION_TRANSFORMER_QUICK_REFERENCE.md** "Data Flow Checklist"

### For Troubleshooting
→ Check **VISION_TRANSFORMER_QUICK_REFERENCE.md** "Common Issues & Solutions"

### For Modifications
→ See **VISION_TRANSFORMER_QUICK_REFERENCE.md** "Advanced: Modifying Architecture"

---

## 🔗 Integration Context

This Vision Transformer serves as the **image feature extractor** in the multimodal pipeline:

```
Image Input [B, T, 3, 256, 256]
            ↓
    ViT_Hierarchical (THIS DOCUMENTATION)
            ↓
    [B, T, 128] ← Image Features (Key/Value)
            ↓
Cross-Attention Module
            ↓
Fused with Motion Features (Query)
            ↓
Pedestrian Behavior Predictions
```

The 128-dimensional embeddings represent high-level spatial features about:
- Pedestrian appearance
- Clothing/body shape
- Surrounding context
- Scene composition

These are then fused with temporal motion information for behavior prediction.

---

**Documentation Coverage:**
- ✅ Architecture overview
- ✅ Data flow end-to-end
- ✅ Window attention deep dive
- ✅ Computational complexity
- ✅ Memory requirements
- ✅ Hyperparameter reference
- ✅ Debugging guide
- ✅ Integ
