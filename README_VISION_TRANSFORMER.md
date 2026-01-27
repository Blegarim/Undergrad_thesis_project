# Vision Transformer Documentation Index

## 📚 Complete Documentation Package

Three comprehensive documents describing the Hierarchical Vision Transformer architecture with data flow diagrams and implementation details.

### 📖 Document Overview

#### 1. **VISION_TRANSFORMER_DATA_FLOW.md** 
**Purpose**: Visual diagrams and architecture explanation  
**Length**: 571 lines

**Includes:**
- Diagram 1: Full pipeline from input to output with spatial resolution tracking
- Diagram 2: Single block deep dive (window partition → attention → reverse)
- ASCII art visualization of both high-level and detailed views
- Computing resource impact analysis
- Attention complexity comparison (61× efficiency gain from windowing)

**Best for**: Understanding how data flows through the model

---

#### 2. **VISION_TRANSFORMER_QUICK_REFERENCE.md**
**Purpose**: Quick lookup and implementation guide  
**Length**: 240 lines

**Includes:**
- Architecture overview and default configuration
- Data flow checklist with all tensor shapes
- Window attention mechanics explanation
- Stage design pattern
- Hyperparameter table (dim, window_size, heads per stage)
- Stochastic depth and relative position bias explanation
- FLOPs and memory usage estimates
- Common issues and solutions
- Testing instructions
- Debug shapes for validation
- Advanced modifications guide

**Best for**: Quick references and troubleshooting

---

#### 3. **DOCUMENTATION_SUMMARY.md**
**Purpose**: Navigation guide and consolidated overview  
**Length**: ~250 lines

**Includes:**
- Summary of all three documents
- Key visualizations and metrics
- Data dimension tracking table
- Efficiency metrics and complexity analysis
- Implementation details (window partition, attention, bias)
- Learning resources and debugging guide
- Document navigation recommendations

**Best for**: Navigating the full documentation set

---

## 🗺️ Quick Navigation

### "I want to understand the architecture"
→ Read VISION_TRANSFORMER_DATA_FLOW.md - Diagram 1

### "I want to understand one block in detail"
→ Read VISION_TRANSFORMER_DATA_FLOW.md - Diagram 2

### "I need to troubleshoot an issue"
→ Check VISION_TRANSFORMER_QUICK_REFERENCE.md - Common Issues & Solutions

### "I want to modify the architecture"
→ See VISION_TRANSFORMER_QUICK_REFERENCE.md - Advanced: Modifying Architecture

### "I need the tensor shapes"
→ Use VISION_TRANSFORMER_QUICK_REFERENCE.md - Data Flow Checklist

### "I want to see efficiency metrics"
→ Find VISION_TRANSFORMER_DATA_FLOW.md - Computing Resource Impact

---

## 🎯 Key Takeaways

### Data Transformation Pipeline
```
[B, T, 3, 384, 384]              # Batch of T frames (context crop: 128*3.0)
        ↓ Stem (Conv 3→36, stride=4)
[B*T, 36, 96, 96]                # 4x downsampling
        ↓ Stage 1 (8×8 windows, 2 layers)
[B*T, 36, 96, 96]                # 144 windows
        ↓ Stage 2 (4×4 windows, 4 layers, downsample)
[B*T, 36, 48, 48]                # 144 windows
        ↓ Stage 3 (2×2 windows, 5 layers, downsample)
[B*T, 288, 24, 24]               # 144 windows
        ↓ Stage 4 (global window, 7 layers, downsample)
[B*T, 36, 12, 12]                # 1 window
        ↓ Global pooling
[B*T, 36]
        ↓ LayerNorm + Projection
[B, T, 128]                      # Final embeddings (d_model=128)
```

### Efficiency Advantages
- **Window attention**: Significantly faster than global attention on large feature maps
- **Hierarchical stages**: Early stages process low-dim features, middle stage peaks at high dimension, late stage reduces back down
- **Progressive stochastic depth**: Regularization schedule from 0.0→0.15 across all 18 blocks (2+4+5+7=18)

### Key Components
1. **Stem**: Conv2d(3→36, k=7, stride=4) - 4x downsampling
2. **4 Stages**: Increasing then decreasing channels (36→36→288→36), decreasing resolution, expanding receptive field
   - Stage 1: 36 channels, 8×8 windows, 2 layers
   - Stage 2: 36 channels, 4×4 windows, 4 layers
   - Stage 3: 288 channels (peak), 2×2 windows, 5 layers
   - Stage 4: 36 channels, global window, 7 layers
3. **Window attention**: Local attention within fixed windows (8×8, 4×4, 2×2, global)
4. **Global pooling**: Collapse spatial dimensions (HxW → 1)
5. **Projection**: Map to embedding dimension (36→128)

---

## 📊 Architecture at a Glance

| Component | Input | Output | Windows | Heads | Layers |
|-----------|-------|--------|---------|-------|--------|
| Stem | [B*T, 3, 384, 384] | [B*T, 36, 96, 96] | — | — | — |
| Stage 1 | [B*T, 36, 96, 96] | [B*T, 36, 96, 96] | 8×8 (144) | 2 | 2 |
| Stage 2 | [B*T, 36, 48, 48] | [B*T, 36, 48, 48] | 4×4 (144) | 2 | 4 |
| Stage 3 | [B*T, 36, 24, 24] | [B*T, 288, 24, 24] | 2×2 (144) | 16 | 5 |
| Stage 4 | [B*T, 288, 12, 12] | [B*T, 36, 12, 12] | global (1) | 2 | 7 |

---

## 🔍 Understanding Window Attention

### Window Partition (32×32 → 4×4 windows)
```
32×32 feature map tiled into 4×4 window regions
(32÷4) × (32÷4) = 8×8 = 64 windows
Each window: 4×4 = 16 tokens
Total tokens: 64 × 16 = 1024 (same as 32×32)

Benefit: Attention only within 16 tokens (O(256)) 
         instead of all 1024 tokens (O(1M))
```

### Window Attention (Scaled Dot-Product)
```
For each window's 16 tokens:
1. Q/K/V projection: 16 tokens × 128 dims → Q,K,V
2. 4 attention heads process in parallel
3. Scale by 1/√32 (head dimension)
4. Attention = softmax(Q @ K^T)
5. Add relative position bias (learnable)
6. Output = Attention @ V
```

### Window Reverse (Reconstruct spatial layout)
```
Reverse operation of window partition
64 windows of 16 tokens → 1024 spatial positions
Reconstructed: [B*T, 32, 32, 128]
Now ready for MLP and next layer
```

---

## ⚡ Performance Characteristics

### Speed
- Stage 1: ~50M FLOPs (low resolution, local patterns)
- Stage 2: ~100M FLOPs (medium resolution, contextual features)
- Stage 3: ~200M FLOPs (high resolution, full context)
- **Total per frame**: ~360M FLOPs (256×256 input)
- **Total per sequence**: ~7.2B FLOPs (20 frames)

### Memory
- Activations peak at Stage 2/3: ~800MB
- Parameters: ~40MB
- **Total GPU requirement**: ~900MB for B=1, T=20

### Comparison to Dense Attention
- 61× speedup on 32×32 (Stage 2)
- 4.4× faster on 16×16 (Stage 3) due to lower resolution
- Global pooling ensures fixed output dimension

---

## 🔧 Implementation Reference

### Default Configuration (from config.py)
```python
vit_args_config(
    in_channels=3,
    stage_dims=[36, 36, 288, 36],
    layer_nums=[2, 4, 5, 7],
    head_nums=[2, 2, 16, 2],
    window_size=[8, 4, 2, None],
    mlp_ratio=[4, 4, 4, 4],
    d_model=128,  # Output dimension (unified across all modules)
    drop_path=0.15,
    attn_dropout=0.15,
    proj_dropout=0.15,
    dropout=0.15
)
```

**Note:** The configuration from `config.py` is the production config used in the project. The ViT_Hierarchical class defaults are different (3 stages, lower dimensions, lower dropout). When using the model, always use the config from `vit_args_config()` function.

### Usage
```python
# Input: batch of video frames
x = torch.randn(2, 20, 3, 256, 256)  # B=2, T=20

vit = ViT_Hierarchical()
output = vit(x)  # [2, 20, 128]
```

---

## 🐛 Troubleshooting Quick Links

| Problem | Solution | Reference |
|---------|----------|-----------|
| OOM error | Reduce batch size or T | QUICK_REF - Memory Usage |
| NaN in attention | Check position bias init | QUICK_REF - Common Issues |
| Wrong output shape | Verify input dimensions | QUICK_REF - Debug Shapes |
| Slow inference | Use smaller window size | DATA_FLOW - Computing Impact |
| Window not divisible | Fix H/W dimensions | QUICK_REF - Common Issues |

---

## 📈 Architecture Progression

### Design Philosophy
1. **Stem**: Aggressive initial downsampling (4×) to reduce spatial complexity
2. **Early stages**: Small windows (8×8), few layers → process local patterns efficiently
3. **Middle stages**: Medium windows (4×4), more layers → capture mid-range context
4. **Late stages**: Global window, deepest network → full image understanding
5. **Pooling**: Collapse to single vector regardless of input size

### Why This Works
- Progressive resolution reduction matches visual hierarchy
- Window sizes decrease with feature map size → maintains relative context
- More layers in deeper stages → more capacity where needed
- Stochastic depth increases with depth → regularization where needed

---

## 📝 File Organization

```
models/
  └── Vision_Transformer.py          ← Implementation
  
Documentation/
  ├── VISION_TRANSFORMER_DATA_FLOW.md        ← This package
  ├── VISION_TRANSFORMER_QUICK_REFERENCE.md  ← This package
  ├── DOCUMENTATION_SUMMARY.md               ← This package
  └── README_VISION_TRANSFORMER.md           ← This file
```

---

## 🎓 Learning Path

**Beginner**: Start with "Architecture Overview" in QUICK_REF, then Diagram 1 in DATA_FLOW

**Intermediate**: Study Diagram 2 (window attention deep dive) in DATA_FLOW

**Advanced**: Read implementation details in QUICK_REF and modify architecture

**Expert**: Refer to Vision_Transformer.py source code with documentation as guide

---

## ✅ Documentation Checklist

- [x] Architecture overview
- [x] End-to-en
