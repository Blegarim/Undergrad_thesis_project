# Vision Transformer Hierarchical Architecture - Data Flow Diagrams

## Diagram 1: Overall Pipeline Data Transformation

```
INPUT: [B, T, C=3, H=256, W=256]
│
├─ Reshape to Process Frames in Batch
│  └─ [B*T, C=3, H=256, W=256]
│
├─ STEM: Conv2d(3→36, k=7, stride=4, padding=3) + BatchNorm + GELU
│  └─ [B*T, D=36, H=64, W=64]          (H/4, W/4)
│
├─────────────────────────────────────────────────────────────────┐
│  STAGE 1: dim=36, layers=2, heads=2, window_size=8             │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Identity (i=0)                                 │
│  │  └─ [B*T, 36, 64, 64]                                       │
│  │                                                              │
│  ├─ Block 1: Window Attention (8×8)                            │
│  │  └─ [B*T, 36, 64, 64]                                       │
│  │                                                              │
│  └─ Block 2: Window Attention (8×8)                            │
│     └─ [B*T, 36, 64, 64]                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────────────────────────────┐
│  STAGE 2: dim=36, layers=4, heads=2, window_size=4            │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(36→36, k=3, stride=2, padding=1)       │
│  │  └─ [B*T, 36, 32, 32]          (H/2, W/2)                  │
│  │                                                              │
│  ├─ Blocks 1-4: Window Attention (4×4)                         │
│  │  └─ [B*T, 36, 32, 32]                                       │
│  │                                                              │
│  └─ (Stochastic depth progressively increases)                 │
│     └─ [B*T, 36, 32, 32]                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────────────────────────────┐
│  STAGE 3: dim=288, layers=5, heads=16, window_size=2×2        │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(36→288, k=3, stride=2, padding=1)      │
│  │  └─ [B*T, 288, 16, 16]          (H/2, W/2)                 │
│  │                                                              │
│  ├─ Blocks 1-5: Window Attention (2×2 windows)                │
│  │  └─ Each block processes in 2×2 = 4-token windows         │
│  │  └─ [B*T, 288, 16, 16]                                     │
│  │                                                              │
│  └─ (Higher capacity for feature learning)                     │
│     └─ [B*T, 288, 16, 16]                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────────────────────────────┐
│  STAGE 4: dim=36, layers=7, heads=2, window_size="global"     │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(288→36, k=3, stride=2, padding=1)      │
│  │  └─ [B*T, 36, 8, 8]          (H/2, W/2)                    │
│  │                                                              │
│  ├─ Blocks 1-7: Window Attention (global window = full HxW)    │
│  │  └─ Each block sees all 8×8=64 tokens                      │
│  │  └─ [B*T, 36, 8, 8]                                        │
│  │                                                              │
│  └─ (Max stochastic depth at final blocks)                     │
│     └─ [B*T, 36, 8, 8]                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│
├─ Global Average Pooling: mean([2, 3])
│  └─ [B*T, 36]
│
├─ LayerNorm
│  └─ [B*T, 36]
│
├─ Reshape back to sequence
│  └─ [B, T, 36]
│
├─ Projection Linear(36→128)
│  └─ [B, T, 128]
│
OUTPUT: [B, T, 128]  ← Passed to Cross-Attention Module
```


═══════════════════════════════════════════════════════════════════
SPATIAL RESOLUTION TRACKING:
═══════════════════════════════════════════════════════════════════
Input Image:        384×384  (147,456 pixels) - context crop: 128*3.0
After Stem:         96×96    (9,216 pixels)   1/4 resolution
Stage 1:            96×96    (9,216 tokens)   8×8=64 tokens/window
Stage 2:            48×48    (2,304 tokens)   4×4=16 tokens/window
Stage 3:            24×24    (576 tokens)     2×2=4 tokens/window
Stage 4:            12×12    (144 tokens)     All tokens in 1 global window
After GAP:          1        (36 features)
Final:              1        (128 features)   ← Temporal features

═══════════════════════════════════════════════════════════════════
LAYER CONFIGURATION SUMMARY:
═══════════════════════════════════════════════════════════════════
Stage  │ In→Out Dim  │ Layers │ Heads │ Window │ Tokens │ Attention Type
───────┼─────────────┼────────┼───────┼────────┼────────┼──────────────────
   1   │ 36→36       │   2    │   2   │  8×8   │   64   │ Window (64 tokens)
   2   │ 36→36       │   4    │   2   │  4×4   │   16   │ Window (16 tokens)
   3   │ 36→288      │   5    │   16  │  2×2   │   4    │ Window (4 tokens)
   4   │ 288→36      │   7    │   2   │ global │  144   │ Global (144 tokens)
   GAP │ 36→36       │   -    │   -   │   -    │   1    │ Spatial pooling
   Proj│ 36→128      │   -    │   -   │   -    │   1    │ Linear projection
```

---

## Diagram 2: Single Stage Deep Dive - Data Flow Through One Window Transformer Block

### Example: Stage 2, Block 1 (Window Size = 4×4)

```
INPUT TO BLOCK: [B*T, 36, 48, 48]
│
│  ╔════════════════════════════════════════════════════════════════╗
│  ║               WINDOW TRANSFORMER BLOCK DETAILED                ║
│  ╚════════════════════════════════════════════════════════════════╝
│
├─ Step 0: Reshape for spatial operations
│  Input:  [B*T, 36, 48, 48]
│  Perm:   [B*T, 48, 48, 36]  ← [B, H, W, C]
│  └─ Now in format for window partitioning
│
│
├─ ╔════════════════════════════════════════════════════════════════╗
│  ║            FIRST RESIDUAL BLOCK: ATTENTION                     ║
│  ╚════════════════════════════════════════════════════════════════╝
│
│
├─ Step 1: LayerNorm
│  Input:  [B*T, 48, 48, 36]
│  Output: [B*T, 48, 48, 36]  (normalized features)
│
│
├─ Step 2: WINDOW PARTITION
│  ┌─────────────────────────────────────────────────────────────┐
│  │ Input: [B*T, 48, 48, 36]                                   │
│  │                                                              │
│  │ Configuration: window_size = 4×4                            │
│  │ Feature map: 48×48 → tiled into (48/4)×(48/4) = 12×12 grid│
│  │                                                              │
│  │ Step-by-step reshape:                                       │
│  │   view(B*T, 12, 4, 12, 4, 36)  ← group pixels into windows│
│  │   permute(0,1,3,2,4,5)           ← rearrange window tiles  │
│  │   view(-1, 16, 36)               ← flatten into windows    │
│  │                                                              │
│  │ Output: [num_windows*B*T, 16, 36]                          │
│  │  where num_windows = (48/4) × (48/4) = 144 windows        │
│  │        16 = 4×4 (tokens per window)                        │
│  │        36 = channel dimension                              │
│  │                                                              │
│  │ Example: Batch of 2, 144 windows each                      │
│  │   Input shape:  [2, 48, 48, 36]                           │
│  │   Output shape: [2*144=288, 16, 36]                       │
│  │                                                              │
│  │   Visual representation:                                    │
│  │   48×48 feature map (B=1 for clarity):                     │
│  │   ┌──────────────────────────────────┐                     │
│  │   │ W1│ W2│ W3│...│W12│             │                     │
│  │   ├──────────────────────────────────┤                     │
│  │   │W13│W14│W15│..│W24│             │                     │
│  │   │  ...                             │  Each 4×4 window   │
│  │   ├──────────────────────────────────┤  becomes [16, 36]  │
│  │   │W133│W134│W135│..│W144│         │  token sequence    │
│  │   └──────────────────────────────────┘                     │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────┘
│  └─ Output: [288, 16, 36]
│
│
├─ Step 3: WINDOW ATTENTION (Multi-Head Self-Attention)
│  ┌─────────────────────────────────────────────────────────────┐
│  │ Input: [288, 16, 36] (each window is independent)          │
│  │                                                              │
│  │ Configuration: num_heads=2, head_dim=36/2=18               │
│  │                                                              │
│  │ 3.1) Compute Q, K, V from linear projection:              │
│  │      QKV_proj(x) → [288, 16, 108]  (3 × 36 = 108)        │
│  │      Reshape: [288, 16, 3, 2, 18] → 3, 288, 2, 16, 18    │
│  │      Split:   Q,K,V each [288, 2, 16, 18]                │
│  │                 (batch, heads, tokens, head_dim)          │
│  │                                                              │
│  │ 3.2) Scaled Dot-Product Attention per head:               │
│  │      Scale factor: 1/√18 ≈ 0.236                          │
│  │      Q_scaled = Q × 0.236 → [288, 2, 16, 18]            │
│  │      Attention = softmax(Q_scaled @ K^T) → [288,2,16,16] │
│  │                  Each token attends to 16 tokens in window│
│  │                                                              │
│  │ 3.3) Apply relative position bias:                         │
│  │      rel_bias_table: [(2*4-1)²] × 2 = 49×2 learnable vals│
│  │      Maps 2D relative distances to bias per head           │
│  │      Attention += rel_pos_bias → [288, 2, 16, 16]        │
│  │                                                              │
│  │ 3.4) Weighted sum with values:                             │
│  │      Output = Attention @ V → [288, 2, 16, 18]           │
│  │      Reshape: [288, 16, 36]                               │
│  │                                                              │
│  │ Result: Each token in window attends to all other tokens  │
│  │         with position-aware weights                        │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────┘
│  └─ Output: [288, 16, 36]
│
│
├─ Step 4: WINDOW REVERSE (Reconstruct spatial layout)
│  ┌─────────────────────────────────────────────────────────────┐
│  │ Input: [288, 16, 36]  (288 windows of 16 tokens each)     │
│  │                                                              │
│  │ Reverse the window_partition operation:                    │
│  │   view(B*T, 12, 12, 4, 4, 36)  ← ungroup windows         │
│  │   permute(0,1,3,2,4,5)           ← rearrange back to grid │
│  │   view(B*T, 48, 48, 36)          ← flatten to spatial map │
│  │                                                              │
│  │ Example: [288, 16, 36] → [2, 48, 48, 36]                 │
│  │                                                              │
│  │ Attention applied within windows is now fused back         │
│  │ into the original spatial feature map                      │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────┘
│  └─ Output: [B*T, 48, 48, 36]
│
│
├─ Step 5: Residual Connection + DropPath
│  Input attention output: [2, 48, 48, 36]
│  Shortcut (from Step 0):  [2, 48, 48, 36]
│  DropPath applied to attention output (stochastic depth)
│  Output: [2, 48, 48, 36] + DropPath([2, 48, 48, 36])
│  └─ Output: [2, 48, 48, 36]
│
│
├─ ╔════════════════════════════════════════════════════════════════╗
│  ║            SECOND RESIDUAL BLOCK: MLP (FFN)                    ║
│  ╚════════════════════════════════════════════════════════════════╝
│
│
├─ Step 6: Reshape to token sequence for MLP
│  Input:  [B*T, 48, 48, 36]
│  Reshape: [B*T, 2304, 36]  ← 48×48=2304 spatial positions
│  └─ Each spatial position becomes a token
│
│
├─ Step 7: LayerNorm
│  Input:  [B*T, 2304, 36]
│  Output: [B*T, 2304, 36]
│
│
├─ Step 8: MLP (Feed-Forward Network)
│  ┌─────────────────────────────────────────────────────────────┐
│  │ Configuration: MLP(36, hidden_dim=144, dropout=0.15)       │
│  │                (mlp_ratio=4, so 36×4=144)                  │
│  │                                                              │
│  │ Forward pass (applied per token):                           │
│  │   x: [B*T, 2304, 36]                                       │
│  │   fc1(x): [B*T, 2304, 36] → [B*T, 2304, 144]             │
│  │   GELU activation                                           │
│  │   Dropout: [B*T, 2304, 144]                               │
│  │   fc2(x): [B*T, 2304, 144] → [B*T, 2304, 36]            │
│  │                                                              │
│  │ Effectively: each token processed through                   │
│  │   shared MLP(36 → 144 → 36)                               │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────┘
│  └─ Output: [B*T, 2304, 36]
│
│
├─ Step 9: Dropout
│  Input:  [B*T, 2304, 36]
│  Output: [B*T, 2304, 36]
│
│
├─ Step 10: Residual Connection + DropPath
│  MLP output:      [B*T, 2304, 36]
│  Shortcut (token view of Step 5): [B*T, 2304, 36]
│  DropPath applied to MLP output
│  Output: [B*T, 2304, 36] + DropPath([B*T, 2304, 36])
│  └─ Output: [B*T, 2304, 36]
│
│
├─ Step 11: Reshape back to spatial
│  Input:  [B*T, 2304, 36]
│  Reshape: [B*T, 36, 48, 48]  ← Permute: [B*T, C, H, W]
│  └─ Output: [B*T, 36, 48, 48]
│
│
OUTPUT FROM BLOCK: [B*T, 36, 48, 48]


═══════════════════════════════════════════════════════════════════
COMPUTATION SUMMARY FOR SINGLE BLOCK:
═══════════════════════════════════════════════════════════════════
Window Size:     4×4 = 16 tokens per window
Num Windows:     (32/4) × (32/4) = 64 windows per frame
Attention Scope: Each token attends to 16 tokens (within window)
                 NOT to all 1024 tokens (saves computation)
                 
MLP Scope:       Applied to all 1024 tokens independently
                 Each token has same MLP weights

Parameters:
   - LayerNorm1:     36 (scale + bias)
   - Attention Q/K/V: 36 × 108 = 3,888
   - Attention Proj:  36 × 36 = 1,296
   - Attention Bias:  (2×4-1)² × 2 = 49 × 2 = 98
   - LayerNorm2:      36
   - MLP fc1:         36 × 144 = 5,184
   - MLP fc2:         144 × 36 = 5,184
   
Total per block: ~15,600 parameters (rough estimate)

Key Insight: Window attention limits computation to O(HW) instead of O((HW)²)
            for the quadratic attention operation
═══════════════════════════════════════════════════════════════════
```

---

## Key Architectural Insights

### Why Hierarchical Stages?
1. **Stage 1** (8×8 windows): Process local details, reduce spatial dims
2. **Stage 2** (4×4 windows): Capture medium-range dependencies  
3. **Stage 3** (2×2 windows): Peak capacity (288 dims) for learning rich features
4. **Stage 4** (global): Capture full image context and reduce dimensions back

### Why Window Attention?
- **Complexity**: O(N) instead of O(N²) where N is number of tokens
- **Memory**: Quadratic attention only within windows (64, 16, 4, or 64 tokens)
- **Efficiency**: Can process high-resolution images (256×256 = 65,536 pixels)
- **Position bias**: Learned relative position embeddings per window

### Why Global Pooling?
- Converts spatial feature map (8×8×36) → fixed vector (36 features)
- Enables temporal processing regardless of input image size
- Acts as implicit classification token similar to Vision Transformer [CLS]

### Data Dimensions at Each Stage
```
Input:      [B, T, 3, 256, 256]     → 65,536 pixels per frame
Stem:       [B*T, 36, 64, 64]       → 4,096 features
Stage 1:    [B*T, 36, 64, 64]       → 4,096 tokens
Stage 2:    [B*T, 36, 32, 32]       → 1,024 tokens  
Stage 3:    [B*T, 288, 16, 16]      → 256 tokens (2×2 windows)
Stage 4:    [B*T, 36, 8, 8]         → 64 tokens (all in 1 global window)
Output:     [B, T, 128]             → 128-dim temporal features
```

---

## Integration with Cross-Attention Module

The output [B, T, 128] from ViT_Hierarchical serves as:
- **Key/Value** for the Cross-Attention module
- Enables attention between motion features and image features
- Fuses spatial-temporal information for pedestrian behavior prediction

---

## ASCII Art Summary: Two Complementary Views

### VIEW 1: End-to-End Architecture (High Level)
```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT [B=2, T=20, C=3, H=384, W=384] - context crop: 128×3.0      │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │  STEM                  │
                    │  Conv(3→36, 7/4) +BN   │
                    │  + GELU                │
                    └────────────────────────┘
                    [B*T=40, 36, 96, 96]
                                 ↓
        ┌────────────────────────────────────────────┐
        │ STAGE 1: 8×8 Windows, 2 Layers, 2 Heads   │
        │ • Processes: 96×96 feature maps           │
        │ • 144 windows of 64 tokens each           │
        │ • Local detail extraction                  │
        └────────────────────────────────────────────┘
        [B*T=40, 36, 96, 96]
                                 ↓
        ┌────────────────────────────────────────────┐
        │ STAGE 2: 4×4 Windows, 4 Layers, 2 Heads   │
        │ • Downsample: Conv2d (stride=2)           │
        │ • Processes: 48×48 feature maps           │
        │ • 144 windows of 16 tokens each           │
        │ • Medium-range dependencies               │
        └────────────────────────────────────────────┘
        [B*T=40, 36, 48, 48]
                                 ↓
        ┌────────────────────────────────────────────┐
        │ STAGE 3: 2×2 Windows, 5 Layers, 16 Heads  │
        │ • Downsample: Conv2d (stride=2)           │
        │ • Processes: 24×24 feature maps           │
        │ • 144 windows of 4 tokens each            │
        │ • Peak capacity (288 channels)            │
        └────────────────────────────────────────────┘
        [B*T=40, 288, 24, 24]
                                 ↓
        ┌────────────────────────────────────────────┐
        │ STAGE 4: Global, 7 Layers, 2 Heads        │
        │ • Downsample: Conv2d (stride=2)           │
        │ • Processes: 12×12 feature maps           │
        │ • 1 global window: all 144 tokens         │
        │ • Full image context, reduce to 36 dims   │
        └────────────────────────────────────────────┘
        [B*T=40, 36, 12, 12]
                                 ↓
                ┌────────────────────────┐
                │ Global Avg Pool        │
                │ mean(dim=[2,3])        │
                └────────────────────────┘
                [B*T=40, 36]
                                 ↓
                ┌────────────────────────┐
                │ LayerNorm + Projection │
                │ Linear(36→128)         │
                └────────────────────────┘
                [B, T=20, 128]
                                 ↓
        ┌────────────────────────────────────────────┐
        │ OUTPUT: Temporal feature vectors           │
        │ Ready for Cross-Attention Module           │
        │ 128-dim features per frame                 │
        └────────────────────────────────────────────┘
```

---

### VIEW 2: Single Block Deep Dive (Window Attention Details)
```
Input: [B*T, 36, 32, 32] - One stage's feature maps (Stage 2 example)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1: SPATIAL REORGANIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [B*T, 36, 32, 32]  ──Permute─→  [B*T, 32, 32, 36]
   (channels first)                 (channels last)
                                    for spatial ops

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2: WINDOW PARTITIONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [B*T, 32, 32, 36]
      32×32 spatial grid
        ↓
   Tile into 4×4 window pattern
     (32÷4) × (32÷4) = 8×8 = 64 windows
        ↓
  Each 4×4 region → 16 tokens
        ↓
  [64*B*T, 16, 36] - 64 independent windows
   per batch frame    tokens in each   channel
                      window dimension


   Visual: 32×32 feature map split into 4×4 windows
   
   ┌─┬─┬─┬─┬─┬─┬─┬─┐
   │1│2│3│4│5│6│7│8│ W01 = window 1 (4×4)
   ├─┼─┼─┼─┼─┼─┼─┼─┤ W09 = window 9 (4×4)
   │9│:│;│<│=│>│?│@│ ...
   ├─┼─┼─┼─┼─┼─┼─┼─┤ W64 = window 64 (4×4)
   │·│·│·│·│·│·│·│·│
   └─┴─┴─┴─┴─┴─┴─┴─┘
   (Each character = 1 4×4 window)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3: ATTENTION WITHIN WINDOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  For each window independently:

  [16, 128] ───┬─ Linear(128→384) ─┬─ Reshape(3,4,32) ─┬─ Q: [4, 16, 32]
  16 tokens    │                   │                    ├─ K: [4, 16, 32]
  128 dims     └─ Q, K, V split ──┘                    └─ V: [4, 16, 32]
                                                        4 heads
                                   ↓
                        Scaled Dot-Product per head:
                        
                        Q_scaled = Q × (1/√32)
                        
                        Attention = softmax(Q_scaled @ K^T)
                                   ├─ Dim: [4, 16, 16]
                                   ├─ Each token attends to
                                   │  all 16 tokens in window
                                   └─ Weights summing to 1.0
                                   
                        +Relative Position Bias:
                        
                        bias_table: (2×4-1)² × 4 heads
                                  = 49 × 4 learned params
                        
                        Attention = Attention + rel_bias
                                   ├─ 4×4 region only
                                   └─ Enables position awareness
                                   
                        Output = Attention @ V
                               = [4, 16, 32]
                               ┣━━ 4 heads
                               ┣━━ 16 tokens
                               └━━ 32 dims per head
                                   ↓
                        Concat heads: [16, 128]
                        Output proj: [16, 128]

  [16, 128] ← Result: Updated window features

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4: WINDOW REVERSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [64*B*T, 16, 128] (64 separate windows)
         ↓
   Reverse window_partition:
   Reconstruct 32×32 spatial layout
         ↓
  [B*T, 32, 32, 128] - Attended features
                       ready for next layer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 5: FEEDFORWARD (MLP) - Applied Spatially
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [B*T, 32, 32, 128]
      ↓
  Reshape to token sequence: [B*T, 1024, 128]
      ↓
  For each of 1024 spatial positions:
  
  [128] → Linear(128→512) → GELU → Dropout
       → Linear(512→128) → [128]
       
  Shared weights for all 1024 positions
      ↓
  [B*T, 1024, 128]
      ↓
  Reshape back to spatial: [B*T, 32, 32, 128]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 6: RESIDUAL CONNECTIONS & OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ Attention output
   │  [B*T, 32, 32, 36]
   │    + DropPath (stochastic depth)
   ├─ Shortcut (input)
   │  [B*T, 32, 32, 36]
   │
   └──→ [B*T, 32, 32, 36] ← Residual sum
               ↓
        Continue to MLP block
               ↓
        [B*T, 32, 32, 36] (same shape)


Output: [B*T, 36, 32, 32] ← Back to channels-first format
```

---

## Computing Resource Impact

### Attention Complexity Analysis

```
DENSE ATTENTION (if applied globally):
   All 1024 tokens attend to all 1024 tokens
   Memory: O(1024²) = ~1 million attention weights
   Computation: O(1024²) ≈ 1 million MACs per head
   
WINDOWED ATTENTION (4×4 windows):
   64 windows × 16 tokens per window
   Each window: 16 tokens attend to 16 tokens
   Memory per window: O(16²) = 256 weights
   Total: 64 × 256 = 16,384 attention weights
   Computation: 64 × 16² ≈ 16,384 MACs per head
   
SAVINGS:
   Memory reduction: 1,000,000 / 16,384 ≈ 61× less
   Compute reduction: 1,000,000 / 16,384 ≈ 61× faster
   
GLOBAL ATTENTION (Stage 4):
   8×8 = 64 tokens all in 1 window
   Memory: O(64²) ≈ 4,096 attention weights
   Computation: O(64²) ≈ 4,096 MACs per head
   BUT: Only happens at the final, very low-resolution stage
```

---

## How Hierarchical Design Balances Efficiency and Context

| Stage | Window | Tokens | Attention Type | Use Case | Tradeoff |
|-------|--------|----
