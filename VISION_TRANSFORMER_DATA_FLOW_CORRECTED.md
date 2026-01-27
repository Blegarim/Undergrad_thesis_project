# Vision Transformer Hierarchical Architecture - Data Flow Diagrams
## Configuration: 4-Stage Hierarchical ViT (from config.py)

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
│  STAGE 2: dim=36, layers=4, heads=2, window_size=4             │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(36→36, k=3, stride=2, padding=1)        │
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
│  STAGE 3: dim=288, layers=5, heads=16, window_size=2           │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(36→288, k=3, stride=2, padding=1)       │
│  │  └─ [B*T, 288, 16, 16]          (H/2, W/2)                 │
│  │                                                              │
│  ├─ Blocks 1-5: Window Attention (2×2 windows)                 │
│  │  └─ Each window is only 2×2=4 tokens (highly local!)        │
│  │  └─ [B*T, 288, 16, 16]                                      │
│  │                                                              │
│  └─ (Further stochastic depth increase)                        │
│     └─ [B*T, 288, 16, 16]                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────────────────────────────┐
│  STAGE 4: dim=36, layers=7, heads=2, window_size="global"      │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Downsample: Conv2d(288→36, k=3, stride=2, padding=1)       │
│  │  └─ [B*T, 36, 8, 8]              (H/2, W/2)                 │
│  │  └─ NOTE: Major dimension reduction 288→36 (8× compression) │
│  │                                                              │
│  ├─ Blocks 1-7: Global Window Attention                        │
│  │  └─ Each block sees all 8×8=64 tokens globally             │
│  │  └─ [B*T, 36, 8, 8]                                         │
│  │                                                              │
│  └─ (Max stochastic depth at final blocks)                     │
│     └─ [B*T, 36, 8, 8]                                         │
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


═══════════════════════════════════════════════════════════════════
SPATIAL RESOLUTION TRACKING:
═══════════════════════════════════════════════════════════════════
Input Image:        256×256  (65,536 pixels)
After Stem:         64×64    (4,096 pixels)   1/4 resolution
Stage 1:            64×64    (4,096 tokens)   8×8=64 tokens/window
Stage 2:            32×32    (1,024 tokens)   4×4=16 tokens/window
Stage 3:            16×16    (256 tokens)     2×2=4 tokens/window  ← UNIQUE!
Stage 4:            8×8      (64 tokens)      All tokens in 1 global window
After GAP:          1        (36 features)
Final Proj:         1        (128 features)   ← Temporal features for cross-attention

═══════════════════════════════════════════════════════════════════
UNIQUE FEATURE: Stage 3 (2×2 Windows)
═══════════════════════════════════════════════════════════════════
Stage 3 is EXTREMELY local with 2×2 windows (only 4 tokens each)
This creates 64 independent attention heads (16×16 spatial = 256 total tokens)
256 tokens / 4 tokens per window = 64 windows
BUT Stage 3 has 16 heads (highest of all stages!)
Each head operates on 4-token windows with 16 parallel heads = rich local detail

Stage 4 compensates with GLOBAL attention across all 64 tokens (8×8)
Plus dimension reduction 288→36 acts as a bottleneck/compression layer

═══════════════════════════════════════════════════════════════════
LAYER CONFIGURATION SUMMARY (ACTUAL):
═══════════════════════════════════════════════════════════════════
Stage  │ In→Out Dim  │ Layers │ Heads │ Window │ Tokens │ Purpose
───────┼─────────────┼────────┼───────┼────────┼────────┼──────────────────
  1    │ 36→36       │   2    │   2   │  8×8   │   64   │ Initial detail
  2    │ 36→36       │   4    │   2   │  4×4   │   16   │ Medium context
  3    │ 36→288      │   5    │  16   │  2×2   │    4   │ RICH LOCAL (2×2!)
  4    │ 288→36      │   7    │   2   │ global │   64   │ GLOBAL context
  GAP  │ 36→36       │   -    │   -   │   -    │    1   │ Spatial pooling
  Proj │ 36→128      │   -    │   -   │   -    │    1   │ Cross-attn compat
```

---

## Diagram 2: Single Stage Deep Dive - Stage 3 Example (2×2 Windows)

### Unique: Stage 3 with 16 Heads and 2×2 Windows

```
INPUT TO STAGE 3: [B*T, 36, 16, 16]  (after Stage 2 downsample)
│
│  ╔════════════════════════════════════════════════════════════════╗
│  ║          DOWNSAMPLE: Conv2d(36→288, stride=2)                 ║
│  ║          Spatial: 16×16 → 8×8                                 ║
│  ║          Channel expansion: 36 → 288 (8× capacity!)           ║
│  ╚════════════════════════════════════════════════════════════════╝
│
│  [B*T, 288, 8, 8]  ← Now ready for Stage 3 blocks
│
├─────────────────────────────────────────────────────────────────────
│  STAGE 3 BLOCK #1 (Representative of all 5 blocks)
├─────────────────────────────────────────────────────────────────────
│
├─ Step 0: Reshape for spatial operations
│  Input:  [B*T, 288, 8, 8]
│  Perm:   [B*T, 8, 8, 288]  ← [B, H, W, C]
│  └─ Now in format for window partitioning
│
│
├─ ╔════════════════════════════════════════════════════════════════╗
│  ║            FIRST RESIDUAL BLOCK: ATTENTION                     ║
│  ╚════════════════════════════════════════════════════════════════╝
│
│
├─ Step 1: LayerNorm
│  Input:  [B*T, 8, 8, 288]
│  Output: [B*T, 8, 8, 288]  (normalized features)
│
│
├─ Step 2: WINDOW PARTITION (2×2 WINDOWS - UNIQUE!)
│  ┌─────────────────────────────────────────────────────────────┐
│  │ Input: [B*T, 8, 8, 288]                                    │
│  │                                                              │
│  │ Configuration: window_size = 2×2  ← EXTREMELY LOCAL!        │
│  │ Feature map: 8×8 → tiled into (8/2)×(8/2) = 4×4 grid      │
│  │      
