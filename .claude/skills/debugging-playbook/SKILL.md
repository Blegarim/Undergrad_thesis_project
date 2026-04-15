---
name: debugging-playbook
description: Decision-tree debugging guide for the pedestrian prediction PyTorch project. Use whenever diagnosing errors, crashes, silent failures, NaN losses, OOM, shape mismatches, slow training, or unexpected model outputs. Trigger on any error traceback, "why is X happening", "model not converging", "CUDA error", "loss is NaN", "shape mismatch", or "it's slow" in the context of this project.
---

# Debugging Playbook

## Rule 0: Identify the Stage First

```
Data loading → Forward pass → Loss → Backward → Optimizer → Metrics
```
Isolate which stage fails before fixing anything. Use the checklist in `references/debug-checklist.md`.

---

## OOM (CUDA Out of Memory)

**Order of fixes — try in sequence, stop when resolved:**

1. Reduce `batch_size` in `config.py` (halve it)
2. Ensure `torch.no_grad()` wraps all eval/inference code
3. Check for tensor accumulation in loops — `.detach()` before appending to lists
4. Add `gc.collect(); torch.cuda.empty_cache()` between epochs
5. Reduce `SEQ_LEN` or `IMG_SIZE` in config (last resort — affects accuracy)
6. Enable gradient checkpointing in ViT (see `Vision_Transformer.py` comments)

**Diagnosis:**
```python
print(torch.cuda.memory_summary())  # before crash
```

---

## NaN / Inf Loss

**Causes in priority order:**

1. **LR too high** — halve `learning_rate` in config, most common cause
2. **AMP overflow** — check `scaler.get_scale()` is not dropping to 1.0 repeatedly
3. **Zero division in loss** — check class weights sum and label range matches `num_classes_dict`
4. **Bad input** — NaN in batch:
   ```python
   assert not torch.isnan(images_tight).any(), "NaN in tight crops"
   assert not torch.isnan(motion).any(), "NaN in motion"
   ```
5. **LogSumExp instability** in CrossAttentionModule — add `eps=1e-6` to denominator if traced here

**Quick check:**
```python
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"NaN grad: {name}")
```

---

## Shape Mismatch

**Always print shapes at the boundary:**
```python
print(f"tight: {images_tight.shape}")   # expect (B,T,3,H,W)
print(f"context: {images_context.shape}")
print(f"motion: {motion.shape}")         # expect (B,T,D_motion)
```

**Common mismatches:**

| Error | Cause | Fix |
|-------|-------|-----|
| `mat1 dim 1 != mat2 dim 0` | Embedding dim mismatch | Check `d_model` in config vs CrossAttention init |
| `Expected 4D tensor` | Missing batch or time dim | Add `.unsqueeze(0)` or check collate_fn |
| `size mismatch at dim 1` | SEQ_LEN inconsistency | Check all components use same `SEQ_LEN` from config |
| `Expected input batch_size == target batch_size` | Loss getting wrong labels | Verify label key names match `num_classes_dict` keys |

---

## CUDA / Device Errors

**`RuntimeError: Expected all tensors to be on same device`**
- Find the offending tensor: print `.device` for each input before forward
- Ensure `model.to(device)` called after ALL components assembled
- Ensure labels moved: `labels = {k: v.to(device) for k, v in labels.items()}`

**`CUDA error: device-side assert triggered`**
- Run with `CUDA_LAUNCH_BLOCKING=1 python train.py` to get real traceback
- Usually label index out of range — check `label_crossing` values are in `[0, num_classes-1]`

---

## Slow Training

**Benchmark first:**
```python
# At start of epoch, time one batch
import time
t = time.time(); next(iter(dataloader)); print(f"Batch load: {time.time()-t:.2f}s")
```

**If load time > forward time:** bottleneck is data
- Increase `num_workers` (try 4 → 8)
- Confirm `pin_memory=True`
- Check LMDB is on SSD not network drive

**If forward time is slow:**
- Confirm `torch.backends.cudnn.benchmark = True` is set
- Confirm AMP is active: wrap with `autocast`, check `scaler` is used
- Profile: `torch.utils.bottleneck python train.py`

---

## Model Not Converging

1. Sanity check: overfit one batch
   ```python
   # In train.py, set dataset to 1 sample, run 100 epochs
   # Loss should go to ~0. If not, bug in loss/labels.
   ```
2. Check learning rate schedule — `warmup_steps` in config too long?
3. Verify all three task heads (`crossing`, `looking`, `action`) have non-zero loss contribution
4. Check loss weights in `config.py` — one task dominating?
5. Inspect predictions: are all samples predicting the same class? (collapsed model)

---

## Component Import Errors

Quick isolation:
```bash
python -c "from models.Vision_Transformer import ViT_Hierarchical; print('ViT OK')"
python -c "from models.Motion_Encoder import MotionEncoder; print('ME OK')"
python -c "from models.Cross_Attention_Module import CrossAttentionModule; print('CA OK')"
python -c "from models.Unified_Module import EnsembleModel; print('Ensemble OK')"
```

---

## See Also

`references/debug-checklist.md` — step-by-step checklist to run before asking for help or changing code.
