# Debug Checklist

Run through this before changing any code.

## Stage 1: Reproduce

- [ ] Can you reproduce consistently? (not intermittent)
- [ ] Note exact error message + full traceback
- [ ] Note which script + which line

## Stage 2: Locate

- [ ] Which stage? Data / Forward / Loss / Backward / Metrics
- [ ] Add shape/device prints at stage boundary
- [ ] Does it fail on first batch or after N steps?

## Stage 3: Isolate

- [ ] Does it fail with `batch_size=1`?
- [ ] Does it fail on CPU (`device="cpu"`)?
- [ ] Does it fail with random tensors (bypass data loading)?
  ```python
  dummy_tight   = torch.randn(1, SEQ_LEN, 3, H, W).to(device)
  dummy_context = torch.randn(1, SEQ_LEN, 3, H_ctx, W_ctx).to(device)
  dummy_motion  = torch.randn(1, SEQ_LEN, MOTION_DIM).to(device)
  out = model(dummy_tight, dummy_context, dummy_motion)
  ```

## Stage 4: Fix

- [ ] One change at a time
- [ ] Re-run import check after any model file change
- [ ] Re-run full forward pass with dummy tensors after fix
- [ ] Run one real batch before full training run
