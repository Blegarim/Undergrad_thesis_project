---
name: experiment-tracking
description: Conventions for logging, naming, checkpointing, and interpreting training runs for the pedestrian prediction project. Use whenever starting a new training run, saving/loading checkpoints, comparing results across runs, reading CSV logs, deciding if a result is an improvement, or writing evaluation code. Trigger on mentions of "run", "checkpoint", "results", "metrics", "is this better", "compare runs", "save model", "resume training", or any question about whether a model improved.
---

# Experiment Tracking

## Run Naming Convention

```
{model_variant}_{date}_{short_note}
e.g.: ensemble_20250414_lrsched
      ensemble_20250415_dropout02
```

- `date`: YYYYMMDD
- `short_note`: what changed vs previous run (1-3 words, underscored)
- Keep names filesystem-safe — no spaces, no slashes

## Directory Structure Per Run

```
runs/
└── {run_name}/
    ├── config_snapshot.json   # copy of config.py values at run start
    ├── train_log.csv          # per-epoch metrics
    ├── checkpoints/
    │   ├── best.pth           # best val loss checkpoint
    │   └── last.pth           # end-of-epoch checkpoint
    └── eval_results.csv       # output of test.py
```

Always snapshot config at run start — reconstructing hyperparams from memory later is unreliable.

## CSV Log Schema

`train_log.csv` columns:
```
epoch, train_loss, val_loss,
acc_crossing, f1_crossing, auc_crossing,
acc_looking,  f1_looking,  auc_looking,
acc_action,   f1_action,   auc_action,
lr, epoch_time_s
```

`eval_results.csv` columns (from `test.py`):
```
task, accuracy, f1, auc, precision, recall,
flops, latency_ms, fps
```

## Checkpoint Save/Load Pattern

**Save:**
```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler_state": scaler.state_dict(),
    "val_loss": val_loss,
    "config": config_dict,
}, path)
```

**Load for resume:**
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
scaler.load_state_dict(ckpt["scaler_state"])
start_epoch = ckpt["epoch"] + 1
```

**Load for eval only:**
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
# Don't load optimizer/scaler
```

## What "Better" Means

**Primary metric: `f1_crossing`** (most task-critical, imbalanced classes)

**Decision rule:**
```
New run is better if:
  f1_crossing improves by > 0.5pp   ← meaningful
  AND val_loss does not increase by > 5%
```

Do not optimize for accuracy alone — crossing class imbalance makes it misleading. AUC is a secondary confirmation signal.

**Metric priority:**
1. `f1_crossing` — primary
2. `auc_crossing` — secondary confirmation
3. `f1_looking`, `f1_action` — should not regress > 1pp vs best run
4. `val_loss` — sanity check, not primary criterion

## Baseline Numbers

See `references/baseline-results.md` for current best run metrics. Update that file when a new best is established. Never compare new results to numbers recalled from memory — always read the file.

## Comparing Runs

When asked to compare runs:
1. Read `eval_results.csv` for each run
2. Diff `config_snapshot.json` to identify what changed
3. Use metric priority above to determine winner
4. Flag if any non-target task regressed > 1pp

## Common Pitfalls

- `best.pth` saves lowest val_loss, not best f1 — always run `test.py` to get true eval metrics before declaring a run "best"
- Resuming from `last.pth` restores LR scheduler state — double-check LR is expected after resume
- FLOPs/latency in `eval_results.csv` are measured on eval hardware — don't compare across machines

## See Also

`references/baseline-results.md` — current best run metrics per task.
