"""
Standalone CLI script that reads CSV/NPZ artifacts from training and
evaluation runs and writes PNG figures to plots/.

Usage
-----
Run from the project root. Each phase is independent — pass only the
flags for the figures you want. Outputs are written to ``--out_dir``
(default ``plots/``), which is created if missing.

    python scripts/plot_results.py [options]

Phase 1 — Training curves (loss + per-head F1/accuracy):
    python scripts/plot_results.py \
        --train_log results/training_log.csv \
        --out_dir plots/

    Input: training_log CSV with columns ``Epoch``, ``Avg Train Loss``,
    ``Val Loss`` and either ``{Actions,Looks,Crosses} F1`` (+ optional
    ``Macro F1``) or ``{Actions,Looks,Crosses} Acc`` + ``Overall Val Acc``.
    Outputs: ``loss_curves.png``, ``per_head_f1_curves.png``.

Phase 2 — Precision-recall and threshold sweep:
    python scripts/plot_results.py \
        --predictions results/predictions.csv \
        --out_dir plots/

    Input: per-sample predictions CSV with columns
    ``{actions,looks,crosses}_true`` and ``{actions,looks,crosses}_prob_1``.
    Outputs: ``pr_curves.png`` (with t=0.5 and best-F1 markers),
    ``f1_threshold.png``.

Phase 3 — Ablation comparison bar charts:
    python scripts/plot_results.py \
        --ablation_logs full=results/test_full.csv,motion_only=results/test_motion.csv,visual_only=results/test_visual.csv,vanilla_concat=results/test_concat.csv \
        --out_dir plots/

    Input: comma-separated ``name=path`` pairs pointing at test-log CSVs
    that contain a summary table (either under a ``=== Default Threshold
    (0.5) ===`` header or a bare ``Heads,Accuracy,F1,AUC,...`` row).
    A ``full`` entry adds a macro-metric reference line.
    Outputs: ``ablation_f1.png``, ``ablation_auc.png``.

Phase 4 — Temporal attention heatmap:
    python scripts/plot_results.py \
        --temporal_weights results/temporal_weights.npz \
        --out_dir plots/

    Input: NPZ with key ``temporal_weights`` of shape ``[N, T]`` and at
    least one of ``{actions,looks,crosses}_true`` for class splitting.
    Output: ``temporal_attention.png`` (mean ± std weight per frame,
    split by class).

Phases can be combined in one invocation; only the flags supplied are
run. With no flags, the script prints a hint and exits.
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

# ── Consistent colour palette ────────────────────────────────────────────
COLOR_ACTIONS = "#1f77b4"
COLOR_LOOKS = "#ff7f0e"
COLOR_CROSSES = "#2ca02c"
COLOR_MACRO = "#9467bd"
COLOR_TRAIN = "#1f77b4"
COLOR_VAL = "#d62728"

ABLATION_COLORS = {
    "full": "#1f77b4",
    "motion_only": "#ff7f0e",
    "visual_only": "#2ca02c",
    "vanilla_concat": "#d62728",
}

HEAD_COLORS = {
    "actions": COLOR_ACTIONS,
    "looks": COLOR_LOOKS,
    "crosses": COLOR_CROSSES,
}

DPI = 150


# ── Phase 1: Training-time curves ────────────────────────────────────────

def _load_training_log(path: str) -> dict:
    """Parse a training_log CSV into a dict of lists, keyed by header."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def plot_loss_curves(train_log_path: str, out_dir: str) -> str:
    data = _load_training_log(train_log_path)
    epochs = data["Epoch"]
    train_loss = data["Avg Train Loss"]
    val_loss = data["Val Loss"]

    best_idx = int(np.argmin(val_loss))
    best_epoch = epochs[best_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color=COLOR_TRAIN, label="Train loss")
    ax.plot(epochs, val_loss, color=COLOR_VAL, label="Val loss")
    ax.axvline(best_epoch, ls="--", color="gray", lw=0.9,
               label=f"Best val epoch {int(best_epoch)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dst = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    return dst


def plot_per_head_f1_curves(train_log_path: str, out_dir: str) -> str:
    data = _load_training_log(train_log_path)
    epochs = data["Epoch"]

    has_f1 = all(k in data for k in ("Actions F1", "Looks F1", "Crosses F1"))

    fig, ax = plt.subplots(figsize=(7, 4))

    if has_f1:
        ax.plot(epochs, data["Actions F1"], color=COLOR_ACTIONS, label="Actions F1")
        ax.plot(epochs, data["Looks F1"], color=COLOR_LOOKS, label="Looks F1")
        ax.plot(epochs, data["Crosses F1"], color=COLOR_CROSSES, label="Crosses F1")
        if "Macro F1" in data:
            ax.plot(epochs, data["Macro F1"], color=COLOR_MACRO, ls="--",
                    label="Macro F1")
        ax.set_ylabel("F1")
    else:
        ax.plot(epochs, data["Actions Acc"], color=COLOR_ACTIONS,
                label="Actions Acc")
        ax.plot(epochs, data["Looks Acc"], color=COLOR_LOOKS,
                label="Looks Acc")
        ax.plot(epochs, data["Crosses Acc"], color=COLOR_CROSSES,
                label="Crosses Acc")
        ax.plot(epochs, data["Overall Val Acc"], color=COLOR_MACRO, ls="--",
                label="Overall Val Acc")
        ax.set_ylabel("Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dst = os.path.join(out_dir, "per_head_f1_curves.png")
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    return dst


# ── Phase 2: PR curves + threshold sweep ─────────────────────────────────

def _load_predictions(path: str) -> dict:
    """Return dict with arrays: actions_true, actions_prob_1, looks_true, etc."""
    out: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if v == "":
                    continue
                out.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in out.items()}


def plot_pr_curves(predictions_path: str, out_dir: str) -> str:
    pred = _load_predictions(predictions_path)
    heads = ["actions", "looks", "crosses"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, head in zip(axes, heads):
        y_true = pred[f"{head}_true"].astype(int)
        y_prob = pred[f"{head}_prob_1"]

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=HEAD_COLORS[head], lw=1.5)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.annotate(f"AP = {ap:.3f}", xy=(0.02, 0.02),
                    xycoords="axes fraction", fontsize=9)

        # Mark default threshold=0.5
        pred_05 = (y_prob >= 0.5).astype(int)
        p_05 = precision_at_threshold(y_true, pred_05)
        r_05 = recall_at_threshold(y_true, pred_05)
        ax.plot(r_05, p_05, "o", ms=7, color="black", label="t=0.5")

        # Mark F1-optimal threshold
        f1_scores = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        best_idx = int(np.argmax(f1_scores))
        ax.plot(recall[best_idx], precision[best_idx], "*", ms=11,
                color="red", label=f"best F1 t={thresholds[best_idx]:.2f}")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(head.capitalize(), fontsize=10)

    fig.tight_layout()
    dst = os.path.join(out_dir, "pr_curves.png")
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    return dst


def precision_at_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / (tp + fp + 1e-12)


def recall_at_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fn + 1e-12)


def plot_f1_threshold(predictions_path: str, out_dir: str) -> str:
    pred = _load_predictions(predictions_path)
    heads = ["actions", "looks", "crosses"]
    thresholds = np.arange(0.05, 0.96, 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, head in zip(axes, heads):
        y_true = pred[f"{head}_true"].astype(int)
        y_prob = pred[f"{head}_prob_1"]

        f1s = []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_true, preds, zero_division=0))
        f1s = np.array(f1s)

        ax.plot(thresholds, f1s, color=HEAD_COLORS[head], lw=1.5)
        best_idx = int(np.argmax(f1s))
        best_t = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        ax.axvline(best_t, ls="--", color="gray", lw=0.9)
        ax.annotate(f"t={best_t:.2f}\nF1={best_f1:.3f}",
                    xy=(best_t, best_f1), fontsize=8,
                    xytext=(best_t + 0.08, best_f1 - 0.08),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(head.capitalize(), fontsize=10)

    fig.tight_layout()
    dst = os.path.join(out_dir, "f1_threshold.png")
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    return dst


# ── Phase 3: Ablation comparison bar charts ───────────────────────────────

def _parse_summary_table(log_path: str) -> dict:
    """Extract the summary table from a test log.

    Handles both the new format (with '=== Default Threshold (0.5) ===' header)
    and the older format (bare 'Heads,...' row after chunk rows).

    Returns e.g. {"actions_f1": 0.64, "actions_auc": 0.79, ...}.
    """
    all_rows: list[list[str]] = []
    with open(log_path, newline="") as f:
        for row in csv.reader(f):
            all_rows.append(row)

    data_rows: list[list[str]] = []
    for i, row in enumerate(all_rows):
        if not row:
            continue
        if "Default Threshold" in row[0]:
            for j in range(i + 1, len(all_rows)):
                r = all_rows[j]
                if not r or r[0] == "" or r[0].startswith("Overall"):
                    break
                if r[0] == "Heads":
                    continue
                data_rows.append(r)
            break
        if row[0] == "Heads" and len(row) >= 6 and row[1] == "Accuracy":
            for j in range(i + 1, len(all_rows)):
                r = all_rows[j]
                if not r or r[0] == "" or r[0].startswith("Overall"):
                    break
                data_rows.append(r)
            break

    metrics: dict[str, float] = {}
    for row in data_rows:
        head = row[0].lower()
        try:
            metrics[f"{head}_acc"] = float(row[1])
            metrics[f"{head}_f1"] = float(row[2])
            metrics[f"{head}_auc"] = float(row[3])
            metrics[f"{head}_p"] = float(row[4])
            metrics[f"{head}_r"] = float(row[5])
        except (IndexError, ValueError):
            continue
    return metrics


def _ablation_bar_chart(
    ablation_data: dict[str, dict[str, float]],
    metric_suffix: str,
    ylabel: str,
    out_path: str,
) -> str:
    heads = ["actions", "looks", "crosses"]
    model_names = list(ablation_data.keys())
    n_groups = len(heads)
    n_bars = len(model_names)
    bar_w = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model_name in enumerate(model_names):
        vals = [ablation_data[model_name].get(f"{h}_{metric_suffix}", 0)
                for h in heads]
        ax.bar(x + i * bar_w, vals, bar_w,
               label=model_name.replace("_", " "),
               color=ABLATION_COLORS.get(model_name, f"C{i}"))

    # Reference dashed line at full model's macro metric
    if "full" in ablation_data:
        full_vals = [ablation_data["full"].get(f"{h}_{metric_suffix}", 0)
                     for h in heads]
        macro = np.mean(full_vals)
        ax.axhline(macro, ls="--", lw=0.9, color="gray",
                    label=f"Full macro {ylabel} ({macro:.2f})")

    ax.set_xticks(x + bar_w * (n_bars - 1) / 2)
    ax.set_xticklabels([h.capitalize() for h in heads])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_ablation_f1(ablation_logs: dict[str, str], out_dir: str) -> str:
    ablation_data = {name: _parse_summary_table(path)
                     for name, path in ablation_logs.items()}
    return _ablation_bar_chart(
        ablation_data, "f1", "F1",
        os.path.join(out_dir, "ablation_f1.png"),
    )


def plot_ablation_auc(ablation_logs: dict[str, str], out_dir: str) -> str:
    ablation_data = {name: _parse_summary_table(path)
                     for name, path in ablation_logs.items()}
    return _ablation_bar_chart(
        ablation_data, "auc", "AUC",
        os.path.join(out_dir, "ablation_auc.png"),
    )


# ── Phase 4: Temporal attention heatmap ───────────────────────────────────

def plot_temporal_attention(tw_path: str, out_dir: str) -> str:
    npz = np.load(tw_path)
    weights = npz["temporal_weights"]  # [N, T]
    T = weights.shape[1]
    frames = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    head_labels = {
        "crosses": ("crosses_true", "Crosses"),
        "actions": ("actions_true", "Actions"),
        "looks": ("looks_true", "Looks"),
    }

    for ax, (head, (label_key, display)) in zip(axes, head_labels.items()):
        if label_key not in npz:
            ax.set_visible(False)
            continue

        y_true = npz[label_key].astype(int)
        for cls_val, cls_label, color, ls in [
            (0, f"{display}=0", HEAD_COLORS.get(head, "gray"), "-"),
            (1, f"{display}=1", HEAD_COLORS.get(head, "gray"), "--"),
        ]:
            mask = y_true == cls_val
            if mask.sum() == 0:
                continue
            w_sub = weights[mask]
            mean = w_sub.mean(axis=0)
            std = w_sub.std(axis=0)
            ax.plot(frames, mean, ls=ls, color=color, lw=1.5, label=cls_label)
            ax.fill_between(frames, mean - std, mean + std,
                            color=color, alpha=0.12)

        ax.set_xlabel("Frame index")
        ax.set_ylabel("Mean softmax weight")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(display, fontsize=10)

    fig.tight_layout()
    dst = os.path.join(out_dir, "temporal_attention.png")
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    return dst


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_ablation_arg(raw: str) -> dict[str, str]:
    """Parse 'name=path,name=path,...' into dict."""
    out = {}
    for pair in raw.split(","):
        if "=" not in pair:
            raise ValueError(f"Expected name=path, got: {pair}")
        name, path = pair.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from training artifacts.",
    )
    parser.add_argument("--train_log", type=str, default=None,
                        help="Path to training_log CSV (Phase 1).")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions CSV (Phase 2).")
    parser.add_argument("--ablation_logs", type=str, default=None,
                        help="Comma-separated name=path pairs (Phase 3).")
    parser.add_argument("--temporal_weights", type=str, default=None,
                        help="Path to temporal_weights.npz (Phase 4).")
    parser.add_argument("--out_dir", type=str, default="plots",
                        help="Output directory for PNG figures.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generated: list[str] = []

    # Phase 1
    if args.train_log:
        print(f"[Phase 1] Reading {args.train_log}")
        generated.append(plot_loss_curves(args.train_log, args.out_dir))
        generated.append(plot_per_head_f1_curves(args.train_log, args.out_dir))

    # Phase 2
    if args.predictions:
        print(f"[Phase 2] Reading {args.predictions}")
        generated.append(plot_pr_curves(args.predictions, args.out_dir))
        generated.append(plot_f1_threshold(args.predictions, args.out_dir))

    # Phase 3
    if args.ablation_logs:
        print(f"[Phase 3] Reading ablation logs")
        abl = _parse_ablation_arg(args.ablation_logs)
        generated.append(plot_ablation_f1(abl, args.out_dir))
        generated.append(plot_ablation_auc(abl, args.out_dir))

    # Phase 4
    if args.temporal_weights:
        print(f"[Phase 4] Reading {args.temporal_weights}")
        generated.append(plot_temporal_attention(args.temporal_weights,
                                                 args.out_dir))

    if not generated:
        print("No inputs provided — nothing to plot. Use --help for options.")
        sys.exit(0)

    print(f"\nGenerated {len(generated)} figure(s):")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
