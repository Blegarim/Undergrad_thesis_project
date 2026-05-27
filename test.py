import os
import csv
import time
import gc
import math
import re
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

import argparse

# ==== Model Imports ====
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from models.AblationModels import MotionOnlyModel, VisualOnlyModel, VanillaConcatModel
from scripts.lmdb_dataset import LMDBChunkDataset
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model

# ==== Shared utilities ====
from scripts.train_utils import remap_cross_labels, collate_fn
from scripts.model_utils import get_model, model_forward

def evaluate(model, dataloader, device, model_type, use_amp=False):
    model.eval()
    correct, total = {}, {}
    all_preds, all_labels, all_probs = {}, {}, {}
    all_temporal_weights = []

    with torch.no_grad():
        for images_tight, images_context, motions, labels in dataloader:
            images_tight = images_tight.to(device, non_blocking=True)
            images_context = images_context.to(device, non_blocking=True)
            motions = motions.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True).long() for k, v in labels.items()}

            remap_cross_labels(labels)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model_forward(model, model_type, images_tight, images_context, motions)

            if "temporal_weights" in outputs:
                all_temporal_weights.append(outputs["temporal_weights"].float().cpu())

            # Process each head (actions, looks, crosses)
            for name in ["actions", "looks", "crosses"]:
                if name == "crosses":
                    logits = outputs["crosses_frame"]
                else:
                    logits = outputs[name]

                # Ensure float for softmax when using AMP
                if use_amp:
                    logits = logits.float()

                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(probs, 1)

                # Accuracy
                correct[name] = correct.get(name, 0) + (preds == labels[name]).sum().item()
                total[name] = total.get(name, 0) + labels[name].numel()

                # store for F1/AUC
                all_preds.setdefault(name, []).append(preds.cpu())
                all_labels.setdefault(name, []).append(labels[name].cpu())
                all_probs.setdefault(name, []).append(probs.cpu())

    metrics = {}
    for name in all_labels.keys():  # Only iterate over the 3 heads we processed
        y_true = torch.cat(all_labels[name]).numpy()
        y_pred = torch.cat(all_preds[name]).numpy()
        y_prob = torch.cat(all_probs[name]).numpy()

        avg_type = "binary" if y_prob.shape[1] == 2 else "macro"
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=avg_type)
        precision = precision_score(y_true, y_pred, average=avg_type)
        recall = recall_score(y_true, y_pred, average=avg_type)
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc = float("nan")

        metrics[name + "_acc"] = acc
        metrics[name + "_f1"] = f1
        metrics[name + "_auc"] = auc
        metrics[name + "_p"] = precision
        metrics[name + "_r"] = recall

    overall = sum(correct.values()) / sum(total.values())
    metrics["overall_acc"] = overall
    temporal_weights = (
        torch.cat(all_temporal_weights).numpy() if all_temporal_weights else None
    )
    return metrics, all_labels, all_preds, all_probs, temporal_weights


def round_metric(metrics, key):
    return round(metrics.get(key, 0.0), 2)

def find_optimal_thresholds(y_true, y_prob, task_name=""):
    """
    Find optimal threshold that maximizes F1 score for binary classification.
    Uses probability of positive class (column 1) for thresholding.
    
    Args:
        y_true: Ground truth labels (numpy array)
        y_prob: Probability predictions (numpy array, shape: [N, 2])
        task_name: Name for logging
    
    Returns:
        optimal_threshold: Float threshold value
        best_f1: Best F1 score achieved
    """
    if y_prob.shape[1] != 2:
        return 0.5, 0.0
    
    pos_probs = y_prob[:, 1]
    
    if len(set(y_true)) < 2:
        return 0.5, 0.0
    
    thresholds = [round(t * 0.05, 2) for t in range(2, 19)]
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        preds = (pos_probs >= thresh).astype(int)
        try:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        except Exception:
            continue
    
    return best_threshold, best_f1

def _infer_window_hw(index_tensor, table_size):
    if index_tensor.ndim != 2 or index_tensor.shape[0] != index_tensor.shape[1]:
        return None, None
    n = index_tensor.shape[0]
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h != 0:
            continue
        w = n // h
        if (2 * h - 1) * (2 * w - 1) == table_size:
            return h, w
    return None, None

def _init_global_rel_pos_from_ckpt(model, state_dict):
    pattern = re.compile(r"^vit\.stages\.(\d+)\.block\.(\d+)\.attn\.relative_position_index$")
    for key, index_tensor in state_dict.items():
        match = pattern.match(key)
        if not match:
            continue
        stage_idx = int(match.group(1))
        block_idx = int(match.group(2))
        bias_key = key.replace("relative_position_index", "relative_position_bias_table")
        bias_table = state_dict.get(bias_key)
        if bias_table is None:
            continue
        H, W = _infer_window_hw(index_tensor, bias_table.shape[0])
        if H is None:
            continue
        try:
            block = model.vit.stages[stage_idx]["block"][block_idx]
        except (IndexError, KeyError, AttributeError, TypeError):
            continue
        if getattr(block, "window_size", None) is None:
            block.init_relative_position_bias(H, W)


def compute_flops(model, model_type, img_height, img_width, context_scale, device):
    """
    Compute FLOPs for the model.
    
    Args:
        model: The model instance
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', or 'full'
        img_height: Image height for tight crop
        img_width: Image width for tight crop
        context_scale: Scale factor for context images
        device: torch device
    
    Returns:
        flops_per_frame: FLOPs per frame
    """
    from fvcore.nn import FlopCountAnalysis
    
    dummy_images_tight = torch.randn(1, 20, 3, img_height, img_width).to(device)
    dummy_images_context = torch.randn(1, 20, 3, int(img_height * context_scale), int(img_width * context_scale)).to(device)
    dummy_motions = torch.randn(1, 20, 8).to(device)
    
    model.eval()
    
    with torch.no_grad():
        if model_type == 'motion_only':
            inputs = (dummy_motions, dummy_images_tight)
        elif model_type == 'visual_only':
            inputs = (dummy_images_context,)
        else:
            inputs = (dummy_images_tight, dummy_images_context, dummy_motions)
        
        flops = FlopCountAnalysis(model, inputs)
        flops.unsupported_ops_warnings(False)
        flops_total = flops.total()
        flops_per_frame = flops_total / dummy_images_tight.size(1)
    
    print(f'Total FLOPs per {dummy_images_tight.size(1)}-frame input: {flops_total/1e9:.2f} GFLOPs')
    print(f'Average FLOPs per frame: {flops_per_frame/1e6:.2f} MFLOPs')
    return flops_per_frame


def inference_latency(model, model_type, img_height, img_width, context_scale, device, num_trials=50):
    """
    Measure inference latency for the model.
    
    Args:
        model: The model instance
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', or 'full'
        img_height: Image height for tight crop
        img_width: Image width for tight crop
        context_scale: Scale factor for context images
        device: torch device
        num_trials: Number of trials for averaging
    
    Returns:
        avg_fps: Average frames per second
        avg_latency_per_frame: Average latency per frame in seconds
    """
    dummy_images_tight = torch.randn(1, 20, 3, img_height, img_width).to(device)
    dummy_images_context = torch.randn(1, 20, 3, int(img_height * context_scale), int(img_width * context_scale)).to(device)
    dummy_motions = torch.randn(1, 20, 8).to(device)
    
    model.eval()
    
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            if model_type == 'motion_only':
                _ = model(dummy_motions, dummy_images_tight)
            elif model_type == 'visual_only':
                _ = model(dummy_images_context)
            else:
                _ = model(dummy_images_tight, dummy_images_context, dummy_motions)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(num_trials):
            if model_type == 'motion_only':
                _ = model(dummy_motions, dummy_images_tight)
            elif model_type == 'visual_only':
                _ = model(dummy_images_context)
            else:
                _ = model(dummy_images_tight, dummy_images_context, dummy_motions)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
    
    avg_latency = (end - start) / num_trials
    avg_fps = 1.0 / avg_latency
    avg_latency_per_frame = avg_latency / dummy_images_tight.size(1)
    
    print(f"\nInference latency (averaged over {num_trials} runs):")
    print(f"  {avg_latency*1000:.2f} ms per {dummy_images_tight.size(1)}-frame sequence")
    print(f"  {avg_latency_per_frame*1000:.2f} ms per frame")
    print(f"  {avg_fps:.2f} FPS equivalent")
    
    return avg_fps, avg_latency_per_frame


# ============================================================
# === Main Testing Script ====================================
# ============================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test pedestrian behavior prediction model')
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['motion_only', 'visual_only', 'vanilla_concat', 'full'],
                        help='Model type for ablation study')
    parser.add_argument('--model_path', type=str, 
                        default="best_model_outputs/best_model_epoch28_0122_1511.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save detailed predictions with probabilities to CSV')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    use_amp = device.type == "cuda"

    # ==== CONFIGURATION ====
    embedding_dim = get_unified_dim_model()
    batch_size = 16
    img_size = 128
    context_scale = 3
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    num_workers = 4
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 2
    }
    model_path = args.model_path
    test_chunk_folder = "preprocessed_test"
    log_dir = "training_log"
    os.makedirs(log_dir, exist_ok=True)
    transform_tight = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_context = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ==== Prepare log file ====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_csv = os.path.join(log_dir, f"test_log_{timestamp}.csv")

    # headers
    csv_headers = [
        "timestamp", "chunk",
        "actions_acc", "actions_f1", "actions_auc", "actions_p", "actions_r",
        "looks_acc", "looks_f1", "looks_auc", "looks_p", "looks_r",
        "crosses_acc", "crosses_f1", "crosses_auc", "crosses_p", "crosses_r",
        "overall_acc"
    ]

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # ==== Load model ====
    print(f"Loading model from {model_path}")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # Initialize base components
    motion_enc = MotionEncoder(**motion_enc_args)
    vit = ViT_Hierarchical(**vit_args)
    
    # Get model based on type selection
    model = get_model(args.model_type, motion_enc, vit, embedding_dim, num_classes_dict).to(device)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    _init_global_rel_pos_from_ckpt(model, state_dict)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Compute FLOPs and latency
    flops_per_frame = compute_flops(model, args.model_type, img_size, img_size, context_scale, device)
    fps, latency_per_frame = inference_latency(model, args.model_type, img_size, img_size, context_scale, device)

    # ==== Find test chunks ====
    chunk_files = sorted(
        [os.path.join(test_chunk_folder, f)
         for f in os.listdir(test_chunk_folder)
         if f.endswith(".lmdb")]
    )
    assert len(chunk_files) > 0, f"No .lmdb chunks found in {test_chunk_folder}"

    print(f"Found {len(chunk_files)} test chunks.")

    # ==== Process each chunk ====
    all_metrics = []
    all_labels_global, all_preds_global, all_probs_global = {}, {}, {}
    all_temporal_weights_global = []
    heads = ["actions", "looks", "crosses"]
    metric_suffixes = ["acc", "f1", "auc", "p", "r"]

    for i, chunk_path in tqdm(enumerate(chunk_files), desc= "Evaluating Chunks", total=len(chunk_files)):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}] {os.path.basename(chunk_path)}")
        start = time.time()

        dataset = LMDBChunkDataset(
            chunk_path,
            transform_tight=transform_tight,
            transform_context=transform_context,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        metrics, all_labels_chunk, all_preds_chunk, all_probs_chunk, tw_chunk = evaluate(model, dataloader, device, args.model_type, use_amp=use_amp)
        duration = time.time() - start

        for name in all_labels_chunk.keys():
            all_labels_global.setdefault(name, []).extend(all_labels_chunk[name])
            all_preds_global.setdefault(name, []).extend(all_preds_chunk[name])
            all_probs_global.setdefault(name, []).extend(all_probs_chunk[name])

        if tw_chunk is not None:
            all_temporal_weights_global.append(tw_chunk)

        metrics_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(chunk_path),
        ]

        for h in heads:
            metrics_row += [round_metric(metrics, f"{h}_{s}") for s in metric_suffixes]
        metrics_row.append(round_metric(metrics, 'overall_acc'))
        all_metrics.append(metrics_row)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row)

        print(f"  Chunk done in {duration:.2f}s")
        del dataset, dataloader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==== Compute Average Metrics ====
    avg_metrics = {}
    for name in all_labels_global.keys():
        y_true = torch.cat(all_labels_global[name]).numpy()
        y_pred = torch.cat(all_preds_global[name]).numpy()
        y_prob = torch.cat(all_probs_global[name]).numpy()

        avg_type = "binary" if y_prob.shape[1] == 2 else "macro"
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=avg_type)
        precision = precision_score(y_true, y_pred, average=avg_type)
        recall = recall_score(y_true, y_pred, average=avg_type)
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc = float("nan")

        avg_metrics[name + '_acc'] = acc
        avg_metrics[name + "_f1"] = f1 
        avg_metrics[name + "_auc"] = auc 
        avg_metrics[name + "_p"] = precision
        avg_metrics[name + "_r"] = recall

    total_correct = 0
    total_samples = 0
    for name in all_labels_global.keys():
        y_true = torch.cat(all_labels_global[name]).numpy()
        y_pred = torch.cat(all_preds_global[name]).numpy()
        total_correct += (y_true == y_pred).sum()
        total_samples += len(y_true)
    avg_metrics["overall_acc"] = total_correct / total_samples if total_samples > 0 else 0.0

    # ==== Threshold Optimization ====
    optimal_thresholds = {}
    optimized_metrics = {}
    
    print("\n=== Threshold Optimization ===")
    for name in all_labels_global.keys():
        y_true = torch.cat(all_labels_global[name]).numpy()
        y_prob = torch.cat(all_probs_global[name]).numpy()
        
        if y_prob.shape[1] == 2:
            opt_thresh, opt_f1 = find_optimal_thresholds(y_true, y_prob, name)
            optimal_thresholds[name] = opt_thresh
            
            pos_probs = y_prob[:, 1]
            opt_preds = (pos_probs >= opt_thresh).astype(int)
            
            optimized_metrics[f'{name}_f1'] = f1_score(y_true, opt_preds, average='binary', zero_division=0)
            optimized_metrics[f'{name}_p'] = precision_score(y_true, opt_preds, average='binary', zero_division=0)
            optimized_metrics[f'{name}_r'] = recall_score(y_true, opt_preds, average='binary', zero_division=0)
            optimized_metrics[f'{name}_acc'] = accuracy_score(y_true, opt_preds)
            
            print(f"  {name}: threshold={opt_thresh:.2f}, F1={opt_f1:.4f} (default: {avg_metrics.get(f'{name}_f1', 0):.4f})")
    
    optimized_metrics["overall_acc"] = (
        sum(v for k, v in optimized_metrics.items() if k.endswith("_acc")) / 3.0
    ) if any(k.endswith("_acc") for k in optimized_metrics) else 0.0

    # Summary Table
    score_row = ["Heads", "Accuracy", "F1", "AUC", "P", "R"]
    rows = [score_row]
    for h in heads:
        row = [h.capitalize()] + [round_metric(avg_metrics, f"{h}_{s}") for s in metric_suffixes]
        rows.append(row)
    # Pad to match header width (6 columns) so downstream parsers stay aligned.
    overall_row = ['Overall', round_metric(avg_metrics, 'overall_acc'), '', '', '', '']

    # Threshold-optimized summary
    opt_score_row = ["Heads (Optimized)", "Threshold", "Accuracy", "F1", "P", "R"]
    opt_rows = [opt_score_row]
    for h in heads:
        thresh = optimal_thresholds.get(h, 0.5)
        row = [h.capitalize(), thresh,
               round_metric(optimized_metrics, f"{h}_acc"),
               round_metric(optimized_metrics, f"{h}_f1"),
               round_metric(optimized_metrics, f"{h}_p"),
               round_metric(optimized_metrics, f"{h}_r")]
        opt_rows.append(row)
    opt_overall_row = ['Overall (Optimized)', '', round_metric(optimized_metrics, 'overall_acc'), '', '', '']

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    computational = [
        ['Parameters count:', f'{param_count} params'],
        ['Model Type:', args.model_type],
        ['Per-frame FLOPs:', f'{flops_per_frame/1e6:.2f} MFLOPs'],
        ['Per-frame Latency:', f'{latency_per_frame*1000:.2f} ms'],
        ['FPS Equivalent:', f'{fps:.2f}'],
    ]

    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["=== Default Threshold (0.5) ==="])
        for r in rows:
            writer.writerow(r)
        writer.writerow(overall_row)
        writer.writerow([])
        writer.writerow(["=== Threshold-Optimized ==="])
        for r in opt_rows:
            writer.writerow(r)
        writer.writerow(opt_overall_row)
        writer.writerow([])
        for r in computational:
            writer.writerow(r)

    print("\nTesting complete.")
    print("Average metrics (default threshold 0.5):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.2f}")
    print("\nThreshold-optimized metrics:")
    print(f"  Optimal thresholds: {optimal_thresholds}")
    print(f"  Overall accuracy: {optimized_metrics.get('overall_acc', 0) * 100:.2f}%")
    print(f"Results logged to: {log_csv}")
    
    # Save detailed predictions with probabilities
    if args.save_predictions:
        pred_csv = os.path.join(log_dir, f"predictions_{timestamp}.csv")
        print(f"\nSaving detailed predictions to: {pred_csv}")
        
        all_labels_flat = {}
        all_preds_flat = {}
        all_probs_flat = {}
        for name in heads:
            if all_labels_global.get(name):
                all_labels_flat[name] = torch.cat(all_labels_global[name]).numpy()
                all_preds_flat[name] = torch.cat(all_preds_global[name]).numpy()
                all_probs_flat[name] = torch.cat(all_probs_global[name]).numpy()
        
        n_samples = len(all_labels_flat.get('actions', []))
        
        with open(pred_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_idx',
                'actions_true', 'actions_prob_0', 'actions_prob_1', 'actions_pred',
                'looks_true', 'looks_prob_0', 'looks_prob_1', 'looks_pred',
                'crosses_true', 'crosses_prob_0', 'crosses_prob_1', 'crosses_pred'
            ])
            
            for i in range(n_samples):
                row = [i]
                for name in heads:
                    if name in all_labels_flat:
                        row.extend([
                            int(all_labels_flat[name][i]),
                            round(float(all_probs_flat[name][i][0]), 4),
                            round(float(all_probs_flat[name][i][1]), 4),
                            int(all_preds_flat[name][i])
                        ])
                    else:
                        row.extend(['', '', '', ''])
                writer.writerow(row)
        print(f"Predictions saved: {n_samples} samples")

        if all_temporal_weights_global:
            tw_all = np.concatenate(all_temporal_weights_global, axis=0)
            tw_path = os.path.join("plots", f"temporal_weights.npz")
            os.makedirs("plots", exist_ok=True)
            labels_for_tw = {}
            for name in heads:
                if name in all_labels_flat:
                    labels_for_tw[name] = all_labels_flat[name]
            np.savez_compressed(
                tw_path,
                temporal_weights=tw_all,
                **{f"{k}_true": v for k, v in labels_for_tw.items()},
            )
            print(f"Temporal weights saved: {tw_path} ({tw_all.shape})")

    # Save results with model type suffix
    if args.model_type != 'full':
        import shutil
        base_log = log_csv.replace('.csv', '')
        new_log = f"{base_log}_{args.model_type}.csv"
        shutil.copy2(log_csv, new_log)
        print(f"Results also copied to: {new_log}")

if __name__ == "__main__":
    main()
