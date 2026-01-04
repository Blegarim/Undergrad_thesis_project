import os
import csv
import time
import gc
import math
import re
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from fvcore.nn import FlopCountAnalysis

# ==== Model Imports ====
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from scripts.lmdb_dataset import LMDBChunkDataset
from config import vit_args_config, motion_enc_args_config
from train import remap_cross_labels, collate_fn

# ============================================================
# === Evaluation Function ====================================
# ============================================================

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = {}, {}
    all_preds, all_labels, all_probs = {}, {}, {}

    with torch.no_grad():
        for images_tight, images_context, motions, labels in dataloader:
            images_tight = images_tight.to(device, non_blocking=True)
            images_context = images_context.to(device, non_blocking=True)
            motions = motions.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images_tight, images_context, motions)

            for name, logits in outputs.items():
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
    for name in correct.keys():
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

        # print(f"    {name}: Acc={acc:.2f} | F1={f1:.2f} | AUC={metrics[name + '_auc']:.2f} | Precision={precision:.2f} | Recall={recall:.2f}")

    overall = 100.0 * sum(correct.values()) / sum(total.values())
    metrics["overall_acc"] = overall
    # print(f"    Overall Accuracy: {overall:.2f}%")
    return metrics, all_labels, all_preds, all_probs

def compute_flops(model, img_height, img_width, context_scale, device):
    dummy_imgages_tight = torch.randn(1, 20, 3, img_height, img_width).to(device)
    dummy_imgages_context = torch.randn(1, 20, 3, img_height * context_scale, img_width * context_scale).to(device)
    dummy_motions = torch.randn(1, 20, 8).to(device)
    model.eval()
    flops = FlopCountAnalysis(model, (dummy_imgages_tight, dummy_imgages_context, dummy_motions))
    flops = flops.unsupported_ops_warnings(False)
    flops_total = flops.total()
    flops_per_frame = flops_total / (dummy_imgages_tight.size(0) * dummy_imgages_tight.size(1))

    print(f'Total FLOPs per {dummy_imgages_tight.size(0) * dummy_imgages_tight.size(1)}-frame input: {flops_total/1e9:.2f} GFLOPs')
    print(f'Average FLOPs per frame: {flops_per_frame/1e6:.2f} MFLOPs\n')
    return flops_per_frame

def inference_latency(model, img_height, img_width, context_scale, device):
    dummy_imgages_tight = torch.randn(1, 20, 3, img_height, img_width).to(device)
    dummy_imgages_context = torch.randn(1, 20, 3, img_height * context_scale, img_width * context_scale).to(device)
    dummy_motions = torch.randn(1, 20, 8).to(device)
    model.eval()
    # Warm up
    for _ in range(10):
        _ = model(dummy_imgages_tight, dummy_imgages_context, dummy_motions)
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.time()
    num_trials = 50
    for _ in range(num_trials):
        _ = model(dummy_imgages_tight, dummy_imgages_context, dummy_motions)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / num_trials  # seconds per 20-frame sequence
    avg_fps = 1.0 / avg_latency
    avg_latency_per_frame = avg_latency / 20.0

    print(f"\n Inference latency (averaged over {num_trials} runs):")
    print(f"  {avg_latency*1000:.2f} ms per {dummy_imgages_tight.size(1)}-frame sequence")
    print(f"  {avg_latency_per_frame*1000:.2f} ms per frame")
    print(f"  {avg_fps:.2f} FPS equivalent\n")
    return avg_fps, avg_latency_per_frame

def round_metric(metrics, key):
    return round(metrics.get(key, 0.0), 2)

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
# ============================================================
# === Main Testing Script ====================================
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== CONFIGURATION ====
    embedding_dim = 128
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
    model_path = "best_model_outputs/best_model_epoch2_0102_1544.pth"
    test_chunk_folder = "preprocessed_test_lmdb"
    log_dir = "training_log"
    os.makedirs(log_dir, exist_ok=True)
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
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

    model = EnsembleModel(
        motion_enc=MotionEncoder(**motion_enc_args),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(
            d_model=embedding_dim,
            num_heads=4,
            num_classes_dict=num_classes_dict,
            use_frame_crosses=True,
            frame_pool="logsumexp",
        ),
    ).to(device)

    state_dict = torch.load(model_path, map_location="cpu")
    _init_global_rel_pos_from_ckpt(model, state_dict)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    flops_per_frame = compute_flops(model, img_size, img_size, context_scale, device)
    fps, latency_per_frame = inference_latency(model, img_size, img_size, context_scale, device)

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
    heads = ["actions", "looks", "crosses"]
    metric_suffixes = ["acc", "f1", "auc", "p", "r"]

    for i, chunk_path in tqdm(enumerate(chunk_files), desc= "Evaluating Chunks", total=len(chunk_files)):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}] {os.path.basename(chunk_path)}")
        start = time.time()

        dataset = LMDBChunkDataset(
            chunk_path,
            transform_tight=base_transforms,
            transform_context=base_transforms,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        metrics, all_labels_chunk, all_preds_chunk, all_probs_chunk = evaluate(model, dataloader, device)
        duration = time.time() - start

        for name in all_labels_chunk.keys():
            all_labels_global.setdefault(name, []).extend(all_labels_chunk[name])
            all_preds_global.setdefault(name, []).extend(all_preds_chunk[name])
            all_probs_global.setdefault(name, []).extend(all_probs_chunk[name])

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

    avg_metrics["overall_acc"] = (
        100 * sum(v for k, v in avg_metrics.items() if k.endswith("_acc")) / 3.0
    )

    # Summary Table
    score_row = ["Heads", "Accuracy", "F1", "AUC", "P", "R"]
    rows = [score_row]
    for h in heads:
        row = [h.capitalize()] + [round_metric(avg_metrics, f"{h}_{s}") for s in metric_suffixes]
        rows.append(row)
    overall_row = ['Overall', round_metric(avg_metrics, 'overall_acc')]
    computational = [
        'Parameters count:',
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} params',
        '',
        'Per-frame FLOPs:',
        f'{flops_per_frame/1e6:.2f} MFLOPs',
        '',
        'Per-frame Latency:',
        f'{latency_per_frame*1000:.2f} ms',
        '',
        'FPS Equivalent:',
        f'{fps:.2f}'
    ]

    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        for r in rows:
            writer.writerow(r)
        writer.writerow(overall_row)
        writer.writerow(computational)

    print("\nTesting complete.")
    print("Average metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.2f}")
    print(f"Results logged to: {log_csv}")

if __name__ == "__main__":
    main()
