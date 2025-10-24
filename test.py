import os
import csv
import time
import gc
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from fvcore.nn import FlopCountAnalysis
import builtins

# ==== Model Imports ====
from models.Vision_Transformer import ViT_Hierarchical
from models.Regression import TCNGRU
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from train import remap_cross_labels, filter_irrelevant


# ============================================================
# === Dataset for preprocessed .pt chunks ====================
# ============================================================

class PTChunkDataset(Dataset):
    """
    For preprocessed .pt chunks that store dict samples:
    {
        'images': Tensor[T, C, H, W],
        'motions': Tensor[T, 4],
        'bboxes': ...,
        'actions': Tensor,
        'looks': Tensor,
        'crosses': Tensor
    }
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict):
            images = sample["images"]
            motions = sample["motions"]
            labels = {
                "actions": sample["actions"],
                "looks": sample["looks"],
                "crosses": sample["crosses"],
            }
            return images, motions, labels
        elif isinstance(sample, (list, tuple)) and len(sample) == 3:
            return sample  # backward compatibility
        else:
            raise ValueError(f"Unexpected sample structure at idx {idx}: {type(sample)}")


# ============================================================
# === Evaluation Function ====================================
# ============================================================

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = {}, {}
    all_preds, all_labels, all_probs = {}, {}, {}

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device, non_blocking=True)
            motions = motions.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images, motions)

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

        print(f"    {name}: Acc={acc:.2f} | F1={f1:.2f} | AUC={metrics[name + '_auc']:.2f} | Precision={precision:.2f} | Recall={recall:.2f}")

    overall = 100.0 * sum(correct.values()) / sum(total.values())
    metrics["overall_acc"] = overall
    print(f"    Overall Accuracy: {overall:.2f}%")
    return metrics, all_labels, all_preds, all_probs

def compute_flops(model, images, motions):
    model.eval()
    flops = FlopCountAnalysis(model, (images, motions))
    flops = flops.unsupported_ops_warnings(False)
    flops_total = flops.total()
    flops_per_frame = flops_total / (images.size(0) * images.size(1))

    print(f'Total FLOPs per {images.size(0) * images.size(1)}-frame input: {flops_total/1e9:.2f} GFLOPs')
    print(f'Average FLOPs per frame: {flops_per_frame/1e6:.2f} MFLOPs\n')
    return flops_per_frame

def inference_latency(model, images, motions):
    model.eval()
    # Warm up
    for _ in range(10):
        _ = model(images, motions)
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.time()
    num_trials = 50
    for _ in range(num_trials):
        _ = model(images, motions)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / num_trials  # seconds per 20-frame sequence
    avg_fps = 1.0 / avg_latency
    avg_latency_per_frame = avg_latency / 20.0

    print(f"\n Inference latency (averaged over {num_trials} runs):")
    print(f"  {avg_latency*1000:.2f} ms per {images.size(1)}-frame sequence")
    print(f"  {avg_latency_per_frame*1000:.2f} ms per frame")
    print(f"  {avg_fps:.2f} FPS equivalent\n")
    return avg_fps, avg_latency_per_frame

def round_metric(metrics, key):
    return round(metrics.get(key, 0.0), 2)
# ============================================================
# === Main Testing Script ====================================
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== CONFIGURATION ====
    embedding_dim = 128
    batch_size = 32
    img_size = 160
    vit_args = dict(
        img_size=img_size, 
        in_channels=3,
        stage_dims=[48, 96, 168],
        layer_nums=[2, 4, 5],
        head_nums=[2, 4, 7],
        window_size=[8, 4, None],
        mlp_ratio=[4, 4, 4],
        drop_path=0.15, 
        attn_dropout=0.1,
        proj_dropout=0.1, 
        dropout=0.15,
    )
    num_workers = 4
    num_classes_dict = {"actions": 2, "looks": 2, "crosses": 2}
    model_path = "outputs/final_model_epoch5_1023_1349.pth"
    test_chunk_folder = "preprocessed_test_128"
    log_dir = "training_log"
    os.makedirs(log_dir, exist_ok=True)

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
        tcngru=TCNGRU(input_dim=4, num_layers=2, kernel_size=3, dropout=0.1),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(
            d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict
        ),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # ==== Computing FLOPs/Inference Latency (dummy input) ====
    dummy_images = torch.randn(1, 20, 3, img_size, img_size).to(device)
    dummy_motions = torch.randn(1, 20, 4).to(device)

    flops_per_frame = compute_flops(model, dummy_images, dummy_motions)
    fps, latency_per_frame = inference_latency(model, dummy_images, dummy_motions)

    # ==== Find test chunks ====
    chunk_files = sorted(
        [os.path.join(test_chunk_folder, f)
         for f in os.listdir(test_chunk_folder)
         if f.endswith(".pt")]
    )
    assert len(chunk_files) > 0, f"No .pt chunks found in {test_chunk_folder}"

    print(f"Found {len(chunk_files)} test chunks.")

    # ==== Process each chunk ====
    all_metrics = []
    all_labels_global, all_preds_global, all_probs_global = {}, {}, {}

    for i, chunk_path in enumerate(chunk_files):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}] {os.path.basename(chunk_path)}")
        start = time.time()

        chunk_data = torch.load(chunk_path, map_location="cpu")
        chunk_data = filter_irrelevant(chunk_data)
        dataset = PTChunkDataset(chunk_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
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
            round_metric(metrics, 'actions_acc'),
            round_metric(metrics, 'actions_f1'),
            round_metric(metrics, 'actions_auc'),
            round_metric(metrics, 'actions_p'),
            round_metric(metrics, 'actions_r'),
            round_metric(metrics, 'looks_acc'),
            round_metric(metrics, 'looks_f1'),
            round_metric(metrics, 'looks_auc'),
            round_metric(metrics, 'looks_p'),
            round_metric(metrics, 'looks_r'),
            round_metric(metrics, 'crosses_acc'),
            round_metric(metrics, 'crosses_f1'),
            round_metric(metrics, 'crosses_auc'),
            round_metric(metrics, 'crosses_p'),
            round_metric(metrics, 'crosses_r'),
            round_metric(metrics, 'overall_acc'),
        ]
        all_metrics.append(metrics_row)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row)

        print(f"  Chunk done in {duration:.2f}s")
        del dataset, dataloader, chunk_data
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

    avg_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'AVERAGE',
            round_metric(avg_metrics, 'actions_acc'),
            round_metric(avg_metrics, 'actions_f1'),
            round_metric(avg_metrics, 'actions_auc'),
            round_metric(avg_metrics, 'actions_p'),
            round_metric(avg_metrics, 'actions_r'),
            round_metric(avg_metrics, 'looks_acc'),
            round_metric(avg_metrics, 'looks_f1'),
            round_metric(avg_metrics, 'looks_auc'),
            round_metric(avg_metrics, 'looks_p'),
            round_metric(avg_metrics, 'looks_r'),
            round_metric(avg_metrics, 'crosses_acc'),
            round_metric(avg_metrics, 'crosses_f1'),
            round_metric(avg_metrics, 'crosses_auc'),
            round_metric(avg_metrics, 'crosses_p'),
            round_metric(avg_metrics, 'crosses_r'),
            round_metric(avg_metrics, 'overall_acc'),
        ]

    computational = [
        'Parameters count:',
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} params',
        '',
        'Per-frame FLOPs:',
        f'{flops_per_frame/1e6:2f} MFLOPs',
        '',
        'Per-frame Latency:',
        f'{latency_per_frame*1000:.2f} ms',
        '',
        'FPS Equivalent:',
        f'{fps:.2f}'
    ]

    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow(avg_row)
        csv.writer(f).writerow(computational)

    print("\n✅ Testing complete.")
    print("Average metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.2f}")
    print(f"Results logged to: {log_csv}")

if __name__ == "__main__":
    main()
