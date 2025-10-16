import os
import csv
import time
import gc
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

# ==== Model Imports ====
from models.Vision_Transformer import ViT_Hierarchical
from models.Regression import TCNGRU
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from train import remap_cross_labels


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

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device, non_blocking=True)
            motions = motions.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images, motions)

            for name in outputs.keys():
                _, preds = torch.max(outputs[name], 1)
                correct[name] = correct.get(name, 0) + (preds == labels[name]).sum().item()
                total[name] = total.get(name, 0) + labels[name].numel()

    metrics = {}
    for name in correct:
        acc = 100.0 * correct[name] / total[name]
        metrics[name] = acc
        print(f"    {name} accuracy: {acc:.2f}%")

    overall = 100.0 * sum(correct.values()) / sum(total.values())
    metrics["overall"] = overall
    print(f"    Overall accuracy: {overall:.2f}%")
    return metrics


# ============================================================
# === Main Testing Script ====================================
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== Config ====
    embedding_dim = 128
    batch_size = 32
    num_workers = 4
    num_classes_dict = {"actions": 2, "looks": 2, "crosses": 3}
    model_path = "outputs/best_model_epoch3.pth"
    test_chunk_folder = "preprocessed_test"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # ==== Prepare log file ====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_csv = os.path.join(log_dir, f"test_log_{timestamp}.csv")
    csv_headers = ["timestamp", "chunk", "actions", "looks", "crosses", "overall", "duration_sec"]

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # ==== Load model ====
    print(f"Loading model from {model_path}")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = EnsembleModel(
        tcngru=TCNGRU(input_dim=4, num_layers=2, kernel_size=3, dropout=0.1),
        vit=ViT_Hierarchical(
            img_size=128, in_channels=3,
            stage_dims=[64, 128, 224],
            layer_nums=[2, 4, 5],
            head_nums=[2, 4, 7],
            window_size=[8, 4, None],
            mlp_ratio=[4, 4, 4],
            drop_path=0.15, attn_dropout=0.1,
            proj_dropout=0.1, dropout=0.15,
        ),
        cross_attention=CrossAttentionModule(
            d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict
        ),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

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
    total_start = time.time()

    for i, chunk_path in enumerate(chunk_files):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}] {os.path.basename(chunk_path)}")
        start = time.time()

        # Load the chunk
        chunk_data = torch.load(chunk_path, map_location="cpu")
        dataset = PTChunkDataset(chunk_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        metrics = evaluate(model, dataloader, device)
        duration = time.time() - start

        metrics_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(chunk_path),
            round(metrics.get("actions", 0.0), 2),
            round(metrics.get("looks", 0.0), 2),
            round(metrics.get("crosses", 0.0), 2),
            round(metrics.get("overall", 0.0), 2),
            round(duration, 2),
        ]
        all_metrics.append(metrics_row)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row)

        print(f"  Chunk done in {duration:.2f}s")
        del dataset, dataloader, chunk_data
        gc.collect()

    # ==== Compute Average Metrics ====
    avg_actions = sum(float(m[2]) for m in all_metrics) / len(all_metrics)
    avg_looks = sum(float(m[3]) for m in all_metrics) / len(all_metrics)
    avg_crosses = sum(float(m[4]) for m in all_metrics) / len(all_metrics)
    avg_overall = sum(float(m[5]) for m in all_metrics) / len(all_metrics)
    total_time = (time.time() - total_start) / 60

    avg_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AVERAGE",
        round(avg_actions, 2),
        round(avg_looks, 2),
        round(avg_crosses, 2),
        round(avg_overall, 2),
        round(total_time * 60, 2),
    ]

    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow(avg_row)

    print("\n✅ Testing complete.")
    print(f"Average Accuracies - Actions: {avg_actions:.2f}%, Looks: {avg_looks:.2f}%, Crosses: {avg_crosses:.2f}%, Overall: {avg_overall:.2f}%")
    print(f"Total time: {total_time:.2f} min")
    print(f"Results logged to: {log_csv}")


if __name__ == "__main__":
    main()
