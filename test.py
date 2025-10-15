import os
import torch
from torch.utils.data import DataLoader
import csv, threading, time, gc
from datetime import datetime

from models.Vision_Transformer import ViT_Hierarchical
from models.Regression import TCNGRU
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from train import remap_cross_labels, PTChunkDataset


def test(model, dataloader, device):
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
                total[name] = total.get(name, 0) + labels[name].size(0)

    test_metrics = {}
    for name in correct:
        acc = 100.0 * correct[name] / total[name]
        test_metrics[name] = acc
        print(f"  {name} accuracy: {acc:.2f}%")

    overall = 100.0 * sum(correct.values()) / sum(total.values())
    test_metrics["overall"] = overall
    print(f"  Overall accuracy: {overall:.2f}%")
    return test_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Config ---
    embedding_dim = 128
    batch_size = 32
    num_workers = 4
    num_classes_dict = {"actions": 2, "looks": 2, "crosses": 3}
    model_path = "outputs/best_model_epoch5.pth"
    test_chunk_folder = "preprocessed_test"
    csv_output = f"training_log/test_metrics_summary_{datetime_str}.csv"

    # --- Load model ---
    model = EnsembleModel(
        tcngru=TCNGRU(input_dim=3, num_layers=2, kernel_size=3, dropout=0.1),
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
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict),
    ).to(device)

    assert os.path.exists(model_path), f"Model path {model_path} not found"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # --- Chunk loading helper ---
    def async_load_chunk(path, holder):
        holder["data"] = torch.load(path, map_location="cpu")

    # --- Prepare chunk list ---
    test_chunk_files = sorted(
        [os.path.join(test_chunk_folder, f) for f in os.listdir(test_chunk_folder) if f.endswith(".pt")]
    )
    assert len(test_chunk_files) > 0, "No test chunks found!"

    # --- Prepare CSV file ---
    csv_headers = ["chunk", "actions", "looks", "crosses", "overall"]
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # --- Process each chunk ---
    current_holder = {"data": torch.load(test_chunk_files[0], map_location="cpu")}
    for chunk_idx, chunk_path in enumerate(test_chunk_files):
        print(f"\n[Chunk {chunk_idx+1}/{len(test_chunk_files)}] {chunk_path}")
        start_time = time.time()

        next_holder = {}
        if chunk_idx + 1 < len(test_chunk_files):
            thread = threading.Thread(target=async_load_chunk, args=(test_chunk_files[chunk_idx + 1], next_holder))
            thread.start()
        else:
            thread = None

        dataset = PTChunkDataset([chunk_path])
        if hasattr(dataset, "data"):
            dataset.data = current_holder["data"]

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=False,
            pin_memory=True, prefetch_factor=2
        )

        metrics = test(model, dataloader, device)
        metrics_row = {"chunk": chunk_idx + 1}
        for key in ["actions", "looks", "crosses", "overall"]:
            metrics_row[key] = metrics.get(key, 0.0)

        with open(csv_output, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row.values())

        print(f"  Chunk {chunk_idx+1} done in {(time.time()-start_time):.2f}s")
        del dataset, dataloader, current_holder
        gc.collect()

        if thread:
            thread.join()
            current_holder = next_holder

    print(f"\n✅ Testing completed. Metrics saved to {csv_output}")


if __name__ == "__main__":
    main()
