import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models.Vision_Transformer import ViT_Hierarchical
from models.Regression import TCNGRU
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel

import time
import gc
import random
import matplotlib.pyplot as plt
import csv
from datetime import datetime

'''
Training script for the PIE dataset using a multimodal model with CNN, Transformer, and Cross-Attention.
'''

def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :4]
    labels = {k: torch.stack([item[k] for item in batch], dim=0) for k in ['actions', 'looks', 'crosses']}
    return images, motions, labels

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def remap_cross_labels(labels):
    crosses = labels['crosses']
    crosses = crosses.clone()
    crosses[crosses == -1] = 2
    labels['crosses'] = crosses

def train_one_chunk(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, motions, labels) in enumerate(dataloader):
        start_time = time.time()
        images = images.to(device)
        motions = motions.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        remap_cross_labels(labels)
        optimizer.zero_grad()
        outputs = model(images, motions)
        loss = 0
        for name in outputs:
            logits = outputs[name]
            targets = labels[name]
            loss += criterion[name](logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        end_time = time.time()
        batch_time = end_time - start_time
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Time per batch: {batch_time:.3f} sec")

    avg_loss = total_loss / len(dataloader)
    print(f"Average chunk Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = {}
    total = {}

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device)
            motions = motions.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images, motions)

            batch_loss = 0.0
            for name in outputs:  # 'actions', 'looks', 'crosses'
                logits = outputs[name]
                targets = labels[name]
                loss_i = criterion[name](logits, targets)
                batch_loss += loss_i.item()

                _, preds = torch.max(logits, 1)
                correct[name] = correct.get(name, 0) + (preds == targets).sum().item()
                total[name] = total.get(name, 0) + targets.size(0)

            running_loss += batch_loss

    # --- Compute metrics ---
    epoch_loss = running_loss / len(dataloader)

    val_metrics = {}
    for name in correct:
        acc = correct[name] / total[name] if total[name] > 0 else 0.0
        val_metrics[name] = acc
        print(f"Validation Accuracy for {name}: {acc:.4f}")

    overall_acc = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0.0
    val_metrics["overall"] = overall_acc

    print(f"Overall Validation Accuracy: {overall_acc:.4f}")
    print(f"Validation Loss: {epoch_loss:.4f}")

    return epoch_loss, val_metrics


# Define PTChunkDataset once
class PTChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def finetune(model, enable_finetune=False):
    if not enable_finetune:
        return
    for name, param in model.named_parameters():
        param.requires_grad = False
        if ('cross_attention' in name) or ('classifier' in name) or ('cross_attn' in name):
            param.requires_grad = True


def gather_chunks(folders):
    if isinstance(folders, str):
        folders = [folders]
    all_files = []
    for folder in folders:
        chunk_files = sorted([os.path.join(folder, f) 
                            for f in os.listdir(folder) 
                            if f.endswith('.pt')])
        all_files.extend(chunk_files)
    
    return all_files

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datetime_str = datetime.now().strftime("%m%d_%H%M")
    log_file = f'training_log/training_log_{datetime_str}.csv'

    os.makedirs('training_log', exist_ok=True)
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Epoch',
            'Avg Train Loss',
            'Actions Acc',
            'Looks Acc',
            'Crosses Acc',
            'Val Loss',
            'Overall Val Acc'
        ])

    print(f'Logging training progress to {log_file}')

    # Configuration
    embedding_dim = 128
    learning_rate = 1e-5
    batch_size = 32
    vit_args = dict(
        img_size=128,
        in_channels=3,
        stage_dims=[48, 96, 168],
        layer_nums=[2, 4, 5],
        head_nums=[2, 4, 7],
        window_size=[8, 4, None],
        mlp_ratio=[4, 4, 4],
        drop_path=0.15,
        attn_dropout=0.15,
        proj_dropout=0.15,
        dropout=0.15
    )
    sequence_length = 20
    num_epochs = 10

    num_workers = 2

    # Number of prediction classes per head
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 3
    }

    model = EnsembleModel(
        tcngru=TCNGRU(input_dim=4, num_layers=2, kernel_size=3, dropout=0.1),
        vit=ViT_Hierarchical(**vit_args),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict)
    ).to(device)

    # Load model
    checkpoint_path = 'outputs/best_model_epoch.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f'Checkpoint {checkpoint_path} not found. Starting from scratch.')

    finetune(model, enable_finetune=False)

    # Class weighting for 'looks' labels. Criterion, optimizer, scheduler
    y = np.array([0]*45000 + [1]*5000)
    looks_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    criterion = {
        name: (
            nn.CrossEntropyLoss(weight=torch.tensor(looks_weight, dtype=torch.float).to(device))
            if name == 'looks'
            else nn.CrossEntropyLoss()
        )
        for name in num_classes_dict
    }
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=0.0001, threshold_mode='rel'
    )

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    best_val_loss = float('inf')

    os.makedirs('outputs', exist_ok=True)

    # --- Training loop ---
    train_chunk_folder = ['preprocessed_train_base', 'preprocessed_train_augmented']
    val_chunk_folder = 'preprocessed_val_base'
    train_chunk_files = gather_chunks(train_chunk_folder)
    val_chunk_files = gather_chunks(val_chunk_folder)

    print(f'Total trainable parameters: {count_parameters(model)}')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_chunk_files)
        epoch_loss = []

        for chunk_idx, chunk_path in enumerate(train_chunk_files):
            print(f"\n[Chunk {chunk_idx + 1}/{len(train_chunk_files)}] Loading from {chunk_path}")
            try:
                current_data = torch.load(chunk_path, map_location='cpu')
            except Exception as e:
                print(f"Failed to load chunk {chunk_path}: {e}")
                continue

            if not current_data:
                print(f"Chunk {chunk_path} is empty, skipping.")
                continue

            dataset = PTChunkDataset(current_data)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers, 
                collate_fn=collate_fn,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=1
            )

            print(f"→ Training {len(loader)} batches in this chunk")
            avg_loss = train_one_chunk(model, loader, criterion, optimizer, device)
            epoch_loss.append(avg_loss)

            del current_data, dataset, loader
            gc.collect()
            torch.cuda.empty_cache()

        # ---- end of chunks ----
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # ---- validation ----
        val_data = []
        for chunk_path in val_chunk_files:
            val_data.extend(torch.load(chunk_path, map_location='cpu'))

        val_dataset = PTChunkDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False
        )

        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                round(avg_epoch_loss, 4),
                round(val_metric.get('actions', 0.0), 4),
                round(val_metric.get('looks', 0.0), 4),
                round(val_metric.get('crosses', 0.0), 4),
                round(val_loss, 4),
                round(val_metric.get('overall', 0.0), 4)
            ])

        del val_data, val_dataset, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), f'outputs/best_model_epoch{epoch+1}_{datetime_str}.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Saving final model and stopping.")
            torch.save(model.state_dict(), f'outputs/final_model_epoch{epoch+1}_{datetime_str}.pth')
            break


if __name__ == "__main__":
    main()