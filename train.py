import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel

from scripts.PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset

import time
import gc

'''
Training script for the PIE dataset using a multimodal model with CNN, Transformer, and Cross-Attention.
'''

def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :3]
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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
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
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {batch_time:.3f} sec")

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = {}
    total = {}

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device)
            motions = motions.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            remap_cross_labels(labels)
            outputs = model(images, motions)
            batch_loss = 0
            for name in outputs:
                logits = outputs[name]
                targets = labels[name]
                loss_i = criterion[name](logits, targets)
                batch_loss += loss_i.item()
                _, preds = torch.max(outputs[name], 1)
                correct[name] = correct.get(name, 0) + (preds == labels[name]).sum().item()
                total[name] = total.get(name, 0) + labels[name].size(0)
            running_loss += batch_loss

    epoch_loss = running_loss / len(dataloader)
    for name in correct:
        if total[name] > 0:
            acc = correct[name] / total[name]
            print(f"Validation Accuracy for {name}: {acc:.4f}")
        else:
            print(f"No samples for {name}, accuracy set to 0.0")
    overall_acc = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0.0
    return epoch_loss, overall_acc

# Define PTChunkDataset once
class PTChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dim = 128
    learning_rate = 5e-5
    batch_size = 32
    sequence_length = 20
    num_epochs = 10
    num_workers = 0

    # Number of prediction classes per head
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 3
    }

    model = MultimodalModel(
        cnn_backbone=CNNFeatureExtractor(backbone='efficientnet_b0', embedding_dim=embedding_dim),
        motion_transformer=MotionTransformer(d_model=embedding_dim, max_len=sequence_length, num_heads=8, num_layers=2, dropout=0.3),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=8, num_classes_dict=num_classes_dict)
    ).to(device)

    checkpoint_path = 'outputs/best_model_epoch1.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f'Checkpoint {checkpoint_path} not found. Starting from scratch.')

    criterion = {name: nn.CrossEntropyLoss() for name in num_classes_dict}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    best_val_loss = float('inf')

    os.makedirs('outputs', exist_ok=True)

    # --- Training loop ---
    train_chunk_folder = 'preprocessed_train'
    train_chunk_files = sorted([os.path.join(train_chunk_folder, f) for f in os.listdir(train_chunk_folder) if f.endswith('.pt')])

    val_chunk_folder = 'preprocessed_val'
    val_chunk_files = sorted([os.path.join(val_chunk_folder, f) for f in os.listdir(val_chunk_folder) if f.endswith('.pt')])

    print(f'Total trainable parameters: {count_parameters(model)}')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        import random; random.shuffle(train_chunk_files)

        for chunk_idx, chunk_path in enumerate(train_chunk_files):
            print(f"Loading chunk {chunk_idx + 1}/{len(train_chunk_files)}: {chunk_path}")
            chunk_data = torch.load(chunk_path, map_location='cpu')
            dataset = PTChunkDataset(chunk_data)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            train_one_epoch(model, loader, criterion, optimizer, device)
            del chunk_data, dataset, loader
            gc.collect()

        # Validation (load all val data once per epoch)
        val_data = []
        for chunk_path in val_chunk_files:
            val_data.extend(torch.load(chunk_path, map_location='cpu'))
        val_dataset = PTChunkDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Overall Accuracy: {val_acc:.4f}")

        del val_data, val_dataset, val_loader
        gc.collect()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), f'outputs/best_model_epoch{epoch+1}.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Saving final model and stopping.")
            torch.save(model.state_dict(), f'outputs/final_model_epoch{epoch+1}.pth')
            break

if __name__ == "__main__":
    main()