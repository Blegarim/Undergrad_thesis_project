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

'''
Training script for the PIE dataset using a multimodal model with CNN, Transformer, and Cross-Attention.
'''

# --- Collate function ---
# Collate a batch of data from the PIESequenceDataset.
def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)  # [B, T, 3, H, W]
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :3]  # [B, T, 3]
    labels = {k: torch.stack([item[k] for item in batch], dim=0) for k in ['actions', 'looks', 'crosses', 'gestures']}
    return images, motions, labels

# --- Early stopping utility ---
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

class ChunkedPIEDataset(Dataset):
    def __init__(self, chunk_folder):
        self.chunk_paths = sorted([
            os.path.join(chunk_folder, f)
            for f in os.listdir(chunk_folder) if f.endswith('.pt')
        ])
        self.index_map = []  # (chunk_id, index_within_chunk)
        self.chunk_cache = {}

        for chunk_id, path in enumerate(self.chunk_paths):
            chunk_data = torch.load(path, map_location='cpu')
            self.index_map.extend([(chunk_id, i) for i in range(len(chunk_data))])
            self.chunk_cache[chunk_id] = None  # Lazy loading

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_id, item_idx = self.index_map[idx]

        if self.chunk_cache[chunk_id] is None:
            self.chunk_cache[chunk_id] = torch.load(self.chunk_paths[chunk_id], map_location='cpu')
        chunk = self.chunk_cache[chunk_id]

        return chunk[item_idx]

def remap_cross_labels(labels):
    # Map only: -1 → 2 (irrelevant)
    crosses = labels['crosses']
    crosses = crosses.clone()  # avoid in-place mutation
    crosses[crosses == -1] = 2
    labels['crosses'] = crosses

# --- Training and validation loops ---
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
        outputs = model(images, motions)  # Dict of logits for each head

        loss = 0
        for name in outputs:
            logits = outputs[name]
            targets = labels[name][:, -1]
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
                targets = labels[name][:, -1]
                loss_i = criterion[name](logits, targets)
                batch_loss += loss_i.item()

                _, preds = torch.max(outputs[name], 1)
                correct[name] = correct.get(name, 0) + (preds == labels[name][:, -1]).sum().item()
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

def get_dataset(split):
    folder_path = f'preprocessed_{split}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Using lazy chunked loader from {folder_path}")
        return ChunkedPIEDataset(folder_path)
    else:
        print(f"Loading raw {split} set from PKL")
        sequences = load_sequences_from_pkl(f'sequences_{split}.pkl')
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        return PIESequenceDataset(sequences, transform=transform, crop=True, preload=True)


# --- Main training function ---
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    embedding_dim = 128
    learning_rate = 5e-5
    batch_size = 64
    sequence_length = 20
    num_epochs = 10
    num_workers = 4

    # Load datasets (already Dataset objects)
    train_dataset = get_dataset('train')
    val_dataset = get_dataset('val')
    print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val sequences.")

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Number of prediction classes per head
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 3,
        'gestures': 6
    }

    # Model
    model = MultimodalModel(
        cnn_backbone=CNNFeatureExtractor(backbone='efficientnet_b0', embedding_dim=embedding_dim),
        motion_transformer=MotionTransformer(d_model=embedding_dim, max_len=sequence_length, num_heads=8, num_layers=2),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=8, num_classes_dict=num_classes_dict)
    ).to(device)

    # Loss and optimizer
    criterion = {name: nn.CrossEntropyLoss() for name in num_classes_dict}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    best_val_loss = float('inf')

    os.makedirs('outputs', exist_ok=True)

    # --- Training loop ---
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Overall Accuracy: {val_acc:.4f}")

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