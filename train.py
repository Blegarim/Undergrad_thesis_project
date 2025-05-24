import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel
from scripts.PIE_sequence_Dataset import build_dataloader, build_dataset  # Make sure this import path is correct

# Training and validation loops

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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, motions, labels) in enumerate(dataloader):
        images = images.to(device)
        motions = motions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, motions) # Shape: [batch_size, num_classes]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch_{batch_idx}, Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device)
            motions = motions.to(device)
            labels = labels.to(device)

            outputs = model(images, motions)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return epoch_loss, accuracy

# Main training function

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({device})

    # Hyperparameters
    embedding_dim = 128
    num_classes = 3
    learning_rate = 1e-4
    batch_size = 8
    sequence_length = 10
    num_epochs = 10
    val_ratio = 0.2

    train_dataset, val_dataset = build_dataset(set_ids=['set01'], sequence_length=sequence_length, crop=True, train_split=1-val_ratio, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = MultimodalModel(
        cnn_backbone = CNNFeatureExtractor(),
        motion_transformer = MotionTransformer(),
        cross_attention = CrossAttentionModule()
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=2, min_delta=0.01)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training. Do better.")
            break

if __name__ == "__main__":
    main()


