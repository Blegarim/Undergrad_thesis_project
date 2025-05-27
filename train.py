import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel

from scripts.PIE_sequence_Dataset import build_dataloader, build_dataset 

'''This script trains a multimodal model on the PIE dataset.
It includes a CNN feature extractor, a motion transformer, and a cross-attention module.
The model is trained using a cross-entropy loss function and an Adam optimizer.
The training process includes early stopping based on validation loss.
'''
# Early stopping utility
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

# Training and validation loops
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, motions, labels) in enumerate(dataloader):
        loss = 0
        images = images.to(device)
        motions = motions.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()

        outputs = model(images, motions) # Shape: [batch_size, num_classes]

        for name in outputs:
            loss += criterion[name](outputs[name], labels[name])
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
    correct = {}
    total = {}

    with torch.no_grad():
        for images, motions, labels in dataloader:
            images = images.to(device)
            motions = motions.to(device)
            # Move each label tensor to device
            labels = {k: v.to(device) for k, v in labels.items()}

            outputs = model(images, motions)
            batch_loss = 0
            for name in outputs:
                loss = criterion[name](outputs[name], labels[name])
                batch_loss += loss.item()

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
            acc = 0.0
            print(f"No samples for {name}, accuracy set to 0.0")
    accuracy = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0.0

    return epoch_loss, accuracy

# Main training function
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Hyperparameters
    embedding_dim = 128
    learning_rate = 5e-5
    batch_size = 16
    sequence_length = 10
    num_epochs = 10
    val_ratio = 0.2

    train_dataset, val_dataset, label_vocab = build_dataset(set_ids=['set01'], 
                                                            sequence_length=sequence_length, 
                                                            crop=True, 
                                                            train_split=1-val_ratio, 
                                                            seed=42,
                                                            )
    print('Label vocabulary: ', label_vocab)
    
    num_classes_dict = {key: len(vocab) for key, vocab in label_vocab.items()}
    print('Number of classes per label: ', num_classes_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model initialization
    model = MultimodalModel(
        cnn_backbone = CNNFeatureExtractor(backbone='efficientnet_b0'),
        motion_transformer = MotionTransformer(),
        cross_attention = CrossAttentionModule(num_classes_dict=num_classes_dict)
    ).to(device)

    # Loss and optimizer
    criterion = {name: nn.CrossEntropyLoss() for name in num_classes_dict}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    # Initialize best validation loss
    best_val_loss = float('inf')

    os.makedirs('outputs', exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy (overall): {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), f'outputs/best_model_epoch{epoch+1}.pth')

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training. Do better.")
            torch.save(model.state_dict(), f'outputs/final_model_epoch{epoch+1}.pth')
            break

if __name__ == "__main__":
    main()


