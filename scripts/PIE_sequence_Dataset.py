from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path
import sys
from ComputeMotionFeatures import compute_motion_features

# Add PIE root directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# PIE-specific imports
from PIE.utilities.pie_data import PIE
from generate_sequences import generate_all_sequences

class PIESequenceDataset(Dataset):
    """
    Custom Dataset for loading sequences of pedestrian crops from PIE dataset.
    Each sample is a sequence of cropped images and a label (default: last frame label).
    """
    def __init__(self, sequences, transform=None, crop=True, include_motion_deltas=False, label_key=None):
        self.sequences = sequences
        self.transform = transform
        self.crop = crop
        self.include_motion_deltas = include_motion_deltas
        self.label_key = label_key

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        images = []

        for img_path, bbox in zip(sequence['image_paths'], sequence['bboxes']):
            img = Image.open(img_path).convert('RGB')

            if self.crop:
                x1, y1, x2, y2 = map(int, bbox)
                img = img.crop((x1, y1, x2, y2))

            if self.transform:
                img = self.transform(img)

            images.append(img)

        # Shape: [seq_len, C, H, W]
        images = torch.stack(images)

        # Compute motion features tensor [seq_len, 2 or 3]
        motions = compute_motion_features(
            sequence['bboxes'],
            sequence['frame_numbers'],
            include_frame_deltas=self.include_motion_deltas
        )

        # Use last-frame label by default (can be changed later)
        labels_dict = sequence['labels']
        if self.label_key:
            label = torch.tensor(labels_dict[self.label_key][-1], dtype=torch.long)
        else:
            label = {
                key: torch.tensor(labels_dict[key][-1], dtype=torch.long) for key in labels_dict
            }

        return images, motions, label

def build_dataset(set_ids=['set01'], sequence_length=10, crop=True, train_split=0.8, seed=42):
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    data_root = ROOT_DIR / 'data'
    pie = PIE(data_path=str(data_root))
    pie.set_ids = set_ids

    database = pie.generate_database()
    train_sequences, val_sequences = generate_all_sequences(database, pie, sequence_length=sequence_length,train_split=train_split, seed=seed)

    train_dataset = PIESequenceDataset(
        sequences=train_sequences,
        transform=transform,
        crop=crop,
        include_motion_deltas=True
    )
    val_dataset = PIESequenceDataset(
        sequences=val_sequences,
        transform=transform,
        crop=crop,
        include_motion_deltas=True
    )

    return train_dataset, val_dataset



def build_dataloader(set_ids=['set01'], sequence_length=10, batch_size=8, crop=True):
    """
    Build train and validation DataLoaders from datasets.
    """
    train_dataset, val_dataset = build_dataset(set_ids=set_ids, sequence_length=sequence_length, crop=crop)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = build_dataloader()
    print("Type of dataloader:", type(train_loader))
    print("Type of first element in dataloader:", type(next(iter(train_loader))))

    for batch in train_loader:
        imgs, motions, labels = batch  # labels is now a dict

        print("Images shape:", imgs.shape)   # Expected: [B, T, 3, 128, 128]
        print("Motion shape:", motions.shape) # Expected: [B, T, 3]

        # Labels is a dict of tensors: { 'crossing': [B], 'looking': [B], ... }
        print("Label keys:", labels.keys())
        for key, value in labels.items():
            print(f"{key} label shape: {value.shape}")

        print("Example motion features (first sample):", motions[0])
        break


