import sys
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from scripts.ComputeMotionFeatures import compute_motion_features
from scripts.vocab_utils import build_label_vocab, encode_labels, decode_labels
from scripts.generate_sequences import generate_all_sequences

from pathlib import Path
# Add PIE root directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# PIE-specific imports
from PIE.utilities.pie_data import PIE


class PIESequenceDataset(Dataset):
    """
    Dataset for PIE sequences with label encoding.

    Args:
        sequences: list of dicts, each with keys 'image_paths', 'bboxes', 'labels' etc.
        label_vocab: dict of {label_key: {label_str: idx}} for encoding labels
        label_key: str or None, which label to return (if None, return dict of all)
        transform: torchvision transforms to apply to crops
        crop: bool, whether to crop pedestrians
        include_motion_deltas: bool, whether to compute motion features with dt
    """
    def __init__(self, 
                 sequences, 
                 label_vocab=None, 
                 label_key=None, 
                 transform=None, 
                 crop=True, 
                 include_motion_deltas=False, 
                 label_mode='final'):
        
        self.sequences = sequences
        self.label_vocab = label_vocab
        self.label_key = label_key
        self.transform = transform
        self.crop = crop
        self.include_motion_deltas = include_motion_deltas
        self.label_mode = label_mode

        if self.label_vocab:
            self._encode_all_labels()

    def _encode_all_labels(self):
        for seq in self.sequences:
            encoded_labels = {}
            for key, labels_list in seq['labels'].items():
                if key in self.label_vocab:
                    encoded_labels[key] = encode_labels(labels_list, self.label_vocab[key])  # Encode labels using the vocab
                else:
                    raise ValueError(f"No vocabulary found for label key: {key}")
            seq['encoded_labels'] = encoded_labels
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        images = []

        for img_path, bbox in zip(seq['image_paths'], seq['bboxes']):
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
            seq['bboxes'],
            seq['frame_numbers'],
            include_frame_deltas=self.include_motion_deltas
        )

        # Convert motions to tensor
        if self.label_key:
            if self.label_vocab:
                label = torch.tensor(seq['encoded_labels'][self.label_key], dtype=torch.long)
            else:
                label = torch.tensor(seq['labels'][self.label_key], dtype=torch.long)
        else:
            if self.label_vocab:
                label = {key: torch.tensor(vals, dtype=torch.long) for key, vals in seq['encoded_labels'].items()}
            else:
                label = {key: torch.tensor(vals, dtype=torch.long) for key, vals in seq['labels'].items()}

        # Handle label_mode
        if self.label_mode == 'final':
            if isinstance(label, dict):
                label = {k: v[-1] for k, v in label.items()}
            else:
                label = label[-1]


        return images, motions, label

def build_dataset(set_ids=['set01'], sequence_length=10, crop=True, train_split=0.8, seed=42, label_keys=None):
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    data_root = ROOT_DIR / 'data'
    pie = PIE(data_path=str(data_root))
    pie.set_ids = set_ids

    database = pie.generate_database()
    train_sequences, val_sequences = generate_all_sequences(database, pie, sequence_length=sequence_length,train_split=train_split, seed=seed)

    if label_keys is None:
        label_keys = ['gesture', 'look', 'action', 'cross']

    label_vocab = build_label_vocab(train_sequences, behavior_keys=label_keys)

    train_dataset = PIESequenceDataset(
        sequences=train_sequences,
        label_vocab=label_vocab,
        transform=transform,
        crop=crop,
        include_motion_deltas=True
    )

    val_dataset = PIESequenceDataset(
        sequences=val_sequences,
        label_vocab=label_vocab,
        transform=transform,
        crop=crop,
        include_motion_deltas=True
    )

    return train_dataset, val_dataset, label_vocab

def build_dataloader(set_ids=['set01'], sequence_length=10, batch_size=8, crop=True, label_keys=None):
    """
    Build train and validation DataLoaders from datasets.
    """
    train_dataset, val_dataset, label_vocab = build_dataset(set_ids=set_ids, sequence_length=sequence_length, crop=crop, label_keys=label_keys)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, label_vocab

def sanity_check():
    train_loader, val_loader, _ = build_dataloader()

    for images, motions, labels, in train_loader:
        print("Images:", images.shape)    # [B, T, 3, 128, 128]
        print("Motion:", motions.shape)   # [B, T, 3]

        if isinstance(labels, dict):
            for k, v in labels.items():
                print(f"{k} label shape: {v.shape}")  # [B, T] if per-frame, [B] if final
        else:
            print("Labels:", labels.shape)

        break



