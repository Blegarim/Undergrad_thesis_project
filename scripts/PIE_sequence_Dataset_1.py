import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
from pathlib import Path
import numpy as np

def load_sequences_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        sequences = pickle.load(f)
    print("Number of sequences loaded:", len(sequences))
    print("Available keys in one sample:", sequences[0].keys())
    return sequences


class PIESequenceDataset(Dataset):
    def __init__(self, sequences, transform=None, crop=True, 
                 return_metadata=False):
        self.sequences = sequences
        self.transform = transform
        self.crop = crop
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        images = []
        for img_path, bbox in zip(seq['images'], seq['bboxes']):
            img_path = Path(img_path)
            if not img_path.exists():
                alt_path = img_path.with_suffix('.jpg')
                if alt_path.exists():
                    img_path = alt_path
                else:
                    raise FileNotFoundError(f"Image not found: {img_path} or {alt_path}")
            img = Image.open(img_path).convert('RGB')
            if self.crop:
                x1, y1, x2, y2 = map(int, bbox)
                img = img.crop((x1, y1, x2, y2))
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images, dim=0)

        # Extract labels
        actions = seq['actions']
        looks = seq['looks']
        crosses = seq['crosses']
        gestures = seq['gestures']

        # Compute motions (center coordinates + deltas)
        centers = []
        for bbox in seq['bboxes']:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
        centers = torch.tensor(centers, dtype=torch.float32)
        dt = centers[1:] - centers[:-1]
        dt = torch.cat([dt[0:1], dt], dim=0)
        motions = torch.cat([centers, dt], dim=1)  # [T, 4]

        sample = {
            'images': images,   # Tensor [T, C, H, W]
            'motions': motions, # Tensor [T, 4] (cx, cy, dx, dy)
            'bboxes': seq['bboxes'],
            'actions': torch.tensor(actions, dtype=torch.long),
            'looks': torch.tensor(looks, dtype=torch.long),
            'crosses': torch.tensor(crosses, dtype=torch.long),
            'gestures': torch.tensor(gestures, dtype=torch.long),
        }

        if self.return_metadata:
            sample['meta'] = {
               'ped_id': seq.get('ped_id', None),
               'video_id': seq.get('video_id', None) 
            }

        print(f"idx={idx} len(images)={len(images)}, len(bboxes)={len(seq['bboxes'])}, len(actions)={len(actions)}, motions.shape={motions.shape}")

        return sample
    
def pad_sequence_tensor(tensor_list, pad_value=0):
    """
    Pads a list of tensors [T_i, ...] into [B, T_max, ...] along the first dimension.
    Works for any shape: [T], [T, D], [T, C, H, W], etc.
    """
    max_len = max(t.shape[0] for t in tensor_list)
    batch = []
    for t in tensor_list:
        pad_len = max_len - t.shape[0]
        if pad_len > 0:
            pad_shape = (pad_len,) + t.shape[1:]
            pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        batch.append(t_padded)
    return torch.stack(batch)

def collate_with_padding(batch):
    """
    Custom collate function to pad variable-length sequences
    """
    images = pad_sequence_tensor([item['images'] for item in batch])
    motions = pad_sequence_tensor([item['motions'] for item in batch])
    actions = pad_sequence_tensor([item['actions'] for item in batch])
    looks = pad_sequence_tensor([item['looks'] for item in batch])
    crosses = pad_sequence_tensor([item['crosses'] for item in batch])
    gestures = pad_sequence_tensor([item['gestures'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]  # Leave bboxes unpadded
    meta = [item['meta'] for item in batch] if 'meta' in batch[0] else None

    out = {
        'images': images,     # [B, T, C, H, W]
        'motions': motions,  # [B, T, 4] (cx, cy, dx, dy)
        'actions': actions,   # [B, T]
        'looks': looks,
        'crosses': crosses,
        'gestures': gestures,
        'bboxes': bboxes,
    }
    if meta:
        out['meta'] = meta
    return out   

def build_dataloader(sequences, batch_size=32, shuffle=True, transform=None, crop=True, pad=False):
    """
    sequences: list of dicts loaded from your PKL
    batch_size: number of sequences per batch
    shuffle: shuffle dataset each epoch
    transform: torchvision transform to apply to each image (optional)
    crop: whether to crop images to bbox (default True)
    Returns: PyTorch DataLoader
    """
    dataset = PIESequenceDataset(sequences, transform=transform, crop=crop)
    collate_fn = collate_with_padding if pad else (lambda x: x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader