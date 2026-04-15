import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    jpeg = TurboJPEG(lib_path="C:\\libjpeg-turbo64\\bin\\turbojpeg.dll")
except Exception:
    jpeg = None
    TJPF_RGB = None
from pathlib import Path
from tqdm import tqdm

def load_sequences_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        sequences = pickle.load(f)
    print("Number of sequences loaded:", len(sequences))
    print("Available keys in one sample:", sequences[0].keys())
    return sequences

class PIESequenceDataset(Dataset):
    def __init__(self, sequences, 
                 transform_tight=None, transform_context=None, 
                 crop=True, context_scale=2.0,
                 return_metadata=False, preload=True):
        self.transform_tight = transform_tight
        self.transform_context = transform_context
        self.crop = crop
        self.context_scale = context_scale
        self.return_metadata = return_metadata
        self.preload = preload
        
        if self.preload:
            print("Preloading images into memory...")
            self.data = []
            for i, seq in enumerate(tqdm(sequences, desc="Preloading")):
                self.data.append(self._process_sequence(seq))
            print(f"Finished preloading {len(self.data)} sequences.")
        else:
            self.sequences = sequences

    def _process_sequence(self, seq):
        images_tight, images_context = [], []
        centers, width, height = [], [], []
        for img_path, bbox in zip(seq['images'], seq['bboxes']):
            img_path = Path(img_path)
            if not img_path.exists():
                candidates = [img_path.with_suffix('.jpg')]
                if img_path.stem.isdigit():
                    padded = img_path.stem.zfill(6)
                    candidates.append(img_path.with_name(padded + img_path.suffix))
                    candidates.append(img_path.with_name(padded + '.jpg'))
                for alt_path in candidates:
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    raise FileNotFoundError(
                        f"Image not found: {img_path} or any of {candidates}"
                    )
            try:
                if jpeg is not None:
                    with open(img_path, 'rb') as in_file:
                        buff = in_file.read()
                    img_array = jpeg.decode(buff, pixel_format=TJPF_RGB)
                    img = Image.fromarray(img_array)
                else:
                    raise Exception("JPEG decoder not available")
            except Exception:
                img = Image.open(img_path).convert('RGB')
            if self.crop:
                x1, y1, x2, y2 = map(int, bbox)
                tight = img.crop((x1, y1, x2, y2))
            scale = self.context_scale
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w2, h2 = w * scale, h * scale
            x1c, y1c, x2c, y2c = cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2
            # Clamp to image bounds
            x1c = max(0, x1c); y1c = max(0, y1c)
            x2c = min(img.width, x2c); y2c = min(img.height, y2c)
            context = img.crop((x1c, y1c, x2c, y2c))
            if self.transform_tight:
                tight = self.transform_tight(tight)
            if self.transform_context:
                context = self.transform_context(context)
            
            images_tight.append(tight)
            images_context.append(context)

            centers.append([cx, cy])
            width.append(w)
            height.append(h)

        centers = torch.tensor(centers, dtype=torch.float32)
        widths = torch.tensor(width, dtype=torch.float32)
        heights = torch.tensor(height, dtype=torch.float32)

        # --- Compute motion deltas (dx, dy, dw, dh) ---
        dt = centers[1:] - centers[:-1]
        dt = torch.cat([dt[0:1], dt], dim=0)
        dw = torch.cat([widths[0:1], widths[1:] - widths[:-1]], dim=0)
        dh = torch.cat([heights[0:1], heights[1:] - heights[:-1]], dim=0)

        motions = torch.cat([centers, dt, widths.unsqueeze(1),
                            heights.unsqueeze(1),
                            dw.unsqueeze(1), dh.unsqueeze(1)], dim=1)
        # [T, 8] → (cx, cy, dx, dy, w, h, dw, dh)

        images_tight = torch.stack(images_tight, dim=0)
        images_context = torch.stack(images_context, dim=0)

        sample = {
            'images_tight': images_tight,               # Tensor [T, C, H, W]
            'images_context': images_context,           # Tensor [T, C, H, W]
            'motions': motions,                         # Tensor [T, 8] (cx, cy, dx, dy, w, h, dw, dh)
            'bboxes': seq['bboxes'],
            'actions': torch.tensor(seq['actions'], dtype=torch.long),
            'looks': torch.tensor(seq['looks'], dtype=torch.long),
            'crosses': torch.tensor(seq['crosses'], dtype=torch.long),
        }

        if self.return_metadata:
            sample['meta'] = {
            'ped_id': seq.get('ped_id', None),
            'video_id': seq.get('video_id', None)
            }

        return sample

    def __len__(self):
        if self.preload:
            return len(self.data)
        else:
            return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            return self._process_sequence(self.sequences[idx])

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
            pad = torch.full(pad_shape, pad_value, dtype=t.dtype)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        batch.append(t_padded)
    return torch.stack(batch)

def collate_with_padding(batch):
    """
    Custom collate function to pad variable-length sequences
    """
    images_tight = pad_sequence_tensor([item['images_tight'] for item in batch])
    images_context = pad_sequence_tensor([item['images_context'] for item in batch])
    motions = pad_sequence_tensor([item['motions'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    looks = torch.stack([item['looks'] for item in batch])
    crosses = torch.stack([item['crosses'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]  # Leave bboxes unpadded
    meta = [item['meta'] for item in batch] if 'meta' in batch[0] else None

    out = {
        'images_tight': images_tight,     # [B, T, C, Ht, Wt]
        'images_context': images_context, # [B, T, C, Hc, Wc]
        'motions': motions,  # [B, T, 8] (cx, cy, dx, dy, w, h, dw, dh)
        'actions': actions,  # [B, 1]
        'looks': looks,      # [B, 1]
        'crosses': crosses,  # [B, 1]
        'bboxes': bboxes,
    }
    if meta:
        out['meta'] = meta
    return out   

def build_dataloader(sequences, batch_size=32, shuffle=True, transform=None, crop=True, pad=False, preload=False):
    """
    sequences: list of dicts loaded from your PKL
    batch_size: number of sequences per batch
    shuffle: shuffle dataset each epoch
    transform: torchvision transform to apply to each image (optional)
    crop: whether to crop images to bbox (default True)
    pad: whether to pad variable-length sequences
    preload: whether to preload all data into RAM
    Returns: PyTorch DataLoader
    """
    dataset = PIESequenceDataset(sequences, transform=transform, crop=crop, preload=preload)
    collate_fn = collate_with_padding if pad else (lambda x: x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn)
    return dataloader
