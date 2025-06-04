import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def load_sequences_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        sequences = pickle.load(f)
    print("Keys in loaded sequences dict:", sequences.keys())
    return sequences

class PIESequenceDataset(Dataset):
    def __init__(self, sequences, transform=None, crop=True):
        self.sequences = sequences
        self.transform = transform
        self.crop = crop

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
        
        labels = seq['labels']
        return {
            'images': images,   # list of images
            'bboxes': seq['bboxes'],
            'actions': labels['action'],
            'looks': labels['look'],
            'crosses': labels['cross'],
            'gestures': labels['gesture'],
        }
    
def build_dataloader(sequences, batch_size=32, shuffle=True, transform=None, crop=True):
    """
    sequences: list of dicts loaded from your PKL
    batch_size: number of sequences per batch
    shuffle: shuffle dataset each epoch
    transform: torchvision transform to apply to each image (optional)
    crop: whether to crop images to bbox (default True)
    Returns: PyTorch DataLoader
    """
    dataset = PIESequenceDataset(sequences, transform=transform, crop=crop)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    return dataloader