from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path
import sys

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
    def __init__(self, sequences, transform=None, crop=True):
        self.sequences = sequences
        self.transform = transform
        self.crop = crop

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

        # Use last-frame label by default (can be changed later)
        label = torch.tensor(sequence['labels'][-1], dtype=torch.long)

        return images, label

def build_dataloader(set_ids=['set01'], sequence_length=10, batch_size=8, crop=True):
    """
    Initializes the PIE dataset, generates sequences, and returns a DataLoader.
    """
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    data_root = ROOT_DIR / 'data'
    pie = PIE(data_path=str(data_root))
    pie.set_ids = set_ids  # Specify which dataset splits to use

    database = pie.generate_database()
    sequences = generate_all_sequences(database, pie, sequence_length=sequence_length)

    dataset = PIESequenceDataset(sequences=sequences, transform=transform, crop=crop)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

if __name__ == '__main__':
    # Quick sanity check
    dataloader = build_dataloader()

    for batch in dataloader:
        imgs, labels = batch
        print("Image batch shape:", imgs.shape)    # [B, T, C, H, W]
        print("Label batch shape:", labels.shape)  # [B]
        break
    
