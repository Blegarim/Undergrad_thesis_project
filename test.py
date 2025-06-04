from scripts.PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset, build_dataloader
from torchvision import transforms
from torchvision.transforms import ToTensor

sequences = load_sequences_from_pkl("sequences_train.pkl")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Change size if needed
    transforms.ToTensor(),
])
loader = build_dataloader(sequences, batch_size=8, shuffle=True, transform=transform, crop=True, pad=True)

for batch in loader:
    print("Batch images:", batch['images'].shape)  # Expect [B, T, C, H, W]
    print("Actions shape:", batch['actions'].shape)  # [B, T]
    print("First bboxes:", batch['bboxes'][0])  # List of boxes
    print("Meta:", batch.get('meta', None))  # Optional
    break
