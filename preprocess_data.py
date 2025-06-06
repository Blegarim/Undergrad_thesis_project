import torch
from torchvision import transforms
from scripts.PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset

def main():
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load raw data
    train_sequences = load_sequences_from_pkl('sequences_train.pkl')
    val_sequences = load_sequences_from_pkl('sequences_val.pkl')

    # Preload and process everything
    train_dataset = PIESequenceDataset(train_sequences, transform=transform, crop=True, preload=True)
    val_dataset = PIESequenceDataset(val_sequences, transform=transform, crop=True, preload=True)

    # Save to disk
    torch.save(train_dataset.data, 'preprocessed_train.pt')

    torch.save(val_dataset.data, 'preprocessed_val.pt')
    print("Precomputed and saved datasets.")

if __name__ == "__main__":
    main()