import torch
from torchvision import transforms
from scripts.PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset
import os
from tqdm import tqdm
import gc

def save_dataset_in_chunks(sequences, out_dir, chunk_size=10000, transform=None):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        dataset = PIESequenceDataset(chunk, transform=transform, crop=True, preload=True)

        # Save the data list (preprocessed tensors) directly
        torch.save(dataset.data, os.path.join(out_dir, f'chunk_{i:06d}.pt'))

        print(f"Saved chunk {i}–{i + len(chunk) - 1} to {out_dir}/chunk_{i:06d}.pt")
        del dataset  # Free memory
        gc.collect() # Force garbage collection

def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_sequences = load_sequences_from_pkl('sequences_train.pkl')
    val_sequences = load_sequences_from_pkl('sequences_val.pkl')

    # Save in chunks
    save_dataset_in_chunks(train_sequences, out_dir='preprocessed_train', chunk_size=10000, transform=transform)
    save_dataset_in_chunks(val_sequences, out_dir='preprocessed_val', chunk_size=10000, transform=transform)

    print("All dataset chunks saved successfully.")

if __name__ == "__main__":
    main()
