import torch
from torchvision import transforms
from scripts.PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset
import os
from tqdm import tqdm
import gc

def save_dataset_in_chunks(sequences, out_dir, chunk_size=5000, transform=None, start_idx=0, end_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    if end_idx is None:
        end_idx = len(sequences)

    for i in range(start_idx, end_idx, chunk_size):
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

    train_start_idx = 0
    train_end_idx = len(train_sequences)
    val_start_idx = 5000
    val_end_idx = len(train_sequences)

    # Save in chunks
    save_dataset_in_chunks(train_sequences, 
                           out_dir='preprocessed_train', 
                           chunk_size=5000, 
                           transform=transform, 
                           start_idx=train_start_idx,
                           end_idx=train_end_idx)
    save_dataset_in_chunks(val_sequences, 
                           out_dir='preprocessed_val', 
                           chunk_size=7500, 
                           transform=transform,
                           start_idx=val_start_idx,
                           end_idx=val_end_idx)

    print("All dataset chunks saved successfully.")

def test():
    test_sequences = load_sequences_from_pkl('sequences_test.pkl')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    test_start_idx = 0
    test_end_idx = len(test_sequences)
    save_dataset_in_chunks(test_sequences,
                            out_dir='preprocessed_test', 
                           chunk_size=5000, 
                           transform=transform,
                           start_idx=test_start_idx,
                           end_idx=test_end_idx)

if __name__ == "__main__":
    test()
