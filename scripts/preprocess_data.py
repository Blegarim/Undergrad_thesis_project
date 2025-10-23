import torch
from torchvision import transforms
from PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset
import os
from tqdm import tqdm
import gc

def save_dataset_in_chunks(sequences, out_dir, chunk_size=5000, transform=None, start_idx=0, end_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    if end_idx is None:
        end_idx = len(sequences)
    total = end_idx - start_idx
    print(f'\nSaving {total} sequences into {out_dir} (chunk_size = {chunk_size})')

    for i in range(start_idx, end_idx, chunk_size):
        chunk = sequences[i:i+chunk_size]
        dataset = PIESequenceDataset(chunk, transform=transform, crop=True, preload=True)

        # Save the data list (preprocessed tensors) directly
        #torch.save(dataset.data, os.path.join(out_dir, f'chunk_{i:06d}.pt'))
        tmp_path = os.path.join(out_dir, f"tmp_chunk_{i:06d}.pt")
        final_path = os.path.join(out_dir, f"chunk_{i:06d}.pt")

        torch.save(dataset.data, tmp_path)
        os.replace(tmp_path, final_path)

        print(f"Saved chunk {i}–{i + len(chunk) - 1} to {out_dir}/chunk_{i:06d}.pt")
        del dataset  # Free memory
        torch.cuda.empty_cache()
        gc.collect() # Force garbage collection

def main():

    base_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    base_160 = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    augmented_160 = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    train_sequences = load_sequences_from_pkl('sequences_train.pkl')
    val_sequences = load_sequences_from_pkl('sequences_val.pkl')
    test_sequences = load_sequences_from_pkl('sequences_test.pkl')

    train_start_idx = 0
    train_end_idx = len(train_sequences)
    val_start_idx = 5000
    val_end_idx = len(val_sequences)
    test_start_idx = 0
    test_end_idx = len(test_sequences)

    # Save in chunks
    save_dataset_in_chunks(train_sequences, 
                           out_dir='preprocessed_train_base', 
                           chunk_size=7500, 
                           transform=base_160, 
                           start_idx=train_start_idx,
                           end_idx=train_end_idx)
    
    save_dataset_in_chunks(train_sequences, 
                           out_dir='preprocessed_train_augmented', 
                           chunk_size=7500, 
                           transform=augmented_160, 
                           start_idx=train_start_idx,
                           end_idx=train_end_idx)
    
    save_dataset_in_chunks(val_sequences, 
                           out_dir='preprocessed_val_base', 
                           chunk_size=7500, 
                           transform=base_160,
                           start_idx=val_start_idx,
                           end_idx=val_end_idx)
    
    save_dataset_in_chunks(val_sequences, 
                           out_dir='preprocessed_val_augmented', 
                           chunk_size=7500, 
                           transform=augmented_160,
                           start_idx=val_start_idx,
                           end_idx=val_end_idx)
    
    save_dataset_in_chunks(test_sequences,
                           out_dir='preprocessed_test_128',
                           chunk_size=7500,
                           transform=base_128,
                           start_idx=test_start_idx,
                           end_idx=test_end_idx)
    
    save_dataset_in_chunks(test_sequences,
                           out_dir='preprocessed_test_160',
                           chunk_size=7500,
                           transform=base_160,
                           start_idx=test_start_idx,
                           end_idx=test_end_idx)

    print("All dataset chunks saved successfully.")

if __name__ == "__main__":
    data = torch.load('preprocessed_train_augmented/chunk_000000.pt')
    print(type(data[0]['images']))
