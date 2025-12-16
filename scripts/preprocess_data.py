import torch
from torchvision import transforms
from PIE_sequence_Dataset_1 import load_sequences_from_pkl, PIESequenceDataset
import os
import gc

def save_dataset_in_chunks(sequences, out_dir, chunk_size=5000, 
                           transform_tight=None, transform_context=None,
                           start_idx=0, end_idx=None, context_scale=2.0):
    os.makedirs(out_dir, exist_ok=True)
    if end_idx is None:
        end_idx = len(sequences)
    total = end_idx - start_idx
    print(f'\nSaving {total} sequences into {out_dir} (chunk_size = {chunk_size})')

    for i in range(start_idx, end_idx, chunk_size):
        chunk = sequences[i:i+chunk_size]
        dataset = PIESequenceDataset(chunk, transform_tight=transform_tight, transform_context=transform_context, crop=True, preload=True, context_scale=context_scale)

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

def img_resize(height=160, width=160):
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
def img_augment(height=160, width=160):
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

def main(img_height=128, img_width=128, context_scale=2.0,
         train=True, data_aug=False, val=True, test=True):

    transform_tight = img_resize(img_height, img_width)
    transform_context = img_resize(img_height * context_scale, img_width * context_scale)
    augmented_tight = img_augment(img_height, img_width)
    augmented_context = img_augment(img_height * context_scale, img_width * context_scale)

    train_sequences = load_sequences_from_pkl('sequences_train.pkl')
    val_sequences = load_sequences_from_pkl('sequences_val.pkl')
    test_sequences = load_sequences_from_pkl('sequences_test.pkl')

    train_start_idx = 0
    train_end_idx = len(train_sequences)
    val_start_idx = 0
    val_end_idx = 13499
    test_start_idx = 0
    test_end_idx = len(test_sequences)

    # Save in chunks
    if train:
        save_dataset_in_chunks(train_sequences, 
                            out_dir='preprocessed_train_base', 
                            chunk_size=1500, 
                            transform_tight=transform_tight, 
                            transform_context=transform_context,
                            context_scale=context_scale,
                            start_idx=train_start_idx,
                            end_idx=train_end_idx)
    if data_aug:
        save_dataset_in_chunks(train_sequences, 
                           out_dir='preprocessed_train_augmented', 
                           chunk_size=1500, 
                           transform_tight=augmented_tight, 
                           transform_context=augmented_context,
                           context_scale=context_scale,
                           start_idx=train_start_idx,
                           end_idx=train_end_idx)
    if val:
        save_dataset_in_chunks(val_sequences, 
                            out_dir='preprocessed_val', 
                            chunk_size=1500, 
                            transform_tight=transform_tight, 
                            transform_context=transform_context,
                            context_scale=context_scale,
                            start_idx=val_start_idx,
                            end_idx=val_end_idx)
    if test:
        save_dataset_in_chunks(train_sequences, 
                            out_dir='preprocessed_test', 
                            chunk_size=1500, 
                            transform_tight=transform_tight, 
                            transform_context=transform_context,
                            context_scale=context_scale,
                            start_idx=test_start_idx,
                            end_idx=test_end_idx)

    print("All dataset chunks saved successfully.")

if __name__ == "__main__":
    main(img_height=128, img_width=128, context_scale=2, 
        train=False, 
        data_aug=True, 
        val=False, 
        test=False)
