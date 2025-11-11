import os
import torch
from torchvision import transforms
import gc
from PIE_sequence_Dataset_1 import PIESequenceDataset, load_sequences_from_pkl
import io
import lmdb
import pickle
from PIL import Image
from tqdm import tqdm

def save_dataset_in_chunks_lmdb(sequences, out_dir, chunk_size=5000,
                                transform_tight=None, transform_context=None,
                                start_idx=0, end_idx=None, context_scale=2.0,
                                jpeg_quality=90):
    """
    Saves preprocessed crops and metadata into LMDB chunks with JPEG compression.
    Each chunk ≈ 1–3 GB depending on sequence count and crop size.
    """
    os.makedirs(out_dir, exist_ok=True)
    if end_idx is None:
        end_idx = len(sequences)
    total = end_idx - start_idx
    print(f'\nSaving {total} sequences into LMDB at {out_dir} (chunk_size = {chunk_size})')

    for i in range(start_idx, end_idx, chunk_size):
        chunk = sequences[i:i+chunk_size]
        dataset = PIESequenceDataset(chunk, 
                                     transform_tight=transform_tight, 
                                     transform_context=transform_context,
                                     crop=True, 
                                     preload=True, 
                                     context_scale=context_scale)

        lmdb_path = os.path.join(out_dir, f"chunk_{i:06d}.lmdb")
        print(f"Writing LMDB {lmdb_path} ...")

        env = lmdb.open(lmdb_path, map_size=100 * 1024**3)  # 100 GB per chunk max
        with env.begin(write=True) as txn:
            for j, sample in enumerate(tqdm(dataset.data, desc=f"Chunk {i}")):
                # Encode tight/context crops as JPEG
                for k, img in enumerate(sample['images_tight']):
                    buf = io.BytesIO()
                    # Convert tensor -> uint8 image
                    img_pil = Image.fromarray((img.permute(1,2,0).numpy() * 255).astype('uint8'))
                    img_pil.save(buf, format='JPEG', quality=jpeg_quality)
                    txn.put(f"{j}_{k}_tight".encode(), buf.getvalue())

                for k, img in enumerate(sample['images_context']):
                    buf = io.BytesIO()
                    img_pil = Image.fromarray((img.permute(1,2,0).numpy() * 255).astype('uint8'))
                    img_pil.save(buf, format='JPEG', quality=jpeg_quality)
                    txn.put(f"{j}_{k}_context".encode(), buf.getvalue())

                # Save metadata (motions, actions, etc.)
                meta = {key: val for key, val in sample.items() 
                        if not key.startswith("images")}
                txn.put(f"{j}_meta".encode(), pickle.dumps(meta))

        env.sync()
        env.close()
        print(f"Saved LMDB chunk {i}–{i + len(chunk) - 1}")
        del dataset
        torch.cuda.empty_cache()
        gc.collect()

    print("✅ All LMDB chunks saved successfully.")

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
    from PIE_sequence_Dataset_1 import load_sequences_from_pkl
    print("⚙️ Starting LMDB preprocessing pipeline...")

    transform_tight = img_resize(img_height, img_width)
    transform_context = img_resize(int(img_height * context_scale), int(img_width * context_scale))
    augmented_tight = img_augment(img_height, img_width)
    augmented_context = img_augment(int(img_height * context_scale), int(img_width * context_scale))

    # --- Load PKL sequences (only for preprocessing) ---
    if train:
        train_sequences = load_sequences_from_pkl('sequences_train.pkl')
    if val:
        val_sequences = load_sequences_from_pkl('sequences_val.pkl')
    if test:
        test_sequences = load_sequences_from_pkl('sequences_test.pkl')

    # --- Preprocess into LMDB ---
    if train:
        save_dataset_in_chunks_lmdb(train_sequences,
            out_dir='preprocessed_train_lmdb',
            chunk_size=1500,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)
    
    if data_aug:
        save_dataset_in_chunks_lmdb(train_sequences,
            out_dir='preprocessed_train_lmdb_aug',
            chunk_size=1500,
            transform_tight=augmented_tight,
            transform_context=augmented_context,
            context_scale=context_scale)
    
    if val:
        save_dataset_in_chunks_lmdb(val_sequences,
            out_dir='preprocessed_val_lmdb',
            chunk_size=1500,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)
    
    if test:
        save_dataset_in_chunks_lmdb(test_sequences,
            out_dir='preprocessed_test_lmdb',
            chunk_size=1500,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)

    print("✅ All LMDB datasets saved successfully.")

if __name__ == "__main__":
    main(img_height=128, img_width=128, context_scale=2,
         train=True,
         data_aug=False,
         val=True,
         test=True)