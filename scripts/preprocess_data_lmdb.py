import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import encode_jpeg
import gc
from PIE_sequence_Dataset_1 import PIESequenceDataset, load_sequences_from_pkl
import lmdb
import pickle
from tqdm import tqdm

def save_dataset_in_chunks_lmdb(sequences, out_dir, chunk_size=5000,
                                transform_tight=None, transform_context=None,
                                start_idx=0, end_idx=None, context_scale=2.0,
                                jpeg_quality=90, num_workers=8,
                                prefetch_factor=2):
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
        dataset = PIESequenceDataset(
            chunk,
            transform_tight=transform_tight,
            transform_context=transform_context,
            crop=True,
            preload=False,
            context_scale=context_scale,
        )
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(dataset, **loader_kwargs)

        lmdb_path = os.path.join(out_dir, f"chunk_{i:06d}.lmdb")
        print(f"Writing LMDB {lmdb_path} ...")

        est_bytes = len(chunk) * 2 * (512 * 512 * 3) * 0.25 * 5
        map_size = max(int(est_bytes * 1.5), 4 * 1024**3)  
        print(f"→ Allocating map_size ≈ {map_size / 1024**3:.2f} GB")

        env = lmdb.open(lmdb_path, map_size=map_size)  # use calculated map_size
        with env.begin(write=True) as txn:
            for j, sample in enumerate(tqdm(loader, desc=f"Chunk {i}", total=len(chunk))):
                def unbatch(value):
                    if torch.is_tensor(value):
                        return value[0]
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        return value[0]
                    return value

                sample = {k: unbatch(v) for k, v in sample.items()}

                # Encode tight/context crops as JPEG (torchvision, no PIL roundtrip)
                for k, img in enumerate(sample['images_tight']):
                    img_uint8 = (img * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
                    jpg = encode_jpeg(img_uint8, quality=jpeg_quality)
                    txn.put(f"{j}_{k}_tight".encode(), jpg.numpy().tobytes())

                for k, img in enumerate(sample['images_context']):
                    img_uint8 = (img * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
                    jpg = encode_jpeg(img_uint8, quality=jpeg_quality)
                    txn.put(f"{j}_{k}_context".encode(), jpg.numpy().tobytes())

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

    print("All LMDB chunks saved successfully.")

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
         train=True, data_aug=False, val=True, test=True, 
         chunk_size=4500):
    from PIE_sequence_Dataset_1 import load_sequences_from_pkl
    print("Starting LMDB preprocessing pipeline...")

    transform_tight = img_resize(img_height, img_width)
    transform_context = img_resize(int(img_height * context_scale), int(img_width * context_scale))
    augmented_tight = img_augment(img_height, img_width)
    augmented_context = img_augment(int(img_height * context_scale), int(img_width * context_scale))

    # --- Load PKL sequences (only for preprocessing) ---
    # Use augmented sequences for training, balanced for val/test
    if train:
        train_sequences = load_sequences_from_pkl('sequences_train_augmented.pkl')
    if val:
        val_sequences = load_sequences_from_pkl('sequences_val.pkl')
    if test:
        test_sequences = load_sequences_from_pkl('sequences_test.pkl')

    # --- Preprocess into LMDB ---
    if train:
        save_dataset_in_chunks_lmdb(train_sequences,
            out_dir='preprocessed_train_augmented',
            chunk_size=chunk_size,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)
    
    if data_aug:
        save_dataset_in_chunks_lmdb(train_sequences,
            out_dir='preprocessed_train_augmented_dataaug',
            chunk_size=chunk_size,
            transform_tight=augmented_tight,
            transform_context=augmented_context,
            context_scale=context_scale)
    
    if val:
        save_dataset_in_chunks_lmdb(val_sequences,
            out_dir='preprocessed_val',
            chunk_size=chunk_size,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)
    
    if test:
        save_dataset_in_chunks_lmdb(test_sequences,
            out_dir='preprocessed_test',
            chunk_size=chunk_size,
            transform_tight=transform_tight,
            transform_context=transform_context,
            context_scale=context_scale)

    print("✅ All LMDB datasets saved successfully.")

if __name__ == "__main__":
    main(img_height=128, img_width=128, context_scale=3.0,
         train=True,
         data_aug=True,
         val=True,
         test=True,
         chunk_size=5060)
