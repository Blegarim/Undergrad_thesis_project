import os
import torch

# ====== CONFIGURATION ======
INPUT_DIR = 'preprocessed_val'           # Where your big .pt files live
OUTPUT_DIR = 'preprocessed_val_split'    # Where new, smaller chunks will go
CHUNK_SIZE = 5000                          # Number of sequences per split chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .pt files in the input folder
input_files = sorted([
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.endswith('.pt')
])

print(f"Found {len(input_files)} .pt files to split.")

for big_idx, big_file in enumerate(input_files):
    print(f"\nLoading {big_file} ...")
    data = torch.load(big_file, map_location='cpu')
    print(f"Loaded {len(data)} samples.")

    n = len(data)
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        chunk_data = data[start:end]
        out_name = f"chunk_{big_idx+1:02d}_{i+1:03d}.pt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        print(f"  Saving {out_name} with {end-start} samples ...")
        torch.save(chunk_data, out_path)
        del chunk_data

    del data

print("\nSplitting complete! New chunks are saved in:", OUTPUT_DIR)