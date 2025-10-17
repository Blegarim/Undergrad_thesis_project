from collections import Counter
import torch
import os

def label_counts(chunk_data):
    cnts = {'actions': Counter(), 'looks': Counter(), 'crosses': Counter()}
    for s in chunk_data:
        cnts['actions'][int(s['actions'])] += 1
        cnts['looks'][int(s['looks'])] += 1
        cnts['crosses'][int(s['crosses'])] += 1
    for k, c in cnts.items():
        print(f'{k} label counts: {dict(c)}')

if __name__ == "__main__":


    test_chunk_folder = "preprocessed_test"

    chunk_files = sorted(
        [os.path.join(test_chunk_folder, f)
         for f in os.listdir(test_chunk_folder)
         if f.endswith(".pt")]
    )

    for i, chunk_path in enumerate(chunk_files):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}] {os.path.basename(chunk_path)}")
        chunk_data = torch.load(chunk_path, map_location="cpu")
        label_counts(chunk_data)
        del chunk_data