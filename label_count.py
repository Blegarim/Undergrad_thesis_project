from collections import Counter
import torch
import os
import csv

def label_counts(chunk_data):
    cnts = {'actions': Counter(), 'looks': Counter(), 'crosses': Counter()}
    for s in chunk_data:
        cnts['actions'][int(s['actions'])] += 1
        cnts['looks'][int(s['looks'])] += 1
        cnts['crosses'][int(s['crosses'])] += 1
    for k in cnts:
        for label in [0, 1, -1] if k == 'crosses' else [0, 1]:
            cnts[k].setdefault(label, 0)
    return cnts

if __name__ == "__main__":
    test_chunk_folder = "preprocessed_test"
    os.makedirs('training_log', exist_ok=True)
    label_count_csv = os.path.join('training_log', 'label_count.csv')

    csv_header = [
        'chunk',
        'actions[walking]',
        'actions[standing]',
        'looks[looking]',
        'looks[not-looking]',
        'crosses[crossing]',
        'crosses[not-crossing]',
        'crosses[irrelevant]'
    ]

    chunk_files = sorted(
        [os.path.join(test_chunk_folder, f)
         for f in os.listdir(test_chunk_folder)
         if f.endswith(".pt")]
    )
    print("Found chunks:", len(chunk_files))

    with open(label_count_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for i, chunk_path in enumerate(chunk_files):
            chunk_name = os.path.basename(chunk_path)
            print(f"\n[Chunk {i+1}/{len(chunk_files)}] {chunk_name}")
            chunk_data = torch.load(chunk_path, map_location="cpu")

            if not chunk_data:
                print(f"⚠️ Empty chunk: {chunk_name}")
                continue

            cnts = label_counts(chunk_data)
            row = [
                chunk_name,
                cnts['actions'][1], cnts['actions'][0],
                cnts['looks'][1], cnts['looks'][0],
                cnts['crosses'][1], cnts['crosses'][0], cnts['crosses'][-1]
            ]
            writer.writerow(row)
            print(f"Logged counts: {row[1:]}")
            del chunk_data

    print(f"\n✅ Label counts saved to: {label_count_csv}")
