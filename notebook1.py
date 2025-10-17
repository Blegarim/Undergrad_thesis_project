from collections import Counter

def label_counts(chunk_data):
    cnts = {'actions': Counter(), 'looks': Counter(), 'crosses': Counter()}
    for s in chunk_data:
        cnts['actions'][int(s['actions'])] += 1
        cnts['looks'][int(s['looks'])] += 1
        cnts['crosses'][int(s['crosses'])] += 1
    for k, c in cnts.items():
        print(f'{k} label counts: {dict(c)}')

if __name__ == "__main__":
    import torch

    # Load a sample .pt chunk file
    chunk_path = 'preprocessed_test/chunk_000000.pt'
    chunk_data = torch.load(chunk_path, map_location='cpu')

    # Print label counts
    label_counts(chunk_data)