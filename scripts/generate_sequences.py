import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from PIE.utilities.pie_data import PIE

import pickle

def generate_sequences(imdb, split='train', seq_type='all', 
                       min_track_size=10, out_path='sequences.pkl'):
    """
    imdb: PIE instance
    split: 'train', 'val', 'test', or 'all'
    seq_type: 'crossing' or 'all' (to include all behaviors)
    min_track_size: minimum sequence length
    Returns: List of dicts, each with images, bboxes, and behavior labels (per frame)
    """

    data_opts = {
        'fstride': 1,
        'data_split_type': 'default',
        'seq_type': seq_type,
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'min_track_size': min_track_size,
    }

    sequences = imdb.generate_data_trajectory_sequence(split, **data_opts)
    num_sequences = len(sequences['image'])
    dataset = []
    print("Available sequence keys:", sequences.keys())

    for i in range(num_sequences):
        images = sequences['image'][i]
        bboxes = sequences['bbox'][i]
        actions = [a[0] for a in sequences['actions'][i]] 
        looks = [l[0] for l in sequences['looks'][i]]
        crosses = [c[0] for c in sequences['cross'][i]]
        gestures = [g[0] for g in sequences['gesture'][i]]

        dataset.append({
            'images': images,
            'bboxes': bboxes,
            'actions': actions,
            'looks': looks,
            'crosses': crosses,
            'gestures': gestures
        })
    # Save the dataset to a pickle file
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} sequences to {out_path}")

    return dataset

if __name__ == '__main__':
    pie_path = ROOT_DIR / 'data'
    imdb = PIE(data_path=pie_path)

    generate_sequences(imdb, split='train', seq_type='all', out_path=  'sequences_train.pkl')
    generate_sequences(imdb, split='val', seq_type='all', out_path='sequences_val.pkl')
    generate_sequences(imdb, split='test', seq_type='all', out_path='sequences_test.pkl')