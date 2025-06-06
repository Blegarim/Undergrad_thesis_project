import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from PIE.utilities.pie_data import PIE

import pickle

SSD_ROOT = '/mnt/ssd/pie_images'  # Adjust this path as needed

def redirect_image_paths(img_path):
    """
    Redirects image paths to start from SSD_ROOT, keeping the subdirectory structure after 'data'.
    """
    img_path = str(img_path)
    data_str = str(ROOT_DIR / 'data')
    if img_path.startswith(data_str):
        rel_path = img_path[len(data_str):].lstrip('/')
        return str(Path(SSD_ROOT) / rel_path)
    else:
        return img_path

def generate_sequences(
        imdb, 
        split='train', 
        seq_type='all', 
        min_track_size=10, 
        out_path='sequences.pkl',
        seq_len=20,           # <<< new: desired subsequence length
        stride=1              # <<< new: sliding window stride
    ):
    """
    imdb: PIE instance
    split: 'train', 'val', 'test', or 'all'
    seq_type: 'crossing' or 'all' (to include all behaviors)
    min_track_size: minimum sequence length
    seq_len: desired length of output sub-sequences
    stride: sliding window step size
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
        images = [redirect_image_paths(img) for img in sequences['image'][i]]
        bboxes = sequences['bbox'][i]
        actions = [a[0] for a in sequences['actions'][i]] 
        looks = [l[0] for l in sequences['looks'][i]]
        crosses = [c[0] for c in sequences['cross'][i]]
        gestures = [g[0] for g in sequences['gesture'][i]]

        n = len(images)
        if n < seq_len:
            continue  # skip too-short tracks

        # Sliding window: create many fixed-length sub-sequences
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            dataset.append({
                'images': images[start:end],
                'bboxes': bboxes[start:end],
                'actions': actions[start:end],
                'looks': looks[start:end],
                'crosses': crosses[start:end],
                'gestures': gestures[start:end]
            })

    # Save the dataset to a pickle file
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} sequences to {out_path}")

    return dataset

if __name__ == '__main__':
    pie_path = ROOT_DIR / 'data'
    imdb = PIE(data_path=pie_path)

    generate_sequences(imdb, split='train', seq_type='all', out_path='sequences_train_ssd.pkl', seq_len=20, stride=1)
    generate_sequences(imdb, split='val', seq_type='all', out_path='sequences_val_ssd.pkl', seq_len=20, stride=1)
    generate_sequences(imdb, split='test', seq_type='all', out_path='sequences_test_ssd.pkl', seq_len=20, stride=1)