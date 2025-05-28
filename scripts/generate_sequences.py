import sys
from pathlib import Path
import random
from collections import defaultdict, Counter
import pickle

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from PIE.utilities.pie_data import PIE
from pprint import pprint
'''
This script generates sequences of frames and annotations for a given video
and pedestrian ID. It uses the PIE dataset to extract sequences of frames
and their corresponding bounding boxes and labels. The sequences are
generated based on a specified sequence length and are stored in a list
of dictionaries, each containing the pedestrian ID, frame numbers,
image paths, bounding boxes, and labels.
'''
def generate_sequences(database, dataset_obj, set_id, video_id, sequence_length=10, strict=False):
    
    video_db = database[set_id][video_id]
    output_sequences = []
    ped_ids = list(video_db['ped_annotations'].keys())


    for pid in ped_ids:
        track = video_db['ped_annotations'][pid]
        frames = track['frames']
        bboxes = track['bbox']

        # Extract multiple behavior labels
        behavior_keys = ['gesture', 'look', 'action', 'cross']
        all_labels = {}

        for key in behavior_keys:
            key_labels = track['behavior'].get(key, [])
            if len(key_labels) < len(frames):
                key_labels += ['__undefined__'] * (len(frames) - len(key_labels))
            all_labels[key] = key_labels

        if len(frames) < sequence_length:
            continue

        for i in range(len(frames) - sequence_length + 1):
            seq_frames = frames[i:i + sequence_length]

             # Strict continuity check (optional)
            if strict:
                if not all(seq_frames[j+1] == seq_frames[j] + 1 for j in range(sequence_length - 1)):
                    continue  # Skip sequences with non-consecutive frames

            seq_bboxes = bboxes[i:i + sequence_length]
            seq_labels = {key: all_labels[key][i:i + sequence_length] for key in behavior_keys}


            image_root = Path(ROOT_DIR) / "data" / "images"

            seq_image_paths = [
                str(image_root / set_id / video_id / f"{int(f):05d}.jpg")
                for f in seq_frames
            ]

            seq_motions = []
            for box in seq_bboxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                seq_motions.append([cx, cy])


            output_sequences.append({
                'pid': pid,
                'frame_numbers': seq_frames,
                'image_paths': seq_image_paths,
                'bboxes': seq_bboxes,
                'labels': seq_labels,
                'motions': seq_motions
            })
    
    return output_sequences

# Loop through all sets and videos to generate sequences
# This function generates sequences for all videos in the dataset
# and returns a list of all sequences.
def generate_all_sequences(database, dataset_obj, sequence_length=10, strict=False, train_split=0.8, seed=42):
    ped_sequences = defaultdict(list)
    
    for set_id in dataset_obj.set_ids:
        video_ids = [vid for vid in database[set_id].keys() if vid != 'ped_annotations']
        for video_id in video_ids:
            # Generate sequences for each video
            sequences = generate_sequences(database, dataset_obj, set_id, video_id, sequence_length, strict)
            for seq in sequences:
                # Create a unique key for each pedestrian ID
                # This allows grouping sequences by pedestrian ID
                # and split them into training and validation sets
                pid_key = f"{set_id}_{video_id}_{seq['pid']}"
                ped_sequences[pid_key].append(seq)

    # Shuffle and split pid keys
    pid_keys = list(ped_sequences.keys())

    rng = random.Random(seed)
    rng.shuffle(pid_keys)
    # Split into training and validation sets

    split_idx = int(train_split * len(pid_keys))
    train_keys = set(pid_keys[:split_idx])
    val_keys = set(pid_keys[split_idx:])

    train_sequences = []
    val_sequences = []

    for pid_key, seqs in ped_sequences.items():
        if pid_key in train_keys:
            train_sequences.extend(seqs)
        else:
            val_sequences.extend(seqs)

    print(f"Split completed: {len(train_sequences)} train sequences, {len(val_sequences)} val sequences.")
    print(f"Train PIDs: {len(train_keys)}, Val PIDs: {len(val_keys)}")

    return train_sequences, val_sequences

def sanity_check_sequences(sequences, sequence_length=10, sample_size=1000):
    print(f"Sanity checking {sample_size} random sequences...\n")
    
    sampled_sequences = random.sample(sequences, min(sample_size, len(sequences)))

    for seq_idx, seq in enumerate(sampled_sequences):  
        frames = seq['frame_numbers']
        bboxes = seq['bboxes']
        labels = seq['labels']

        for key, label_seqs in labels.items():
            assert len(label_seqs) == sequence_length, f"{key} label in seq {seq_idx} length mismatch"
        assert len(bboxes) == sequence_length, f"Seq {seq_idx} bboxes length mismatch"

        for i in range(sequence_length - 1):
            assert frames[i+1] > frames[i], f"Seq {seq_idx} frames not increasing at position {i}"

        print(f"Sequence {seq_idx} passed length and continuity checks.")

    print("All sampled sequences passed sanity checks.")

# Example usage
if __name__ == "__main__":
    data_root = ROOT_DIR / 'data'
    dataset = PIE(data_path=str(data_root))
    dataset.set_ids = ['set01']  # or more sets if you have them

    database = dataset.generate_database()

    train_sequences, val_sequences = generate_all_sequences(
        database, dataset, sequence_length=10, strict=False
    )

    # Optional: check integrity
    #sanity_check_sequences(train_sequences, sequence_length=10, sample_size=500)
    #sanity_check_sequences(val_sequences, sequence_length=10, sample_size=500)
    label_counts = {key: Counter() for key in ['gesture', 'look', 'action', 'cross']}

    for seq in train_sequences:
        for key in label_counts:
            label_counts[key].update(seq['labels'][key])

    print("\n== Label statistics in training sequences ==")
    for key, counter in label_counts.items():
        print(f"{key}: {dict(counter)}")



