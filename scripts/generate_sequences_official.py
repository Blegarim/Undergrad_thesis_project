import sys
import pickle
from pathlib import Path
import argparse

# Project root setup
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from PIE.utilities.pie_data import PIE

def safe_patch_image_set_ids(dataset):
    """
    Patch the dataset instance so _get_image_set_ids can handle list-of-tuples.
    """
    original_func = dataset._get_image_set_ids

    def patched(image_set):
        if isinstance(image_set, str):
            return original_func(image_set)
        elif isinstance(image_set, list):
            return list(set(s for (s, _) in image_set))
        else:
            raise TypeError(f"Unsupported type for image_set: {type(image_set)}")

    dataset._get_image_set_ids = patched

def flatten_ped_id(ped_id):
    if isinstance(ped_id, list):
        result = []
        for item in ped_id:
            result.extend(flatten_ped_id(item))
        return result
    else:
        return [ped_id]

def postprocess_add_behavior_labels(sequences, dataset, annot_db):
    """
    Adds per-frame behavior labels (action, look, gesture, cross) to the trajectory sequences.
    """
    print("🧩 Post-processing to add 'action', 'look', 'gesture', 'cross' labels...")

    sequences['action'] = []
    sequences['look'] = []
    sequences['gesture'] = []
    sequences['cross'] = []

    for seq_idx, (images, ped_id) in enumerate(zip(sequences['image'], sequences['ped_id'])):
        # Fix nested image list if needed
        if isinstance(images, list) and len(images) == 1 and isinstance(images[0], list):
            images = images[0]
        if not images:
            print(f"⚠️ Skipping empty image list at sequence {seq_idx}")
            continue

        # Extract set_id and video_id
        try:
            first_image_path = Path(images[0])
            parts = first_image_path.parts
            set_id, video_id = parts[-4], parts[-3]
        except Exception as e:
            print(f"⚠️ Failed to parse image path for sequence {seq_idx}: {e}")
            continue

        # Flatten ped_id to list of strings
        ped_ids = flatten_ped_id(ped_id)

        # Remove duplicates while preserving order
        seen = set()
        unique_ped_ids = []
        for pid in ped_ids:
            if pid not in seen:
                unique_ped_ids.append(pid)
                seen.add(pid)

        if len(unique_ped_ids) == 0:
            print(f"⚠️ Empty ped_id list at sequence {seq_idx}, skipping")
            continue
        elif len(unique_ped_ids) > 1:
            print(f"⚠️ Multiple ped_ids {ped_ids} at sequence {seq_idx}, skipping")
            continue

        ped_id_key = ped_ids[0]

        try:
            frames = annot_db[set_id][video_id]['ped_annotations'][ped_id_key]['frames']
        except KeyError:
            print(f"⚠️ Missing annotation for {set_id}/{video_id}/{ped_id_key}")
            frames = {}

        behavior_labels = {'action': [], 'look': [], 'gesture': [], 'cross': []}

        for img_path in images:
            frame_num = Path(img_path).stem  # e.g., '000345'
            frame_data = frames.get(frame_num, {})

            for key in behavior_labels:
                ann_key = key if key != 'cross' else 'crossing'
                label_str = frame_data.get(ann_key, None)

                try:
                    if label_str is None:
                        scalar = -1  # sentinel for missing label
                    else:
                        scalar = dataset._map_text_to_scalar(ann_key, label_str)
                except KeyError:
                    print(f"⚠️ Unmapped label '{label_str}' for {ann_key} in frame {frame_num}")
                    scalar = -1

                behavior_labels[key].append(scalar)

        # Append labels to sequences output
        for key in behavior_labels:
            sequences[key].append(behavior_labels[key])


def generate_sequences(set_ids, seq_type, output_dir):
    from PIE.utilities.pie_data import PIE

    DATA_ROOT = ROOT_DIR / 'data'
    OUTPUT_FILENAME = f'sequences_{"_".join(set_ids)}_{seq_type}.pkl'

    data_opts = {
        'fstride': 1,
        'data_split_type': 'default',
        'seq_type': seq_type,
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'min_track_size': 0,
        'random_params': {'ratios': None, 'val_data': True, 'regen_data': True},
        'kfold_params': {'num_folds': 5, 'fold': 1}
    }

    dataset = PIE(data_path=str(DATA_ROOT))
    dataset.set_ids = set_ids
    dataset.data_opts = data_opts

    safe_patch_image_set_ids(dataset)

    print("Generating annotation database...")
    annot_db = dataset.generate_database()

    print(f"Generating '{seq_type}' sequences...")

    if seq_type == 'trajectory':
        image_set = []
        for set_id in set_ids:
            for video_id in annot_db[set_id].keys():
                if video_id != 'ped_annotations':
                    image_set.append((set_id, video_id))
        sequences = dataset.generate_data_trajectory_sequence(image_set)

        # Add per-frame behavior labels (post-process)
        postprocess_add_behavior_labels(sequences, dataset, annot_db)

    elif seq_type in ['intention', 'crossing']:
        sequences = dataset.generate_data_intention_sequence(set_ids)
        print("Behavior labels not added for non-trajectory types.")

    else:
        raise ValueError(f"Unsupported sequence type: {seq_type}")

    output_path = Path(output_dir or (DATA_ROOT / 'PIE' / set_ids[0])) / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f)

    print(f"✅ Saved {len(sequences['image'])} sequences to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['set01'], help='List of dataset set IDs')
    parser.add_argument('--seq_type', default='trajectory', help='Sequence type: trajectory / intention / crossing')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory (optional)')
    args = parser.parse_args()

    generate_sequences(args.sets, args.seq_type, args.output_dir)

if __name__ == '__main__':
    main()
