"""
Augment minority class sequences (crosses=1, looks=1) to expand the dataset.
Applies safe, realistic transformations that preserve label validity.
"""

import pickle
import random
import torch
import torchvision.transforms as T
import copy


class SequenceAugmenter:
    def __init__(self, p_flip=0.5, p_color=0.4, p_noise=0.3, p_erase=0.2):
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_noise = p_noise
        self.p_erase = p_erase
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1
        )

    def horizontal_flip(self, seq):
        seq = copy.deepcopy(seq)
        seq['images_tight'] = torch.flip(seq['images_tight'], dims=[3])
        seq['images_context'] = torch.flip(seq['images_context'], dims=[3])
        seq['motions'][:, 2] *= -1
        return seq

    def color_augment(self, seq):
        seq = copy.deepcopy(seq)
        for t in range(len(seq['images_tight'])):
            seq['images_tight'][t] = self.color_jitter(seq['images_tight'][t])
            seq['images_context'][t] = self.color_jitter(seq['images_context'][t])
        return seq

    def motion_noise(self, seq, noise_std=0.02):
        seq = copy.deepcopy(seq)
        noise = torch.randn_like(seq['motions']) * noise_std
        seq['motions'] = seq['motions'] + noise
        return seq

    def random_erase_frames(self, seq, n_frames=2):
        seq = copy.deepcopy(seq)
        T = len(seq['images_tight'])
        if T < n_frames:
            return seq
        erase_frames = random.sample(range(T), n_frames)
        for f in erase_frames:
            prev_f = max(0, f - 1)
            next_f = min(T - 1, f + 1)
            seq['images_tight'][f] = (seq['images_tight'][prev_f] + seq['images_tight'][next_f]) / 2
            seq['images_context'][f] = (seq['images_context'][prev_f] + seq['images_context'][next_f]) / 2
        return seq

    def __call__(self, seq):
        augmented = [copy.deepcopy(seq)]
        n_augs = random.randint(2, 4)
        choices = ['flip', 'color', 'noise', 'erase']
        selected = random.sample(choices, n_augs)

        for aug_type in selected:
            aug_seq = copy.deepcopy(seq)
            if aug_type == 'flip' and random.random() < self.p_flip:
                augmented.append(self.horizontal_flip(aug_seq))
            elif aug_type == 'color' and random.random() < self.p_color:
                augmented.append(self.color_augment(aug_seq))
            elif aug_type == 'noise' and random.random() < self.p_noise:
                augmented.append(self.motion_noise(aug_seq))
            elif aug_type == 'erase' and random.random() < self.p_erase:
                augmented.append(self.random_erase_frames(aug_seq))

        return augmented


def augment_minority_sequences(input_path, output_path, 
                                crosses_multiplier=6, looks_multiplier=3,
                                seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Loading sequences from {input_path}...")
    with open(input_path, 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"Total sequences: {len(sequences)}")
    
    crosses_pos = [s for s in sequences if s['crosses'] == 1]
    looks_pos = [s for s in sequences if s['looks'] == 1]
    original_neg = [s for s in sequences if s['crosses'] == 0]
    
    print(f"\nOriginal: crosses={len(crosses_pos)}, looks={len(looks_pos)}, negative={len(original_neg)}")
    
    augmenter = SequenceAugmenter()
    final_sequences = list(original_neg)
    
    def expand_subset(subset, multiplier, label):
        if not subset:
            return []
        target = len(subset) * multiplier
        result = []
        idx = 0
        while len(result) < target:
            result.extend(augmenter(subset[idx % len(subset)]))
            idx += 1
        result = result[:target]
        print(f"  {label}: {len(subset)} -> {len(result)} (x{multiplier})")
        return result
    
    print(f"\nAugmenting...")
    final_sequences.extend(expand_subset(crosses_pos, crosses_multiplier, "crosses"))
    final_sequences.extend(expand_subset(looks_pos, looks_multiplier, "looks"))
    
    random.shuffle(final_sequences)
    
    total = len(final_sequences)
    crosses_pos_final = sum(1 for s in final_sequences if s['crosses'] == 1)
    looks_pos_final = sum(1 for s in final_sequences if s['looks'] == 1)
    
    print(f"\nResult: {total} sequences")
    print(f"  crosses: {crosses_pos_final} ({crosses_pos_final/total*100:.1f}%)")
    print(f"  looks: {looks_pos_final} ({looks_pos_final/total*100:.1f}%)")
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_sequences, f)
    print(f"\nSaved to {output_path}")
    
    return final_sequences


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sequences_train.pkl')
    parser.add_argument('--output', default='sequences_train_augmented.pkl')
    parser.add_argument('--crosses_mult', type=int, default=6)
    parser.add_argument('--looks_mult', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    augment_minority_sequences(
        args.input, args.output,
        crosses_multiplier=args.crosses_mult,
        looks_multiplier=args.looks_mult,
        seed=args.seed
    )
