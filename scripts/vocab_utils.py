from collections import defaultdict

def build_label_vocab(sequences, behavior_keys=['gesture', 'look', 'action', 'cross']):
    """
    Build a vocabulary mapping for each behavior key from the sequence labels.

    Args:
        sequences (list of dict): List of sequence dicts from generate_sequences.
        behavior_keys (list): Keys to extract labels for.

    Returns:
        dict: A dictionary where each key is a behavior type and value is a label-to-index mapping.
    """
    vocab = {key: set() for key in behavior_keys}

    #Gather all labels string for each behavior key
    for seq in sequences:
        for key in behavior_keys:
            vocab[key].update(seq['labels'][key])

    # Convert sets to sorted lists and create label-to-index mappings
    vocab_maps = {}
    for key, label_set in vocab.items():
        labels_sorted = sorted(label_set)
        vocab_maps[key] = {label: idx for idx, label in enumerate(labels_sorted)}

    return vocab_maps 

def encode_labels(label_seq, vocab_map):
    """
    Encode a list of raw labels into indices using the given vocab_map.

    Args:
        label_seq (list): List of raw labels.
        vocab_map (dict): Mapping from raw label to index.

    Returns:
        list: List of encoded labels (integers).
    """
    return [vocab_map[label] for label in label_seq]


def decode_labels(encoded_seq, reverse_vocab_map):
    """
    Decode a list of integer labels back to raw labels.

    Args:
        encoded_seq (list): List of encoded labels.
        reverse_vocab_map (dict): Mapping from index to raw label.

    Returns:
        list: List of raw labels.
    """
    return [reverse_vocab_map[idx] for idx in encoded_seq]