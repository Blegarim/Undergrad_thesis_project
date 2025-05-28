from collections import defaultdict

def build_label_vocab(sequences, behavior_keys=['cross']):
    """
    Builds vocabularies (unique label -> index) for each behavior key.

    Args:
        sequences: list of dicts, each with a 'labels' field that contains multiple label types.
        behavior_keys: list of label keys to process.

    Returns:
        A dictionary mapping each behavior key to its label vocab: {label_str: index}
    """
    from collections import defaultdict

    vocab = {key: set() for key in behavior_keys}

    for seq in sequences:
        for key in behavior_keys:
            labels_per_frame = seq['labels'][key]
            # Flatten list of lists: frame-level multi-label → flat label list
            for frame_labels in labels_per_frame:
                if isinstance(frame_labels, list):
                    vocab[key].update(frame_labels)
                else:
                    vocab[key].add(frame_labels)

    vocab = {
        key: {label: idx for idx, label in enumerate(sorted(vocab[key]))}
        for key in vocab
    }

    return vocab

def encode_labels(label_seq, vocab_map):
    """
    label_seq: List of labels per frame, where each frame can have multiple labels.
    Returns a list of lists of label indices.
    """
    encoded = []

    for label in label_seq:
        if isinstance(label, list):
            encoded.append([vocab_map[l] for l in label])
        else:
            encoded.append([vocab_map[label]])

    return encoded



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