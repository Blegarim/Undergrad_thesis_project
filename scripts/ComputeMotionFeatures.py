import torch

def compute_motion_features(bboxes, frame_numbers, include_frame_deltas=False):
    """
    bboxes: list of [x1, y1, x2, y2] per frame
    frame_numbers: list of frame indices (ints)
    include_frame_deltas: whether to append frame-to-frame delta as a feature

    Returns: Tensor shape [sequence_length, feature_dim]
    """
    centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = torch.tensor(centers, dtype=torch.float32)

    # Compute motion deltas (dx, dy)
    deltas = torch.zeros_like(centers)
    deltas[1:] = centers[1:] - centers[:-1]

    if include_frame_deltas:
        dt = []
        for i in range(len(frame_numbers)):
            if i == 0:
                dt.append(0)
            else:
                dt.append(frame_numbers[i] - frame_numbers[i-1])
        dt = torch.tensor(dt, dtype=torch.float32).unsqueeze(1)  # shape [seq_len, 1]

        motion = torch.cat([deltas, dt], dim=1)  # [seq_len, 3]
    else:
        motion = deltas  # [seq_len, 2]

    return motion
