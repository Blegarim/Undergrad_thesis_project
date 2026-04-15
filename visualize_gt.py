"""
Visualize Ground Truth for Future Prediction Task from PIE Dataset

This script visualizes:
- Input sequence: 20 consecutive frames that the model uses as input
- Future labels: Whether the pedestrian will [walk/look/cross] within the next 30 frames

This matches the format used in scripts/generate_sequences.py:
- Input: seq_len=20 frames
- Future window: future_offset=30 frames
- Labels: Binary (1 if action occurs in future window, 0 otherwise)

Usage:
    python visualize_gt.py
    
Or import and use functions directly:
    from visualize_gt import visualize_sequences_with_future_labels
"""

import os
import cv2
import numpy as np
import pickle
from PIE.utilities.pie_data import PIE

LABEL_COLORS = {
    'action': (0, 255, 255),   # Yellow - for action (will walk)
    'look': (255, 0, 255),     # Magenta - for look (will look)
    'cross': (255, 255, 0),    # Cyan - for cross (will cross)
}
TEXT_COLOR = (255, 255, 255)  # White text
BOX_COLOR = (0, 255, 0)  # Green for bounding box
SEQ_LEN = 20
FUTURE_OFFSET = 30
TOL = 2


def clamp_to_binary(signal):
    return [1 if v == 1 else 0 for v in signal]


def draw_sequence_frame(frame, bbox, frame_idx, total_frames, pie=None, track_id=None):
    """
    Draw bounding box on a single frame.
    
    Args:
        frame: Video frame (numpy array)
        bbox: Bounding box [x1, y1, x2, y2]
        frame_idx: Current frame index in sequence
        total_frames: Total frames in this sequence
        pie: PIE dataset instance (optional)
        track_id: Optional pedestrian ID
    
    Returns:
        Frame with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    
    if track_id is not None:
        id_text = f'ID:{track_id}'
        cv2.putText(frame, id_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
    
    return frame


def draw_future_labels(frame, action_future, look_future, cross_future, width):
    """
    Draw future prediction labels on the frame.
    
    Args:
        frame: Video frame
        action_future: 1 if will walk in future, 0 otherwise
        look_future: 1 if will look in future, 0 otherwise  
        cross_future: 1 if will cross in future, 0 otherwise
        width: Frame width for positioning
    
    Returns:
        Frame with labels drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    labels = [
        ('action', 'WILL WALK', action_future, LABEL_COLORS['action']),
        ('look', 'WILL LOOK', look_future, LABEL_COLORS['look']),
        ('cross', 'WILL CROSS', cross_future, LABEL_COLORS['cross']),
    ]
    
    y_offset = 40
    x_start = width - 200
    
    cv2.rectangle(frame, (x_start - 10, 10), (width - 10, y_offset + 80), (0, 0, 0), -1)
    cv2.putText(frame, "FUTURE PREDICTION", (x_start, 30), font, 0.6, (255, 255, 255), 2)
    
    for key, text, val, color in labels:
        status = text if val == 1 else "---"
        bg_color = color if val == 1 else (50, 50, 50)
        text_color = (0, 0, 0) if val == 1 else (100, 100, 100)
        
        cv2.rectangle(frame, (x_start, y_offset), (x_start + 180, y_offset + 22), bg_color, -1)
        cv2.putText(frame, status, (x_start + 5, y_offset + 16), font, font_scale, text_color, thickness)
        y_offset += 28
    
    return frame


def visualize_from_pickle(pkl_path, output_path='gt_visualization.mp4', max_sequences=50):
    """
    Visualize sequences from generated pickle file (matching generate_sequences.py output).
    
    Args:
        pkl_path: Path to pickle file with sequences
        output_path: Output video path
        max_sequences: Maximum number of sequences to visualize
    
    Returns:
        None (saves video)
    """
    print(f"Loading sequences from {pkl_path}...")
    
    if not os.path.exists(pkl_path):
        print(f"Pickle file not found: {pkl_path}")
        print("Please run scripts/generate_sequences.py first to generate the data.")
        return
    
    with open(pkl_path, 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"Loaded {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print("No sequences to visualize!")
        return
    
    first_seq = sequences[0]
    first_img_path = first_seq['images'][0]
    
    if os.path.exists(first_img_path):
        first_img = cv2.imread(first_img_path)
        if first_img is not None:
            height, width = first_img.shape[:2]
        else:
            width, height = 1920, 1080
    else:
        width, height = 1920, 1080
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
    print(f"Output video: {output_path}")
    
    num_sequences = min(len(sequences), max_sequences)
    
    for seq_idx in range(num_sequences):
        seq = sequences[seq_idx]
        images = seq['images']
        bboxes = seq['bboxes']
        action_label = seq['actions']
        look_label = seq['looks']
        cross_label = seq['crosses']
        
        seq_len = len(images)
        
        for frame_idx in range(seq_len):
            img_path = images[frame_idx]
            
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Image not found: {img_path}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            h, w = frame.shape[:2]
            
            bbox = bboxes[frame_idx]
            frame = draw_sequence_frame(frame, bbox, frame_idx, seq_len)
            
            if frame_idx == seq_len - 1:
                frame = draw_future_labels(frame, action_label, look_label, cross_label, w)
                
                info_y = h - 100
                cv2.rectangle(frame, (10, info_y - 10), (400, h - 10), (0, 0, 0), -1)
                cv2.putText(frame, f"Sequence {seq_idx + 1}/{num_sequences}", (20, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Input: {seq_len} frames | Future: {FUTURE_OFFSET} frames", (20, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame)
        
        if (seq_idx + 1) % 10 == 0:
            print(f"Processed {seq_idx + 1}/{num_sequences} sequences...")
    
    out.release()
    print(f"Video saved: {output_path}")
    print(f"Total output frames: {num_sequences * SEQ_LEN}")


def generate_and_visualize(pie_path='PIE', output_path='gt_visualization.mp4', 
                          split='test', max_sequences=50):
    """
    Generate sequences on-the-fly and visualize them.
    
    Args:
        pie_path: Path to PIE dataset
        output_path: Output video path
        split: 'train', 'val', 'test', or 'all'
        max_sequences: Maximum sequences to visualize
    
    Returns:
        None
    """
    print(f"Loading PIE dataset from {pie_path}...")
    pie = PIE(data_path=pie_path)
    
    print("Generating sequences...")
    
    data_opts = {
        'fstride': 1,
        'data_split_type': 'default',
        'seq_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'min_track_size': SEQ_LEN + FUTURE_OFFSET,
    }
    
    sequences = pie.generate_data_trajectory_sequence(split, **data_opts)
    num_sequences = len(sequences['image'])
    print(f"Found {num_sequences} tracks in {split} split")
    
    dataset = []
    
    for i in range(num_sequences):
        images = sequences['image'][i]
        bboxes = sequences['bbox'][i]
        actions = [a[0] for a in sequences['actions'][i]]
        looks = [l[0] for l in sequences['looks'][i]]
        crosses = [c[0] for c in sequences['cross'][i]]
        crosses = clamp_to_binary(crosses)
        
        n = len(images)
        if n < SEQ_LEN:
            continue
        
        for start in range(0, n - SEQ_LEN + 1, 3):
            end = start + SEQ_LEN
            
            if any(crosses[start:end]):
                continue
            
            future_start = end
            future_end = min(end + FUTURE_OFFSET + TOL, n)
            
            action_event = 1 if any(actions[future_start:future_end]) else 0
            look_event = 1 if any(looks[future_start:future_end]) else 0
            cross_event = 1 if any(crosses[future_start:future_end]) else 0
            
            dataset.append({
                'images': images[start:end],
                'bboxes': bboxes[start:end],
                'actions': action_event,
                'looks': look_event,
                'crosses': cross_event
            })
            
            if len(dataset) >= max_sequences * 2:
                break
        
        if len(dataset) >= max_sequences * 2:
            break
    
    dataset = dataset[:max_sequences]
    print(f"Generated {len(dataset)} sequences for visualization")
    
    first_seq = dataset[0]
    first_img_path = first_seq['images'][0]
    
    if os.path.exists(first_img_path):
        first_img = cv2.imread(first_img_path)
        if first_img is not None:
            height, width = first_img.shape[:2]
        else:
            width, height = 1920, 1080
    else:
        print(f"Warning: Image not found: {first_img_path}")
        width, height = 1920, 1080
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
    print(f"Output video: {output_path}")
    
    for seq_idx, seq in enumerate(dataset):
        images = seq['images']
        bboxes = seq['bboxes']
        action_label = seq['actions']
        look_label = seq['looks']
        cross_label = seq['crosses']
        
        for frame_idx in range(SEQ_LEN):
            img_path = images[frame_idx]
            
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Image not found", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            h, w = frame.shape[:2]
            
            bbox = bboxes[frame_idx]
            frame = draw_sequence_frame(frame, bbox, frame_idx, SEQ_LEN)
            
            if frame_idx == SEQ_LEN - 1:
                frame = draw_future_labels(frame, action_label, look_label, cross_label, w)
                
                info_y = h - 100
                cv2.rectangle(frame, (10, info_y - 10), (450, h - 10), (0, 0, 0), -1)
                cv2.putText(frame, f"Sequence {seq_idx + 1}/{len(dataset)}", (20, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Input: {SEQ_LEN} frames | Future: {FUTURE_OFFSET} frames", (20, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame)
    
    out.release()
    print(f"Video saved: {output_path}")
    print(f"Total sequences: {len(dataset)}")
    print(f"Total frames: {len(dataset) * SEQ_LEN}")


def visualize_blank_with_labels(output_path='gt_visualization_blank.mp4', num_sequences=20):
    """
    Visualize with blank frames to show the label structure (when images unavailable).
    
    Args:
        output_path: Output video path
        num_sequences: Number of sequences to generate
    
    Returns:
        None
    """
    width, height = 1920, 1080
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
    print(f"Output video: {output_path}")
    
    import random
    random.seed(42)
    
    for seq_idx in range(num_sequences):
        action_label = random.randint(0, 1)
        look_label = random.randint(0, 1)
        cross_label = random.randint(0, 1)
        
        for frame_idx in range(SEQ_LEN):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            cv2.rectangle(frame, (200, 200), (600, 800), BOX_COLOR, 3)
            
            cv2.putText(frame, f"Frame {frame_idx + 1}/{SEQ_LEN}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Bounding Box (Input)", (200, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)
            
            if frame_idx == SEQ_LEN - 1:
                frame = draw_future_labels(frame, action_label, look_label, cross_label, width)
                
                info_y = height - 100
                cv2.rectangle(frame, (10, info_y - 10), (500, height - 10), (0, 0, 0), -1)
                cv2.putText(frame, f"Sequence {seq_idx + 1}/{num_sequences}", (20, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Input: {SEQ_LEN} frames | Future: {FUTURE_OFFSET} frames", (20, info_y + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                label_info = f"GT Labels -> Action: {action_label}, Look: {look_label}, Cross: {cross_label}"
                cv2.putText(frame, label_info, (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
    
    out.release()
    print(f"Video saved: {output_path}")
    print(f"Total frames: {num_sequences * SEQ_LEN}")


def main():
    """Main function"""
    print("=" * 70)
    print("PIE Dataset Ground Truth Visualization - Future Prediction Task")
    print("=" * 70)
    print()
    print("This visualization shows:")
    print(f"  - Input: {SEQ_LEN} consecutive frames (what the model sees)")
    print(f"  - Future: {FUTURE_OFFSET} frame window (what we're predicting)")
    print("  - Labels: Binary (1 = will perform action, 0 = won't)")
    print()
    
    pkl_path = 'sequences_test.pkl'
    
    if os.path.exists(pkl_path):
        print(f"Found pre-generated sequences: {pkl_path}")
        visualize_from_pickle(pkl_path, max_sequences=30)
    else:
        print(f"Pickle file not found: {pkl_path}")
        print("Generating sample sequences from PIE dataset...")
        
        try:
            generate_and_visualize(
                pie_path='PIE',
                output_path='gt_visualization_output.mp4',
                split='test',
                max_sequences=30
            )
        except Exception as e:
            print(f"Error generating sequences: {e}")
            print("Creating demo visualization with blank frames instead...")
            visualize_blank_with_labels(output_path='gt_visualization_demo.mp4', num_sequences=20)
    
    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
