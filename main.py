import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config
from scripts.pedestrian_detection import extract_tracks_from_video, smooth_track
from scripts.PIE_sequence_Dataset_1 import PIESequenceDataset
from PIE.utilities.pie_data import PIE

# ============================================================
# Configuration
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Video inference configuration
video_path = 'test_clip2.mp4'
out_video = 'output_with_predictions.mp4'
model_path = "best_model_outputs/best_model_epoch10_0121_2029.pth"

# Model and data configuration
embedding_dim = 128
sequence_length = 20
vit_args = vit_args_config()
motion_enc_args = motion_enc_args_config()
num_classes_dict = {
    'actions': 2,
    'looks': 2,
    'crosses': 2
}

# Image transforms (aligned with train.py/test.py)
base_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# PIE dataset utils
pie = PIE(data_path='PIE')

# ============================================================
# Model Initialization
# ============================================================
model = EnsembleModel(
    motion_enc=MotionEncoder(**motion_enc_args),
    vit=ViT_Hierarchical(**vit_args),
    cross_attention=CrossAttentionModule(
        d_model=embedding_dim,
        num_heads=4,
        num_classes_dict=num_classes_dict,
        use_frame_crosses=True,
        frame_pool="logsumexp",
    )
).to(device)

if os.path.exists(model_path):
    print(f'Loading model from {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
else:
    print(f'WARNING: Model checkpoint not found at {model_path}')

model.eval()
print(f'Model loaded and set to evaluation mode.\n')

# ============================================================
# Extract Tracks and Sequences from Video
# ============================================================
print(f"Extracting pedestrian tracks from {video_path}...")
tracks = extract_tracks_from_video(
    video_path=video_path,
    model_path='yolo11n.pt',
    class_idx=0,
    conf=0.3,
    show=False
)
print(f"Total pedestrian tracks: {len(tracks)}\n")

# Convert tracks to PIESequenceDataset format
# The format expected: list of dicts with keys: 'images', 'bboxes', 'actions', 'looks', 'crosses'
sequences_for_dataset = []

for track_id, track_data in tracks.items():
    smoothed_track = smooth_track(track_data)
    
    # Skip short tracks
    if len(smoothed_track) < sequence_length:
        continue
    
    # Extract sequences from each track using sliding window
    for i in range(len(smoothed_track) - sequence_length + 1):
        window = smoothed_track[i:i+sequence_length]
        
        # Prepare sequence data for PIESequenceDataset
        sequence_dict = {
            'images': [item['image'] for item in window],  # List of PIL Images or image arrays
            'bboxes': [item['bbox'] for item in window],    # List of (x1, y1, x2, y2) tuples
            'actions': 0,    # Placeholder - will be predicted
            'looks': 0,      # Placeholder - will be predicted
            'crosses': 0,    # Placeholder - will be predicted
            'ped_id': track_id,
            'video_id': 'inference'
        }
        sequences_for_dataset.append(sequence_dict)

print(f"Extracted {len(sequences_for_dataset)} sequences from all tracks.")

# ============================================================
# Prepare Dataset and DataLoader
# ============================================================
if len(sequences_for_dataset) == 0:
    print("No sequences extracted from video. Exiting.")
    exit(1)

dataset = PIESequenceDataset(
    sequences_for_dataset,
    transform_tight=base_transforms,
    transform_context=base_transforms,
    crop=True,
    context_scale=2.0,
    return_metadata=True,
    preload=True
)

# Collate function that matches train.py/test.py
def inference_collate_fn(batch):
    """Collate function for inference - handles variable-length sequences"""
    images_tight = torch.stack([item['images_tight'] for item in batch], dim=0)
    images_context = torch.stack([item['images_context'] for item in batch], dim=0)
    motions = torch.stack([item['motions'] for item in batch], dim=0)[..., :8]
    bboxes = [item['bboxes'] for item in batch]
    meta = [item['meta'] for item in batch]
    
    return images_tight, images_context, motions, bboxes, meta

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=inference_collate_fn,
    pin_memory=(device.type == 'cuda')
)

# ============================================================
# Inference
# ============================================================
print("Running inference...")
batch_size = 32
all_preds = {'actions': [], 'looks': [], 'crosses_frame': []}
all_metadata = []

with torch.no_grad():
    for images_tight, images_context, motions, bboxes, metadata in dataloader:
        images_tight = images_tight.to(device)
        images_context = images_context.to(device)
        motions = motions.to(device)
        
        # Forward pass
        outputs = model(images_tight, images_context, motions)
        
        # Extract predictions
        # NOTE: Aligned with train.py which uses specific logic for crosses outputs
        for key in outputs:
            if key == 'crosses':
                # Skip generic 'crosses' key if present (shouldn't be with new model)
                continue
            elif key == 'crosses_frame':
                # This is used for main prediction (aligned with train.py)
                batch_preds = outputs[key].argmax(dim=1).cpu().tolist()
                all_preds['crosses_frame'].extend(batch_preds)
            else:
                # actions, looks
                batch_preds = outputs[key].argmax(dim=1).cpu().tolist()
                all_preds[key].extend(batch_preds)
        
        # Store metadata
        all_metadata.extend(metadata)

print(f"Inference complete. Generated predictions for {len(all_preds['actions'])} sequences.\n")

# ============================================================
# Aggregate Results by Frame
# ============================================================
frame_results = {}  # frame_idx -> list of detections

for seq_idx, meta in enumerate(all_metadata):
    ped_id = meta.get('ped_id', -1)
    bboxes_seq = sequences_for_dataset[seq_idx]['bboxes']
    
    action = all_preds['actions'][seq_idx]
    look = all_preds['looks'][seq_idx]
    cross = all_preds['crosses_frame'][seq_idx]
    
    # Find frame indices for this sequence
    # We need to map back to original frame indices from tracks
    track_id = ped_id
    if track_id in tracks:
        smoothed_track = smooth_track(tracks[track_id])
        # Find which window this sequence belongs to by matching bboxes
        for window_start in range(len(smoothed_track) - sequence_length + 1):
            window = smoothed_track[window_start:window_start+sequence_length]
            window_bboxes = [tuple(item['bbox']) for item in window]
            
            if window_bboxes == [tuple(bbox) for bbox in bboxes_seq]:
                frame_idxs = [item['frame_idx'] for item in window]
                for frame_idx, bbox in zip(frame_idxs, bboxes_seq):
                    if frame_idx not in frame_results:
                        frame_results[frame_idx] = []
                    frame_results[frame_idx].append({
                        'bbox': bbox,
                        'track_id': track_id,
                        'action': action,
                        'look': look,
                        'cross': cross,
                    })
                break

# ============================================================
# Visualization and Output Video
# ============================================================
print(f"Generating output video with predictions...")

LABEL_COLORS = {
    'action': (0, 255, 255),   # Yellow
    'look':   (255, 0, 255),   # Magenta
    'cross':  (255, 255, 0),   # Cyan
}
TEXT_COLOR = (0, 0, 0)  # Black for text

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = frame_results.get(frame_idx, [])
    for res in results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        
        # Draw ID label
        id_text = f'ID {res["track_id"]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
        y1_label = max(y1 - 22, 0)
        cv2.rectangle(frame, (x1, y1_label), (x1 + text_size[0], y1), (200, 200, 200), -1)
        cv2.putText(frame, id_text, (x1, y1 - 7), font, font_scale, TEXT_COLOR, thickness)
        
        # Map predictions to text labels using PIE
        cross_value = res['cross']
        # Ensure cross_value is valid for PIE mapping
        if cross_value == 2 or cross_value < 0:
            cross_value = -1
        
        try:
            action_text = pie._map_scalar_to_text('action', res['action'])
            look_text = pie._map_scalar_to_text('look', res['look'])
            cross_text = pie._map_scalar_to_text('cross', cross_value)
        except:
            action_text = f"A:{res['action']}"
            look_text = f"L:{res['look']}"
            cross_text = f"C:{res['cross']}"
        
        # Draw prediction labels
        label_names = ['action', 'look', 'cross']
        label_texts = [action_text, look_text, cross_text]
        x_offset = x1
        y_offset = y1_label - 22
        
        for label, text in zip(label_names, label_texts):
            color_bg = LABEL_COLORS[label]
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(
                frame,
                (x_offset, y_offset),
                (x_offset + text_size[0] + 6, y_offset + text_size[1] + 8),
                color_bg, -1
            )
            cv2.putText(
                frame,
                text,
                (x_offset + 3, y_offset + text_size[1] + 3),
                font, font_scale, TEXT_COLOR, thickness
            )
            x_offset += text_size[0] + 10
        
        # Draw color indicators below bbox
        color_list = [
            LABEL_COLORS['action'] if res['action'] else (50, 50, 50),
            LABEL_COLORS['look'] if res['look'] else (50, 50, 50),
            LABEL_COLORS['cross'] if res['cross'] else (50, 50, 50),
        ]
        for i, color in enumerate(color_list):
            cv2.rectangle(frame, (x1 + i*15, y2+5), (x1 + (i+1)*15, y2+20), color, -1)
    
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"\nVideo processing complete! Output saved as {out_video}.")
print(f"Total frames processed: {frame_idx}")
print(f"Total detections: {sum(len(v) for v in frame_results.values())}")
