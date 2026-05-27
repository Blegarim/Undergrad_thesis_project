"""
Visualization script for comparing ground truth labels vs model predictions.

Modes:
    --mode=gt      : Ground truth labels only
    --mode=pred    : Model predictions only
    --mode=both    : Side-by-side comparison
    --mode=diff    : Highlight mismatches

Data paths (matching project structure):
    - Annotations: PIE/annotations/annotations/<set>/<video>_annt.xml
    - Images: data/images/<set>/<video>/
    - LMDB: data/lmdb/

Usage:
    python visualize_comparison.py --mode=both --video=set03/video_0010
    python visualize_comparison.py --mode=pred --video=test_clip2.mp4
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from models.AblationModels import MotionOnlyModel, VisualOnlyModel, VanillaConcatModel
from config import vit_args_config, motion_enc_args_config
from scripts.model_utils import get_model, model_forward
from scripts.PIE_sequence_Dataset_1 import PIESequenceDataset
from PIE.utilities.pie_data import PIE

# Constants - matching generate_sequences.py
SEQ_LEN = 20
FUTURE_OFFSET = 30
TOL = 2

# Visualization colors
LABEL_COLORS = {
    'action': (0, 255, 255),   # Yellow - walking
    'look': (255, 0, 255),     # Magenta - looking
    'cross': (255, 255, 0),    # Cyan - crossing
}
BOX_COLOR = (0, 255, 0)        # Green
MISMATCH_COLOR = (0, 0, 255)  # Red


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GT vs Predictions')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['gt', 'pred', 'both', 'diff'],
                       help='Visualization mode')
    parser.add_argument('--video', type=str, default='set03/video_0010',
                       help='Video identifier (set_id/video_id or .mp4 file)')
    parser.add_argument('--output', type=str, default='comparison_output.mp4',
                       help='Output video path')
    parser.add_argument('--model', type=str, 
                       default='best_model_outputs/best_model_epoch10_0121_2029.pth',
                       help='Model checkpoint path')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Max frames to process')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame index')
    parser.add_argument('--stride', type=int, default=3,
                       help='Sequence stride for sliding window')
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model for inference - matching main.py/train.py config."""
    embedding_dim = 128
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    num_classes_dict = {'actions': 2, 'looks': 2, 'crosses': 2}

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
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model


def get_pie_annotations(pie_path, set_id, vid_id):
    """Parse PIE XML annotations for ground truth."""
    import xml.etree.ElementTree as ET
    
    # Try different annotation paths
    annt_paths = [
        os.path.join(pie_path, 'annotations', 'annotations', set_id, f'{vid_id}_annt.xml'),
        os.path.join(pie_path, 'annotations', set_id, f'{vid_id}_annt.xml'),
    ]
    
    annt_file = None
    for path in annt_paths:
        if os.path.exists(path):
            annt_file = path
            break
    
    if annt_file is None:
        return None
    
    tree = ET.parse(annt_file)
    root = tree.getroot()
    
    num_frames = int(root.find("./meta/task/size").text)
    width = int(root.find("./meta/task/original_size/width").text)
    height = int(root.find("./meta/task/original_size/height").text)

    ped_data = {}
    for track in root.findall('./track'):
        if track.get('label') != 'pedestrian':
            continue
        boxes = track.findall('./box')
        if not boxes:
            continue
        
        ped_id = boxes[0].find('./attribute[@name="id"]').text
        frames, bboxes, actions, looks, crosses = [], [], [], [], []
        
        for box in boxes:
            if int(box.get('outside')) == 1:
                continue
            frame_id = int(box.get('frame'))
            x1 = float(box.get('xtl'))
            y1 = float(box.get('ytl'))
            x2 = float(box.get('xbr'))
            y2 = float(box.get('ybr'))
            
            # Parse action
            action_elem = box.find('./attribute[@name="action"]')
            action_val = 1 if (action_elem is not None and action_elem.text == 'walking') else 0
            
            # Parse look
            look_elem = box.find('./attribute[@name="look"]')
            look_val = 1 if (look_elem is not None and look_elem.text == 'looking') else 0
            
            # Parse cross
            cross_elem = box.find('./attribute[@name="cross"]')
            if cross_elem is not None and cross_elem.text:
                cross_map = {'not-crossing': 0, 'crossing': 1, 'crossing-irrelevant': -1}
                cross_val = cross_map.get(cross_elem.text, -1)
            else:
                cross_val = -1
            
            frames.append(frame_id)
            bboxes.append([x1, y1, x2, y2])
            actions.append(action_val)
            looks.append(look_val)
            crosses.append(cross_val)
        
        if frames:
            ped_data[ped_id] = {
                'frames': frames, 
                'bbox': bboxes, 
                'action': actions, 
                'look': looks, 
                'cross': crosses
            }
    
    return {'num_frames': num_frames, 'width': width, 'height': height, 'ped_data': ped_data}


def generate_sequences_from_annotations(annotations, seq_len=SEQ_LEN, stride=3, 
                                      future_offset=FUTURE_OFFSET, tol=TOL):
    """
    Generate sequences from annotations with future labels.
    Exactly matching scripts/generate_sequences.py logic.
    """
    sequences = []
    
    for ped_id, data in annotations['ped_data'].items():
        frames = data['frames']
        bboxes = data['bbox']
        actions = data['action']
        looks = data['look']
        crosses = data['cross']
        
        # Clamp crosses to binary (matching generate_sequences.py)
        crosses_binary = [1 if c == 1 else 0 for c in crosses]
        n = len(frames)
        
        if n < seq_len + future_offset:
            continue
        
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            
            # Skip sequences with crossing in input window (matching generate_sequences.py)
            if any(crosses_binary[start:end]):
                continue
            
            future_start = end
            future_end = min(end + future_offset + tol, n)
            
            # Future labels: 1 if action occurs in future window
            action_event = 1 if any(actions[future_start:future_end]) else 0
            look_event = 1 if any(looks[future_start:future_end]) else 0
            cross_event = 1 if any(crosses_binary[future_start:future_end]) else 0
            
            sequences.append({
                'ped_id': ped_id,
                'frames': frames[start:end],
                'bboxes': bboxes[start:end],
                # Per-frame labels
                'actions': actions[start:end],
                'looks': looks[start:end],
                'crosses': crosses_binary[start:end],
                # Future labels (matching train.py/test.py label keys)
                'gt_action': action_event,
                'gt_look': look_event,
                'gt_cross': cross_event,
            })
    
    return sequences


def draw_labels(frame, action, look, cross, x1, y1, x2, y2, prefix=''):
    """Draw labels on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    labels = [
        ('action', f'{prefix}WALK', action, LABEL_COLORS['action']),
        ('look', f'{prefix}LOOK', look, LABEL_COLORS['look']),
        ('cross', f'{prefix}CROSS', cross, LABEL_COLORS['cross']),
    ]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    
    y_offset = y1 - 22
    x_offset = x1
    
    for _, text, val, color in labels:
        status = text if val == 1 else f'{prefix}---'
        bg_color = color if val == 1 else (50, 50, 50)
        txt_color = (0, 0, 0) if val == 1 else (100, 100, 100)
        
        size, _ = cv2.getTextSize(status, font, font_scale, thickness)
        cv2.rectangle(frame, (x_offset, y_offset), 
                     (x_offset + size[0] + 6, y_offset + size[1] + 8), bg_color, -1)
        cv2.putText(frame, status, (x_offset + 3, y_offset + size[1] + 3),
                    font, font_scale, txt_color, thickness)
        x_offset += size[0] + 10
    
    return frame


def draw_comparison(frame, gt_action, gt_look, gt_cross, 
                   pred_action, pred_look, pred_cross, x1, y1, x2, y2):
    """Draw side-by-side comparison with mismatch highlighting."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    
    labels = [
        ('WALK', gt_action, pred_action, LABEL_COLORS['action']),
        ('LOOK', gt_look, pred_look, LABEL_COLORS['look']),
        ('CROSS', gt_cross, pred_cross, LABEL_COLORS['cross']),
    ]
    
    y_offset = y1 - 22
    x_offset = x1
    
    for name, gt_val, pred_val, color in labels:
        match = gt_val == pred_val
        gt_status = name if gt_val == 1 else '---'
        pred_status = name if pred_val == 1 else '---'
        
        text = f'{gt_status}|{pred_status}'
        bg_color = color if match else MISMATCH_COLOR
        txt_color = (0, 0, 0)
        
        size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x_offset, y_offset),
                     (x_offset + size[0] + 6, y_offset + size[1] + 8), bg_color, -1)
        cv2.putText(frame, text, (x_offset + 3, y_offset + size[1] + 3),
                    font, font_scale, txt_color, thickness)
        x_offset += size[0] + 10
    
    return frame


def process_pie_dataset(args):
    """Process PIE dataset with ground truth."""
    parts = args.video.split('/')
    if len(parts) != 2:
        print(f"Invalid video format: {args.video}. Use format: set_id/video_id (e.g., set03/video_0010)")
        return
    
    set_id, vid_id = parts[0], parts[1]
    
    # Paths - matching project structure
    pie_path = os.path.join(ROOT_DIR, 'PIE')
    data_path = os.path.join(ROOT_DIR, 'data')
    
    # Get annotations
    annotations = get_pie_annotations(pie_path, set_id, vid_id)
    if annotations is None:
        print(f"Annotations not found for {set_id}/{vid_id}")
        print(f"Searched in: {pie_path}/annotations/")
        return
    
    # Generate sequences with GT labels (matching generate_sequences.py)
    sequences = generate_sequences_from_annotations(annotations, stride=args.stride)
    print(f"Generated {len(sequences)} sequences from {set_id}/{vid_id}")
    
    if len(sequences) == 0:
        print("No sequences generated! Try increasing --stride or check annotation data.")
        return
    
    # Filter sequences based on frame range if max_frames specified
    max_frame = args.max_frames if args.max_frames else annotations['num_frames']
    start_frame = args.start_frame
    end_frame = start_frame + max_frame
    
    # Keep only sequences that overlap with our frame range
    sequences = [s for s in sequences if any(f >= start_frame and f < end_frame for f in s['frames'])]
    print(f"Filtered to {len(sequences)} sequences within frame range {start_frame}-{end_frame}")
    
    if len(sequences) == 0:
        print("No sequences in specified frame range!")
        return
    
    # Image paths - check data/images/ first, then PIE/images/
    img_folder = os.path.join(data_path, 'images', set_id, vid_id)
    if not os.path.exists(img_folder):
        img_folder = os.path.join(pie_path, 'images', set_id, vid_id)
    images_exist = os.path.exists(img_folder)
    
    if not images_exist:
        print(f"Warning: Image folder not found: {img_folder}")
    
    # ==== Model Inference (if not GT-only mode) ====
    model = None
    predictions = {}  # key: (ped_id, frame_idx) -> {'action': pred, 'look': pred, 'cross': pred}
    
    if args.mode in ['pred', 'both', 'diff']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model
        model_path = os.path.join(ROOT_DIR, args.model) if not os.path.isabs(args.model) else args.model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, device)
        model.eval()
        
        # Prepare image paths in sequences format for PIESequenceDataset
        for seq in sequences:
            seq['images'] = [os.path.join(img_folder, f'{frame_idx+1:05d}.jpg') for frame_idx in seq['frames']]
        
        # Create transforms (matching train.py) - add Resize for consistent sizes
        base_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset and dataloader
        print("Creating dataset for inference...")
        dataset = PIESequenceDataset(
            sequences,
            transform_tight=base_transforms,
            transform_context=base_transforms,
            crop=True,
            context_scale=2.0,
            return_metadata=True,
            preload=True
        )
        
        def inference_collate_fn(batch):
            images_tight = torch.stack([item['images_tight'] for item in batch])
            images_context = torch.stack([item['images_context'] for item in batch])
            motions = torch.stack([item['motions'] for item in batch])[..., :8]
            meta = [item['meta'] for item in batch]
            return images_tight, images_context, motions, meta
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=inference_collate_fn)
        
        # Run inference
        print("Running model inference...")
        seq_idx = 0
        with torch.no_grad():
            for images_tight, images_context, motions, meta in dataloader:
                images_tight = images_tight.to(device)
                images_context = images_context.to(device)
                motions = motions.to(device)
                
                outputs = model(images_tight, images_context, motions)
                
                # Debug: print output shapes
                if seq_idx == 0:
                    print("Debug - output shapes:")
                    for k, v in outputs.items():
                        print(f"  {k}: {v.shape}")
                    print(f"  batch_size: {images_tight.size(0)}")
                
                # Extract predictions
                batch_size = images_tight.size(0)
                for i in range(batch_size):
                    ped_id = meta[i].get('ped_id', seq_idx)
                    
                    # Get predictions - check output shape first
                    if outputs['actions'].dim() == 2:
                        action_pred = outputs['actions'][i].argmax(dim=0).item()
                    else:
                        action_pred = outputs['actions'][i].item()
                    
                    if outputs['looks'].dim() == 2:
                        look_pred = outputs['looks'][i].argmax(dim=0).item()
                    else:
                        look_pred = outputs['looks'][i].item()
                    
                    if 'crosses_frame' in outputs:
                        if outputs['crosses_frame'].dim() == 2:
                            cross_pred = outputs['crosses_frame'][i].argmax(dim=0).item()
                        else:
                            cross_pred = outputs['crosses_frame'][i].item()
                    else:
                        if outputs['crosses'].dim() == 2:
                            cross_pred = outputs['crosses'][i].argmax(dim=0).item()
                        else:
                            cross_pred = outputs['crosses'][i].item()
                    
                    # Store predictions for each frame in the sequence
                    seq = sequences[seq_idx]
                    for frame_idx in seq['frames']:
                        predictions[(ped_id, frame_idx)] = {
                            'action': action_pred,
                            'look': look_pred,
                            'cross': cross_pred
                        }
                    
                    seq_idx += 1
        
        print(f"Inference complete. Generated predictions for {len(predictions)} frame-ped pairs.")
    
    if not images_exist:
        print(f"Warning: Image folder not found: {img_folder}")
        print("Creating blank frames with annotations...")
    
    width, height = annotations['width'], annotations['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 10, (width, height))
    
    # Build frame index for GT
    frame_gt = {}
    for seq in sequences:
        for frame_idx, bbox in zip(seq['frames'], seq['bboxes']):
            if frame_idx not in frame_gt:
                frame_gt[frame_idx] = []
            frame_gt[frame_idx].append({
                'bbox': bbox,
                'ped_id': seq['ped_id'],
                'gt_action': seq['gt_action'],
                'gt_look': seq['gt_look'],
                'gt_cross': seq['gt_cross'],
            })
    
    for frame_idx in range(max_frame):
        actual_frame = start_frame + frame_idx
        if images_exist:
            img_path = os.path.join(img_folder, f'{actual_frame+1:05d}.jpg')
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if actual_frame in frame_gt:
            for det in frame_gt[actual_frame]:
                x1, y1, x2, y2 = map(int, det['bbox'])
                ped_id = det['ped_id']
                
                if args.mode == 'gt':
                    frame = draw_labels(frame, det['gt_action'], det['gt_look'], 
                                       det['gt_cross'], x1, y1, x2, y2, prefix='GT:')
                
                elif args.mode == 'pred':
                    # Get model predictions
                    pred = predictions.get((ped_id, actual_frame), {'action': 0, 'look': 0, 'cross': 0})
                    frame = draw_labels(frame, pred['action'], pred['look'], 
                                       pred['cross'], x1, y1, x2, y2, prefix='PD:')
                
                elif args.mode in ['both', 'diff']:
                    # Get model predictions
                    pred = predictions.get((ped_id, actual_frame), {'action': 0, 'look': 0, 'cross': 0})
                    frame = draw_comparison(frame, det['gt_action'], det['gt_look'], 
                                          det['gt_cross'], pred['action'], pred['look'], pred['cross'], x1, y1, x2, y2)
        
        # Add mode label
        cv2.putText(frame, f"Mode: {args.mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Processed {frame_idx + 1}/{max_frame} frames")
    
    out.release()
    print(f"Output saved: {args.output}")


def process_custom_video(args):
    """Process custom video with model predictions (similar to main.py)."""
    if not HAS_DETECTION or not HAS_DATASET:
        print("Error: Required modules not available for video processing")
        print(f"  HAS_DETECTION: {HAS_DETECTION}")
        print(f"  HAS_DATASET: {HAS_DATASET}")
        return
    
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(ROOT_DIR, args.model) if not os.path.isabs(args.model) else args.model
    model = load_model(model_path, device)
    
    # Extract tracks (matching main.py)
    print("Extracting pedestrian tracks...")
    tracks = extract_tracks_from_video(
        video_path=video_path,
        model_path='yolo11n.pt',
        class_idx=0,
        conf=0.3,
        show=False
    )
    print(f"Found {len(tracks)} tracks")
    
    # Build sequences (matching main.py)
    sequences = []
    for track_id, track_data in tracks.items():
        smoothed = smooth_track(track_data)
        if len(smoothed) < SEQ_LEN:
            continue
        for i in range(len(smoothed) - SEQ_LEN + 1):
            window = smoothed[i:i+SEQ_LEN]
            sequences.append({
                'images': [item['image'] for item in window],
                'bboxes': [item['bbox'] for item in window],
                'ped_id': track_id,
            })
    
    if not sequences:
        print("No valid sequences!")
        return
    
    # Run inference (matching main.py)
    print("Running inference...")
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PIESequenceDataset(
        sequences, 
        transform_tight=base_transforms,
        transform_context=base_transforms,
        crop=True, 
        context_scale=2.0,
        return_metadata=True, 
        preload=True
    )
    
    def collate_fn(batch):
        images_t = torch.stack([item['images_tight'] for item in batch])
        images_c = torch.stack([item['images_context'] for item in batch])
        motions = torch.stack([item['motions'] for item in batch])[..., :8]
        bboxes = [item['bboxes'] for item in batch]
        meta = [item['meta'] for item in batch]
        return images_t, images_c, motions, bboxes, meta
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    preds = []
    with torch.no_grad():
        for images_t, images_c, motions, _, meta in dataloader:
            images_t = images_t.to(device)
            images_c = images_c.to(device)
            motions = motions.to(device)
            outputs = model(images_t, images_c, motions)
            
            # Extract predictions (matching main.py)
            preds.extend([{
                'action': outputs['actions'].argmax(dim=1).cpu().tolist()[i],
                'look': outputs['looks'].argmax(dim=1).cpu().tolist()[i],
                'cross': outputs['crosses_frame'].argmax(dim=1).cpu().tolist()[i],
            } for i in range(len(meta))])
    
    # Generate output video (matching main.py)
    print("Generating output video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    for seq_idx, seq in enumerate(sequences):
        pred = preds[seq_idx] if seq_idx < len(preds) else {'action': 0, 'look': 0, 'cross': 0}
        
        for frame_idx, (img, bbox) in enumerate(zip(seq['images'], seq['bboxes'])):
            if isinstance(img, str) and os.path.exists(img):
                frame = cv2.imread(img)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            if frame is not None:
                x1, y1, x2, y2 = map(int, bbox)
                frame = draw_labels(frame, pred['action'], pred['look'], 
                                   pred['cross'], x1, y1, x2, y2, prefix='PD:')
                out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output saved: {args.output}")


def main():
    args = parse_args()
    
    print(f"Mode: {args.mode}")
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    
    # Determine processing mode based on video path
    if args.video.endswith('.mp4') or os.path.exists(args.video):
        process_custom_video(args)
    elif '/' in args.video and 'video_' in args.video:
        process_pie_dataset(args)
    else:
        print(f"Unknown video format: {args.video}")
        print("Use either:")
        print("  - PIE dataset: set03/video_0010")
        print("  - Custom video: test_clip2.mp4")
    
    print("Done!")


if __name__ == '__main__':
    main()
