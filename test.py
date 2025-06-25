from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

import torch
from torchvision import transforms
import numpy as np

# Load model
model = YOLO('yolo11n.pt')

# Open video
video_path = 'test_clip.mp4'
cap = cv2.VideoCapture(video_path)

# Output structure: {track_id: list of (frame_idx, bbox, frame)}
tracks = defaultdict(list)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(frame, persist=True, classes=[0], conf=0.3)[0]

    for box in results.boxes:
        if box.id is None:
            continue

        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        cropped = frame[y1:y2, x1:x2].copy()

        tracks[track_id].append({
            "frame_idx": frame_idx,
            "bbox": (x1, y1, x2, y2),
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "image": cropped
        })

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Tracking", frame)

    frame_idx += 1

print(f"\nTotal pedestrian tracks: {len(tracks)}\n")

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # [0,1]
])

def smooth_track(track_data, window=3):
    smoothed = []
    for i in range(len(track_data)):
        start = max(0, i - window)
        end = min(len(track_data), i + window + 1)
        x = np.mean([track_data[j]['cx'] for j in range(start, end)])
        y = np.mean([track_data[j]['cy'] for j in range(start, end)])
        frame_idx = track_data[i]['frame_idx']
        img = track_data[i]['image']
        smoothed.append({'cx': x, 'cy': y, 'dt': 0, 'frame_idx': frame_idx, 'image': img})
    return smoothed

def extract_sequences_from_track(track_data, T=20):
    sequences = []

    # Need at least T frames to form one sequence
    if len(track_data) < T:
        return sequences

    for i in range(len(track_data) - T + 1):
        window = track_data[i:i+T]
        
        # Prepare image tensor list
        imgs = []
        motions = []

        for j in range(T):
            item = window[j]
            img = img_transform(item['image'])  # [3, 128, 128]
            imgs.append(img)

            cx = item['cx']
            cy = item['cy']
            dt = 0 if j == 0 else item['frame_idx'] - window[j-1]['frame_idx']
            motions.append([cx, cy, dt])

        imgs_tensor = torch.stack(imgs)              # [T, 3, 128, 128]
        motion_tensor = torch.tensor(motions)        # [T, 3]
        
        sequences.append((imgs_tensor, motion_tensor))

    return sequences

total_sequences = 0
for track_id, track_data in tracks.items():
    track_data = smooth_track(track_data=track_data)
    sequences = extract_sequences_from_track(track_data, T=20)
    if len(sequences) == 0:
        continue
    print(f"Generated {len(sequences)} sequences from Track {track_id}")
    total_sequences += len(sequences)

print(f"\nTotal sequences extracted: {total_sequences}")

cap.release()


