from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

import torch
from torchvision import transforms

# Load model
model = YOLO('yolov8n.pt')

# Open video
video_path = 'videohaha.mp4'
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
            continue  # skip untracked objects

        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)

        # Optionally store the cropped image too
        cropped = frame[y1:y2, x1:x2].copy()

        tracks[track_id].append({
            "frame_idx": frame_idx,
            "bbox": bbox,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "image": cropped  # you can save or process later
        })

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

    frame_idx += 1

print(f"\nTotal pedestrian tracks: {len(tracks)}\n")

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # [0,1]
])

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

sequences = extract_sequences_from_track(tracks[3], T=10)
print(f"Generated {len(sequences)} sequences from Track 3")

cap.release()


