from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

import torch
from torchvision import transforms
import numpy as np

# Define the default image transform
default_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # [0,1]
])

def extract_tracks_from_video(
    video_path,
    model_path='yolo11n.pt',
    class_idx=0,
    conf=0.3,
    show=False
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracks = defaultdict(list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[class_idx], conf=conf)[0]
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            # Guard against invalid crop (sometimes y2==y1 or x2==x1)
            if y2 <= y1 or x2 <= x1:
                continue
            cropped = frame[y1:y2, x1:x2].copy()
            tracks[track_id].append({
                "frame_idx": frame_idx,
                "bbox": (x1, y1, x2, y2),
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "image": cropped
            })
            if show:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()
    return tracks

def smooth_track(track_data, window=3):
    smoothed = []
    for i in range(len(track_data)):
        start = max(0, i - window)
        end = min(len(track_data), i + window + 1)
        x = np.mean([track_data[j]['cx'] for j in range(start, end)])
        y = np.mean([track_data[j]['cy'] for j in range(start, end)])
        frame_idx = track_data[i]['frame_idx']
        img = track_data[i]['image']
        bbox = track_data[i]['bbox']
        smoothed.append({'cx': x, 'cy': y, 'dt': 0, 'frame_idx': frame_idx, 'image': img, 'bbox': bbox})
    return smoothed

def extract_sequences_from_track(track_data, T=20, img_transform=default_img_transform):
    sequences = []
    if len(track_data) < T:
        return sequences

    for i in range(len(track_data) - T + 1):
        window = track_data[i:i+T]
        imgs = []
        motions = []
        for j in range(T):
            item = window[j]
            # Defensive: make sure the image is not empty
            if item['image'].size == 0:
                continue
            img = img_transform(item['image'])  # [3, 128, 128]
            imgs.append(img)
            cx = item['cx']
            cy = item['cy']
            dt = 0 if j == 0 else item['frame_idx'] - window[j-1]['frame_idx']
            motions.append([cx, cy, dt])
        if len(imgs) == T and len(motions) == T:
            imgs_tensor = torch.stack(imgs)                 # [T, 3, 128, 128]
            motion_tensor = torch.tensor(motions, dtype=torch.float32)  # [T, 3]
            sequences.append((imgs_tensor, motion_tensor))
        # else: skip sequence if not all images present
    return sequences

def main():
    video_path = 'test_clip.mp4'
    model_path = 'yolo11n.pt'
    class_idx = 0  # Assuming class 0 is the pedestrian class
    conf = 0.3
    show = True

    tracks = extract_tracks_from_video(video_path, model_path, class_idx, conf, show)
    
    print(f"\nTotal pedestrian tracks: {len(tracks)}\n")
    
    all_sequences = []
    for track_id, track_data in tracks.items():
        smoothed_track = smooth_track(track_data)
        sequences = extract_sequences_from_track(smoothed_track, T=20)
        all_sequences.extend(sequences)

    print(f"Extracted {len(all_sequences)} sequences from all tracks.")

if __name__ == "__main__":
    main()