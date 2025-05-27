import cv2
from pathlib import Path

# Root directory of the video files
set_id = 'set01'
ROOT_DIR = Path(__file__).resolve().parents[1]

video_dir = ROOT_DIR / 'data' / 'raw_videos' / set_id
output_base = ROOT_DIR / 'data' / 'images' / set_id
frame_skip = 1

# Check if the video directory exists
output_base.mkdir(parents=True, exist_ok=True)

for file in video_dir.glob('*.mp4'):
    video_name = file.stem
    output_dir = output_base / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(file))
    if not cap.isOpened():
        print(f"[!] Failed to open video file: {file}")
        continue

    frame_id = 0
    saved_id = 0

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip == 0:
            if count % 100 == 0:
                print(f"Processing {video_name}, Frame: {frame_id}, Saved: {saved_id}")
            frame_name = f"{saved_id:05d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_id += 1
            count += 1
        frame_id += 1

    cap.release()
    print(f"Finished extracting {saved_id} frames from {video_name}")

print("✅ All videos processed.")
