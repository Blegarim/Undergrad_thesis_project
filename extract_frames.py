import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps=None):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = video_fps
    
    frame_interval = int(round(video_fps / fps)) if fps < video_fps else 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 video")
    parser.add_argument("video", help="Path to input MP4 video")
    parser.add_argument("output", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=None, help="Extract frames at this FPS (default: video FPS)")
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.fps)
