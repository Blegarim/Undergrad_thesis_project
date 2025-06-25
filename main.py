import torch
from torchvision import transforms
import cv2

from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel

from pedestrian_detection import extract_tracks_from_video, smooth_track, extract_sequences_from_track


default_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  
])
embedding_dim = 128
sequence_length = 20
num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 3,
        'gestures': 6
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the multimodal model
model = MultimodalModel(
        cnn_backbone=CNNFeatureExtractor(backbone='efficientnet_b0', embedding_dim=embedding_dim),
        motion_transformer=MotionTransformer(d_model=embedding_dim, max_len=sequence_length, num_heads=8, num_layers=2),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=8, num_classes_dict=num_classes_dict)
    ).to(device)

model.load_state_dict(torch.load('outputs/final_model_epoch5.pth', map_location=device))
model.eval()  # Set model to evaluation mode

tracks = extract_tracks_from_video(
    video_path='test_clip.mp4',
    model_path='yolo11n.pt',
    class_idx=0, 
    conf=0.3,
    show=False
)
print(f"\nTotal pedestrian tracks: {len(tracks)}\n")
    
all_sequences = []
all_seq_meta = []

for track_id, track_data in tracks.items():
    smoothed_track = smooth_track(track_data)
    sequences = extract_sequences_from_track(smoothed_track, T=20)
    for i, (imgs, motions) in enumerate(sequences):
        meta = {
            'track_id': track_id,
            'frame_idxs': [item['frame_idx'] for item in smoothed_track[i:i+20]],
            'bboxes': [item['bbox'] for item in smoothed_track[i:i+20]]
        }
        all_sequences.append((imgs, motions))
        all_seq_meta.append(meta)

print(f"Extracted {len(all_sequences)} sequences from all tracks.")

images = torch.stack([seq[0] for seq in all_sequences]).to(device)  # [N, T, 3, 128, 128]
motions = torch.stack([seq[1] for seq in all_sequences]).to(device)  # [N, T, 3]

batch_size = 32
all_preds = {k: [] for k in num_classes_dict.keys()}
for i in range(0, len(images), batch_size):
    img_batch = images[i:i+batch_size]
    motion_batch = motions[i:i+batch_size]
    with torch.no_grad():
        outputs = model(img_batch, motion_batch)
    for k in outputs:
        batch_preds = outputs[k].argmax(dim=1).cpu().tolist()
        all_preds[k].extend(batch_preds)

# {frame_idx: [ (bbox, track_id, action, cross, gesture, ...), ... ]}
frame_results = {}

for idx, meta in enumerate(all_seq_meta):
    action = all_preds['actions'][idx]
    look = all_preds['looks'][idx]
    cross = all_preds['crosses'][idx]
    gesture = all_preds['gestures'][idx]
    for f, bbox in zip(meta['frame_idxs'], meta['bboxes']):
        if f not in frame_results:
            frame_results[f] = []
        frame_results[f].append({
            'bbox': bbox,
            'track_id': meta['track_id'],
            'action': action,
            'look': look,
            'cross': cross,
            'gesture': gesture
        })

# Save the results to a video file with predictions
cap = cv2.VideoCapture('test_clip.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_predictions.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

LABEL_COLORS = {
    'action': (0, 255, 255),   # Yellow
    'look':   (255, 0, 255),   # Magenta
    'cross':  (255, 255, 0),   # Cyan
    'gesture':(0, 165, 255),   # Orange
}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = frame_results.get(frame_idx, [])
    for res in results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw bounding box
        text = f'ID {res["track_id"]} | Action: {res["action"]}, Look: {res["look"]}, Cross: {res["cross"]}, Gesture: {res["gesture"]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        color_bg = LABEL_COLORS['gesture']
        # Ensure text fits within the frame
        x2_label = min(x1 + text_size[0], frame.shape[1]-1)
        y1_label = max(y1 - 22, 0)
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + text_size[0], y1), color_bg, -1)
        cv2.putText(frame, text, (x1, y1 - 7), font, font_scale, (0, 0, 0), 2)

        color_list = [
            LABEL_COLORS['action'] if res['action'] else (50, 50, 50),
            LABEL_COLORS['look']   if res['look']   else (50, 50, 50),
            LABEL_COLORS['cross']  if res['cross']  else (50, 50, 50),
            LABEL_COLORS['gesture']if res['gesture']else (50, 50, 50),
        ]
        for i, color in enumerate(color_list):
            cv2.rectangle(frame, (x1 + i*15, y2+5), (x1 + (i+1)*15, y2+20), color, -1)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Video processing complete! Output saved as 'output_with_predictions.mp4'.")