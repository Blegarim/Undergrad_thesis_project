import torch
from torchvision import transforms
import cv2

from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel

from pedestrian_detection import extract_tracks_from_video, smooth_track, extract_sequences_from_track
from PIE.utilities.pie_data import PIE
pie = PIE(data_path='PIE')

# Configuration
video_path = 'test_clip.mp4'
out_video = 'output_with_predictions_3.mp4'
model_path = 'outputs/best_model_epoch1.pth'



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
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the multimodal model
model = MultimodalModel(
        cnn_backbone=CNNFeatureExtractor(backbone='efficientnet_b0', embedding_dim=embedding_dim, pretrained=True, freeze_backbone=True),
        motion_transformer=MotionTransformer(d_model=embedding_dim, max_len=sequence_length, num_heads=4, num_layers=2),
        cross_attention=CrossAttentionModule(d_model=embedding_dim, num_heads=4, num_classes_dict=num_classes_dict)
    ).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set model to evaluation mode



tracks = extract_tracks_from_video(
    video_path=video_path,
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
    for f, bbox in zip(meta['frame_idxs'], meta['bboxes']):
        if f not in frame_results:
            frame_results[f] = []
        frame_results[f].append({
            'bbox': bbox,
            'track_id': meta['track_id'],
            'action': action,
            'look': look,
            'cross': cross,
        })

# Save the results to a video file with predictions
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(out_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))



LABEL_COLORS = {
    'action': (0, 255, 255),   # Yellow
    'look':   (255, 0, 255),   # Magenta
    'cross':  (255, 255, 0),   # Cyan
}
TEXT_COLOR = (0, 0, 0)  # Black for text

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = frame_results.get(frame_idx, [])
    for res in results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw bounding box

        # Draw ID label above all attribute labels
        id_text = f'ID {res["track_id"]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
        y1_label = max(y1 - 22, 0)
        cv2.rectangle(frame, (x1, y1_label), (x1 + text_size[0], y1), (200, 200, 200), -1)
        cv2.putText(frame, id_text, (x1, y1 - 7), font, font_scale, TEXT_COLOR, 2)

        cross_value = res['cross']
        if cross_value == 2:
            cross_value = -1

        action_text = pie._map_scalar_to_text('action', res['action'])
        look_text = pie._map_scalar_to_text('look', res['look'])
        cross_text = pie._map_scalar_to_text('cross', cross_value)

        # Draw each attribute label in its color, side by side above the box
        label_names = ['action', 'look', 'cross']
        label_texts = [
            f'{action_text}',
            f'{look_text}',
            f'{cross_text}',
        ]
        x_offset = x1
        y_offset = y1_label - 22  # Stack labels above ID label
        for label, text in zip(label_names, label_texts):
            color_bg = LABEL_COLORS[label]
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            # Draw background rectangle for each label
            cv2.rectangle(
                frame,
                (x_offset, y_offset),
                (x_offset + text_size[0] + 6, y_offset + text_size[1] + 8),
                color_bg, -1
            )
            # Put label text
            cv2.putText(
                frame,
                text,
                (x_offset + 3, y_offset + text_size[1] + 3),
                font, font_scale, TEXT_COLOR, 2
            )
            x_offset += text_size[0] + 10  # Add spacing between labels

        # Draw color indicators below the box (as before)
        color_list = [
            LABEL_COLORS['action'] if res['action'] else (50, 50, 50),
            LABEL_COLORS['look']   if res['look']   else (50, 50, 50),
            LABEL_COLORS['cross']  if res['cross']  else (50, 50, 50),
        ]
        for i, color in enumerate(color_list):
            cv2.rectangle(frame, (x1 + i*15, y2+5), (x1 + (i+1)*15, y2+20), color, -1)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Video processing complete! Output saved as {out_video}.")