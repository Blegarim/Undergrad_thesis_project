import torch
from torchvision import transforms
from ultralytics import YOLO

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
for track_id, track_data in tracks.items():
    smoothed_track = smooth_track(track_data)
    sequences = extract_sequences_from_track(smoothed_track, T=20)
    all_sequences.extend(sequences)

print(f"Extracted {len(all_sequences)} sequences from all tracks.")

images = torch.stack([seq[0] for seq in all_sequences]).to(device)  # [N, T, 3, 128, 128]
motions = torch.stack([seq[1] for seq in all_sequences]).to(device)  # [N, T, 3]

batch_size = 32
for i in range(0, len(images), batch_size):
    img_batch = images[i:i+batch_size]
    motion_batch = motions[i:i+batch_size]
    with torch.no_grad():
        outputs = model(img_batch, motion_batch)

preds = {k: v.argmax(dim=1) for k, v in outputs.items()}
print(preds)
