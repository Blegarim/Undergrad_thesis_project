from models.CNN_Feature_Extractor import CNNFeatureExtractor
from models.Motion_Transformer import MotionTransformer
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import MultimodalModel  # your class that combines all 3

import torch

cnn = CNNFeatureExtractor(backbone='mobilenetv2', embedding_dim=128)
motion = MotionTransformer(d_model=128, max_len=100, num_heads=8, num_layers=2)
cross_attn = CrossAttentionModule(d_model=128, num_heads=8, num_classes=2)

model = MultimodalModel(cnn, motion, cross_attn)

# test with dummy inputs here...
# Define dummy input sizes
batch_size = 4
seq_len = 10
img_channels, H, W = 3, 128, 128
motion_dim = 3

# Dummy image sequence: [B, T, 3, H, W]
dummy_images = torch.randn(batch_size, seq_len, img_channels, H, W)

# Dummy motion sequence: [B, T, 3]  (e.g., cx, cy, dt)
dummy_motions = torch.randn(batch_size, seq_len, motion_dim)

# Run through model
output = model(dummy_images, dummy_motions)

print("Output shape:", output.shape)  # Expect [B, num_classes], e.g., [4, 2]






