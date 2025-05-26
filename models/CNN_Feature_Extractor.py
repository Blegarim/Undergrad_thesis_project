import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='mobilenetv2', embedding_dim=128, pretrained=True, freeze_backbone=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        if backbone == 'mobilenetv2':
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            cnn_out_dim = base_model.classifier[1].in_features
            self.feature_extractor = base_model.features
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            cnn_out_dim = base_model.classifier[1].in_features
            self.feature_extractor = base_model.features
        elif backbone == 'custom':
            # Custom CNN backbone (not implemented)
            raise NotImplementedError("Custom CNN not implemented yet.")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x shape: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)                # Merge batch and time
        features = self.feature_extractor(x)      # Shape: [B*T, C, H', W']
        pooled = self.pool(features)              # Shape: [B*T, C, 1, 1]
        flattened = pooled.view(B * T, -1)        # Shape: [B*T, C]
        embeddings = self.fc_layers(flattened)    # Shape: [B*T, D]
        return embeddings.view(B, T, -1)          # Shape: [B, T, D]

if __name__ == '__main__':
    dummy_input = torch.randn(8, 10, 3, 128, 128)  # [B, T, C, H, W]
    model = CNNFeatureExtractor(backbone='efficientnet_b0', embedding_dim=128, freeze_backbone=True)
    output = model(dummy_input)
    print(output.shape)  # Expected: [8, 10, 128]


