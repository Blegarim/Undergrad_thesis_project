import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEncoder(nn.Module):
    def __init__(self, 
                 motion_dim=8, 
                 hidden_dim=128, 
                 d_model=128, 
                 num_layers=2, 
                 num_heads=8, 
                 dropout=0.3):
        super().__init__()

        #Images Feature Extraction
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1,1))
        )

        #Motion feature encoding
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )

        #Temporal processing
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Gated Recurrent Unit (GRU) ---
        self.gru = nn.GRU(input_size=hidden_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0
        )
        
        # --- Attention ---
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, d_model) if hidden_dim != d_model else nn.Identity()
    
    def forward(self, motion_data, images_data):
        """
        Args:
            motion_data: [batch_size, seq_len, motion_dim] - raw motion features
            image_data: [batch_size, seq_len, 3, H, W] - tight crops
        Returns:
            [B, T, d_model]
        """
        B, T = motion_data.shape[:2]
        
        # Process images
        images_data = images_data.flatten(0, 1)  # [B*T, 3, H, W]
        img_feats = self.img_encoder(images_data)
        img_feats = img_feats.squeeze(-1).squeeze(-1)  # [B*T, hidden_dim]
        img_feats = img_feats.view(B, T, -1)  # [B, T, hidden_dim]

        motion_norm = (motion_data - motion_data.mean(dim=1, keepdim=True)) / (motion_data.std(dim=1, keepdim=True) + 1e-6)

        motion_data = motion_norm.transpose(1, 2) # [B, motion_dim, T]
        motion_feats = self.motion_encoder(motion_data) # [B, hidden_dim/2, T]
        motion_feats = motion_feats.transpose(1, 2) # [B, T, hidden_dim/2]
        
        combined = torch.cat([img_feats, motion_feats], dim=-1)  # [B, T, hidden_dim * 1.5]
        x = self.fusion(combined)  # [B, T, hidden_dim]

        x, _ = self.gru(x) # [B, T, hidden_dim]

        x = x + self.pos_encoding[:, :T, :]

        residual = x
        x = self.norm(x)
        x, _ = self.temporal_attn(x, x, x)
        x = self.proj(residual + self.dropout(x)) # [B, T, d_model]
        
        return x
if __name__ == "__main__":
    batch_size = 8
    seq_len = 20
    motion_dim = 8
    img_size = 128
    
    motion_input = torch.randn(batch_size, seq_len, motion_dim)
    image_input = torch.randn(batch_size, seq_len, 3, img_size, img_size)
    
    model = MotionEncoder(
        motion_dim=motion_dim,
        hidden_dim=224,
        d_model=128
    )
    
    out = model(motion_input, image_input)
    print("Output shape:", out.shape)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")