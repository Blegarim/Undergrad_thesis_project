import torch, torch.nn as nn, torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1) 
        return self.dropout(self.fc2(F.gelu(x1) * x2))
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 num_heads=8, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, 
                                               num_heads=num_heads, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = GEGLU(d_model, dim_feedforward, dropout)

    def forward(self, x, key_padding_mask=None):
        y = self.norm1(x)
        attn_output, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)

        y = self.norm2(x)
        x = x +self.dropout(self.ffn(y))

        return x
    
class ViT_Hierarchical(nn.Module):
    def __init__(self,
                 img_size=128,
                 in_channels=3,
                 stage_dims=[64, 128, 256],
                 layer_nums=[2, 4, 6],
                 head_nums=[2, 4, 8],
                 ffn_ratios=3,
                 dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=stage_dims[0], 
                      kernel_size=7, 
                      stride=4, 
                      padding=3, 
                      bias=False),
            nn.BatchNorm2d(stage_dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        in_dim = stage_dims[0]
        for i, (dim, num_layers, num_heads) in enumerate(zip(stage_dims, layer_nums, head_nums)):
            if i !=0:
                down_sample = nn.Conv2d(in_channels=in_dim, 
                                        out_channels=dim, 
                                        kernel_size=3, 
                                        stride=2, 
                                        padding=1, 
                                        bias=False)
            block = nn.ModuleList([
                TransformerEncoderBlock(d_model=dim, 
                                        num_heads=num_heads, 
                                        dim_feedforward=dim*ffn_ratios, 
                                        dropout=dropout)
                for _ in range(num_layers)
            ])

            self.stages.append(nn.ModuleDict({
                'down_sample': down_sample if i != 0 else nn.Identity(),
                'block': block
            }))
            in_dim=dim
        self.norm = nn.LayerNorm(stage_dims[-1])
        self.frame_proj = nn.Linear(stage_dims[-1], 128)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # [B*T, C, H, W]
        x = self.stem(x)           # [B*T, D, H/4, W/4]
        for stage in self.stages:
            x = stage['down_sample'](x) # Downsampling
            B_T, D, H_s, W_s = x.shape
            tokens = x.flatten(2).transpose(1, 2) # [B*T, H_s*W_s, D]
            for block in stage['block']:
                tokens = block(tokens)                 # Transformer blocks
            
            x = tokens.transpose(1, 2).view(B_T, D, H_s, W_s) # Reshape back to image-like
        
        x = x.mean([2, 3]) # Global average pooling (B*T, D)
        x = self.norm(x)
        x = x.view(B, T, -1) # [B, T, D]
        x = self.frame_proj(x) # Project to desired 128 dim for cross-attention
        return x
    
if __name__ == "__main__":
    # Test the ViT_Hierarchical module
    batch_size = 2
    seq_len = 8
    img_size = 128
    in_channels = 3
    x = torch.randn(batch_size, seq_len, in_channels, img_size, img_size) # Example input

    model = ViT_Hierarchical(img_size=img_size, in_channels=in_channels, 
                             stage_dims=[32, 64, 128], 
                             layer_nums=[2, 4, 6], 
                             head_nums=[2, 4, 8], 
                             ffn_ratios=3, dropout=0.1)
    out = model(x)
    print("Output shape:", out.shape) # Expected: [batch_size, seq_len, 128]
    print ("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


