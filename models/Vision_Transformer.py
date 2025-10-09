import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

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
    '''
    A standard Transformer encoder block with global multi-head self-attention and feedforward network.'''
    def __init__(self, d_model=128, num_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout = dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = GEGLU(d_model, dim_feedforward, dropout)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        mask: Optional mask tensor of shape [batch_size, seq_len]
        """
        # Self-attention
        y = self.norm1(x)
        attn_output, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)

        y = self.norm2(x)
        ffn_output = self.ffn(y)
        x = x + self.dropout(ffn_output)

        return x

class MLP(nn.Module):
    def __init__(self, dim=128, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

def window_partition(x, window_size):
    '''
    Args: 
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    '''
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # (B, H//ws, ws, W//ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size*window_size, C)  # (num_windows*B, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    Ws = window_size
    num_windows = windows.shape[0]
    B = num_windows // ((H // Ws) * (W // Ws))
    x = windows.view(B, H // Ws, W // Ws, Ws, Ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    '''
    Window based multi-head self attention (W-MSA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        dropout (float, optional): Dropout ratio. Default: 0.0
    Returns:
        Tensor: (B, N, C)
    '''
    def __init__(self, dim, window_size, num_heads,qkv_bias=True, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.dim = dim 
        self.window_size = window_size # (Wh, Ww)
        Wh, Ww = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads 
        self.scale = (dim // num_heads) ** -0.5 # Scaling factor for dot-product attention
        assert self.dim % self.num_heads == 0, "dim should be divisible by num_heads"

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std = 0.02)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij')) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] -1 # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] -1 # Shift to start from 0
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # Multiply by width
        relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index) # Not a parameter

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B_, num_heads, N, C//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2] # Each has shape (B_, num_heads, N, C//num_heads)
        # Scaled dot-product attention
        q = q *self.scale
        attn = (q @ k.transpose(-2, -1)) # (B_, num_heads, N, N)
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0) # (B_, num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # (B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x   

class WindowTransformerBlock(nn.Module):
    '''
    A Transformer block that applies window-based multi-head self-attention (W-MSA).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio to determine the hidden dimension in feedforward networks.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        dropout (float, optional): Dropout rate. Default: 0.1
        attn_dropout (float, optional): Attention dropout rate. Default: 0.1
        proj_dropout (float, optional): Projection dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.1
        fused_window_process (bool, optional): If True, process all windows in a batch together for efficiency. Default: False    
    '''
    def __init__(self, dim=128, num_heads=8, window_size=4, mlp_ratio=4.0, 
                 qkv_bias=True, dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, drop_path=0.1,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, 
                                    attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, C = x.shape
        Wh, Ww = self.window_size

        # Dynamically infer H, W from sequence length
        H = W = int(L ** 0.5)
        if H * W != L:
            raise ValueError(f"Non-square feature map: cannot reshape {L} tokens into H*W grid")

        # Update stored resolution to match runtime shape
        self.input_resolution = (H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Partition windows
        x = window_partition(x, Wh) # (num_windows*B, window_size*window_size, C)

        # W-MSA
        x = self.attn(x) # (num_windows*B, window_size*window_size, C)

        # Merge windows
        x = x.view(-1, Wh, Ww, C) # (num_windows*B, window_size, window_size, C)
        x = window_reverse(x, Wh, H, W) # (B, H, W, C)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.dropout(self.drop_path(self.mlp(self.norm2(x))))

        return x



class ViT_Hierarchical(nn.Module):
    '''
    Vision Transformer with hierarchical stages and window-based attention.
    Args:
        img_size (int): Input image size (assumed square).
        in_channels (int): Number of input image channels.
        stage_dims (list): List of embedding dimensions for each stage.
        layer_nums (list): List of number of transformer layers for each stage.
        head_nums (list): List of number of attention heads for each stage.
        window_size (list): List of window sizes for each stage (None for global attention).
        mlp_ratios (list): List of feedforward network expansion ratios for each stage.
        drop_path (float, optional): Stochastic depth rate. Default: 0.1
        attn_dropout (float, optional): Attention dropout rate. Default: 0.1
        proj_dropout (float, optional): Projection dropout rate. Default: 0.1
        dropout (float, optional): Dropout rate. Default: 0.1
    Returns:
        Tensor: (B, T, 128) where B is batch size, T is sequence length, and 128 is the final embedding dimension.
        '''
    def __init__(self,
                 img_size=128,
                 in_channels=3,
                 stage_dims=[64, 128, 256],
                 layer_nums=[2, 4, 6],
                 head_nums=[2, 4, 8],
                 window_size=[8, 4, None],   # None → global attention
                 mlp_ratio=[4, 4, 4],
                 drop_path=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 dropout=0.1):
        super().__init__()

        # --- helper ---
        def _to_list(x, n):
            return x if isinstance(x, (list, tuple)) else [x] * n

        num_stages = len(stage_dims)
        mlp_ratio    = _to_list(mlp_ratio, num_stages)
        window_size  = _to_list(window_size, num_stages)
        attn_dropout = _to_list(attn_dropout, num_stages)
        proj_dropout = _to_list(proj_dropout, num_stages)

        # progressive stochastic depth schedule
        total_blocks = sum(layer_nums)
        dpr = torch.linspace(0, drop_path, total_blocks).tolist()
        block_idx = 0

        # --- stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_dims[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(stage_dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        self.stage_types = []
        in_dim = stage_dims[0]

        # --- build stages ---
        for i, (dim, num_layers, num_heads, w_size, mlp_r, attn_dp, proj_dp) in \
            enumerate(zip(stage_dims, layer_nums, head_nums, window_size, mlp_ratio, attn_dropout, proj_dropout)):

            # downsample between stages
            down_sample = (
                nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False)
                if i != 0 else nn.Identity()
            )

            # build blocks
            blocks = []
            for j in range(num_layers):
                dp_rate = dpr[block_idx]
                block_idx += 1

                if i < len(stage_dims) - 1 and w_size is not None:
                    # window-based local transformer
                    blocks.append(WindowTransformerBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=w_size,
                        mlp_ratio=mlp_r,
                        dropout=dropout,
                        attn_dropout=attn_dp,
                        proj_dropout=proj_dp,
                        drop_path=dp_rate
                    ))
                else:
                    # global attention stage
                    blocks.append(TransformerEncoderBlock(
                        d_model=dim,
                        num_heads=num_heads,
                        dim_feedforward=int(dim * mlp_r),
                        dropout=dropout
                    ))

            # register stage
            self.stages.append(nn.ModuleDict({
                'down_sample': down_sample,
                'block': nn.ModuleList(blocks)
            }))
            self.stage_types.append('window' if i < len(stage_dims) - 1 else 'global')
            in_dim = dim

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.frame_proj = nn.Linear(stage_dims[-1], 128)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # [B*T, C, H, W]
#        print(f"Input shape: {x.shape}")
        x = self.stem(x)           # [B*T, D, H/4, W/4]
#        print(f"[Stem] -> {x.shape}")
        for stage_idx, (stage, block_type) in enumerate(zip(self.stages, self.stage_types)):
            x = stage['down_sample'](x) # Downsampling
            B_T, D, H_s, W_s = x.shape
#            print(f"[Downsample] -> ({B_T}, {D}, {H_s}, {W_s}), Block type: {block_type}")

            if block_type == 'window':
                for blk_idx, block in enumerate(stage['block']):
                    x_window = x.flatten(2).transpose(1, 2) # [B*T, H_s*W_s, D]
#                    print(f" Stage {stage_idx} | Block {blk_idx} | Window in: {x_window.shape}")
                    x_window = block(x_window)          # Window Transformer blocks
#                    print(f" Stage {stage_idx} | Block {blk_idx} | Window out: {x_window.shape}")
                    x = x_window.transpose(1, 2).view(B_T, D, H_s, W_s) # Reshape back to image-like
            else:
                tokens = x.flatten(2).transpose(1, 2) # [B*T, H_s*W_s, D]
#                print(f" Stage {stage_idx} | Global Block | Tokens in: {tokens.shape}")
                for blk_idx, block in enumerate(stage['block']):
                    tokens = block(tokens)                 # Transformer blocks
#                    print(f" Stage {stage_idx} | Block {blk_idx} | Tokens out: {tokens.shape}")
                x = tokens.transpose(1, 2).view(B_T, D, H_s, W_s) # Reshape back to image-like
#                print(f" Stage {stage_idx} | Global reshape | Tokens out: {tokens.shape}")
        
        x = x.mean([2, 3]) # Global average pooling (B*T, D)
#        print(f"[Global Avg Pool] -> {x.shape}")
        x = self.norm(x)
        x = x.view(B, T, -1) # [B, T, D]
        x = self.frame_proj(x) # Project to desired 128 dim for cross-attention
        return x

if __name__ == '__main__':
    # Test the ViT_Hierarchical module
    batch_size = 32
    seq_len = 8
    img_size = 128
    in_channels = 3
    x = torch.randn(batch_size, seq_len, in_channels, img_size, img_size) # Example input

    vit = ViT_Hierarchical(
        img_size=128,
        in_channels=3,
        stage_dims=[64, 128, 224],
        layer_nums=[2, 4, 5],
        head_nums=[2, 4, 7],
        window_size=[8, 4, None],
        mlp_ratio=[4, 4, 4],
        drop_path=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        dropout=0.1
    )
    out = vit(x)
    print("Output shape:", out.shape) # Expected: [batch_size, seq_len, 128]
    print ("Total parameters:", sum(p.numel() for p in vit.parameters() if p.requires_grad))