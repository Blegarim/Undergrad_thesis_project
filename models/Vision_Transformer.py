import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

class MLP(nn.Module):
    def __init__(self, dim=128, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)
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
        # preserve None to indicate a dynamic/global window; otherwise normalize to tuple
        if window_size is None:
            self.window_size = None
            Wh = Ww = None
        else:
            self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
            Wh, Ww = self.window_size
        self.num_heads = num_heads 
        self.scale = (dim // num_heads) ** -0.5 # Scaling factor for dot-product attention
        assert self.dim % self.num_heads == 0, "dim should be divisible by num_heads"

        #Prebuild relative position table for fixed windows; keep None if dynamic/global windows
        if self.window_size is not None:
            table_size = (2 * Wh - 1) * (2 * Ww - 1)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(table_size, num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            device = self.relative_position_bias_table.device
            coords_h = torch.arange(Wh, device=device)
            coords_w = torch.arange(Ww, device=device)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
            relative_coords[:, :, 0] += Wh - 1
            relative_coords[:, :, 1] += Ww - 1
            relative_coords[:, :, 0] *= 2 * Ww - 1
            relative_position_index = relative_coords.sum(-1).long().contiguous()  # N, N
            # register buffer (overwrites existing if present)
            self._buffers.pop("relative_position_index", None)
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.relative_position_bias_table = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def init_relative_position_bias(self, Wh, Ww):
        """
        Build or rebuild the relative position bias table and index for a window of size (Wh, Ww).
        Safe to call multiple times; if the current table already matches requested size this is a no-op.
        """
        if getattr(self, "window_size", None) == (Wh, Ww) and hasattr(self, "relative_position_index"):
            return
        
        # determine device for new parameter: prefer existing param device, else any param/buffer device, else cpu
        if hasattr(self, "relative_position_bias_table") and isinstance(self.relative_position_bias_table, nn.Parameter):
            device = self.relative_position_bias_table.device
        else:
            # fallback: check parameters, buffers, else cpu
            params = list(self.parameters())
            buffers = list(self.buffers())
            if params:
                device = params[0].device
            elif buffers:
                device = buffers[0].device
            else:
                device = torch.device("cpu")
        
        self.window_size = (Wh, Ww)
        table_size = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, self.num_heads, device=device)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std = 0.02)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(Wh, device=device)
        coords_w = torch.arange(Ww, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij')) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += Wh -1 # Shift to start from 0
        relative_coords[:, :, 1] += Ww -1 # Shift to start from 0
        relative_coords[:, :, 0] *= 2 * Ww - 1 # Multiply by width
        relative_position_index = relative_coords.sum(-1).long().contiguous() # Wh*Ww, Wh*Ww
        # register (or replace) buffer
        self._buffers.pop("relative_position_index", None)
        self.register_buffer("relative_position_index", relative_position_index) # Not a parameter

    def forward(self, x):
        B_, N, C = x.shape
        if self.relative_position_bias_table is None or not hasattr(self, "relative_position_index"):
            raise RuntimeError("WindowAttention: call init_relative_position_bias(Wh, Ww) before forward for dynamic/global window")
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
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio to determine the hidden dimension in feedforward networks.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        dropout (float, optional): Dropout rate. Default: 0.1
        attn_dropout (float, optional): Attention dropout rate. Default: 0.1
        proj_dropout (float, optional): Projection dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.1   
    '''
    def __init__(self, dim=128, num_heads=8, window_size=4, mlp_ratio=4.0, 
                 qkv_bias=True, dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, drop_path=0.1,
                 ):
        super().__init__()
        self.dim = dim
        if window_size is None:
            self.window_size = None
        else:
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
    
    def init_relative_position_bias(self, H, W):
        """
        Forwarder so ViT can request a block to (re)initialize its attention's relative position bias
        for a runtime HxW feature map.
        """
        # delegate to the WindowAttention instance
        Wh, Ww = (H, W) if self.window_size is None else self.window_size
        self.attn.init_relative_position_bias(Wh, Ww)

    def forward(self, x):
        '''
        x: [B, C, H, W] - spatial feature map
        Returns: [B, C, H, W]
        '''
        B, C, H, W = x.shape

        # Infer window sizes
        if self.window_size is None:
            Wh, Ww = H, W
        else:
            Wh, Ww = self.window_size if isinstance(self.window_size, tuple) else (self.window_size, self.window_size)
        
        # Validate window tiling
        if (H % Wh != 0) or (W % Ww != 0):
            raise ValueError(f"Feature map ({H}x{W}) not divisible by window ({Wh}x{Ww})")
        
        # Permute to [B, H, W, C] for spatial operations
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        shortcut = x_perm

        # First residual block: attention
        x_perm = self.norm1(x_perm)
        x_windows = window_partition(x_perm, Wh)  # [num_windows*B, Wh*Ww, C]
        x_windows = self.attn(x_windows)
        x_perm = x_windows.view(-1, Wh, Ww, C)
        x_perm = window_reverse(x_perm, Wh, H, W)  # [B, H, W, C]
        x_perm = shortcut + self.drop_path(x_perm)

        # Second residual block: MLP
        x_flat = x_perm.view(B, H * W, C)  # [B, H*W, C]
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = self.dropout(x_flat)
        x_flat = x_perm.view(B, H * W, C) + self.drop_path(x_flat)
        x_perm = x_flat.view(B, H, W, C)

        # Permute back to [B, C, H, W]
        return x_perm.permute(0, 3, 1, 2)

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
                 in_channels=3,
                 stage_dims=[64, 128, 256],
                 layer_nums=[2, 4, 6],
                 head_nums=[2, 4, 8],
                 window_size=[8, 4, "global"],   # global window for last layer
                 mlp_ratio=[4, 4, 4],
                 d_model=128,
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
            actual_w_size = None if w_size == "global" else w_size
            blocks = []
            for j in range(num_layers):
                dp_rate = dpr[block_idx]
                block_idx += 1

                blocks.append(WindowTransformerBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=actual_w_size,
                        mlp_ratio=mlp_r,
                        dropout=dropout,
                        attn_dropout=attn_dp,
                        proj_dropout=proj_dp,
                        drop_path=dp_rate
                    ))

            # register stage
            self.stages.append(nn.ModuleDict({
                'down_sample': down_sample,
                'block': nn.ModuleList(blocks)
            }))
            in_dim = dim

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.frame_proj = nn.Linear(stage_dims[-1], d_model) if stage_dims[-1] != d_model else nn.Identity()

    def forward(self, x):
        '''
        x: Context cropped images, tensor of shape [B, T, C, Hc, Wc]
        Returns:
            Tensor of shape [B, T, 128]
        '''
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # [B*T, C, H, W]
#        print(f"Input shape: {x.shape}")
        x = self.stem(x)           # [B*T, D, H/4, W/4]
#        print(f"[Stem] -> {x.shape}")
        for stage_idx, stage in enumerate(self.stages):
            x = stage['down_sample'](x)
            B_T, D, H_s, W_s = x.shape

            for blk_idx, block in enumerate(stage['block']):
                
                # Ensure the block's relative-position tables are initialized for runtime H_s,W_s.
            # Use the block API instead of mutating internals.
                if getattr(block, "window_size", None) is None:
                    block.init_relative_position_bias(H_s, W_s)

                x = block(x)  # x: [B*T, D, H_s, W_s] -> same shape
        
        x = x.mean([2, 3]) # Global average pooling (B*T, D)
#        print(f"[Global Avg Pool] -> {x.shape}")
        x = self.norm(x)
        x = x.view(B, T, -1) # [B, T, D]
        x = self.frame_proj(x) # Project to desired 128 dim for cross-attention
        return x

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    # Test the ViT_Hierarchical module
    batch_size = 1
    seq_len = 10
    img_size = 480
    in_channels = 3
    x = torch.randn(batch_size, seq_len, in_channels, img_size, img_size) # Example input

    vit = ViT_Hierarchical(
        in_channels=3,
        stage_dims=[48, 96, 168, 96],
        layer_nums=[2, 4, 5, 7],
        head_nums=[2, 4, 7, 4],
        window_size=[8, 4, 2, None],
        mlp_ratio=[4, 4, 4, 4],
        d_model=224,
        drop_path=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        dropout=0.1
    )

    out = vit(x)
    flops = FlopCountAnalysis(vit, x)
    print("Total parameters:", sum(p.numel() for p in vit.parameters() if p.requires_grad))

    #graph = draw_graph(vit, input_size=(batch_size, seq_len, in_channels, img_size, img_size))
    #graph.visual_graph.render("vit_hierarchical", format="plain", cleanup=True)