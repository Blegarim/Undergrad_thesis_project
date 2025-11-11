import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    # Configuration
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    d_model = get_unified_dim_model()
    num_classes_dict = {"actions": 2, "looks": 2, "crosses": 2}
    
    # Initialize model components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = ViT_Hierarchical(**vit_args)
    motion_enc = MotionEncoder(**motion_enc_args)
    cross_attention = CrossAttentionModule(d_model=d_model, num_classes_dict=num_classes_dict, num_heads=8, dropout=0.3)
    model = EnsembleModel(motion_enc, vit, cross_attention, d_model=d_model).to(device)

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    print("Per-module parameter breakdown:")
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        train_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<20} total={mod_params:,} | trainable={train_params:,}")

    B, T, C, H, W = 1, 20, 3, 128, 128   # batch, sequence, channels, height, width
    M = 8                                 # motion feature dim

    images_tight = torch.randn(B, T, C, H, W, device=device)
    images_context = torch.randn(B, T, C, H, W, device=device)
    motions = torch.randn(B, T, M, device=device)

    with torch.no_grad():
        outputs = model(images_tight, images_context, motions)

    print("\nModel output heads:")
    for head, out in outputs.items():
        print(f"  {head:<10} -> {tuple(out.shape)}")

if __name__ == "__main__":
    main()