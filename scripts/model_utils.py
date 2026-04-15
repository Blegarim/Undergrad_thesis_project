"""
Shared model utilities for training and evaluation.
Provides model factory and forward pass functions for ablation studies.
"""

import torch
import torch.nn as nn

from models.AblationModels import MotionOnlyModel, VisualOnlyModel, VanillaConcatModel
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel


def get_model(model_type, motion_enc, vit, d_model, num_classes_dict, dropout=0.1):
    """
    Get specified model for ablation study.
    
    Args:
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', or 'full'
        motion_enc: MotionEncoder instance
        vit: ViT_Hierarchical instance  
        d_model: model dimension
        num_classes_dict: dictionary of class counts
        dropout: dropout rate (default: 0.1)
    
    Returns:
        Model instance
    """
    if model_type == 'motion_only':
        return MotionOnlyModel(
            motion_enc=motion_enc,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=dropout
        )
    elif model_type == 'visual_only':
        return VisualOnlyModel(
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=dropout
        )
    elif model_type == 'vanilla_concat':
        return VanillaConcatModel(
            motion_enc=motion_enc,
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=dropout
        )
    elif model_type == 'full':
        cross_attention = CrossAttentionModule(
            d_model=d_model,
            num_heads=4,
            num_classes_dict=num_classes_dict,
            use_frame_crosses=True,
            frame_pool="logsumexp",
        )
        return EnsembleModel(
            motion_enc=motion_enc,
            vit=vit,
            cross_attention=cross_attention,
            d_model=d_model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: 'motion_only', 'visual_only', 'vanilla_concat', 'full'")


def model_forward(model, model_type, images_tight, images_context, motions):
    """
    Forward pass wrapper for different model types.
    
    Args:
        model: model instance
        model_type: 'motion_only', 'visual_only', 'vanilla_concat', 'full'
        images_tight: [B, T, C, H, W]
        images_context: [B, T, C, H, W] 
        motions: [B, T, motion_dim]
    
    Returns:
        logits dict
    """
    if model_type == 'motion_only':
        logits = model(motions, images_tight)
    elif model_type == 'visual_only':
        logits = model(images_context)
    else:
        logits = model(images_tight, images_context, motions)
    
    return logits
