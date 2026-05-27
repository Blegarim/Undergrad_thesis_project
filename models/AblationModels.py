import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionOnlyModel(nn.Module):
    """Motion encoder only model for ablation study."""
    
    def __init__(self, motion_enc, d_model=128, num_classes_dict=None, dropout=0.1):
        super().__init__()
        self.motion_enc = motion_enc
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal pooling MLP
        self.pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Classification heads matching CrossAttentionModule output format
        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
            for name, num_classes in num_classes_dict.items()
        })
        
        # Frame-level crosses head
        self.crosses_frame_head = nn.Linear(d_model, num_classes_dict["crosses"])
        
    def forward(self, motions, images_tight, frame_pool="logsumexp"):
        """Forward pass with motion features only."""
        motion_feats = self.motion_enc(motions, images_tight)
        motion_feats = self.norm(motion_feats)  # [B, T, D]
        
        # Temporal pooling
        scores = self.pool_mlp(motion_feats)    # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (motion_feats * weights).sum(dim=1)  # [B, D]
        
        logits = {}
        
        # Action and looks classification from pooled features
        for key, head in self.classifier.items():
            if key != "crosses":
                logits[key] = head(pooled)
        
        frame_logits = self.crosses_frame_head(motion_feats)  # [B, T, C]
        
        # Frame pooling for crosses
        if frame_pool == "logsumexp":
            frame_crosses = torch.logsumexp(frame_logits, dim=1)
        elif frame_pool == "max":
            frame_crosses = frame_logits.max(dim=1).values
        elif frame_pool == "mean":
            frame_crosses = frame_logits.mean(dim=1)
        else:
            raise ValueError(f"Unsupported frame_pool: {frame_pool}")
        
        logits["crosses_frame"] = frame_crosses
        
        return logits


class VisualOnlyModel(nn.Module):
    """Visual encoder only model for ablation study."""
    
    def __init__(self, vit, d_model=128, num_classes_dict=None, dropout=0.1):
        super().__init__()
        self.vit = vit
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal pooling MLP
        self.pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Classification heads matching CrossAttentionModule output format
        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
            for name, num_classes in num_classes_dict.items()
        })
        
        # Frame-level crosses head
        self.crosses_frame_head = nn.Linear(d_model, num_classes_dict["crosses"])
        
    def forward(self, images_context, frame_pool="logsumexp"):
        """Forward pass with visual features only."""
        image_feats = self.vit(images_context)  # [B, T, D]
        image_feats = self.norm(image_feats)
        
        # Temporal pooling
        scores = self.pool_mlp(image_feats)     # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (image_feats * weights).sum(dim=1)  # [B, D]
        
        logits = {}
        
        # Action and looks classification from pooled features
        for key, head in self.classifier.items():
            if key != "crosses":
                logits[key] = head(pooled)
        
        frame_logits = self.crosses_frame_head(image_feats)  # [B, T, C]
        
        # Frame pooling for crosses
        if frame_pool == "logsumexp":
            frame_crosses = torch.logsumexp(frame_logits, dim=1)
        elif frame_pool == "max":
            frame_crosses = frame_logits.max(dim=1).values
        elif frame_pool == "mean":
            frame_crosses = frame_logits.mean(dim=1)
        else:
            raise ValueError(f"Unsupported frame_pool: {frame_pool}")
        
        logits["crosses_frame"] = frame_crosses
        
        return logits


class VanillaConcatModel(nn.Module):
    """Vanilla concatenation model without cross-attention for ablation study."""
    
    def __init__(self, motion_enc, vit, d_model=128, num_classes_dict=None, dropout=0.1):
        super().__init__()
        self.motion_enc = motion_enc
        self.vit = vit
        self.d_model = d_model
        
        # Normalization for both modalities
        self.motion_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        
        # Fusion layer after concatenation
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Temporal pooling MLP
        self.pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Classification heads matching CrossAttentionModule output format
        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
            for name, num_classes in num_classes_dict.items()
        })
        
        # Frame-level crosses head
        self.crosses_frame_head = nn.Linear(d_model, num_classes_dict["crosses"])
        
    def forward(self, images_tight, images_context, motions, frame_pool="logsumexp"):
        """Forward pass with vanilla concatenation."""
        # Extract features from both encoders
        image_feats = self.vit(images_context)        # [B, T, D]
        image_feats = self.visual_norm(image_feats)
        
        motion_out = self.motion_enc(motions, images_tight)  # [B, T, D]
        motion_feats = self.motion_norm(motion_out)
        
        # Vanilla concatenation and fusion
        concatenated = torch.cat([motion_feats, image_feats], dim=-1)  # [B, T, 2*D]
        fused_feats = self.fusion(concatenated)  # [B, T, D]
        
        # Temporal pooling
        scores = self.pool_mlp(fused_feats)      # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (fused_feats * weights).sum(dim=1)  # [B, D]
        
        logits = {}
        
        # Action and looks classification from pooled features
        for key, head in self.classifier.items():
            if key != "crosses":
                logits[key] = head(pooled)
        
        frame_logits = self.crosses_frame_head(fused_feats)  # [B, T, C]
        
        # Frame pooling for crosses
        if frame_pool == "logsumexp":
            frame_crosses = torch.logsumexp(frame_logits, dim=1)
        elif frame_pool == "max":
            frame_crosses = frame_logits.max(dim=1).values
        elif frame_pool == "mean":
            frame_crosses = frame_logits.mean(dim=1)
        else:
            raise ValueError(f"Unsupported frame_pool: {frame_pool}")
        
        logits["crosses_frame"] = frame_crosses
        
        return logits