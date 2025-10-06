import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, d_model=128, num_heads=8, num_classes_dict=None, dropout=0.1):
        super().__init__()
        # Cross Attention: Query from motion, Key and Value from CNN features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )

        self.pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Classifier: MLP head for prediction
        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            ) for name, num_classes in num_classes_dict.items()   
        })

    def forward(self, motion_feats, image_feats, key_padding_mask=None):
        """
        motion_feats: Tensor of shape [batch_size, seq_len, d_model]
        image_feats: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Cross Attention
        attn_output, _ = self.cross_attn(
            query = motion_feats,  # Shape: [batch_size, seq_len, d_model]
            key = image_feats,   # Shape: [batch_size, seq_len, d_model]
            value = image_feats, # Shape: [batch_size, seq_len, d_model]
            key_padding_mask = key_padding_mask
        ) # Shape: [batch_size, seq_len, d_model]

        # Pooling: Weighted average pooling using learned weights
        scores = self.pool_mlp(attn_output) # Shape: [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1) # Shape: [batch_size, seq_len, 1]

        pooled = (attn_output * weights).sum(dim=1) # Shape: [batch_size, d_model]

        logits = {key: head(pooled) for key, head in self.classifier.items()}
        return logits




