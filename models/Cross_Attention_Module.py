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

        self.pool = nn.AdaptiveAvgPool1d(1) # Pooling layer to aggregate features

        # Classifier: MLP head for prediction
        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            ) for name, num_classes in num_classes_dict.items()   
        })

    def forward(self, motion_feats, image_feats):
        """
        motion_feats: Tensor of shape [batch_size, seq_len, d_model]
        image_feats: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Cross Attention
        attn_output, _ = self.cross_attn(
            query = motion_feats,  # Shape: [batch_size, seq_len, d_model]
            key = image_feats,   # Shape: [batch_size, seq_len, d_model]
            value = image_feats, # Shape: [batch_size, seq_len, d_model]
        ) # Shape: [batch_size, seq_len, d_model]

        # Pooling
        pooled_output = self.pool(attn_output.transpose(1, 2)).squeeze(-1)

        logits = {key: head(pooled_output) for key, head in self.classifier.items()}
        return logits

if __name__ == "__main__":
    B, T, D = 4, 10, 128
    img_feats = torch.randn(B, T, D)
    motion_feats = torch.randn(B, T, D)

    model = CrossAttentionModule(d_model=D, num_heads=4)
    out = model(img_feats, motion_feats)
    print("Output:", {k: v.shape for k, v in out.items()})  # [4, 2]


