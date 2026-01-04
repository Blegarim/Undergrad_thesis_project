import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        num_classes_dict=None,
        dropout=0.1,
        use_frame_crosses=False,
        frame_pool="logsumexp",
    ):
        super().__init__()
        # Cross Attention: Query from motion, Key and Value from CNN features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )

        self.use_frame_crosses = use_frame_crosses
        self.frame_pool = frame_pool

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

        self.crosses_frame_head = None
        if self.use_frame_crosses and num_classes_dict and "crosses" in num_classes_dict:
            self.crosses_frame_head = nn.Linear(d_model, num_classes_dict["crosses"])

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

        logits = {}
        for key, head in self.classifier.items():
            if key == "crosses" and self.use_frame_crosses:
                continue
            logits[key] = head(pooled)

        if self.use_frame_crosses and self.crosses_frame_head is not None:
            frame_logits = self.crosses_frame_head(attn_output)  # [B, T, C]
            if self.frame_pool == "logsumexp":
                crosses_logits = torch.logsumexp(frame_logits, dim=1)
            elif self.frame_pool == "max":
                crosses_logits = frame_logits.max(dim=1).values
            elif self.frame_pool == "mean":
                crosses_logits = frame_logits.mean(dim=1)
            else:
                raise ValueError(f"Unsupported frame_pool: {self.frame_pool}")
            logits["crosses"] = crosses_logits
        return logits




