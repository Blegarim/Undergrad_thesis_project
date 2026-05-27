import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        num_classes_dict=None,
        dropout=0.1,
        use_frame_crosses=True,    # kept for config / logging
        frame_pool="logsumexp",
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.use_frame_crosses = use_frame_crosses
        self.frame_pool = frame_pool

        self.pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
            for name, num_classes in num_classes_dict.items()
        })

        self.crosses_frame_head = nn.Linear(
            d_model, num_classes_dict["crosses"]
        )

    def forward(self, motion_feats, image_feats, key_padding_mask=None):
        attn_output, _ = self.cross_attn(
            query=motion_feats,
            key=image_feats,
            value=image_feats,
            key_padding_mask=key_padding_mask
        )  # [B, T, D]

        scores = self.pool_mlp(attn_output)          # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (attn_output * weights).sum(dim=1) # [B, D]

        logits = {}

        for key, head in self.classifier.items():
            if key != "crosses":
                logits[key] = head(pooled)

        if self.use_frame_crosses:
            frame_logits = self.crosses_frame_head(attn_output)  # [B, T, C]

            if self.frame_pool == "logsumexp":
                logits["crosses_frame"] = torch.logsumexp(frame_logits, dim=1)
            elif self.frame_pool == "max":
                logits["crosses_frame"] = frame_logits.max(dim=1).values
            elif self.frame_pool == "mean":
                logits["crosses_frame"] = frame_logits.mean(dim=1)
            else:
                raise ValueError(f"Unsupported frame_pool: {self.frame_pool}")

        logits["temporal_weights"] = weights.squeeze(-1)  # [B, T]

        return logits





