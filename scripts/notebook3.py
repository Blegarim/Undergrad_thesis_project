'''Input: motion tensor [B, T, 3]  ← (cx, cy, dt)

1. Linear Projection (input_dim=3 → d_model)
    - Linear layer to project motion features to d_model
    - Shape: [B, T, d_model]
    - d_model = 128 (or any other dimension)

2. Add Positional Encoding (sinusoidal or learned)
    - Shape: [B, T, d_model]
    - Positional encoding helps the model understand the order of the sequence
    - Can be learned or sinusoidal (e.g., using torch.nn.Embedding)

3. Dropout (optional)
    - Dropout layer to prevent overfitting
    - Shape: [B, T, d_model]
    - Dropout rate: 0.1 (or any other value)

4. N x TransformerEncoderBlock(s), each block has:
    - MultiHeadAttention (Self-attention)
    - LayerNorm + Residual
    - FeedForward (MLP)
    - LayerNorm + Residual
    - Dropout (optional)
    - Shape: [B, T, d_model]
    - MultiHeadAttention: [B, T, d_model] → [B, T, d_model]
    - FeedForward: [B, T, d_model] → [B, T, d_model]
    - LayerNorm: [B, T, d_model] → [B, T, d_model]
    - Residual connections: [B, T, d_model] + [B, T, d_model] → [B, T, d_model]
    - Dropout: [B, T, d_model] → [B, T, d_model]


5. Output Aggregation (choose one):
    a. Use last time step
    b. Mean pooling over time
    c. [CLS] token (if you prepend one)

→ Final vector [B, d_model] → MLP Head for prediction
'''