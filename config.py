def get_unified_dim_model():
    # Ensure consistent d_model across all modules for proper cross-attention
    d_model = 128
    return int(d_model)

def vit_args_config(in_channels=3,
                stage_dims=[36, 36, 288, 36],
                layer_nums=[2, 4, 5, 7],
                head_nums=[2, 2, 16, 2],
                window_size=[8, 4, 2, None],
                mlp_ratio=[4, 4, 4, 4],
                drop_path=0.15,
                attn_dropout=0.15,
                proj_dropout=0.15,
                dropout=0.15):
    d_model = get_unified_dim_model()
    return {
            "in_channels": in_channels,
            "stage_dims": stage_dims,
            "layer_nums": layer_nums,
            "head_nums": head_nums,
            "window_size": window_size,
            "d_model": d_model,
            "mlp_ratio": mlp_ratio,
            "drop_path": drop_path,
            "attn_dropout": attn_dropout,
            "proj_dropout": proj_dropout,
            "dropout": dropout}

def motion_enc_args_config(motion_dim=8,
                        hidden_dim=168,
                        num_layers=2,
                        num_heads=8, 
                        dropout=0.3):
    d_model = get_unified_dim_model()
    return {"motion_dim": motion_dim,
            "hidden_dim": hidden_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout}


