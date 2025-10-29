def vit_args_config(img_size=160,
            in_channels=3,
            stage_dims=[48, 96, 168],
            layer_nums=[2, 4, 5],
            head_nums=[2, 4, 7],
            window_size=[8, 4, None],
            mlp_ratio=[4, 4, 4],
            drop_path=0.15,
            attn_dropout=0.15,
            proj_dropout=0.15,
            dropout=0.15):
    
    return {"img_size": img_size,
            "in_channels": in_channels,
            "stage_dims": stage_dims,
            "layer_nums": layer_nums,
            "head_nums": head_nums,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "drop_path": drop_path,
            "attn_dropout": attn_dropout,
            "proj_dropout": proj_dropout,
            "dropout": dropout}

def motion_enc_args_config(input_dim=4, 
                           tcn_channels=(128, 256), 
                           d_model=128, 
                           num_layers=2, 
                           kernel_size=3, 
                           num_heads=8, 
                           dropout=0.1):
    
    return {"input_dim": input_dim,
            "tcn_channels": tcn_channels,
            "d_model": d_model,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "num_heads": num_heads,
            "dropout": dropout}

num_class_dict = {}