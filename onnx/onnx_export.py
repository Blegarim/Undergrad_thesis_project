import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import onnx
from models.Unified_Module import EnsembleModel
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model

class EnsembleModelONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images_tight, images_context, motions):
        out = self.model(images_tight, images_context, motions)

        return out["actions"], out["looks"], out["crosses_frame"]
    
def main():
    device = torch.device("cpu")
    embedding_dim = get_unified_dim_model()
    num_classes_dict = {
        "actions": 2,
        "looks": 2,
        "crosses": 2
    }

    motion_enc = MotionEncoder(**motion_enc_args_config())
    vit = ViT_Hierarchical(**vit_args_config())

    from scripts.model_utils import get_model
    base_model = get_model("full", motion_enc, vit, embedding_dim, num_classes_dict)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_dict = torch.load(os.path.join(project_root, "model_outputs/best_model_epoch28_0122_1511.pth"), map_location="cpu")
    base_model.load_state_dict(state_dict)
    base_model.eval()

    onnx_model = EnsembleModelONNX(base_model)

    # Dummy inputs — must match your real data shapes exactly
    T = 20
    H, W = 128, 128
    dummy_tight    = torch.randn(1, T, 3, H, W)
    dummy_context  = torch.randn(1, T, 3, H*3, W*3)
    dummy_motions  = torch.randn(1, T, 8)

    torch.onnx.export(
        onnx_model,
        (dummy_tight, dummy_context, dummy_motions),
        os.path.join(project_root, "onnx", "pedestrian_model.onnx"),
        input_names=["images_tight", "images_context", "motions"],
        output_names=["actions", "looks", "crosses_frame"],
        dynamic_axes={          # allow variable batch size
            "images_tight":    {0: "batch"},
            "images_context":  {0: "batch"},
            "motions":         {0: "batch"},
            "actions":         {0: "batch"},
            "looks":           {0: "batch"},
            "crosses_frame":  {0: "batch"},
        },
        opset_version=17
    )
    out_path = os.path.join(project_root, "onnx", "pedestrian_model.onnx")
    print(f"Exported to {out_path}")

if __name__ == "__main__":
    main()
