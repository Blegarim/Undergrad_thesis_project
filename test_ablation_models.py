import torch
import torch.nn as nn
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
from models.AblationModels import MotionOnlyModel, VisualOnlyModel, VanillaConcatModel
from config import vit_args_config, motion_enc_args_config, get_unified_dim_model

def test_ablation_models():
    """Phantom test for all ablation models to ensure they work correctly."""
    
    print("Testing Ablation Models...")
    
    # Test configuration
    batch_size = 2
    seq_len = 10
    img_size_tight = 224
    img_size_context = 480
    motion_dim = 8
    
    # Number of classes
    num_classes_dict = {
        'actions': 2,
        'looks': 2,
        'crosses': 2
    }
    
    # Create dummy input data
    images_tight = torch.randn(batch_size, seq_len, 3, img_size_tight, img_size_tight)
    images_context = torch.randn(batch_size, seq_len, 3, img_size_context, img_size_context)
    motions = torch.randn(batch_size, seq_len, motion_dim)
    
    print(f"Input shapes:")
    print(f"  Images tight: {images_tight.shape}")
    print(f"  Images context: {images_context.shape}")
    print(f"  Motions: {motions.shape}")
    
    # Get model configurations
    d_model = get_unified_dim_model()
    vit_args = vit_args_config()
    motion_enc_args = motion_enc_args_config()
    
    print(f"\nModel configuration:")
    print(f"  d_model: {d_model}")
    print(f"  ViT d_model: {vit_args['d_model']}")
    print(f"  Motion encoder d_model: {motion_enc_args['d_model']}")
    
    # Initialize base components
    vit = ViT_Hierarchical(**vit_args)
    motion_enc = MotionEncoder(**motion_enc_args)
    cross_attention = CrossAttentionModule(
        d_model=d_model,
        num_heads=4,
        num_classes_dict=num_classes_dict,
        use_frame_crosses=True,
        frame_pool="logsumexp",
    )
    
    # Test original EnsembleModel for reference
    print("\n=== Testing Original EnsembleModel ===")
    try:
        original_model = EnsembleModel(
            motion_enc=motion_enc,
            vit=vit,
            cross_attention=cross_attention,
            d_model=d_model
        )
        
        with torch.no_grad():
            original_output = original_model(images_tight, images_context, motions)
            print(f"✓ Original model output keys: {list(original_output.keys())}")
            for key, value in original_output.items():
                print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"✗ Original model failed: {e}")
        return False
    
    # Test 1: Motion Only Model
    print("\n=== Testing Motion Only Model ===")
    try:
        motion_only_model = MotionOnlyModel(
            motion_enc=motion_enc,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
        
        # We need to get motion features from the motion encoder first
        with torch.no_grad():
            motion_feats = motion_enc(motions, images_tight)  # [B, T, D]
            motion_output = motion_only_model(motion_feats)
            print(f"✓ Motion only model output keys: {list(motion_output.keys())}")
            for key, value in motion_output.items():
                print(f"  {key}: {value.shape}")
                
            # Verify output shapes match original
            for key in motion_output.keys():
                if key in original_output.keys():
                    if motion_output[key].shape != original_output[key].shape:
                        print(f"✗ Shape mismatch for {key}: {motion_output[key].shape} vs {original_output[key].shape}")
                        return False
                    
    except Exception as e:
        print(f"✗ Motion only model failed: {e}")
        return False
    
    # Test 2: Visual Only Model
    print("\n=== Testing Visual Only Model ===")
    try:
        visual_only_model = VisualOnlyModel(
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
        
        with torch.no_grad():
            visual_output = visual_only_model(images_context)
            print(f"✓ Visual only model output keys: {list(visual_output.keys())}")
            for key, value in visual_output.items():
                print(f"  {key}: {value.shape}")
                
            # Verify output shapes match original
            for key in visual_output.keys():
                if key in original_output.keys():
                    if visual_output[key].shape != original_output[key].shape:
                        print(f"✗ Shape mismatch for {key}: {visual_output[key].shape} vs {original_output[key].shape}")
                        return False
                        
    except Exception as e:
        print(f"✗ Visual only model failed: {e}")
        return False
    
    # Test 3: Vanilla Concatenation Model
    print("\n=== Testing Vanilla Concatenation Model ===")
    try:
        vanilla_model = VanillaConcatModel(
            motion_enc=motion_enc,
            vit=vit,
            d_model=d_model,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
        
        with torch.no_grad():
            vanilla_output = vanilla_model(images_tight, images_context, motions)
            print(f"✓ Vanilla concat model output keys: {list(vanilla_output.keys())}")
            for key, value in vanilla_output.items():
                print(f"  {key}: {value.shape}")
                
            # Verify output shapes match original
            for key in vanilla_output.keys():
                if key in original_output.keys():
                    if vanilla_output[key].shape != original_output[key].shape:
                        print(f"✗ Shape mismatch for {key}: {vanilla_output[key].shape} vs {original_output[key].shape}")
                        return False
                        
    except Exception as e:
        print(f"✗ Vanilla concat model failed: {e}")
        return False
    
    # Test frame pooling variants
    print("\n=== Testing Frame Pooling Variants ===")
    for pool_type in ["logsumexp", "max", "mean"]:
        try:
            vanilla_output_variant = vanilla_model(images_tight, images_context, motions, frame_pool=pool_type)
            print(f"✓ {pool_type} pooling: crosses_frame shape {vanilla_output_variant['crosses_frame'].shape}")
        except Exception as e:
            print(f"✗ {pool_type} pooling failed: {e}")
            return False
    
    print("\n🎉 All ablation models passed phantom tests!")
    return True

if __name__ == "__main__":
    success = test_ablation_models()
    if success:
        print("\n✅ All tests passed. Ablation models are ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the models.")