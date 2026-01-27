"""
Simple test to verify our class imbalance configuration logic.
This test doesn't require torch - just basic Python to validate setup.
"""

import os
import sys

def test_configuration_loading():
    """Test if our configuration files load correctly."""
    print("Testing Configuration Loading...")
    
    try:
        # Test basic imports
        from imbalance_config import get_training_config, get_preset_config
        print("[OK] Configuration modules imported successfully")
        
        # Test base config
        base_config = get_training_config()
        print(f"[OK] Base config loaded with {len(base_config)} parameters")
        
        # Test presets
        for preset in ['conservative', 'recommended', 'aggressive']:
            config = get_preset_config(preset)
            print(f"[OK] Preset '{preset}' loaded successfully")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files are present."""
    print("\nTesting File Structure...")
    
    required_files = [
        'class_imbalance_strategies.py',
        'imbalance_config.py', 
        'train_enhanced.py',
        'train.py',
        'CLASS_IMBALANCE_README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} exists")
        else:
            print(f"[ERROR] {file} missing")
            return False
    
    return True

def test_configuration_logic():
    """Test configuration logic without torch."""
    print("\nTesting Configuration Logic...")
    
    try:
        from imbalance_config import get_preset_config
        
        # Test recommended preset
        config = get_preset_config('recommended')
        
        # Check key parameters
        expected_keys = [
            'loss_types', 'focal_params', 'use_dynamic_loss_weighting',
            'monitor_f1_score', 'embedding_dim', 'learning_rate'
        ]
        
        for key in expected_keys:
            if key in config:
                print(f"[OK] Key '{key}' present")
            else:
                print(f"[ERROR] Key '{key}' missing")
                return False
        
        # Test loss types
        loss_types = config['loss_types']
        print(f"[OK] Loss types configured: {loss_types}")
        
        # Validate loss type values
        valid_loss_types = ['standard', 'focal', 'class_balanced_focal', 'weighted_ce']
        for task, loss_type in loss_types.items():
            if loss_type in valid_loss_types:
                print(f"[OK] {task}: {loss_type}")
            else:
                print(f"[ERROR] {task}: Invalid loss type {loss_type}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration logic test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Simple Setup Validation\n")
    
    success = True
    
    if not test_file_structure():
        success = False
    
    if not test_configuration_loading():
        success = False
        
    if not test_configuration_logic():
        success = False
    
    if success:
        print("\nAll basic tests passed!")
        print("\nNext Steps:")
        print("1. Run: python train_enhanced.py --preset recommended --enable_advanced True")
        print("2. Monitor training logs for F1-score improvements")
        print("3. Compare with original training.py performance")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)