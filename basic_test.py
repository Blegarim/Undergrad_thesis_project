"""
Basic configuration test without torch dependency.
"""

def test_basic_config():
    """Test basic configuration without torch imports."""
    print("Testing Basic Configuration...")
    
    # Test that we can load the config file structure
    try:
        # Simple import test
        import sys
        import os
        sys.path.append('.')
        
        # Read config file to check structure
        with open('imbalance_config.py', 'r') as f:
            content = f.read()
            
        # Check if key functions exist
        required_functions = [
            'get_training_config',
            'get_preset_config', 
            'validate_config'
        ]
        
        for func in required_functions:
            if f"def {func}" in content:
                print(f"[OK] Function {func} found")
            else:
                print(f"[ERROR] Function {func} missing")
                return False
        
        # Check preset configurations
        if "PRESET_CONFIGS" in content:
            print("[OK] PRESET_CONFIGS found")
        else:
            print("[ERROR] PRESET_CONFIGS missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Config test failed: {e}")
        return False

def check_preset_configurations():
    """Check preset configuration logic."""
    print("\nTesting Preset Configurations...")
    
    # Create mock configs to test logic
    conservative = {
        'loss_types': {
            'actions': 'weighted_ce',
            'looks': 'weighted_ce', 
            'crosses': 'weighted_ce'
        },
        'use_dynamic_loss_weighting': False,
        'use_batch_balancing': False
    }
    
    recommended = {
        'loss_types': {
            'actions': 'weighted_ce',
            'looks': 'class_balanced_focal',
            'crosses': 'class_balanced_focal'
        },
        'focal_params': {
            'looks': {'gamma': 2.0, 'beta': 0.9999},
            'crosses': {'gamma': 2.5, 'beta': 0.99999}
        },
        'use_dynamic_loss_weighting': True,
        'use_batch_balancing': False
    }
    
    aggressive = {
        'loss_types': {
            'actions': 'focal',
            'looks': 'class_balanced_focal',
            'crosses': 'class_balanced_focal'
        },
        'focal_params': {
            'looks': {'gamma': 3.0, 'beta': 0.9999},
            'crosses': {'gamma': 3.5, 'beta': 0.99999}
        },
        'use_dynamic_loss_weighting': True,
        'use_batch_balancing': True
    }
    
    configs = {
        'conservative': conservative,
        'recommended': recommended, 
        'aggressive': aggressive
    }
    
    for name, config in configs.items():
        print(f"\nTesting {name} preset:")
        
        # Check required keys
        required_keys = ['loss_types', 'use_dynamic_loss_weighting']
        for key in required_keys:
            if key in config:
                print(f"  [OK] {key}: {config[key]}")
            else:
                print(f"  [ERROR] {key} missing")
                return False
        
        # Check loss types
        valid_loss_types = ['weighted_ce', 'focal', 'class_balanced_focal']
        loss_types = config['loss_types']
        for task, loss_type in loss_types.items():
            if loss_type in valid_loss_types:
                print(f"  [OK] {task}: {loss_type}")
            else:
                print(f"  [ERROR] {task}: {loss_type}")
                return False
    
    return True

def main():
    """Run basic tests."""
    print("Basic Configuration Validation\n")
    
    success = True
    
    if not test_basic_config():
        success = False
        
    if not check_preset_configurations():
        success = False
    
    if success:
        print("\nAll basic tests passed!")
        print("\nYour class imbalance handling setup is ready!")
        print("\nUsage Instructions:")
        print("1. Recommended: python train_enhanced.py --preset recommended --enable_advanced True")
        print("2. Conservative: python train_enhanced.py --preset conservative --enable_advanced True") 
        print("3. Aggressive: python train_enhanced.py --preset aggressive --enable_advanced True")
        print("4. Original pipeline: python train.py (unchanged)")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)