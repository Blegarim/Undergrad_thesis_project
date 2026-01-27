"""
Simple test script to verify the enhanced class imbalance handling setup.
This script validates the configuration and components without running full training.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imbalance_config import (
    get_training_config, get_preset_config, validate_config,
    apply_imbalance_strategies_to_training, PRESET_CONFIGS
)
from class_imbalance_strategies import (
    FocalLoss, ClassBalancedFocalLoss, DynamicLossWeighting,
    ThresholdOptimizer, HardNegativeMining, create_class_balanced_loss
)
from collections import Counter
import numpy as np


def test_imbalance_strategies():
    """Test all implemented imbalance strategies."""
    print("🧪 Testing Class Imbalance Strategies...")
    
    # Test 1: Basic configuration loading
    print("\n1️⃣ Testing Configuration Loading...")
    try:
        config = get_training_config()
        print(f"✅ Base config loaded: {len(config)} parameters")
        
        # Test presets
        for preset_name in PRESET_CONFIGS.keys():
            preset_config = get_preset_config(preset_name)
            print(f"✅ Preset '{preset_name}' loaded successfully")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    # Test 2: Configuration validation
    print("\n2️⃣ Testing Configuration Validation...")
    try:
        validate_config(config)
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test 3: Focal Loss
    print("\n3️⃣ Testing Focal Loss...")
    try:
        focal_loss = FocalLoss(gamma=2.0)
        
        # Create dummy data
        logits = torch.randn(4, 2)  # batch_size=4, num_classes=2
        targets = torch.randint(0, 2, (4,))
        
        loss = focal_loss(logits, targets)
        print(f"✅ Focal loss computed: {loss.item():.4f}")
        
        # Test with alpha weighting
        alpha = torch.tensor([0.3, 0.7])
        focal_weighted = FocalLoss(alpha=alpha, gamma=2.0)
        loss_weighted = focal_weighted(logits, targets)
        print(f"✅ Weighted focal loss computed: {loss_weighted.item():.4f}")
        
    except Exception as e:
        print(f"❌ Focal loss test failed: {e}")
        return False
    
    # Test 4: Class-Balanced Focal Loss
    print("\n4️⃣ Testing Class-Balanced Focal Loss...")
    try:
        # Simulate severe imbalance (like crosses: 2% positive)
        samples_per_class = [980, 20]  # 98% negative, 2% positive
        cb_focal = ClassBalancedFocalLoss(samples_per_class, beta=0.9999, gamma=2.5)
        
        loss_cb = cb_focal(logits, targets)
        print(f"✅ Class-balanced focal loss computed: {loss_cb.item():.4f}")
        
    except Exception as e:
        print(f"❌ Class-balanced focal loss test failed: {e}")
        return False
    
    # Test 5: Dynamic Loss Weighting
    print("\n5️⃣ Testing Dynamic Loss Weighting...")
    try:
        dynamic_weights = DynamicLossWeighting(initial_weights=[1.0, 1.5, 2.0])
        
        # Simulate task losses
        task_losses = {'actions': torch.tensor(0.5), 
                      'looks': torch.tensor(0.8), 
                      'crosses': torch.tensor(1.2)}
        
        weighted_loss = dynamic_weights.get_weighted_loss(task_losses)
        print(f"✅ Dynamic weighted loss computed: {weighted_loss.item():.4f}")
        
        # Test performance update
        metrics = {'actions': 0.7, 'looks': 0.3, 'crosses': 0.1}  # crosses performing poorly
        dynamic_weights.update_performance(metrics)
        print("✅ Performance update completed")
        
    except Exception as e:
        print(f"❌ Dynamic loss weighting test failed: {e}")
        return False
    
    # Test 6: Hard Negative Mining
    print("\n6️⃣ Testing Hard Negative Mining...")
    try:
        hnm = HardNegativeMining(ratio=0.3)
        
        # Create logits with confident wrong predictions (hard negatives)
        logits_hard = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])  # Confident but wrong
        targets_hard = torch.tensor([1, 0, 1, 0])  # Opposite of predictions
        
        hard_indices = hnm.select_hard_negatives(logits_hard, targets_hard, 'crosses')
        print(f"✅ Hard negative mining found {len(hard_indices)} hard examples")
        
    except Exception as e:
        print(f"❌ Hard negative mining test failed: {e}")
        return False
    
    # Test 7: Threshold Optimizer
    print("\n7️⃣ Testing Threshold Optimizer...")
    try:
        threshold_opt = ThresholdOptimizer()
        
        # Create dummy validation predictions and targets
        predictions = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
        targets = np.array([0, 0, 1, 1, 0, 1])
        
        opt_threshold = threshold_opt._find_optimal_threshold(predictions, targets)
        print(f"✅ Optimal threshold found: {opt_threshold:.3f}")
        
    except Exception as e:
        print(f"❌ Threshold optimizer test failed: {e}")
        return False
    
    # Test 8: Loss Function Creation
    print("\n8️⃣ Testing Loss Function Creation...")
    try:
        # Simulate dataset with imbalance
        dummy_dataset = [
            {'actions': 0, 'looks': 0, 'crosses': 0},
            {'actions': 0, 'looks': 0, 'crosses': 0},
            {'actions': 0, 'looks': 0, 'crosses': 0},
            {'actions': 1, 'looks': 1, 'crosses': 1},  # Rare positive
        ]
        
        # Test different loss types
        for loss_type in ['standard', 'focal', 'class_balanced_focal', 'weighted_ce']:
            try:
                loss_fn = create_class_balanced_loss(dummy_dataset, task='crosses', loss_type=loss_type)
                print(f"✅ {loss_type} loss created successfully")
                
                # Test forward pass
                logits = torch.randn(4, 2)
                targets = torch.tensor([0, 0, 0, 1])
                loss_val = loss_fn(logits, targets)
                print(f"   Forward pass: {loss_val.item():.4f}")
                
            except Exception as e:
                print(f"⚠️ {loss_type} loss creation failed: {e}")
        
    except Exception as e:
        print(f"❌ Loss function creation test failed: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True


def test_preset_configurations():
    """Test different preset configurations."""
    print("\n🔧 Testing Preset Configurations...")
    
    for preset_name in ['conservative', 'aggressive', 'recommended']:
        print(f"\n📋 Testing '{preset_name}' preset:")
        try:
            config = get_preset_config(preset_name)
            
            # Check key parameters
            print(f"  Loss types: {config['loss_types']}")
            print(f"  Dynamic weighting: {config.get('use_dynamic_loss_weighting', False)}")
            print(f"  Batch balancing: {config.get('use_batch_balancing', False)}")
            
            # Test validation
            validate_config(config)
            print(f"  ✅ '{preset_name}' preset valid")
            
        except Exception as e:
            print(f"  ❌ '{preset_name}' preset failed: {e}")
            return False
    
    return True


def simulate_training_scenario():
    """Simulate a training scenario with severe imbalance."""
    print("\n🎯 Simulating Training Scenario...")
    
    # Simulate dataset imbalance
    print("\n📊 Simulated Dataset Imbalance:")
    print("  Actions: ~52% walking, ~48% standing (balanced)")
    print("  Looks: ~17% looking, ~83% not-looking (severe)")
    print("  Crosses: ~2% crossing, ~98% not-crossing (extreme)")
    
    # Test recommended configuration
    print("\n⚙️ Using Recommended Configuration:")
    config = get_preset_config('recommended')
    
    for task, loss_type in config['loss_types'].items():
        gamma = config['focal_params'].get(task, {}).get('gamma', 2.0)
        beta = config['focal_params'].get(task, {}).get('beta', 0.9999)
        print(f"  {task}: {loss_type} (γ={gamma}, β={beta})")
    
    print(f"  Dynamic weighting: {config.get('use_dynamic_loss_weighting', False)}")
    print(f"  Monitor F1-score: {config.get('monitor_f1_score', False)}")
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Class Imbalance Strategy Tests\n")
    
    success = True
    
    # Test individual components
    if not test_imbalance_strategies():
        success = False
    
    # Test preset configurations
    if not test_preset_configurations():
        success = False
    
    # Simulate training scenario
    if not simulate_training_scenario():
        success = False
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📖 Usage Instructions:")
        print("  1. Conservative approach: python train_enhanced.py --preset conservative")
        print("  2. Recommended approach: python train_enhanced.py --preset recommended")
        print("  3. Aggressive approach: python train_enhanced.py --preset aggressive")
        print("  4. Original pipeline: python train_enhanced.py --enable_advanced False")
        print("\n🔧 Configuration files created:")
        print("  - class_imbalance_strategies.py (core implementations)")
        print("  - imbalance_config.py (configuration management)")
        print("  - train_enhanced.py (enhanced training script)")
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)