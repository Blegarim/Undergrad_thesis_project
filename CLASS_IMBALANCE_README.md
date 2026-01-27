# Class Imbalance Handling Implementation

## Overview

This implementation provides advanced strategies for handling severe class imbalance in the pedestrian behavior prediction task. The original dataset has the following imbalance:

- **Actions**: ~52% walking, ~48% standing (relatively balanced)
- **Looks**: ~17% looking, ~83% not-looking (severe imbalance)
- **Crosses**: ~2% crossing, ~98% not-crossing (extreme imbalance)

## Files Created

### 1. `class_imbalance_strategies.py`
Core implementations of advanced imbalance handling techniques:

- **FocalLoss**: Addresses class imbalance by focusing on hard-to-classify examples
- **ClassBalancedFocalLoss**: Combines focal loss with class-balanced weighting
- **DynamicLossWeighting**: Automatically adjusts task weights based on performance
- **BatchBalancedSampler**: Ensures balanced representation in mini-batches
- **HardNegativeMining**: Focuses training on difficult examples
- **ThresholdOptimizer**: Optimizes decision thresholds for each task

### 2. `imbalance_config.py`
Configuration management system with preset options:

- **Conservative**: Basic weighted loss, minimal changes
- **Recommended**: Class-balanced focal loss for imbalanced tasks, dynamic weighting
- **Aggressive**: Maximum imbalance handling with all techniques enabled

### 3. `train_enhanced.py`
Enhanced training script that integrates all strategies safely:
- Maintains full backward compatibility with original `train.py`
- Toggleable advanced strategies via command-line arguments
- Enhanced metrics (F1, precision, recall) beyond just accuracy
- Better logging and monitoring for imbalanced tasks

### 4. `test_imbalance_setup.py`
Comprehensive test suite for validating all components.

## Usage

### Basic Usage (Recommended)
```bash
python train_enhanced.py --preset recommended --enable_advanced True
```

### Conservative Approach
```bash
python train_enhanced.py --preset conservative --enable_advanced True
```

### Aggressive Approach
```bash
python train_enhanced.py --preset aggressive --enable_advanced True
```

### Original Pipeline (No Advanced Strategies)
```bash
python train_enhanced.py --enable_advanced False
```

### Original Training Script (Unchanged)
```bash
python train.py  # Original pipeline unchanged
```

## Configuration Details

### Recommended Preset Configuration
```python
{
    'loss_types': {
        'actions': 'weighted_ce',           # Balanced, use weighted CE
        'looks': 'class_balanced_focal',    # Severe imbalance
        'crosses': 'class_balanced_focal',  # Extreme imbalance
    },
    'focal_params': {
        'looks': {'gamma': 2.0, 'beta': 0.9999},
        'crosses': {'gamma': 2.5, 'beta': 0.99999},
    },
    'use_dynamic_loss_weighting': True,
    'use_batch_balancing': False,           # Can be enabled if needed
    'monitor_f1_score': True,              # Better metric than accuracy
}
```

### Conservative Preset Configuration
```python
{
    'loss_types': {
        'actions': 'weighted_ce',
        'looks': 'weighted_ce',
        'crosses': 'weighted_ce',
    },
    'use_dynamic_loss_weighting': False,
    'use_batch_balancing': False,
}
```

### Aggressive Preset Configuration
```python
{
    'loss_types': {
        'actions': 'focal',
        'looks': 'class_balanced_focal',
        'crosses': 'class_balanced_focal',
    },
    'focal_params': {
        'looks': {'gamma': 3.0, 'beta': 0.9999},
        'crosses': {'gamma': 3.5, 'beta': 0.99999},
    },
    'use_dynamic_loss_weighting': True,
    'use_batch_balancing': True,
    'use_hard_negative_mining': True,
}
```

## Key Features

### 1. **Safe Integration**
- Zero modifications to existing `train.py`
- All new functionality is additive and toggleable
- Maintains original WeightedRandomSampler behavior

### 2. **Task-Specific Handling**
- Different strategies for different imbalance levels
- Higher gamma values for more severe imbalance
- Task-specific loss function selection

### 3. **Enhanced Metrics**
- F1-score monitoring (more informative than accuracy)
- Precision and recall tracking
- Macro F1 for overall performance assessment

### 4. **Dynamic Adaptation**
- Automatic loss weight adjustment based on task performance
- Higher weights for underperforming tasks
- Responsive to training dynamics

### 5. **Multiple Strategies**
- **Loss-level**: Focal loss, class-balanced loss, weighted CE
- **Sampling-level**: Batch balancing, weighted sampling
- **Training-level**: Hard negative mining, threshold optimization
- **Optimization-level**: Dynamic weighting, task-specific learning rates

## Expected Improvements

Based on the severe imbalance analysis, we expect:

### For Looks Task (9:1 imbalance):
- **F1-score**: From ~0.1 → 0.3-0.5
- **Precision**: From ~0.05 → 0.2-0.4
- **Recall**: From ~0.2 → 0.4-0.6

### For Crosses Task (50:1 imbalance):
- **F1-score**: From ~0.2 → 0.4-0.6
- **Precision**: From ~0.1 → 0.3-0.5
- **Recall**: From ~0.4 → 0.5-0.7

### For Actions Task (balanced):
- Minimal change or slight improvement
- More stable training dynamics

## Implementation Details

### Focal Loss Formula
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- `γ` (gamma): Focusing parameter (2.0-3.5)
- `α_t` (alpha): Class balancing weight

### Class-Balanced Weighting
```
Effective number = (1 - β^n) / (1 - β)
CB weight = (1 - β) / (1 - β^n)
```
- `β` (beta): Hyperparameter controlling class balance (0.9999-0.99999)
- `n`: Number of samples in class

### Dynamic Weight Adjustment
- Monitor task performance every epoch
- Increase weight for tasks not improving after `patience` epochs
- Adaptive rebalancing based on F1-score trends

## Validation and Testing

Run the test suite to validate implementation:
```bash
python test_imbalance_setup.py
```

The test suite validates:
- Configuration loading and validation
- All loss function implementations
- Dynamic weighting mechanisms
- Hard negative mining
- Threshold optimization
- Preset configurations

## Backward Compatibility

The implementation is fully backward compatible:
- Original `train.py` remains unchanged
- All new functionality is opt-in
- Same data loading and preprocessing pipeline
- Same model architecture and training loop structure

## Monitoring and Debugging

Enhanced training logs include:
- Task-specific F1, precision, recall
- Dynamic weight evolution
- Loss component breakdown
- Early stopping based on meaningful metrics

## Future Extensions

The framework supports easy addition of:
- Additional loss functions (LDAM loss, GHM loss)
- More sophisticated sampling strategies
- Ensemble methods for imbalance
- Curriculum learning approaches

---

**Note**: This implementation is designed to address the specific severe imbalance in your pedestrian behavior prediction dataset while maintaining the integrity and stability of the existing training pipeline.