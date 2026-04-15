"""
Training configuration with advanced class imbalance handling options.
This file extends the existing config.py with toggleable imbalance strategies.
"""

from class_imbalance_strategies import (
    FocalLoss, ClassBalancedFocalLoss, BatchBalancedSampler,
    DynamicLossWeighting, ThresholdOptimizer, create_class_balanced_loss,
    get_imbalance_config
)

def get_training_config():
    """
    Main training configuration with imbalance handling options.
    These settings can be safely toggled without affecting the core pipeline.
    """
    
    # Basic training settings (from original)
    base_config = {
        'embedding_dim': 128,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'num_epochs': 30,
        'num_workers': 2,
        
        # Original loss weights
        'loss_weight': {'actions': 0.8, 'looks': 0.8, 'crosses': 1.2},
        
        # Original sampler settings
        'use_weighted_sampler': True,
        'sampler_powers': {"crosses": 1, "actions": 0.5, "looks": 0.5},
    }
    
    # Advanced imbalance handling (toggleable)
    imbalance_config = {
        # Enable/disable advanced strategies
        'use_advanced_imbalance_handling': True,
        
        # Loss function choices per task: 'standard', 'focal', 'class_balanced_focal', 'weighted_ce'
        'loss_types': {
            'actions': 'standard',  # Balanced, use standard CE
            'looks': 'class_balanced_focal',  # Severe imbalance
            'crosses': 'class_balanced_focal',  # Extreme imbalance
        },
        
        # Focal loss parameters
        'focal_params': {
            'looks': {'gamma': 2.0, 'beta': 0.9999},
            'crosses': {'gamma': 2.5, 'beta': 0.99999},
        },
        
        # Batch balancing options
        'use_batch_balancing': False,  # Can be enabled for severe cases
        'batch_balance_ratios': {
            'actions': 0.5,  # 50% positive
            'looks': 0.3,   # 30% positive (upweight minority)
            'crosses': 0.2, # 20% positive (upweight minority)
        },
        
        # Dynamic loss weighting
        'use_dynamic_loss_weighting': True,
        'dynamic_weighting_patience': 3,
        
        # Hard negative mining
        'use_hard_negative_mining': False,  # Can be enabled for difficult cases
        'hard_negative_ratio': 0.3,
        
        # Threshold optimization
        'use_threshold_optimization': False,  # For inference, not training
        
        # Validation metrics for imbalance
        'monitor_f1_score': True,  # Monitor F1 instead of just accuracy
        'early_stopping_metric': 'f1_macro',  # Use F1 for early stopping
        
        # Learning rate adjustments for imbalanced tasks
        'use_task_specific_lr': False,
        'task_lr_multipliers': {
            'actions': 1.0,
            'looks': 1.2,  # Slightly higher LR for difficult task
            'crosses': 1.5,  # Higher LR for most difficult task
        }
    }
    
    return {**base_config, **imbalance_config}


def create_loss_functions(config, datasets):
    """
    Create appropriate loss functions based on configuration.
    This can be safely used in the existing training pipeline.
    """
    loss_functions = {}
    
    for task in ['actions', 'looks', 'crosses']:
        loss_type = config['loss_types'].get(task, 'standard')
        
        if loss_type == 'standard':
            import torch.nn as nn
            loss_functions[task] = nn.CrossEntropyLoss()
            
        elif loss_type in ['focal', 'class_balanced_focal', 'weighted_ce']:
            # Use the dataset to compute class weights
            dataset = datasets.get(task, datasets.get('all', []))
            loss_functions[task] = create_class_balanced_loss(
                dataset, task=task, loss_type=loss_type
            )
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions


def create_advanced_components(config, dataset=None):
    """
    Create advanced imbalance handling components.
    Returns dictionary of components that can be used in training.
    """
    components = {}
    
    # Dynamic loss weighting
    if config.get('use_dynamic_loss_weighting', False):
        initial_weights = list(config['loss_weight'].values())
        components['dynamic_weighting'] = DynamicLossWeighting(
            initial_weights=initial_weights,
            patience=config.get('dynamic_weighting_patience', 3)
        )
    
    # Batch balancing sampler
    if config.get('use_batch_balancing', False):
        if dataset is not None:
            components['batch_sampler'] = BatchBalancedSampler(
                dataset=dataset,
                batch_size=config['batch_size'],
                target_balance=config['batch_balance_ratios']
            )
    
    # Threshold optimizer
    if config.get('use_threshold_optimization', False):
        components['threshold_optimizer'] = ThresholdOptimizer()
    
    return components


def apply_imbalance_strategies_to_training(model, config, datasets=None):
    """
    Apply all configured imbalance strategies to the training pipeline.
    This function integrates with the existing training code safely.
    """
    
    # Create loss functions
    loss_functions = create_loss_functions(config, datasets)
    
    # Create advanced components
    advanced_components = create_advanced_components(config, datasets.get('all', []))
    
    # Setup for monitoring
    setup_imbalance_monitoring(config)
    
    return {
        'loss_functions': loss_functions,
        'advanced_components': advanced_components,
        'config': config
    }


def setup_imbalance_monitoring(config):
    """
    Setup additional monitoring for imbalanced training.
    """
    if config.get('monitor_f1_score', False):
        print("Enabling F1-score monitoring for imbalanced tasks")
        
    if config.get('use_dynamic_loss_weighting', False):
        print("Enabling dynamic loss weighting")
        
    if config.get('loss_types')['looks'] in ['focal', 'class_balanced_focal']:
        print("Using focal loss for severely imbalanced 'looks' task")
        
    if config.get('loss_types')['crosses'] in ['focal', 'class_balanced_focal']:
        print("Using class-balanced focal loss for extremely imbalanced 'crosses' task")


def get_recommended_config():
    """
    Get recommended configuration based on the dataset imbalance analysis.
    This provides a safe starting point for training.
    """
    
    config = get_training_config()
    
    # Adjust based on severity of imbalance
    config['loss_types'] = {
        'actions': 'weighted_ce',  # Slightly balanced
        'looks': 'class_balanced_focal',  # Severe (17% positive)
        'crosses': 'class_balanced_focal',  # Extreme (2% positive)
    }
    
    config['focal_params'] = {
        'looks': {'gamma': 2.0, 'beta': 0.9999},
        'crosses': {'gamma': 2.5, 'beta': 0.99999},
    }
    
    # Enable dynamic weighting for difficult tasks
    config['use_dynamic_loss_weighting'] = True
    
    # Keep batch balancing off initially (can be enabled if needed)
    config['use_batch_balancing'] = False
    
    # Monitor appropriate metrics
    config['monitor_f1_score'] = True
    config['early_stopping_metric'] = 'f1_macro'
    
    return config


def validate_config(config):
    """
    Validate the configuration before training.
    """
    required_keys = [
        'embedding_dim', 'learning_rate', 'batch_size', 'num_epochs',
        'loss_types', 'focal_params', 'loss_weight'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate loss types
    valid_loss_types = ['standard', 'focal', 'class_balanced_focal', 'weighted_ce']
    for task, loss_type in config['loss_types'].items():
        if loss_type not in valid_loss_types:
            raise ValueError(f"Invalid loss type for {task}: {loss_type}")
    
    # Validate focal params
    for task in ['looks', 'crosses']:
        if task in config['focal_params']:
            required_params = ['gamma', 'beta']
            for param in required_params:
                if param not in config['focal_params'][task]:
                    raise ValueError(f"Missing focal param {param} for {task}")
    
    print("✅ Configuration validation passed")
    return True


# Preset configurations for easy experimentation
PRESET_CONFIGS = {
    'conservative': {
        'loss_types': {'actions': 'weighted_ce', 'looks': 'weighted_ce', 'crosses': 'weighted_ce'},
        'use_dynamic_loss_weighting': False,
        'use_batch_balancing': False,
    },
    
    'aggressive': {
        'loss_types': {'actions': 'focal', 'looks': 'class_balanced_focal', 'crosses': 'class_balanced_focal'},
        'focal_params': {
            'looks': {'gamma': 3.0, 'beta': 0.9999},
            'crosses': {'gamma': 3.5, 'beta': 0.99999},
        },
        'use_dynamic_loss_weighting': True,
        'use_batch_balancing': True,
    },
    
    'recommended': get_recommended_config(),
}


def get_preset_config(preset_name):
    """
    Get a preset configuration by name.
    Available presets: 'conservative', 'aggressive', 'recommended'
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    base_config = get_training_config()
    preset_config = PRESET_CONFIGS[preset_name]
    
    # Merge with base config
    return {**base_config, **preset_config}