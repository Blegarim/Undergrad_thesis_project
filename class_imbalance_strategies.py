"""
Advanced class imbalance handling strategies for pedestrian behavior prediction.
This module provides additional techniques that can be safely toggled during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score as sklearn_f1_score
from collections import Counter
import random


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance.
    Focuses on hard-to-classify examples and reduces weight of easy majority class examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
            else:
                alpha_t = 1.0
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss that accounts for both class imbalance and sample difficulty.
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        
        # Handle edge cases for class counts
        samples_per_class = np.array(samples_per_class, dtype=np.float32)
        
        # Ensure no zero counts (avoid division by zero)
        samples_per_class = np.maximum(samples_per_class, 1.0)
        
        # Calculate effective number of samples
        self.effective_num = 1.0 - np.power(self.beta, samples_per_class)
        
        # Calculate class weights with safeguards
        denominator = np.array(self.effective_num)
        denominator = np.maximum(denominator, 1e-8)  # Avoid division by zero
        
        self.class_weights = (1.0 - self.beta) / denominator
        
        # Normalize weights
        weight_sum = np.sum(self.class_weights)
        if weight_sum > 0:
            self.class_weights = self.class_weights / weight_sum * len(self.class_weights)
        else:
            self.class_weights = np.ones_like(self.class_weights)
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class-balanced weights
        class_weights = torch.tensor(self.class_weights, dtype=torch.float, device=inputs.device)
        weights = class_weights.gather(0, targets)
        weighted_loss = weights * focal_loss
        
        return weighted_loss.mean()


class BatchBalancedSampler:
    """
    Ensures each mini-batch has balanced representation of classes.
    Particularly useful for extremely imbalanced datasets.
    """
    def __init__(self, dataset, batch_size, target_balance=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_balance = target_balance or {'actions': 0.5, 'looks': 0.5, 'crosses': 0.5}
        
        # Group indices by label combinations
        self._group_indices()
        
    def _group_indices(self):
        """Group dataset indices by their label combinations."""
        self.groups = {
            'actions': {0: [], 1: []},
            'looks': {0: [], 1: []},
            'crosses': {0: [], 1: []}
        }
        
        for idx, item in enumerate(self.dataset):
            for task in ['actions', 'looks', 'crosses']:
                label = int(item[task])
                self.groups[task][label].append(idx)
                
    def _sample_balanced_batch(self, task):
        """Sample a batch with balanced representation for a specific task."""
        batch_indices = []
        target_ratio = self.target_balance[task]
        
        pos_count = int(self.batch_size * target_ratio)
        neg_count = self.batch_size - pos_count
        
        # Sample positive and negative indices
        pos_indices = random.sample(
            self.groups[task][1], 
            min(pos_count, len(self.groups[task][1]))
        )
        neg_indices = random.sample(
            self.groups[task][0], 
            min(neg_count, len(self.groups[task][0]))
        )
        
        # If not enough samples, fill with available ones
        if len(pos_indices) < pos_count:
            remaining = pos_count - len(pos_indices)
            pos_indices.extend(random.choices(self.groups[task][1], k=remaining))
        if len(neg_indices) < neg_count:
            remaining = neg_count - len(neg_indices)
            neg_indices.extend(random.choices(self.groups[task][0], k=remaining))
            
        return pos_indices + neg_indices
    
    def __iter__(self):
        """Iterate over balanced batches."""
        # Create balanced batches for each task and cycle through them
        task_batches = []
        for task in ['actions', 'looks', 'crosses']:
            num_batches = len(self.dataset) // self.batch_size
            for _ in range(num_batches):
                batch_indices = self._sample_balanced_batch(task)
                task_batches.append(batch_indices)
                
        if self.shuffle:
            random.shuffle(task_batches)
            
        for batch_indices in task_batches:
            yield batch_indices
            
    def __len__(self):
        return len(self.dataset) // self.batch_size


class DynamicLossWeighting(nn.Module):
    """
    Dynamically adjusts loss weights based on task performance and class imbalance.
    """
    TASK_INDEX = {'actions': 0, 'looks': 1, 'crosses': 2}

    def __init__(self, initial_weights=None, adaptive=True, patience=3):
        super(DynamicLossWeighting, self).__init__()
        self.adaptive = adaptive
        self.patience = patience

        if initial_weights is None:
            self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        else:
            self.weights = nn.Parameter(torch.tensor(initial_weights))

        self.performance_history = {'actions': [], 'looks': [], 'crosses': []}
        self.last_improvement = {'actions': 0, 'looks': 0, 'crosses': 0}
        
        # Device management - will be set to device later
        self._device = None
        
    def update_performance(self, metrics):
        """Update performance history and adjust weights if needed."""
        if not self.adaptive:
            return
            
        for task, metric in metrics.items():
            if task in self.performance_history:
                self.performance_history[task].append(metric)
                
                # Check if performance improved
                if len(self.performance_history[task]) > 1:
                    if metric > self.performance_history[task][-2]:
                        self.last_improvement[task] = 0
                    else:
                        self.last_improvement[task] += 1
                        
                # Increase weight for tasks that aren't improving
                if self.last_improvement[task] > self.patience:
                    idx = self.TASK_INDEX[task]
                    with torch.no_grad():
                        self.weights[idx] *= 1.1
                    self.last_improvement[task] = 0
                    
    def to(self, *args, **kwargs):
        """Move dynamic weighting to specified device."""
        result = super().to(*args, **kwargs)
        # Extract device from args/kwargs
        if args:
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']
        else:
            device = None
        
        if device is not None:
            self._device = device
        return result
    
    def get_weighted_loss(self, losses):
        """Apply dynamic weights to individual task losses."""
        weighted_loss = 0.0
        for task, loss in losses.items():
            weight = self.weights[self.TASK_INDEX[task]]
            if hasattr(loss, 'device'):
                weight = weight.to(loss.device)
            elif self._device is not None:
                weight = weight.to(self._device)
            weighted_loss += weight * loss
        return weighted_loss


class HardNegativeMining:
    """
    Focuses training on hard-to-classify examples, particularly useful for minority classes.
    """
    def __init__(self, ratio=0.3, min_examples=5):
        self.ratio = ratio
        self.min_examples = min_examples
        
    def select_hard_negatives(self, logits, targets, task):
        """Select hard negative examples for focused training."""
        batch_size = logits.size(0)
        
        # Get predictions and confidence scores
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        # Find misclassified examples
        misclassified = (predictions != targets)
        
        # Among misclassified, find those with high confidence (hard negatives)
        hard_mask = misclassified & (probs.max(dim=1)[0] > 0.7)
        
        # Ensure minimum number of examples
        if hard_mask.sum() < self.min_examples:
            # Take the most confident misclassified examples
            misclassified_probs = probs[misclassified].max(dim=1)[0]
            _, top_indices = torch.topk(misclassified_probs, min(self.min_examples, misclassified.sum()))
            hard_indices = torch.where(misclassified)[0][top_indices]
        else:
            hard_indices = torch.where(hard_mask)[0]
            
        return hard_indices


class ThresholdOptimizer:
    """
    Optimizes decision thresholds for each task based on validation performance.
    Particularly useful for imbalanced classification tasks.
    """
    def __init__(self, task_names=['actions', 'looks', 'crosses']):
        self.task_names = task_names
        self.thresholds = {task: 0.5 for task in task_names}
        
    def optimize_thresholds(self, model, val_loader, device, task_names=None):
        """Find optimal thresholds for each task using validation data."""
        task_names = task_names or self.task_names
        model.eval()
        
        # Collect all predictions and targets
        all_predictions = {task: [] for task in task_names}
        all_targets = {task: [] for task in task_names}
        
        with torch.no_grad():
            for batch in val_loader:
                images_tight, images_context, motions, labels = batch
                images_tight = images_tight.to(device)
                images_context = images_context.to(device)
                motions = motions.to(device)
                
                outputs = model(images_tight, images_context, motions)
                
                for task in task_names:
                    if task == 'crosses':
                        logits = outputs['crosses_frame']
                    else:
                        logits = outputs[task]
                        
                    probs = F.softmax(logits, dim=1)[:, 1]  # Probability of positive class
                    all_predictions[task].extend(probs.cpu().numpy())
                    all_targets[task].extend(labels[task].numpy())
        
        # Find optimal threshold for each task
        for task in task_names:
            if len(set(all_targets[task])) > 1:  # Only if both classes present
                threshold_value = self._find_optimal_threshold(
                    all_predictions[task], all_targets[task]
                )
                self.thresholds[task] = float(threshold_value)  # Ensure float type
                
        return self.thresholds
    
    def _find_optimal_threshold(self, predictions, targets):
        """Find threshold that maximizes F1 score."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0.0

        targets = np.array(targets)
        for threshold in thresholds:
            pred_binary = (np.array(predictions) > threshold).astype(int)
            f1 = sklearn_f1_score(targets, pred_binary, average='binary', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold


def create_class_balanced_loss(dataset, task='crosses', loss_type='focal'):
    """
    Create a loss function that accounts for class imbalance.
    """
    # Handle empty or invalid dataset
    if not dataset:
        print(f"Warning: Empty dataset for task {task}, using standard CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    
    # Count class frequencies
    try:
        labels = [int(item[task]) for item in dataset]
    except (KeyError, TypeError, ValueError) as e:
        print(f"Warning: Cannot extract labels for task {task} ({e}), using standard CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    
    class_counts = Counter(labels)
    
    # Ensure we have both classes or handle gracefully
    if len(class_counts) < 2:
        print(f"Warning: Only {len(class_counts)} class(es) found for task {task}: {dict(class_counts)}")
        if len(class_counts) == 1:
            only_class = list(class_counts.keys())[0]
            print(f"Using synthetic opposite class for balanced loss calculation")
            class_counts[1 - only_class] = 1  # Add synthetic opposite class
    
    if loss_type == 'focal':
        # Use inverse class frequencies as alpha for focal loss
        total = sum(class_counts.values())
        alpha = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        alpha_tensor = torch.tensor([alpha[0], alpha[1]], dtype=torch.float32)
        return FocalLoss(alpha=alpha_tensor, gamma=2.0)
    
    elif loss_type == 'class_balanced_focal':
        samples_per_class = [class_counts[0], class_counts[1]]
        return ClassBalancedFocalLoss(samples_per_class, beta=0.9999, gamma=2.0)
    
    elif loss_type == 'weighted_ce':
        # Standard weighted cross entropy with robust handling
        try:
            # Check if we have both classes
            unique_classes = np.unique(labels)
            if len(unique_classes) < 2:
                print(f"Warning: Only one class found ({unique_classes}), using equal weights")
                class_weights = np.array([1.0, 1.0])
            else:
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=np.array(labels)
                )
                
                # Ensure we have weights for both classes (0 and 1)
                if len(class_weights) == 1:
                    if 0 in unique_classes:
                        class_weights = np.array([class_weights[0], 1.0])
                    else:
                        class_weights = np.array([1.0, class_weights[0]])
                elif len(class_weights) == 2:
                    pass  # Good case
                else:
                    print(f"Warning: Unexpected number of classes ({len(class_weights)}), using equal weights")
                    class_weights = np.array([1.0, 1.0])
            
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=class_weights)
            
        except Exception as e:
            print(f"Warning: compute_class_weight failed ({e}), using inverse frequency weighting")
            # Fallback to manual calculation
            class_counts = Counter(labels)
            total = sum(class_counts.values())
            n_classes = len(class_counts)
            
            if len(class_counts) == 1:
                # Only one class present
                weights = [1.0, 1.0]
            else:
                weights = []
                for cls in [0, 1]:  # Ensure binary order
                    count = class_counts.get(cls, 0)
                    if count == 0:
                        weights.append(1.0)  # Avoid division by zero
                    else:
                        weight = total / (n_classes * count)
                        weights.append(weight)
            
            class_weights = torch.tensor(weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=class_weights)
    
    else:
        return nn.CrossEntropyLoss()


# Utility functions for easy integration
def get_imbalance_config():
    """
    Returns a configuration dictionary with recommended settings for different imbalance scenarios.
    """
    return {
        'looks_severe': {
            'loss_type': 'class_balanced_focal',
            'gamma': 2.0,
            'beta': 0.9999,
            'batch_balance': 0.3,  # 30% positive in each batch
            'dynamic_weighting': True,
            'hard_negative_mining': True
        },
        'crosses_extreme': {
            'loss_type': 'class_balanced_focal',
            'gamma': 2.5,
            'beta': 0.99999,
            'batch_balance': 0.2,  # 20% positive in each batch
            'dynamic_weighting': True,
            'hard_negative_mining': True,
            'threshold_optimization': True
        },
        'actions_balanced': {
            'loss_type': 'weighted_ce',
            'batch_balance': 0.5,
            'dynamic_weighting': False,
            'hard_negative_mining': False
        }
    }