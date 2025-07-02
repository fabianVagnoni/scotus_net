#!/usr/bin/env python3
"""
Loss functions for SCOTUS Voting Model.

This module provides various loss functions to handle dataset imbalance and 
different training objectives, including custom implementations like Focal Loss
with gamma annealing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base class for all loss functions."""
    
    def __init__(self, **kwargs):
        """Initialize loss function with configuration parameters."""
        pass
    
    @abstractmethod
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions tensor
            targets: Ground truth targets tensor
            **kwargs: Additional parameters (e.g., epoch for annealing)
            
        Returns:
            Loss tensor
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss function configuration."""
        return {}


class MSELoss(BaseLoss):
    """Mean Squared Error Loss wrapper."""
    
    def __init__(self, reduction: str = 'mean', **kwargs):
        """
        Initialize MSE Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute MSE loss."""
        return self.criterion(predictions, targets)
    
    def get_config(self) -> Dict[str, Any]:
        return {'type': 'mse', 'reduction': self.reduction}


class KLDivLoss(BaseLoss):
    """KL Divergence Loss wrapper."""
    
    def __init__(self, reduction: str = 'batchmean', **kwargs):
        """
        Initialize KL Divergence Loss.
        
        Args:
            reduction: Reduction method ('batchmean', 'sum', 'mean', 'none')
        """
        super().__init__(**kwargs)
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.reduction = reduction
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Note: Predictions should be log-probabilities for KL divergence.
        """
        # Apply log_softmax to predictions for KL divergence
        log_predictions = F.log_softmax(predictions, dim=1)
        return self.criterion(log_predictions, targets)
    
    def get_config(self) -> Dict[str, Any]:
        return {'type': 'kl_div', 'reduction': self.reduction}


class FocalLoss(BaseLoss):
    """
    Focal Loss implementation with gamma annealing for handling class imbalance.
    
    The Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples. The gamma parameter controls the rate of
    down-weighting, and we implement annealing to start with high focus on
    hard examples and gradually reduce it.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma_initial: float = 2.0,
                 gamma_final: float = 1.0,
                 gamma_decay_rate: float = 0.1,
                 reduction: str = 'mean',
                 **kwargs):
        """
        Initialize Focal Loss with gamma annealing.
        
        Args:
            alpha: Class balancing weights tensor. If None, no class balancing.
            gamma_initial: Initial gamma value (higher = more focus on hard examples)
            gamma_final: Final gamma value (lower bound for annealing)
            gamma_decay_rate: Exponential decay rate for gamma annealing
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma_initial = gamma_initial
        self.gamma_final = gamma_final
        self.gamma_decay_rate = gamma_decay_rate
        self.reduction = reduction
        
        # Current gamma (will be updated during training)
        self.current_gamma = gamma_initial
    
    def _compute_gamma(self, epoch: int) -> float:
        """
        Compute current gamma using exponential decay annealing.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current gamma value
        """
        # Exponential decay: gamma = gamma_final + (gamma_initial - gamma_final) * exp(-decay_rate * epoch)
        gamma = self.gamma_final + (self.gamma_initial - self.gamma_final) * math.exp(-self.gamma_decay_rate * epoch)
        return max(gamma, self.gamma_final)  # Ensure gamma doesn't go below final value
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            predictions: Model predictions (logits) [batch_size, num_classes]
            targets: Ground truth probabilities [batch_size, num_classes]
            **kwargs: Additional parameters including 'epoch' for gamma annealing
            
        Returns:
            Focal loss tensor
        """
        # Update gamma based on current epoch
        epoch = kwargs.get('epoch', 0)
        self.current_gamma = self._compute_gamma(epoch)
        
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Convert target probabilities to class indices for standard focal loss computation
        target_classes = torch.argmax(targets, dim=1)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(predictions, target_classes, reduction='none')
        
        # Get probabilities for the true classes
        pt = probs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.current_gamma
        
        # Apply alpha balancing if provided
        if self.alpha is not None:
            if self.alpha.device != predictions.device:
                self.alpha = self.alpha.to(predictions.device)
            alpha_t = self.alpha.gather(0, target_classes)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'focal',
            'gamma_initial': self.gamma_initial,
            'gamma_final': self.gamma_final,
            'gamma_decay_rate': self.gamma_decay_rate,
            'current_gamma': self.current_gamma,
            'reduction': self.reduction,
            'alpha': self.alpha.tolist() if self.alpha is not None else None
        }


class WeightedFocalLoss(FocalLoss):
    """
    Focal Loss with automatic class weight computation for severe imbalance.
    
    This variant automatically computes alpha weights based on class frequencies
    to handle severe dataset imbalance like the SCOTUS dataset.
    """
    
    def __init__(self, 
                 class_counts: Optional[torch.Tensor] = None,
                 weight_power: float = 1.0,
                 **kwargs):
        """
        Initialize Weighted Focal Loss.
        
        Args:
            class_counts: Tensor with count of samples per class
            weight_power: Power to raise the inverse frequency weights (1.0 = standard inverse frequency)
            **kwargs: Arguments passed to parent FocalLoss
        """
        # Compute alpha weights if class counts provided
        alpha = None
        if class_counts is not None:
            # Compute inverse frequency weights
            total_samples = class_counts.sum().float()
            weights = total_samples / (class_counts.float() + 1e-8)  # Add epsilon to avoid division by zero
            
            # Apply power scaling
            if weight_power != 1.0:
                weights = weights ** weight_power
            
            # Normalize weights
            alpha = weights / weights.sum() * len(weights)
        
        super().__init__(alpha=alpha, **kwargs)
        self.class_counts = class_counts
        self.weight_power = weight_power
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'class_counts': self.class_counts.tolist() if self.class_counts is not None else None,
            'weight_power': self.weight_power
        })
        return config


def create_loss_function(loss_type: str, **kwargs) -> BaseLoss:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function ('mse', 'kl_div', 'focal', 'weighted_focal')
        **kwargs: Loss-specific configuration parameters
        
    Returns:
        Configured loss function instance
        
    Raises:
        ValueError: If loss_type is not supported
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'mse':
        return MSELoss(**kwargs)
    elif loss_type == 'kl_div':
        return KLDivLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'weighted_focal':
        return WeightedFocalLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}. "
                        f"Supported types: 'mse', 'kl_div', 'focal', 'weighted_focal'")


def get_recommended_loss_for_imbalance(class_counts: torch.Tensor, 
                                     imbalance_threshold: float = 0.7) -> str:
    """
    Recommend a loss function based on dataset class imbalance.
    
    Args:
        class_counts: Tensor with count of samples per class
        imbalance_threshold: Threshold for considering dataset imbalanced
        
    Returns:
        Recommended loss function name
    """
    total_samples = class_counts.sum().float()
    max_class_ratio = (class_counts.max().float() / total_samples).item()
    
    if max_class_ratio > imbalance_threshold:
        # Severe imbalance - recommend weighted focal loss
        return 'weighted_focal'
    elif max_class_ratio > 0.5:
        # Moderate imbalance - recommend focal loss
        return 'focal'
    else:
        # Balanced dataset - standard losses work fine
        return 'kl_div'


# Convenience function for SCOTUS dataset
def create_scotus_loss_function(loss_type: str, config, **kwargs) -> BaseLoss:
    """
    Create loss function optimized for SCOTUS dataset characteristics.
    
    Args:
        loss_type: Type of loss function
        config: Model configuration object
        **kwargs: Additional loss-specific parameters
        
    Returns:
        Configured loss function for SCOTUS dataset
    """
    # Default parameters optimized for SCOTUS dataset imbalance
    scotus_defaults = {
        'focal': {
            'gamma_initial': 2.0,
            'gamma_final': 1.0,
            'gamma_decay_rate': 0.1,
            'reduction': 'mean'
        },
        'weighted_focal': {
            'gamma_initial': 2.5,  # Higher initial gamma for severe imbalance
            'gamma_final': 1.2,
            'gamma_decay_rate': 0.15,
            'weight_power': 0.8,   # Soften the inverse frequency weights slightly
            'reduction': 'mean'
        },
        'kl_div': {
            'reduction': getattr(config, 'kl_reduction', 'batchmean')
        },
        'mse': {
            'reduction': 'mean'
        }
    }
    
    # Merge defaults with provided kwargs
    if loss_type in scotus_defaults:
        loss_kwargs = {**scotus_defaults[loss_type], **kwargs}
    else:
        loss_kwargs = kwargs
    
    return create_loss_function(loss_type, **loss_kwargs) 