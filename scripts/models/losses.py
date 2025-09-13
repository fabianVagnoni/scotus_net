"""
Simplified loss functions for SCOTUS voting model.
Only includes KL Divergence Loss as requested for simplification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class KLDivLoss(nn.Module):
    """
    KL Divergence Loss for probability distributions.
    """
    
    def __init__(self, reduction: str = 'batchmean', **kwargs):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.KLDivLoss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            predictions: Raw logits from model, shape (batch_size, 3)
            targets: Target probability distributions, shape (batch_size, 3)
            
        Returns:
            KL divergence loss
        """
        # Convert logits to log probabilities
        log_predictions = F.log_softmax(predictions, dim=-1)
        
        # Ensure targets sum to 1 (normalize if needed)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        
        return self.loss_fn(log_predictions, targets)
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        return {
            'loss_type': 'KLDivLoss',
            'reduction': self.reduction
        }

class MSELoss(nn.Module):
    """
    Mean Squared Error Loss for probability distributions.
    """
    
    def __init__(self, reduction: str = 'batchmean', **kwargs):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Mean Squared Error Loss.
        
        Args:
            predictions: Raw logits from model, shape (batch_size, 3)
            targets: Target probability distributions, shape (batch_size, 3)
            
        Returns:
            Mean Squared Error Loss
        """
        predictions = F.softmax(predictions, dim=-1)
        return self.loss_fn(predictions, targets)
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        return {
            'loss_type': 'MSELoss',
            'reduction': self.reduction
        }


def create_loss_function(**kwargs) -> KLDivLoss:
    """
    Create KL divergence loss function.
        
        Args:
        **kwargs: Additional arguments for loss function
        
    Returns:
        KLDivLoss instance
    """
    return KLDivLoss(**kwargs)


def create_scotus_loss_function(config=None, **kwargs):
    """
    Create SCOTUS-specific loss function (KL divergence or MSE based on config).
    
    Args:
        config: Configuration object with loss_type attribute
        **kwargs: Additional arguments
        
    Returns:
        KLDivLoss or MSELoss instance
    """
    # Extract relevant parameters from config if provided
    loss_kwargs = {}
    if config and hasattr(config, 'loss_reduction'):
        loss_kwargs['reduction'] = config.loss_reduction
    
    # Override with any explicit kwargs
    loss_kwargs.update(kwargs)
    
    # Determine loss type from config
    loss_type = 'KL'  # Default to KL
    if config and hasattr(config, 'loss_type'):
        loss_type = config.loss_type.upper()
    
    if loss_type == 'MSE':
        return MSELoss(**loss_kwargs)
    else:  # Default to KL
        return KLDivLoss(**loss_kwargs)