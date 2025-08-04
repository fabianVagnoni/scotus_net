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


def create_loss_function(**kwargs) -> KLDivLoss:
    """
    Create KL divergence loss function.
        
        Args:
        **kwargs: Additional arguments for loss function
        
    Returns:
        KLDivLoss instance
    """
    return KLDivLoss(**kwargs)


def create_scotus_loss_function(config=None, **kwargs) -> KLDivLoss:
    """
    Create SCOTUS-specific loss function (simplified to only KL divergence).
    
    Args:
        config: Configuration object (for compatibility)
        **kwargs: Additional arguments
        
    Returns:
        KLDivLoss instance
    """
    # Extract relevant parameters from config if provided
    loss_kwargs = {}
    if config and hasattr(config, 'loss_reduction'):
        loss_kwargs['reduction'] = config.loss_reduction
    
    # Override with any explicit kwargs
    loss_kwargs.update(kwargs)
    
    return KLDivLoss(**loss_kwargs)