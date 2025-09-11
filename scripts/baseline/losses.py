import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class PerJusticeKLDivLoss(nn.Module):
    """
    KL divergence over 3-class distributions for each justice independently.

    Expects:
      - predictions: logits of shape (batch_size, 3, num_justices)
      - targets: probabilities of shape (batch_size, 3, num_justices)
    Computes KL-div along class dim (size 3), averaged over justices and batch.
    """

    def __init__(self, reduction: str = "batchmean") -> None:
        super().__init__()
        self.reduction = reduction
        # We'll implement reduction manually to keep behavior simple and explicit

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Log-softmax over class dim
        log_probs = F.log_softmax(predictions, dim=1)

        # Normalize targets along class dim to be safe
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # KLDivLoss element-wise: sum_c targets * (log_targets - log_probs)
        log_targets = torch.log(targets.clamp_min(1e-12))
        kld_per_class = targets * (log_targets - log_probs)

        # Sum over classes -> (batch, num_justices)
        kld_per_justice = kld_per_class.sum(dim=1)

        # Mean over justices, then mean over batch (batchmean)
        if self.reduction == "batchmean":
            return kld_per_justice.mean()
        elif self.reduction == "mean":
            return kld_per_justice.mean()
        elif self.reduction == "sum":
            return kld_per_justice.sum()
        else:
            # 'none': return per-sample per-justice losses
            return kld_per_justice

    def get_config(self) -> Dict[str, Any]:
        return {"loss_type": "PerJusticeKLDivLoss", "reduction": self.reduction}


def create_baseline_loss(reduction: str = "batchmean") -> PerJusticeKLDivLoss:
    return PerJusticeKLDivLoss(reduction=reduction)


