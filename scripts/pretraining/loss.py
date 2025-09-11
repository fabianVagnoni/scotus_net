import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, alpha: float = 0.5, rho: float = 0.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rho = rho
    
    def to(self, device):
        """Override to method to ensure all components are moved to device."""
        super().to(device)
        self.device = device
        self.cross_entropy_loss = self.cross_entropy_loss.to(device)
        return self

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def forward(self, e_t, e_f):
        """Loss function for contrastive learning based on NT-Xent Loss & MSE."""
        e_t , e_f = self._to_device(e_t) , self._to_device(e_f).detach()
        e_t = F.normalize(e_t, dim=-1)
        e_f = F.normalize(e_f, dim=-1)
        nx_loss = self.nx_loss(e_t, e_f)
        mse_loss = self.mse_loss(e_t, e_f)
        return self.alpha * nx_loss + (1 - self.alpha) * mse_loss
    
    def nx_loss(self, e_t, e_f):
        """NT-Xent Loss."""
        pos = (e_t * e_f).sum(dim=-1, keepdim=True) / self.temperature            # (B,1)

        neg = e_t @ e_f.T / self.temperature                                      # (B,B)
        neg.fill_diagonal_(-1e9)                                                # mask positives

        if self.rho > 0:
            pos_exp = torch.exp(pos)
            neg_exp = torch.exp(neg).sum(dim=1, keepdim=True)
            denom = pos_exp + torch.clamp((1 - self.rho) * neg_exp, min=1e-12)
            return -(pos - torch.log(denom)).mean()

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(e_t.size(0), dtype=torch.long, device=e_t.device)
        return F.cross_entropy(logits, labels)
    
    def mse_loss(self, e_t, e_f):
        """MSE Loss."""
        return 1 - F.cosine_similarity(e_t, e_f, dim=-1).mean()
    