import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, alpha: float = 0.5, device: str = 'cuda'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = device
        self.to(self.device)

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def forward(self, e_t, e_f):
        """Loss function for contrastive learning based on NT-Xent Loss & MSE."""
        e_t , e_f = self._to_device(e_t) , self._to_device(e_f)
        nx_loss = self.nx_loss(e_t, e_f)
        mse_loss = self.mse_loss(e_t, e_f)
        return self.alpha * nx_loss + (1 - self.alpha) * mse_loss
    
    def nx_loss(self, e_t, e_f):
        """NT-Xent Loss."""
        e_t = F.normalize(e_t, dim=-1)
        e_f = F.normalize(e_f, dim=-1)
        sim = (e_t @ e_f.T) / self.temperature # (B,D) @ (D,B) => (B,B)
        labels = torch.arange(sim.size(0))
        c1 = self.cross_entropy_loss(sim, labels)
        c2 = self.cross_entropy_loss(sim.T, labels)
        return (c1 + c2) * .5
    
    def mse_loss(self, e_t, e_f):
        """MSE Loss."""
        return torch.mean((e_t - e_f)**2)
    

