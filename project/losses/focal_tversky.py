import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
    def forward(self, logits, targets):
        targets = (targets > 0.5).float()
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = torch.pow(1.0 - tversky, self.gamma)
        return loss.mean()
