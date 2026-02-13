import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = 1e-6
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()
        return bce + dice
