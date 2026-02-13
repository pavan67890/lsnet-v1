import math
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou

class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.total_epochs - self.warmup_epochs)
        return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t / T)) for base_lr in self.base_lrs]
