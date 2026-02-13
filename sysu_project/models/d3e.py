import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        mid = max(1, channels // r)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc2(self.act(self.fc1(s)))
        s = self.sig(s)
        return x * s

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        a = self.sig(self.conv(a))
        return x * a

class D3EBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se = SE(channels, r=8)
        self.sa = SpatialAttention()
    def forward(self, f1, f2):
        d = torch.abs(f1 - f2)
        d = self.se(d)
        d = self.sa(d)
        return d
