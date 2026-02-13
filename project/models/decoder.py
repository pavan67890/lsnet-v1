import torch
import torch.nn as nn
import torch.nn.functional as F

class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.step1 = DWSeparableConv(56, 96)
        self.dec2 = DWSeparableConv(96 + 32, 64)
        self.dec1 = DWSeparableConv(64 + 24, 32)
        class IR(nn.Module):
            def __init__(self, c, e):
                super().__init__()
                mid = c * e
                self.pw1 = nn.Conv2d(c, mid, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(mid)
                self.dw = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
                self.bn2 = nn.BatchNorm2d(mid)
                self.pw2 = nn.Conv2d(mid, c, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(c)
                self.act = nn.ReLU(inplace=True)
            def forward(self, x):
                h = self.act(self.bn1(self.pw1(x)))
                h = self.act(self.bn2(self.dw(h)))
                h = self.bn3(self.pw2(h))
                return self.act(h + x)
        self.ref_s1 = nn.Sequential(IR(96, 4), IR(96, 4))
        self.ref_x2 = nn.Sequential(IR(64, 4), IR(64, 4))
        self.ref_x1 = nn.Sequential(IR(32, 4), IR(32, 4), IR(32, 4))
        self.conv3x3a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn3x3a = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3c = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn3x3c = nn.BatchNorm2d(32)
        self.conv3x3b = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn3x3b = nn.BatchNorm2d(16)
        self.conv1x1 = nn.Conv2d(16, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
    def forward(self, d3, d2, d1):
        u3 = F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        s1 = self.step1(u3)
        s1 = self.ref_s1(s1)
        x2 = self.dec2(torch.cat([s1, d2], dim=1))
        x2 = self.ref_x2(x2)
        u2 = F.interpolate(x2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.dec1(torch.cat([u2, d1], dim=1))
        x1 = self.ref_x1(x1)
        u1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        z = self.relu(self.bn3x3a(self.conv3x3a(u1)))
        z = self.relu(self.bn3x3c(self.conv3x3c(z)))
        z = self.relu(self.bn3x3b(self.conv3x3b(z)))
        o = self.conv1x1(z)
        p = self.sig(o)
        return o, p
