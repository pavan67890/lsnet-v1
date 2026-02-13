import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class EfficientNetB4SlicedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        feats = list(m.features)
        x = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            o = x
            outs = []
            modules = []
            for i, mod in enumerate(feats):
                o = mod(o)
                modules.append(mod)
                outs.append(o)
                if o.shape[1] == 48 and o.shape[-2] == 128 and o.shape[-1] == 128:
                    idx0 = i
                if o.shape[1] == 24 and o.shape[-2] == 128 and o.shape[-1] == 128:
                    idx1 = i
                if o.shape[1] == 32 and o.shape[-2] == 64 and o.shape[-1] == 64:
                    idx2 = i
                if o.shape[1] == 56 and o.shape[-2] == 32 and o.shape[-1] == 32:
                    idx3 = i
        self.m0 = nn.Sequential(*feats[: idx0 + 1])
        self.m1 = nn.Sequential(*feats[idx0 + 1 : idx1 + 1])
        self.m2 = nn.Sequential(*feats[idx1 + 1 : idx2 + 1])
        self.m3 = nn.Sequential(*feats[idx2 + 1 : idx3 + 1])
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
        self.ref0 = nn.Sequential(IR(48, 8), IR(48, 8))
        self.ref1 = nn.Sequential(IR(24, 8), IR(24, 8), IR(24, 8), IR(24, 8))
        self.ref2 = nn.Sequential(IR(32, 8), IR(32, 8), IR(32, 8), IR(32, 8))
        self.ref3 = nn.Sequential(IR(56, 8), IR(56, 8), IR(56, 8), IR(56, 8))

    def forward(self, x):
        c0 = self.m0(x)
        c0 = self.ref0(c0)
        c1 = self.m1(c0)
        c1 = self.ref1(c1)
        c2 = self.m2(c1)
        c2 = self.ref2(c2)
        c3 = self.m3(c2)
        c3 = self.ref3(c3)
        return c0, c1, c2, c3
