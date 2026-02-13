import torch
import torch.nn as nn
from .encoder import EfficientNetB4SlicedEncoder
from .d3e import D3EBlock
from .decoder import Decoder

class LSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetB4SlicedEncoder()
        self.d3e1 = D3EBlock(24)
        self.d3e2 = D3EBlock(32)
        self.d3e3 = D3EBlock(56)
        self.decoder = Decoder()
    def forward(self, t1, t2):
        c0_1, c1_1, c2_1, c3_1 = self.encoder(t1)
        c0_2, c1_2, c2_2, c3_2 = self.encoder(t2)
        d1 = self.d3e1(c1_1, c1_2)
        d2 = self.d3e2(c2_1, c2_2)
        d3 = self.d3e3(c3_1, c3_2)
        logits, probs = self.decoder(d3, d2, d1)
        return logits, probs
