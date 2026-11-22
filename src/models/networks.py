"""Dua arsitektur pembanding dengan backbone VGG16-BN pra-latih (stride 8).

- CountRegressor: fitur -> global average pooling -> MLP -> satu angka (jumlah orang).
- CSRNet: fitur -> konvolusi dilasi -> peta kepadatan 1/8; jumlah orang = jumlah seluruh piksel peta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn


def vgg_frontend(pretrained: bool = True) -> nn.Sequential:
    vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(vgg.features.children())[:33])  # sampai conv4_3 + BN + ReLU, stride 8, 512 kanal


class CSRNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.frontend = vgg_frontend(pretrained)
        layers, c_in = [], 512
        for c in (512, 512, 512, 256, 128, 64):
            layers += [nn.Conv2d(c_in, c, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
            c_in = c
        self.backend = nn.Sequential(*layers)
        self.head = nn.Conv2d(64, 1, 1)
        for m in list(self.backend.modules()) + [self.head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.relu(self.head(self.backend(self.frontend(x))))


class CountRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.frontend = vgg_frontend(pretrained)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, x):
        # memprediksi log(1 + jumlah) agar rentang 1..3000 stabil untuk dilatih
        return self.head(self.frontend(x)).squeeze(1)
