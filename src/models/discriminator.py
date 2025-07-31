# src/models/discriminator.py
import torch, torch.nn as nn, torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=not norm)]
            if norm: layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            block(in_channels * 2, 64, norm=False),
            block(64, 128), block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_scales=2):
        super().__init__()
        self.discriminators = nn.ModuleList([Discriminator(in_channels) for _ in range(num_scales)])
    def forward(self, x, y):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x, y))
            x = F.avg_pool2d(x, 3, 2, 1)
            y = F.avg_pool2d(y, 3, 2, 1)
        return outputs