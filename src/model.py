from typing import Tuple
from torch import Tensor

import torch
from torch import nn


class Deconv2d(nn.ConvTranspose2d):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = super().forward(x)
        x = torch.cat((x, y), dim=1)
        return x


def ConvBlock(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int]):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.conv4 = ConvBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x1 = self.conv1(x)
        x = self.pool(x1)
        x2 = self.conv2(x)
        x = self.pool(x2)
        x3 = self.conv3(x)
        x = self.pool(x3)
        x4 = self.conv4(x)
        x = self.pool(x4)
        return x, x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, channels: Tuple[int], dropout: float):
        super().__init__()
        self.deconv4 = Deconv2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.conv4 = ConvBlock(channels[4], channels[3])
        self.deconv3 = Deconv2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.conv3 = ConvBlock(channels[3], channels[2])
        self.deconv2 = Deconv2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.conv2 = ConvBlock(channels[2], channels[1])
        self.deconv1 = Deconv2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.conv1 = ConvBlock(channels[1], channels[0])

    def forward(self, x: Tensor, x4: Tensor, x3: Tensor, x2: Tensor, x1: Tensor) -> Tensor:
        x = self.deconv4(x, x4)
        x = self.conv4(x)
        x = self.deconv3(x, x3)
        x = self.conv3(x)
        x = self.deconv2(x, x2)
        x = self.conv2(x)
        x = self.deconv1(x, x1)
        x = self.conv1(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, channels: Tuple[int] = (32, 64, 128, 256, 512), dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder(in_channels, channels)
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.decoder = Decoder(channels, dropout)
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, x4, x3, x2, x1)
        x = self.head(x)
        return x


def LinearBlock(in_features, out_features):
    return (
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
    )


class UNetClassifier(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            *LinearBlock(512 * 14 * 14, 8192),
            *LinearBlock(8192, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.bottleneck(x)
        y = self.classifier(x.flatten(1))
        x = self.decoder(x, x4, x3, x2, x1)
        x = self.head(x)
        return x, y
