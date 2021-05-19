from typing import Tuple
from torch import Tensor

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

def get_vgg_blocks():
    vgg13 = models.vgg13_bn(pretrained=True, progress=True)
    block = []
    blocks = []
    for layer in vgg13.features:
        if isinstance(layer, nn.MaxPool2d):
            blocks.append(block)
            block = []
        else:
            block.append(layer)
    return blocks


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = get_vgg_blocks()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(*blocks[0])
        self.conv2 = nn.Sequential(*blocks[1])
        self.conv3 = nn.Sequential(*blocks[2])
        self.conv4 = nn.Sequential(*blocks[3])
        self.bottleneck = ConvBlock(512, 1024)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x = self.bottleneck(self.pool(x4))
        return x, x1, x2, x3, x4


class Upconv2d(nn.ConvTranspose2d):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = super().forward(x)
        x = torch.cat((x, y), dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv4 = Upconv2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(1024, 512)
        self.deconv3 = Upconv2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(512, 256)
        self.deconv2 = Upconv2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(256, 128)
        self.deconv1 = Upconv2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(128, 64)
        self.header = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self,
        x: Tensor,
        x4: Tensor,
        x3: Tensor,
        x2: Tensor,
        x1: Tensor) -> Tensor:

        x = self.deconv4(x, x4)
        x = self.deconv3(self.conv4(x), x3)
        x = self.deconv2(self.conv3(x), x2)
        x = self.deconv1(self.conv2(x), x1)
        x = self.header(self.conv1(x))
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: Tensor) -> Tensor:
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x, x4, x3, x2, x1)
        return x


class LinearBlock(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.header = nn.Sequential(
            nn.Flatten(1),
            LinearBlock(1024 * 7 * 7, 2048),
            LinearBlock(2048, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.header(x)
        return x


class EncoderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class UNetClassifier(UNet):
    def __init__(self):
        super().__init__()
        self.classifier = Classifier()

    def forward(self, x: Tensor) -> Tensor:
        x, x1, x2, x3, x4 = self.encoder(x)
        y = self.classifier(x)
        x = self.decoder(x, x4, x3, x2, x1)
        return x, y
