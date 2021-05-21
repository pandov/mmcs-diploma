from typing import Tuple
from torch import Tensor

import torch
from torch import nn


class VGGBlock(nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool):

        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class VGGEncoder(nn.Module):
    def __init__(self,
    in_channels: int,
    num_features: int,
    bias: bool):

        super().__init__()
        self.conv1 = VGGBlock(
            in_channels, num_features, bias)
        self.conv2 = VGGBlock(
            num_features, num_features * 2, bias)
        self.conv3 = VGGBlock(
            num_features * 2, num_features * 4, bias)
        self.conv4 = VGGBlock(
            num_features * 4, num_features * 8, bias)
        self.bottleneck = VGGBlock(
            num_features * 8, num_features * 16, bias)
        self.pool = nn.MaxPool2d(2)

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
    def __init__(self, num_features: int, bias):
        super().__init__()
        self.deconv4 = Upconv2d(
            num_features * 16, num_features * 8,
            kernel_size=2, stride=2)
        self.conv4 = VGGBlock(
            num_features * 16, num_features * 8, bias)
        self.deconv3 = Upconv2d(
            num_features * 8, num_features * 4,
            kernel_size=2, stride=2)
        self.conv3 = VGGBlock(
            num_features * 8, num_features * 4, bias)
        self.deconv2 = Upconv2d(
            num_features * 4, num_features * 2,
            kernel_size=2, stride=2)
        self.conv2 = VGGBlock(
            num_features * 4, num_features * 2, bias)
        self.deconv1 = Upconv2d(
            num_features * 2, num_features,
            kernel_size=2, stride=2)
        self.conv1 = VGGBlock(
            num_features * 2, num_features, bias)
        self.header = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1),
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
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        bias: bool = False):
        super().__init__()
        self.encoder = VGGEncoder(
            in_channels, init_features, bias)
        self.decoder = Decoder(init_features, bias)

    def forward(self, x: Tensor) -> Tensor:
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x, x4, x3, x2, x1)
        return x


class LinearBlock(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2)
        self.header = nn.Sequential(
            nn.Flatten(1),
            LinearBlock(512 * 7 * 7, 1024),
            LinearBlock(1024, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.header(x)
        return x


class VGGEncoderClassifier(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        init_features: int = 32,
        bias: bool = False):

        super().__init__()
        self.encoder = VGGEncoder(
            in_channels, init_features, bias)
        self.encoder.requires_grad_(False)
        self.classifier = Classifier()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x, _, _, _, _ = self.encoder(x)
        x = self.classifier(x)
        return x


class UNetClassifier(UNet):
    def __init__(self,
        in_channels: int = 3,
        init_features: int = 32,
        bias: bool = False):

        super().__init__()
        self.classifier = Classifier()
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        z = torch.zeros_like(x)
        x, x1, x2, x3, x4 = self.encoder(x)
        y = self.classifier(x)
        idx = torch.where(y > 0.5)[0]
        z[idx] = self.decoder(
            x[idx], x4[idx], x3[idx], x2[idx], x1[idx])
        return z, y
