from typing import Tuple
from torch import Tensor

import torch
from torch import nn

def ConvBlock(in_channels: int, out_channels: int, bias: bool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Deconv2d(nn.ConvTranspose2d):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = super().forward(x)
        x = torch.cat((x, y), dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_features: int, bias: bool):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, num_features, bias)
        self.conv2 = ConvBlock(num_features, num_features * 2, bias)
        self.conv3 = ConvBlock(num_features * 2, num_features * 4, bias)
        self.conv4 = ConvBlock(num_features * 4, num_features * 8, bias)
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
    def __init__(self, num_features: int, bias):
        super().__init__()
        self.deconv4 = Deconv2d(num_features * 16, num_features * 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(num_features * 16, num_features * 8, bias)
        self.deconv3 = Deconv2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(num_features * 8, num_features * 4, bias)
        self.deconv2 = Deconv2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(num_features * 4, num_features * 2, bias)
        self.deconv1 = Deconv2d(num_features * 2, num_features, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(num_features * 2, num_features, bias)

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
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32, bias: bool = False):
        super().__init__()
        num_feautres = init_features
        self.encoder = Encoder(in_channels, num_feautres, bias)
        self.bottleneck = ConvBlock(num_feautres * 8, num_feautres * 16, bias)
        self.decoder = Decoder(num_feautres, bias)
        self.header = nn.Sequential(
            nn.Conv2d(num_feautres, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, x4, x3, x2, x1)
        x = self.header(x)
        return x

    @staticmethod
    def load_from_torch_hub():
        model = UNet()
        state_dict = model.state_dict()
        state_dict_hub = torch.load('torch_hub_unet.pt')
        for key_hub, key in zip(state_dict_hub.keys(), state_dict.keys()):
            state_dict[key] = state_dict_hub[key_hub]
        model.load_state_dict(state_dict)
        return model


def LinearBlock(in_features, out_features):
    return (
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
    )


class EncoderClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, init_features: int = 32, bias: bool = False):
        super().__init__()
        self.encoder = Encoder(in_channels, init_features, bias)
        self.encoder.requires_grad_(False)
        self.classifier = nn.Sequential(
            *LinearBlock(256 * 14 * 14, 4096),
            *LinearBlock(4096, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.encoder(x)[0]
        x = self.classifier(x.flatten(1))
        return x
