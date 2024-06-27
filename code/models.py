import torch
import torch.nn as nn
import torch.nn.functional
from torchvision.models import vgg19
from torchvision.models._utils import IntermediateLayerGetter


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.downscaling = nn.Sequential(
            ConvBlock(in_channels=6, out_channels=64, kernel_size=7, stride=1),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2),
        )
        self.residual_part = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=256, out_channels=256, kernel_size=3, stride=1
                )
                for _ in range(8)
            ]
        )

        self.upscaling = nn.Sequential(
            TransposeConvBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            TransposeConvBlock(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ConvBlock(
                in_channels=64, out_channels=3, kernel_size=7, stride=1, last_layer=True
            ),
        )

    def forward(self, x):
        x = self.downscaling(x)
        x = self.residual_part(x)
        x = self.upscaling(x)
        return x


import math


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=None,
        last_layer=False,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = math.ceil((kernel_size / stride - 1) * (stride / 2))

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_channels),
        )
        self.last_layer = last_layer

    def forward(self, x):
        x = self.layer(x)
        if self.last_layer:
            return torch.sigmoid(x)

        return torch.nn.functional.relu(x, inplace=True)


class TransposeConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding=1
        )
        self.instance = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.instance(out)
        return out + residual


class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.head(encoded)


class VGG19_intermediate_layers_only(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG19_intermediate_layers_only, self).__init__(*args, **kwargs)
        self.vgg19 = vgg19(pretrained=True)
        self.return_layers = {
            "1": "out_layer1",
            "6": "out_layer2",
            "11": "out_layer3",
            "20": "out_layer4",
            "29": "out_layer5",
        }
        self.model_with_multuple_layer = IntermediateLayerGetter(
            self.vgg19.features, return_layers=self.return_layers
        )

    def forward(self, x):
        return self.model_with_multuple_layer(x)


if __name__ == "__main__":

    model = Generator()
    # model = TransposeConvBlock(
    # in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
    # )
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")
    x = torch.rand(5, 6, 512, 512)
    result = model(x)
    print(result.shape)
