import torch
import torch.nn as nn
import torch.nn.functional
from torchvision.models import vgg19
from torchvision.models._utils import IntermediateLayerGetter


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Generator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        output = (torch.tanh(x) + 1) / 2
        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=dilation,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            #
            nn.ReflectionPad2d(1),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class Generator_v0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.downscaling = nn.Sequential(
            # nn.ReflectionPad2d(3),
            ConvBlock(in_channels=2, out_channels=64, kernel_size=7, stride=1),
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
            # nn.ReflectionPad2d(3),
            ConvBlock(
                in_channels=64, out_channels=1, kernel_size=7, stride=1, last_layer=True
            ),
        )

    def forward(self, x):
        x = self.downscaling(x)
        x = self.residual_part(x)
        x = self.upscaling(x)
        print("mine g: ", x.sum(), x.mean())
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
            return (torch.tanh(x) + 1) / 2

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


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(256, 1, kernel_size=3, stride=2, padding=1),
            ),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.head(encoded)


class VGG19_intermediate_layers_only(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG19_intermediate_layers_only, self).__init__(*args, **kwargs)
        model_vgg19 = vgg19(pretrained=True)
        self.return_layers = {
            "1": "out_layer1",
            "6": "out_layer2",
            "11": "out_layer3",
            "20": "out_layer4",
            "29": "out_layer5",
        }
        self.model_with_multuple_layer = IntermediateLayerGetter(
            model_vgg19.features, return_layers=self.return_layers
        )
        for param in self.model_with_multuple_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model_with_multuple_layer(x)
