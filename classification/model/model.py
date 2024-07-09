from pathlib import Path
from typing import Literal, Optional
import pretrainedmodels

import torch
import torch.nn as nn
import torchvision.models


def get_model_2d(
        model_name : Literal["efficientnet"] = "efficientnet",
        in_dim : int = 3,
        out_dim : int = 6,
        ) -> nn.Module:
    if model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0(pretrained=True)
        if in_dim != 3:
            new_features = nn.Sequential(*list(model.features.children()))
            new_features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=2, stride=2, padding=1, bias=False)
            pretrained_weights = model.features[0][0].weight
            new_features[0][0].weight.data.normal_(0, 0.001)
            new_features[0][0].weight.data[:, :3, :, :] = pretrained_weights
            model.features = new_features
        model.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model = nn.Sequential(
            model,
            nn.Linear(1000, out_dim),
        )

    model = model.to(torch.float)
    return model


class Model3D(torch.nn.Module):
    def __init__(self, in_dim : int = 3, out_dim : int = 6):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv3d(in_dim, 32, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # 
            nn.Conv3d(32, 64, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # 
            nn.Conv3d(64, 64, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # 
            nn.Conv3d(64, 128, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # 
            nn.Conv3d(128, 256, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # 
            nn.Conv3d(256, 256, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 4, 4)),
        )
        self.head = nn.Conv1d(256, out_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.squeeze(x, (3, 4))
        x = self.head(x)
        return x


def get_model(
        mode : Literal["2d_single_channel", "2d_multichannel", "3d"] = "2d_single_channel",
        filename_checkpoint : Optional[Path] = None,
        in_dim : int = 3,
        **kwargs,
        ) -> nn.Module:
    if mode in ["2d_single_channel", "2d_multichannel"]:
        if mode == "2d_single_channel":
            in_dim = 1
        model = get_model_2d(in_dim=in_dim, **kwargs)
    elif mode == "3d":
        model = Model3D(in_dim=in_dim, **kwargs)
    else:
        raise ValueError(f"Unknown mode for model initialization: {mode}")
    
    if filename_checkpoint is not None:
        state_dict = torch.load(filename_checkpoint)["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(state_dict)
        model.eval()
    return model