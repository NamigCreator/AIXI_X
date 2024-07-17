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
    elif model_name in ["se_resnext50_32x4d", "se_resnext101_32x4d"]:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model = nn.Sequential(
            model,
            nn.Linear(1000, out_dim),
        )
    elif model_name in ["se_resnext50_32x4d_2", "se_resnext101_32x4d_2"]:
        # from https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/2DNet/src/net/models.py
        model = pretrainedmodels.__dict__[model_name[:-2]](num_classes=1000, pretrained="imagenet")
        num_ftrs = model.last_linear.in_features
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

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
        mode : Literal["2d_single_channel", "2d_multichannel", "3d", "seq"] = "2d_single_channel",
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
    elif mode == "seq":
        model = SequenceModel(**kwargs)
    else:
        raise ValueError(f"Unknown mode for model initialization: {mode}")
    
    if filename_checkpoint is not None:
        state_dict = torch.load(filename_checkpoint)["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(state_dict)
        model.eval()
    return model


# from https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/SequenceModel/seq_model.py
class SequenceModel(nn.Module):
    def __init__(self, 
            feature_dim : int = 128,
            lstm_layers : int = 2,
            hidden : int = 96,
            dropout : float = 0.5,
            n_out : int = 6,
            ):
        super().__init__()
        
        self.fea_conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fea_first_final = nn.Conv1d(128, n_out, 1, bias=True)
        
        self.hidden_fea = hidden
        self.fea_lstm = nn.LSTM(128, self.hidden_fea, 
            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fea_lstm_final = nn.Conv1d(self.hidden_fea*2, n_out, kernel_size=1)
        
        ratio = 4
            
        self.conv_first = nn.Sequential(
            nn.Conv1d(feature_dim+2*n_out, 128*ratio, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128*ratio),
            nn.ReLU(inplace=True),
            nn.Conv1d(128*ratio, 64*ratio, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(64*ratio),
            nn.ReLU(inplace=True),
        )
        self.conv_res = nn.Sequential(
            nn.Conv1d(64*ratio, 64*ratio, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(64*ratio),
            nn.ReLU(inplace=True),
            nn.Conv1d(64*ratio, 64*ratio, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(64*ratio),
            nn.ReLU(inplace=True),
        )
        self.conv_final = nn.Conv1d(64*ratio, n_out, kernel_size=3, padding=1, bias=True)
        
        self.hidden = hidden
        self.lstm = nn.LSTM(64*ratio, self.hidden, 
            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.final = nn.Conv1d(self.hidden*2, n_out, kernel_size=1, bias=True)
        
        
    def forward(self, embeds: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        features = embeds.permute(0, 2, 1)
        # B x 128 x L
        features = self.fea_conv(features)
        # B x 6 x L
        features_first_final = self.fea_first_final(features)
        out0 = features_first_final
        
        # B x hidden x L
        features = features.permute(0, 2, 1)
        features, _ = self.fea_lstm(features)
        features = features.permute(0, 2, 1)
        # B x 6 x L
        features_lstm_final = self.fea_lstm_final(features)
        out0 += features_lstm_final
        
        # B x 6 x L
        out0_sigmoid = torch.sigmoid(out0)
        # B x (C+6) x L
        x = torch.cat([embeds.permute(0, 2, 1), preds.permute(0, 2, 1), out0_sigmoid], dim=1)
        # B x 64*4 x L
        x = self.conv_first(x)
        # B x 64*4 x L
        x = self.conv_res(x)
        # B x 6 x L
        x_cnn = self.conv_final(x)
        out = x_cnn
        
        # B x 64*ratio * L
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        # B x 6 x L
        x = self.final(x)
        out += x
        return out, out0