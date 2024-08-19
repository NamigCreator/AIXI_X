from pathlib import Path
from typing import Literal, Optional
import pretrainedmodels
import math

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
        model_type : Literal["2d_single_channel", "2d_multichannel", "3d", "seq", "segm"] = "2d_single_channel",
        filename_checkpoint : Optional[Path] = None,
        in_dim : int = 3,
        **kwargs,
        ) -> nn.Module:
    if model_type in ["2d_single_channel", "2d_multichannel"]:
        if model_type == "2d_single_channel":
            in_dim = 1
        model = get_model_2d(in_dim=in_dim, **kwargs)
    elif model_type == "3d":
        model = Model3D(in_dim=in_dim, **kwargs)
    elif model_type == "seq":
        model = SequenceModel(**kwargs)
    elif model_type == "segm":
        model = UNet(**kwargs)
    else:
        raise ValueError(f"Unknown model_type for model initialization: {model_type}")
    
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
    
    
class SliceNorm(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        input_shape = x.shape
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = super().forward(x, **kwargs)
        x = x.view(input_shape[0], input_shape[2], input_shape[1], input_shape[3], input_shape[4])
        x = torch.permute(x, (0, 2, 1, 3, 4))
        return x
    
    
class UNet(nn.Module):
    def __init__(self,
            in_dim : int = 3,
            out_dim : int = 6,
            min_channels : int = 16,
            max_channels : int = 256,
            n_down_blocks : int = 4,
            mode : Literal["3d", "2d"] = "3d",
            ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.n_down_blocks = n_down_blocks
        self.mode = mode

        self.down_blocks = []
        self.pool_layers = []
        for i in range(n_down_blocks):
            if i == 0:
                channels_in = in_dim
                channels_1 = min_channels
            else:
                channels_in = min_channels * 2**i
                channels_1 = min_channels * 2**(i+1)
            channels_out = min_channels * 2**(i+1)
            if self.mode == "3d":
                down_block = nn.Sequential(
                    nn.Conv3d(channels_in, channels_1, (1, 3, 3), padding=(0, 1, 1), bias=False),
                    # nn.BatchNorm3d(channels_1),
                    SliceNorm(channels_1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels_1, channels_out, (1, 3, 3), padding=(0, 1, 1), bias=False),
                    # nn.BatchNorm3d(channels_out),
                    SliceNorm(channels_out),
                    nn.ReLU(inplace=True),
                )
                pool_layer = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:
                down_block = nn.Sequential(
                    nn.Conv2d(channels_in, channels_1, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(channels_1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels_1, channels_out, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(inplace=True),
                )
                pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.down_blocks.append(down_block)
            self.pool_layers.append(pool_layer)
            
        if self.mode == "3d":
            self.bottleneck = nn.Sequential(
                nn.Conv3d(max_channels, max_channels, 3, padding=1, bias=False),
                # nn.BatchNorm3d(max_channels),
                SliceNorm(max_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(max_channels, max_channels, 3, padding=1, bias=False),
                # nn.BatchNorm3d(max_channels),
                SliceNorm(max_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(max_channels, max_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(max_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(max_channels, max_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(max_channels),
                nn.ReLU(inplace=True),
            )
        
        self.up_blocks = []
        self.upsample_layers = []
        for i in range(n_down_blocks):
            channels_in = max_channels // 2**i
            if i == n_down_blocks - 1:
                channels_1 = min_channels * 2
            else:
                channels_1 = max_channels // 2**(i+1)
            channels_out = max_channels // 2**(i+1)
            if self.mode == "3d":
                upsample_layer = nn.ConvTranspose3d(channels_in, channels_in, 3,
                    stride=2, padding=1, output_padding=1)
                up_block = nn.Sequential(
                    nn.Dropout3d(p=0.5),
                    nn.Conv3d(channels_in*2, channels_1, (1, 3, 3), padding=(0, 1, 1), bias=False),
                    # nn.BatchNorm3d(channels_1),
                    SliceNorm(channels_1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels_1, channels_out, (1, 3, 3), padding=(0, 1, 1), bias=False),
                    # nn.BatchNorm3d(channels_out),
                    SliceNorm(channels_out),
                    nn.ReLU(inplace=True),
                )
            else:
                upsample_layer = nn.ConvTranspose2d(channels_in, channels_in, 3,
                    stride=2, padding=1, output_padding=1)
                up_block = nn.Sequential(
                    nn.Dropout2d(p=0.5),
                    nn.Conv2d(channels_in*2, channels_1, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(channels_1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels_1, channels_out, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(inplace=True),
                )
            self.upsample_layers.append(upsample_layer)
            self.up_blocks.append(up_block)
            
        if self.mode == "3d":
            self.out = nn.Sequential(
                nn.Conv3d(min_channels, out_dim, 1),
                # nn.Softmax(dim=1),
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(min_channels, out_dim, 1),
            )
        
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.pool_layers = nn.ModuleList(self.pool_layers)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.upsample_layers = nn.ModuleList(self.upsample_layers)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "3d":
            input_shape = (inputs.shape[2], inputs.shape[3], inputs.shape[4])
            input_shape_corrected = (
                int(math.ceil(input_shape[0]/2**len(self.down_blocks)))*2**len(self.down_blocks),
                int(math.ceil(input_shape[1]/2**len(self.down_blocks)))*2**len(self.down_blocks),
                int(math.ceil(input_shape[2]/2**len(self.down_blocks)))*2**len(self.down_blocks),
            )
            if input_shape[0] != input_shape_corrected[0] \
                    or input_shape[1] != input_shape_corrected[1] \
                    or input_shape[2] != input_shape_corrected[2]:
                x = torch.nn.functional.interpolate(inputs, size=input_shape_corrected, mode="bilinear")
                interpolated = True
            else:
                x = inputs
                interpolated = False
        else:
            input_shape = (inputs.shape[2], inputs.shape[3])
            input_shape_corrected = (
                int(math.ceil(input_shape[0]/2**len(self.down_blocks)))*2**len(self.down_blocks),
                int(math.ceil(input_shape[1]/2**len(self.down_blocks)))*2**len(self.down_blocks),
            )
            if input_shape[0] != input_shape_corrected[0] \
                    or input_shape[1] != input_shape_corrected[1]:
                x = torch.nn.functional.interpolate(inputs, size=input_shape_corrected, mode="bilinear")
                interpolated = True
            else:
                x = inputs
                interpolated = False
            
        downs = []
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
            downs.append(x)
            x = self.pool_layers[i](x)
            
        for i in range(len(self.up_blocks)):
            x = self.upsample_layers[i](x)
            x = torch.cat([downs[len(downs)-i-1], x], dim=1)
            x = self.up_blocks[i](x)
        
        logits = self.out(x)
        
        if interpolated:
            logits = torch.nn.functional.interpolate(logits, size=input_shape, mode="bilinear")
            
        if self.mode == "3d":
            assert logits.shape == (inputs.shape[0], self.out_dim, inputs.shape[2], inputs.shape[3], inputs.shape[4]), "Wrong shape of the logits"
        else:
            assert logits.shape == (inputs.shape[0], self.out_dim, inputs.shape[2], inputs.shape[3]), "Wrong shape of the logits"
        return logits