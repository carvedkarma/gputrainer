import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base import BaseModel, ResidualBlock

class ResNetPrice(BaseModel):
    def __init__(
        self,
        input_dim: int,
        channels: List[int] = [64, 128, 256, 512],
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("resnet_price", input_dim, output_dim)
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels[0], kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(channels[0], channels[0], 2)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[3], channels[3] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[3] // 2, output_dim)
        )
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        logits = self.classifier(x)
        return logits


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_1x1: int, out_3x3: int, 
                 out_5x5: int, out_pool: int):
        super().__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm1d(out_1x1),
            nn.ReLU()
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_3x3),
            nn.ReLU()
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_5x5),
            nn.ReLU()
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm1d(out_pool),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class InceptionNet(BaseModel):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("inception_net", input_dim, output_dim)
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        self.inception1 = InceptionModule(64, 16, 32, 16, 16)
        self.inception2 = InceptionModule(80, 32, 64, 32, 32)
        self.inception3 = InceptionModule(160, 64, 128, 64, 64)
        
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.inception1(x)
        x = self.pool(x)
        x = self.inception2(x)
        x = self.pool(x)
        x = self.inception3(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        logits = self.classifier(x)
        return logits


class WaveNet(BaseModel):
    def __init__(
        self,
        input_dim: int,
        residual_channels: int = 64,
        dilation_channels: int = 64,
        skip_channels: int = 128,
        num_blocks: int = 4,
        layers_per_block: int = 8,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("wavenet", input_dim, output_dim)
        
        self.input_conv = nn.Conv1d(input_dim, residual_channels, kernel_size=1)
        
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for b in range(num_blocks):
            for l in range(layers_per_block):
                dilation = 2 ** l
                self.dilated_convs.append(
                    nn.Conv1d(
                        residual_channels, 
                        dilation_channels * 2,
                        kernel_size=2,
                        dilation=dilation,
                        padding=dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
                )
                self.skip_convs.append(
                    nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)
                )
                
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(skip_channels, skip_channels // 2),
            nn.ReLU(),
            nn.Linear(skip_channels // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        
        skip_sum = 0
        for dilated, residual, skip in zip(
            self.dilated_convs, self.residual_convs, self.skip_convs
        ):
            dilated_out = dilated(x)
            dilated_out = dilated_out[:, :, :x.size(2)]
            
            filter_out, gate_out = torch.chunk(dilated_out, 2, dim=1)
            gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            skip_sum = skip_sum + skip(gated)
            x = x + residual(gated)
            
        out = self.output_conv(skip_sum)
        out = self.global_pool(out)
        out = out.flatten(1)
        
        logits = self.classifier(out)
        return logits
