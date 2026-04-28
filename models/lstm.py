import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base import BaseModel

class BidirectionalLSTM(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 3,
        use_attention: bool = True
    ):
        super().__init__("bidirectional_lstm", input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_attention:
            self.attention = TemporalAttention(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            
        logits = self.classifier(context)
        return logits
    
    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, features = x.shape
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out, hidden


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = self.attention(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        context = torch.sum(attn_weights * lstm_output, dim=1)
        
        return context, attn_weights.squeeze(-1)


class StackedLSTM(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("stacked_lstm", input_dim, output_dim)
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.LSTM(prev_dim, hidden_dim, batch_first=True))
            self.dropouts.append(nn.Dropout(dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        self.attention = TemporalAttention(hidden_dims[-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lstm, dropout, norm in zip(self.layers, self.dropouts, self.layer_norms):
            x, _ = lstm(x)
            x = norm(x)
            x = dropout(x)
            
        context, _ = self.attention(x)
        logits = self.classifier(context)
        return logits


class ConvLSTM(BaseModel):
    def __init__(
        self,
        input_dim: int,
        conv_channels: list = [64, 128],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("conv_lstm", input_dim, output_dim)
        
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
            
        self.conv = nn.Sequential(*conv_layers)
        
        self.lstm = nn.LSTM(
            conv_channels[-1],
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.attention = TemporalAttention(lstm_hidden * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        
        logits = self.classifier(context)
        return logits
