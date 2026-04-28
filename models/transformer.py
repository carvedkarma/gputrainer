import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseModel, PositionalEncoding, AttentionBlock

class TransformerPriceModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        output_dim: int = 3
    ):
        super().__init__("transformer_price", input_dim, output_dim)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for block in self.attention_blocks:
            x = block(x, mask)
            
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        logits = self.classifier(x)
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        attn_weights = []
        for block in self.attention_blocks:
            _, weights = block.attention(x, x, x, need_weights=True)
            attn_weights.append(weights)
            x = block(x)
            
        return torch.stack(attn_weights)


class TemporalFusionTransformer(BaseModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_quantiles: int = 3,
        output_dim: int = 3
    ):
        super().__init__("temporal_fusion_transformer", input_dim, output_dim)
        
        self.d_model = d_model
        
        self.static_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.temporal_encoder = nn.LSTM(
            input_dim, d_model // 2, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.variable_selection = VariableSelectionNetwork(d_model, input_dim, dropout)
        
        self.enrichment = GatedResidualNetwork(d_model, d_model, dropout)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        
        self.attention_blocks = nn.ModuleList([
            InterpretableMultiHeadAttention(d_model, nhead, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.post_attention = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.quantile_output = nn.Linear(d_model, num_quantiles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.temporal_encoder(x)
        
        static_features = self.static_encoder(x.mean(dim=1))
        
        enriched = self.enrichment(lstm_out, static_features.unsqueeze(1).expand(-1, lstm_out.size(1), -1))
        
        enriched = enriched.transpose(0, 1)
        enriched = self.pos_encoding(enriched)
        enriched = enriched.transpose(0, 1)
        
        for attn_block, grn in zip(self.attention_blocks, self.post_attention):
            attn_out, _ = attn_block(enriched)
            enriched = grn(attn_out + enriched)
            
        output_repr = enriched[:, -1, :]
        
        logits = self.output_layer(output_repr)
        return logits


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model: int, num_inputs: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(d_model * num_inputs, d_model, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.num_inputs = num_inputs
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        flattened = x.view(batch_size, seq_len, -1)
        weights = self.grn(flattened)
        weights = self.softmax(weights)
        return weights


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, 
                 context_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim else None
        
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gate_sigmoid = nn.Sigmoid()
        
        self.skip_fc = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip_fc(x) if self.skip_fc else x
        
        hidden = self.fc1(x)
        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        
        gate = self.gate_sigmoid(self.gate_fc(hidden))
        gated = gate * hidden
        
        output = self.layer_norm(gated + residual)
        return output


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        
        return output, attn_weights.mean(dim=1)
