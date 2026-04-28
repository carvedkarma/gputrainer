"""
V6 Forecaster: Temporal-MoE-Attention Multi-Head Trading Model

Architecture:
  Input (batch, seq_len, features)
    -> Feature Masking (training only, 15% random zero)
    -> Causal Conv1D Block (3 layers, residual)
    -> Learned Positional Encoding
    -> Pre-Norm Transformer Blocks x2 (4-head self-attention, causal mask)
    -> Last-token extraction
    -> Symbol Embedding concat
    -> Mixture-of-Experts Trunk (4 experts, top-2 sparse gating)
    -> Output Heads (same interface as V5 + confidence + aux_next_bar)

Backward compatible with V5: same output dict keys, same forward(x, symbol_ids) signature.
Supports 2D input (single bar) and 3D input (sequence window).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import math


class CausalConv1DLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.15):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        if self.padding > 0:
            x_padded = F.pad(x, (self.padding, 0))
        else:
            x_padded = x
        out = self.conv(x_padded)
        if out.size(-1) != residual.size(-1):
            out = out[..., :residual.size(-1)]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + residual


class CausalConv1DBlock(nn.Module):
    def __init__(self, input_dim: int, conv_channels: int, n_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.15):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(n_layers):
            out_ch = conv_channels
            layers.append(CausalConv1DLayer(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.output_dim = conv_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15, use_layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.skip(x)


class ExpertMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, dropout=dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, n_experts: int = 4, top_k: int = 2,
                 expert_hidden_dims: list = None, dropout: float = 0.15,
                 expert_dropout_p: float = 0.1):
        super().__init__()
        if expert_hidden_dims is None:
            expert_hidden_dims = [192, 128, 96]
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_dropout_p = expert_dropout_p
        self.gate = nn.Linear(input_dim, n_experts)
        self.gate_noise = nn.Linear(input_dim, n_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, expert_hidden_dims, dropout)
            for _ in range(n_experts)
        ])
        self.output_dim = expert_hidden_dims[-1]
        self._last_gate_probs = None
        self._last_hard_usage = None
        self._entropy_bonus = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)

        if self.training:
            noise_std = F.softplus(self.gate_noise(x))
            noise = noise_std * torch.randn_like(gate_logits)
            gate_logits = gate_logits + noise

            if self.expert_dropout_p > 0 and self.n_experts > self.top_k + 1:
                if torch.rand(1).item() < self.expert_dropout_p:
                    drop_idx = torch.randint(0, self.n_experts, (1,)).item()
                    gate_logits[:, drop_idx] = -1e9

        gate_probs = F.softmax(gate_logits, dim=-1)
        self._last_gate_probs = gate_probs.detach()

        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_vals = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        n = gate_probs.size(0)
        hard_usage = torch.zeros(self.n_experts, device=gate_probs.device)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                hard_usage[e] += (top_k_idx[:, k] == e).float().sum()
        fraction_hard = hard_usage / (n * self.top_k)
        self._last_hard_usage = fraction_hard.detach()

        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_idx[:, k]
            weight = top_k_vals[:, k].unsqueeze(-1)
            for e_idx in range(self.n_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask])
                    output[mask] += weight[mask] * expert_out

        load_balance_loss = self._compute_load_balance_loss(gate_probs, fraction_hard)
        return output, load_balance_loss

    def _compute_load_balance_loss(self, gate_probs: torch.Tensor,
                                   fraction_hard: torch.Tensor) -> torch.Tensor:
        fraction_soft = gate_probs.mean(dim=0)
        loss = self.n_experts * (fraction_hard.detach() * fraction_soft).sum()

        if self._entropy_bonus and self.training:
            avg_probs = gate_probs.mean(dim=0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
            max_entropy = math.log(self.n_experts)
            loss = loss + 2.0 * (max_entropy - entropy)

        return loss

    def reinit_dead_experts(self, dead_indices: list):
        if not dead_indices:
            return
        alive = [i for i in range(self.n_experts) if i not in dead_indices]
        if not alive:
            return
        best_idx = alive[0]
        if self._last_hard_usage is not None:
            best_val = -1.0
            for i in alive:
                if self._last_hard_usage[i].item() > best_val:
                    best_val = self._last_hard_usage[i].item()
                    best_idx = i
        with torch.no_grad():
            best_expert_state = self.experts[best_idx].state_dict()
            for dead_idx in dead_indices:
                self.experts[dead_idx].load_state_dict(
                    {k: v.clone() for k, v in best_expert_state.items()}
                )
                for p in self.experts[dead_idx].parameters():
                    p.add_(torch.randn_like(p) * 0.02)

                self.gate.weight[dead_idx].copy_(self.gate.weight[best_idx])
                self.gate.weight[dead_idx].add_(torch.randn_like(self.gate.weight[dead_idx]) * 0.05)
                if self.gate.bias is not None:
                    self.gate.bias[dead_idx] = self.gate.bias[best_idx] + 0.2

                self.gate_noise.weight[dead_idx].copy_(self.gate_noise.weight[best_idx])
                self.gate_noise.weight[dead_idx].add_(torch.randn_like(self.gate_noise.weight[dead_idx]) * 0.05)
                if self.gate_noise.bias is not None:
                    self.gate_noise.bias[dead_idx] = self.gate_noise.bias[best_idx]


@dataclass
class V6ForecasterConfig:
    input_dim: int = 85
    seq_len: int = 16
    conv_channels: int = 128
    n_conv_layers: int = 3
    conv_kernel_size: int = 3
    n_attn_layers: int = 2
    n_attn_heads: int = 4
    attn_ff_dim: int = 256
    n_experts: int = 4
    expert_top_k: int = 2
    expert_hidden_dims: list = None
    trunk_output_dim: int = 96
    dropout: float = 0.15
    n_symbols: int = 1
    symbol_embed_dim: int = 16
    feature_mask_ratio: float = 0.15
    enable_aux_head: bool = True
    enable_confidence_head: bool = True
    n_barrier_presets: int = 0
    enable_regime_head: bool = False

    def __post_init__(self):
        if self.expert_hidden_dims is None:
            self.expert_hidden_dims = [192, 128, 96]
        self.trunk_output_dim = self.expert_hidden_dims[-1]


class V6Forecaster(nn.Module):
    """V6 Temporal-MoE-Attention Forecaster.

    Next-generation multi-head model with causal convolutions, self-attention,
    mixture-of-experts trunk, feature masking, confidence calibration,
    and auxiliary self-supervised learning.
    """

    def __init__(self, config: V6ForecasterConfig):
        super().__init__()
        self.config = config
        self.name = "V6Forecaster"
        self.input_dim = config.input_dim
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0

        self.conv_block = CausalConv1DBlock(
            input_dim=config.input_dim,
            conv_channels=config.conv_channels,
            n_layers=config.n_conv_layers,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

        self.pos_embedding = nn.Embedding(config.seq_len, config.conv_channels)

        self.attn_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.conv_channels,
                n_heads=config.n_attn_heads,
                ff_dim=config.attn_ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.n_attn_layers)
        ])
        self.attn_norm = nn.LayerNorm(config.conv_channels)

        if config.n_symbols > 1:
            self.symbol_embedding = nn.Embedding(config.n_symbols, config.symbol_embed_dim)
            moe_input_dim = config.conv_channels + config.symbol_embed_dim
        else:
            self.symbol_embedding = None
            moe_input_dim = config.conv_channels

        self.moe = MixtureOfExperts(
            input_dim=moe_input_dim,
            n_experts=config.n_experts,
            top_k=config.expert_top_k,
            expert_hidden_dims=config.expert_hidden_dims,
            dropout=config.dropout,
        )
        trunk_dim = self.moe.output_dim

        self.ret_dist_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        self.mfe_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.mae_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.action_head = nn.Sequential(
            nn.Linear(trunk_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 3),
        )

        if config.enable_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )
        else:
            self.confidence_head = None

        if config.n_barrier_presets > 1:
            self.barrier_head = nn.Sequential(
                nn.Linear(trunk_dim, 48),
                nn.LayerNorm(48),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(48, config.n_barrier_presets),
            )
        else:
            self.barrier_head = None

        if config.enable_regime_head:
            self.regime_head = nn.Sequential(
                nn.Linear(trunk_dim, 48),
                nn.LayerNorm(48),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(48, 3),
            )
        else:
            self.regime_head = None

        if config.enable_aux_head:
            self.aux_next_bar_head = nn.Linear(trunk_dim, config.input_dim)
        else:
            self.aux_next_bar_head = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            nn.init.uniform_(self.moe.gate.bias, -0.1, 0.1)
            nn.init.xavier_uniform_(self.moe.gate.weight, gain=1.0)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def _apply_feature_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.feature_mask_ratio <= 0:
            return x
        mask = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > self.config.feature_mask_ratio
        return x * mask.float()

    def forward(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, feat_dim = x.shape

        x = self._apply_feature_mask(x)

        x = self.conv_block(x)

        positions = torch.arange(seq_len, device=x.device)
        if seq_len > self.config.seq_len:
            positions = positions[-self.config.seq_len:]
            x = x[:, -self.config.seq_len:, :]
            seq_len = self.config.seq_len
        else:
            positions = positions % self.config.seq_len
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb.unsqueeze(0)

        causal_mask = self._get_causal_mask(seq_len, x.device)
        for block in self.attn_blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.attn_norm(x)
        last_token = x[:, -1, :]

        if self.symbol_embedding is not None and symbol_ids is not None:
            sym_emb = self.symbol_embedding(symbol_ids)
            last_token = torch.cat([last_token, sym_emb], dim=-1)

        trunk_features, moe_balance_loss = self.moe(last_token)

        ret_raw = self.ret_dist_head(trunk_features)
        ret_mu = torch.clamp(ret_raw[:, 0:1], -10.0, 10.0)
        ret_log_sigma = torch.clamp(ret_raw[:, 1:2], -8.0, 2.0)
        ret_sigma = torch.exp(ret_log_sigma)

        mfe_pred = self.mfe_head(trunk_features)
        mfe_pred = torch.clamp(mfe_pred, 0.0, 20.0)

        mae_pred = self.mae_head(trunk_features)
        mae_pred = torch.clamp(mae_pred, 0.0, 20.0)

        action_logits = self.action_head(trunk_features)
        action_logits = torch.clamp(action_logits, -10.0, 10.0)

        result = {
            'ret_mu': ret_mu,
            'ret_log_sigma': ret_log_sigma,
            'ret_sigma': ret_sigma,
            'mfe': mfe_pred,
            'mae': mae_pred,
            'action_logits': action_logits,
            'moe_balance_loss': moe_balance_loss,
        }

        if self.confidence_head is not None:
            conf = self.confidence_head(trunk_features)
            conf = torch.sigmoid(conf)
            result['confidence'] = conf

        if self.barrier_head is not None:
            barrier_logits = self.barrier_head(trunk_features)
            barrier_logits = torch.clamp(barrier_logits, -10.0, 10.0)
            result['barrier_logits'] = barrier_logits

        if self.regime_head is not None:
            regime_logits = self.regime_head(trunk_features)
            regime_logits = torch.clamp(regime_logits, -10.0, 10.0)
            result['regime_logits'] = regime_logits

        if self.aux_next_bar_head is not None and self.training:
            aux_pred = self.aux_next_bar_head(trunk_features)
            result['aux_next_bar'] = aux_pred

        return result

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_expert_usage(self) -> Optional[Dict[str, float]]:
        if self.moe._last_hard_usage is None:
            return None
        return {f"E{i}": self.moe._last_hard_usage[i].item() * 100
                for i in range(self.config.n_experts)}

    def get_expert_soft_probs(self) -> Optional[Dict[str, float]]:
        if self.moe._last_gate_probs is None:
            return None
        probs = self.moe._last_gate_probs.mean(dim=0)
        return {f"E{i}": probs[i].item() * 100 for i in range(self.config.n_experts)}
