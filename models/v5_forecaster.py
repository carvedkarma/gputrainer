"""
V5 Forecaster: Separating Market Forecasting from Decision Layer

Architecture:
- Shared trunk (ResidualBlock-based MLP, same as EnhancedMultiHeadMLP)
- ret_dist_head: predicts Normal(mu, sigma) for ret_h distribution
- mfe_head: regression for max favorable excursion (Huber)
- mae_head: regression for max adverse excursion (Huber)
- action_head: 3-class logits {HOLD=0, LONG=1, SHORT=2}
- barrier_head: N-class logits for preset selection (optional)
- regime_head: 3-class {chop, trend, highvol} (optional)

All outputs produced in one forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3, use_layer_norm: bool = True):
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


class TemporalConvBlock(nn.Module):
    """Lightweight Conv1D temporal encoder for V5Forecaster.

    Takes (batch, seq_len, n_features) input, applies three Conv1d layers
    with GELU + LayerNorm, then projects the last-timestep representation
    back to n_features with a residual bypass.  Output shape: (batch, n_features).
    This keeps trunk_input_dim unchanged so old checkpoints remain compatible.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=0)
        self.ln1 = nn.LayerNorm(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        self.ln2 = nn.LayerNorm(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=0)
        self.ln3 = nn.LayerNorm(64)
        self.out_proj = nn.Linear(64, n_features)
        self.residual_proj = nn.Linear(n_features, n_features)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)
        h = self.act(self.ln1(self.conv1(F.pad(h, (2, 0))).permute(0, 2, 1))).permute(0, 2, 1)
        h = self.act(self.ln2(self.conv2(F.pad(h, (2, 0))).permute(0, 2, 1))).permute(0, 2, 1)
        h = self.act(self.ln3(self.conv3(F.pad(h, (2, 0))).permute(0, 2, 1))).permute(0, 2, 1)
        h_last = h[:, :, -1]
        out = self.out_proj(h_last)
        res = self.residual_proj(x[:, -1, :])
        return out + res


@dataclass
class V5ForecasterConfig:
    input_dim: int = 63
    hidden_dims: list = None
    dropout: float = 0.3
    use_layer_norm: bool = True
    use_residual: bool = True
    n_barrier_presets: int = 0
    enable_regime_head: bool = False
    n_symbols: int = 1
    symbol_embed_dim: int = 8
    n_features: int = None  # Alias for input_dim — accepted for back-compat with old configs
    use_temporal: bool = True  # Enable Conv1D temporal block for sequence input
    # Bias initialization for MAE/MFE regression heads.  Default 0.0 preserves existing
    # behavior.  Setting to a small positive value (e.g. 1.0–2.0) lifts the head off the
    # dead-gradient region of clamp(0, 20) so trunk features that get pushed into a
    # regime with negative pre-activations (e.g. SHORT specialist runs with strong KL
    # pressure on bear bars) do not collapse mae_pred / mfe_pred to 0 across all bars.
    mae_head_init_bias: float = 0.0
    mfe_head_init_bias: float = 0.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]
        # n_features is a legacy alias for input_dim.  If both are explicitly set to
        # different non-default values, the caller has a config conflict.
        if self.n_features is not None:
            default_input_dim = 63
            if self.input_dim != default_input_dim and self.input_dim != self.n_features:
                raise ValueError(
                    f"V5ForecasterConfig: n_features={self.n_features} and "
                    f"input_dim={self.input_dim} are both set to different non-default values. "
                    f"Use only one."
                )
            self.input_dim = self.n_features


class V5Forecaster(nn.Module):
    """V5 multi-head forecaster separating market state from decision."""

    def __init__(self, config: V5ForecasterConfig):
        super().__init__()
        self.config = config
        self.name = "V5Forecaster"

        self.input_dim = config.input_dim
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0

        if config.use_temporal:
            self.temporal_block = TemporalConvBlock(config.input_dim)
        else:
            self.temporal_block = None

        if config.n_symbols > 1:
            self.symbol_embedding = nn.Embedding(config.n_symbols, config.symbol_embed_dim)
            trunk_input_dim = config.input_dim + config.symbol_embed_dim
        else:
            self.symbol_embedding = None
            trunk_input_dim = config.input_dim

        trunk_layers = []
        prev_dim = trunk_input_dim
        for hidden_dim in config.hidden_dims:
            if config.use_residual:
                trunk_layers.append(ResidualBlock(
                    prev_dim, hidden_dim,
                    dropout=config.dropout,
                    use_layer_norm=config.use_layer_norm
                ))
            else:
                trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
                if config.use_layer_norm:
                    trunk_layers.append(nn.LayerNorm(hidden_dim))
                trunk_layers.append(nn.GELU())
                trunk_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_dim = prev_dim

        self.ret_dist_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        self.mfe_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.mae_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.action_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, 3)
        )

        if config.n_barrier_presets > 1:
            self.barrier_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, config.n_barrier_presets)
            )
        else:
            self.barrier_head = None

        if config.enable_regime_head:
            self.regime_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 3)
            )
        else:
            self.regime_head = None

        self.confidence_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for head in [self.ret_dist_head, self.mfe_head, self.mae_head]:
            final = list(head.children())[-1]
            if isinstance(final, nn.Linear):
                nn.init.normal_(final.weight, std=0.01)
                nn.init.zeros_(final.bias)
        # Apply configurable positive bias to MAE/MFE heads to escape the
        # clamp(0, 20) dead-gradient trap when trunk features get pushed
        # negative (observed in SHORT specialist runs where mae_pred = 0 across
        # all folds).  Single-element bias is broadcast across the batch.
        if self.config.mae_head_init_bias != 0.0:
            mae_final = list(self.mae_head.children())[-1]
            if isinstance(mae_final, nn.Linear):
                nn.init.constant_(mae_final.bias, float(self.config.mae_head_init_bias))
        if self.config.mfe_head_init_bias != 0.0:
            mfe_final = list(self.mfe_head.children())[-1]
            if isinstance(mfe_final, nn.Linear):
                nn.init.constant_(mfe_final.bias, float(self.config.mfe_head_init_bias))

    def forward(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            if self.temporal_block is not None:
                x = self.temporal_block(x)
            else:
                x = x[:, -1, :]

        if self.symbol_embedding is not None and symbol_ids is not None:
            sym_emb = self.symbol_embedding(symbol_ids)
            x = torch.cat([x, sym_emb], dim=-1)

        features = self.trunk(x)

        ret_raw = self.ret_dist_head(features)
        ret_mu = torch.clamp(ret_raw[:, 0:1], -10.0, 10.0)
        ret_log_sigma = torch.clamp(ret_raw[:, 1:2], -8.0, 2.0)
        ret_sigma = torch.exp(ret_log_sigma)

        mfe_pred = self.mfe_head(features)
        mfe_pred = torch.clamp(mfe_pred, 0.0, 20.0)

        mae_pred = self.mae_head(features)
        mae_pred = torch.clamp(mae_pred, 0.0, 20.0)

        action_logits = self.action_head(features)
        action_logits = torch.clamp(action_logits, -10.0, 10.0)

        result = {
            'ret_mu': ret_mu,
            'ret_log_sigma': ret_log_sigma,
            'ret_sigma': ret_sigma,
            'mfe': mfe_pred,
            'mae': mae_pred,
            'action_logits': action_logits,
        }

        if self.barrier_head is not None:
            barrier_logits = self.barrier_head(features)
            barrier_logits = torch.clamp(barrier_logits, -10.0, 10.0)
            result['barrier_logits'] = barrier_logits

        if self.regime_head is not None:
            regime_logits = self.regime_head(features)
            regime_logits = torch.clamp(regime_logits, -10.0, 10.0)
            result['regime_logits'] = regime_logits

        result['confidence'] = torch.sigmoid(self.confidence_head(features))

        return result

    @property
    def model(self):
        """Backward-compat accessor: model.model.symbol_embed.weight resolves correctly.

        Returns self so that the legacy path model.model.symbol_embed.weight works without
        registering a self-referential child module (which would break PyTorch traversal).
        Implemented as a @property so PyTorch's __setattr__ never sees it and _modules stays clean.
        """
        return self

    @property
    def symbol_embed(self):
        """Backward-compat alias for self.symbol_embedding.

        Tests access model.model.symbol_embed.weight; implemented as a @property so the
        nn.Embedding is not registered twice in _modules and the state_dict stays clean.
        """
        return self.symbol_embedding

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
