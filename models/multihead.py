"""
Multi-Head Model Architecture for Institutional Trading

Full trading output vector:
1. Classification head → direction_score [-1, 1]
2. Regression head → expected return (μ) and volatility (σ)
3. Quantile head → uncertainty quantiles (q10, q25, q50, q75, q90)
4. Trading head → entry_offset, sl_distance, tp_distance
5. Candle head → future candle deltas (Δclose, Δhigh, Δlow) for N steps

This enables complete trade planning from neural network output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, NamedTuple, List
from dataclasses import dataclass, field
import math
from .base import BaseModel, PositionalEncoding, AttentionBlock


class MultiScaleTemporalEmbedding(nn.Module):
    """
    Multi-Scale Temporal Embedding for capturing patterns at different resolutions.
    
    2024 SOTA: Instead of using a single fixed positional encoding, this learns
    separate temporal patterns at multiple scales (15m, 1h, 4h equivalents).
    
    Implementation based on:
    - "Multi-Scale Temporal Attention for Time Series" (2024)
    - "Temporal Fusion Transformers" temporal embedding approach
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 200, scales: List[int] = None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scales = scales or [1, 4, 16]  # 15m, 1h, 4h in 15m candle counts
        self.n_scales = len(self.scales)
        
        # Learned temporal embeddings per scale
        self.scale_embeddings = nn.ModuleList([
            nn.Embedding(max_seq_len, d_model // self.n_scales)
            for _ in self.scales
        ])
        
        # Scale fusion weights (learned)
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)
        
        # Project concatenated scale embeddings to d_model
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate multi-scale temporal embeddings.
        
        Returns: [seq_len, d_model] temporal embeddings
        """
        embeddings = []
        
        for scale_idx, scale in enumerate(self.scales):
            # Create scaled position indices
            positions = torch.arange(seq_len, device=device)
            scaled_positions = (positions // scale) % self.max_seq_len
            
            # Get embeddings for this scale
            scale_emb = self.scale_embeddings[scale_idx](scaled_positions)
            embeddings.append(scale_emb * self.scale_weights[scale_idx])
        
        # Concatenate and project
        combined = torch.cat(embeddings, dim=-1)
        return self.projection(combined)


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-Head Multi-Scale Attention (MHMSA) block.
    
    2024 SOTA: Extends standard attention to process information at multiple
    temporal resolutions simultaneously, then fuses the results.
    
    Key innovations:
    1. Separate attention heads for each temporal scale
    2. Cross-scale attention for capturing inter-scale dependencies
    3. Gated fusion for adaptive scale combination
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        scales: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scales = scales or [1, 4, 16]
        self.n_scales = len(self.scales)
        
        # Ensure even distribution of heads across scales
        heads_per_scale = max(1, num_heads // self.n_scales)
        
        # Per-scale attention
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads_per_scale, dropout=dropout, batch_first=True)
            for _ in self.scales
        ])
        
        # Cross-scale attention (global to combine scales)
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        
        # Gated fusion for combining scale outputs
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * self.n_scales, embed_dim),
            nn.Sigmoid()
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _downsample_for_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample sequence by taking every nth element (average pooling)."""
        if scale == 1:
            return x
        
        batch_size, seq_len, embed_dim = x.shape
        
        # Pad sequence to be divisible by scale
        pad_len = (scale - seq_len % scale) % scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape and average pool
        new_len = (seq_len + pad_len) // scale
        x = x.view(batch_size, new_len, scale, embed_dim)
        return x.mean(dim=2)
    
    def _upsample_to_original(self, x: torch.Tensor, target_len: int, scale: int) -> torch.Tensor:
        """Upsample back to original sequence length by repeating."""
        if scale == 1:
            return x[:, :target_len]
        
        # Repeat each element 'scale' times
        x = x.repeat_interleave(scale, dim=1)
        return x[:, :target_len]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Process at each scale
        scale_outputs = []
        
        for scale_idx, scale in enumerate(self.scales):
            # Downsample for this scale
            x_scaled = self._downsample_for_scale(x, scale)
            
            # Apply attention at this scale
            attn_out, _ = self.scale_attentions[scale_idx](x_scaled, x_scaled, x_scaled)
            
            # Upsample back to original length
            attn_out = self._upsample_to_original(attn_out, seq_len, scale)
            scale_outputs.append(attn_out)
        
        # Gated fusion of scale outputs
        scale_concat = torch.cat(scale_outputs, dim=-1)
        gate_weights = self.gate(scale_concat)
        
        # Weighted combination (use first scale as base, gate the rest)
        fused = scale_outputs[0]
        for i in range(1, self.n_scales):
            fused = fused + gate_weights * scale_outputs[i]
        
        # Residual connection and norm
        x = self.norm1(x + self.dropout(fused))
        
        # Cross-scale attention for global context
        cross_out, _ = self.cross_scale_attention(x, x, x, attn_mask=mask)
        x = self.norm_cross(x + self.dropout(cross_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


@dataclass
class MultiHeadOutput:
    """Output from multi-head model."""
    # Classification head: direction probabilities
    class_logits: torch.Tensor  # [batch, 3] for SHORT/HOLD/LONG
    
    # Regression head: expected return
    mu: torch.Tensor  # [batch, 1] expected forward return
    
    # Quantile head: return distribution
    quantiles: torch.Tensor  # [batch, 5] for q10, q25, q50, q75, q90
    
    # Optional: uncertainty estimate (std of predictions)
    sigma: Optional[torch.Tensor] = None  # [batch, 1]
    
    # Trading head outputs
    entry_offset: Optional[torch.Tensor] = None  # [batch, 1] entry price offset
    sl_distance: Optional[torch.Tensor] = None   # [batch, 1] stop loss distance
    tp_distance: Optional[torch.Tensor] = None   # [batch, 1] take profit distance
    
    # Candle prediction head outputs
    candle_deltas: Optional[torch.Tensor] = None  # [batch, n_steps, 3] for Δclose, Δhigh, Δlow
    
    # Flow Forecast heads (for regime-conditioned path generation)
    vol_state_logits: Optional[torch.Tensor] = None  # [batch, 3] for CONTRACTION/NEUTRAL/EXPANSION
    acceleration: Optional[torch.Tensor] = None      # [batch, 1] momentum change over horizon
    
    # Entry quality head (binary: should we enter this trade?)
    enter_logits: Optional[torch.Tensor] = None  # [batch, 1] raw logits for BCEWithLogitsLoss

    # Value head (regression: predicted E[net R])
    value_logits: Optional[torch.Tensor] = None  # [batch, 1] predicted net R expectation

    # Edge head (regression: net MFE - MAE quality proxy)
    edge_logits: Optional[torch.Tensor] = None  # [batch, 1] predicted edge quality

    # v4.6 Direction head (binary: 1=LONG better, 0=SHORT better)
    dir_logits: Optional[torch.Tensor] = None  # [batch, 1] raw logits for direction BCE

    # v4.6 HTF score head (4-class: 0=none, 1=weak, 2=moderate, 3=strong)
    htf_logits: Optional[torch.Tensor] = None  # [batch, 4] raw logits for HTF classification

    # v4.9 Distributional heads
    win_logits: Optional[torch.Tensor] = None  # [batch, 1] raw logits for p(R>0) BCE
    dist_quantiles: Optional[torch.Tensor] = None  # [batch, 3] q10/q50/q90 of realized R
    regime_logits: Optional[torch.Tensor] = None  # [batch, 3] chop/trend/highvol classification


class QuantileHead(nn.Module):
    """
    Quantile regression head using monotonic constraint.
    
    Outputs q10, q25, q50, q75, q90 with enforced ordering.
    Uses delta parameterization: each quantile = prev + softplus(delta)
    """
    
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.n_quantiles = len(self.QUANTILES)
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # q50 (median) prediction - can be any value
        self.median_head = nn.Linear(hidden_dim // 2, 1)
        
        # Lower deltas (q50 - q25, q25 - q10) - must be positive
        self.lower_deltas = nn.Linear(hidden_dim // 2, 2)
        
        # Upper deltas (q75 - q50, q90 - q75) - must be positive
        self.upper_deltas = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns quantiles [batch, 5] in order: q10, q25, q50, q75, q90
        """
        h = self.shared(x)
        
        # Median (q50) - unrestricted
        q50 = self.median_head(h)  # [batch, 1]
        
        # Lower deltas - use softplus to ensure positive
        lower_d = F.softplus(self.lower_deltas(h))  # [batch, 2]
        d_25_10 = lower_d[:, 0:1]  # q25 - q10
        d_50_25 = lower_d[:, 1:2]  # q50 - q25
        
        # Upper deltas - use softplus to ensure positive
        upper_d = F.softplus(self.upper_deltas(h))  # [batch, 2]
        d_75_50 = upper_d[:, 0:1]  # q75 - q50
        d_90_75 = upper_d[:, 1:2]  # q90 - q75
        
        # Build quantiles with monotonic ordering
        q25 = q50 - d_50_25
        q10 = q25 - d_25_10
        q75 = q50 + d_75_50
        q90 = q75 + d_90_75
        
        # Stack in order: q10, q25, q50, q75, q90
        quantiles = torch.cat([q10, q25, q50, q75, q90], dim=-1)
        
        return quantiles


class RegressionHead(nn.Module):
    """
    Regression head - ENTIRELY DISABLED (Feb 2026).
    
    CRITICAL STABILITY FIX: ENTIRE regression path removed.
    Diagnostics showed ALL regression layers causing gradient explosions:
    - regression_head.shared layers: 8-59 (exploding)
    - regression_head.sigma_head: 6-42 (exploding)
    - classifier: 0.68-1.71 (stable)
    
    Model now outputs ONLY direction classification.
    Use external ATR/rolling volatility for position sizing.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1,
                 use_log_sigma: bool = True, disable_mu: bool = True, disable_all: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            use_log_sigma: If True (default), output log(σ) directly for Phase 1b.
                          If False, use softplus for backward compatibility.
            disable_mu: If True (default), μ head is disabled - returns zeros
            disable_all: If True (default), ENTIRE regression head is disabled - all frozen
        """
        super().__init__()
        
        self.use_log_sigma = use_log_sigma
        self.disable_mu = disable_mu
        self.disable_all = disable_all
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Expected return - DISABLED
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        
        # Uncertainty - DISABLED
        self.sigma_head = nn.Linear(hidden_dim // 2, 1)
        
        # FREEZE ENTIRE REGRESSION HEAD - no gradients flow through ANY layer
        if self.disable_all:
            for param in self.parameters():
                param.requires_grad = False
        elif self.disable_mu:
            # Legacy: only freeze mu_head
            for param in self.mu_head.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, sigma_or_log_sigma) tensors.
        
        If disable_all=True: returns zeros for both (no computation, no gradients)
        If disable_mu=True: mu is zeros (no gradient flow)
        If use_log_sigma=True: returns (mu, log_sigma) where log_sigma is clamped
        If use_log_sigma=False: returns (mu, sigma) where sigma = softplus(raw) + eps
        """
        # ENTIRE REGRESSION HEAD DISABLED - Return zeros with no gradient, skip all computation
        if self.disable_all:
            mu = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            sigma = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return mu, sigma
        
        h = self.shared(x)
        
        # μ HEAD DISABLED - Return zeros with no gradient
        if self.disable_mu:
            mu = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        else:
            mu = self.mu_head(h)
        
        if self.use_log_sigma:
            log_sigma = self.sigma_head(h)
            log_sigma = log_sigma.clamp(min=-8, max=2)
            return mu, log_sigma
        else:
            sigma = F.softplus(self.sigma_head(h)) + 1e-6
            return mu, sigma


class ClassificationHead(nn.Module):
    """
    Classification head for direction (SHORT/HOLD/LONG).
    
    PHASE 2 UPGRADE: Prior probability bias initialization.
    
    Problem: With imbalanced data (e.g., 60% HOLD, 20% LONG, 20% SHORT),
    the model starts with uniform probabilities (33% each), wasting early
    training epochs re-learning the prior distribution.
    
    Solution: Initialize output layer bias = log(prior_prob), so initial
    outputs match the training label distribution. The model then focuses
    on learning deviations from prior rather than the prior itself.
    
    Reference: "Deep Learning - Goodfellow et al." Chapter 8.4
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_classes: int = 3, dropout: float = 0.1,
                 class_priors: Optional[torch.Tensor] = None):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (3 for SHORT/HOLD/LONG)
            dropout: Dropout rate
            class_priors: Optional [num_classes] tensor of prior probabilities
                          e.g., [0.20, 0.60, 0.20] for SHORT/HOLD/LONG
                          If provided, initializes output bias = log(prior)
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Build layers individually for bias initialization access
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim // 2, num_classes)
        
        # STABILITY FIX: Prior bias initialization DISABLED
        # Was causing model to collapse to one class early in training
        # Re-enable after stable training is achieved
        # if class_priors is not None:
        #     self._init_prior_bias(class_priors)
        
        # Use default Xavier initialization instead
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def _init_prior_bias(self, priors: torch.Tensor):
        """
        Initialize output layer bias based on class priors.
        
        For softmax, if we want initial P(class_i) = prior_i, we set:
            bias_i = log(prior_i)
        
        This ensures the model starts by predicting the marginal distribution,
        then learns to adjust based on features.
        """
        # Clamp priors to avoid log(0)
        priors = priors.clamp(min=1e-6)
        # Normalize to ensure they sum to 1
        priors = priors / priors.sum()
        # Compute log-prior biases
        log_prior_bias = torch.log(priors)
        
        # Set the output layer bias
        with torch.no_grad():
            self.output.bias.copy_(log_prior_bias)
        
        # Zero the output weights to start neutral
        nn.init.zeros_(self.output.weight)
        
        import logging
        logging.getLogger(__name__).info(
            f"[ClassificationHead] Prior bias initialized: "
            f"priors={priors.tolist()}, bias={log_prior_bias.tolist()}"
        )
    
    def set_class_priors(self, priors: torch.Tensor):
        """
        Update output bias based on new class priors (e.g., after computing from dataset).
        
        Call this before training to set biases from actual label distribution.
        """
        self._init_prior_bias(priors)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, num_classes]."""
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return self.output(x)


class TradingHead(nn.Module):
    """
    Trading head for entry/SL/TP distance prediction.
    
    Outputs:
    - entry_offset: Price offset from current (can be small positive/negative)
    - sl_distance: Stop loss distance (always positive, applied directionally)
    - tp_distance: Take profit distance (always positive, applied directionally)
    
    Distances are normalized by current volatility during training.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Entry offset (small, can be + or -)
        self.entry_head = nn.Linear(hidden_dim // 2, 1)
        
        # SL distance (must be positive)
        self.sl_head = nn.Linear(hidden_dim // 2, 1)
        
        # TP distance (must be positive)
        self.tp_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (entry_offset, sl_distance, tp_distance).
        
        Distances are positive, offset can be small +/-.
        """
        h = self.shared(x)
        
        # Entry offset: small value, allow +/-
        entry_offset = torch.tanh(self.entry_head(h)) * 0.01  # Max 1% offset
        
        # SL distance: positive, typically 0.5% - 5%
        sl_distance = F.softplus(self.sl_head(h)) * 0.01 + 0.003  # Min 0.3%, scale to typical
        
        # TP distance: positive, typically 1% - 10%
        tp_distance = F.softplus(self.tp_head(h)) * 0.02 + 0.005  # Min 0.5%, scale to typical
        
        return entry_offset, sl_distance, tp_distance


class CandlePredictionHead(nn.Module):
    """
    Predicts future candle deltas (NOT raw prices).
    
    For each future step, predicts:
    - Δclose: Close price change from current
    - Δhigh: High price change from current  
    - Δlow: Low price change from current
    
    Uses percentage changes for scale invariance.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_future_steps: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.n_steps = n_future_steps
        self.n_outputs_per_step = 3  # Δclose, Δhigh, Δlow
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each future step (more capacity)
        self.step_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.n_outputs_per_step)
            for _ in range(n_future_steps)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns candle deltas [batch, n_steps, 3].
        
        Output[:, i, 0] = Δclose for step i
        Output[:, i, 1] = Δhigh for step i
        Output[:, i, 2] = Δlow for step i
        
        Values are percentage changes (e.g., 0.01 = 1% move).
        """
        h = self.shared(x)
        
        step_outputs = []
        for step_head in self.step_heads:
            step_pred = step_head(h)  # [batch, 3]
            step_outputs.append(step_pred)
            
        # Stack: [batch, n_steps, 3]
        candle_deltas = torch.stack(step_outputs, dim=1)
        
        # Scale to reasonable range (typically -10% to +10%)
        candle_deltas = torch.tanh(candle_deltas) * 0.10
        
        return candle_deltas


class VolStateHead(nn.Module):
    """
    Volatility State Classification Head for Flow Forecast.
    
    Predicts 3-class volatility regime:
    - 0: CONTRACTION (forward_vol/current_vol < 0.9)
    - 1: NEUTRAL (0.9 <= ratio <= 1.1)
    - 2: EXPANSION (ratio > 1.1)
    
    Used to shape alpha parameter in quantile path generation.
    """
    
    VOL_STATE_NAMES = ["CONTRACTION", "NEUTRAL", "EXPANSION"]
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, 3] for CONTRACTION/NEUTRAL/EXPANSION."""
        return self.classifier(x)


class AccelerationHead(nn.Module):
    """
    Acceleration Prediction Head for Flow Forecast.
    
    Predicts momentum change over horizon period:
    acceleration = momentum_forward - momentum_now
    
    Where momentum = 4-bar return (for 15m, this is 1 hour momentum).
    
    Positive acceleration = momentum increasing (trend strengthening)
    Negative acceleration = momentum decreasing (trend weakening)
    
    Used for path shaping - adjusts how quickly paths diverge.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns acceleration prediction [batch, 1].
        
        Output is unbounded momentum change (typically -0.05 to +0.05).
        Positive = momentum increasing, negative = momentum decreasing.
        """
        # Clamp output to reasonable range (±10% momentum change)
        raw_accel = self.predictor(x)
        return torch.tanh(raw_accel) * 0.10


class ConstrainedCandleHead(nn.Module):
    """
    PHASE 2: Constrained candle parameterization.
    
    Problem: Raw high/low predictions can violate high >= low constraint.
    Solution: Predict (Δclose, log_range, skew) and reconstruct valid candles.
    
    Parameters:
    - Δclose: Close price change from current (unbounded, scaled by tanh)
    - log_range: log(high - low), always positive after exp()
    - skew ∈ [-1, 1]: Where close sits within the range (0 = middle)
    
    Reconstruction:
        range = exp(log_range)  # Always positive
        high = close + range * (0.5 + 0.5 * skew)  # Upper portion
        low = close - range * (0.5 - 0.5 * skew)   # Lower portion
    
    This GUARANTEES high >= low for all predictions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_future_steps: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.n_steps = n_future_steps
        self.n_raw_outputs = 3  # Δclose, log_range, skew
        self.n_reconstructed_outputs = 3  # Δclose, Δhigh, Δlow (for compatibility)
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each future step
        self.step_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.n_raw_outputs)
            for _ in range(n_future_steps)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstructed candle deltas [batch, n_steps, 3].
        
        Output[:, i, 0] = Δclose for step i (percentage change)
        Output[:, i, 1] = Δhigh for step i (reconstructed, >= Δlow guaranteed)
        Output[:, i, 2] = Δlow for step i (reconstructed, <= Δhigh guaranteed)
        
        Internally predicts (Δclose, log_range, skew) and reconstructs.
        """
        h = self.shared(x)
        
        reconstructed_candles = []
        for step_head in self.step_heads:
            raw = step_head(h)  # [batch, 3]: (Δclose_raw, log_range_raw, skew_raw)
            
            # Extract components
            delta_close_raw = raw[:, 0]   # Unbounded
            log_range_raw = raw[:, 1]     # Unbounded (will exp())
            skew_raw = raw[:, 2]          # Unbounded (will tanh())
            
            # Constrain outputs
            # Δclose: scale to ±10% range
            delta_close = torch.tanh(delta_close_raw) * 0.10
            
            # log_range: clip to prevent explosion, then exp() for positive range
            # Typical range: exp(-5) ≈ 0.007 to exp(-1) ≈ 0.37 (0.7% to 37% range)
            log_range = torch.clamp(log_range_raw, -8, 0)  # Outputs 0.03% to 100% range
            range_val = torch.exp(log_range) * 0.10  # Scale to reasonable %
            
            # skew: constrain to [-1, 1] via tanh
            skew = torch.tanh(skew_raw)
            
            # Reconstruct high/low from close, range, and skew
            # skew = 0: close in middle (high = close + range/2, low = close - range/2)
            # skew = 1: close at low (high = close + range, low = close)
            # skew = -1: close at high (high = close, low = close - range)
            delta_high = delta_close + range_val * (0.5 + 0.5 * skew)
            delta_low = delta_close - range_val * (0.5 - 0.5 * skew)
            
            # Stack [batch, 3]
            step_candle = torch.stack([delta_close, delta_high, delta_low], dim=1)
            reconstructed_candles.append(step_candle)
        
        # Stack all steps: [batch, n_steps, 3]
        return torch.stack(reconstructed_candles, dim=1)
    
    def forward_with_params(self, x: torch.Tensor) -> dict:
        """
        Alternative forward that returns both raw params and reconstructed candles.
        Useful for debugging and monitoring the constrained parameterization.
        
        Returns dict with:
        - 'candle_deltas': [batch, n_steps, 3] - reconstructed Δclose, Δhigh, Δlow
        - 'raw_params': [batch, n_steps, 3] - Δclose, log_range, skew (before reconstruction)
        """
        h = self.shared(x)
        
        reconstructed_candles = []
        raw_params = []
        
        for step_head in self.step_heads:
            raw = step_head(h)  # [batch, 3]
            
            delta_close_raw = raw[:, 0]
            log_range_raw = raw[:, 1]
            skew_raw = raw[:, 2]
            
            delta_close = torch.tanh(delta_close_raw) * 0.10
            log_range = torch.clamp(log_range_raw, -8, 0)
            range_val = torch.exp(log_range) * 0.10
            skew = torch.tanh(skew_raw)
            
            delta_high = delta_close + range_val * (0.5 + 0.5 * skew)
            delta_low = delta_close - range_val * (0.5 - 0.5 * skew)
            
            step_candle = torch.stack([delta_close, delta_high, delta_low], dim=1)
            step_params = torch.stack([delta_close, log_range, skew], dim=1)
            
            reconstructed_candles.append(step_candle)
            raw_params.append(step_params)
        
        return {
            'candle_deltas': torch.stack(reconstructed_candles, dim=1),
            'raw_params': torch.stack(raw_params, dim=1)
        }


class MultiHeadTransformer(BaseModel):
    """
    Transformer with multi-head output for institutional trading.
    
    Outputs:
    - Classification: direction probabilities
    - Regression: expected return (μ) and uncertainty (σ)
    - Quantiles: q10, q25, q50, q75, q90 for SL/TP derivation
    - Trading: entry_offset, sl_distance, tp_distance
    - Candles: future candle deltas (Δclose, Δhigh, Δlow)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_transformer", input_dim, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        
        # Shared encoder backbone
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-head outputs (8 heads total)
        self.class_head = ClassificationHead(d_model, d_model // 2, num_classes, dropout)
        self.regression_head = RegressionHead(d_model, d_model // 2, dropout, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(d_model, d_model // 2, dropout)
        self.trading_head = TradingHead(d_model, d_model // 2, dropout)
        self.candle_head = CandlePredictionHead(d_model, d_model // 2, n_future_candles, dropout)
        
        # Flow Forecast heads (for regime-conditioned path generation)
        self.vol_state_head = VolStateHead(d_model, d_model // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(d_model, d_model // 2, dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared encoder - returns pooled representation."""
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for block in self.attention_blocks:
            x = block(x, mask)
            
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - returns class logits for backward compatibility.
        Use forward_multihead() for full multi-head output.
        """
        features = self.encode(x, mask)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MultiHeadOutput:
        """
        Full multi-head forward pass.
        
        Returns MultiHeadOutput with all heads:
        - class_logits: [batch, 3]
        - mu: [batch, 1]
        - sigma: [batch, 1]
        - quantiles: [batch, 5]
        - entry_offset, sl_distance, tp_distance: [batch, 1] each
        - candle_deltas: [batch, n_steps, 3]
        - vol_state_logits: [batch, 3] for flow forecast
        - acceleration: [batch, 1] for flow forecast
        """
        features = self.encode(x, mask)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        # Flow Forecast heads
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )
    
    def predict_with_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference.
        
        Returns dict with full trading output:
        - probabilities: softmax of class logits
        - direction: argmax of probabilities
        - mu: expected return
        - sigma: uncertainty
        - quantiles: {q10, q25, q50, q75, q90}
        - trading: {entry_offset, sl_distance, tp_distance}
        - candle_deltas: future candle predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward_multihead(x)
            
            probs = F.softmax(output.class_logits, dim=-1)
            direction = torch.argmax(probs, dim=-1)
            
            return {
                'probabilities': probs,
                'direction': direction,
                'mu': output.mu,
                'sigma': output.sigma,
                'q10': output.quantiles[:, 0:1],
                'q25': output.quantiles[:, 1:2],
                'q50': output.quantiles[:, 2:3],
                'q75': output.quantiles[:, 3:4],
                'q90': output.quantiles[:, 4:5],
                'entry_offset': output.entry_offset,
                'sl_distance': output.sl_distance,
                'tp_distance': output.tp_distance,
                'candle_deltas': output.candle_deltas,
            }


class MultiScaleTransformer(BaseModel):
    """
    Multi-Scale Temporal Transformer with MHMSA for Institutional Trading.
    
    2024 SOTA Architecture: Extends the standard transformer with multi-scale
    temporal attention that captures patterns at 15m, 1h, and 4h resolutions
    simultaneously using learned temporal embeddings and gated fusion.
    
    Key improvements over standard transformer:
    1. Multi-scale temporal embeddings instead of fixed positional encoding
    2. MHMSA blocks that process at multiple resolutions with cross-scale attention
    3. Better capture of both short-term momentum and longer-term trends
    
    Use this when you have single-timeframe data but want to capture
    multi-resolution patterns implicitly.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,  # Fewer layers since MHMSA is more expressive
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        scales: List[int] = None,
        use_log_sigma: bool = True
    ):
        super().__init__("multihead_multiscale", input_dim, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        self.scales = scales or [1, 4, 16]  # 15m, 1h, 4h equivalents
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Multi-scale temporal embedding (replaces standard pos encoding)
        self.temporal_embedding = MultiScaleTemporalEmbedding(
            d_model, max_seq_len, self.scales
        )
        
        # Multi-Scale Attention blocks
        self.msa_blocks = nn.ModuleList([
            MultiScaleAttentionBlock(d_model, nhead, dim_feedforward, self.scales, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-head outputs (same as standard transformer)
        self.class_head = ClassificationHead(d_model, d_model // 2, num_classes, dropout)
        self.regression_head = RegressionHead(d_model, d_model // 2, dropout, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(d_model, d_model // 2, dropout)
        self.trading_head = TradingHead(d_model, d_model // 2, dropout)
        self.candle_head = CandlePredictionHead(d_model, d_model // 2, n_future_candles, dropout)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(d_model, d_model // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(d_model, d_model // 2, dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared encoder with multi-scale attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add multi-scale temporal embeddings
        temporal_emb = self.temporal_embedding(seq_len, x.device)
        x = x + temporal_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply multi-scale attention blocks
        for block in self.msa_blocks:
            x = block(x, mask)
            
        # Global pooling
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - returns class logits for backward compatibility."""
        features = self.encode(x, mask)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MultiHeadOutput:
        """Full multi-head forward pass with all prediction heads."""
        features = self.encode(x, mask)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        # Flow Forecast heads
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )
    
    def predict_with_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference with full trading output."""
        self.eval()
        with torch.no_grad():
            output = self.forward_multihead(x)
            
            probs = F.softmax(output.class_logits, dim=-1)
            direction = torch.argmax(probs, dim=-1)
            
            return {
                'probabilities': probs,
                'direction': direction,
                'mu': output.mu,
                'sigma': output.sigma,
                'q10': output.quantiles[:, 0:1],
                'q25': output.quantiles[:, 1:2],
                'q50': output.quantiles[:, 2:3],
                'q75': output.quantiles[:, 3:4],
                'q90': output.quantiles[:, 4:5],
                'entry_offset': output.entry_offset,
                'sl_distance': output.sl_distance,
                'tp_distance': output.tp_distance,
                'candle_deltas': output.candle_deltas,
            }


class MultiHeadLSTM(BaseModel):
    """
    Bidirectional LSTM with multi-head output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_lstm", input_dim, num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output dimension is 2x hidden due to bidirectional
        encoder_dim = hidden_dim * 2
        
        # Multi-head outputs (8 heads total)
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        self.trading_head = TradingHead(encoder_dim, encoder_dim // 2, dropout)
        self.candle_head = CandlePredictionHead(encoder_dim, encoder_dim // 2, n_future_candles, dropout)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(encoder_dim, encoder_dim // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(encoder_dim, encoder_dim // 2, dropout)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final hidden state from LSTM."""
        output, (h_n, c_n) = self.lstm(x)
        
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        features = torch.cat([h_forward, h_backward], dim=-1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )


class MultiHeadCNN(BaseModel):
    """
    1D CNN (ResNet-style) with multi-head output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_cnn", input_dim, num_classes)
        
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        
        # Initial projection
        self.input_conv = nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        
        # Residual blocks with increasing channels
        self.blocks = nn.ModuleList()
        in_ch = hidden_channels
        for i in range(num_blocks):
            out_ch = hidden_channels * (2 ** min(i, 2))  # Cap at 4x
            self.blocks.append(self._make_block(in_ch, out_ch, dropout))
            in_ch = out_ch
            
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Encoder output dimension
        encoder_dim = in_ch
        
        # Multi-head outputs (8 heads total)
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        self.trading_head = TradingHead(encoder_dim, encoder_dim // 2, dropout)
        self.candle_head = CandlePredictionHead(encoder_dim, encoder_dim // 2, n_future_candles, dropout)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(encoder_dim, encoder_dim // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(encoder_dim, encoder_dim // 2, dropout)
        
    def _make_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, seq_len, features]
        Output: [batch, encoder_dim]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x) + F.interpolate(x, size=x.size(-1)) if x.size(1) == block[0].out_channels else block(x)
            
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )


class MultiHeadGNN(BaseModel):
    """
    Graph Neural Network with multi-head output for cross-asset trading.
    Based on CrossAssetGNN architecture with temporal encoding + graph attention.
    Handles 3D input (batch, seq, features) for training compatibility.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_assets: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_gnn", input_dim, num_classes)
        
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        
        # Temporal encoder (from CrossAssetGNN)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Node encoder for graph structure (from CrossAssetGNN)
        features_per_asset = max(input_dim // num_assets, 1)
        self.node_encoder = nn.Sequential(
            nn.Linear(features_per_asset, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers (simplified for 3D input)
        self.graph_attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.graph_attention_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            
        # Edge predictor for adjacency (from CrossAssetGNN)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Multi-head outputs (8 heads total)
        self.class_head = ClassificationHead(hidden_dim, num_classes)
        self.regression_head = RegressionHead(hidden_dim, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(hidden_dim, num_quantiles)
        self.trading_head = TradingHead(hidden_dim)
        self.candle_head = CandlePredictionHead(hidden_dim, n_future_steps=n_future_candles)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(hidden_dim, hidden_dim // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(hidden_dim, hidden_dim // 2, dropout)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using temporal + attention encoding."""
        # x: (batch, seq, features)
        batch_size, seq_len, _ = x.shape
        
        # Temporal encoding
        h = self.temporal_encoder(x)
        h, _ = self.temporal_lstm(h)
        
        # Apply graph attention layers (self-attention across sequence)
        for attn_layer in self.graph_attention_layers:
            h_attn, _ = attn_layer(h, h, h)
            h = h + h_attn  # Residual connection
        
        # Take last hidden state
        h = h[:, -1, :]  # (batch, hidden)
        
        return h
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )


class MultiHeadVAE(BaseModel):
    """
    Variational Autoencoder with multi-head output for regime detection.
    Based on MarketVAE architecture with encoder/decoder + latent space.
    Uses latent representations for multi-head predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 25,
        latent_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_vae", input_dim, num_classes)
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_quantiles = num_quantiles
        self.use_log_sigma = use_log_sigma
        self.n_future_candles = n_future_candles
        
        # Encoder (from MarketVAE)
        encoder_layers = []
        in_features = input_dim * sequence_length
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (from MarketVAE)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (from MarketVAE) - for reconstruction loss if needed
        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim * sequence_length))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Multi-head outputs (8 heads total, from latent space)
        self.class_head = ClassificationHead(latent_dim, num_classes)
        self.regression_head = RegressionHead(latent_dim, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(latent_dim, num_quantiles)
        self.trading_head = TradingHead(latent_dim)
        self.candle_head = CandlePredictionHead(latent_dim, n_future_steps=n_future_candles)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(latent_dim, latent_dim // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(latent_dim, latent_dim // 2, dropout)
    
    def encode_to_latent(self, x: torch.Tensor) -> tuple:
        """Encode input to latent distribution parameters (mu, log_var)."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, self.sequence_length, self.input_dim)
        return x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features from input using reparameterization."""
        mu, log_var = self.encode_to_latent(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )
    
    def forward_with_reconstruction(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both multi-head outputs and reconstruction for VAE loss."""
        mu_latent, log_var = self.encode_to_latent(x)
        z = self.reparameterize(mu_latent, log_var)
        x_recon = self.decode(z)
        
        # Multi-head outputs from latent
        class_logits = self.class_head(z)
        mu, sigma = self.regression_head(z)
        quantiles = self.quantile_head(z)
        entry_offset, sl_distance, tp_distance = self.trading_head(z)
        candle_deltas = self.candle_head(z)
        
        multihead_output = MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )
        
        return multihead_output, x_recon, mu_latent, log_var


# Import TFT components from transformer module
from .transformer import GatedResidualNetwork, InterpretableMultiHeadAttention


class MultiHeadTFT(BaseModel):
    """
    Temporal Fusion Transformer with multi-head output for institutional trading.
    
    TFT-inspired architecture combining:
    - Bidirectional LSTM for temporal encoding (captures sequential patterns)
    - Gated residual networks (GRN) for non-linear processing with skip connections
    - Static context encoder for aggregated sequence representation  
    - Interpretable multi-head attention for temporal pattern recognition
    - Multi-head outputs: classification, regression, quantile, trading, candle prediction
    
    Follows Google's TFT paper core concepts: https://arxiv.org/abs/1912.09363
    Simplified for raw feature input without per-variable embedding.
    Key TFT components: GRN gating, static enrichment, interpretable attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5,
        use_log_sigma: bool = True  # PHASE 1b: Enable log-sigma by default
    ):
        super().__init__("multihead_tft", input_dim, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_encoder_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        self.use_log_sigma = use_log_sigma
        
        # Static context encoder (processes aggregated sequence features)
        self.static_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Temporal encoder (LSTM for sequence processing)
        self.temporal_encoder = nn.LSTM(
            input_dim, d_model // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Gated residual networks for static enrichment
        self.static_enrichment_grn = GatedResidualNetwork(d_model, d_model, dropout, context_dim=d_model)
        
        # Interpretable multi-head attention layers
        self.attention_blocks = nn.ModuleList([
            InterpretableMultiHeadAttention(d_model, nhead, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Post-attention GRNs
        self.post_attention_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        
        # Multi-head outputs (8 heads total)
        self.class_head = ClassificationHead(d_model, d_model // 2, num_classes, dropout)
        self.regression_head = RegressionHead(d_model, d_model // 2, dropout, use_log_sigma=use_log_sigma)
        self.quantile_head = QuantileHead(d_model, d_model // 2, dropout)
        self.trading_head = TradingHead(d_model, d_model // 2, dropout)
        self.candle_head = CandlePredictionHead(d_model, d_model // 2, n_future_candles, dropout)
        
        # Flow Forecast heads
        self.vol_state_head = VolStateHead(d_model, d_model // 2, 3, dropout)
        self.acceleration_head = AccelerationHead(d_model, d_model // 2, dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        TFT-style encoding: LSTM + static enrichment + attention.
        Returns pooled representation for multi-head outputs.
        """
        # Temporal encoding via bidirectional LSTM
        lstm_out, _ = self.temporal_encoder(x)  # [batch, seq, d_model]
        
        # Static context from aggregated features
        static_features = self.static_encoder(x.mean(dim=1))  # [batch, d_model]
        
        # Static enrichment via GRN
        # Expand static to match sequence length
        static_expanded = static_features.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        enriched = self.static_enrichment_grn(lstm_out, static_expanded)
        
        # Positional encoding
        enriched = enriched.transpose(0, 1)
        enriched = self.pos_encoding(enriched)
        enriched = enriched.transpose(0, 1)
        
        # Interpretable attention layers
        for attn_block, grn in zip(self.attention_blocks, self.post_attention_grns):
            attn_out, _ = attn_block(enriched)
            enriched = grn(attn_out + enriched)
        
        # Take final timestep representation (like TFT)
        output_repr = enriched[:, -1, :]  # [batch, d_model]
        
        return output_repr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns class logits for backward compatibility.
        Use forward_multihead() for full multi-head output.
        """
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """
        Full multi-head forward pass.
        
        Returns MultiHeadOutput with all heads:
        - class_logits: [batch, 3]
        - mu: [batch, 1]
        - sigma: [batch, 1]
        - quantiles: [batch, 5]
        - entry_offset, sl_distance, tp_distance: [batch, 1] each
        - candle_deltas: [batch, n_steps, 3]
        - vol_state_logits: [batch, 3] for flow forecast
        - acceleration: [batch, 1] for flow forecast
        """
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        vol_state_logits = self.vol_state_head(features)
        acceleration = self.acceleration_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration
        )
    
    def predict_with_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference.
        Returns dict with full trading output.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward_multihead(x)
            
            probs = F.softmax(output.class_logits, dim=-1)
            direction = torch.argmax(probs, dim=-1)
            
            return {
                'probabilities': probs,
                'direction': direction,
                'mu': output.mu,
                'sigma': output.sigma,
                'q10': output.quantiles[:, 0:1],
                'q25': output.quantiles[:, 1:2],
                'q50': output.quantiles[:, 2:3],
                'q75': output.quantiles[:, 3:4],
                'q90': output.quantiles[:, 4:5],
                'entry_offset': output.entry_offset,
                'sl_distance': output.sl_distance,
                'tp_distance': output.tp_distance,
                'candle_deltas': output.candle_deltas,
                'vol_state_logits': output.vol_state_logits,
                'acceleration': output.acceleration,
            }
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns attention weights from all attention layers for interpretability.
        TFT's interpretable attention allows understanding which time steps matter.
        """
        # Temporal encoding via bidirectional LSTM
        lstm_out, _ = self.temporal_encoder(x)
        
        # Static context
        static_features = self.static_encoder(x.mean(dim=1))
        static_expanded = static_features.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        enriched = self.static_enrichment_grn(lstm_out, static_expanded)
        
        # Positional encoding
        enriched = enriched.transpose(0, 1)
        enriched = self.pos_encoding(enriched)
        enriched = enriched.transpose(0, 1)
        
        # Collect attention weights
        all_weights = []
        for attn_block, grn in zip(self.attention_blocks, self.post_attention_grns):
            attn_out, weights = attn_block(enriched)
            all_weights.append(weights)
            enriched = grn(attn_out + enriched)
        
        return all_weights


# Factory function to get multi-head model
def get_multihead_model(
    architecture: str,
    input_dim: int,
    **kwargs
) -> BaseModel:
    """
    Factory function to create multi-head models.
    
    Args:
        architecture: One of 'transformer', 'tft', 'lstm', 'cnn', 'gnn', 'vae'
        input_dim: Number of input features
        **kwargs: Architecture-specific arguments
        
    Returns:
        Multi-head model instance
    """
    models = {
        'transformer': MultiHeadTransformer,
        'multiscale': MultiScaleTransformer,
        'multiscale_transformer': MultiScaleTransformer,
        'tft': MultiHeadTFT,
        'lstm': MultiHeadLSTM,
        'cnn': MultiHeadCNN,
        'gnn': MultiHeadGNN,
        'vae': MultiHeadVAE,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    return models[architecture](input_dim=input_dim, **kwargs)
