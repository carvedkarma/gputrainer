"""
Regime Detection and Mixture-of-Experts

Implements institutional-grade regime-aware trading:
1. Regime Detection: Classify market state (trend, range, high-vol, low-vol)
2. Expert Models: Specialized models for each regime
3. Gating Network: Learns to weight experts based on current regime
4. Ensemble: Combines expert predictions with learned weights

"This is how you get both aggressive and accurate."
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current market regime state."""
    regime_id: int
    regime_name: str
    confidence: float
    features: Dict[str, float]


class RegimeDetector:
    """
    Detects market regime from price data.
    
    Regimes:
    0 - TREND_UP: Strong upward momentum
    1 - TREND_DOWN: Strong downward momentum
    2 - RANGE_BOUND: Low directional movement
    3 - HIGH_VOLATILITY: High volatility, choppy
    4 - BREAKOUT: Transitional state
    5 - CHAOS: News-driven, unpredictable (often "don't trade")
    """
    
    REGIMES = {
        0: "TREND_UP",
        1: "TREND_DOWN", 
        2: "RANGE_BOUND",
        3: "HIGH_VOLATILITY",
        4: "BREAKOUT",
        5: "CHAOS"
    }
    
    def __init__(self,
                 trend_threshold: float = 0.02,
                 volatility_lookback: int = 48,
                 trend_lookback: int = 96):
        self.trend_threshold = trend_threshold
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features used for regime detection.
        
        Returns DataFrame with columns:
        - trend_strength: Directional movement strength
        - volatility_level: Current vs historical volatility
        - range_ratio: (high-low)/close ratio
        - momentum: Rate of change
        - volume_spike: Volume vs average
        - price_position: Position within recent range
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        log_returns = np.log(close / close.shift(1))
        volatility = log_returns.rolling(self.volatility_lookback).std()
        avg_volatility = volatility.rolling(self.volatility_lookback * 4).mean()
        volatility_level = volatility / avg_volatility.clip(lower=1e-8)
        
        momentum = close.pct_change(self.trend_lookback)
        trend_strength = momentum / volatility.clip(lower=1e-8)
        
        atr = (high - low).rolling(self.volatility_lookback).mean()
        range_ratio = atr / close
        
        rolling_high = high.rolling(self.trend_lookback).max()
        rolling_low = low.rolling(self.trend_lookback).min()
        price_position = (close - rolling_low) / (rolling_high - rolling_low + 1e-8)
        
        avg_volume = volume.rolling(self.volatility_lookback * 4).mean()
        volume_spike = volume / avg_volume.clip(lower=1)
        
        recent_high = high.rolling(self.trend_lookback // 2).max()
        recent_low = low.rolling(self.trend_lookback // 2).min()
        breakout_up = (close > recent_high.shift(1)).astype(float)
        breakout_down = (close < recent_low.shift(1)).astype(float)
        
        return pd.DataFrame({
            "trend_strength": trend_strength,
            "volatility_level": volatility_level,
            "range_ratio": range_ratio,
            "momentum": momentum,
            "volume_spike": volume_spike,
            "price_position": price_position,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "raw_volatility": volatility
        })
    
    def classify_regime(self, features: pd.Series) -> Tuple[int, float]:
        """
        Classify regime from feature row.
        
        Returns (regime_id, confidence)
        """
        trend_strength = features["trend_strength"]
        volatility_level = features["volatility_level"]
        volume_spike = features["volume_spike"]
        breakout_up = features["breakout_up"]
        breakout_down = features["breakout_down"]
        
        if volume_spike > 3 and volatility_level > 2:
            return 5, 0.8  # CHAOS
        
        if breakout_up > 0 or breakout_down > 0:
            if volatility_level > 1.5:
                return 4, 0.7  # BREAKOUT
        
        if volatility_level > 1.8:
            return 3, min(volatility_level / 2, 1.0)  # HIGH_VOLATILITY
        
        if trend_strength > 1.5:
            return 0, min(trend_strength / 3, 1.0)  # TREND_UP
        
        if trend_strength < -1.5:
            return 1, min(abs(trend_strength) / 3, 1.0)  # TREND_DOWN
        
        return 2, 0.6  # RANGE_BOUND
    
    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regime for each row in DataFrame.
        
        Returns DataFrame with columns:
        - regime_id: Numeric regime identifier
        - regime_name: String regime name
        - regime_confidence: Confidence in regime classification
        - regime_features: All computed features
        """
        features_df = self.compute_regime_features(df)
        
        regimes = []
        confidences = []
        
        for idx in range(len(features_df)):
            features = features_df.iloc[idx]
            if pd.isna(features["trend_strength"]):
                regimes.append(2)  # Default to RANGE_BOUND
                confidences.append(0.0)
            else:
                regime_id, confidence = self.classify_regime(features)
                regimes.append(regime_id)
                confidences.append(confidence)
        
        result = features_df.copy()
        result["regime_id"] = regimes
        result["regime_name"] = [self.REGIMES[r] for r in regimes]
        result["regime_confidence"] = confidences
        
        return result


class ExpertModel(nn.Module):
    """
    A single expert model specialized for a specific regime.
    
    Outputs (mu, sigma) predictions for regression.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 expert_type: str = "trend"):
        super().__init__()
        
        self.expert_type = expert_type
        self.hidden_dim = hidden_dim
        
        layers = []
        in_features = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive sigma
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, seq, features] or [batch, features]
            
        Returns:
            mu: Expected return [batch, 1]
            sigma: Uncertainty [batch, 1]
        """
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep
        
        hidden = self.encoder(x)
        
        mu = self.mu_head(hidden)
        sigma = self.sigma_head(hidden) + 0.001  # Minimum sigma
        
        return mu, sigma


class GatingNetwork(nn.Module):
    """
    Gating network that learns to weight experts based on regime features.
    
    Implements soft gating: outputs a probability distribution over experts.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden_dim: int = 64,
                 use_regime_features: bool = True):
        super().__init__()
        
        self.num_experts = num_experts
        self.use_regime_features = use_regime_features
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, 
                regime_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute expert weights.
        
        Args:
            x: Input features [batch, features] or [batch, seq, features]
            regime_features: Optional explicit regime features [batch, regime_features]
            
        Returns:
            weights: Expert weights [batch, num_experts], sum to 1
        """
        if x.dim() == 3:
            x = x[:, -1, :]
        
        if self.use_regime_features and regime_features is not None:
            x = torch.cat([x, regime_features], dim=-1)
        
        logits = self.gate(x)
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return weights


class MixtureOfExpertsRegressor(nn.Module):
    """
    Mixture-of-Experts model for regime-aware trading.
    
    Architecture:
    1. Expert models: Each specializes in a regime (trend, mean-reversion, volatility, chaos)
    2. Gating network: Learns to weight experts based on current market state
    3. Output: Weighted combination of expert predictions (mu, sigma)
    
    "Train specialists, then train a gating network that selects/weights experts based on regime features."
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_experts: int = 4,
                 expert_hidden_dim: int = 128,
                 dropout: float = 0.2,
                 regime_feature_dim: int = 9):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        expert_types = ["trend", "mean_reversion", "volatility", "chaos"]
        self.experts = nn.ModuleList([
            ExpertModel(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                num_layers=2,
                dropout=dropout,
                expert_type=expert_types[i % len(expert_types)]
            )
            for i in range(num_experts)
        ])
        
        gate_input_dim = input_dim + regime_feature_dim
        self.gate = GatingNetwork(
            input_dim=gate_input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim // 2,
            use_regime_features=True
        )
        
        self.expert_usage = torch.zeros(num_experts)
        
    def forward(self, 
                x: torch.Tensor,
                regime_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through mixture of experts.
        
        Args:
            x: Input features [batch, seq, features] or [batch, features]
            regime_features: Regime detection features [batch, regime_features]
            
        Returns:
            mu: Combined expected return [batch, 1]
            sigma: Combined uncertainty [batch, 1]
            gate_weights: Expert weights [batch, num_experts]
        """
        if regime_features is None:
            regime_features = torch.zeros(x.shape[0], 9, device=x.device)
        
        gate_weights = self.gate(x, regime_features)
        
        if self.training:
            self.expert_usage += gate_weights.detach().sum(dim=0).cpu()
        
        expert_mus = []
        expert_sigmas = []
        
        for expert in self.experts:
            mu, sigma = expert(x)
            expert_mus.append(mu)
            expert_sigmas.append(sigma)
        
        expert_mus = torch.stack(expert_mus, dim=1)  # [batch, num_experts, 1]
        expert_sigmas = torch.stack(expert_sigmas, dim=1)  # [batch, num_experts, 1]
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch, num_experts, 1]
        
        combined_mu = (expert_mus * gate_weights_expanded).sum(dim=1)  # [batch, 1]
        
        combined_sigma = torch.sqrt(
            (expert_sigmas ** 2 * gate_weights_expanded).sum(dim=1) +
            ((expert_mus - combined_mu.unsqueeze(1)) ** 2 * gate_weights_expanded).sum(dim=1)
        )
        
        return combined_mu, combined_sigma, gate_weights
    
    def get_expert_predictions(self, 
                                x: torch.Tensor,
                                regime_features: Optional[torch.Tensor] = None
                                ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Get individual expert predictions for analysis."""
        predictions = {}
        
        expert_names = ["trend", "mean_reversion", "volatility", "chaos"]
        for i, expert in enumerate(self.experts):
            name = expert_names[i % len(expert_names)]
            mu, sigma = expert(x)
            predictions[name] = (mu, sigma)
        
        return predictions
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        total_usage = self.expert_usage.sum()
        if total_usage == 0:
            return {}
        
        normalized = self.expert_usage / total_usage
        expert_names = ["trend", "mean_reversion", "volatility", "chaos"]
        
        return {
            expert_names[i % len(expert_names)]: normalized[i].item()
            for i in range(self.num_experts)
        }


class RegimeAwareTrainer:
    """
    Trainer for mixture-of-experts with regime detection.
    
    Training procedure:
    1. Compute regime features for all data
    2. Train experts on regime-specific subsets (warm-up)
    3. Train gating network to select experts
    4. Fine-tune entire system end-to-end
    """
    
    def __init__(self,
                 model: MixtureOfExpertsRegressor,
                 regime_detector: RegimeDetector,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.regime_detector = regime_detector
        self.device = device
        
    def compute_loss(self,
                     mu_pred: torch.Tensor,
                     sigma_pred: torch.Tensor,
                     mu_target: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss assuming Gaussian distribution.
        
        Loss = 0.5 * log(sigma^2) + 0.5 * (mu_target - mu_pred)^2 / sigma^2
        
        Plus regularization to prevent expert collapse.
        """
        variance = sigma_pred ** 2
        nll = 0.5 * torch.log(variance + 1e-6) + 0.5 * (mu_target - mu_pred) ** 2 / (variance + 1e-6)
        
        nll_loss = nll.mean()
        
        avg_weights = weights.mean(dim=0)
        load_balance_loss = self.model.num_experts * (avg_weights ** 2).sum()
        
        total_loss = nll_loss + 0.1 * load_balance_loss
        
        return total_loss
    
    def train_epoch(self,
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        for batch in dataloader:
            x = batch["features"].to(self.device)
            regime_features = batch.get("regime_features")
            if regime_features is not None:
                regime_features = regime_features.to(self.device)
            mu_target = batch["mu_target"].to(self.device)
            
            optimizer.zero_grad()
            
            mu_pred, sigma_pred, weights = self.model(x, regime_features)
            
            loss = self.compute_loss(mu_pred, sigma_pred, mu_target, weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += (mu_pred - mu_target).abs().mean().item()
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "mae": total_mae / n_batches,
            "expert_usage": self.model.get_expert_usage_stats()
        }


def create_regime_moe_model(input_dim: int = 81,
                             num_experts: int = 4,
                             hidden_dim: int = 128,
                             dropout: float = 0.2) -> MixtureOfExpertsRegressor:
    """
    Factory function to create a regime-aware mixture-of-experts model.
    """
    model = MixtureOfExpertsRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        expert_hidden_dim=hidden_dim,
        dropout=dropout,
        regime_feature_dim=9  # From RegimeDetector
    )
    
    logger.info(f"Created MoE model: {input_dim} input, {num_experts} experts, "
               f"{hidden_dim} hidden dim")
    
    return model
