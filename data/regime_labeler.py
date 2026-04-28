"""
Regime Labeling Module for Training Data

Creates regime labels using candle-only rules:
- BULL (0): Strong upward trend with positive rolling returns
- BEAR (1): Strong downward trend with negative rolling returns  
- HIGH_VOL (2): High volatility period (realized vol > 1.5x median)
- LOW_VOL_CHOP (3): Low volatility choppy/ranging market

Labels are based on:
- Rolling return (momentum direction)
- Drawdown from rolling high
- Realized volatility (log returns std)
- Bollinger Band width / ATR ratio

This module is for TRAINING BALANCE ONLY - inference unchanged.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds"""
    # Rolling window sizes (in candles)
    return_window: int = 96  # 24 hours at 15m
    volatility_window: int = 48  # 12 hours at 15m  
    bb_window: int = 20  # Standard BB period
    atr_window: int = 14  # Standard ATR period
    
    # Trend thresholds
    bull_return_threshold: float = 0.015  # 1.5% cumulative return = bull
    bear_return_threshold: float = -0.015  # -1.5% = bear
    
    # Volatility thresholds
    high_vol_multiplier: float = 1.5  # > 1.5x median vol = high vol
    low_vol_multiplier: float = 0.7  # < 0.7x median vol = low vol
    
    # Drawdown thresholds
    significant_drawdown: float = 0.02  # 2% drawdown overrides bull


class RegimeLabeler:
    """
    Labels each candle with a market regime for training balance.
    
    Regime IDs:
    0 = BULL: Positive momentum, low drawdown
    1 = BEAR: Negative momentum, high drawdown
    2 = HIGH_VOL: High realized volatility (choppy/news-driven)
    3 = LOW_VOL_CHOP: Low volatility, ranging/consolidation
    """
    
    REGIME_NAMES = {
        0: "BULL",
        1: "BEAR",
        2: "HIGH_VOL",
        3: "LOW_VOL_CHOP"
    }
    
    NUM_REGIMES = 4
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime-relevant features from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            DataFrame with computed features
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        # Log returns
        log_returns = np.log(close / close.shift(1))
        
        # 1. Rolling cumulative return
        rolling_return = close.pct_change(self.config.return_window)
        
        # 2. Realized volatility (annualized from log returns)
        realized_vol = log_returns.rolling(self.config.volatility_window).std()
        median_vol = realized_vol.rolling(self.config.volatility_window * 10).median()
        vol_ratio = realized_vol / median_vol.clip(lower=1e-8)
        
        # 3. Drawdown from rolling high
        rolling_high = close.rolling(self.config.return_window).max()
        drawdown = (close - rolling_high) / rolling_high
        
        # 4. Bollinger Band width
        bb_ma = close.rolling(self.config.bb_window).mean()
        bb_std = close.rolling(self.config.bb_window).std()
        bb_width = (2 * bb_std) / bb_ma.clip(lower=1)
        
        # 5. ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_window).mean()
        atr_pct = atr / close
        
        # 6. BB Width / ATR ratio (choppiness indicator)
        chop_ratio = bb_width / atr_pct.clip(lower=1e-8)
        
        return pd.DataFrame({
            "rolling_return": rolling_return,
            "realized_vol": realized_vol,
            "vol_ratio": vol_ratio,
            "drawdown": drawdown,
            "bb_width": bb_width,
            "atr_pct": atr_pct,
            "chop_ratio": chop_ratio
        }, index=df.index)
    
    def classify_regime(self, features: pd.Series) -> int:
        """
        Classify single sample into regime.
        
        Priority order:
        1. HIGH_VOL if vol_ratio > 1.5x (market stress takes precedence)
        2. BULL if rolling_return > threshold and drawdown small
        3. BEAR if rolling_return < threshold or significant drawdown
        4. LOW_VOL_CHOP otherwise
        
        Args:
            features: Series with regime features
            
        Returns:
            Regime ID (0-3)
        """
        vol_ratio = features.get("vol_ratio", 1.0)
        rolling_return = features.get("rolling_return", 0.0)
        drawdown = features.get("drawdown", 0.0)
        
        # Handle NaN values
        if pd.isna(vol_ratio) or pd.isna(rolling_return) or pd.isna(drawdown):
            return 3  # Default to LOW_VOL_CHOP
        
        # 1. HIGH_VOL takes precedence (market stress)
        if vol_ratio > self.config.high_vol_multiplier:
            return 2  # HIGH_VOL
        
        # 2. BULL: Strong positive momentum with small drawdown
        if (rolling_return > self.config.bull_return_threshold and 
            drawdown > -self.config.significant_drawdown):
            return 0  # BULL
        
        # 3. BEAR: Strong negative momentum or significant drawdown
        if (rolling_return < self.config.bear_return_threshold or 
            drawdown < -self.config.significant_drawdown):
            return 1  # BEAR
        
        # 4. LOW_VOL_CHOP: Everything else (low vol, ranging)
        return 3  # LOW_VOL_CHOP
    
    def label_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label all samples with regime IDs.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            numpy array of regime IDs (0-3) with same length as df
        """
        features = self.compute_features(df)
        
        regime_ids = np.zeros(len(df), dtype=np.int64)
        
        for i in range(len(df)):
            regime_ids[i] = self.classify_regime(features.iloc[i])
        
        return regime_ids
    
    def get_regime_distribution(self, regime_ids: np.ndarray) -> Dict[str, float]:
        """
        Get distribution of regimes in dataset.
        
        Args:
            regime_ids: Array of regime IDs
            
        Returns:
            Dict mapping regime name to percentage
        """
        total = len(regime_ids)
        distribution = {}
        
        for regime_id, name in self.REGIME_NAMES.items():
            count = np.sum(regime_ids == regime_id)
            distribution[name] = count / total if total > 0 else 0.0
        
        return distribution
    
    def get_sample_weights(self, regime_ids: np.ndarray, 
                           target_balance: float = 0.25) -> np.ndarray:
        """
        Compute sample weights for balanced training.
        
        Each regime should ideally represent ~25% of training samples.
        Underrepresented regimes get higher weights.
        
        Args:
            regime_ids: Array of regime IDs
            target_balance: Target proportion for each regime (default 0.25)
            
        Returns:
            numpy array of sample weights
        """
        total = len(regime_ids)
        weights = np.ones(total, dtype=np.float32)
        
        # Count samples per regime
        regime_counts = {}
        for regime_id in range(self.NUM_REGIMES):
            regime_counts[regime_id] = np.sum(regime_ids == regime_id)
        
        # Calculate inverse frequency weights
        for regime_id, count in regime_counts.items():
            if count > 0:
                # Weight = target_proportion / actual_proportion
                actual_proportion = count / total
                regime_weight = target_balance / actual_proportion
                
                # Clip weights to prevent extreme values
                regime_weight = np.clip(regime_weight, 0.25, 4.0)
                
                # Apply weight to samples in this regime
                mask = regime_ids == regime_id
                weights[mask] = regime_weight
        
        # Normalize weights to sum to len(weights)
        weights = weights * (total / weights.sum())
        
        return weights
    
    def log_distribution(self, regime_ids: np.ndarray, dataset_name: str = "Dataset"):
        """Log regime distribution for debugging."""
        dist = self.get_regime_distribution(regime_ids)
        
        logger.info(f"[{dataset_name}] Regime Distribution:")
        for name, pct in dist.items():
            count = np.sum(regime_ids == list(self.REGIME_NAMES.keys())[
                list(self.REGIME_NAMES.values()).index(name)
            ])
            logger.info(f"  {name}: {pct*100:.1f}% ({count:,} samples)")


def create_regime_sampler(regime_ids: np.ndarray, 
                          target_balance: float = 0.25) -> "torch.utils.data.WeightedRandomSampler":
    """
    Create a WeightedRandomSampler for regime-balanced batching.
    
    Args:
        regime_ids: Array of regime IDs for all training samples
        target_balance: Target proportion for each regime
        
    Returns:
        WeightedRandomSampler for use with DataLoader
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    
    labeler = RegimeLabeler()
    weights = labeler.get_sample_weights(regime_ids, target_balance)
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(regime_ids),
        replacement=True
    )
    
    return sampler


def test_regime_labeler():
    """Test regime labeler with synthetic data."""
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 1000
    
    # Simulate price series with different regimes
    prices = np.cumsum(np.random.randn(n) * 0.001 + 0.0001) + 100
    
    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 0.1,
        "high": prices + abs(np.random.randn(n) * 0.2),
        "low": prices - abs(np.random.randn(n) * 0.2),
        "close": prices,
        "volume": np.random.rand(n) * 1000000
    })
    
    labeler = RegimeLabeler()
    regime_ids = labeler.label_regimes(df)
    
    print("Regime distribution:")
    dist = labeler.get_regime_distribution(regime_ids)
    for name, pct in dist.items():
        print(f"  {name}: {pct*100:.1f}%")
    
    print("\nSample weights range:")
    weights = labeler.get_sample_weights(regime_ids)
    print(f"  Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
    
    return regime_ids


if __name__ == "__main__":
    test_regime_labeler()
