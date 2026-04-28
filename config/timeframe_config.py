"""
Centralized Timeframe Configuration

SINGLE SOURCE OF TRUTH for all timeframe-dependent parameters.
This prevents hidden mismatches between training and inference.

Usage:
    from config.timeframe_config import get_config, DEFAULT_TIMEFRAME
    config = get_config("15m")
    horizon = config.horizon_bars  # 16 for 15m
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# === PRODUCTION DEFAULT: BTCUSDT 15m ===
DEFAULT_TIMEFRAME = "15m"
DEFAULT_SYMBOL = "BTCUSDT"


@dataclass(frozen=True)
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    
    timeframe: str
    horizon_bars: int           # Prediction horizon in bars (4h target)
    lookback_bars: int          # Lookback for volatility (2x horizon)
    
    # Cost and threshold defaults
    default_cost: float         # Round-trip cost estimate
    min_net_edge: float         # Minimum edge after costs
    min_confidence: float       # Minimum mu/sigma ratio
    
    # Feature engineering
    feature_count_stf: int      # Expected STF feature count
    feature_count_mtf: int      # Expected MTF feature count
    
    # Annualization factor for volatility
    bars_per_day: int           # Number of bars per 24 hours
    
    def __post_init__(self):
        logger.info(f"TimeframeConfig({self.timeframe}): horizon={self.horizon_bars}, "
                   f"lookback={self.lookback_bars}, cost={self.default_cost:.4f}")


# === TIMEFRAME CONFIGURATIONS ===
# All horizons target ~4 hours forward prediction
# horizon_bars = 4 hours / bar_duration

TIMEFRAME_CONFIGS: Dict[str, TimeframeConfig] = {
    "1m": TimeframeConfig(
        timeframe="1m",
        horizon_bars=240,           # 4h * 60 = 240 bars
        lookback_bars=480,          # 2x horizon
        default_cost=0.0009,        # 0.09%
        min_net_edge=0.0,           # Debug: no minimum
        min_confidence=0.40,        # Lowered from 0.7 to reduce HOLD-heavy labels
        feature_count_stf=41,
        feature_count_mtf=66,
        bars_per_day=1440,
    ),
    "5m": TimeframeConfig(
        timeframe="5m",
        horizon_bars=48,            # 4h * 12 = 48 bars
        lookback_bars=96,           # 2x horizon
        default_cost=0.0009,
        min_net_edge=0.0,
        min_confidence=0.40,        # Lowered from 0.7 to reduce HOLD-heavy labels
        feature_count_stf=41,
        feature_count_mtf=66,
        bars_per_day=288,
    ),
    "15m": TimeframeConfig(
        timeframe="15m",
        horizon_bars=16,            # 4h * 4 = 16 bars  <-- PRODUCTION DEFAULT
        lookback_bars=32,           # 2x horizon
        default_cost=0.0009,        # 0.09% round-trip
        min_net_edge=0.0,           # Debug: accept all edges
        min_confidence=0.40,        # Lowered from 0.7 to reduce HOLD-heavy labels
        feature_count_stf=41,       # compute_technical_features output
        feature_count_mtf=66,       # MTF fusion output
        bars_per_day=96,
    ),
    "1h": TimeframeConfig(
        timeframe="1h",
        horizon_bars=4,             # 4h / 1h = 4 bars
        lookback_bars=8,
        default_cost=0.0009,
        min_net_edge=0.0,
        min_confidence=0.40,        # Lowered from 0.7 to reduce HOLD-heavy labels
        feature_count_stf=41,
        feature_count_mtf=66,
        bars_per_day=24,
    ),
    "4h": TimeframeConfig(
        timeframe="4h",
        horizon_bars=1,             # 4h / 4h = 1 bar
        lookback_bars=4,            # Need minimum 4 for vol calc
        default_cost=0.0009,
        min_net_edge=0.0,
        min_confidence=0.40,        # Lowered from 0.7 to reduce HOLD-heavy labels
        feature_count_stf=41,
        feature_count_mtf=66,
        bars_per_day=6,
    ),
}


def get_config(timeframe: str = DEFAULT_TIMEFRAME) -> TimeframeConfig:
    """
    Get configuration for a timeframe.
    
    Args:
        timeframe: One of "1m", "5m", "15m", "1h", "4h"
        
    Returns:
        TimeframeConfig with all parameters
        
    Raises:
        ValueError if timeframe not supported
    """
    if timeframe not in TIMEFRAME_CONFIGS:
        valid = list(TIMEFRAME_CONFIGS.keys())
        raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {valid}")
    
    return TIMEFRAME_CONFIGS[timeframe]


def get_horizon_for_timeframe(timeframe: str = DEFAULT_TIMEFRAME) -> int:
    """Convenience function to get horizon bars for a timeframe."""
    return get_config(timeframe).horizon_bars


def get_lookback_for_timeframe(timeframe: str = DEFAULT_TIMEFRAME) -> int:
    """Convenience function to get lookback bars for a timeframe."""
    return get_config(timeframe).lookback_bars


def get_default_cost(timeframe: str = DEFAULT_TIMEFRAME) -> float:
    """Convenience function to get default cost for a timeframe."""
    return get_config(timeframe).default_cost


# Log default config on import
logger.info(f"=== Timeframe Config: DEFAULT_TIMEFRAME={DEFAULT_TIMEFRAME} ===")
logger.info(f"=== Default horizon: {TIMEFRAME_CONFIGS[DEFAULT_TIMEFRAME].horizon_bars} bars ===")
