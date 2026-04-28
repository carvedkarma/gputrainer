"""
GPU Trainer Configuration Module

Centralized configuration for timeframes, training parameters, and defaults.
"""

from .timeframe_config import (
    TimeframeConfig,
    TIMEFRAME_CONFIGS,
    DEFAULT_TIMEFRAME,
    DEFAULT_SYMBOL,
    get_config,
    get_horizon_for_timeframe,
    get_lookback_for_timeframe,
    get_default_cost,
)

# Re-export training configuration classes (previously in config.py)
from .training_config import (
    Config,
    config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    RLConfig,
    HorizonConfig,
    NoTradeConfig,
    CostModeConfig,
    InstitutionConfig,
)

__all__ = [
    # Timeframe configuration
    "TimeframeConfig",
    "TIMEFRAME_CONFIGS",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_SYMBOL",
    "get_config",
    "get_horizon_for_timeframe",
    "get_lookback_for_timeframe",
    "get_default_cost",
    # Training configuration (from original config.py)
    "Config",
    "config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "RLConfig",
    "HorizonConfig",
    "NoTradeConfig",
    "CostModeConfig",
    "InstitutionConfig",
]
