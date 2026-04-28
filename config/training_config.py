import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class DataConfig:
    # BTCUSDT only for focused institutional-grade training
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    # 15m timeframe only for deep learning models (3 year training horizon)
    timeframes: List[str] = field(default_factory=lambda: ["15m"])
    # 3 years of 15m candles: 3 * 365 * 24 * 4 = 105,120 candles
    lookback_candles: int = 105120
    sequence_length: int = 100
    prediction_horizon: int = 16  # 16-bar horizon = 4 hours at 15m
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Regime-balanced training
    regime_balanced: bool = True
    target_regime_balance: float = 0.25  # Target ~25% per regime in each batch

@dataclass
class ModelConfig:
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 6
    transformer_dropout: float = 0.1
    
    lstm_hidden: int = 256
    lstm_layers: int = 3
    lstm_dropout: float = 0.2
    
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    vae_latent_dim: int = 64
    
    gnn_hidden: int = 128
    gnn_layers: int = 3
    
    sentiment_model: str = "distilbert-base-uncased"
    
@dataclass 
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4  # Increased from 1e-5 to 1e-4 for better regularization
    epochs: int = 300
    patience: int = 50
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 0.0  # Set dynamically as lr * 0.05 if 0
    
    use_curriculum: bool = True
    use_contrastive: bool = True
    use_online_learning: bool = True
    
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
@dataclass
class RLConfig:
    algorithm: str = "PPO"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    
    initial_capital: float = 10000.0
    max_position_size: float = 1.0
    transaction_cost: float = 0.001

@dataclass
class HorizonConfig:
    """Institution-grade horizon-specific thresholds"""
    bars: int = 15
    purpose: str = "active_trades"
    min_edge: float = 0.0015          # Minimum edge after costs (log)
    min_confidence: float = 1.25       # μ/σ ratio threshold
    weight: float = 0.5                # Training/prediction weight
    max_hold_bars: int = 15            # Time stop: forced exit

@dataclass
class NoTradeConfig:
    """Institution-grade NO-TRADE conditions"""
    dead_zone_threshold: float = 0.0015     # ±15 bps dead zone
    uncertainty_percentile: float = 0.95
    max_uncertainty_multiplier: float = 2.5  # σ > 2.5x median = panic
    horizon_disagreement_veto: bool = True
    max_loss_streak: int = 3
    loss_streak_size_reduction: float = 0.5
    funding_flip_window: int = 4
    funding_flip_threshold: float = 0.001

@dataclass
class CostModeConfig:
    """Trading cost modes for label creation"""
    # Cost modes: defines round-trip trading costs for different execution strategies
    TAKER_TAKER = 0.0009   # Conservative: taker entry + taker exit (0.09%)
    MAKER_TAKER = 0.0006   # Optimistic: maker entry + taker exit (0.06%)
    MAKER_MAKER = 0.0004   # Aggressive: maker entry + maker exit (0.04%)
    
    current_mode: str = "taker_taker"  # Default to conservative
    
    def get_cost(self) -> float:
        """Get round-trip cost for current mode"""
        costs = {
            "taker_taker": self.TAKER_TAKER,
            "maker_taker": self.MAKER_TAKER,
            "maker_maker": self.MAKER_MAKER
        }
        return costs.get(self.current_mode, self.TAKER_TAKER)

@dataclass
class InstitutionConfig:
    """Institution-grade trading configuration"""
    # Horizon-specific configs
    h15: HorizonConfig = field(default_factory=lambda: HorizonConfig(
        bars=15, purpose="active_trades", min_edge=0.0015, min_confidence=1.25, weight=0.5, max_hold_bars=15
    ))
    h60: HorizonConfig = field(default_factory=lambda: HorizonConfig(
        bars=60, purpose="swing_intraday", min_edge=0.0025, min_confidence=1.10, weight=0.35, max_hold_bars=60
    ))
    h240: HorizonConfig = field(default_factory=lambda: HorizonConfig(
        bars=240, purpose="trend_filter", min_edge=0.0040, min_confidence=0.90, weight=0.15, max_hold_bars=240
    ))
    
    no_trade: NoTradeConfig = field(default_factory=NoTradeConfig)
    cost_mode: CostModeConfig = field(default_factory=CostModeConfig)
    
    # Trading costs (futures) - base values
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001
    spread_estimate: float = 0.0002
    total_round_trip: float = 0.0009  # Default taker/taker
    
    # Position sizing
    risk_cap_per_trade: float = 0.0025   # 0.25% equity per trade
    min_size_pct: float = 0.0005         # 0.05% minimum
    max_size_pct: float = 0.003          # 0.30% maximum
    
    # Expected performance bounds (overfit detection)
    max_realistic_sharpe: float = 3.0
    max_realistic_win_rate: float = 0.65
    
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    institution: InstitutionConfig = field(default_factory=InstitutionConfig)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    
    base_dir: Path = Path(__file__).parent.parent  # gpu_trainer directory
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_cache")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "saved_models")
    
    db_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/btc_signals")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    api_port: int = 8000
    
    # Replit proxy URL for fetching Binance data (bypasses Australian geoblocking)
    # Set this to your Replit app URL, e.g., "https://your-app.replit.app"
    replit_proxy_url: str = os.getenv("REPLIT_PROXY_URL", "")
    
    # Dashboard URL for GPU Export API (multi-timeframe aligned data)
    # This is the preferred method - provides properly aligned data with as-of joins
    dashboard_url: str = os.getenv("DASHBOARD_URL", "https://your-app.replit.app")
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.training.log_dir).mkdir(exist_ok=True)
        
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

config = Config()
