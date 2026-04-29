from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MythosConfig:
    # Data / walk-forward
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    interval: str = "15m"
    horizon: int = 48
    train_months: int = 12
    test_months: int = 1
    max_folds: int | None = None
    lookback: int = 96
    threshold_quantile: float = 0.75
    random_state: int = 42
    tp_mult: float = 3.0
    sl_mult: float = 1.0

    # World model / router
    n_regimes: int = 4
    min_regime_confidence: float = 0.45
    min_router_confidence: float = 0.55
    min_expected_r: float = 0.01
    min_edge_threshold: float = 0.02
    min_confidence: float = 0.55
    reliability_window: int = 120
    uncertainty_penalty: float = 0.40
    router_reliability_weight: float = 0.20
    abstain_edge_floor: float = 0.01

    # Risk constitution
    daily_loss_cap_r: float = -4.0
    weekly_loss_cap_r: float = -12.0
    trailing_stop_r: float = -25.0
    cooldown_bars: int = 4
    max_trades_per_day: int = 8
    max_leverage: float = 1.8
    vol_target: float = 0.012
    min_size_mult: float = 0.5
    max_size_mult: float = 1.8
    min_trades_for_confidence: int = 25
    min_trades_per_fold: int = 25
    save_best_model: bool = True
    best_model_metric: str = "total_r"
    model_output_dir: str = "checkpoints/mythos_models"
    analog_k: int = 48
    analog_blend: float = 0.35
    online_reliability_alpha: float = 0.08
    reliability_regime_window: int = 80
    robust_score_dd_penalty: float = 0.35
    side_balance_window: int = 160
    side_imbalance_soft_cap: float = 0.82
    side_imbalance_edge_penalty: float = 0.015
    drawdown_edge_start_r: float = 8.0
    drawdown_edge_step_r: float = 4.0
    drawdown_edge_boost: float = 0.0025
    loss_streak_trigger: int = 4
    loss_streak_cooldown_bars: int = 12
    online_allocator_lr: float = 0.06
    online_allocator_min_mult: float = 0.75
    online_allocator_max_mult: float = 1.55
    change_detect_z_thresh: float = 2.6
    change_detect_confirm_bars: int = 2
    change_detect_cooldown_bars: int = 24
    change_edge_floor_boost: float = 0.004
    change_confidence_boost: float = 0.03
    change_uncertainty_mult: float = 1.15
    counterfactual_min_advantage_r: float = 0.006
    counterfactual_risk_penalty: float = 0.6
    counterfactual_margin: float = 0.006
    counterfactual_uncertainty_weight: float = 0.50
    counterfactual_min_alt_hits: int = 8

    def __post_init__(self) -> None:
        # Keep legacy/new naming aligned for callers.
        self.min_trades_per_fold = int(self.min_trades_for_confidence)
        # Route confidence should honor the public router confidence knob.
        self.min_confidence = float(self.min_router_confidence)
        # Keep edge gate aligned with the exposed minimum expected-R control.
        self.abstain_edge_floor = float(self.min_expected_r)

