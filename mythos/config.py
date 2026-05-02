from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


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
    adaptive_side_target_strength: float = 0.22
    adaptive_side_target_min: float = 0.35
    adaptive_side_target_max: float = 0.65
    side_health_penalty: float = 0.02
    side_health_boost: float = 0.008
    side_health_decay: float = 0.97
    side_aggression_boost: float = 0.006
    side_aggression_min_gap: float = 0.05
    side_aggression_underweight_gain: float = 0.60
    precision_min_confidence: float = 0.0
    precision_min_edge: float = 0.0
    precision_min_conviction: float = 0.52
    precision_high_conviction: float = 0.72
    conviction_weight_edge: float = 0.30
    conviction_weight_confidence: float = 0.30
    conviction_weight_uncertainty: float = 0.25
    conviction_weight_meta: float = 0.15
    conviction_score_threshold: float = 0.62
    conviction_boost: float = 0.35
    conviction_max_size_mult: float = 2.2
    conviction_recent_window: int = 64
    conviction_recent_min_trades: int = 12
    conviction_recent_min_expectancy: float = 0.03
    sure_min_analog_hits: int = 12
    sure_min_analog_ratio: float = 0.20
    sure_meta_strength_min: float = 0.10
    sure_edge_buffer: float = 0.001
    sure_confidence_buffer: float = 0.03
    sure_recent_window: int = 96
    sure_recent_min_trades: int = 24
    sure_recent_min_hit_rate: float = 0.52
    sure_recent_min_expectancy: float = 0.04
    sure_cold_start_conviction_extra: float = 0.04
    sure_cold_start_meta_extra: float = 0.06
    leverage_edge_buffer: float = 0.003
    leverage_confidence_buffer: float = 0.04
    leverage_conviction_buffer: float = 0.04
    leverage_recent_window: int = 120
    leverage_recent_min_trades: int = 20
    leverage_recent_min_hit_rate: float = 0.53
    leverage_recent_min_expectancy: float = 0.05
    conviction_guard_window: int = 32
    conviction_guard_min_trades: int = 8
    conviction_guard_min_expectancy_r: float = 0.03
    conviction_requires_meta_ready: bool = True
    # Backward-compatible aliases for older CLI wiring.
    conviction_size_gate: float = 0.70
    conviction_size_boost: float = 1.30
    drawdown_edge_start_r: float = 8.0
    drawdown_edge_step_r: float = 4.0
    drawdown_edge_boost: float = 0.0025
    loss_streak_trigger: int = 4
    loss_streak_cooldown_bars: int = 12
    side_fail_window: int = 48
    side_fail_min_trades: int = 10
    side_fail_expectancy_r: float = -0.12
    side_fail_cooldown_bars: int = 24
    side_fail_ema_alpha: float = 0.25
    side_fail_hard_pause: bool = False
    short_boost_enable: bool = True
    short_boost_window: int = 96
    short_boost_min_trades: int = 24
    short_boost_threshold_r: float = 0.08
    short_boost_edge: float = 0.0045
    short_boost_confidence: float = 0.03
    flip_intensity_trigger: float = 0.35
    flip_harden_hold_bars: int = 24
    instability_edge_mult: float = 0.70
    instability_confidence_drop: float = 0.08
    instability_uncertainty_mult: float = 1.35
    transition_min_samples: int = 6
    # Canonical transition learner knobs (wired from CLI).
    transition_learn_rate: float = 0.12
    transition_edge_gain: float = 0.35
    transition_confidence_gain: float = 0.06
    transition_uncertainty_gain: float = 0.30
    # Optional GPU-backed neural experts (v6 evolution).
    use_gpu_expert: bool = True
    gpu_expert_hidden: int = 64
    gpu_expert_dropout: float = 0.05
    gpu_expert_epochs: int = 10
    gpu_expert_batch_size: int = 1024
    gpu_expert_lr: float = 1e-3
    # Backward-compatible aliases (older revisions/checkpoints).
    transition_memory_alpha: float = 0.12
    transition_memory_edge_scale: float = 0.25
    transition_memory_confidence_scale: float = 0.06
    transition_memory_uncertainty_scale: float = 0.45
    enable_gpu_neural_experts: bool = True
    neural_expert_hidden: int = 96
    neural_expert_epochs: int = 8
    neural_expert_lr: float = 8e-4
    neural_expert_batch_size: int = 2048
    neural_expert_dropout: float = 0.10
    neural_expert_device: str = "auto"
    # Deep meta-learner (decision quality model on top of expert outputs).
    use_meta_learner: bool = True
    meta_learner_device: str = "auto"
    meta_learner_hidden: int = 96
    meta_learner_epochs: int = 6
    meta_learner_batch_size: int = 1024
    meta_learner_lr: float = 8e-4
    meta_learner_dropout: float = 0.10
    meta_learner_reg_weight: float = 0.25
    meta_learner_edge_blend: float = 0.30
    meta_learner_conf_blend: float = 0.20
    meta_learner_uncertainty_penalty: float = 0.80
    meta_learner_fallback: bool = True
    meta_learner_fallback_lr: float = 0.03
    meta_bootstrap_enable: bool = True
    meta_bootstrap_samples: int = 1024
    meta_bootstrap_epochs: int = 1
    meta_bootstrap_min_samples: int = 64
    meta_bootstrap_edge_cap: float = 0.04
    meta_bootstrap_conf_gain: float = 0.12
    # Aliases used by CLI for readability.
    meta_learner_edge_gain: float = 0.30
    meta_learner_confidence_gain: float = 0.20
    meta_learner_min_side_prob: float = 0.50
    meta_learner_min_train_samples: int = 512
    meta_learner_warmup_samples: int = 1024
    meta_learner_ready_prob_floor: float = 0.42
    meta_learner_ready_prob_ceiling: float = 0.58
    regime_flip_confidence_boost: float = 0.05
    regime_flip_min_analog_edge: float = 0.0
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
        # Keep transition learner legacy/new naming aligned.
        self.transition_memory_alpha = float(self.transition_learn_rate)
        self.transition_memory_edge_scale = float(self.transition_edge_gain)
        self.transition_memory_confidence_scale = float(self.transition_confidence_gain)
        self.transition_memory_uncertainty_scale = float(self.transition_uncertainty_gain)
        self.adaptive_side_target_strength = float(
            np.clip(getattr(self, "adaptive_side_target_strength", 0.22), 0.0, 1.0)
        )
        tmin = float(np.clip(getattr(self, "adaptive_side_target_min", 0.35), 0.05, 0.95))
        tmax = float(np.clip(getattr(self, "adaptive_side_target_max", 0.65), 0.05, 0.95))
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        self.adaptive_side_target_min = tmin
        self.adaptive_side_target_max = tmax
        self.side_health_penalty = float(max(getattr(self, "side_health_penalty", 0.02), 0.0))
        self.side_health_boost = float(max(getattr(self, "side_health_boost", 0.008), 0.0))
        self.side_health_decay = float(np.clip(getattr(self, "side_health_decay", 0.97), 0.80, 0.999))
        self.side_aggression_boost = float(
            np.clip(getattr(self, "side_aggression_boost", 0.006), 0.0, 0.05)
        )
        self.side_aggression_min_gap = float(
            np.clip(getattr(self, "side_aggression_min_gap", 0.05), 0.0, 1.0)
        )
        self.side_aggression_underweight_gain = float(
            np.clip(getattr(self, "side_aggression_underweight_gain", 0.60), 0.0, 3.0)
        )
        self.precision_min_confidence = float(
            np.clip(getattr(self, "precision_min_confidence", 0.0), 0.0, 1.0)
        )
        self.precision_min_edge = float(max(getattr(self, "precision_min_edge", 0.0), 0.0))
        self.precision_min_conviction = float(
            np.clip(getattr(self, "precision_min_conviction", 0.52), 0.0, 1.0)
        )
        self.precision_high_conviction = float(
            np.clip(getattr(self, "precision_high_conviction", 0.72), 0.0, 1.0)
        )
        if self.precision_high_conviction < self.precision_min_conviction:
            self.precision_high_conviction = self.precision_min_conviction
        self.conviction_weight_edge = float(max(getattr(self, "conviction_weight_edge", 0.30), 0.0))
        self.conviction_weight_confidence = float(max(getattr(self, "conviction_weight_confidence", 0.30), 0.0))
        self.conviction_weight_uncertainty = float(max(getattr(self, "conviction_weight_uncertainty", 0.25), 0.0))
        self.conviction_weight_meta = float(max(getattr(self, "conviction_weight_meta", 0.15), 0.0))
        self.conviction_size_gate = float(np.clip(getattr(self, "conviction_size_gate", 0.70), 0.0, 1.0))
        self.conviction_size_boost = float(np.clip(getattr(self, "conviction_size_boost", 1.30), 1.0, 3.0))
        self.conviction_score_threshold = float(
            np.clip(getattr(self, "conviction_score_threshold", self.conviction_size_gate), 0.0, 1.0)
        )
        boost_raw = float(getattr(self, "conviction_boost", 0.35))
        # Backward compatibility: old aliases used multiplicative form (e.g. 1.30 means +0.30).
        if boost_raw > 1.0:
            boost_raw -= 1.0
        self.conviction_boost = float(np.clip(boost_raw, 0.0, 2.0))
        self.conviction_max_size_mult = float(
            max(getattr(self, "conviction_max_size_mult", 2.2), self.max_size_mult)
        )
        self.conviction_recent_window = int(max(getattr(self, "conviction_recent_window", 64), 8))
        self.conviction_recent_min_trades = int(max(getattr(self, "conviction_recent_min_trades", 12), 1))
        self.conviction_recent_min_expectancy = float(
            getattr(self, "conviction_recent_min_expectancy", 0.03)
        )
        self.sure_min_analog_hits = int(max(getattr(self, "sure_min_analog_hits", 12), 0))
        self.sure_min_analog_ratio = float(np.clip(getattr(self, "sure_min_analog_ratio", 0.20), 0.0, 1.0))
        self.sure_meta_strength_min = float(np.clip(getattr(self, "sure_meta_strength_min", 0.10), 0.0, 1.0))
        self.sure_edge_buffer = float(max(getattr(self, "sure_edge_buffer", 0.001), 0.0))
        self.sure_confidence_buffer = float(np.clip(getattr(self, "sure_confidence_buffer", 0.03), 0.0, 1.0))
        self.sure_recent_window = int(max(getattr(self, "sure_recent_window", 96), 8))
        self.sure_recent_min_trades = int(max(getattr(self, "sure_recent_min_trades", 24), 1))
        self.sure_recent_min_hit_rate = float(
            np.clip(getattr(self, "sure_recent_min_hit_rate", 0.52), 0.0, 1.0)
        )
        self.sure_recent_min_expectancy = float(getattr(self, "sure_recent_min_expectancy", 0.04))
        self.sure_cold_start_conviction_extra = float(
            np.clip(getattr(self, "sure_cold_start_conviction_extra", 0.04), 0.0, 0.5)
        )
        self.sure_cold_start_meta_extra = float(
            np.clip(getattr(self, "sure_cold_start_meta_extra", 0.06), 0.0, 0.5)
        )
        self.leverage_edge_buffer = float(max(getattr(self, "leverage_edge_buffer", 0.003), 0.0))
        self.leverage_confidence_buffer = float(
            np.clip(getattr(self, "leverage_confidence_buffer", 0.04), 0.0, 1.0)
        )
        self.leverage_conviction_buffer = float(
            np.clip(getattr(self, "leverage_conviction_buffer", 0.04), 0.0, 1.0)
        )
        self.leverage_recent_window = int(max(getattr(self, "leverage_recent_window", 120), 8))
        self.leverage_recent_min_trades = int(max(getattr(self, "leverage_recent_min_trades", 20), 1))
        self.leverage_recent_min_hit_rate = float(
            np.clip(getattr(self, "leverage_recent_min_hit_rate", 0.53), 0.0, 1.0)
        )
        self.leverage_recent_min_expectancy = float(
            getattr(self, "leverage_recent_min_expectancy", 0.05)
        )
        self.conviction_guard_window = int(
            max(getattr(self, "conviction_guard_window", self.conviction_recent_window), 8)
        )
        self.conviction_guard_min_trades = int(
            max(getattr(self, "conviction_guard_min_trades", self.conviction_recent_min_trades), 1)
        )
        self.conviction_guard_min_expectancy_r = float(
            getattr(self, "conviction_guard_min_expectancy_r", self.conviction_recent_min_expectancy)
        )
        # Compatibility aliases consumed by risk sizing and older walk-forward revisions.
        self.high_conviction_expectancy_window = int(self.conviction_recent_window)
        self.conviction_requires_meta_ready = bool(getattr(self, "conviction_requires_meta_ready", True))
        self.side_fail_hard_pause = bool(getattr(self, "side_fail_hard_pause", False))
        self.short_boost_enable = bool(getattr(self, "short_boost_enable", True))
        self.short_boost_window = int(max(getattr(self, "short_boost_window", 96), 8))
        self.short_boost_min_trades = int(max(getattr(self, "short_boost_min_trades", 24), 1))
        self.short_boost_threshold_r = float(getattr(self, "short_boost_threshold_r", 0.08))
        self.short_boost_edge = float(max(getattr(self, "short_boost_edge", 0.0045), 0.0))
        self.short_boost_confidence = float(
            np.clip(getattr(self, "short_boost_confidence", 0.03), 0.0, 1.0)
        )
        # Keep neural-expert naming aligned across revisions.
        self.use_gpu_neural_expert = bool(self.enable_gpu_neural_experts)
        self.neural_expert_hidden_dim = int(self.neural_expert_hidden)
        dev = str(self.neural_expert_device or "auto").strip().lower()
        if dev not in {"auto", "cuda", "cpu"}:
            dev = "auto"
        self.neural_expert_device = dev
        self.allow_cpu_neural_expert = bool(dev in {"auto", "cpu"})
        meta_dev = str(self.meta_learner_device or "auto").strip().lower()
        if meta_dev not in {"auto", "cuda", "cpu"}:
            meta_dev = "auto"
        self.meta_learner_device = meta_dev
        self.allow_cpu_meta_learner = bool(meta_dev in {"auto", "cpu"})
        self.meta_learner_fallback = bool(getattr(self, "meta_learner_fallback", True))
        self.meta_learner_fallback_lr = float(
            np.clip(getattr(self, "meta_learner_fallback_lr", 0.03), 1e-5, 0.5)
        )
        self.meta_bootstrap_enable = bool(getattr(self, "meta_bootstrap_enable", True))
        self.meta_bootstrap_samples = int(max(getattr(self, "meta_bootstrap_samples", 1024), 0))
        self.meta_bootstrap_epochs = int(max(getattr(self, "meta_bootstrap_epochs", 1), 1))
        self.meta_bootstrap_trades = int(
            max(getattr(self, "meta_bootstrap_trades", self.meta_bootstrap_samples), 0)
        )
        self.meta_bootstrap_min_samples = int(max(getattr(self, "meta_bootstrap_min_samples", 64), 16))
        self.meta_bootstrap_edge_cap = float(
            np.clip(getattr(self, "meta_bootstrap_edge_cap", 0.04), 0.001, 1.0)
        )
        self.meta_bootstrap_conf_gain = float(
            np.clip(getattr(self, "meta_bootstrap_conf_gain", 0.12), 0.0, 0.49)
        )
        floor = float(np.clip(getattr(self, "meta_learner_ready_prob_floor", 0.42), 0.0, 1.0))
        ceil = float(np.clip(getattr(self, "meta_learner_ready_prob_ceiling", 0.58), 0.0, 1.0))
        if floor > ceil:
            floor, ceil = ceil, floor
        self.meta_learner_ready_prob_floor = floor
        self.meta_learner_ready_prob_ceiling = ceil

