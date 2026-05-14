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
    side_rebalance_enable: bool = True
    side_rebalance_warmup_trades: int = 40
    side_rebalance_window: int = 96
    side_rebalance_short_target: float = 0.32
    side_rebalance_short_boost: float = 0.0035
    side_rebalance_long_penalty: float = 0.0030
    side_rebalance_conf_boost: float = 0.02
    side_rebalance_quality_guard: float = 0.06
    side_rebalance_max_adjust: float = 0.012
    intelligence_enable: bool = True
    intelligence_min_samples: int = 24
    intelligence_ema_alpha: float = 0.08
    intelligence_hit_weight: float = 0.55
    intelligence_expectancy_weight: float = 0.45
    intelligence_variance_penalty: float = 0.18
    intelligence_edge_scale: float = 0.010
    intelligence_negative_edge_scale: float = 0.012
    intelligence_conf_scale: float = 0.06
    intelligence_uncertainty_scale: float = 0.30
    intelligence_max_edge_adjust: float = 0.020
    intelligence_side_switch_enable: bool = True
    intelligence_side_switch_min_gap: float = 0.30
    intelligence_side_switch_min_analog_adv: float = 0.0015
    intelligence_side_switch_conviction_guard: float = 0.58
    intelligence_side_switch_warmup_trades: int = 40
    intelligence_side_switch_min_samples: int = 48
    intelligence_side_switch_cooldown_bars: int = 48
    intelligence_side_switch_max_rate: float = 0.20
    adaptive_side_target_strength: float = 0.22
    adaptive_side_target_min: float = 0.35
    adaptive_side_target_max: float = 0.65
    side_health_penalty: float = 0.02
    side_health_boost: float = 0.008
    side_health_decay: float = 0.97
    side_aggression_boost: float = 0.006
    side_aggression_min_gap: float = 0.05
    side_aggression_underweight_gain: float = 0.60
    adaptive_tp_sl_enable: bool = True
    adaptive_tp_min_mult: float = 1.2
    adaptive_tp_max_mult: float = 3.6
    adaptive_sl_min_mult: float = 0.8
    adaptive_sl_max_mult: float = 2.4
    adaptive_tp_quality_gain: float = 0.35
    adaptive_tp_trend_gain: float = 0.20
    adaptive_tp_vol_penalty: float = 0.18
    adaptive_sl_quality_tighten: float = 0.25
    adaptive_sl_uncertainty_widen: float = 0.30
    adaptive_sl_vol_widen: float = 0.20
    adaptive_short_tp_bias: float = 0.05
    adaptive_short_sl_bias: float = 0.04
    time_adaptive_enable: bool = True
    time_adaptive_switch_enable: bool = True
    time_adaptive_warmup_trades: int = 36
    time_adaptive_window: int = 240
    time_adaptive_min_bucket_trades: int = 8
    time_adaptive_edge_scale: float = 0.010
    time_adaptive_conf_scale: float = 0.05
    time_adaptive_max_edge_adjust: float = 0.018
    time_adaptive_switch_min_gap_r: float = 0.04
    time_adaptive_switch_conviction_guard: float = 0.62
    time_adaptive_switch_min_samples: int = 10
    time_adaptive_report_top_n: int = 6
    precision_selective_enable: bool = False
    precision_selective_min_trades: int = 48
    precision_selective_score_window: int = 512
    precision_selective_score_min_samples: int = 128
    precision_selective_base_quantile: float = 0.70
    precision_selective_max_quantile: float = 0.95
    precision_selective_target_win_rate: float = 0.52
    precision_selective_adapt_gain: float = 0.40
    precision_selective_edge_weight: float = 0.45
    precision_selective_conf_weight: float = 0.35
    precision_selective_uncertainty_weight: float = 0.20
    precision_selective_conviction_weight: float = 0.25
    robust_validation_enable: bool = True
    robust_validation_min_folds: int = 5
    robust_validation_metric: str = "expectancy_r"
    cpcv_test_fraction: float = 0.40
    cpcv_max_paths: int = 256
    cpcv_random_seed: int = 42
    robust_validation_trial_count: int = 8
    robust_validation_sr_benchmark: float = 0.0
    robust_validation_spa_bootstrap_samples: int = 400
    robust_validation_report_top_paths: int = 5
    opportunity_rescue_enable: bool = True
    opportunity_rescue_start_bars: int = 96
    opportunity_rescue_full_bars: int = 384
    opportunity_rescue_edge_relax: float = 0.010
    opportunity_rescue_conf_relax: float = 0.08
    opportunity_rescue_min_edge: float = 0.004
    opportunity_rescue_min_confidence: float = 0.47
    opportunity_rescue_override_start: float = 0.55
    opportunity_rescue_override_conviction: float = 0.70
    opportunity_rescue_override_edge_buffer: float = 0.002
    opportunity_rescue_override_conf_buffer: float = 0.02
    participation_adapt_enable: bool = True
    participation_target_trades_per_fold: int = 40
    participation_relax_start_progress: float = 0.25
    participation_conf_relax_max: float = 0.10
    participation_edge_relax_max: float = 0.008
    participation_conviction_relax_max: float = 0.08
    participation_min_confidence: float = 0.44
    participation_min_edge: float = 0.003
    participation_min_conviction: float = 0.46
    participation_override_conviction: float = 0.74
    participation_override_edge_buffer: float = 0.0015
    participation_override_conf_buffer: float = 0.01
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
    leverage_policy_window: int = 160
    leverage_policy_min_trades: int = 24
    leverage_policy_min_hit_rate: float = 0.54
    leverage_policy_min_expectancy: float = 0.06
    leverage_policy_context_weight: float = 0.60
    leverage_policy_cold_start_conviction_extra: float = 0.08
    leverage_net_edge_floor: float = 0.002
    leverage_side_policy_enable: bool = True
    leverage_side_min_trades: int = 12
    leverage_side_min_hit_rate: float = 0.52
    leverage_side_min_expectancy: float = 0.03
    execution_fee_bps: float = 4.0
    execution_slippage_bps: float = 2.0
    execution_cost_cap_r: float = 0.35
    # Capital-protection controls (CLI names preserved for compatibility).
    emergency_stop_r: float = -35.0
    dd_size_throttle_start_r: float = 10.0
    dd_size_throttle_end_r: float = 24.0
    dd_size_throttle_min: float = 0.40
    dd_disable_leverage_r: float = 12.0
    dd_risk_recovery_r: float = 6.0
    # Internal/legacy aliases kept for backward compatibility.
    emergency_stop_enable: bool = True
    emergency_max_drawdown_r: float = 12.0
    emergency_equity_floor_r: float = -18.0
    drawdown_size_start_r: float = 6.0
    drawdown_size_full_r: float = 14.0
    drawdown_size_min_scale: float = 0.35
    disable_conviction_boost_drawdown_r: float = 6.0
    disable_leverage_drawdown_r: float = 6.0
    risk_state_normalize_by_size: bool = True
    risk_state_min_size_for_norm: float = 1.0
    risk_state_max_size_for_norm: float = 250.0
    risk_cap_override_enable: bool = True
    risk_cap_override_conviction: float = 0.80
    risk_cap_override_edge_buffer: float = 0.002
    risk_cap_override_max_uncertainty: float = 0.70
    bayes_quality_enable: bool = True
    bayes_quality_warmup_trades: int = 20
    bayes_quality_decay: float = 0.995
    bayes_quality_prior_alpha: float = 2.0
    bayes_quality_prior_beta: float = 2.0
    bayes_quality_regime_weight: float = 0.45
    bayes_quality_min_win_prob: float = 0.50
    bayes_quality_min_expectancy: float = -0.01
    bayes_quality_edge_scale: float = 0.22
    bayes_quality_confidence_scale: float = 0.10
    bayes_quality_uncertainty_scale: float = 0.18
    bayes_quality_reject_margin: float = 0.05
    nonconformity_enable: bool = True
    nonconformity_warmup_trades: int = 24
    nonconformity_window: int = 160
    nonconformity_quantile: float = 0.86
    nonconformity_margin: float = 0.03
    nonconformity_min_winners: int = 16
    nonconformity_weight_uncertainty: float = 0.36
    nonconformity_weight_confidence: float = 0.22
    nonconformity_weight_edge: float = 0.20
    nonconformity_weight_meta: float = 0.14
    nonconformity_weight_analog: float = 0.08
    nonconformity_override_conviction: float = 0.88
    nonconformity_override_edge_buffer: float = 0.003
    nonconformity_override_confidence_buffer: float = 0.04
    nonconformity_target_reject_rate: float = 0.48
    nonconformity_reject_tolerance: float = 0.12
    nonconformity_adaptive_relax: float = 0.16
    nonconformity_adaptive_max_relax: float = 0.18
    nonconformity_soft_override_margin: float = 0.04
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
    counterfactual_target_reject_rate: float = 0.70
    counterfactual_reject_tolerance: float = 0.10
    counterfactual_adaptive_relax: float = 0.35
    counterfactual_adaptive_min_adv_floor: float = 0.25

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
        # Normalize leverage/sizing rails so runtime clamps remain coherent.
        self.max_leverage = float(max(getattr(self, "max_leverage", 1.8), 0.1))
        self.min_size_mult = float(max(getattr(self, "min_size_mult", 0.5), 0.0))
        self.max_size_mult = float(
            max(
                getattr(self, "max_size_mult", self.max_leverage),
                self.min_size_mult,
                self.max_leverage,
            )
        )
        # In high-leverage profiles, trade counts are naturally lower; relax
        # warmup/sample gates so adaptive modules can still activate.
        if self.min_size_mult >= 10.0:
            self.intelligence_min_samples = int(min(max(getattr(self, "intelligence_min_samples", 24), 1), 12))
            self.time_adaptive_warmup_trades = int(
                min(max(getattr(self, "time_adaptive_warmup_trades", 36), 0), 12)
            )
            self.time_adaptive_min_bucket_trades = int(
                min(max(getattr(self, "time_adaptive_min_bucket_trades", 8), 1), 4)
            )
            self.precision_selective_min_trades = int(
                min(max(getattr(self, "precision_selective_min_trades", 48), 1), 18)
            )
            self.precision_selective_score_min_samples = int(
                min(max(getattr(self, "precision_selective_score_min_samples", 128), 16), 64)
            )
        self.adaptive_side_target_strength = float(
            np.clip(getattr(self, "adaptive_side_target_strength", 0.22), 0.0, 1.0)
        )
        self.side_rebalance_enable = bool(getattr(self, "side_rebalance_enable", True))
        self.side_rebalance_warmup_trades = int(max(getattr(self, "side_rebalance_warmup_trades", 40), 1))
        self.side_rebalance_window = int(max(getattr(self, "side_rebalance_window", 96), 8))
        self.side_rebalance_short_target = float(
            np.clip(getattr(self, "side_rebalance_short_target", 0.32), 0.05, 0.50)
        )
        self.side_rebalance_short_boost = float(
            np.clip(getattr(self, "side_rebalance_short_boost", 0.0035), 0.0, 0.05)
        )
        self.side_rebalance_long_penalty = float(
            np.clip(getattr(self, "side_rebalance_long_penalty", 0.0030), 0.0, 0.05)
        )
        self.side_rebalance_conf_boost = float(
            np.clip(getattr(self, "side_rebalance_conf_boost", 0.02), 0.0, 0.20)
        )
        self.side_rebalance_quality_guard = float(
            np.clip(getattr(self, "side_rebalance_quality_guard", 0.06), 0.0, 0.50)
        )
        self.side_rebalance_max_adjust = float(
            np.clip(getattr(self, "side_rebalance_max_adjust", 0.012), 0.0, 0.10)
        )
        self.intelligence_enable = bool(getattr(self, "intelligence_enable", True))
        self.intelligence_min_samples = int(max(getattr(self, "intelligence_min_samples", 24), 1))
        self.intelligence_ema_alpha = float(
            np.clip(getattr(self, "intelligence_ema_alpha", 0.08), 0.01, 1.0)
        )
        self.intelligence_hit_weight = float(
            np.clip(getattr(self, "intelligence_hit_weight", 0.55), 0.0, 2.0)
        )
        self.intelligence_expectancy_weight = float(
            np.clip(getattr(self, "intelligence_expectancy_weight", 0.45), 0.0, 2.0)
        )
        self.intelligence_variance_penalty = float(
            np.clip(getattr(self, "intelligence_variance_penalty", 0.18), 0.0, 2.0)
        )
        self.intelligence_edge_scale = float(
            np.clip(getattr(self, "intelligence_edge_scale", 0.010), 0.0, 0.20)
        )
        self.intelligence_negative_edge_scale = float(
            np.clip(getattr(self, "intelligence_negative_edge_scale", 0.012), 0.0, 0.20)
        )
        self.intelligence_conf_scale = float(
            np.clip(getattr(self, "intelligence_conf_scale", 0.06), 0.0, 0.50)
        )
        self.intelligence_uncertainty_scale = float(
            np.clip(getattr(self, "intelligence_uncertainty_scale", 0.30), 0.0, 1.50)
        )
        self.intelligence_max_edge_adjust = float(
            np.clip(getattr(self, "intelligence_max_edge_adjust", 0.020), 0.0, 0.50)
        )
        self.intelligence_side_switch_enable = bool(
            getattr(self, "intelligence_side_switch_enable", True)
        )
        self.intelligence_side_switch_min_gap = float(
            np.clip(getattr(self, "intelligence_side_switch_min_gap", 0.30), 0.0, 2.0)
        )
        self.intelligence_side_switch_min_analog_adv = float(
            np.clip(getattr(self, "intelligence_side_switch_min_analog_adv", 0.0015), 0.0, 0.50)
        )
        self.intelligence_side_switch_conviction_guard = float(
            np.clip(getattr(self, "intelligence_side_switch_conviction_guard", 0.58), 0.0, 1.0)
        )
        self.intelligence_side_switch_warmup_trades = int(
            max(getattr(self, "intelligence_side_switch_warmup_trades", 40), 0)
        )
        self.intelligence_side_switch_min_samples = int(
            max(getattr(self, "intelligence_side_switch_min_samples", 48), 1)
        )
        self.intelligence_side_switch_cooldown_bars = int(
            max(getattr(self, "intelligence_side_switch_cooldown_bars", 48), 0)
        )
        self.intelligence_side_switch_max_rate = float(
            np.clip(getattr(self, "intelligence_side_switch_max_rate", 0.20), 0.0, 1.0)
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
        self.adaptive_tp_sl_enable = bool(getattr(self, "adaptive_tp_sl_enable", True))
        self.adaptive_tp_min_mult = float(
            np.clip(getattr(self, "adaptive_tp_min_mult", 1.2), 0.10, 20.0)
        )
        self.adaptive_tp_max_mult = float(
            max(getattr(self, "adaptive_tp_max_mult", 3.6), self.adaptive_tp_min_mult + 1e-6)
        )
        self.adaptive_sl_min_mult = float(
            np.clip(getattr(self, "adaptive_sl_min_mult", 0.8), 0.10, 20.0)
        )
        self.adaptive_sl_max_mult = float(
            max(getattr(self, "adaptive_sl_max_mult", 2.4), self.adaptive_sl_min_mult + 1e-6)
        )
        self.adaptive_tp_quality_gain = float(
            np.clip(getattr(self, "adaptive_tp_quality_gain", 0.35), 0.0, 2.0)
        )
        self.adaptive_tp_trend_gain = float(
            np.clip(getattr(self, "adaptive_tp_trend_gain", 0.20), 0.0, 2.0)
        )
        self.adaptive_tp_vol_penalty = float(
            np.clip(getattr(self, "adaptive_tp_vol_penalty", 0.18), 0.0, 2.0)
        )
        self.adaptive_sl_quality_tighten = float(
            np.clip(getattr(self, "adaptive_sl_quality_tighten", 0.25), 0.0, 2.0)
        )
        self.adaptive_sl_uncertainty_widen = float(
            np.clip(getattr(self, "adaptive_sl_uncertainty_widen", 0.30), 0.0, 2.0)
        )
        self.adaptive_sl_vol_widen = float(
            np.clip(getattr(self, "adaptive_sl_vol_widen", 0.20), 0.0, 2.0)
        )
        self.adaptive_short_tp_bias = float(
            np.clip(getattr(self, "adaptive_short_tp_bias", 0.05), -1.0, 1.0)
        )
        self.adaptive_short_sl_bias = float(
            np.clip(getattr(self, "adaptive_short_sl_bias", 0.04), -1.0, 1.0)
        )
        self.time_adaptive_enable = bool(getattr(self, "time_adaptive_enable", True))
        self.time_adaptive_switch_enable = bool(
            getattr(self, "time_adaptive_switch_enable", True)
        )
        self.time_adaptive_warmup_trades = int(
            max(getattr(self, "time_adaptive_warmup_trades", 36), 0)
        )
        self.time_adaptive_window = int(max(getattr(self, "time_adaptive_window", 240), 8))
        self.time_adaptive_min_bucket_trades = int(
            max(getattr(self, "time_adaptive_min_bucket_trades", 8), 1)
        )
        self.time_adaptive_edge_scale = float(
            np.clip(getattr(self, "time_adaptive_edge_scale", 0.010), 0.0, 0.25)
        )
        self.time_adaptive_conf_scale = float(
            np.clip(getattr(self, "time_adaptive_conf_scale", 0.05), 0.0, 1.0)
        )
        self.time_adaptive_max_edge_adjust = float(
            np.clip(getattr(self, "time_adaptive_max_edge_adjust", 0.018), 0.0, 0.50)
        )
        self.time_adaptive_switch_min_gap_r = float(
            np.clip(getattr(self, "time_adaptive_switch_min_gap_r", 0.04), 0.0, 2.0)
        )
        self.time_adaptive_switch_conviction_guard = float(
            np.clip(getattr(self, "time_adaptive_switch_conviction_guard", 0.62), 0.0, 1.0)
        )
        self.time_adaptive_switch_min_samples = int(
            max(getattr(self, "time_adaptive_switch_min_samples", 10), 1)
        )
        self.time_adaptive_report_top_n = int(max(getattr(self, "time_adaptive_report_top_n", 6), 1))
        self.precision_selective_enable = bool(getattr(self, "precision_selective_enable", False))
        self.precision_selective_min_trades = int(
            max(getattr(self, "precision_selective_min_trades", 48), 0)
        )
        self.precision_selective_score_window = int(
            max(getattr(self, "precision_selective_score_window", 512), 32)
        )
        self.precision_selective_score_min_samples = int(
            max(getattr(self, "precision_selective_score_min_samples", 128), 16)
        )
        self.precision_selective_base_quantile = float(
            np.clip(getattr(self, "precision_selective_base_quantile", 0.70), 0.50, 0.999)
        )
        self.precision_selective_max_quantile = float(
            np.clip(getattr(self, "precision_selective_max_quantile", 0.95), 0.50, 0.999)
        )
        if self.precision_selective_max_quantile < self.precision_selective_base_quantile:
            self.precision_selective_max_quantile = self.precision_selective_base_quantile
        self.precision_selective_target_win_rate = float(
            np.clip(getattr(self, "precision_selective_target_win_rate", 0.52), 0.0, 1.0)
        )
        self.precision_selective_adapt_gain = float(
            np.clip(getattr(self, "precision_selective_adapt_gain", 0.40), 0.0, 2.0)
        )
        self.precision_selective_edge_weight = float(
            np.clip(getattr(self, "precision_selective_edge_weight", 0.45), 0.0, 5.0)
        )
        self.precision_selective_conf_weight = float(
            np.clip(getattr(self, "precision_selective_conf_weight", 0.35), 0.0, 5.0)
        )
        self.precision_selective_uncertainty_weight = float(
            np.clip(getattr(self, "precision_selective_uncertainty_weight", 0.20), 0.0, 5.0)
        )
        self.precision_selective_conviction_weight = float(
            np.clip(getattr(self, "precision_selective_conviction_weight", 0.25), 0.0, 5.0)
        )
        self.robust_validation_enable = bool(getattr(self, "robust_validation_enable", True))
        self.robust_validation_min_folds = int(
            max(getattr(self, "robust_validation_min_folds", 5), 3)
        )
        metric = str(getattr(self, "robust_validation_metric", "expectancy_r")).strip().lower()
        if metric not in {"total_r", "expectancy_r", "win_rate", "robust_score"}:
            metric = "expectancy_r"
        self.robust_validation_metric = metric
        self.cpcv_test_fraction = float(
            np.clip(getattr(self, "cpcv_test_fraction", 0.40), 0.10, 0.90)
        )
        self.cpcv_max_paths = int(max(getattr(self, "cpcv_max_paths", 256), 1))
        self.cpcv_random_seed = int(max(getattr(self, "cpcv_random_seed", 42), 0))
        self.robust_validation_trial_count = int(
            max(getattr(self, "robust_validation_trial_count", 8), 1)
        )
        self.robust_validation_sr_benchmark = float(
            getattr(self, "robust_validation_sr_benchmark", 0.0)
        )
        self.robust_validation_spa_bootstrap_samples = int(
            np.clip(getattr(self, "robust_validation_spa_bootstrap_samples", 400), 32, 5000)
        )
        self.robust_validation_report_top_paths = int(
            max(getattr(self, "robust_validation_report_top_paths", 5), 1)
        )
        self.opportunity_rescue_enable = bool(getattr(self, "opportunity_rescue_enable", True))
        self.opportunity_rescue_start_bars = int(max(getattr(self, "opportunity_rescue_start_bars", 96), 1))
        self.opportunity_rescue_full_bars = int(
            max(getattr(self, "opportunity_rescue_full_bars", 384), self.opportunity_rescue_start_bars + 1)
        )
        self.opportunity_rescue_edge_relax = float(
            np.clip(getattr(self, "opportunity_rescue_edge_relax", 0.010), 0.0, 0.25)
        )
        self.opportunity_rescue_conf_relax = float(
            np.clip(getattr(self, "opportunity_rescue_conf_relax", 0.08), 0.0, 0.50)
        )
        self.opportunity_rescue_min_edge = float(
            max(getattr(self, "opportunity_rescue_min_edge", 0.004), 0.0)
        )
        self.opportunity_rescue_min_confidence = float(
            np.clip(getattr(self, "opportunity_rescue_min_confidence", 0.47), 0.0, 1.0)
        )
        self.opportunity_rescue_override_start = float(
            np.clip(getattr(self, "opportunity_rescue_override_start", 0.55), 0.0, 1.0)
        )
        self.opportunity_rescue_override_conviction = float(
            np.clip(getattr(self, "opportunity_rescue_override_conviction", 0.70), 0.0, 1.0)
        )
        self.opportunity_rescue_override_edge_buffer = float(
            max(getattr(self, "opportunity_rescue_override_edge_buffer", 0.002), 0.0)
        )
        self.opportunity_rescue_override_conf_buffer = float(
            np.clip(getattr(self, "opportunity_rescue_override_conf_buffer", 0.02), 0.0, 1.0)
        )
        self.participation_adapt_enable = bool(getattr(self, "participation_adapt_enable", True))
        self.participation_target_trades_per_fold = int(
            max(getattr(self, "participation_target_trades_per_fold", 40), 1)
        )
        self.participation_relax_start_progress = float(
            np.clip(getattr(self, "participation_relax_start_progress", 0.25), 0.0, 1.0)
        )
        self.participation_conf_relax_max = float(
            np.clip(getattr(self, "participation_conf_relax_max", 0.10), 0.0, 0.60)
        )
        self.participation_edge_relax_max = float(
            np.clip(getattr(self, "participation_edge_relax_max", 0.008), 0.0, 0.20)
        )
        self.participation_conviction_relax_max = float(
            np.clip(getattr(self, "participation_conviction_relax_max", 0.08), 0.0, 0.50)
        )
        self.participation_min_confidence = float(
            np.clip(getattr(self, "participation_min_confidence", 0.44), 0.0, 1.0)
        )
        self.participation_min_edge = float(max(getattr(self, "participation_min_edge", 0.003), 0.0))
        self.participation_min_conviction = float(
            np.clip(getattr(self, "participation_min_conviction", 0.46), 0.0, 1.0)
        )
        self.participation_override_conviction = float(
            np.clip(getattr(self, "participation_override_conviction", 0.74), 0.0, 1.0)
        )
        self.participation_override_edge_buffer = float(
            max(getattr(self, "participation_override_edge_buffer", 0.0015), 0.0)
        )
        self.participation_override_conf_buffer = float(
            np.clip(getattr(self, "participation_override_conf_buffer", 0.01), 0.0, 1.0)
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
        self.leverage_policy_window = int(max(getattr(self, "leverage_policy_window", 160), 8))
        self.leverage_policy_min_trades = int(max(getattr(self, "leverage_policy_min_trades", 24), 1))
        self.leverage_policy_min_hit_rate = float(
            np.clip(getattr(self, "leverage_policy_min_hit_rate", 0.54), 0.0, 1.0)
        )
        self.leverage_policy_min_expectancy = float(
            getattr(self, "leverage_policy_min_expectancy", 0.06)
        )
        self.leverage_policy_context_weight = float(
            np.clip(getattr(self, "leverage_policy_context_weight", 0.60), 0.0, 1.0)
        )
        self.leverage_policy_cold_start_conviction_extra = float(
            np.clip(getattr(self, "leverage_policy_cold_start_conviction_extra", 0.08), 0.0, 0.5)
        )
        self.leverage_net_edge_floor = float(max(getattr(self, "leverage_net_edge_floor", 0.002), 0.0))
        self.leverage_side_policy_enable = bool(getattr(self, "leverage_side_policy_enable", True))
        self.leverage_side_min_trades = int(max(getattr(self, "leverage_side_min_trades", 12), 1))
        self.leverage_side_min_hit_rate = float(
            np.clip(getattr(self, "leverage_side_min_hit_rate", 0.52), 0.0, 1.0)
        )
        self.leverage_side_min_expectancy = float(
            getattr(self, "leverage_side_min_expectancy", 0.03)
        )
        self.execution_fee_bps = float(max(getattr(self, "execution_fee_bps", 4.0), 0.0))
        self.execution_slippage_bps = float(max(getattr(self, "execution_slippage_bps", 2.0), 0.0))
        self.execution_cost_cap_r = float(
            np.clip(getattr(self, "execution_cost_cap_r", 0.35), 0.0, 5.0)
        )
        self.emergency_stop_r = float(getattr(self, "emergency_stop_r", -35.0))
        raw_dd_start = float(getattr(self, "dd_size_throttle_start_r", 10.0))
        raw_drawdown_start = float(getattr(self, "drawdown_size_start_r", 6.0))
        if abs(raw_dd_start - 10.0) <= 1e-12 and abs(raw_drawdown_start - 6.0) > 1e-12:
            raw_dd_start = raw_drawdown_start
        self.dd_size_throttle_start_r = float(max(raw_dd_start, 0.0))
        raw_dd_end = float(getattr(self, "dd_size_throttle_end_r", 24.0))
        raw_drawdown_end = float(getattr(self, "drawdown_size_full_r", 14.0))
        if abs(raw_dd_end - 24.0) <= 1e-12 and abs(raw_drawdown_end - 14.0) > 1e-12:
            raw_dd_end = raw_drawdown_end
        self.dd_size_throttle_end_r = float(max(raw_dd_end, self.dd_size_throttle_start_r + 1e-6))
        raw_dd_min = float(getattr(self, "dd_size_throttle_min", 0.40))
        raw_drawdown_min = float(getattr(self, "drawdown_size_min_scale", 0.35))
        if abs(raw_dd_min - 0.40) <= 1e-12 and abs(raw_drawdown_min - 0.35) > 1e-12:
            raw_dd_min = raw_drawdown_min
        self.dd_size_throttle_min = float(np.clip(raw_dd_min, 0.05, 1.0))
        raw_dd_disable_lev = float(getattr(self, "dd_disable_leverage_r", 12.0))
        raw_disable_lev = float(getattr(self, "disable_leverage_drawdown_r", 6.0))
        if abs(raw_dd_disable_lev - 12.0) <= 1e-12 and abs(raw_disable_lev - 6.0) > 1e-12:
            raw_dd_disable_lev = raw_disable_lev
        self.dd_disable_leverage_r = float(max(raw_dd_disable_lev, 0.0))
        self.dd_risk_recovery_r = float(
            np.clip(
                getattr(self, "dd_risk_recovery_r", max(0.5 * self.dd_disable_leverage_r, 0.0)),
                0.0,
                max(self.dd_disable_leverage_r, 1e-6),
            )
        )
        self.size_throttle_dd_start_r = float(self.dd_size_throttle_start_r)
        self.size_throttle_dd_max_r = float(self.dd_size_throttle_end_r)
        self.size_throttle_min_fraction = float(self.dd_size_throttle_min)
        self.emergency_stop_enable = bool(getattr(self, "emergency_stop_enable", True))
        self.emergency_max_drawdown_r = float(max(getattr(self, "emergency_max_drawdown_r", 12.0), 0.0))
        self.emergency_equity_floor_r = float(
            np.clip(getattr(self, "emergency_equity_floor_r", self.emergency_stop_r), -200.0, 0.0)
        )
        self.drawdown_size_start_r = float(self.dd_size_throttle_start_r)
        self.drawdown_size_full_r = float(self.dd_size_throttle_end_r)
        self.drawdown_size_min_scale = float(self.dd_size_throttle_min)
        self.disable_conviction_boost_drawdown_r = float(
            max(getattr(self, "disable_conviction_boost_drawdown_r", self.dd_disable_leverage_r), 0.0)
        )
        self.disable_leverage_drawdown_r = float(self.dd_disable_leverage_r)
        self.risk_state_normalize_by_size = bool(getattr(self, "risk_state_normalize_by_size", True))
        self.risk_state_min_size_for_norm = float(
            max(getattr(self, "risk_state_min_size_for_norm", 1.0), 1e-6)
        )
        self.risk_state_max_size_for_norm = float(
            max(
                getattr(self, "risk_state_max_size_for_norm", 250.0),
                self.risk_state_min_size_for_norm,
            )
        )
        self.risk_cap_override_enable = bool(getattr(self, "risk_cap_override_enable", True))
        self.risk_cap_override_conviction = float(
            np.clip(getattr(self, "risk_cap_override_conviction", 0.80), 0.0, 1.0)
        )
        self.risk_cap_override_edge_buffer = float(
            max(getattr(self, "risk_cap_override_edge_buffer", 0.002), 0.0)
        )
        self.risk_cap_override_max_uncertainty = float(
            np.clip(getattr(self, "risk_cap_override_max_uncertainty", 0.70), 0.01, 5.0)
        )
        self.bayes_quality_enable = bool(getattr(self, "bayes_quality_enable", True))
        self.bayes_quality_warmup_trades = int(max(getattr(self, "bayes_quality_warmup_trades", 20), 1))
        self.bayes_quality_decay = float(
            np.clip(getattr(self, "bayes_quality_decay", 0.995), 0.90, 1.0)
        )
        self.bayes_quality_prior_alpha = float(
            np.clip(getattr(self, "bayes_quality_prior_alpha", 2.0), 0.10, 100.0)
        )
        self.bayes_quality_prior_beta = float(
            np.clip(getattr(self, "bayes_quality_prior_beta", 2.0), 0.10, 100.0)
        )
        self.bayes_quality_regime_weight = float(
            np.clip(getattr(self, "bayes_quality_regime_weight", 0.45), 0.0, 1.0)
        )
        self.bayes_quality_min_win_prob = float(
            np.clip(getattr(self, "bayes_quality_min_win_prob", 0.50), 0.0, 1.0)
        )
        self.bayes_quality_min_expectancy = float(
            getattr(self, "bayes_quality_min_expectancy", -0.01)
        )
        self.bayes_quality_edge_scale = float(
            np.clip(getattr(self, "bayes_quality_edge_scale", 0.22), 0.0, 2.0)
        )
        self.bayes_quality_confidence_scale = float(
            np.clip(getattr(self, "bayes_quality_confidence_scale", 0.10), 0.0, 1.0)
        )
        self.bayes_quality_uncertainty_scale = float(
            np.clip(getattr(self, "bayes_quality_uncertainty_scale", 0.18), 0.0, 2.0)
        )
        self.bayes_quality_reject_margin = float(
            np.clip(getattr(self, "bayes_quality_reject_margin", 0.05), 0.0, 0.5)
        )
        self.nonconformity_enable = bool(getattr(self, "nonconformity_enable", True))
        self.nonconformity_warmup_trades = int(max(getattr(self, "nonconformity_warmup_trades", 24), 1))
        self.nonconformity_window = int(max(getattr(self, "nonconformity_window", 160), 8))
        self.nonconformity_quantile = float(
            np.clip(getattr(self, "nonconformity_quantile", 0.86), 0.50, 0.99)
        )
        self.nonconformity_margin = float(
            np.clip(getattr(self, "nonconformity_margin", 0.03), 0.0, 1.0)
        )
        self.nonconformity_min_winners = int(max(getattr(self, "nonconformity_min_winners", 16), 1))
        self.nonconformity_weight_uncertainty = float(
            max(getattr(self, "nonconformity_weight_uncertainty", 0.36), 0.0)
        )
        self.nonconformity_weight_confidence = float(
            max(getattr(self, "nonconformity_weight_confidence", 0.22), 0.0)
        )
        self.nonconformity_weight_edge = float(
            max(getattr(self, "nonconformity_weight_edge", 0.20), 0.0)
        )
        self.nonconformity_weight_meta = float(
            max(getattr(self, "nonconformity_weight_meta", 0.14), 0.0)
        )
        self.nonconformity_weight_analog = float(
            max(getattr(self, "nonconformity_weight_analog", 0.08), 0.0)
        )
        self.nonconformity_override_conviction = float(
            np.clip(getattr(self, "nonconformity_override_conviction", 0.88), 0.0, 1.0)
        )
        self.nonconformity_override_edge_buffer = float(
            max(getattr(self, "nonconformity_override_edge_buffer", 0.003), 0.0)
        )
        self.nonconformity_override_confidence_buffer = float(
            np.clip(getattr(self, "nonconformity_override_confidence_buffer", 0.04), 0.0, 1.0)
        )
        self.nonconformity_target_reject_rate = float(
            np.clip(getattr(self, "nonconformity_target_reject_rate", 0.48), 0.0, 0.99)
        )
        self.nonconformity_reject_tolerance = float(
            np.clip(getattr(self, "nonconformity_reject_tolerance", 0.12), 0.0, 0.5)
        )
        self.nonconformity_adaptive_relax = float(
            np.clip(getattr(self, "nonconformity_adaptive_relax", 0.16), 0.0, 1.0)
        )
        self.nonconformity_adaptive_max_relax = float(
            np.clip(getattr(self, "nonconformity_adaptive_max_relax", 0.18), 0.0, 0.5)
        )
        self.nonconformity_soft_override_margin = float(
            np.clip(getattr(self, "nonconformity_soft_override_margin", 0.04), 0.0, 0.5)
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
        self.counterfactual_target_reject_rate = float(
            np.clip(getattr(self, "counterfactual_target_reject_rate", 0.70), 0.0, 0.99)
        )
        self.counterfactual_reject_tolerance = float(
            np.clip(getattr(self, "counterfactual_reject_tolerance", 0.10), 0.0, 0.5)
        )
        self.counterfactual_adaptive_relax = float(
            np.clip(getattr(self, "counterfactual_adaptive_relax", 0.35), 0.0, 1.0)
        )
        self.counterfactual_adaptive_min_adv_floor = float(
            np.clip(getattr(self, "counterfactual_adaptive_min_adv_floor", 0.25), 0.05, 1.0)
        )
        floor = float(np.clip(getattr(self, "meta_learner_ready_prob_floor", 0.42), 0.0, 1.0))
        ceil = float(np.clip(getattr(self, "meta_learner_ready_prob_ceiling", 0.58), 0.0, 1.0))
        if floor > ceil:
            floor, ceil = ceil, floor
        self.meta_learner_ready_prob_floor = floor
        self.meta_learner_ready_prob_ceiling = ceil

