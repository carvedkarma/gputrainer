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
    side_fail_window: int = 48
    side_fail_min_trades: int = 10
    side_fail_expectancy_r: float = -0.12
    side_fail_cooldown_bars: int = 24
    side_fail_ema_alpha: float = 0.25
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
    # Aliases used by CLI for readability.
    meta_learner_edge_gain: float = 0.30
    meta_learner_confidence_gain: float = 0.20
    meta_learner_min_side_prob: float = 0.50
    meta_learner_min_train_samples: int = 512
    meta_learner_warmup_samples: int = 1024
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

