from mythos.config import MythosConfig
from mythos.promotion import evaluate_promotion


def test_mythos_config_defaults_are_sane():
    cfg = MythosConfig()
    assert cfg.train_months > 0
    assert cfg.test_months > 0
    assert cfg.n_regimes >= 2
    assert 0.0 < cfg.min_router_confidence < 1.0
    assert cfg.max_trades_per_day > 0


def test_promotion_gate_smoke():
    good = evaluate_promotion(
        expectancy=0.12,
        win_rate=0.52,
        trades=80,
        min_trades=25,
        edge_drift=0.6,
        score_monotonic=True,
        side_balance=0.35,
    )
    bad = evaluate_promotion(
        expectancy=-0.03,
        win_rate=0.41,
        trades=8,
        min_trades=25,
        edge_drift=2.2,
        score_monotonic=False,
        side_balance=0.02,
    )
    assert good.should_promote is True
    assert bad.should_promote is False


def test_new_bayes_and_nonconformity_knobs_are_normalized():
    cfg = MythosConfig(
        bayes_quality_warmup_trades=0,
        bayes_quality_decay=0.2,
        bayes_quality_prior_alpha=-1.0,
        bayes_quality_prior_beta=-1.0,
        bayes_quality_regime_weight=2.0,
        bayes_quality_min_win_prob=2.0,
        bayes_quality_edge_scale=-1.0,
        bayes_quality_confidence_scale=-1.0,
        bayes_quality_uncertainty_scale=-1.0,
        bayes_quality_reject_margin=2.0,
        nonconformity_warmup_trades=0,
        nonconformity_window=0,
        nonconformity_quantile=0.1,
        nonconformity_margin=-1.0,
        nonconformity_min_winners=0,
        nonconformity_weight_uncertainty=-1.0,
        nonconformity_override_conviction=2.0,
        nonconformity_override_edge_buffer=-1.0,
        nonconformity_override_confidence_buffer=2.0,
    )
    assert cfg.bayes_quality_warmup_trades >= 1
    assert cfg.bayes_quality_decay == 0.9
    assert cfg.bayes_quality_prior_alpha == 0.1
    assert cfg.bayes_quality_prior_beta == 0.1
    assert cfg.bayes_quality_regime_weight == 1.0
    assert cfg.bayes_quality_min_win_prob == 1.0
    assert cfg.bayes_quality_edge_scale == 0.0
    assert cfg.bayes_quality_confidence_scale == 0.0
    assert cfg.bayes_quality_uncertainty_scale == 0.0
    assert cfg.bayes_quality_reject_margin == 0.5
    assert cfg.nonconformity_warmup_trades >= 1
    assert cfg.nonconformity_window >= 8
    assert cfg.nonconformity_quantile == 0.5
    assert cfg.nonconformity_margin == 0.0
    assert cfg.nonconformity_min_winners >= 1
    assert cfg.nonconformity_weight_uncertainty == 0.0
    assert cfg.nonconformity_override_conviction == 1.0
    assert cfg.nonconformity_override_edge_buffer == 0.0
    assert cfg.nonconformity_override_confidence_buffer == 1.0


def test_new_adaptive_reject_and_side_rebalance_knobs_are_normalized():
    cfg = MythosConfig(
        side_rebalance_warmup_trades=0,
        side_rebalance_window=0,
        side_rebalance_short_target=0.9,
        side_rebalance_short_boost=-0.1,
        side_rebalance_long_penalty=-0.2,
        side_rebalance_conf_boost=9.0,
        side_rebalance_quality_guard=9.0,
        side_rebalance_max_adjust=9.0,
        counterfactual_target_reject_rate=2.0,
        counterfactual_reject_tolerance=-1.0,
        counterfactual_adaptive_relax=2.0,
        counterfactual_adaptive_min_adv_floor=0.0,
        nonconformity_target_reject_rate=2.0,
        nonconformity_reject_tolerance=-1.0,
        nonconformity_adaptive_relax=2.0,
        nonconformity_adaptive_max_relax=2.0,
        nonconformity_soft_override_margin=-1.0,
    )
    assert cfg.side_rebalance_warmup_trades >= 1
    assert cfg.side_rebalance_window >= 8
    assert cfg.side_rebalance_short_target == 0.5
    assert cfg.side_rebalance_short_boost == 0.0
    assert cfg.side_rebalance_long_penalty == 0.0
    assert cfg.side_rebalance_conf_boost == 0.2
    assert cfg.side_rebalance_quality_guard == 0.5
    assert cfg.side_rebalance_max_adjust == 0.1
    assert cfg.counterfactual_target_reject_rate == 0.99
    assert cfg.counterfactual_reject_tolerance == 0.0
    assert cfg.counterfactual_adaptive_relax == 1.0
    assert cfg.counterfactual_adaptive_min_adv_floor == 0.05
    assert cfg.nonconformity_target_reject_rate == 0.99
    assert cfg.nonconformity_reject_tolerance == 0.0
    assert cfg.nonconformity_adaptive_relax == 1.0
    assert cfg.nonconformity_adaptive_max_relax == 0.5
    assert cfg.nonconformity_soft_override_margin == 0.0


def test_capital_protection_knobs_are_normalized():
    cfg = MythosConfig(
        emergency_max_drawdown_r=-1.0,
        emergency_equity_floor_r=-999.0,
        drawdown_size_start_r=-1.0,
        drawdown_size_full_r=-2.0,
        drawdown_size_min_scale=-1.0,
        disable_conviction_boost_drawdown_r=-1.0,
        disable_leverage_drawdown_r=-1.0,
    )
    assert cfg.emergency_max_drawdown_r == 0.0
    assert cfg.emergency_equity_floor_r == -200.0
    assert cfg.drawdown_size_start_r == 0.0
    assert cfg.drawdown_size_full_r > cfg.drawdown_size_start_r
    assert cfg.drawdown_size_min_scale == 0.05
    assert cfg.disable_conviction_boost_drawdown_r == 0.0
    assert cfg.disable_leverage_drawdown_r == 0.0


def test_intelligence_knobs_are_normalized():
    cfg = MythosConfig(
        intelligence_min_samples=0,
        intelligence_ema_alpha=2.0,
        intelligence_hit_weight=-1.0,
        intelligence_expectancy_weight=3.0,
        intelligence_variance_penalty=-1.0,
        intelligence_edge_scale=-1.0,
        intelligence_negative_edge_scale=-1.0,
        intelligence_conf_scale=2.0,
        intelligence_uncertainty_scale=3.0,
        intelligence_max_edge_adjust=3.0,
        intelligence_side_switch_min_gap=3.0,
        intelligence_side_switch_min_analog_adv=-1.0,
        intelligence_side_switch_conviction_guard=2.0,
    )
    assert cfg.intelligence_min_samples == 1
    assert cfg.intelligence_ema_alpha == 1.0
    assert cfg.intelligence_hit_weight == 0.0
    assert cfg.intelligence_expectancy_weight == 2.0
    assert cfg.intelligence_variance_penalty == 0.0
    assert cfg.intelligence_edge_scale == 0.0
    assert cfg.intelligence_negative_edge_scale == 0.0
    assert cfg.intelligence_conf_scale == 0.5
    assert cfg.intelligence_uncertainty_scale == 1.5
    assert cfg.intelligence_max_edge_adjust == 0.5
    assert cfg.intelligence_side_switch_min_gap == 2.0
    assert cfg.intelligence_side_switch_min_analog_adv == 0.0
    assert cfg.intelligence_side_switch_conviction_guard == 1.0
