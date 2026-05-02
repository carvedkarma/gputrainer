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
