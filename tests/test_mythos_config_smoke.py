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
