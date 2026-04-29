import numpy as np

from mythos.config import MythosConfig
from mythos.experts import build_experts
from mythos.features import build_feature_frame
from mythos.promotion import evaluate_promotion
from mythos.risk import RiskConstitution
from mythos.router import MetaRouter
from mythos.world_model import WorldModel


def _synthetic_ohlcv_df(n: int = 900):
    import pandas as pd

    rng = np.random.default_rng(11)
    base = np.cumsum(rng.normal(0.0, 14.0, size=n)) + 25000.0
    close = np.maximum(base, 100.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.1, 30.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 30.0, size=n)
    volume = rng.lognormal(mean=10.0, sigma=0.35, size=n)
    ts0 = 1640995200000  # 2022-01-01 UTC in ms
    timestamps = ts0 + np.arange(n) * 15 * 60 * 1000
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(np.int64),
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume.astype(np.float64),
        }
    )


def test_mythos_components_pipeline_shapes_and_types():
    cfg = MythosConfig()
    df = _synthetic_ohlcv_df(1000)
    feat = build_feature_frame(df)
    assert len(feat) > 400

    wm = WorldModel(cfg.random_state)
    wm.fit(feat)
    reg = wm.predict_regime(feat)
    assert reg.shape[0] == len(feat)
    assert np.all(reg >= 0)

    experts = build_experts(cfg.random_state)
    xcols = ["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"]
    X = feat[xcols].to_numpy(dtype=np.float64)
    y1 = feat["fwd_ret_1"].to_numpy(dtype=np.float64)
    y4 = feat["fwd_ret_4"].to_numpy(dtype=np.float64)
    y16 = feat["fwd_ret_16"].to_numpy(dtype=np.float64)
    for ex in experts:
        ex.fit(X, y1, y4, y16, reg)
        p = ex.predict_one(X[0], int(reg[0]))
        assert p.side in (-1, 1)
        assert np.isfinite(p.expected_r)
        assert 0.0 <= p.confidence <= 1.0
        assert p.uncertainty >= 0.0

    router = MetaRouter(cfg.random_state)
    router.fit(feat, reg, experts)
    routed = router.route_one(
        x=X[10],
        regime=int(reg[10]),
        experts=experts,
        vol_16=float(feat.iloc[10]["vol_16"]),
        trend_ema=float(feat.iloc[10]["trend_ema"]),
    )
    assert routed["side"] in (-1, 0, 1)
    assert np.isfinite(float(routed["edge"]))
    assert np.isfinite(float(routed["confidence"]))
    assert np.isfinite(float(routed["uncertainty"]))

    risk = RiskConstitution(cfg)
    allowed = risk.allow_trade(
        ts_ms=int(feat.iloc[10]["timestamp"]),
        side=int(routed["side"]),
        edge=float(routed["edge"]),
        uncertainty=float(routed["uncertainty"]),
    )
    assert isinstance(allowed, bool)
    mult = risk.position_size_multiplier(
        edge=float(routed["edge"]),
        uncertainty=float(routed["uncertainty"]),
        regime=int(reg[10]),
    )
    assert np.isfinite(mult)


def test_promotion_gate_evaluate_pass_and_fail():
    ok = evaluate_promotion(
        expectancy=0.12,
        win_rate=0.42,
        trades=120,
        min_trades=25,
        edge_drift=0.6,
        score_monotonic=True,
        side_balance=0.45,
    )
    bad = evaluate_promotion(
        expectancy=-0.01,
        win_rate=0.22,
        trades=12,
        min_trades=25,
        edge_drift=1.8,
        score_monotonic=False,
        side_balance=0.02,
    )
    assert ok.should_promote is True
    assert bad.should_promote is False
    assert len(bad.reasons) > 0
