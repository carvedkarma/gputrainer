import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from mythos.config import MythosConfig
from mythos.experts import build_experts
from mythos.features import build_feature_frame
from mythos.promotion import evaluate_promotion
from mythos.risk import RiskConstitution
from mythos.router import MetaRouter
from mythos.world_model import WorldModel
from mythos.walkforward import (
    NeuralMetaLearner,
    V3ExecutionGovernor,
    _adaptive_counterfactual_pass,
    _adaptive_nonconformity_gate,
    _adaptive_rebalance_adjustment,
    _apply_intelligence_adjustment,
    _bayes_quality_gate,
    _conviction_score,
    _counterfactual_pass,
    _allow_conviction_leverage,
    _estimate_execution_cost_r,
    _intelligence_bucket,
    _nonconformity_gate,
    _nonconformity_score,
    _side_policy_ok,
    _update_bayes_quality_state,
    _update_intelligence_state,
)


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
    xcols = [
        "ret_1",
        "ret_4",
        "ret_16",
        "ret_64",
        "vol_16",
        "vol_64",
        "vol_256",
        "zscore_64",
        "trend_ema",
        "trend_slope_8",
        "range_break_48",
        "atr_pct",
        "rsi_14",
        "adx_14",
        "vol_z_128",
    ]
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


def test_execution_governor_adaptive_side_penalty_nudges_dominant_side():
    cfg = MythosConfig(
        side_balance_window=32,
        side_imbalance_soft_cap=0.65,
        side_imbalance_edge_penalty=0.015,
        adaptive_side_target_strength=0.30,
        adaptive_side_target_min=0.40,
        adaptive_side_target_max=0.60,
        side_health_penalty=0.03,
        side_health_boost=0.01,
    )
    gov = V3ExecutionGovernor(cfg)
    gov.side_hist = [1] * 30 + [-1] * 2
    gov.side_health[1] = -0.8
    gov.side_health[-1] = 0.7

    long_pen = gov.side_penalty(1)
    short_pen = gov.side_penalty(-1)
    assert long_pen > 0.0
    assert short_pen <= 0.0


def test_execution_governor_side_fail_is_soft_by_default():
    cfg = MythosConfig(side_fail_hard_pause=False, side_fail_window=8, side_fail_min_trades=3)
    gov = V3ExecutionGovernor(cfg)
    gov.side_recent_rr[1] = [-0.5, -0.4, -0.35, -0.3]
    gov._update_side_health(side=1, bar_idx=50)
    assert gov.side_pause_until[1] == -1
    assert gov.side_health[1] < 0.0


def test_neural_meta_learner_fallback_trains_without_torch():
    cfg = MythosConfig(
        use_meta_learner=True,
        meta_learner_fallback=True,
        meta_learner_min_train_samples=16,
        side_balance_window=64,
    )
    meta = NeuralMetaLearner(cfg=cfg, n_features=5)
    assert meta.is_active() is True
    x = np.array([0.2, -0.1, 0.05, 0.4, -0.2], dtype=np.float64)
    for i in range(80):
        rr = 0.7 if (i % 3) else -0.6
        meta.update(
            x=x,
            edge=0.02,
            confidence=0.56,
            uncertainty=0.2,
            regime=1,
            side=1,
            realized_r=rr,
        )
    p = meta.score(x=x, edge=0.02, confidence=0.56, uncertainty=0.2, regime=1, side=1)
    assert 0.0 <= p <= 1.0
    assert meta.is_ready() is True


def test_counterfactual_pass_keeps_strong_live_edge():
    class _StubAnalog:
        def __init__(self, choose, alt):
            self.choose = choose
            self.alt = alt

        def query(self, x, side):
            return self.choose if side == 1 else self.alt

    cfg = MythosConfig(
        counterfactual_min_advantage_r=0.006,
        counterfactual_risk_penalty=0.6,
        counterfactual_margin=0.006,
        counterfactual_uncertainty_weight=0.5,
        counterfactual_min_alt_hits=8,
    )
    analog = _StubAnalog(
        choose={"analog_edge": 0.002, "analog_conf": 0.52, "analog_hits": 20.0},
        alt={"analog_edge": 0.003, "analog_conf": 0.51, "analog_hits": 20.0},
    )
    ok = _counterfactual_pass(
        analog_mem=analog,
        x=np.array([0.0], dtype=np.float64),
        side=1,
        edge=0.025,
        uncertainty=0.18,
        cfg=cfg,
    )
    assert ok is True


def test_conviction_score_increases_with_better_signal_quality():
    cfg = MythosConfig(
        min_expected_r=0.005,
        conviction_weight_edge=0.30,
        conviction_weight_confidence=0.30,
        conviction_weight_uncertainty=0.25,
        conviction_weight_meta=0.15,
    )
    low = _conviction_score(
        edge=0.004,
        confidence=0.52,
        uncertainty=0.60,
        meta_p=0.51,
        cfg=cfg,
    )
    high = _conviction_score(
        edge=0.03,
        confidence=0.78,
        uncertainty=0.12,
        meta_p=0.69,
        cfg=cfg,
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_meta_bootstrap_activates_learner_readiness():
    cfg = MythosConfig(
        use_meta_learner=True,
        meta_learner_fallback=True,
        meta_bootstrap_samples=128,
        meta_bootstrap_epochs=2,
        meta_learner_min_train_samples=256,
    )
    meta = NeuralMetaLearner(cfg=cfg, n_features=5)
    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 1.0, size=(200, 5)).astype(np.float64)
    rr = rng.normal(0.05, 0.6, size=200).astype(np.float64)
    sides = np.where(rng.random(200) > 0.5, 1, -1).astype(np.int64)
    applied = meta.bootstrap_from_memory(X=X, realized_r=rr, sides=sides)
    assert applied is True
    assert meta.is_ready() is True


def test_conviction_boost_requires_positive_recent_quality():
    cfg = MythosConfig(
        conviction_score_threshold=0.6,
        conviction_boost=0.6,
        conviction_max_size_mult=2.3,
        conviction_recent_window=16,
        conviction_recent_min_trades=6,
        conviction_recent_min_expectancy=0.05,
    )
    risk = RiskConstitution(cfg)
    base = risk.position_size_multiplier(edge=0.03, uncertainty=0.15, conviction=0.9)
    neg_hist = [-0.3, -0.2, -0.4, -0.1, -0.2, -0.25]
    with_neg = risk.position_size_multiplier(
        edge=0.03,
        uncertainty=0.15,
        conviction=0.9,
        high_conviction_recent=neg_hist,
    )
    pos_hist = [0.4, 0.2, 0.3, 0.1, 0.5, 0.2]
    with_pos = risk.position_size_multiplier(
        edge=0.03,
        uncertainty=0.15,
        conviction=0.9,
        high_conviction_recent=pos_hist,
    )
    assert with_neg <= base
    assert with_pos > base


def test_conviction_boost_disabled_without_recent_quality():
    cfg = MythosConfig(
        conviction_score_threshold=0.6,
        conviction_boost=0.6,
        conviction_max_size_mult=2.3,
        conviction_guard_window=8,
        conviction_guard_min_trades=4,
        conviction_guard_min_expectancy_r=0.05,
    )
    risk = RiskConstitution(cfg)
    no_quality = risk.position_size_multiplier(
        edge=0.03,
        uncertainty=0.15,
        conviction=0.9,
        high_conviction_recent=[0.01, -0.02, 0.02, 0.01],
        allow_conviction_boost=True,
    )
    blocked = risk.position_size_multiplier(
        edge=0.03,
        uncertainty=0.15,
        conviction=0.9,
        high_conviction_recent=[0.01, -0.02, 0.02, 0.01],
        allow_conviction_boost=False,
    )
    assert abs(no_quality - blocked) < 1e-9


def test_sure_and_leverage_gate_thresholds_are_normalized():
    cfg = MythosConfig(
        sure_min_analog_hits=-5,
        sure_min_analog_ratio=1.5,
        sure_meta_strength_min=-0.2,
        sure_recent_window=0,
        sure_recent_min_trades=0,
        sure_recent_min_hit_rate=1.5,
        sure_cold_start_conviction_extra=0.8,
        leverage_recent_window=0,
        leverage_recent_min_trades=0,
        leverage_recent_min_hit_rate=-0.5,
    )
    assert cfg.sure_min_analog_hits == 0
    assert cfg.sure_min_analog_ratio == 1.0
    assert cfg.sure_meta_strength_min == 0.0
    assert cfg.sure_recent_window >= 8
    assert cfg.sure_recent_min_trades >= 1
    assert cfg.sure_recent_min_hit_rate == 1.0
    assert cfg.sure_cold_start_conviction_extra == 0.5
    assert cfg.leverage_recent_window >= 8
    assert cfg.leverage_recent_min_trades >= 1
    assert cfg.leverage_recent_min_hit_rate == 0.0


def test_execution_cost_and_leverage_policy_knobs_are_normalized():
    cfg = MythosConfig(
        leverage_policy_window=0,
        leverage_policy_min_trades=0,
        leverage_policy_min_hit_rate=2.0,
        leverage_policy_context_weight=1.5,
        leverage_policy_cold_start_conviction_extra=0.9,
        execution_fee_bps=-5.0,
        execution_slippage_bps=-2.0,
        execution_cost_cap_r=9.0,
    )
    assert cfg.leverage_policy_window >= 8
    assert cfg.leverage_policy_min_trades >= 1
    assert cfg.leverage_policy_min_hit_rate == 1.0
    assert cfg.leverage_policy_context_weight == 1.0
    assert cfg.leverage_policy_cold_start_conviction_extra == 0.5
    assert cfg.execution_fee_bps == 0.0
    assert cfg.execution_slippage_bps == 0.0
    assert cfg.execution_cost_cap_r == 5.0


def test_execution_cost_r_increases_with_fee_and_uncertainty():
    low_cost_cfg = MythosConfig(
        min_expected_r=0.01,
        execution_fee_bps=1.0,
        execution_slippage_bps=1.0,
        execution_cost_cap_r=0.5,
    )
    high_cost_cfg = MythosConfig(
        min_expected_r=0.01,
        execution_fee_bps=10.0,
        execution_slippage_bps=8.0,
        execution_cost_cap_r=0.5,
    )
    low = _estimate_execution_cost_r(edge=0.015, uncertainty=0.2, cfg=low_cost_cfg)
    high = _estimate_execution_cost_r(edge=0.015, uncertainty=1.2, cfg=high_cost_cfg)
    assert high > low
    assert 0.0 <= low <= low_cost_cfg.execution_cost_cap_r
    assert 0.0 <= high <= high_cost_cfg.execution_cost_cap_r


def test_leverage_policy_gate_blocks_weak_context_when_history_ready():
    cfg = MythosConfig(
        leverage_policy_window=32,
        leverage_policy_min_trades=8,
        leverage_policy_min_hit_rate=0.55,
        leverage_policy_min_expectancy=0.02,
        leverage_policy_context_weight=0.7,
        leverage_recent_window=32,
        leverage_recent_min_trades=8,
        leverage_recent_min_hit_rate=0.5,
        leverage_recent_min_expectancy=0.0,
    )
    weak_history = [-0.2, -0.1, -0.15, -0.05, -0.08, -0.04, -0.03, -0.06]
    weak_ctx = [0.35] * len(weak_history)
    strong_history = [0.3, 0.2, 0.15, 0.1, 0.2, 0.25, 0.05, 0.18]
    strong_ctx = [0.8] * len(strong_history)
    blocked = _allow_conviction_leverage(
        is_sure_signal=True,
        edge=0.03,
        confidence=0.68,
        conviction=0.9,
        context_score=0.4,
        leveraged_recent_rr=weak_history,
        leveraged_recent_ctx=weak_ctx,
        cfg=cfg,
    )
    allowed = _allow_conviction_leverage(
        is_sure_signal=True,
        edge=0.03,
        confidence=0.68,
        conviction=0.9,
        context_score=0.85,
        leveraged_recent_rr=strong_history,
        leveraged_recent_ctx=strong_ctx,
        cfg=cfg,
    )
    assert blocked is False
    assert allowed is True


def test_side_policy_blocks_leverage_on_weak_side_history():
    cfg = MythosConfig(
        leverage_side_policy_enable=True,
        leverage_side_min_trades=6,
        leverage_side_min_hit_rate=0.55,
        leverage_side_min_expectancy=0.02,
    )
    weak_side = {1: [-0.2, -0.1, -0.15, -0.05, -0.04, -0.08]}
    strong_side = {-1: [0.2, 0.1, 0.15, 0.05, 0.18, 0.07]}
    assert _side_policy_ok(side=1, side_rr_hist=weak_side, cfg=cfg) is False
    assert _side_policy_ok(side=-1, side_rr_hist=strong_side, cfg=cfg) is True


def test_net_edge_floor_blocks_leverage_when_cost_dominates():
    cfg = MythosConfig(
        min_expected_r=0.01,
        execution_fee_bps=10.0,
        execution_slippage_bps=10.0,
        execution_cost_cap_r=0.5,
        leverage_net_edge_floor=0.02,
    )
    edge = 0.015
    uncertainty = 1.0
    net_edge = edge - _estimate_execution_cost_r(edge=edge, uncertainty=uncertainty, cfg=cfg)
    assert net_edge < cfg.leverage_net_edge_floor


def test_bayes_quality_gate_rejects_persistently_weak_side_regime():
    cfg = MythosConfig(
        bayes_quality_enable=True,
        bayes_quality_warmup_trades=8,
        bayes_quality_prior_alpha=2.0,
        bayes_quality_prior_beta=2.0,
        bayes_quality_min_win_prob=0.52,
        bayes_quality_min_expectancy=0.0,
        bayes_quality_reject_margin=0.02,
    )
    side_stats = {1: {"alpha": 2.0, "beta": 2.0, "n": 0.0, "sum_r": 0.0}}
    regime_stats = {}
    for _ in range(18):
        _update_bayes_quality_state(
            side=1,
            regime=2,
            realized_r=-0.35,
            side_stats=side_stats,
            regime_side_stats=regime_stats,
            cfg=cfg,
        )
    gate = _bayes_quality_gate(
        side=1,
        regime=2,
        edge=0.002,
        confidence=0.51,
        uncertainty=1.2,
        total_trades=24,
        side_stats=side_stats,
        regime_side_stats=regime_stats,
        cfg=cfg,
    )
    assert gate["ready"] == 1.0
    assert gate["pass"] == 0.0


def test_nonconformity_gate_blocks_outlier_but_allows_override_for_extreme_signal():
    cfg = MythosConfig(
        nonconformity_enable=True,
        nonconformity_warmup_trades=12,
        nonconformity_window=64,
        nonconformity_quantile=0.80,
        nonconformity_margin=0.01,
        nonconformity_min_winners=10,
        nonconformity_override_conviction=0.9,
        nonconformity_override_edge_buffer=0.004,
        nonconformity_override_confidence_buffer=0.05,
        min_edge_threshold=0.01,
        min_confidence=0.55,
    )
    winner_scores = [0.12, 0.16, 0.18, 0.20, 0.22, 0.24, 0.19, 0.21, 0.23, 0.17, 0.15, 0.18]
    blocked = _nonconformity_gate(
        score=0.40,
        conviction=0.82,
        edge=0.018,
        confidence=0.61,
        total_trades=20,
        winner_scores=winner_scores,
        cfg=cfg,
    )
    assert blocked["ready"] == 1.0
    assert blocked["pass"] == 0.0
    assert blocked["override"] == 0.0

    overridden = _nonconformity_gate(
        score=0.40,
        conviction=0.95,
        edge=0.016,
        confidence=0.62,
        total_trades=20,
        winner_scores=winner_scores,
        cfg=cfg,
    )
    assert overridden["ready"] == 1.0
    assert overridden["pass"] == 1.0
    assert overridden["override"] == 1.0


def test_nonconformity_score_behaves_monotonically_with_quality():
    cfg = MythosConfig()
    bad = _nonconformity_score(
        edge=0.002,
        confidence=0.52,
        uncertainty=1.4,
        meta_p=0.51,
        analog_hits=4.0,
        cfg=cfg,
    )
    good = _nonconformity_score(
        edge=0.03,
        confidence=0.76,
        uncertainty=0.10,
        meta_p=0.70,
        analog_hits=36.0,
        cfg=cfg,
    )
    assert 0.0 <= bad <= 1.0
    assert 0.0 <= good <= 1.0
    assert good < bad


def test_adaptive_counterfactual_relaxes_when_reject_rate_overshoots():
    class _StubAnalog:
        def query(self, x, side):
            return {"analog_edge": 0.0015 if side == 1 else 0.0012, "analog_conf": 0.52, "analog_hits": 20.0}

    cfg = MythosConfig(
        counterfactual_min_advantage_r=0.01,
        counterfactual_margin=0.01,
        counterfactual_target_reject_rate=0.55,
        counterfactual_reject_tolerance=0.05,
        counterfactual_adaptive_relax=0.8,
        counterfactual_adaptive_min_adv_floor=0.2,
    )
    strict = _counterfactual_pass(
        analog_mem=_StubAnalog(),
        x=np.array([0.0], dtype=np.float64),
        side=1,
        edge=0.006,
        uncertainty=0.3,
        cfg=cfg,
    )
    relaxed = _adaptive_counterfactual_pass(
        analog_mem=_StubAnalog(),
        x=np.array([0.0], dtype=np.float64),
        side=1,
        edge=0.006,
        uncertainty=0.3,
        cfg=cfg,
        accepted_trades=20,
        cf_rejects=90,
    )
    assert strict is False
    assert relaxed is True


def test_adaptive_nonconformity_soft_override_activates():
    cfg = MythosConfig(
        nonconformity_enable=True,
        nonconformity_warmup_trades=12,
        nonconformity_window=64,
        nonconformity_quantile=0.80,
        nonconformity_margin=0.01,
        nonconformity_min_winners=10,
        nonconformity_target_reject_rate=0.45,
        nonconformity_reject_tolerance=0.05,
        nonconformity_adaptive_relax=0.5,
        nonconformity_adaptive_max_relax=0.2,
        nonconformity_soft_override_margin=0.06,
    )
    winner_scores = [0.10, 0.13, 0.15, 0.17, 0.18, 0.20, 0.14, 0.16, 0.19, 0.21, 0.12, 0.13]
    gate = _adaptive_nonconformity_gate(
        score=0.28,
        conviction=0.86,
        edge=0.018,
        confidence=0.62,
        total_trades=24,
        winner_scores=winner_scores,
        cfg=cfg,
        accepted_trades=20,
        nonconformity_rejects=80,
    )
    assert gate["ready"] == 1.0
    assert gate["pass"] == 1.0
    assert gate["override"] == 1.0
    assert gate["adaptive_soft_override"] == 1.0


def test_side_rebalance_boosts_shorts_when_underrepresented():
    cfg = MythosConfig(
        side_rebalance_enable=True,
        side_rebalance_warmup_trades=12,
        side_rebalance_window=20,
        side_rebalance_short_target=0.35,
        side_rebalance_short_boost=0.006,
        side_rebalance_long_penalty=0.004,
        side_rebalance_quality_guard=0.08,
    )
    long_hist = [0.01] * 14
    short_hist = [0.03] * 2
    short_adj = _adaptive_rebalance_adjustment(side=-1, long_trades=long_hist, short_trades=short_hist, cfg=cfg)
    long_adj = _adaptive_rebalance_adjustment(side=1, long_trades=long_hist, short_trades=short_hist, cfg=cfg)
    assert short_adj["edge_adjust"] > 0.0
    assert short_adj["conf_adjust"] >= 0.0
    assert long_adj["edge_adjust"] < 0.0


def test_risk_emergency_stop_and_size_throttle_behave_safely():
    cfg = MythosConfig(
        emergency_stop_enable=True,
        emergency_max_drawdown_r=2.0,
        emergency_equity_floor_r=-3.0,
        drawdown_size_start_r=1.0,
        drawdown_size_full_r=3.0,
        drawdown_size_min_scale=0.4,
        disable_leverage_drawdown_r=1.0,
    )
    risk = RiskConstitution(cfg)
    risk.record_trade(-1.5, bar_index=1, conviction=0.8)
    risk.record_trade(-1.0, bar_index=2, conviction=0.8)
    assert risk.should_disable_leverage() is True
    assert risk.should_stop_trading() is True
    base = 1.0 / (1.0 + 0.2) * (1.0 + 0.03)
    sized = risk.position_size_multiplier(edge=0.03, uncertainty=0.2, conviction=0.9)
    assert sized <= base


def test_intelligence_side_switch_is_capped_and_not_forced():
    cfg = MythosConfig(
        intelligence_enable=True,
        intelligence_min_samples=4,
        intelligence_side_switch_enable=True,
        intelligence_side_switch_min_gap=0.2,
        intelligence_side_switch_min_analog_adv=0.01,
        intelligence_side_switch_conviction_guard=0.8,
        intelligence_switch_rate_cap=0.2,
        intelligence_switch_cooldown_bars=5,
    )
    side_stats = {
        1: {"n": 20.0, "hit_ema": 0.35, "exp_ema": -0.06, "var_ema": 0.02},
        -1: {"n": 20.0, "hit_ema": 0.66, "exp_ema": 0.09, "var_ema": 0.02},
    }
    reg_stats = {
        (2, 1): {"n": 20.0, "hit_ema": 0.34, "exp_ema": -0.06, "var_ema": 0.02},
        (2, -1): {"n": 20.0, "hit_ema": 0.70, "exp_ema": 0.10, "var_ema": 0.02},
    }
    exp_stats = {"trend_long": {"n": 20.0, "hit_ema": 0.40, "exp_ema": -0.02, "var_ema": 0.02}}
    # Force clamp by setting already-high switch rate and cooldown active.
    adj = _apply_intelligence_adjustment(
        side=1,
        regime=2,
        expert_name="trend_long",
        edge=0.012,
        confidence=0.58,
        uncertainty=0.35,
        conviction=0.4,
        analog_edge=0.0,
        side_stats=side_stats,
        regime_side_stats=reg_stats,
        expert_stats=exp_stats,
        accepted_trades=50,
        side_switches=20,
        bar_idx=10,
        last_switch_bar=8,
        cfg=cfg,
    )
    assert int(round(float(adj["side"]))) == 1
    assert adj["switched"] == 0.0


def test_intelligence_adjustment_boosts_quality_after_positive_history():
    cfg = MythosConfig(
        intelligence_enable=True,
        intelligence_min_samples=6,
        intelligence_ema_alpha=0.2,
        intelligence_edge_scale=0.02,
        intelligence_conf_scale=0.08,
        intelligence_uncertainty_scale=0.25,
    )
    side_stats = {1: _intelligence_bucket(), -1: _intelligence_bucket()}
    regime_stats = {}
    expert_stats = {}
    for _ in range(20):
        _update_intelligence_state(
            side=1,
            regime=2,
            expert_name="trend_long",
            realized_r=0.9,
            side_stats=side_stats,
            regime_side_stats=regime_stats,
            expert_stats=expert_stats,
            cfg=cfg,
        )
    out = _apply_intelligence_adjustment(
        side=1,
        regime=2,
        expert_name="trend_long",
        edge=0.01,
        confidence=0.55,
        uncertainty=0.40,
        conviction=0.70,
        analog_advantage=0.01,
        side_stats=side_stats,
        regime_side_stats=regime_stats,
        expert_stats=expert_stats,
        cfg=cfg,
    )
    assert out["side"] == 1.0
    assert out["edge"] > 0.01
    assert out["confidence"] >= 0.55
    assert out["uncertainty"] <= 0.40
    assert out["score"] > 0.0


def test_intelligence_adjustment_can_switch_side_when_opposite_is_clearly_better():
    cfg = MythosConfig(
        intelligence_enable=True,
        intelligence_min_samples=6,
        intelligence_side_switch_enable=True,
        intelligence_side_switch_min_gap=0.20,
        intelligence_side_switch_min_analog_adv=0.001,
        intelligence_side_switch_conviction_guard=0.75,
    )
    side_stats = {1: _intelligence_bucket(), -1: _intelligence_bucket()}
    regime_stats = {}
    expert_stats = {}
    for _ in range(24):
        _update_intelligence_state(
            side=1,
            regime=0,
            expert_name="router_long",
            realized_r=-0.8,
            side_stats=side_stats,
            regime_side_stats=regime_stats,
            expert_stats=expert_stats,
            cfg=cfg,
        )
        _update_intelligence_state(
            side=-1,
            regime=0,
            expert_name="router_short",
            realized_r=0.9,
            side_stats=side_stats,
            regime_side_stats=regime_stats,
            expert_stats=expert_stats,
            cfg=cfg,
        )
    out = _apply_intelligence_adjustment(
        side=1,
        regime=0,
        expert_name="router_long",
        edge=0.012,
        confidence=0.56,
        uncertainty=0.35,
        conviction=0.40,
        analog_advantage=-0.01,
        side_stats=side_stats,
        regime_side_stats=regime_stats,
        expert_stats=expert_stats,
        cfg=cfg,
    )
    assert out["side"] == -1.0
    assert out["switched"] == 1.0
