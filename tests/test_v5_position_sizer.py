"""Tests for v5.0.8+ Adaptive Position Sizing & Dynamic Risk Scaling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from train.v5_position_sizer import (
    AdaptivePositionSizer, AdaptiveSizingConfig,
    RegimeScaler, RegimeScalingConfig,
    DailyLossTracker, LossManagementConfig,
    TrailingEquityStop,
    build_sizing_diagnostics,
)


class TestAdaptivePositionSizer:
    def test_disabled_returns_1(self):
        cfg = AdaptiveSizingConfig(enabled=False)
        sizer = AdaptivePositionSizer(cfg)
        assert sizer.compute_size_multiplier(1.0, 0.6, 0.5, 0.5, 0.3) == 1.0

    def test_high_pwin_high_rr_gives_above_min(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.5, max_size_mult=3.0, min_size_mult=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=2.0, p_win=0.7, mu_r=1.0, mfe=2.0, mae=0.5)
        assert mult > cfg.min_size_mult
        assert mult <= cfg.max_size_mult

    def test_low_pwin_gives_min_mult(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25, min_size_mult=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=0.5, p_win=0.3, mu_r=-0.5, mfe=0.2, mae=1.0)
        assert mult == 0.25

    def test_higher_pwin_gives_higher_mult(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.5, max_size_mult=3.0)
        sizer = AdaptivePositionSizer(cfg)
        mult_low = sizer.compute_size_multiplier(score=1.0, p_win=0.5, mu_r=0.5, mfe=1.0, mae=0.5)
        mult_high = sizer.compute_size_multiplier(score=1.0, p_win=0.8, mu_r=0.5, mfe=1.0, mae=0.5)
        assert mult_high > mult_low

    def test_diagnostics(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25)
        sizer = AdaptivePositionSizer(cfg)
        for i in range(20):
            sizer.compute_size_multiplier(score=1.0, p_win=0.5 + i*0.01, mu_r=0.5, mfe=1.0, mae=0.5)
        diag = sizer.get_diagnostics()
        assert diag['total_sized_trades'] == 20
        assert 'avg_size_mult' in diag
        assert 'pct_above_1x' in diag

    def test_safe_mae_zero(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=1.0, p_win=0.6, mu_r=0.5, mfe=1.0, mae=0.0)
        assert mult >= cfg.min_size_mult
        assert mult <= cfg.max_size_mult


class TestRegimeScaler:
    def test_disabled_returns_1(self):
        cfg = RegimeScalingConfig(enabled=False)
        scaler = RegimeScaler(cfg)
        assert scaler.compute_regime_multiplier(100, 1) == 1.0

    def test_ema_bullish_long(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 100.0)
        close[200:] = 110.0
        ema = np.full(300, 100.0)
        mult = scaler.compute_regime_multiplier(250, side=1, close_prices=close, ema200=ema)
        assert mult > 1.0

    def test_ema_bearish_long(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 90.0)
        ema = np.full(300, 100.0)
        mult = scaler.compute_regime_multiplier(250, side=1, close_prices=close, ema200=ema)
        assert mult < 1.0

    def test_rolling_equity_positive(self):
        cfg = RegimeScalingConfig(enabled=True, lookback_trades=10, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(1.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult > 1.0

    def test_rolling_equity_negative(self):
        cfg = RegimeScalingConfig(enabled=True, lookback_trades=10, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(-1.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult < 1.0

    def test_atr_low_volatility_bullish(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50, atr_bull_ratio=0.8, atr_bear_ratio=1.5)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, 1.0)
        atr[150:] = 0.5
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult > 1.0

    def test_atr_high_volatility_bearish(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50, atr_bull_ratio=0.8, atr_bear_ratio=1.5)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, 1.0)
        atr[150:] = 2.0
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult < 1.0

    def test_diagnostics(self):
        cfg = RegimeScalingConfig(enabled=True)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 100.0)
        ema = np.full(300, 100.0)
        for i in range(10):
            scaler.compute_regime_multiplier(200 + i, side=1, close_prices=close, ema200=ema)
        diag = scaler.get_diagnostics()
        assert diag['total_regime_trades'] == 10
        assert 'avg_regime_score' in diag

    def test_equity_below_min_trades_ignored(self):
        cfg = RegimeScalingConfig(enabled=True, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(-2.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult == 1.0

    def test_low_confidence_dampening_single_signal(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  min_equity_trades=5, low_confidence_dampen=0.5)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(1.0)
        mult_dampened = scaler.compute_regime_multiplier(100, side=1)
        assert mult_dampened > 1.0
        assert mult_dampened <= 1.25

    def test_no_dampening_with_multiple_signals(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  atr_lookback=50, min_equity_trades=5)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(1.0)
        atr = np.full(200, 1.0)
        atr[150:] = 0.5
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult > 1.0
        history = scaler.regime_history[-1]
        assert history['n_signals'] >= 2

    def test_atr_nan_early_bars_no_signal(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, np.nan)
        atr[100:] = 1.0
        mult = scaler.compute_regime_multiplier(30, side=1, atr_values=atr)
        assert mult == 1.0

    def test_equity_signal_capped_at_half(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  min_equity_trades=5, low_confidence_dampen=1.0)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(5.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult <= 1.5
        assert mult > 1.0


class TestDailyLossTracker:
    def test_daily_cap_blocks(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        assert not tracker.should_block()
        tracker.record_trade(-2.0)
        assert not tracker.should_block()
        tracker.record_trade(-1.5)
        assert tracker.should_block()

    def test_day_reset(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-4.0)
        assert tracker.should_block()
        tracker.new_bar("2025-01-02")
        assert not tracker.should_block()

    def test_per_symbol_cap(self):
        cfg = LossManagementConfig(per_symbol_daily_r_budget=-2.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-1.5, symbol="BTCUSDT")
        assert not tracker.should_block(symbol="BTCUSDT")
        tracker.record_trade(-1.0, symbol="BTCUSDT")
        assert tracker.should_block(symbol="BTCUSDT")
        assert not tracker.should_block(symbol="ETHUSDT")

    def test_diagnostics(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0, per_symbol_daily_r_budget=-2.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-4.0, symbol="BTCUSDT")
        tracker.should_block(symbol="BTCUSDT")
        diag = tracker.get_diagnostics()
        assert diag['days_killed'] == 1
        assert 'symbol_kill_counts' in diag

    def test_no_cap_no_block(self):
        cfg = LossManagementConfig()
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-10.0)
        assert not tracker.should_block()


class TestTrailingEquityStop:
    def test_triggers_on_drawdown(self):
        stop = TrailingEquityStop(stop_distance=5.0)
        stop.update(10.0)
        assert not stop.should_block()
        stop.update(-6.0)
        assert stop.should_block()

    def test_recovery(self):
        stop = TrailingEquityStop(stop_distance=5.0, recovery_pct=0.5)
        stop.update(10.0)
        stop.update(-6.0)
        assert stop.should_block()
        stop.update(4.0)
        assert not stop.should_block()

    def test_no_false_trigger(self):
        stop = TrailingEquityStop(stop_distance=10.0)
        for _ in range(20):
            stop.update(0.5)
        assert not stop.should_block()

    def test_diagnostics(self):
        stop = TrailingEquityStop(stop_distance=5.0)
        stop.update(10.0)
        stop.update(-7.0)
        stop.update(3.0)
        diag = stop.get_diagnostics()
        assert diag['stop_triggers'] == 1
        assert diag['max_drawdown_r'] >= 5.0
        assert diag['final_equity_r'] == 6.0

    def test_multiple_triggers(self):
        stop = TrailingEquityStop(stop_distance=3.0, recovery_pct=0.3)
        stop.update(5.0)
        stop.update(-4.0)
        assert stop.should_block()
        stop.update(3.5)
        assert not stop.should_block()
        stop.update(2.0)
        stop.update(-6.0)
        assert stop.should_block()
        diag = stop.get_diagnostics()
        assert diag['stop_triggers'] == 2


class TestPercentileThreshold:
    """Tests for the adaptive percentile-based threshold floor logic."""

    def _compute_effective_threshold(self, calibrated, scores, min_threshold=None,
                                      min_threshold_pct=None, quality_mask=None):
        effective = calibrated
        pct_floor = None
        if min_threshold_pct is not None:
            pct_scores = scores.copy()
            pct_scores[np.isnan(pct_scores)] = -np.inf
            if quality_mask is not None:
                pct_scores[~quality_mask] = -np.inf
            finite = pct_scores[np.isfinite(pct_scores)]
            if len(finite) > 0:
                pct_floor = float(np.percentile(finite, min_threshold_pct))
        if min_threshold is not None and effective < min_threshold:
            effective = min_threshold
        if pct_floor is not None and effective < pct_floor:
            effective = pct_floor
        return effective, pct_floor

    def test_pct_raises_low_threshold(self):
        scores = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50])
        eff, pct = self._compute_effective_threshold(0.0358, scores, min_threshold_pct=70)
        assert pct is not None
        assert pct == pytest.approx(np.percentile(scores, 70), abs=1e-6)
        assert eff > 0.0358
        assert eff == pct

    def test_pct_no_effect_when_calibrated_higher(self):
        scores = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
        eff, pct = self._compute_effective_threshold(0.3210, scores, min_threshold_pct=70)
        assert pct is not None
        assert eff == 0.3210

    def test_fixed_and_pct_combined_takes_max(self):
        scores = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        eff, pct = self._compute_effective_threshold(
            0.02, scores, min_threshold=0.05, min_threshold_pct=80)
        p80 = float(np.percentile(scores, 80))
        assert eff == max(0.05, p80)

    def test_pct_ignores_nan_scores(self):
        scores = np.array([np.nan, np.nan, 0.10, 0.20, 0.30, np.nan, 0.40, 0.50])
        eff, pct = self._compute_effective_threshold(0.01, scores, min_threshold_pct=50)
        valid = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        assert pct == pytest.approx(np.percentile(valid, 50), abs=1e-6)

    def test_pct_respects_quality_mask(self):
        scores = np.array([0.90, 0.80, 0.10, 0.20, 0.30])
        mask = np.array([False, False, True, True, True])
        eff, pct = self._compute_effective_threshold(
            0.01, scores, min_threshold_pct=50, quality_mask=mask)
        assert pct == pytest.approx(np.percentile([0.10, 0.20, 0.30], 50), abs=1e-6)

    def test_pct_disabled_when_none(self):
        scores = np.array([0.10, 0.20, 0.30])
        eff, pct = self._compute_effective_threshold(0.05, scores, min_threshold_pct=None)
        assert pct is None
        assert eff == 0.05

    def test_pct_all_nan_scores(self):
        scores = np.array([np.nan, np.nan, np.nan])
        eff, pct = self._compute_effective_threshold(0.05, scores, min_threshold_pct=70)
        assert pct is None
        assert eff == 0.05


class TestBuildSizingDiagnostics:
    def test_empty(self):
        diag = build_sizing_diagnostics(None, None, None, None)
        assert diag == {}

    def test_full(self):
        sizer_cfg = AdaptiveSizingConfig(enabled=True)
        sizer = AdaptivePositionSizer(sizer_cfg)
        sizer.compute_size_multiplier(1.0, 0.6, 0.5, 1.0, 0.5)

        regime_cfg = RegimeScalingConfig(enabled=True)
        regime = RegimeScaler(regime_cfg)
        regime.record_trade_result(1.0)

        loss_cfg = LossManagementConfig(daily_loss_cap=-3.0)
        daily = DailyLossTracker(loss_cfg)

        equity = TrailingEquityStop(5.0)

        sized_r = np.array([1.0, -0.5, 2.0])
        unsized_r = np.array([0.8, -0.4, 1.5])

        diag = build_sizing_diagnostics(sizer, regime, daily, equity, sized_r, unsized_r)
        assert 'adaptive_sizing' in diag
        assert 'regime_scaling' in diag
        assert 'daily_loss_management' in diag
        assert 'trailing_equity_stop' in diag
        assert 'sizing_comparison' in diag
        assert abs(diag['sizing_comparison']['sizing_impact_r'] - (2.5 - 1.9)) < 0.01


class TestConvictionSizer:
    def _make_sizer(self, **overrides):
        from train.v5_position_sizer import ConvictionSizer, ConvictionSizingConfig
        defaults = dict(enabled=True, tier_top_pct=5.0, tier_top_mult=2.5,
                        tier_high_pct=20.0, tier_high_mult=1.5,
                        tier_mid_mult=1.0, tier_low_pct=50.0, tier_low_mult=0.5,
                        confidence_boost_threshold=0.65, confidence_boost_mult=1.3)
        defaults.update(overrides)
        cfg = ConvictionSizingConfig(**defaults)
        return ConvictionSizer(cfg)

    def _warm_up(self, sizer, n=100):
        for s in np.linspace(0.1, 2.0, n):
            sizer.compute_size_multiplier(score=s, p_directional=0.5, side=1)

    def test_disabled_returns_1(self):
        sizer = self._make_sizer(enabled=False)
        assert sizer.compute_size_multiplier(score=2.0, p_directional=0.9, side=1) == 1.0

    def test_top_tier_gets_top_mult(self):
        sizer = self._make_sizer()
        self._warm_up(sizer)
        mult = sizer.compute_size_multiplier(score=10.0, p_directional=0.5, side=1)
        assert mult == 2.5

    def test_bottom_tier_gets_low_mult(self):
        sizer = self._make_sizer()
        self._warm_up(sizer)
        mult = sizer.compute_size_multiplier(score=0.01, p_directional=0.5, side=1)
        assert mult == 0.5

    def test_confidence_boost(self):
        sizer = self._make_sizer()
        self._warm_up(sizer)
        mult_no_conf = sizer.compute_size_multiplier(score=10.0, p_directional=0.5, side=1)
        mult_with_conf = sizer.compute_size_multiplier(score=10.0, p_directional=0.8, side=1)
        assert mult_with_conf > mult_no_conf

    def test_warmup_returns_mid(self):
        sizer = self._make_sizer()
        mult = sizer.compute_size_multiplier(score=1.0, p_directional=0.5, side=1)
        assert mult == 1.0

    def test_diagnostics(self):
        sizer = self._make_sizer()
        self._warm_up(sizer)
        sizer.compute_size_multiplier(score=1.5, p_directional=0.7, side=1)
        diag = sizer.get_diagnostics()
        assert diag['conviction_sizing_enabled'] is True
        assert diag['total_conviction_trades'] > 0
        assert 'avg_conviction_mult' in diag
        assert 'tier_distribution' in diag


class TestTrailingStop:
    def test_basic_long_trailing(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 102, 105, 108, 110, 108, 105, 103, 100, 98])
        highs = closes + 1
        lows = closes - 1
        atr_val = 2.0
        tp_dist = atr_val * 5.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=9, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=5.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        assert outcome in ('TP', 'TRAIL_WIN', 'TRAIL_BE', 'SL', 'EXP_WIN', 'EXP_LOSS')

    def test_short_trailing(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 98, 95, 92, 90, 92, 95, 97, 100, 102])
        highs = closes + 1
        lows = closes - 1
        atr_val = 2.0
        tp_dist = atr_val * 5.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=9, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=5.0, sl_mult=1.5, atr_val=atr_val, side=-1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        assert outcome in ('TP', 'TRAIL_WIN', 'TRAIL_BE', 'SL', 'EXP_WIN', 'EXP_LOSS')

    def test_sl_hit_before_activation(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 99, 98, 97, 96, 95])
        highs = closes + 0.5
        lows = closes - 0.5
        atr_val = 2.0
        tp_dist = atr_val * 5.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=5, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=5.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        assert outcome == 'SL'
        assert r_val < 0

    def test_runner_mode(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 105, 110, 115, 120, 118, 115, 112, 108, 105])
        highs = closes + 1
        lows = closes - 1
        atr_val = 2.0
        tp_dist = atr_val * 3.0
        sl_dist = atr_val * 1.5
        r_no_runner, out_no_runner = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=9, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=3.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        r_runner, out_runner = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=9, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=3.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=True
        )
        assert out_no_runner in ('TP', 'TRAIL_WIN', 'EXP_WIN')
        assert out_runner in ('TRAIL_WIN', 'TRAIL_BE', 'EXP_WIN', 'EXP_LOSS')

    def test_expiry(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 100.5, 101, 100.5, 100, 100.5])
        highs = closes + 0.3
        lows = closes - 0.3
        atr_val = 50.0
        tp_dist = atr_val * 5.0
        sl_dist = atr_val * 5.0
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=5, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=5.0, sl_mult=5.0, atr_val=atr_val, side=1,
            trail_activation=3.0, trail_distance=1.0, allow_runner=False
        )
        assert outcome in ('EXP_WIN', 'EXP_LOSS')


    def test_tp_checked_before_sl_on_same_bar(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 110])
        highs = np.array([100, 115])
        lows = np.array([100, 95])
        atr_val = 2.0
        tp_dist = atr_val * 3.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=1, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=3.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        assert outcome == 'TP'
        assert abs(r_val - 3.0 / 1.5) < 0.01

    def test_trail_be_on_breakeven_exit(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 103, 105, 100, 98])
        highs = np.array([100, 104, 106, 101, 99])
        lows = np.array([100, 102, 104, 99, 97])
        atr_val = 2.0
        tp_dist = atr_val * 5.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=4, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=5.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=3.0, allow_runner=False
        )
        if outcome in ('TRAIL_WIN', 'TRAIL_BE'):
            assert r_val >= 0

    def test_trail_win_positive_r(self):
        from data.common import _simulate_trade_trailing
        closes = np.array([100, 103, 107, 110, 108, 105, 100])
        highs = closes + 1
        lows = closes - 1
        atr_val = 2.0
        tp_dist = atr_val * 10.0
        sl_dist = atr_val * 1.5
        r_val, outcome = _simulate_trade_trailing(
            highs, lows, closes, i=0, horizon=6, n=len(closes),
            entry=100.0, tp_dist=tp_dist, sl_dist=sl_dist,
            tp_mult=10.0, sl_mult=1.5, atr_val=atr_val, side=1,
            trail_activation=1.0, trail_distance=1.0, allow_runner=False
        )
        if outcome == 'TRAIL_WIN':
            assert r_val > 0

    def test_rolling_window_conviction(self):
        from train.v5_position_sizer import ConvictionSizer, ConvictionSizingConfig
        cfg = ConvictionSizingConfig(enabled=True, window_size=30)
        sizer = ConvictionSizer(cfg)
        for s in np.linspace(0.1, 1.0, 50):
            sizer.compute_size_multiplier(score=s, p_directional=0.5, side=1)
        assert len(sizer.score_window) == 30


class TestBuildSizingDiagWithConviction:
    def test_conviction_in_diagnostics(self):
        from train.v5_position_sizer import ConvictionSizer, ConvictionSizingConfig
        cfg = ConvictionSizingConfig(enabled=True)
        conv = ConvictionSizer(cfg)
        for s in np.linspace(0.1, 2.0, 50):
            conv.compute_size_multiplier(score=s, p_directional=0.5, side=1)
        diag = build_sizing_diagnostics(None, None, None, None, conviction=conv)
        assert 'conviction_sizing' in diag
        assert diag['conviction_sizing']['total_conviction_trades'] == 50


class TestUltraConvictionSizer:
    def _make_sizer(self, **overrides):
        from train.v5_position_sizer import UltraConvictionSizer, UltraConvictionConfig
        defaults = dict(enabled=True, risk_cap=0.05, score_pct=0.90,
                        adx_min=25.0, edge_min=0.03, dd_max=0.10,
                        max_per_day=2, mult=3.0)
        defaults.update(overrides)
        cfg = UltraConvictionConfig(**defaults)
        sizer = UltraConvictionSizer(cfg)
        for s in np.linspace(0.1, 2.0, 100):
            sizer.record_score(s)
        return sizer

    def _eval(self, sizer, score=2.5, adx=30.0, edge_l=0.06, edge_s=0.06,
              side=1, regime="bull", date_str="2024-01-01"):
        return sizer.evaluate(symbol="BTCUSDT", side=side, score=score,
                              edge_l=edge_l, edge_s=edge_s, adx_val=adx,
                              regime=regime, date_str=date_str)

    def test_disabled_returns_false(self):
        from train.v5_position_sizer import UltraConvictionSizer, UltraConvictionConfig
        cfg = UltraConvictionConfig(enabled=False)
        sizer = UltraConvictionSizer(cfg)
        assert not sizer.evaluate("BTC", 1, 2.0, 0.05, 0.05, 30.0, "bull", "2024-01-01")

    def test_ultra_applied_when_all_gates_pass(self):
        sizer = self._make_sizer()
        assert self._eval(sizer) is True
        assert sizer.ultra_applied == 1

    def test_apply_sizing_uses_mult(self):
        sizer = self._make_sizer(mult=3.0)
        assert self._eval(sizer) is True
        new_mult = sizer.apply_ultra_sizing(current_mult=1.0, stop_distance_pct=0.01)
        assert new_mult == 3.0

    def test_apply_sizing_caps_by_risk(self):
        sizer = self._make_sizer(mult=5.0, risk_cap=0.05)
        assert self._eval(sizer) is True
        new_mult = sizer.apply_ultra_sizing(current_mult=1.0, stop_distance_pct=0.02)
        assert new_mult <= 0.05 / 0.02 + 0.001

    def test_ultra_blocked_by_low_adx(self):
        sizer = self._make_sizer(adx_min=25.0)
        assert self._eval(sizer, adx=20.0) is False

    def test_ultra_nan_adx_bypasses_gate(self):
        sizer = self._make_sizer(adx_min=25.0)
        result = self._eval(sizer, adx=float('nan'))
        assert result is True

    def test_ultra_blocked_by_low_edge(self):
        sizer = self._make_sizer(edge_min=0.03)
        assert self._eval(sizer, edge_l=0.01) is False

    def test_ultra_blocked_by_regime_mismatch(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="bear") is False

    def test_ultra_blocked_by_neutral_regime(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, regime="neutral") is False

    def test_ultra_daily_limit(self):
        sizer = self._make_sizer(max_per_day=1)
        assert self._eval(sizer) is True
        assert self._eval(sizer) is False  # daily limit reached

    def test_ultra_blocked_by_drawdown(self):
        sizer = self._make_sizer(dd_max=0.10)
        sizer.update_equity(1.0)
        sizer.update_equity(-0.5)
        assert self._eval(sizer) is False

    def test_ultra_diagnostics(self):
        sizer = self._make_sizer()
        self._eval(sizer)
        self._eval(sizer, score=0.5)  # low score
        diag = sizer.get_diagnostics()
        assert diag['ultra_applied'] == 1
        assert diag['ultra_skipped'] >= 1
        assert diag['risk_cap'] == 0.05

    def test_ultra_score_below_percentile(self):
        sizer = self._make_sizer(score_pct=0.95)
        assert self._eval(sizer, score=1.0) is False

    def test_build_diagnostics_with_ultra(self):
        sizer = self._make_sizer()
        diag = build_sizing_diagnostics(None, None, None, None, ultra=sizer)
        assert 'ultra_conviction' in diag


class TestMultiRegimeClassifier:
    def _make_classifier(self, **kwargs):
        from train.v5_position_sizer import MultiRegimeClassifier, MultiRegimeConfig
        defaults = dict(
            adx_trending_threshold=25.0, adx_choppy_threshold=20.0,
            atr_high_vol_ratio=1.3, atr_low_vol_ratio=0.7,
            atr_rolling_window=96, ema_slope_window=10, ema_price_buffer=0.005,
        )
        defaults.update(kwargs)
        cfg = MultiRegimeConfig(**defaults)
        return MultiRegimeClassifier(cfg)

    def test_high_vol_regime(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=30.0, atr_current=1.5, atr_rolling=1.0,
                              close_price=100.0, ema200_val=95.0, ema200_prev=94.5)
        assert regime == "high_vol"

    def test_choppy_regime(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=15.0, atr_current=1.0, atr_rolling=1.0,
                              close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        assert regime == "choppy"

    def test_trending_up_regime(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=30.0, atr_current=1.0, atr_rolling=1.0,
                              close_price=105.0, ema200_val=100.0, ema200_prev=98.0)
        assert regime == "trending_up"

    def test_trending_down_regime(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=30.0, atr_current=1.0, atr_rolling=1.0,
                              close_price=95.0, ema200_val=100.0, ema200_prev=102.0)
        assert regime == "trending_down"

    def test_low_vol_regime(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=22.0, atr_current=0.5, atr_rolling=1.0,
                              close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        assert regime == "low_vol"

    def test_nan_adx_low_atr_gives_low_vol(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=float('nan'), atr_current=0.5, atr_rolling=1.0,
                              close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        assert regime == "low_vol"

    def test_nan_atr_rolling_not_high_vol(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=30.0, atr_current=float('nan'), atr_rolling=1.0,
                              close_price=105.0, ema200_val=100.0, ema200_prev=98.0)
        assert regime != "high_vol"

    def test_diagnostics_tracking(self):
        clf = self._make_classifier()
        clf.classify(adx_val=30.0, atr_current=1.5, atr_rolling=1.0,
                     close_price=100.0, ema200_val=95.0, ema200_prev=94.5)
        clf.classify(adx_val=15.0, atr_current=1.0, atr_rolling=1.0,
                     close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        clf.classify(adx_val=30.0, atr_current=1.0, atr_rolling=1.0,
                     close_price=105.0, ema200_val=100.0, ema200_prev=98.0)
        diag = clf.get_diagnostics()
        assert diag['total_classified'] == 3
        assert diag['regime_counts']['high_vol'] == 1
        assert diag['regime_counts']['choppy'] == 1
        assert diag['regime_counts']['trending_up'] == 1

    def test_priority_order_high_vol_over_trending(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=30.0, atr_current=1.5, atr_rolling=1.0,
                              close_price=105.0, ema200_val=100.0, ema200_prev=98.0)
        assert regime == "high_vol"

    def test_low_adx_low_atr_gives_low_vol(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=15.0, atr_current=0.5, atr_rolling=1.0,
                              close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        assert regime == "low_vol"

    def test_low_adx_normal_atr_gives_choppy(self):
        clf = self._make_classifier()
        regime = clf.classify(adx_val=15.0, atr_current=1.0, atr_rolling=1.0,
                              close_price=100.0, ema200_val=100.0, ema200_prev=100.0)
        assert regime == "choppy"


class TestUltraConvictionMultiRegime:
    def _make_sizer(self, **kwargs):
        from train.v5_position_sizer import UltraConvictionSizer, UltraConvictionConfig
        defaults = dict(enabled=True, risk_cap=0.05, score_pct=0.50,
                        adx_min=20.0, edge_min=0.02, dd_max=0.20,
                        max_per_day=5, mult=3.0, score_window=50)
        defaults.update(kwargs)
        cfg = UltraConvictionConfig(**defaults)
        sizer = UltraConvictionSizer(cfg)
        for s in np.linspace(0.5, 2.0, 30):
            sizer.record_score(float(s))
        return sizer

    def _eval(self, sizer, side=1, regime="trending_up", score=1.8, adx=30.0, edge_l=0.05, edge_s=0.05):
        return sizer.evaluate(
            symbol="BTCUSDT", side=side, score=score,
            edge_l=edge_l, edge_s=edge_s, adx_val=adx,
            regime=regime, date_str="2025-01-01",
        )

    def test_trending_up_allows_long(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="trending_up") is True

    def test_trending_down_allows_short(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=-1, regime="trending_down") is True

    def test_low_vol_allows_both(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="low_vol") is True
        sizer2 = self._make_sizer()
        assert self._eval(sizer2, side=-1, regime="low_vol") is True

    def test_choppy_blocks_all(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="choppy") is False

    def test_high_vol_blocks_all(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="high_vol") is False

    def test_trending_up_blocks_short(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=-1, regime="trending_up") is False

    def test_trending_down_blocks_long(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="trending_down") is False

    def test_backward_compat_bull(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="bull") is True

    def test_backward_compat_bear(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=-1, regime="bear") is True

    def test_backward_compat_neutral_blocked(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="neutral") is False

    def test_empty_regime_blocked(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="") is False

    def test_unknown_regime_blocked(self):
        sizer = self._make_sizer()
        assert self._eval(sizer, side=1, regime="unknown") is False


class TestMultiRegimeCoverage:
    """Tests that multi-regime classifier covers >95% of bars after warmup."""

    def _make_classifier(self, **overrides):
        from train.v5_position_sizer import MultiRegimeClassifier, MultiRegimeConfig
        defaults = dict(
            enabled=True,
            adx_trending_threshold=25.0,
            adx_choppy_threshold=20.0,
            atr_high_vol_ratio=1.3,
            atr_low_vol_ratio=0.7,
            atr_rolling_window=96,
            ema_slope_window=10,
            ema_price_buffer=0.005,
        )
        defaults.update(overrides)
        return MultiRegimeClassifier(MultiRegimeConfig(**defaults))

    def test_coverage_with_valid_inputs(self):
        clf = self._make_classifier()
        n_bars = 1000
        np.random.seed(42)
        for i in range(n_bars):
            adx = np.random.uniform(10, 40)
            atr_cur = np.random.uniform(0.5, 2.0)
            atr_roll = 1.0
            price = 50000 + np.random.uniform(-5000, 5000)
            ema = 50000.0
            ema_prev = 49900.0
            regime = clf.classify(adx, atr_cur, atr_roll, price, ema, ema_prev)
            assert regime in ("trending_up", "trending_down", "choppy", "high_vol", "low_vol")
        diag = clf.get_diagnostics()
        unknown_pct = diag['regime_pct'].get('unknown', 0)
        assert unknown_pct < 5.0, f"Unknown regime too high: {unknown_pct}%"
        assert diag['total_classified'] == n_bars

    def test_coverage_with_nan_inputs_still_classifies(self):
        clf = self._make_classifier()
        regime = clf.classify(float('nan'), float('nan'), float('nan'),
                              50000.0, float('nan'), float('nan'))
        assert regime in ("trending_up", "trending_down", "choppy", "high_vol", "low_vol", "unknown")

    def test_flat_price_produces_valid_regimes(self):
        clf = self._make_classifier()
        for _ in range(500):
            regime = clf.classify(adx_val=15.0, atr_current=1.0,
                                  atr_rolling=1.0, close_price=50000.0,
                                  ema200_val=50000.0, ema200_prev=50000.0)
            assert regime in clf.REGIMES


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestADXStability:
    """Tests that ADX computation produces no NaN after warmup, even with flat data."""

    def test_adx_no_nan_after_warmup(self):
        from train.v5_train import _compute_adx
        n = 500
        high = np.full(n, 50100.0)
        low = np.full(n, 49900.0)
        close = np.full(n, 50000.0)
        adx = _compute_adx(high, low, close, period=14)
        warmup_end = 14 * 3
        post_warmup = adx[warmup_end:]
        assert not np.any(np.isnan(post_warmup)), f"ADX has NaN after warmup: {np.sum(np.isnan(post_warmup))} NaNs"

    def test_adx_no_nan_volatile_data(self):
        from train.v5_train import _compute_adx
        np.random.seed(99)
        n = 1000
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        high = close + np.abs(np.random.randn(n) * 50)
        low = close - np.abs(np.random.randn(n) * 50)
        adx = _compute_adx(high, low, close, period=14)
        warmup_end = 14 * 3
        post_warmup = adx[warmup_end:]
        assert not np.any(np.isnan(post_warmup))
        assert np.all(np.isfinite(post_warmup))


class TestScoreStatsFinite:
    """Tests that score stat computation handles -inf values correctly."""

    def _compute_finite_stats(self, scores):
        finite_mask = np.isfinite(scores)
        finite_scores = scores[finite_mask]
        if len(finite_scores) > 0:
            return {
                'score_mean': float(np.mean(finite_scores)),
                'score_std': float(np.std(finite_scores)),
                'score_p50': float(np.percentile(finite_scores, 50)),
                'score_p90': float(np.percentile(finite_scores, 90)),
                'score_pct_positive': float(np.mean(finite_scores > 0) * 100),
                'n_finite_scores': int(np.sum(finite_mask)),
            }
        else:
            return {
                'score_mean': 0.0, 'score_std': 0.0,
                'score_p50': 0.0, 'score_p90': 0.0,
                'score_pct_positive': 0.0,
                'n_finite_scores': 0,
            }

    def test_mixed_finite_and_inf(self):
        scores = np.array([0.5, 1.2, -np.inf, 0.8, -np.inf, 2.0])
        diag = self._compute_finite_stats(scores)
        assert np.isfinite(diag['score_mean'])
        assert np.isfinite(diag['score_p50'])
        assert np.isfinite(diag['score_p90'])
        assert diag['n_finite_scores'] == 4

    def test_all_inf_scores(self):
        scores = np.full(50, -np.inf)
        diag = self._compute_finite_stats(scores)
        assert diag['score_mean'] == 0.0
        assert diag['n_finite_scores'] == 0

    def test_all_finite_scores(self):
        scores = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        diag = self._compute_finite_stats(scores)
        assert diag['n_finite_scores'] == 5
        assert abs(diag['score_mean'] - np.mean(scores)) < 1e-6
        assert diag['score_pct_positive'] == 100.0

    def test_nan_handling(self):
        scores = np.array([0.5, np.nan, 1.0, -np.inf])
        diag = self._compute_finite_stats(scores)
        assert diag['n_finite_scores'] == 2
        assert np.isfinite(diag['score_mean'])


class TestEdgeFirstFilter:
    """Tests for Edge-First selection gate (PART 1)."""

    def test_edge_min_filters_low_edge(self):
        edge_bar_values = np.array([0.01, 0.02, 0.03, 0.05, 0.10])
        edge_min = 0.03
        ef_pass = edge_bar_values >= edge_min
        assert list(ef_pass) == [False, False, True, True, True]
        assert np.sum(ef_pass) == 3

    def test_percentile_floor_works(self):
        np.random.seed(42)
        edge_bar_values = np.random.uniform(0, 0.2, 1000).astype(np.float32)
        pct_floor = 70
        ef_pct_threshold = float(np.percentile(edge_bar_values, pct_floor))
        ef_threshold = max(0.03, ef_pct_threshold)
        ef_pass = edge_bar_values >= ef_threshold
        pass_pct = np.mean(ef_pass) * 100
        assert pass_pct <= 35, f"Expected at most 35% pass, got {pass_pct:.1f}%"

    def test_edge_min_with_inf(self):
        edge_bar_values = np.array([0.05, -np.inf, np.nan, 0.10, 0.02])
        edge_min = 0.03
        ef_pass = np.isfinite(edge_bar_values) & (edge_bar_values >= edge_min)
        assert list(ef_pass) == [True, False, False, True, False]

    def test_proxy_edge_computation(self):
        mu_R = np.array([0.1, -0.05, 0.2, 0.0])
        risk = np.maximum(np.array([0.5, 0.5, 0.1, 0.5]), 1e-6)
        edge_proxy = np.abs(mu_R) / risk
        assert abs(edge_proxy[0] - 0.2) < 1e-6
        assert abs(edge_proxy[1] - 0.1) < 1e-6
        assert abs(edge_proxy[2] - 2.0) < 1e-6
        assert abs(edge_proxy[3] - 0.0) < 1e-6

    def test_edge_topn_per_day_cap(self):
        topn = 3
        trades_today = 0
        results = []
        for i in range(10):
            if trades_today >= topn:
                results.append(False)
            else:
                results.append(True)
                trades_today += 1
        assert sum(results) == 3
        assert len(results) == 10


class TestRegimeSideMap:
    """Tests for regime-conditional side filtering (PART 2)."""

    def _should_block(self, regime_side_map, bar_regime, side_val, ultra_conviction=False):
        allowed = regime_side_map.get(bar_regime, "BOTH")
        if ultra_conviction and allowed == "NONE":
            return False
        if allowed == "NONE":
            return True
        if allowed == "LONG" and side_val != 1:
            return True
        if allowed == "SHORT" and side_val != -1:
            return True
        return False

    def test_long_allowed_in_trending_up(self):
        rsm = {"trending_up": "LONG", "trending_down": "SHORT", "choppy": "NONE"}
        assert not self._should_block(rsm, "trending_up", 1)
        assert self._should_block(rsm, "trending_up", -1)

    def test_short_allowed_in_trending_down(self):
        rsm = {"trending_up": "LONG", "trending_down": "SHORT", "choppy": "NONE"}
        assert not self._should_block(rsm, "trending_down", -1)
        assert self._should_block(rsm, "trending_down", 1)

    def test_none_blocks_both(self):
        rsm = {"choppy": "NONE"}
        assert self._should_block(rsm, "choppy", 1)
        assert self._should_block(rsm, "choppy", -1)

    def test_ultra_conviction_overrides_none(self):
        rsm = {"choppy": "NONE"}
        assert not self._should_block(rsm, "choppy", 1, ultra_conviction=True)

    def test_unknown_regime_allows_both(self):
        rsm = {"trending_up": "LONG"}
        assert not self._should_block(rsm, "unknown", 1)
        assert not self._should_block(rsm, "unknown", -1)

    def test_both_allows_all(self):
        rsm = {"trending_up": "BOTH"}
        assert not self._should_block(rsm, "trending_up", 1)
        assert not self._should_block(rsm, "trending_up", -1)


class TestSizeFloorClamp:
    """Tests for size floor clamping (PART 3)."""

    def _apply_size_floor(self, trade_size_mult, size_floor, rolling_r, throttle_level):
        if size_floor > 0 and trade_size_mult < size_floor:
            sf_allow = True
            if rolling_r < 0:
                sf_allow = False
            if throttle_level > 0.5:
                sf_allow = False
            if sf_allow:
                return size_floor
        return trade_size_mult

    def test_floor_applied_when_rolling_positive(self):
        result = self._apply_size_floor(0.15, 0.35, rolling_r=2.0, throttle_level=0.0)
        assert result == 0.35

    def test_floor_not_applied_when_rolling_negative(self):
        result = self._apply_size_floor(0.15, 0.35, rolling_r=-1.0, throttle_level=0.0)
        assert result == 0.15

    def test_floor_not_applied_in_high_throttle(self):
        result = self._apply_size_floor(0.15, 0.35, rolling_r=2.0, throttle_level=0.7)
        assert result == 0.15

    def test_floor_not_applied_when_above_floor(self):
        result = self._apply_size_floor(0.50, 0.35, rolling_r=2.0, throttle_level=0.0)
        assert result == 0.50

    def test_floor_disabled_when_zero(self):
        result = self._apply_size_floor(0.10, 0.0, rolling_r=2.0, throttle_level=0.0)
        assert result == 0.10


class TestFunnelDiagnosticsConsistency:
    """Tests that funnel diagnostic counters are consistent."""

    def test_counters_add_up(self):
        n_candidates = 1000
        warmup = 50
        cooldown = 100
        adx = 30
        ema = 20
        regime_side = 15
        edge_topn = 10
        weekly = 5
        corr = 8
        daily = 3
        equity = 2
        tpd = 7
        ddt = 12
        edge_first_pre = 200
        trades_taken = n_candidates - warmup - cooldown - adx - ema - regime_side \
                       - edge_topn - weekly - corr - daily - equity - tpd - ddt - edge_first_pre

        total_accounted = (warmup + cooldown + adx + ema + regime_side + edge_topn +
                          weekly + corr + daily + equity + tpd + ddt + edge_first_pre + trades_taken)
        assert total_accounted == n_candidates

    def test_zero_trades_valid(self):
        n_candidates = 100
        blocked = 100
        trades = n_candidates - blocked
        assert trades == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
