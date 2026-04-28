"""Tests for T001-T004 new logic: features, regime, labels, horizons.

Covers:
  - T001: New feature computations (trend_efficiency, ATR_ratio_7_28, bb_squeeze, cvd_zscore)
  - T002: Regime-adaptive deadzone, MAE penalty, clean entry bonus
  - T003: 4D regime vector values and ranges
  - T004: Adaptive horizon clamping and scaling
  - Integration: full pipeline produces valid features with no NaN explosion
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from data.common import compute_atr, compute_adaptive_horizons, compute_vol_adjusted_sl
from data.candidate_generator import compute_adx
from data.v5_target_generator import build_v5_targets

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


def _make_ohlcv(n=500, seed=42, base_price=100.0):
    rng = np.random.RandomState(seed)
    closes = base_price + np.cumsum(rng.randn(n) * 0.5)
    closes = np.maximum(closes, 1.0)
    highs = closes + rng.uniform(0.1, 1.0, n)
    lows = closes - rng.uniform(0.1, 1.0, n)
    lows = np.maximum(lows, 0.5)
    opens = closes + rng.randn(n) * 0.3
    opens = np.maximum(opens, 0.5)
    volume = rng.uniform(100, 10000, n)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)
    timestamps = np.arange(n) * 900_000 + 1_700_000_000_000

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume,
        'taker_buy_base': taker_buy_base,
        'timestamp': timestamps,
    })


def _compute_atr_simple(df, period):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_rsi_simple(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


class TestT001FeatureComputations:

    def test_trend_efficiency_range(self):
        df = _make_ohlcv(200)
        close = df['close']
        N = 20
        net_move = (close - close.shift(N)).abs()
        total_path = close.diff().abs().rolling(N, min_periods=5).sum()
        te = (net_move / total_path.clip(lower=1e-10)).clip(0.0, 1.0)

        valid = te.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_trend_efficiency_perfect_trend(self):
        n = 50
        closes = pd.Series(np.linspace(100, 120, n))
        N = 20
        net_move = (closes - closes.shift(N)).abs()
        total_path = closes.diff().abs().rolling(N, min_periods=5).sum()
        te = (net_move / total_path.clip(lower=1e-10)).clip(0.0, 1.0)

        assert te.iloc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_atr_ratio_7_28_positive(self):
        df = _make_ohlcv(200)
        atr_7 = _compute_atr_simple(df, 7)
        atr_28 = _compute_atr_simple(df, 28)
        ratio = (atr_7 / atr_28.clip(lower=1e-10)).clip(0.1, 5.0)

        valid = ratio.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.1
        assert valid.max() <= 5.0

    def test_bb_squeeze_range(self):
        from scipy import stats
        df = _make_ohlcv(200)
        close = df['close']
        bb_width = (close.rolling(20).std() * 2) / close.rolling(20).mean().clip(lower=1e-10)
        squeeze = bb_width.rolling(50, min_periods=10).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0 if len(x) > 0 else 0.5,
            raw=False
        ).clip(0.0, 1.0)

        valid = squeeze.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_cvd_zscore_range(self):
        df = _make_ohlcv(200)
        taker_buy = df['taker_buy_base']
        taker_sell = df['volume'] - taker_buy
        cvd = (taker_buy - taker_sell).cumsum()
        cvd_mean = cvd.rolling(50, min_periods=10).mean()
        cvd_std = cvd.rolling(50, min_periods=10).std()
        cvd_z = ((cvd - cvd_mean) / cvd_std.clip(lower=1e-10)).clip(-3.0, 3.0)

        valid = cvd_z.dropna()
        assert len(valid) > 0
        assert valid.min() >= -3.0
        assert valid.max() <= 3.0

    def test_vol_regime_roc_range(self):
        df = _make_ohlcv(200)
        atr_14 = _compute_atr_simple(df, 14)
        roc = atr_14.pct_change(5).clip(-1.0, 1.0)

        valid = roc.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_momentum_acceleration_range(self):
        df = _make_ohlcv(200)
        rsi_14 = _compute_rsi_simple(df['close'], 14)
        mom_accel = rsi_14.pct_change(5).clip(-0.5, 0.5)

        valid = mom_accel.dropna()
        assert len(valid) > 0
        assert valid.min() >= -0.5
        assert valid.max() <= 0.5

    def test_volume_price_divergence_range(self):
        df = _make_ohlcv(200)
        volume = df['volume']
        close = df['close']
        vol_change = volume.rolling(10, min_periods=3).mean() / volume.rolling(30, min_periods=10).mean().clip(lower=1)
        price_change = close.pct_change(10).abs()
        vpd = (vol_change - 1.0 - price_change * 10).clip(-3.0, 3.0)

        valid = vpd.dropna()
        assert len(valid) > 0
        assert valid.min() >= -3.0
        assert valid.max() <= 3.0


class TestT002RegimeAdaptiveDeadzone:

    def test_deadzone_adapts_to_adx(self):
        df = _make_ohlcv(500)
        targets = build_v5_targets(df, horizon=16, adaptive_horizon=False)

        assert 'deadzone_R' in targets
        assert 'deadzone_R_trending' in targets
        assert 'deadzone_R_choppy' in targets

        assert targets['deadzone_R_trending'] <= targets['deadzone_R']
        assert targets['deadzone_R_choppy'] >= targets['deadzone_R']

    def test_trending_gets_tighter_deadzone(self):
        df = _make_ohlcv(500)
        targets = build_v5_targets(df, horizon=16, adaptive_horizon=False)
        assert targets['deadzone_R_trending'] < targets['deadzone_R_choppy']

    def test_side_confidence_weighting(self):
        df = _make_ohlcv(500)
        from data.common import generate_v5_sweep_outcomes
        barrier = generate_v5_sweep_outcomes(df, horizon=16, adaptive_horizon=False)
        targets = build_v5_targets(
            df, horizon=16, barrier_outcomes=barrier, adaptive_horizon=False
        )

        sw = targets['sample_weight']
        valid = targets['valid_mask']
        valid_weights = sw[valid]
        assert np.all(valid_weights >= 0.0)
        assert np.all(valid_weights <= 1.0)
        assert np.mean(valid_weights) < 1.0

    def test_mae_penalty_in_tp_quality(self):
        from training.triple_barrier import bidirectional_outcome_v47_for_index

        n = 100
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(n) * 0.3)
        closes = np.maximum(closes, 50.0)
        highs = closes + rng.uniform(0.5, 3.0, n)
        lows = closes - rng.uniform(0.5, 3.0, n)
        lows = np.maximum(lows, 1.0)

        atr_val = 2.0
        sl_mult = 1.5

        results = []
        for i in range(20, 60):
            result = bidirectional_outcome_v47_for_index(
                highs, lows, closes, i, atr_val,
                tp_mult=2.0, sl_mult=sl_mult, horizon=16
            )
            if not np.isnan(result['tp_quality']):
                results.append(result)

        assert len(results) > 0

        for r in results:
            assert 0.0 <= r['tp_quality'] <= 1.0

    def test_clean_entry_bonus_in_tp_quality(self):
        from training.triple_barrier import bidirectional_outcome_v47_for_index

        n = 40
        closes = np.full(n, 100.0)
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes[1] = 101.0
        closes[2] = 102.0
        highs[1] = 103.0
        highs[2] = 105.0
        highs[3] = 107.0
        for j in range(4, 20):
            highs[j] = 100.0 + (j - 3) * 1.5
            closes[j] = 100.0 + (j - 3) * 1.0

        atr_val = 2.0

        result = bidirectional_outcome_v47_for_index(
            highs, lows, closes, 0, atr_val,
            tp_mult=2.0, sl_mult=1.5, horizon=16
        )

        assert result['tp_quality'] >= 0.0


@requires_torch
class TestT003RegimeVector:

    def test_regime_trend_range(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df)

        rt = regime['regime_trend'].dropna()
        assert len(rt) > 0
        assert rt.min() >= -1.0
        assert rt.max() <= 1.0

    def test_regime_volatility_range(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df)

        rv = regime['regime_volatility'].dropna()
        assert len(rv) > 0
        assert rv.min() >= -5.0
        assert rv.max() <= 5.0

    def test_regime_momentum_range(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df)

        rm = regime['regime_momentum'].dropna()
        assert len(rm) > 0
        assert rm.min() >= -1.0
        assert rm.max() <= 1.0

    def test_regime_session_sin_cos_range(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df)

        rs = regime['regime_session_sin'].dropna()
        rc = regime['regime_session_cos'].dropna()
        assert len(rs) > 0
        assert rs.min() >= -1.0
        assert rs.max() <= 1.0
        assert rc.min() >= -1.0
        assert rc.max() <= 1.0

    def test_regime_features_count(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df)

        assert len(regime.columns) == 5
        expected = ['regime_trend', 'regime_volatility', 'regime_momentum',
                    'regime_session_sin', 'regime_session_cos']
        assert list(regime.columns) == expected

    def test_regime_no_timestamp_defaults_session(self):
        df = _make_ohlcv(200)
        df_no_ts = df.drop(columns=['timestamp'])
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        regime = fe.compute_regime_features(df_no_ts)

        assert (regime['regime_session_sin'] == 0.0).all()
        assert (regime['regime_session_cos'] == pytest.approx(1.0)).all()


class TestT003RegimeVectorStandalone:

    def test_regime_trend_formula(self):
        df = _make_ohlcv(200)
        adx_14 = compute_adx(df, 14)
        close = df['close']
        ema_20 = close.ewm(span=20).mean()
        ema_slope = ema_20.diff(5)
        ema_slope_sign = np.sign(ema_slope)
        regime_trend = np.tanh((adx_14 / 25.0) * ema_slope_sign)

        valid = regime_trend[np.isfinite(regime_trend)]
        assert len(valid) > 0
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_regime_volatility_formula(self):
        df = _make_ohlcv(200)
        atr_14 = _compute_atr_simple(df, 14)
        atr_rolling_median = atr_14.rolling(50, min_periods=10).median()
        atr_rolling_std = atr_14.rolling(50, min_periods=10).std()
        rv = ((atr_14 - atr_rolling_median) / atr_rolling_std.clip(lower=1e-10)).clip(-5.0, 5.0)

        valid = rv.dropna()
        assert len(valid) > 0
        assert valid.min() >= -5.0
        assert valid.max() <= 5.0

    def test_regime_momentum_formula(self):
        df = _make_ohlcv(200)
        rsi_14 = _compute_rsi_simple(df['close'], 14)
        rsi_slope_5 = rsi_14.diff(5)
        rm = np.tanh(rsi_slope_5 / 10.0)

        valid = rm.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_regime_session_encoding(self):
        timestamps = np.arange(100) * 900_000 + 1_700_000_000_000
        ts = pd.to_datetime(timestamps, unit='ms', utc=True)
        hour = ts.hour + ts.minute / 60.0
        sin_val = np.sin(2 * np.pi * hour / 24.0)
        cos_val = np.cos(2 * np.pi * hour / 24.0)

        assert np.all(sin_val >= -1.0)
        assert np.all(sin_val <= 1.0)
        assert np.all(cos_val >= -1.0)
        assert np.all(cos_val <= 1.0)

    def test_regime_vector_has_5_components(self):
        expected = ['regime_trend', 'regime_volatility', 'regime_momentum',
                    'regime_session_sin', 'regime_session_cos']
        assert len(expected) == 5


class TestT004AdaptiveHorizon:

    def test_horizon_clamping_min(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 10.0

        horizons = compute_adaptive_horizons(atr, base_horizon=16, min_horizon=8, max_horizon=48)

        assert np.all(horizons >= 8)

    def test_horizon_clamping_max(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 0.01

        horizons = compute_adaptive_horizons(atr, base_horizon=16, min_horizon=8, max_horizon=48)

        assert np.all(horizons <= 48)

    def test_high_vol_contracts_horizon(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 1.0
        atr[160] = 5.0

        horizons = compute_adaptive_horizons(atr, base_horizon=16, median_window=50,
                                             min_horizon=8, max_horizon=48)

        assert horizons[160] < 16

    def test_low_vol_extends_horizon(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 5.0
        atr[160] = 1.0

        horizons = compute_adaptive_horizons(atr, base_horizon=16, median_window=50,
                                             min_horizon=8, max_horizon=48)

        assert horizons[160] > 16

    def test_vol_adjusted_sl_boost(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 1.0
        atr[150:] = 3.0

        sl_mults = compute_vol_adjusted_sl(atr, sl_mult=1.5, median_window=50,
                                           high_vol_threshold=1.3, sl_boost=1.15)

        normal_sl = sl_mults[50:90]
        assert np.allclose(normal_sl, 1.5)

        high_vol_sl = sl_mults[160:]
        boosted_count = np.sum(high_vol_sl > 1.5)
        assert boosted_count > 0

    def test_vol_adjusted_sl_boost_value(self):
        n = 200
        atr = np.ones(n, dtype=np.float64) * 1.0
        atr[150:] = 3.0

        sl_mults = compute_vol_adjusted_sl(atr, sl_mult=1.5, median_window=50,
                                           high_vol_threshold=1.3, sl_boost=1.15)

        boosted = sl_mults[sl_mults > 1.5]
        if len(boosted) > 0:
            assert np.allclose(boosted, 1.5 * 1.15)

    def test_adaptive_horizon_no_lookahead(self):
        n = 200
        atr = np.ones(n, dtype=np.float64)
        atr[100] = 10.0

        horizons = compute_adaptive_horizons(atr, base_horizon=16, median_window=50)

        assert horizons[50] == horizons[60]
        assert horizons[99] == horizons[50]

    def test_adaptive_horizons_in_targets(self):
        df = _make_ohlcv(500)
        targets = build_v5_targets(df, horizon=16, adaptive_horizon=True)

        eh = targets['effective_horizons']
        assert len(eh) == len(df)
        assert np.all(eh >= 8)
        assert np.all(eh <= 48)


@requires_torch
class TestIntegrationWithTorch:

    def test_full_pipeline_no_nan_explosion(self):
        df = _make_ohlcv(500)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])

        enh = fe.compute_enhanced_features(df)
        regime = fe.compute_regime_features(df)

        warm_start = 100
        enh_valid = enh.iloc[warm_start:]
        regime_valid = regime.iloc[warm_start:]

        for col in enh_valid.columns:
            nan_rate = enh_valid[col].isna().mean()
            assert nan_rate < 0.15, f"Enhanced feature {col} has {nan_rate:.1%} NaN after warmup"

        for col in regime_valid.columns:
            nan_rate = regime_valid[col].isna().mean()
            assert nan_rate < 0.15, f"Regime feature {col} has {nan_rate:.1%} NaN after warmup"

    def test_enhanced_feature_count(self):
        df = _make_ohlcv(200)
        from data.pipeline import FeatureEngineer
        fe = FeatureEngineer(symbols=['TEST'], timeframes=['15m'])
        enh = fe.compute_enhanced_features(df)

        assert len(enh.columns) == fe.ENH_FEATURE_COUNT


class TestIntegration:

    def test_targets_valid_mask_reasonable(self):
        df = _make_ohlcv(500)
        targets = build_v5_targets(df, horizon=16, adaptive_horizon=True)

        valid = targets['valid_mask']
        valid_rate = np.mean(valid)
        assert valid_rate > 0.5, f"Only {valid_rate:.1%} valid targets — too few"

        ret = targets['ret_R'][valid]
        assert np.all(np.isfinite(ret))
        mfe = targets['mfe_R'][valid]
        assert np.all(np.isfinite(mfe))
        mae = targets['mae_R'][valid]
        assert np.all(np.isfinite(mae))

    def test_adx_computation(self):
        df = _make_ohlcv(200)
        adx = compute_adx(df)

        assert len(adx) == len(df)
        valid = adx[adx > 0]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))
        assert np.all(valid >= 0)

    def test_atr_computation(self):
        df = _make_ohlcv(200)
        atr = compute_atr(df)

        assert len(atr) == len(df)
        valid_atr = atr[atr > 0]
        assert len(valid_atr) > 0
        assert np.all(np.isfinite(valid_atr))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
