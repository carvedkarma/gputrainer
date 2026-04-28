"""Tests for Drawdown-Adaptive Throttle (DDT)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from train.drawdown_throttle import DrawdownAdaptiveThrottle, DDTConfig


def make_ddt(**overrides):
    defaults = dict(
        enabled=True,
        lookback_trades=10,
        bad_rollr=3.0,
        thr_k=0.60,
        thr_min=0.08,
        thr_max=0.25,
        size_k=0.70,
        min_size_mult=0.25,
        alpha_down=0.30,
        alpha_up=0.05,
        warmup_trades=5,
    )
    defaults.update(overrides)
    return DrawdownAdaptiveThrottle(DDTConfig(**defaults))


class TestDDTBasic:
    def test_initial_state_zero_throttle(self):
        ddt = make_ddt()
        assert ddt.throttle_state == 0.0
        assert ddt.effective_threshold(0.10) == 0.10
        assert ddt.size_multiplier() == 1.0

    def test_warmup_no_throttle(self):
        ddt = make_ddt(warmup_trades=5)
        for _ in range(4):
            ddt.update_on_trade_close(-1.0)
        assert ddt.compute_raw_throttle() == 0.0
        assert ddt.throttle_state == 0.0

    def test_warmup_then_throttle(self):
        ddt = make_ddt(warmup_trades=5, lookback_trades=10, bad_rollr=3.0)
        for _ in range(5):
            ddt.update_on_trade_close(-0.5)
        raw = ddt.compute_raw_throttle()
        assert raw > 0.0, f"raw={raw} should be > 0 after warmup with losses"

    def test_all_losses_max_throttle(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=10, bad_rollr=3.0,
                       alpha_down=1.0)
        for _ in range(10):
            ddt.update_on_trade_close(-1.0)
        raw = ddt.compute_raw_throttle()
        assert raw == 1.0

    def test_all_wins_zero_throttle(self):
        ddt = make_ddt(warmup_trades=3)
        for _ in range(10):
            ddt.update_on_trade_close(1.0)
        raw = ddt.compute_raw_throttle()
        assert raw == 0.0
        assert ddt.throttle_state == 0.0

    def test_mixed_positive_sum_zero_throttle(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=10)
        for _ in range(7):
            ddt.update_on_trade_close(1.0)
        for _ in range(3):
            ddt.update_on_trade_close(-0.5)
        raw = ddt.compute_raw_throttle()
        assert raw == 0.0

    def test_rolling_window_drops_old_trades(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=5, bad_rollr=3.0,
                       alpha_down=1.0, alpha_up=1.0)
        for _ in range(5):
            ddt.update_on_trade_close(-1.0)
        assert ddt.compute_raw_throttle() == 1.0
        for _ in range(5):
            ddt.update_on_trade_close(1.0)
        assert ddt.compute_raw_throttle() == 0.0


class TestDDTEffectiveThreshold:
    def test_threshold_raised_during_drawdown(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=10, bad_rollr=3.0,
                       thr_k=0.60, alpha_down=1.0)
        for _ in range(10):
            ddt.update_on_trade_close(-1.0)
        thr = ddt.effective_threshold(0.10)
        assert thr > 0.10
        assert thr <= 0.25

    def test_threshold_clamp_min(self):
        ddt = make_ddt(thr_min=0.08)
        thr = ddt.effective_threshold(0.05)
        assert thr >= 0.08

    def test_threshold_clamp_max(self):
        ddt = make_ddt(warmup_trades=1, lookback_trades=5, bad_rollr=1.0,
                       thr_k=10.0, thr_max=0.25, alpha_down=1.0)
        for _ in range(5):
            ddt.update_on_trade_close(-1.0)
        thr = ddt.effective_threshold(0.10)
        assert thr == 0.25

    def test_disabled_returns_base(self):
        ddt = DrawdownAdaptiveThrottle(DDTConfig(enabled=False))
        assert ddt.effective_threshold(0.15) == 0.15
        assert ddt.size_multiplier() == 1.0


class TestDDTSizeMultiplier:
    def test_size_reduced_during_drawdown(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=10, bad_rollr=3.0,
                       size_k=0.70, alpha_down=1.0)
        for _ in range(10):
            ddt.update_on_trade_close(-1.0)
        sm = ddt.size_multiplier()
        assert sm < 1.0
        assert sm >= 0.25

    def test_size_clamp_min(self):
        ddt = make_ddt(warmup_trades=1, lookback_trades=5, bad_rollr=1.0,
                       size_k=0.99, min_size_mult=0.25, alpha_down=1.0)
        for _ in range(5):
            ddt.update_on_trade_close(-1.0)
        sm = ddt.size_multiplier()
        assert sm == 0.25

    def test_size_normal_when_winning(self):
        ddt = make_ddt(warmup_trades=3)
        for _ in range(10):
            ddt.update_on_trade_close(0.5)
        assert ddt.size_multiplier() == 1.0


class TestDDTHysteresis:
    def test_tighten_fast_loosen_slow(self):
        ddt = make_ddt(warmup_trades=3, lookback_trades=20, bad_rollr=5.0,
                       alpha_down=0.50, alpha_up=0.05)
        for _ in range(10):
            ddt.update_on_trade_close(-1.0)
        peak_throttle = ddt.throttle_state
        assert peak_throttle > 0.0
        for _ in range(5):
            ddt.update_on_trade_close(2.0)
        assert ddt.throttle_state < peak_throttle
        assert ddt.throttle_state > 0.0, "Should still be partially throttled (slow recovery)"


class TestDDTDiagnostics:
    def test_diagnostics_keys(self):
        ddt = make_ddt()
        d = ddt.diagnostics()
        assert 'throttle_level' in d
        assert 'rolling_sum_r' in d
        assert 'total_trades_seen' in d
        assert 'max_throttle_seen' in d
        assert 'lookback_trades' in d

    def test_get_diagnostics_keys(self):
        ddt = make_ddt()
        d = ddt.get_diagnostics()
        assert 'ddt_enabled' in d
        assert 'params' in d
        assert 'final_throttle_state' in d

    def test_trade_log(self):
        ddt = make_ddt(warmup_trades=2)
        for _ in range(3):
            ddt.update_on_trade_close(-0.5)
        thr = ddt.effective_threshold(0.10)
        sm = ddt.size_multiplier()
        ddt.log_trade_close(-0.5, thr, sm)
        assert len(ddt._trade_log) == 1


class TestDDTRawThrottleLinear:
    def test_partial_throttle(self):
        ddt = make_ddt(warmup_trades=1, lookback_trades=10, bad_rollr=6.0,
                       alpha_down=1.0, alpha_up=1.0)
        for _ in range(5):
            ddt.update_on_trade_close(-0.6)
        raw = ddt.compute_raw_throttle()
        expected = 3.0 / 6.0
        assert abs(raw - expected) < 0.01, f"raw={raw} expected={expected}"

    def test_zero_at_boundary(self):
        ddt = make_ddt(warmup_trades=1, lookback_trades=10, bad_rollr=6.0)
        for _ in range(5):
            ddt.update_on_trade_close(0.0)
        assert ddt.compute_raw_throttle() == 0.0


class TestDDTNanHandling:
    def test_nan_trade_ignored(self):
        ddt = make_ddt(warmup_trades=3)
        ddt.update_on_trade_close(float('nan'))
        assert len(ddt.closed_r_history) == 0
        ddt.update_on_trade_close(1.0)
        assert len(ddt.closed_r_history) == 1


class TestMaxThresholdLogic:
    """Tests for max_threshold ceiling cap logic (mirrors run_v5_forward_test threshold computation)."""

    def _compute_effective(self, score_threshold, min_threshold=None, max_threshold=None):
        hard_floor = min_threshold if min_threshold is not None else 0.10
        effective = max(hard_floor, score_threshold)
        if max_threshold is not None and effective > max_threshold:
            effective = max_threshold
        return effective

    def test_cap_reduces_high_calibrated(self):
        assert self._compute_effective(0.20, min_threshold=0.05, max_threshold=0.10) == 0.10

    def test_cap_no_effect_when_below(self):
        assert self._compute_effective(0.08, min_threshold=0.05, max_threshold=0.15) == 0.08

    def test_cap_none_means_no_ceiling(self):
        assert self._compute_effective(0.30, min_threshold=0.05, max_threshold=None) == 0.30

    def test_cap_equals_threshold(self):
        assert self._compute_effective(0.12, min_threshold=0.05, max_threshold=0.12) == 0.12

    def test_floor_wins_when_cap_above_floor(self):
        effective = self._compute_effective(0.03, min_threshold=0.08, max_threshold=0.15)
        assert effective == 0.08

    def test_cap_below_floor_caps_to_cap(self):
        effective = self._compute_effective(0.20, min_threshold=0.12, max_threshold=0.10)
        assert effective == 0.10
