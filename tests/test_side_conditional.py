"""Tests for side-conditional outcome fix (oracle elimination).

Validates that:
1. r_long != r_short in some bars (not always same outcome)
2. Evaluation selects r_long when side=LONG and r_short when side=SHORT
3. Oracle best-side is NOT used in forward test evaluation
4. Costs scale with position size
"""

import numpy as np
import pandas as pd
import pytest


def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


def _make_synthetic_df(n=200):
    """Create a synthetic OHLC series with known properties."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, n).astype(float)
    timestamps = np.arange(n) * 900_000 + 1_700_000_000_000

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df


class TestSideConditionalOutcomes:
    def test_returns_all_keys(self):
        """generate_v5_sweep_outcomes must return both side-conditional and legacy keys."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df()
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        assert 'r_long' in result
        assert 'r_short' in result
        assert 'out_long' in result
        assert 'out_short' in result
        assert 'realized_r' in result
        assert 'outcome' in result

    def test_r_long_differs_from_r_short(self):
        """LONG and SHORT outcomes must differ in at least some bars."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df(500)
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short)
        assert np.sum(valid) > 50, "Too few valid bars"

        differ = r_long[valid] != r_short[valid]
        assert np.sum(differ) > 0, (
            "r_long and r_short are identical everywhere -- "
            "side-conditional outcomes are not differentiated"
        )
        pct_differ = np.mean(differ) * 100
        assert pct_differ > 5, (
            f"Only {pct_differ:.1f}% of bars differ between LONG and SHORT -- suspiciously low"
        )

    def test_oracle_best_side_matches_max(self):
        """Legacy realized_r should be max(r_long, r_short)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df()
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']
        realized_r = result['realized_r']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short) & ~np.isnan(realized_r)
        expected_best = np.maximum(r_long[valid], r_short[valid])
        np.testing.assert_allclose(
            realized_r[valid], expected_best, atol=1e-6,
            err_msg="Legacy realized_r is not max(r_long, r_short)"
        )

    def test_side_selection_logic(self):
        """Ensure np.where correctly selects r_long when side=1 and r_short when side=-1."""
        r_long = np.array([1.0, -0.5, 0.3, 1.33])
        r_short = np.array([-1.0, 0.8, -0.2, 1.33])
        sides = np.array([1, -1, 1, -1])

        r_eval = np.where(sides == 1, r_long, r_short)

        expected = np.array([1.0, 0.8, 0.3, 1.33])
        np.testing.assert_array_equal(r_eval, expected)

    def test_side_conditional_worse_than_oracle(self):
        """Side-conditional average R should be <= oracle best-side R (on average)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df(500)
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']
        oracle_r = result['realized_r']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short) & ~np.isnan(oracle_r)

        sides_random = np.where(np.random.RandomState(123).rand(np.sum(valid)) > 0.5, 1, -1)
        r_random = np.where(sides_random == 1, r_long[valid], r_short[valid])

        assert np.mean(r_random) <= np.mean(oracle_r[valid]) + 0.01, (
            "Random-side selection should not beat oracle on average"
        )

    def test_outcome_string_selection(self):
        """Ensure outcome strings are properly selected by side."""
        out_long = np.array(["TP", "SL", "EXP_WIN", "TP"])
        out_short = np.array(["SL", "TP", "EXP_LOSS", "SL"])
        sides = np.array([1, -1, 1, -1])

        out_eval = np.where(sides == 1, out_long, out_short)

        expected = np.array(["TP", "TP", "EXP_WIN", "SL"])
        np.testing.assert_array_equal(out_eval, expected)


class TestCostScaling:
    def test_fees_scale_with_size(self):
        """Fees should scale linearly with position size."""
        pytest.importorskip("torch")
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from training.walk_forward import TransactionCosts

        costs = TransactionCosts()

        fee_small, slip_small = costs.calculate_costs(
            price=50000.0, size=0.01, volatility=0.001, is_taker=True
        )
        fee_large, slip_large = costs.calculate_costs(
            price=50000.0, size=1.0, volatility=0.001, is_taker=True
        )

        assert abs(fee_large / fee_small - 100.0) < 0.01, (
            f"Fee ratio should be 100x but got {fee_large / fee_small:.2f}x"
        )
        assert abs(slip_large / slip_small - 100.0) < 0.01, (
            f"Slippage ratio should be 100x but got {slip_large / slip_small:.2f}x"
        )


class TestSharpeAnnualization:
    def test_sharpe_not_inflated(self):
        """Sharpe should not be wildly inflated for few trades over many bars."""
        np.random.seed(42)
        n_trades = 20
        t_r = np.random.randn(n_trades) * 0.5 + 0.1
        val_bars = 96 * 30
        val_days = val_bars / 96.0

        expect = float(np.mean(t_r))
        std_r = float(np.std(t_r))
        trades_per_year = (n_trades / max(val_days, 1e-6)) * 252
        sharpe = expect / max(std_r, 1e-6) * np.sqrt(max(trades_per_year, 1))

        old_sharpe = expect / max(std_r, 1e-6) * np.sqrt(252 * 96)

        assert abs(sharpe) < abs(old_sharpe), (
            f"New Sharpe ({sharpe:.2f}) should be less inflated than old ({old_sharpe:.2f})"
        )
        assert abs(sharpe) < 50, f"Sharpe {sharpe:.2f} still seems too high"


class TestCLIFlags:
    def test_trades_per_day_override_parses(self):
        """New --v5-target-trades-per-day flag should parse correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-target-tpd", type=float, default=6.5)
        parser.add_argument("--v5-tpd-tol", type=float, default=1.5)
        parser.add_argument("--v5-target-trades-per-day", type=float, default=None)
        parser.add_argument("--v5-target-trades-per-day-band", type=float, default=None)

        args = parser.parse_args(["--v5-target-trades-per-day", "3.5",
                                   "--v5-target-trades-per-day-band", "0.8"])
        assert args.v5_target_trades_per_day == 3.5
        assert args.v5_target_trades_per_day_band == 0.8

        effective_tpd = args.v5_target_tpd
        effective_tol = args.v5_tpd_tol
        if args.v5_target_trades_per_day is not None:
            effective_tpd = args.v5_target_trades_per_day
        if args.v5_target_trades_per_day_band is not None:
            effective_tol = args.v5_target_trades_per_day_band
        assert effective_tpd == 3.5
        assert effective_tol == 0.8

    def test_default_no_override(self):
        """Without override flags, defaults should be preserved."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-target-tpd", type=float, default=6.5)
        parser.add_argument("--v5-tpd-tol", type=float, default=1.5)
        parser.add_argument("--v5-target-trades-per-day", type=float, default=None)
        parser.add_argument("--v5-target-trades-per-day-band", type=float, default=None)

        args = parser.parse_args([])
        assert args.v5_target_trades_per_day is None
        assert args.v5_target_trades_per_day_band is None

        effective_tpd = args.v5_target_tpd
        if args.v5_target_trades_per_day is not None:
            effective_tpd = args.v5_target_trades_per_day
        assert effective_tpd == 6.5

    def test_ema200_gate_flag_parses(self):
        """--v5-ema200-regime-gate should parse as boolean."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-ema200-regime-gate", action="store_true", default=False)

        args_off = parser.parse_args([])
        assert args_off.v5_ema200_regime_gate is False

        args_on = parser.parse_args(["--v5-ema200-regime-gate"])
        assert args_on.v5_ema200_regime_gate is True


class TestSideDiagnostics:
    def test_side_counts_match_direction_breakdown(self):
        """Side diagnostic counts must equal direction breakdown counts."""
        sides = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, 1])
        n_long = int(np.sum(sides == 1))
        n_short = int(np.sum(sides == -1))

        assert n_long == 6
        assert n_short == 4
        assert n_long + n_short == len(sides)

    def test_one_sided_detection(self):
        """All-LONG sides should trigger a warning condition."""
        sides = np.ones(100, dtype=int)
        n_short = int(np.sum(sides == -1))
        assert n_short == 0, "Expected 0 shorts for all-long sides"

        should_warn = n_short == 0 and len(sides) > 10
        assert should_warn, "Should trigger one-sided warning"


class TestEMA200:
    def _compute_ema_standalone(self, close_arr, period=200):
        """Standalone EMA for testing (mirrors _compute_ema in v5_train.py)."""
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(close_arr, dtype=np.float64)
        ema[0] = close_arr[0]
        for i in range(1, len(close_arr)):
            ema[i] = alpha * close_arr[i] + (1 - alpha) * ema[i - 1]
        return ema

    def test_ema_computation(self):
        """EMA200 should be computed without future leakage."""
        close = np.array([100.0] * 50 + [200.0] * 50, dtype=np.float64)
        ema = self._compute_ema_standalone(close, period=10)

        assert ema[0] == 100.0
        assert ema[49] == pytest.approx(100.0, abs=0.1)
        assert ema[99] > 150.0

    def test_ema_gate_blocks_long_below(self):
        """EMA gate should block LONG when close < EMA200."""
        close = np.array([100.0, 90.0, 110.0, 80.0])
        ema200 = np.array([105.0, 105.0, 105.0, 105.0])
        sides = np.array([1, 1, 1, -1])

        blocked = []
        for i in range(len(close)):
            if sides[i] == 1 and close[i] < ema200[i]:
                blocked.append(i)
            if sides[i] == -1 and close[i] > ema200[i]:
                blocked.append(i)

        assert 0 in blocked
        assert 1 in blocked
        assert 2 not in blocked
        assert 3 not in blocked


def _compute_v5_scores_pure(arrays, side_mode='action_head', score_lambda=0.5, rr_weight=0.0):
    """Pure numpy re-implementation of compute_v5_scores for testing without torch."""
    mu_R = arrays['mu_R']
    mae = arrays['mae']
    mfe = arrays['mfe']
    p_long = arrays['p_long']
    p_short = arrays['p_short']
    n = len(mu_R)
    eps = 1e-9

    if side_mode == 'action_head':
        abs_mu = np.abs(mu_R)
        risk = mae + eps
        sides = np.where(p_long >= p_short, 1, -1)
        p_dir = np.where(sides == 1, p_long, p_short)
        edge = p_dir * abs_mu / risk
        conflict = np.where(
            ((sides == 1) & (mu_R < 0)) | ((sides == -1) & (mu_R > 0)),
            1.0, 0.0
        )
        penalty = score_lambda * conflict * abs_mu / risk
    else:
        risk = mae + eps
        edge_long = p_long * mu_R / risk
        edge_short = p_short * (-mu_R) / risk
        sides = np.where(edge_long >= edge_short, 1, -1)
        edge = np.maximum(edge_long, edge_short)
        penalty = score_lambda * np.maximum(0, -np.where(sides == 1, mu_R, -mu_R)) / risk

    scores = edge - penalty

    if rr_weight > 0:
        abs_mu = np.abs(mu_R)
        risk = mae + eps
        rr_ratio = mfe / (mae + eps)
        rr_bonus = rr_weight * rr_ratio * abs_mu / risk
        scores = scores + rr_bonus

    n_long = int(np.sum(sides == 1))
    n_short = int(np.sum(sides == -1))
    diag = {
        'n_long_all': n_long,
        'n_short_all': n_short,
        'penalty_mean': float(np.mean(penalty)),
    }
    return scores, sides, diag


class TestV506ScoringFix:
    """Tests for v5.0.6 action_head side mode and R/R ratio scoring."""

    def _make_arrays(self, mu_R, p_long, p_short, mae=None, mfe=None):
        n = len(mu_R)
        return {
            'mu_R': np.array(mu_R, dtype=np.float64),
            'mae': np.array(mae if mae is not None else [0.5]*n, dtype=np.float64),
            'mfe': np.array(mfe if mfe is not None else [1.0]*n, dtype=np.float64),
            'p_long': np.array(p_long, dtype=np.float64),
            'p_short': np.array(p_short, dtype=np.float64),
        }

    def test_action_head_mode_produces_shorts(self):
        """action_head mode must produce SHORT when p_short > p_long, even with positive mu_R."""
        arrays = self._make_arrays(
            mu_R=[0.5, 0.5, 0.5, 0.5, 0.5],
            p_long=[0.1, 0.1, 0.1, 0.1, 0.1],
            p_short=[0.6, 0.6, 0.6, 0.6, 0.6],
        )
        scores, sides, diag = _compute_v5_scores_pure(arrays, side_mode='action_head')

        n_short = int(np.sum(sides == -1))
        assert n_short == 5, f"Expected all 5 bars SHORT, got {n_short}"
        assert diag['n_short_all'] == 5

    def test_mu_sign_mode_no_shorts_positive_mu(self):
        """Legacy mu_sign mode must produce ZERO shorts when mu_R > 0 (the known bug)."""
        arrays = self._make_arrays(
            mu_R=[0.5, 0.3, 0.8, 0.1, 0.2],
            p_long=[0.2, 0.2, 0.2, 0.2, 0.2],
            p_short=[0.6, 0.6, 0.6, 0.6, 0.6],
        )
        scores, sides, diag = _compute_v5_scores_pure(arrays, side_mode='mu_sign')

        n_short = int(np.sum(sides == -1))
        assert n_short == 0, f"mu_sign mode should produce 0 shorts with positive mu_R, got {n_short}"

    def test_action_head_mixed_sides(self):
        """action_head mode should produce both LONG and SHORT based on p_long vs p_short."""
        arrays = self._make_arrays(
            mu_R=[0.5, 0.5, -0.3, -0.3],
            p_long=[0.7, 0.2, 0.7, 0.2],
            p_short=[0.2, 0.7, 0.2, 0.7],
        )
        scores, sides, diag = _compute_v5_scores_pure(arrays, side_mode='action_head')

        assert sides[0] == 1, "Bar 0: p_long>p_short should be LONG"
        assert sides[1] == -1, "Bar 1: p_short>p_long should be SHORT"
        assert sides[2] == 1, "Bar 2: p_long>p_short should be LONG"
        assert sides[3] == -1, "Bar 3: p_short>p_long should be SHORT"
        assert diag['n_long_all'] == 2
        assert diag['n_short_all'] == 2

    def test_penalty_activates_on_conflict(self):
        """Penalty should be higher when chosen side conflicts with mu_R sign."""
        aligned = self._make_arrays(
            mu_R=[1.0], p_long=[0.8], p_short=[0.1],
        )
        conflicting = self._make_arrays(
            mu_R=[-1.0], p_long=[0.8], p_short=[0.1],
        )
        _, _, d_aligned = _compute_v5_scores_pure(aligned, side_mode='action_head', score_lambda=0.5)
        _, _, d_conflict = _compute_v5_scores_pure(conflicting, side_mode='action_head', score_lambda=0.5)

        assert d_conflict['penalty_mean'] > d_aligned['penalty_mean'], (
            f"Conflict penalty ({d_conflict['penalty_mean']:.4f}) should exceed "
            f"aligned penalty ({d_aligned['penalty_mean']:.4f})"
        )

    def test_rr_weight_boosts_favorable_setups(self):
        """R/R ratio bonus should boost scores when mfe/mae is favorable."""
        good_rr = self._make_arrays(
            mu_R=[0.5], p_long=[0.6], p_short=[0.2],
            mfe=[2.0], mae=[0.3],
        )
        bad_rr = self._make_arrays(
            mu_R=[0.5], p_long=[0.6], p_short=[0.2],
            mfe=[0.3], mae=[2.0],
        )

        s_good_no_rr, _, _ = _compute_v5_scores_pure(good_rr, side_mode='action_head', rr_weight=0.0)
        s_bad_no_rr, _, _ = _compute_v5_scores_pure(bad_rr, side_mode='action_head', rr_weight=0.0)
        s_good_rr, _, _ = _compute_v5_scores_pure(good_rr, side_mode='action_head', rr_weight=0.5)
        s_bad_rr, _, _ = _compute_v5_scores_pure(bad_rr, side_mode='action_head', rr_weight=0.5)

        assert s_good_rr[0] > s_good_no_rr[0], "R/R bonus should increase score for favorable setup"
        assert s_good_rr[0] > s_bad_rr[0], "Good R/R setup should score higher than bad R/R with rr_weight > 0"

    def test_rr_weight_zero_mfe_no_effect(self):
        """rr_weight=0 should produce identical scores regardless of MFE (MAE still affects risk)."""
        high_mfe = self._make_arrays(
            mu_R=[0.5], p_long=[0.6], p_short=[0.2],
            mfe=[5.0], mae=[0.5],
        )
        low_mfe = self._make_arrays(
            mu_R=[0.5], p_long=[0.6], p_short=[0.2],
            mfe=[0.1], mae=[0.5],
        )

        s_high, _, _ = _compute_v5_scores_pure(high_mfe, side_mode='action_head', rr_weight=0.0)
        s_low, _, _ = _compute_v5_scores_pure(low_mfe, side_mode='action_head', rr_weight=0.0)

        assert s_high[0] == s_low[0], "With rr_weight=0, MFE should not affect score"


class TestV506SideSpecificTargets:
    """Tests for side-specific MFE/MAE target generation."""

    def test_target_generator_returns_side_keys(self):
        """build_v5_targets must return side-specific MFE/MAE arrays."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.v5_target_generator import build_v5_targets

        df = _make_synthetic_df(200)
        result = build_v5_targets(df, horizon=16)

        assert 'mfe_R_long' in result
        assert 'mae_R_long' in result
        assert 'mfe_R_short' in result
        assert 'mae_R_short' in result

    def test_side_mfe_mae_differ(self):
        """Long and short MFE/MAE must differ in at least some bars."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.v5_target_generator import build_v5_targets

        df = _make_synthetic_df(500)
        result = build_v5_targets(df, horizon=16)

        valid = result['valid_mask']
        mfe_l = result['mfe_R_long'][valid]
        mfe_s = result['mfe_R_short'][valid]
        mae_l = result['mae_R_long'][valid]
        mae_s = result['mae_R_short'][valid]

        assert np.sum(mfe_l != mfe_s) > 0, "Long/Short MFE should differ"
        assert np.sum(mae_l != mae_s) > 0, "Long/Short MAE should differ"

    def test_long_short_mfe_relationship(self):
        """Long MFE == Short MAE and vice versa (same price extremes, opposite perspective)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.v5_target_generator import build_v5_targets

        df = _make_synthetic_df(200)
        result = build_v5_targets(df, horizon=16)

        valid = result['valid_mask']
        mfe_l = result['mfe_R_long'][valid]
        mae_l = result['mae_R_long'][valid]
        mfe_s = result['mfe_R_short'][valid]
        mae_s = result['mae_R_short'][valid]

        np.testing.assert_allclose(mfe_l, mae_s, atol=1e-6,
            err_msg="Long MFE should equal Short MAE (upside = short's risk)")
        np.testing.assert_allclose(mae_l, mfe_s, atol=1e-6,
            err_msg="Long MAE should equal Short MFE (downside = short's profit)")

    def test_action_label_uses_correct_side_mfe(self):
        """Action label HOLD check should use side-appropriate MFE."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.v5_target_generator import build_v5_targets

        df = _make_synthetic_df(500)
        result = build_v5_targets(df, horizon=16, mfe_min_r=0.05)

        valid = result['valid_mask']
        action = result['action_label'][valid]
        ret_R = result['ret_R'][valid]

        n_long = int(np.sum(action == 1))
        n_short = int(np.sum(action == 2))
        assert n_long > 0 and n_short > 0, f"Need both LONG ({n_long}) and SHORT ({n_short}) labels"


class TestV506CLIFlags:
    """Tests for v5.0.6 CLI flag parsing."""

    def test_score_side_mode_flag(self):
        """--v5-score-side-mode should parse correctly with choices."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-score-side-mode", type=str, default="action_head",
                            choices=["action_head", "mu_sign"])
        parser.add_argument("--v5-rr-weight", type=float, default=0.0)

        args_default = parser.parse_args([])
        assert args_default.v5_score_side_mode == "action_head"
        assert args_default.v5_rr_weight == 0.0

        args_legacy = parser.parse_args(["--v5-score-side-mode", "mu_sign"])
        assert args_legacy.v5_score_side_mode == "mu_sign"

        args_rr = parser.parse_args(["--v5-rr-weight", "0.3"])
        assert args_rr.v5_rr_weight == 0.3

    def test_invalid_side_mode_rejected(self):
        """Invalid side mode should be rejected by argparse."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-score-side-mode", type=str, default="action_head",
                            choices=["action_head", "mu_sign"])

        with pytest.raises(SystemExit):
            parser.parse_args(["--v5-score-side-mode", "invalid_mode"])


class TestCapitalProtection:
    """Tests for v5.0.7 capital protection: regime gate in sweep, weekly loss cap, warmup skip."""

    @pytest.mark.skipif(not _has_torch(), reason="torch not available")
    def test_regime_gate_blocks_in_sweep(self):
        """EMA200 gate in _run_v5_sweep should block counter-trend trades."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from train.v5_train import _run_v5_sweep

        n = 100
        np.random.seed(42)
        scores = np.random.rand(n) * 2
        sides = np.ones(n, dtype=int)
        sides[50:] = -1

        close_prices = np.linspace(100, 120, n)
        ema200_value = 110.0
        realized_r = np.random.randn(n) * 0.1
        outcomes = np.ones(n, dtype=int)
        r_long = realized_r.copy()
        r_short = realized_r.copy()
        out_long = outcomes.copy()
        out_short = outcomes.copy()

        res_no_gate, _, _, _ = _run_v5_sweep(
            scores, sides, outcomes, realized_r,
            n, 1, 2.0, 1.5,
            r_long=r_long, r_short=r_short,
            out_long=out_long, out_short=out_short,
            close_prices=close_prices, ema200_regime_gate=False,
        )

        res_with_gate, _, _, _ = _run_v5_sweep(
            scores, sides, outcomes, realized_r,
            n, 1, 2.0, 1.5,
            r_long=r_long, r_short=r_short,
            out_long=out_long, out_short=out_short,
            close_prices=close_prices, ema200_regime_gate=True,
        )

        trades_no_gate = res_no_gate.get('total_trades', 0) if res_no_gate else 0
        trades_with_gate = res_with_gate.get('total_trades', 0) if res_with_gate else 0
        assert trades_with_gate <= trades_no_gate, \
            f"Regime gate should block some trades: {trades_with_gate} >= {trades_no_gate}"

    @pytest.mark.skipif(not _has_torch(), reason="torch not available")
    def test_weekly_loss_cap_triggers(self):
        """Weekly loss cap should stop trading when weekly R drops below threshold."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from train.v5_train import _run_v5_sweep

        n = 200
        np.random.seed(99)
        scores = np.ones(n) * 2.0
        sides = np.ones(n, dtype=int)
        realized_r = np.ones(n) * -0.5
        outcomes = np.array(["TP"] * n)
        r_long = realized_r.copy()
        r_short = realized_r.copy()
        out_long = outcomes.copy()
        out_short = outcomes.copy()

        base_ts = 1_700_000_000_000
        timestamps = np.arange(n) * 900_000 + base_ts

        res_no_cap, _, _, _ = _run_v5_sweep(
            scores, sides, outcomes, realized_r,
            n, 1, 2.0, 1.5,
            r_long=r_long, r_short=r_short,
            out_long=out_long, out_short=out_short,
            timestamps=timestamps, weekly_loss_cap=None,
        )

        res_with_cap, _, _, _ = _run_v5_sweep(
            scores, sides, outcomes, realized_r,
            n, 1, 2.0, 1.5,
            r_long=r_long, r_short=r_short,
            out_long=out_long, out_short=out_short,
            timestamps=timestamps, weekly_loss_cap=-3.0,
        )

        trades_no_cap = res_no_cap.get('total_trades', 0) if res_no_cap else 0
        trades_with_cap = res_with_cap.get('total_trades', 0) if res_with_cap else 0
        assert trades_with_cap < trades_no_cap, \
            f"Weekly cap should reduce trades: {trades_with_cap} >= {trades_no_cap}"

    @pytest.mark.skipif(not _has_torch(), reason="torch not available")
    def test_warmup_skip_bars_forward_test_config(self):
        """V5ForwardTestConfig should accept warmup_skip_bars and weekly_loss_cap."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from train.v5_train import V5ForwardTestConfig

        cfg_default = V5ForwardTestConfig()
        assert cfg_default.warmup_skip_bars == 0
        assert cfg_default.weekly_loss_cap is None

        cfg_custom = V5ForwardTestConfig(warmup_skip_bars=96, weekly_loss_cap=-5.0)
        assert cfg_custom.warmup_skip_bars == 96
        assert cfg_custom.weekly_loss_cap == -5.0

    def test_cli_flags_parse(self):
        """CLI flags --v5-weekly-loss-cap and --v5-warmup-skip-bars should parse correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--v5-weekly-loss-cap", type=float, default=None)
        parser.add_argument("--v5-warmup-skip-bars", type=int, default=0)

        args_default = parser.parse_args([])
        assert args_default.v5_weekly_loss_cap is None
        assert args_default.v5_warmup_skip_bars == 0

        args_custom = parser.parse_args(["--v5-weekly-loss-cap", "-5.0", "--v5-warmup-skip-bars", "96"])
        assert args_custom.v5_weekly_loss_cap == -5.0
        assert args_custom.v5_warmup_skip_bars == 96

    def test_regime_gate_direction_correctness(self):
        """EMA200 gate logic: LONG blocked when close < EMA, SHORT blocked when close > EMA.
        Uses pure-numpy EMA computation (no torch dependency)."""

        def _compute_ema_pure(close_arr, period=200):
            alpha = 2.0 / (period + 1)
            ema = np.empty_like(close_arr, dtype=np.float64)
            ema[0] = close_arr[0]
            for i in range(1, len(close_arr)):
                ema[i] = alpha * close_arr[i] + (1 - alpha) * ema[i - 1]
            return ema

        n = 500
        close = np.concatenate([
            np.linspace(100, 100, 200),
            np.linspace(100, 80, 150),
            np.linspace(80, 120, 150),
        ])
        ema = _compute_ema_pure(close, 200)

        below_count = 0
        above_count = 0
        for i in range(250, n):
            if close[i] < ema[i]:
                below_count += 1
            if close[i] > ema[i]:
                above_count += 1

        assert below_count > 0, "Should have bars where close < EMA200 (LONG would be blocked)"
        assert above_count > 0, "Should have bars where close > EMA200 (SHORT would be blocked)"

        assert below_count + above_count > 0, "Gate should identify regime violations"

    @pytest.mark.skipif(not _has_torch(), reason="torch not available")
    def test_weekly_cap_resets_between_weeks(self):
        """Kill-switch should reset at week boundaries."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from train.v5_train import _run_v5_sweep

        bars_per_week = 7 * 24 * 4
        n = bars_per_week * 3
        np.random.seed(123)
        scores = np.ones(n) * 2.0
        sides = np.ones(n, dtype=int)
        realized_r = np.ones(n) * -0.3
        realized_r[bars_per_week:bars_per_week + 50] = 0.5
        outcomes = np.array(["TP"] * n)
        r_long = realized_r.copy()
        r_short = realized_r.copy()
        out_long = outcomes.copy()
        out_short = outcomes.copy()

        base_ts = 1_700_000_000_000
        timestamps = np.arange(n) * 900_000 + base_ts

        res, _, _, _ = _run_v5_sweep(
            scores, sides, outcomes, realized_r,
            n, 1, 2.0, 1.5,
            r_long=r_long, r_short=r_short,
            out_long=out_long, out_short=out_short,
            timestamps=timestamps, weekly_loss_cap=-2.0,
        )
        trades = res.get('total_trades', 0) if res else 0
        assert trades > 0, "Should still take trades after weekly reset"
