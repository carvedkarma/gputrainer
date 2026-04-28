"""Tests for v5.2.0 precision audit fixes (T001-T006).

Covers:
  T001: Lookahead bias fix in quality gate (ref_arrays)
  T002: Oracle fallback removal (ValueError on missing side-conditional)
  T003: Head disagreement gate
  T004: Statistical edge metrics (Sortino, t-stat, CI)
  T005: Train/test purge gap
  T006: Slippage deduction in score computation
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

if HAS_TORCH:
    from train.v5_train import (
        v5_quality_mask,
        V5QualityGateConfig,
        V5ForwardTestConfig,
        compute_v5_scores,
        _compute_time_split,
        _compute_forward_metrics,
    )


def _make_arrays(n=500, seed=42):
    rng = np.random.RandomState(seed)
    return {
        'mu_R': rng.randn(n).astype(np.float32) * 0.1,
        'mae': np.abs(rng.randn(n).astype(np.float32) * 0.3) + 0.01,
        'mfe': np.abs(rng.randn(n).astype(np.float32) * 0.3) + 0.01,
        'sigma': np.abs(rng.randn(n).astype(np.float32) * 0.2) + 0.01,
        'p_trade': rng.uniform(0.2, 0.9, n).astype(np.float32),
        'p_long': rng.uniform(0.1, 0.8, n).astype(np.float32),
        'p_short': rng.uniform(0.1, 0.8, n).astype(np.float32),
    }


class TestT001LookaheadBiasQualityGate:

    def test_ref_arrays_used_for_percentiles(self):
        test_arrays = _make_arrays(n=500, seed=1)
        test_arrays['mu_R'][:] = 0.5
        # Heterogeneous mae: 450 bars at 0.01, 50 bars at 0.5.
        # Without ref: adaptive_mae ≈ percentile([0.01]*450+[0.5]*50, 90) ≈ 0.059
        #   → bars with mae=0.5 FAIL threshold → ~450 pass.
        # With ref (mae=5.0): adaptive_mae = min(1.0, 5.0 * <cfg factor>) = 1.0
        #   → all 500 bars pass.
        # 450 ≠ 500 confirms ref_arrays change the gate.
        test_arrays['mae'][:450] = 0.01
        test_arrays['mae'][450:] = 0.5
        test_arrays['sigma'][:] = 0.01
        test_arrays['p_trade'][:] = 0.99

        ref_arrays = _make_arrays(n=500, seed=2)
        ref_arrays['mu_R'][:] = 0.01
        ref_arrays['mae'][:] = 5.0
        ref_arrays['sigma'][:] = 5.0
        ref_arrays['p_trade'][:] = 0.1

        cfg = V5QualityGateConfig()

        mask_no_ref, _ = v5_quality_mask(test_arrays, cfg, epoch=999)
        mask_with_ref, _ = v5_quality_mask(test_arrays, cfg, epoch=999, ref_arrays=ref_arrays)

        pass_no_ref = np.sum(mask_no_ref)
        pass_with_ref = np.sum(mask_with_ref)
        assert pass_no_ref != pass_with_ref, \
            "ref_arrays should change which bars pass the quality gate"

    def test_warmup_bypass_ignores_ref(self):
        test_arrays = _make_arrays(n=100)
        ref_arrays = _make_arrays(n=100)
        ref_arrays['sigma'][:] = 0.001

        cfg = V5QualityGateConfig()
        mask, diag = v5_quality_mask(test_arrays, cfg, epoch=2, ref_arrays=ref_arrays)
        assert diag['warmup_bypass'] is True
        assert np.all(mask)

    def test_no_ref_arrays_backward_compatible(self):
        arrays = _make_arrays(n=500)
        cfg = V5QualityGateConfig()
        mask1, _ = v5_quality_mask(arrays, cfg, epoch=999)
        mask2, _ = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=None)
        np.testing.assert_array_equal(mask1, mask2)


class TestT002OracleFallbackRemoval:

    def test_missing_side_arrays_raises_error(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        n = 50
        dummy_out = {
            'ret_mu': torch.randn(n, 1),
            'mae': torch.abs(torch.randn(n, 1)),
            'mfe': torch.abs(torch.randn(n, 1)),
            'action_logits': torch.randn(n, 3),
            'ret_log_sigma': torch.randn(n, 1),
        }
        model.return_value = dummy_out
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])

        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        with pytest.raises(ValueError, match="REQUIRED"):
            from train.v5_train import run_v5_forward_test
            run_v5_forward_test(
                model=model,
                device=torch.device('cpu'),
                test_features=np.random.randn(n, 10).astype(np.float32),
                test_outcomes=np.array(["TP"] * n),
                test_realized_r=np.random.randn(n).astype(np.float32),
                test_sym_ids=np.zeros(n, dtype=np.int64),
                test_cand_mask=np.ones(n, dtype=bool),
                test_valid=np.ones(n, dtype=np.float32),
                test_bars=n,
                config=config,
            )


class TestT003HeadDisagreementGate:

    def test_config_default_off(self):
        cfg = V5ForwardTestConfig()
        assert cfg.head_disagreement_gate is False

    def test_config_can_enable(self):
        cfg = V5ForwardTestConfig(head_disagreement_gate=True)
        assert cfg.head_disagreement_gate is True


class TestT004StatisticalMetrics:

    def test_sortino_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.02
        t_outcomes = np.array(["TP"] * 60 + ["SL"] * 40)
        t_sides = np.where(t_r > 0, 1, -1)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 'sortino' in report
        assert isinstance(report['sortino'], float)
        assert np.isfinite(report['sortino'])

    def test_tstat_pvalue_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.05
        t_outcomes = np.array(["TP"] * n)
        t_sides = np.ones(n, dtype=int)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 't_stat' in report
        assert 'p_value' in report
        assert report['t_stat'] > 0
        assert 0 < report['p_value'] < 1

    def test_bootstrap_ci_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.03
        t_outcomes = np.array(["TP"] * n)
        t_sides = np.ones(n, dtype=int)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 'ci_95_lower' in report
        assert 'ci_95_upper' in report
        assert report['ci_95_lower'] < report['ci_95_upper']
        assert report['ci_95_lower'] < report['expectancy_r'] < report['ci_95_upper']

    def test_few_trades_no_crash(self):
        t_r = np.array([0.1], dtype=np.float32)
        t_outcomes = np.array(["TP"])
        t_sides = np.array([1])
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, 96, config,
            start_date="2024-01-01", end_date="2024-01-02"
        )
        assert report['total_trades'] == 1
        assert report['t_stat'] == 0.0
        assert report['ci_95_lower'] == 0.0


class TestT005PurgeGap:

    def test_purge_removes_end_of_train(self):
        import pandas as pd
        n = 1000
        timestamps = np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_no_purge, test_no_purge = _compute_time_split(sym_df)
        train_purge, test_purge = _compute_time_split(sym_df, purge_bars=24)

        assert len(train_purge) == len(train_no_purge) - 24
        assert len(test_purge) == len(test_no_purge)

    def test_purge_with_date_split(self):
        import pandas as pd
        from datetime import datetime, timezone

        n = 2000
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        timestamps = base_ts + np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_no_purge, test_no_purge = _compute_time_split(
            sym_df, train_end_date="2024-01-15",
            test_start_date="2024-01-15"
        )
        train_purge, test_purge = _compute_time_split(
            sym_df, train_end_date="2024-01-15",
            test_start_date="2024-01-15",
            purge_bars=24
        )

        assert len(train_purge) < len(train_no_purge)
        assert len(train_purge) == len(train_no_purge) - 24
        assert len(test_purge) == len(test_no_purge)

    def test_zero_purge_unchanged(self):
        import pandas as pd
        n = 500
        timestamps = np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_a, test_a = _compute_time_split(sym_df, purge_bars=0)
        train_b, test_b = _compute_time_split(sym_df)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


class TestT006SlippageDeduction:

    def test_slippage_reduces_scores(self):
        arrays = _make_arrays(n=200, seed=42)
        arrays['mu_R'][:] = 0.1
        arrays['mae'][:] = 0.05
        arrays['mfe'][:] = 0.15
        arrays['p_long'][:] = 0.7
        arrays['p_short'][:] = 0.3

        scores_no_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=0.0
        )
        scores_with_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=5.0
        )

        finite_no = scores_no_slip[np.isfinite(scores_no_slip)]
        finite_with = scores_with_slip[np.isfinite(scores_with_slip)]
        assert np.mean(finite_with) < np.mean(finite_no), \
            "Slippage should reduce average scores"

    def test_zero_slippage_unchanged(self):
        arrays = _make_arrays(n=200, seed=42)

        scores_a, sides_a, _ = compute_v5_scores(None, _arrays=arrays, slippage_bps=0.0)
        scores_b, sides_b, _ = compute_v5_scores(None, _arrays=arrays)

        np.testing.assert_array_almost_equal(scores_a, scores_b)
        np.testing.assert_array_equal(sides_a, sides_b)

    def test_high_slippage_kills_small_edge(self):
        arrays = _make_arrays(n=100, seed=42)
        arrays['mu_R'][:] = 0.005
        arrays['mae'][:] = 0.05
        arrays['mfe'][:] = 0.01
        arrays['p_long'][:] = 0.6
        arrays['p_short'][:] = 0.4

        scores_no_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=0.0, min_mu_r_score=0.0
        )
        scores_big_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=10.0, min_mu_r_score=0.0
        )

        finite_no = scores_no_slip[np.isfinite(scores_no_slip)]
        finite_slip = scores_big_slip[np.isfinite(scores_big_slip)]
        if len(finite_slip) > 0 and len(finite_no) > 0:
            assert np.mean(finite_slip) < np.mean(finite_no)

    def test_config_has_slippage(self):
        cfg = V5ForwardTestConfig(slippage_base_bps=2.5)
        assert cfg.slippage_base_bps == 2.5

    def test_default_slippage_is_bitget_roundtrip(self):
        cfg = V5ForwardTestConfig()
        assert cfg.slippage_base_bps == 6.0, (
            "Default slippage must be 6 bps (Bitget 3 bps taker × 2 sides), "
            f"got {cfg.slippage_base_bps}"
        )


class TestProductionReadinessAuditFixes:
    """Regression tests for bugs found in the Mar-2026 production-readiness audit.

    T_BUG1: edge_pass now applied in final_mask (quality gate mu_R filter live)
    T_BUG2: score_threshold default is 0.02 (matches live shared config)
    T_BUG3: slippage_base_bps default is 6.0 bps (tested in T006 class above)
    T_BUG4: _compute_score_decile_table detects monotonicity
    """

    def test_bug1_edge_pass_filters_low_mu_r_bars(self):
        """T1: quality_mask with strict_audit=True must filter bars with |mu_R| < p25 threshold.

        Setup: 500 bars with tiny mu_R (uniform[-0.001, 0.001]) but ref has large mu_R (0.5).
        adaptive_mu_min = p25(ref) = 0.5.  All bars fail edge_pass → pass_rate < 10%.
        Relax loop runs 5 steps but relaxed_mu=0.25 still >> 0.001 → still 0 bars.
        strict_audit=True: keeps the last relaxed_mask (0 bars) without finiteness bypass.
        Expected: n_passed == 0  (well below the 25% threshold).
        """
        n = 500
        rng = np.random.RandomState(0)
        arrays = {
            'mu_R':    rng.uniform(-0.001, 0.001, n).astype(np.float32),  # tiny mu_R, all near-zero
            'mae':     np.full(n, 0.1, dtype=np.float32),
            'sigma':   np.full(n, 0.1, dtype=np.float32),
            'p_trade': np.full(n, 0.7, dtype=np.float32),
        }
        ref = {
            'mu_R':    np.full(n, 0.5, dtype=np.float32),  # ref has large mu_R → p25 will be large
            'mae':     np.full(n, 0.01, dtype=np.float32),
            'sigma':   np.full(n, 0.01, dtype=np.float32),
            'p_trade': np.full(n, 0.99, dtype=np.float32),
        }
        # T1 Fix: use strict_audit=True so the finiteness bypass cannot drop edge_pass.
        cfg = V5QualityGateConfig(mu_R_min=0.05, quality_gate_strict_audit=True)
        mask, diag = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=ref)
        n_passed = int(np.sum(mask))
        assert n_passed == 0, (
            f"strict_audit=True: edge_pass must filter ALL low-mu_R bars (got {n_passed}/{n} passed). "
            "If this test fails, the finiteness fallback is bypassing edge_pass under strict_audit."
        )

    def test_bug1_edge_pass_preserved_in_default_fallback(self):
        """T1 fallback fix: default (strict_audit=False) must also preserve edge floor in fallback.

        Even without strict_audit, the finiteness fallback now includes the last_relaxed_mu floor.
        With ref mu_R=0.5, last_relaxed_mu after 5 steps = 0.5/2.0 = 0.25.
        Arrays mu_R is uniform(-0.001, 0.001) — all below 0.25 → still filtered.
        Expected: n_passed < 5 (effectively 0 since all below last_relaxed_mu=0.25).
        """
        n = 500
        rng = np.random.RandomState(0)
        arrays = {
            'mu_R':    rng.uniform(-0.001, 0.001, n).astype(np.float32),
            'mae':     np.full(n, 0.1, dtype=np.float32),
            'sigma':   np.full(n, 0.1, dtype=np.float32),
            'p_trade': np.full(n, 0.7, dtype=np.float32),
        }
        ref = {
            'mu_R':    np.full(n, 0.5, dtype=np.float32),
            'mae':     np.full(n, 0.01, dtype=np.float32),
            'sigma':   np.full(n, 0.01, dtype=np.float32),
            'p_trade': np.full(n, 0.99, dtype=np.float32),
        }
        cfg = V5QualityGateConfig(mu_R_min=0.05, quality_gate_strict_audit=False)
        mask, diag = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=ref)
        n_passed = int(np.sum(mask))
        assert n_passed < n * 0.25, (
            f"default fallback must preserve edge floor (got {n_passed}/{n} passed). "
            "If this fails, the T1 fix (last_relaxed_mu preserved in fallback) is broken."
        )

    def test_bug1_t4_raw_mu_r_used_for_gate_when_available(self):
        """T4: quality_mask uses '_raw_mu_R_ref' for gate percentile when mu_debias is active.

        Scenario: ref['mu_R'] is debiased (near-zero), but ref['_raw_mu_R_ref'] is the
        original large-magnitude values.  The gate must use raw values to set adaptive_mu_min
        to a meaningful threshold (not ~0 from the debiased values).

        Arrays have mu_R=0.10.  Raw ref=0.50 → adaptive_mu_min=0.50.  After 5 relax steps
        the minimum relaxed_mu = 0.50/2.0 = 0.25, still > 0.10, so all bars fail every step.
        With strict_audit=True the finiteness bypass is skipped → n_passed = 0.

        Debiased ref=0.001 → adaptive_mu_min≈0.001, bars immediately pass → n≈500.

        The function must behave like the raw case (n_raw < n_debiased).
        """
        n = 500
        arrays = {
            'mu_R':    np.full(n, 0.10, dtype=np.float32),  # 0.10 < 0.50/2.0=0.25: fails all relax steps
            'mae':     np.full(n, 0.1, dtype=np.float32),
            'sigma':   np.full(n, 0.1, dtype=np.float32),
            'p_trade': np.full(n, 0.7, dtype=np.float32),
        }
        ref_debiased = {
            'mu_R':          np.full(n, 0.001, dtype=np.float32),  # debiased → near zero
            '_raw_mu_R_ref': np.full(n, 0.50,  dtype=np.float32),  # raw → large
            'mae':     np.full(n, 0.01, dtype=np.float32),
            'sigma':   np.full(n, 0.01, dtype=np.float32),
            'p_trade': np.full(n, 0.99, dtype=np.float32),
        }
        # strict_audit=True: no finiteness bypass → only bars that pass the edge floor survive
        cfg = V5QualityGateConfig(mu_R_min=0.05, quality_gate_strict_audit=True)
        mask_with_raw, _ = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=ref_debiased)

        ref_no_raw = {k: v for k, v in ref_debiased.items() if k != '_raw_mu_R_ref'}
        ref_no_raw['mu_R'] = np.full(n, 0.001, dtype=np.float32)  # debiased only, no raw
        mask_debiased_only, _ = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=ref_no_raw)

        n_raw      = int(np.sum(mask_with_raw))
        n_debiased = int(np.sum(mask_debiased_only))

        # Raw ref (0.5): adaptive_mu_min=0.5, bars[mu_R=0.10] fail all 5 relax steps → n=0
        # Debiased ref (0.001): adaptive_mu_min≈0.001, bars[mu_R=0.10] all pass → n≈500
        assert n_raw < n_debiased, (
            f"T4: '_raw_mu_R_ref' must tighten the gate vs debiased ref. "
            f"raw={n_raw} debiased={n_debiased}. If equal, '_raw_mu_R_ref' is not being used."
        )
        assert n_raw < n * 0.25, (
            f"T4: when raw ref is 0.5 and arrays mu_R=0.10 (below all relaxed floors), "
            f"strict_audit=True must keep n_passed near 0. Got {n_raw}/{n}."
        )
        assert n_debiased > n * 0.75, (
            f"T4: debiased ref (mu_min≈0.001) should pass most bars with mu_R=0.10. "
            f"Got only {n_debiased}/{n}."
        )

    def test_bug2_score_threshold_default_matches_live(self):
        """Bug 2: V5ForwardTestConfig.score_threshold must match shared_v5_trade_config default."""
        from config.shared_v5_trade_config import V5TradeDefaults
        live_default = V5TradeDefaults().score_threshold
        backtest_default = V5ForwardTestConfig().score_threshold
        assert backtest_default == live_default, (
            f"score_threshold mismatch: backtest default={backtest_default}, "
            f"live default={live_default}. Backtest metrics will be optimistic."
        )

    def test_bug4_score_decile_table_monotonic(self):
        """Bug 4: _compute_score_decile_table returns monotonic=True on perfectly sorted data."""
        from train.v5_train import _compute_score_decile_table
        n = 200
        scores = np.linspace(0.0, 1.0, n)
        realized_r = scores * 2.0 - 0.5  # perfectly correlated with scores
        rows, monotonic = _compute_score_decile_table(scores, realized_r, n_deciles=5)
        assert len(rows) == 5
        assert monotonic is True, "Perfectly sorted scores should give monotonic=True"

    def test_bug4_score_decile_table_non_monotonic(self):
        """Bug 4: _compute_score_decile_table returns monotonic=False when top decile underperforms."""
        from train.v5_train import _compute_score_decile_table
        n = 200
        scores = np.linspace(0.0, 1.0, n)
        realized_r = np.full(n, 0.1)
        realized_r[int(n * 0.9):] = -0.5  # top decile is a loser
        rows, monotonic = _compute_score_decile_table(scores, realized_r, n_deciles=5)
        assert monotonic is False, "Top-decile underperformance should give monotonic=False"

    def test_bug4_score_decile_table_too_few_trades(self):
        """Bug 4: _compute_score_decile_table returns empty list when too few trades."""
        from train.v5_train import _compute_score_decile_table
        scores = np.array([0.1, 0.5, 0.9])
        r = np.array([0.1, 0.2, 0.3])
        rows, monotonic = _compute_score_decile_table(scores, r, n_deciles=10)
        assert rows == [] and monotonic is False

    def test_finding5_entry_lag_reduces_r(self):
        """Finding 5: entry_lag_atr_fraction > 0 should reduce avg realized R vs lag=0."""
        from data.common import generate_v5_sweep_outcomes
        import pandas as pd
        rng = np.random.RandomState(42)
        n = 500
        close = np.cumprod(1 + rng.randn(n) * 0.01) * 100
        df = pd.DataFrame({
            'open':  close,
            'high':  close * 1.01,
            'low':   close * 0.99,
            'close': close,
            'volume': np.ones(n) * 1000,
        })
        res_no_lag = generate_v5_sweep_outcomes(df, entry_lag_atr_fraction=0.0)
        res_with_lag = generate_v5_sweep_outcomes(df, entry_lag_atr_fraction=0.5)

        finite_no_lag = res_no_lag['r_long'][np.isfinite(res_no_lag['r_long'])]
        finite_with_lag = res_with_lag['r_long'][np.isfinite(res_with_lag['r_long'])]

        assert np.mean(finite_with_lag) < np.mean(finite_no_lag), (
            "entry_lag should reduce avg realized R for LONG trades "
            f"(no_lag={np.mean(finite_no_lag):.4f}, with_lag={np.mean(finite_with_lag):.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task #58 tests: loss rebalance + sigma threshold + two-phase curriculum
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    from train.v5_train import compute_v5_loss


def _make_dummy_v5_batch(n=32, seed=7, device='cpu'):
    """Create a minimal batch dict that compute_v5_loss accepts.

    Key names match V5Dataset.__getitem__ output:
      ret_R, mfe_R, mae_R (R-unit targets), action_label (long int), valid (bool).
    """
    import torch
    rng = torch.Generator()
    rng.manual_seed(seed)
    feat_n = 95
    batch = {
        'features': torch.randn(n, feat_n, generator=rng),
        'ret_R': torch.randn(n, generator=rng) * 0.02,          # R-unit realized return
        'mfe_R': torch.abs(torch.randn(n, generator=rng)) * 0.01,  # R-unit MFE
        'mae_R': torch.abs(torch.randn(n, generator=rng)) * 0.01,  # R-unit MAE
        'action_label': torch.randint(0, 3, (n,), generator=rng),  # 0=HOLD,1=LONG,2=SHORT
        'valid': torch.ones(n, dtype=torch.bool),
        'vol_h': torch.ones(n) * 0.02,  # not used by loss but often in batch
    }
    return batch


def _make_dummy_v5_outputs(n=32, sigma_val=0.607, seed=7, device='cpu'):
    """Create model output dict with a fixed sigma value."""
    import torch
    rng = torch.Generator()
    rng.manual_seed(seed)
    outputs = {
        'ret_mu': torch.randn(n, 1, generator=rng) * 0.01,
        'ret_log_sigma': torch.full((n, 1), float(torch.tensor(sigma_val).log())),
        'ret_sigma': torch.full((n, 1), sigma_val),
        'mfe': torch.abs(torch.randn(n, 1, generator=rng)) * 0.01,
        'mae': torch.abs(torch.randn(n, 1, generator=rng)) * 0.01,
        'action_logits': torch.randn(n, 3, generator=rng),
    }
    return outputs


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_sigma_penalty_fires_at_0_607():
    """Task #58 T1: sigma=0.607 should trigger penalty when sigma_reg_threshold=0.40.

    At the old threshold of 1.5, sigma=0.607 never triggered the penalty,
    meaning the model was free to use high uncertainty to collapse mu_R gradients.
    With threshold=0.40, penalty must be positive (> 1e-6).
    """
    import torch
    batch = _make_dummy_v5_batch()
    outputs = _make_dummy_v5_outputs(sigma_val=0.607)

    # threshold=0.40 — should fire
    loss_strict, ld_strict = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.1, sigma_reg_threshold=0.40,
        phase1_mode=False,
    )
    # threshold=1.5 (old default) — should NOT fire for sigma=0.607
    loss_old, ld_old = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.1, sigma_reg_threshold=1.5,
        phase1_mode=False,
    )
    sigma_pen_strict = ld_strict.get('L_sigma_reg', 0.0)
    sigma_pen_old = ld_old.get('L_sigma_reg', 0.0)

    assert sigma_pen_strict > 1e-6, (
        f"sigma penalty must fire at threshold=0.40 with sigma=0.607 "
        f"(got L_sigma_reg={sigma_pen_strict:.6f})"
    )
    assert sigma_pen_old < 1e-6, (
        f"sigma penalty must NOT fire at threshold=1.5 with sigma=0.607 "
        f"(got L_sigma_reg={sigma_pen_old:.6f})"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_loss_budget_L_ret_pct_above_50_percent():
    """Task #58 T2: With w_ret=6.0 / w_mfe=0.15 / w_mae=0.15, L_ret must be >50% of gradient.

    Previous weights (w_ret=3, w_mfe=1, w_mae=1) drove L_ret at ~2.4%.
    New weights must drive L_ret to ≥50% so the trunk learns return signal.
    """
    import torch
    batch = _make_dummy_v5_batch()
    outputs = _make_dummy_v5_outputs(sigma_val=0.40)

    _, ld = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
    )
    # Reconstruct gradient budget from loss components
    L_ret = ld.get('L_ret', 0.0) * 6.0
    L_mfe = ld.get('L_mfe', 0.0) * 0.15
    L_mae = ld.get('L_mae', 0.0) * 0.15
    L_act = ld.get('L_action', 0.0) * 2.5
    total = L_ret + L_mfe + L_mae + L_act + 1e-9
    ret_pct = L_ret / total

    assert ret_pct > 0.50, (
        f"L_ret must comprise >50% of gradient budget with new weights "
        f"(got {ret_pct*100:.1f}% | L_ret={L_ret:.4f} L_mfe={L_mfe:.4f} "
        f"L_mae={L_mae:.4f} L_act={L_act:.4f})"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_atr14_in_dataset_and_no_double_normalization():
    """Task #58 T4: mfe_R/mae_R are ALREADY R-units; atr14 must NOT divide targets again.

    Semantic invariant:
      - build_v5_targets returns mfe_R = price_delta / ATR (R-units already normalized).
      - compute_v5_loss must NOT divide by ATR again (would give price_delta / ATR^2).
      - The atr_normalize_risk_heads flag is preserved as an API stub but is a no-op
        in the loss function; the flag does not change L_mfe or L_mae.

    This test verifies:
      1. With atr14 in the batch, L_mfe is IDENTICAL to without atr14 (no double division).
      2. V5Dataset stores atr14 when provided (data plumbing test).
    """
    import torch

    # Loss should be identical whether or not atr14 is in the batch.
    batch_no_atr = _make_dummy_v5_batch()
    batch_with_atr = dict(batch_no_atr)
    batch_with_atr['atr14'] = torch.full((32,), 2.0)  # if double-dividing, targets halved → loss differs

    outputs = _make_dummy_v5_outputs(sigma_val=0.40)

    _, ld_no_atr = compute_v5_loss(
        outputs, batch_no_atr,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False, atr_normalize_risk_heads=True,
    )
    _, ld_with_atr = compute_v5_loss(
        outputs, batch_with_atr,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False, atr_normalize_risk_heads=True,
    )

    l_mfe_no_atr = ld_no_atr.get('L_mfe', 0.0)
    l_mfe_with_atr = ld_with_atr.get('L_mfe', 0.0)
    assert abs(l_mfe_no_atr - l_mfe_with_atr) < 1e-6, (
        f"ATR in batch must NOT change L_mfe (mfe_R is already R-units; no double division). "
        f"If they differ, ATR normalization is incorrectly applied a second time. "
        f"(no_atr={l_mfe_no_atr:.6f}, with_atr={l_mfe_with_atr:.6f})"
    )

    # V5Dataset data plumbing: atr14 stored and emitted when provided.
    import numpy as np
    n = 16
    rng = np.random.RandomState(42)
    dummy_feat = rng.randn(n, 95).astype(np.float32)
    dummy_arr = rng.randn(n).astype(np.float32)
    dummy_act = np.zeros(n, dtype=np.int64)
    dummy_valid = np.ones(n, dtype=bool)
    dummy_atr14 = np.full(n, 1.5, dtype=np.float32)

    if HAS_TORCH:
        from train.v5_train import V5Dataset
        ds_with = V5Dataset(dummy_feat, dummy_arr, abs(dummy_arr), abs(dummy_arr),
                            abs(dummy_arr), dummy_act, dummy_valid, atr14=dummy_atr14)
        ds_without = V5Dataset(dummy_feat, dummy_arr, abs(dummy_arr), abs(dummy_arr),
                               abs(dummy_arr), dummy_act, dummy_valid)
        assert 'atr14' in ds_with[0], "V5Dataset must emit 'atr14' key when atr14 is provided"
        assert 'atr14' not in ds_without[0], "V5Dataset must NOT emit 'atr14' key when atr14 is None"
        assert abs(ds_with[0]['atr14'].item() - 1.5) < 1e-6, "V5Dataset atr14 value must match input"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_phase1_mode_skips_mfe_mae_action():
    """Task #58 T3: phase1_mode=True should zero out MFE, MAE, and action losses.

    Phase 1 is a curriculum phase where only the return head is trained.
    Gradient from MFE/MAE/action during Phase 1 pushes the trunk toward
    risk-head local minima before mu_R has a meaningful gradient signal.
    Verified by comparing ld['L_mfe'], ld['L_mae'], ld['L_action'] to zero.
    """
    import torch
    batch = _make_dummy_v5_batch()
    outputs = _make_dummy_v5_outputs(sigma_val=0.40)

    _, ld_p1 = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=True,
    )
    _, ld_p2 = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
    )

    # Phase 1: MFE, MAE, action must be absent or zero
    for key in ('L_mfe', 'L_mae', 'L_action'):
        p1_val = ld_p1.get(key, 0.0)
        p2_val = ld_p2.get(key, 0.0)
        assert abs(p1_val) < 1e-8, (
            f"phase1_mode=True must zero out {key} "
            f"(got {p1_val:.6f}; Phase2 has {p2_val:.4f})"
        )

    # Phase 2: at least action and return must be non-trivial
    assert ld_p2.get('L_ret', 0.0) > 1e-6 or ld_p2.get('L_action', 0.0) > 1e-6, (
        "Phase 2 must have non-zero L_ret or L_action "
        f"(L_ret={ld_p2.get('L_ret'):.4f} L_action={ld_p2.get('L_action'):.4f})"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_chop_hold_target_changes_kl_loss():
    """Task #56 B1: chop_hold_target changes the KL side-balance loss.

    When chop_hold_target=0.20 (new default), the chop KL target is [0.20, 0.40, 0.40].
    When chop_hold_target=0.35 (old hardcoded), the target is [0.35, 0.325, 0.325].
    A model with uniform action predictions should get different L_action values
    since the KL divergence from the chop target differs.
    """
    import torch

    batch = _make_dummy_v5_batch()
    # Force all bars into chop regime (ret_R near zero)
    batch['ret_R'] = torch.zeros_like(batch['ret_R'])
    outputs = _make_dummy_v5_outputs(sigma_val=0.40)

    _, ld_new = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
        chop_hold_target=0.20,  # new default: more aggressive L/S push
        side_bal_weight=0.05,
    )
    _, ld_old = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
        chop_hold_target=0.35,  # old hardcoded value
        side_bal_weight=0.05,
    )

    l_action_new = ld_new.get('L_action', 0.0)
    l_action_old = ld_old.get('L_action', 0.0)
    assert abs(l_action_new - l_action_old) > 1e-8, (
        f"chop_hold_target 0.20 vs 0.35 must give different L_action when all bars are chop-regime. "
        f"new={l_action_new:.6f} old={l_action_old:.6f} — chop_hold_target not wired into KL targets?"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_ret_mag_ce_weight_changes_action_loss():
    """Task #56 B2: ret_mag_ce_weight upweights high-return bars in CE loss.

    With a batch containing a mix of high-return and near-zero-return bars,
    enabling ret_mag_ce_weight=True must produce a different L_action than False.
    The loss scale is preserved (normalized weights), but gradient distribution changes.
    """
    import torch
    import numpy as np

    batch = _make_dummy_v5_batch()
    # Mix: half bars with high ret_R (bull/bear), half near-zero (chop)
    n = batch['ret_R'].shape[0]
    half = n // 2
    ret_mixed = torch.zeros(n, 1)
    ret_mixed[:half] = 0.5   # high return
    ret_mixed[half:] = 0.001  # near-zero chop
    batch['ret_R'] = ret_mixed

    outputs = _make_dummy_v5_outputs(sigma_val=0.40)

    _, ld_off = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
        ret_mag_ce_weight=False,
        ret_mag_scale=2.0,
    )
    _, ld_on = compute_v5_loss(
        outputs, batch,
        w_ret=6.0, w_mfe=0.15, w_mae=0.15, w_action=2.5,
        sigma_spread_reg=0.0, sigma_reg_threshold=0.40,
        phase1_mode=False,
        ret_mag_ce_weight=True,
        ret_mag_scale=2.0,
    )

    l_action_off = ld_off.get('L_action', 0.0)
    l_action_on  = ld_on.get('L_action', 0.0)
    assert abs(l_action_on - l_action_off) > 1e-8, (
        f"ret_mag_ce_weight=True must change L_action when batch has mixed ret_R magnitudes. "
        f"off={l_action_off:.6f} on={l_action_on:.6f} — upweighting not applied?"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_score_lambda_0_30_via_compute_v5_scores():
    """Task #56 A1: compute_v5_scores with score_lambda=0.30 vs 0.50.

    At p_side=0.31 (above break-even 0.231 for lambda=0.30, below 0.333 for lambda=0.50):
    - score_lambda=0.30 should produce a POSITIVE score
    - score_lambda=0.50 should produce a LOWER (or negative) score
    Uses _arrays interface to call compute_v5_scores directly without torch model outputs.
    """
    import numpy as np
    from train.v5_train import compute_v5_scores

    n = 50
    # p_long=0.31, p_short=0.04, p_hold=0.65 — long signal with low p_long
    arrays = {
        'mu_R':  np.full(n, 0.50, dtype=np.float32),  # positive expected return
        'mae':   np.full(n, 0.50, dtype=np.float32),  # risk = mae (capped)
        'mfe':   np.full(n, 1.00, dtype=np.float32),
        'p_long':  np.full(n, 0.31, dtype=np.float32),
        'p_short': np.full(n, 0.04, dtype=np.float32),
    }

    # Call with lower lambda
    result_030 = compute_v5_scores(
        None, score_lambda=0.30, _arrays=arrays, min_mu_r_score=0.0
    )
    # Call with higher lambda
    result_050 = compute_v5_scores(
        None, score_lambda=0.50, _arrays=arrays, min_mu_r_score=0.0
    )

    score_030 = float(np.mean(result_030['score_long']))
    score_050 = float(np.mean(result_050['score_long']))

    assert score_030 > score_050, (
        f"score_lambda=0.30 must give HIGHER score than 0.50 at p_side=0.31. "
        f"score_030={score_030:.5f} score_050={score_050:.5f}"
    )
    assert score_030 > 0, (
        f"score_lambda=0.30 must give POSITIVE score at p_long=0.31 "
        f"(break-even=0.231). Got {score_030:.5f}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
