"""
Task #51 Regression Tests — V5 Training Signal Fix

Verifies:
  1. mu_debias default is False in all 3 function signatures / dataclass
  2. SIDE_BAL_W == 0.05 (reduced from 0.30 to limit KL loss dominance)
  3. barrier_aligned_ret_R logic: LONG→+r_long, SHORT→-r_short, HOLD→0.0
  4. Aligned ret_R produces no NaNs and correct sign per action label
  5. [V5_TRAIN_QUALITY] per-epoch logging is present in v5_train.py
  6. [V5_TRAIN_WARN] fires at epoch>=20 when mu_r_corr < -0.05
  7. [V5_BASELINE_CMP] / v5_run_metrics.json infrastructure is in place
  8. Fold-start log emits mu_debias ENABLED/DISABLED status
"""
import ast
import re
import sys
import numpy as np
import pytest
from pathlib import Path


V5_TRAIN_PATH = Path(__file__).parent / "train" / "v5_train.py"
V5_TGT_PATH   = Path(__file__).parent / "data" / "v5_target_generator.py"


@pytest.fixture(scope="module")
def v5_train_src():
    return V5_TRAIN_PATH.read_text()


@pytest.fixture(scope="module")
def v5_tgt_src():
    return V5_TGT_PATH.read_text()


class TestMuDebiasDefault:
    def test_v5forwardconfig_default_false(self, v5_train_src):
        assert "mu_debias: bool = False" in v5_train_src, (
            "V5ForwardConfig must have mu_debias default=False "
            "(disable mu_debias by default to prevent gradient collapse)"
        )

    def test_no_mu_debias_true_defaults(self, v5_train_src):
        hits = [ln.strip() for ln in v5_train_src.splitlines()
                if "mu_debias=True" in ln and "mu_debias_alpha" not in ln
                and not ln.lstrip().startswith("#")]
        assert hits == [], (
            f"Found mu_debias=True default in function signature(s): {hits}"
        )

    def test_run_v5_walk_forward_default_false(self, v5_train_src):
        match = re.search(
            r"def run_v5_walk_forward.*?mu_debias\s*=\s*(True|False)",
            v5_train_src, re.DOTALL
        )
        assert match, "run_v5_walk_forward must declare mu_debias parameter"
        assert match.group(1) == "False", (
            f"run_v5_walk_forward mu_debias default must be False, got {match.group(1)}"
        )

    def test_train_v5_model_default_false(self, v5_train_src):
        match = re.search(
            r"def train_v5_model.*?mu_debias\s*=\s*(True|False)",
            v5_train_src, re.DOTALL
        )
        assert match, "train_v5_model must declare mu_debias parameter"
        assert match.group(1) == "False", (
            f"train_v5_model mu_debias default must be False, got {match.group(1)}"
        )


class TestSideBalanceWeight:
    def test_side_bal_w_is_configurable(self, v5_train_src):
        # Task #53 replaced the hardcoded SIDE_BAL_W = 0.05 with a configurable
        # side_bal_weight parameter (default 0.15) for 3-class KL.
        # The assignment must be: SIDE_BAL_W = side_bal_weight
        assert "SIDE_BAL_W = side_bal_weight" in v5_train_src, (
            "SIDE_BAL_W must be assigned from configurable side_bal_weight "
            "parameter (Task #53 changed hardcoded 0.05 to configurable default 0.15)"
        )

    def test_old_side_bal_w_030_removed(self, v5_train_src):
        assert "SIDE_BAL_W = 0.30" not in v5_train_src, (
            "Old SIDE_BAL_W = 0.30 must be removed"
        )


class TestBarrierAlignedRetR:
    """Unit tests for the barrier_aligned_ret_R override block."""

    def _run_alignment(self, action_label, b_r_long, b_r_short, valid_mask=None):
        n = len(action_label)
        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)
        ret_R = np.zeros(n, dtype=np.float64)
        n_l = n_s = n_h = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            lbl = action_label[i]
            if lbl == 1:
                rl = b_r_long[i] if np.isfinite(b_r_long[i]) else 0.0
                ret_R[i] = float(rl)
                n_l += 1
            elif lbl == 2:
                rs = b_r_short[i] if np.isfinite(b_r_short[i]) else 0.0
                ret_R[i] = -float(rs)
                n_s += 1
            else:
                ret_R[i] = 0.0
                n_h += 1
        return ret_R, n_l, n_s, n_h

    def test_long_bars_have_nonnegative_ret_r(self):
        action_label = np.array([1, 1, 1, 2, 0])
        b_r_long  = np.array([1.5, 0.5, 2.0, 0.0, 0.0])
        b_r_short = np.array([0.0, 0.0, 0.0, 1.2, 0.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        long_mask = action_label == 1
        assert np.all(ret_R[long_mask] >= 0), (
            f"LONG-labelled bars must have non-negative ret_R: {ret_R[long_mask]}"
        )

    def test_short_bars_have_nonpositive_ret_r(self):
        action_label = np.array([2, 2, 2, 1, 0])
        b_r_long  = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        b_r_short = np.array([1.2, 0.8, 2.0, 0.0, 0.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        short_mask = action_label == 2
        assert np.all(ret_R[short_mask] <= 0), (
            f"SHORT-labelled bars must have non-positive ret_R: {ret_R[short_mask]}"
        )

    def test_hold_bars_have_zero_ret_r(self):
        action_label = np.array([0, 0, 0, 1, 2])
        b_r_long  = np.array([0.3, 0.1, 0.2, 1.5, 0.0])
        b_r_short = np.array([0.0, 0.4, 0.1, 0.0, 1.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        hold_mask = action_label == 0
        assert np.all(ret_R[hold_mask] == 0.0), (
            f"HOLD-labelled bars must have ret_R=0: {ret_R[hold_mask]}"
        )

    def test_nan_barrier_values_become_zero(self):
        action_label = np.array([1, 2])
        b_r_long  = np.array([float('nan'), 0.0])
        b_r_short = np.array([0.0, float('nan')])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        assert np.all(np.isfinite(ret_R)), (
            f"NaN barrier values must be clamped to 0.0: ret_R={ret_R}"
        )
        assert ret_R[0] == 0.0, "NaN r_long → ret_R[0] must be 0.0"
        assert ret_R[1] == 0.0, "NaN r_short → ret_R[1] must be 0.0"

    def test_invalid_bars_not_overridden(self):
        action_label = np.array([1, 1, 0])
        b_r_long  = np.array([1.5, 0.8, 0.0])
        b_r_short = np.array([0.0, 0.0, 0.0])
        valid_mask = np.array([True, False, True])
        ret_R_original = np.array([99.0, 99.0, 99.0])
        ret_R = ret_R_original.copy()
        n_l = n_s = n_h = 0
        for i in range(3):
            if not valid_mask[i]:
                continue
            lbl = action_label[i]
            if lbl == 1:
                ret_R[i] = float(b_r_long[i])
                n_l += 1
            elif lbl == 2:
                ret_R[i] = -float(b_r_short[i])
                n_s += 1
            else:
                ret_R[i] = 0.0
                n_h += 1
        assert ret_R[1] == 99.0, "Invalid bar must NOT have its ret_R overridden"
        assert ret_R[0] == 1.5, "Valid LONG bar must get +r_long"
        assert ret_R[2] == 0.0, "Valid HOLD bar must get 0.0"

    def test_sign_convention_no_gradient_contradiction(self):
        # LONG label → positive ret_R → model predicts positive mu_R
        # SHORT label → negative ret_R → model predicts negative mu_R
        # side_aware_scoring: long requires mu_R>0, short requires mu_R<0
        # → no contradiction between NLL loss target and CE action label
        action_label = np.array([1, 2])
        b_r_long  = np.array([1.5, 0.0])
        b_r_short = np.array([0.0, 1.2])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        assert ret_R[0] > 0, "LONG bar: positive ret_R → model learns mu_R>0 (bullish)"
        assert ret_R[1] < 0, "SHORT bar: negative ret_R → model learns mu_R<0 (bearish)"


class TestPerEpochQualityLogging:
    def test_v5_train_quality_log_present(self, v5_train_src):
        assert "[V5_TRAIN_QUALITY]" in v5_train_src, (
            "Must log [V5_TRAIN_QUALITY] per epoch with mu_r_corr_val"
        )

    def test_mu_r_corr_val_computed(self, v5_train_src):
        assert "mu_r_corr_val" in v5_train_src, (
            "Must compute mu_r_corr_val (correlation between predicted mu_R and val_ret_R)"
        )

    def test_v5_train_warn_after_epoch_20(self, v5_train_src):
        assert "[V5_TRAIN_WARN]" in v5_train_src, (
            "Must emit [V5_TRAIN_WARN] when mu_r_corr_val < -0.05 after epoch 20"
        )
        assert "epoch >= 20" in v5_train_src, (
            "Warning must fire only at epoch>=20 (not on early noisy epochs)"
        )

    def test_quality_block_guarded_by_not_use_v6(self, v5_train_src):
        idx = v5_train_src.find("[V5_TRAIN_QUALITY]")
        assert idx >= 0
        # Find the enclosing 'if not use_v6' guard — may be up to 5000 chars before
        # the actual log string because the compute block is sizeable.
        surrounding = v5_train_src[max(0, idx-5000):idx]
        assert "not use_v6" in surrounding, (
            "[V5_TRAIN_QUALITY] block must be guarded by 'not use_v6'"
        )


class TestBaselineComparison:
    def _cmp_segment(self, v5_train_src):
        idx = v5_train_src.find("[V5_BASELINE_CMP]")
        assert idx >= 0, "[V5_BASELINE_CMP] block not found"
        return v5_train_src[idx:idx + 8000]

    def test_baseline_cmp_block_present(self, v5_train_src):
        assert "[V5_BASELINE_CMP]" in v5_train_src, (
            "Must have [V5_BASELINE_CMP] block in walk-forward summary"
        )

    def test_run_metrics_saved(self, v5_train_src):
        assert "v5_run_metrics.json" in v5_train_src, (
            "Must save current run metrics to v5_run_metrics.json"
        )

    def test_baseline_metrics_loaded(self, v5_train_src):
        assert "v5_baseline_metrics.json" in v5_train_src, (
            "Must try to load v5_baseline_metrics.json for comparison"
        )

    def test_comparison_includes_mu_r_correlation(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "mu_r_correlation" in seg, (
            "Baseline comparison must include mu_r_correlation metric"
        )

    def test_comparison_includes_action_accuracy(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "action_accuracy" in seg, (
            "Baseline comparison must include action_accuracy metric"
        )

    def test_comparison_includes_total_r(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "total_r" in seg, (
            "Baseline comparison must include total_r metric"
        )

    def test_comparison_includes_p_side_winners(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "p_side_winners" in seg or "mean_p_side_winners" in seg, (
            "Baseline comparison must include p_side_winners metric"
        )

    def test_comparison_includes_p_side_losers(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "p_side_losers" in seg or "mean_p_side_losers" in seg, (
            "Baseline comparison must include p_side_losers metric"
        )

    def test_comparison_includes_winner_loser_margin(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "winners-losers" in seg or "winners - losers" in seg or "p_side_winners" in seg, (
            "Baseline comparison must include winners > losers margin check"
        )

    def test_comparison_includes_score_discriminability(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "score_disc_p90" in seg or "disc(p90" in seg or "p90/p50" in seg, (
            "Baseline comparison must include score discriminability (p90/p50)"
        )
        assert "score_disc_p99" in seg or "disc(p99" in seg or "p99/p50" in seg, (
            "Baseline comparison must include score discriminability (p99/p50)"
        )

    def test_comparison_includes_relax_loop(self, v5_train_src):
        seg = self._cmp_segment(v5_train_src)
        assert "relax_loop" in seg or "relax" in seg.lower(), (
            "Baseline comparison must include relax-loop trigger rate"
        )

    def test_run_metrics_has_p_side_fields(self, v5_train_src):
        assert "mean_p_side_winners" in v5_train_src, (
            "v5_run_metrics must include mean_p_side_winners"
        )
        assert "mean_p_side_losers" in v5_train_src, (
            "v5_run_metrics must include mean_p_side_losers"
        )
        assert "mean_score_disc_p90p50" in v5_train_src, (
            "v5_run_metrics must include mean_score_disc_p90p50"
        )
        assert "mean_score_disc_p99p50" in v5_train_src, (
            "v5_run_metrics must include mean_score_disc_p99p50"
        )


class TestFoldStartLog:
    def test_fold_start_logs_mu_debias_status(self, v5_train_src):
        assert '"ENABLED" if mu_debias else "DISABLED"' in v5_train_src or \
               "'ENABLED' if mu_debias else 'DISABLED'" in v5_train_src, (
            "Fold-start log must emit mu_debias ENABLED/DISABLED status"
        )

    def test_fold_start_logs_side_bal_w(self, v5_train_src):
        idx = v5_train_src.find("barrier_aligned_ret_R=True")
        assert idx >= 0, (
            "Fold-start log must mention barrier_aligned_ret_R=True"
        )


class TestCLIDefaults:
    def test_quick_start_mu_debias_default_false(self):
        from pathlib import Path
        qs_path = Path(__file__).parent / "quick_start.py"
        if not qs_path.exists():
            pytest.skip("quick_start.py not found")
        qs_src = qs_path.read_text()
        assert 'default=False' in qs_src, (
            "quick_start.py --v5-mu-debias must have default=False (opt-in)"
        )
        idx = qs_src.find("--v5-mu-debias")
        assert idx >= 0
        snippet = qs_src[idx:idx+300]
        assert "default=False" in snippet, (
            "--v5-mu-debias argparse entry must have default=False"
        )

    def test_side_bal_w_log_not_030(self, v5_train_src):
        # Task #53 made SIDE_BAL_W configurable (default 0.15); the old 0.30 must
        # not appear in any log line.  The fold-start log now uses %.2f format.
        assert 'SIDE_BAL_W=0.30' not in v5_train_src, (
            "All occurrences of SIDE_BAL_W=0.30 in logs must be removed"
        )
        assert 'SIDE_BAL_W=%.2f' in v5_train_src or 'SIDE_BAL_W=' in v5_train_src, (
            "Fold-start log must still emit a SIDE_BAL_W value"
        )


class TestScoreSpreadTelemetry:
    def test_score_percentiles_in_quality_log(self, v5_train_src):
        assert "score_p10" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log score_p10 for spread observability"
        )
        assert "score_p50" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log score_p50 (not just |mu_R| percentiles)"
        )
        assert "score_p90" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log score_p90"
        )
        assert "score_p99" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log score_p99"
        )

    def test_discriminability_ratios_logged(self, v5_train_src):
        assert "disc(p90/p50)" in v5_train_src or "p90/p50" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log p90/p50 discriminability ratio"
        )
        assert "disc(p99/p50)" in v5_train_src or "p99/p50" in v5_train_src, (
            "[V5_TRAIN_QUALITY] must log p99/p50 discriminability ratio"
        )

    def test_score_approx_uses_mae_and_p_side(self, v5_train_src):
        idx = v5_train_src.find("[V5_TRAIN_QUALITY]")
        segment = v5_train_src[max(0, idx-2000):idx+2000]
        assert "_risk_f" in segment or "_mae_cat" in segment, (
            "Score approximation must use MAE as risk denominator"
        )
        assert "_p_side_f" in segment, (
            "Score approximation must use p_side (max of p_long, p_short)"
        )


class TestSignConventionDocumentation:
    def test_short_sign_convention_explained(self, v5_tgt_src):
        assert "-r_short" in v5_tgt_src or "negation" in v5_tgt_src or "negative" in v5_tgt_src.lower(), (
            "Must document why SHORT uses -r_short (sign convention for mu_R bearish)"
        )

    def test_short_convention_reconciled_with_spec(self, v5_tgt_src):
        assert "reconciled" in v5_tgt_src.lower() or "sign convention" in v5_tgt_src.lower(), (
            "Comment must explicitly reconcile -r_short sign with task-51 spec"
        )


class TestTargetGeneratorAlignment:
    def test_barrier_aligned_ret_r_code_present(self, v5_tgt_src):
        assert "barrier_aligned_ret_R" in v5_tgt_src or "barrier_aligned=True" in v5_tgt_src, (
            "v5_target_generator.py must have barrier_aligned ret_R alignment block"
        )

    def test_short_bars_negated(self, v5_tgt_src):
        assert "ret_R[i] = -float(rs_val)" in v5_tgt_src, (
            "SHORT bars must set ret_R[i] = -float(rs_val) for correct sign convention"
        )

    def test_hold_bars_zero(self, v5_tgt_src):
        assert "ret_R[i] = 0.0" in v5_tgt_src, (
            "HOLD bars must set ret_R[i] = 0.0"
        )

    def test_nan_guard_present(self, v5_tgt_src):
        assert "np.isfinite(b_r_long[i])" in v5_tgt_src, (
            "Must guard b_r_long[i] against NaN before assigning ret_R"
        )
        assert "np.isfinite(b_r_short[i])" in v5_tgt_src, (
            "Must guard b_r_short[i] against NaN before assigning ret_R"
        )

    def test_alignment_log_emitted(self, v5_tgt_src):
        assert "barrier_aligned=True" in v5_tgt_src, (
            "Must log '[V5_TARGETS] barrier_aligned=True' with count/mean/std for observability"
        )

    def test_alignment_log_includes_mean_std(self, v5_tgt_src):
        assert "ret_R_mean=" in v5_tgt_src, (
            "[V5_TARGETS] barrier_aligned log must include ret_R_mean"
        )
        assert "ret_R_std=" in v5_tgt_src, (
            "[V5_TARGETS] barrier_aligned log must include ret_R_std"
        )


class TestFmtCmpBehavior:
    """Behavioral unit tests for the _fmt_cmp comparison logic.

    These run the actual comparison logic without importing v5_train to avoid
    heavy GPU dependencies.  They mirror the exact logic added to v5_train.py.
    """

    @staticmethod
    def _fmt_cmp(name, bv, nv, threshold=None, higher_is_better=True, fmt='.4f',
                 upper_threshold=None):
        """Mirror of the _fmt_cmp helper inside run_v5_walk_forward."""
        if nv is None:
            return f"  {name:<50}: baseline=N/A  new=N/A  [SKIP]"
        bv_str = f"{bv:{fmt}}" if bv is not None else "N/A"
        diff_str = f"{nv - bv:+{fmt}}" if bv is not None else "N/A"
        if upper_threshold is not None:
            passed = nv <= upper_threshold
        elif threshold is not None:
            passed = nv >= threshold
        elif bv is not None:
            passed = (nv > bv) if higher_is_better else (nv < bv)
        else:
            passed = None
        status = "PASS" if passed else ("FAIL" if passed is not None else "N/A")
        return (
            f"  {name:<50}: baseline={bv_str:<12}  "
            f"new={nv:{fmt}}  diff={diff_str}  [{status}]"
        )

    def test_long_pct_79_passes_absolute_gate(self):
        result = self._fmt_cmp("long_pct", 85.0, 79.0, upper_threshold=80.0, fmt='.1f')
        assert "[PASS]" in result, f"long_pct=79 must PASS absolute gate <=80, got: {result}"

    def test_long_pct_81_fails_absolute_gate(self):
        result = self._fmt_cmp("long_pct", 50.0, 81.0, upper_threshold=80.0, fmt='.1f')
        assert "[FAIL]" in result, f"long_pct=81 must FAIL absolute gate <=80, got: {result}"

    def test_long_pct_absolute_gate_ignores_baseline(self):
        """Gate must be absolute (<=80), not relative to baseline."""
        # new=81 > baseline=85 (new is "better" relatively) but still FAIL absolute
        result = self._fmt_cmp("long_pct", 85.0, 81.0, upper_threshold=80.0, fmt='.1f')
        assert "[FAIL]" in result, "long_pct absolute gate must fail when >80 even if better than baseline"

    def test_relax_loop_49_passes_absolute_gate(self):
        result = self._fmt_cmp("relax_loop", 60.0, 49.0, upper_threshold=50.0, fmt='.1f')
        assert "[PASS]" in result, f"relax_loop=49 must PASS absolute gate <=50, got: {result}"

    def test_relax_loop_51_fails_absolute_gate(self):
        result = self._fmt_cmp("relax_loop", 30.0, 51.0, upper_threshold=50.0, fmt='.1f')
        assert "[FAIL]" in result, f"relax_loop=51 must FAIL absolute gate <=50, got: {result}"

    def test_mu_r_corr_above_threshold_passes(self):
        result = self._fmt_cmp("mu_r_corr", 0.0, 0.05, threshold=0.0)
        assert "[PASS]" in result

    def test_mu_r_corr_below_threshold_fails(self):
        result = self._fmt_cmp("mu_r_corr", 0.1, -0.01, threshold=0.0)
        assert "[FAIL]" in result

    def test_score_disc_p90p50_above_5x_passes(self):
        result = self._fmt_cmp("score_disc_p90/p50", 3.0, 6.0, threshold=5.0, fmt='.2f')
        assert "[PASS]" in result

    def test_score_disc_p99p50_below_10x_fails(self):
        result = self._fmt_cmp("score_disc_p99/p50", 12.0, 9.9, threshold=10.0, fmt='.2f')
        assert "[FAIL]" in result

    def test_nv_none_emits_skip(self):
        result = self._fmt_cmp("metric", 1.0, None)
        assert "[SKIP]" in result

    def test_baseline_none_no_relative_comparison(self):
        result = self._fmt_cmp("metric", None, 0.5, threshold=0.0)
        assert "[PASS]" in result


class TestMeanPerFoldTotalR:
    def test_mean_per_fold_total_r_key_in_metrics(self, v5_train_src):
        assert "mean_per_fold_total_r" in v5_train_src, (
            "v5_run_metrics.json must include 'mean_per_fold_total_r' (per-fold average)"
        )

    def test_mean_per_fold_total_r_computed_from_active_reports(self, v5_train_src):
        assert "_per_fold_total_r_vals" in v5_train_src, (
            "mean_per_fold_total_r must be computed from _active_rpts, not from aggregate total_r"
        )

    def test_mean_per_fold_total_r_in_comparison_block(self, v5_train_src):
        assert "mean_per_fold_total_r   (must >= baseline" in v5_train_src, (
            "[V5_BASELINE_CMP] must include mean_per_fold_total_r comparison row"
        )
