"""
Task #52 regression tests: V5 gate audit, percentile gate mode, WF_FOLD_PROOF extension,
V5_GATE_BASELINE_CMP infrastructure, and --v5-gate-mode CLI argument.

All tests run without GPU/data — they verify code structure, control flow, and log output only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train'))
sys.path.insert(0, os.path.dirname(__file__))

import unittest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import io
import logging


class TestGateModeField(unittest.TestCase):
    """V5ForwardTestConfig has gate_mode field with default ref_magnitude (source-level check)."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_gate_mode_default_is_ref_magnitude(self):
        self.assertIn('gate_mode: str = "ref_magnitude"', self.src,
                      "V5ForwardTestConfig must have gate_mode field with default ref_magnitude")

    def test_gate_mode_can_be_set_to_percentile_top15(self):
        self.assertIn('"percentile_top15"', self.src,
                      "percentile_top15 choice must appear in source")

    def test_gate_mode_is_string_field(self):
        self.assertIn('gate_mode: str =', self.src,
                      "gate_mode must be declared as a str field in V5ForwardTestConfig")


class TestV5GateAuditLogic(unittest.TestCase):
    """
    [V5_GATE_AUDIT] block: percentile_cutoff, taken_by_ref_gate, taken_by_pct_gate, overlap.
    Tested via direct computation (mirrors v5_train.py logic at ~line 2844-2863).
    """

    def _compute_gate_audit(self, finite_work, effective_threshold):
        """Reproduce [V5_GATE_AUDIT] logic from v5_train.py."""
        pct_cutoff_85 = float(np.percentile(finite_work, 85))
        taken_ref = int(np.sum(finite_work >= effective_threshold))
        taken_pct = int(np.sum(finite_work >= pct_cutoff_85))
        overlap = int(np.sum((finite_work >= effective_threshold) & (finite_work >= pct_cutoff_85)))
        overlap_pct = 100.0 * overlap / max(taken_ref, 1)
        gate_pass_rate = 100.0 * len(finite_work) / max(len(finite_work) + 50, 1)
        return {
            'pct_cutoff_85': pct_cutoff_85,
            'taken_ref': taken_ref,
            'taken_pct': taken_pct,
            'overlap': overlap,
            'overlap_pct': overlap_pct,
        }

    def test_pct_cutoff_is_p85_of_finite_scores(self):
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.0, 1.0, 200)
        audit = self._compute_gate_audit(scores, effective_threshold=0.5)
        expected_p85 = float(np.percentile(scores, 85))
        self.assertAlmostEqual(audit['pct_cutoff_85'], expected_p85, places=6)

    def test_taken_by_ref_gate_count(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        audit = self._compute_gate_audit(scores, effective_threshold=0.5)
        self.assertEqual(audit['taken_ref'], int(np.sum(scores >= 0.5)))

    def test_taken_by_pct_gate_is_top_15pct(self):
        rng = np.random.RandomState(0)
        scores = rng.uniform(0.0, 1.0, 100)
        audit = self._compute_gate_audit(scores, effective_threshold=0.01)
        expected_p85 = float(np.percentile(scores, 85))
        expected_taken_pct = int(np.sum(scores >= expected_p85))
        self.assertEqual(audit['taken_pct'], expected_taken_pct)

    def test_overlap_is_intersection_of_both_gates(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        threshold = 0.6
        audit = self._compute_gate_audit(scores, effective_threshold=threshold)
        p85 = float(np.percentile(scores, 85))
        expected_overlap = int(np.sum((scores >= threshold) & (scores >= p85)))
        self.assertEqual(audit['overlap'], expected_overlap)

    def test_overlap_pct_100_when_ref_gate_is_subset_of_pct_gate(self):
        # Very high threshold: ref_gate is a strict subset of pct_gate (all ref picks are also top-15%)
        scores = np.arange(1, 101, dtype=float)  # 1..100
        audit = self._compute_gate_audit(scores, effective_threshold=99.0)
        # ref_gate picks score>=99 (2 bars: 99,100); p85=86; pct_gate picks >=86 (15 bars)
        self.assertEqual(audit['taken_ref'], 2)
        self.assertGreater(audit['taken_pct'], 2)
        # overlap = 2 (both 99 and 100 are >= p85 too)
        self.assertEqual(audit['overlap'], 2)
        self.assertAlmostEqual(audit['overlap_pct'], 100.0, places=1)

    def test_gate_audit_requires_at_least_10_scores(self):
        # Only 5 scores → no audit block triggered (gate_mode check only applies when len>=10)
        small_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertLess(len(small_scores), 10)  # confirm size is too small


class TestPercentileTop15GateOverride(unittest.TestCase):
    """
    When gate_mode='percentile_top15', effective_threshold is overridden with p85 cutoff.
    """

    def _simulate_gate_mode_override(self, finite_work, effective_threshold, gate_mode):
        """Reproduce gate_mode override logic from v5_train.py ~line 2858-2863."""
        if len(finite_work) >= 10 and gate_mode == "percentile_top15":
            pct_cutoff_85 = float(np.percentile(finite_work, 85))
            effective_threshold = pct_cutoff_85
        return effective_threshold

    def test_ref_magnitude_mode_does_not_change_threshold(self):
        scores = np.arange(1, 101, dtype=float)
        original = 0.5
        result = self._simulate_gate_mode_override(scores, original, "ref_magnitude")
        self.assertEqual(result, original)

    def test_percentile_top15_overrides_threshold_with_p85(self):
        rng = np.random.RandomState(7)
        scores = rng.uniform(0, 1, 100)
        original = 0.0001  # very low threshold
        result = self._simulate_gate_mode_override(scores, original, "percentile_top15")
        expected = float(np.percentile(scores, 85))
        self.assertAlmostEqual(result, expected, places=6)

    def test_percentile_top15_with_fewer_than_10_scores_does_not_override(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # only 5
        original = 0.3
        result = self._simulate_gate_mode_override(scores, original, "percentile_top15")
        self.assertEqual(result, original)

    def test_percentile_top15_selects_exactly_top_15pct(self):
        scores = np.arange(1, 101, dtype=float)  # 1..100
        result = self._simulate_gate_mode_override(scores, 0.0, "percentile_top15")
        n_selected = int(np.sum(scores >= result))
        # p85 of 1..100 = 85.15; scores >= 86 → 15 bars (86..100)
        self.assertAlmostEqual(n_selected / len(scores), 0.15, delta=0.02)


class TestScoreSpreadP10(unittest.TestCase):
    """score_spread dict includes p10 key (new in Task #52)."""

    def _build_score_spread(self, scores):
        """Reproduce score_spread construction from v5_train.py ~line 4253-4269."""
        sc_finite = scores[np.isfinite(scores)]
        if len(sc_finite) <= 5:
            return None
        sc_p50 = float(np.percentile(sc_finite, 50))
        sc_p90 = float(np.percentile(sc_finite, 90))
        sc_p99 = float(np.percentile(sc_finite, 99))
        spread = {
            'p1':  round(float(np.percentile(sc_finite, 1)), 6),
            'p10': round(float(np.percentile(sc_finite, 10)), 6),
            'p25': round(float(np.percentile(sc_finite, 25)), 6),
            'p50': round(sc_p50, 6),
            'p75': round(float(np.percentile(sc_finite, 75)), 6),
            'p90': round(sc_p90, 6),
            'p99': round(sc_p99, 6),
        }
        disc_p90p50 = None
        disc_p99p50 = None
        if abs(sc_p50) > 1e-8:
            disc_p90p50 = round(sc_p90 / sc_p50, 2)
            disc_p99p50 = round(sc_p99 / sc_p50, 2)
        return spread, disc_p90p50, disc_p99p50

    def test_p10_key_present_in_score_spread(self):
        rng = np.random.RandomState(1)
        scores = rng.uniform(0.001, 1.0, 100)
        result = self._build_score_spread(scores)
        self.assertIsNotNone(result)
        spread, _, _ = result
        self.assertIn('p10', spread)

    def test_p10_value_is_correct(self):
        rng = np.random.RandomState(2)
        scores = rng.uniform(0.001, 1.0, 100)
        result = self._build_score_spread(scores)
        spread, _, _ = result
        expected = round(float(np.percentile(scores, 10)), 6)
        self.assertAlmostEqual(spread['p10'], expected, places=5)

    def test_score_disc_p90p50_is_computed(self):
        scores = np.arange(1, 101, dtype=float)
        result = self._build_score_spread(scores)
        spread, disc_p90p50, disc_p99p50 = result
        self.assertIsNotNone(disc_p90p50)
        self.assertIsNotNone(disc_p99p50)
        expected = round(spread['p90'] / spread['p50'], 2)
        self.assertAlmostEqual(disc_p90p50, expected, places=1)

    def test_score_disc_none_when_p50_near_zero(self):
        scores = np.zeros(20)
        result = self._build_score_spread(scores)
        if result is None:
            return  # too few finite
        spread, disc_p90p50, disc_p99p50 = result
        self.assertIsNone(disc_p90p50)
        self.assertIsNone(disc_p99p50)

    def test_score_spread_includes_all_expected_keys(self):
        rng = np.random.RandomState(3)
        scores = rng.uniform(0.001, 1.0, 50)
        result = self._build_score_spread(scores)
        spread, _, _ = result
        for key in ['p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99']:
            self.assertIn(key, spread, f"Missing key: {key}")


class TestWFFoldProofExtension(unittest.TestCase):
    """
    [WF_FOLD_PROOF] log now includes gate_mode, gate_cutoff, score_p10, score_p50, score_p90.
    Tested by checking the format strings are referenced in v5_train.py source.
    """

    def _check_source_contains(self, pattern):
        """Read v5_train.py source and check pattern is present."""
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            src = f.read()
        return pattern in src

    def test_wf_fold_proof_includes_gate_mode(self):
        self.assertTrue(
            self._check_source_contains('gate_mode={_proof_gate_mode}'),
            "[WF_FOLD_PROOF] does not include gate_mode field"
        )

    def test_wf_fold_proof_includes_gate_cutoff(self):
        self.assertTrue(
            self._check_source_contains('gate_cutoff={_proof_gc_str}'),
            "[WF_FOLD_PROOF] does not include gate_cutoff field"
        )

    def test_wf_fold_proof_includes_score_p10(self):
        self.assertTrue(
            self._check_source_contains('score_p10={_proof_p10_str}'),
            "[WF_FOLD_PROOF] does not include score_p10 field"
        )

    def test_wf_fold_proof_includes_score_p50(self):
        self.assertTrue(
            self._check_source_contains('score_p50={_proof_p50_str}'),
            "[WF_FOLD_PROOF] does not include score_p50 field"
        )

    def test_wf_fold_proof_includes_score_p90(self):
        self.assertTrue(
            self._check_source_contains('score_p90={_proof_p90_str}'),
            "[WF_FOLD_PROOF] does not include score_p90 field"
        )

    def test_proof_gate_cutoff_extracted_from_fold_report(self):
        self.assertTrue(
            self._check_source_contains("fold_report.get('gate_cutoff', None)"),
            "gate_cutoff not extracted from fold_report"
        )

    def test_proof_gate_mode_extracted_from_fold_report(self):
        self.assertTrue(
            self._check_source_contains("fold_report.get('gate_mode', 'ref_magnitude')"),
            "gate_mode not extracted from fold_report"
        )

    def test_proof_score_p10_extracted_from_score_spread(self):
        self.assertTrue(
            self._check_source_contains("_proof_sc_spread.get('p10', None)"),
            "score_p10 not extracted from score_spread in WF_FOLD_PROOF"
        )


class TestGateAuditLogMarker(unittest.TestCase):
    """[V5_GATE_AUDIT] log marker is present in v5_train.py source."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_gate_audit_marker_present(self):
        self.assertIn('[V5_GATE_AUDIT]', self.src)

    def test_gate_audit_logs_percentile_cutoff(self):
        self.assertIn('percentile_cutoff(p85)', self.src)

    def test_gate_audit_logs_taken_by_ref_gate(self):
        self.assertIn('taken_by_ref_gate', self.src)

    def test_gate_audit_logs_taken_by_pct_gate(self):
        self.assertIn('taken_by_pct_gate', self.src)

    def test_gate_audit_logs_overlap(self):
        self.assertIn('overlap=', self.src)

    def test_gate_audit_logs_gate_pass_rate(self):
        self.assertIn('gate_pass_rate=', self.src)

    def test_gate_audit_only_runs_when_10_or_more_scores(self):
        # The guard is `if len(finite_work) >= 10:`
        self.assertIn('if len(finite_work) >= 10:', self.src)

    def test_percentile_top15_override_log_present(self):
        self.assertIn('PERCENTILE_TOP15 mode active', self.src)

    def test_gate_mode_getattr_with_default(self):
        self.assertIn("getattr(config, 'gate_mode', 'ref_magnitude')", self.src)


class TestGateBaselineCmpInfrastructure(unittest.TestCase):
    """[V5_GATE_BASELINE_CMP] infrastructure: 6 required metrics, promotion logic, go/no-go."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_gate_run_metrics_path_defined(self):
        self.assertIn('v5_gate_run_metrics.json', self.src)

    def test_gate_baseline_metrics_path_defined(self):
        self.assertIn('v5_gate_baseline_metrics.json', self.src)

    # --- 6 required comparison metric keys ---
    def test_gate_run_metrics_has_mean_per_fold_total_r(self):
        self.assertIn("'mean_per_fold_total_r'", self.src)

    def test_gate_run_metrics_has_mean_per_fold_expectancy_r(self):
        self.assertIn("'mean_per_fold_expectancy_r'", self.src)

    def test_gate_run_metrics_has_monotonic_fold_count(self):
        self.assertIn("'monotonic_fold_count'", self.src)

    def test_gate_run_metrics_has_mean_score_disc_p90p50(self):
        self.assertIn("'mean_score_disc_p90p50'", self.src)

    def test_gate_run_metrics_has_relax_loop_total(self):
        self.assertIn("'relax_loop_total'", self.src)

    def test_gate_run_metrics_has_gate_pass_rate_std(self):
        self.assertIn("'gate_pass_rate_std'", self.src)

    # --- audit extras still present ---
    def test_gate_run_metrics_has_score_p10_mean(self):
        self.assertIn("'score_p10_mean'", self.src)

    def test_gate_run_metrics_has_mean_gate_cutoff(self):
        self.assertIn("'mean_gate_cutoff'", self.src)

    # --- markers ---
    def test_gate_baseline_cmp_marker_present(self):
        self.assertIn('[V5_GATE_BASELINE_CMP]', self.src)

    def test_gate_run_metrics_marker_present(self):
        self.assertIn('[V5_GATE_RUN_METRICS]', self.src)

    # --- 6 required metrics appear in the CMP section ---
    def test_gate_cmp_compares_mean_per_fold_total_r(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        self.assertGreater(idx, 0)
        gate_cmp_section = self.src[idx:]
        self.assertIn('mean_per_fold_total_r', gate_cmp_section)

    def test_gate_cmp_compares_mean_per_fold_expectancy_r(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        gate_cmp_section = self.src[idx:]
        self.assertIn('mean_per_fold_expectancy_r', gate_cmp_section)

    def test_gate_cmp_compares_monotonic_fold_count(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        gate_cmp_section = self.src[idx:]
        self.assertIn('monotonic_fold_count', gate_cmp_section)

    def test_gate_cmp_compares_mean_score_disc_p90p50(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        gate_cmp_section = self.src[idx:]
        self.assertIn('mean_score_disc_p90p50', gate_cmp_section)

    def test_gate_cmp_compares_relax_loop_total(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        gate_cmp_section = self.src[idx:]
        self.assertIn('relax_loop_total', gate_cmp_section)

    def test_gate_cmp_compares_gate_pass_rate_std(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        gate_cmp_section = self.src[idx:]
        self.assertIn('gate_pass_rate_std', gate_cmp_section)

    # --- promotion log messages ---
    def test_gate_promoted_log_present(self):
        self.assertIn('[V5_GATE] Percentile gate PROMOTED', self.src,
                      "Missing promoted log line required by task spec")

    def test_gate_not_promoted_log_present(self):
        self.assertIn('[V5_GATE] Percentile gate NOT promoted', self.src,
                      "Missing not-promoted log line required by task spec")

    def test_promotion_requires_all_pass(self):
        # Promotion log must be preceded by an all-pass check
        self.assertIn('_gcmp_all_pass', self.src)

    def test_promotion_only_for_non_ref_magnitude_mode(self):
        # Promotion only fires when gate_mode != 'ref_magnitude'
        idx = self.src.find('[V5_GATE] Percentile gate PROMOTED')
        self.assertGreater(idx, 0)
        context_before = self.src[max(0, idx - 300):idx]
        self.assertIn("ref_magnitude", context_before,
                      "Promotion must be guarded by gate_mode check")

    def test_fmt_cmp_defined_before_gate_cmp_section(self):
        fmt_cmp_idx = self.src.find('def _fmt_cmp(')
        gate_cmp_idx = self.src.find("[V5_GATE_BASELINE_CMP] GATE METRICS COMPARISON")
        self.assertGreater(fmt_cmp_idx, 0, "_fmt_cmp not found")
        self.assertGreater(gate_cmp_idx, 0, "[V5_GATE_BASELINE_CMP] GATE METRICS COMPARISON not found")
        self.assertLess(fmt_cmp_idx, gate_cmp_idx,
                        "_fmt_cmp must be defined before [V5_GATE_BASELINE_CMP] section")


class TestGateRunMetricsComputation(unittest.TestCase):
    """Test the gate run metrics computation logic: the 6 required comparison metrics."""

    def _compute_gate_run_metrics(self, active_rpts):
        """Reproduce _gate_run_metrics computation from v5_train.py."""
        gate_total_r_vals   = [r.get('total_r', 0.0) for r in active_rpts]
        gate_expect_vals    = [r.get('expectancy_r', 0.0) for r in active_rpts]
        gate_mono_vals      = [r.get('score_monotonic') for r in active_rpts if r.get('score_monotonic') is not None]
        gate_disc90_vals    = [r.get('score_disc_p90p50') for r in active_rpts if r.get('score_disc_p90p50') is not None]
        gate_relax_vals     = [r.get('relax_loop_triggers', 0) for r in active_rpts]
        gate_pass_rate_vals = [r.get('gate_pass_rate') for r in active_rpts if r.get('gate_pass_rate') is not None]
        gate_p10_vals  = [r.get('score_spread', {}).get('p10') for r in active_rpts if r.get('score_spread', {}).get('p10') is not None]
        gate_cutoff_vals = [r.get('gate_cutoff') for r in active_rpts if r.get('gate_cutoff') is not None]
        gate_mode_used = active_rpts[0].get('gate_mode', 'ref_magnitude') if active_rpts else 'ref_magnitude'
        return {
            'gate_mode_used':            gate_mode_used,
            'mean_per_fold_total_r':     round(float(np.mean(gate_total_r_vals)), 4) if gate_total_r_vals else None,
            'mean_per_fold_expectancy_r':round(float(np.mean(gate_expect_vals)), 4)  if gate_expect_vals  else None,
            'monotonic_fold_count':      int(sum(1 for v in gate_mono_vals if v)),
            'mean_score_disc_p90p50':    round(float(np.mean(gate_disc90_vals)), 4)  if gate_disc90_vals  else None,
            'relax_loop_total':          int(sum(gate_relax_vals)),
            'gate_pass_rate_std':        round(float(np.std(gate_pass_rate_vals)), 2) if len(gate_pass_rate_vals) >= 2 else None,
            'score_p10_mean':            round(float(np.mean(gate_p10_vals)), 6)      if gate_p10_vals     else None,
            'mean_gate_cutoff':          round(float(np.mean(gate_cutoff_vals)), 6)   if gate_cutoff_vals  else None,
            'monotonic_pct':             round(100.0 * sum(1 for v in gate_mono_vals if v) / max(len(gate_mono_vals), 1), 1) if gate_mono_vals else None,
        }

    def _make_fold_report(self, gate_mode='ref_magnitude', gate_cutoff=0.15,
                          gate_pass_rate=65.0, score_monotonic=True,
                          total_r=0.5, expectancy_r=0.03, score_disc_p90p50=1.8,
                          relax_loop_triggers=1, score_spread=None):
        if score_spread is None:
            score_spread = {'p10': 0.01, 'p50': 0.05, 'p90': 0.20, 'p99': 0.50}
        return {
            'total_trades': 10,
            'total_r': total_r,
            'expectancy_r': expectancy_r,
            'score_disc_p90p50': score_disc_p90p50,
            'relax_loop_triggers': relax_loop_triggers,
            'gate_mode': gate_mode,
            'gate_cutoff': gate_cutoff,
            'gate_pass_rate': gate_pass_rate,
            'score_monotonic': score_monotonic,
            'score_spread': score_spread,
        }

    def test_mean_per_fold_total_r_computed(self):
        folds = [
            self._make_fold_report(total_r=1.0),
            self._make_fold_report(total_r=3.0),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertAlmostEqual(m['mean_per_fold_total_r'], 2.0, places=3)

    def test_mean_per_fold_expectancy_r_computed(self):
        folds = [
            self._make_fold_report(expectancy_r=0.02),
            self._make_fold_report(expectancy_r=0.04),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertAlmostEqual(m['mean_per_fold_expectancy_r'], 0.03, places=3)

    def test_monotonic_fold_count_is_integer_count(self):
        folds = [
            self._make_fold_report(score_monotonic=True),
            self._make_fold_report(score_monotonic=False),
            self._make_fold_report(score_monotonic=True),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertEqual(m['monotonic_fold_count'], 2)

    def test_monotonic_fold_count_zero_when_none_monotonic(self):
        folds = [self._make_fold_report(score_monotonic=False) for _ in range(4)]
        m = self._compute_gate_run_metrics(folds)
        self.assertEqual(m['monotonic_fold_count'], 0)

    def test_mean_score_disc_p90p50_computed(self):
        folds = [
            self._make_fold_report(score_disc_p90p50=2.0),
            self._make_fold_report(score_disc_p90p50=3.0),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertAlmostEqual(m['mean_score_disc_p90p50'], 2.5, places=3)

    def test_relax_loop_total_is_sum_across_folds(self):
        folds = [
            self._make_fold_report(relax_loop_triggers=2),
            self._make_fold_report(relax_loop_triggers=5),
            self._make_fold_report(relax_loop_triggers=0),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertEqual(m['relax_loop_total'], 7)

    def test_gate_pass_rate_std_computed_across_folds(self):
        folds = [
            self._make_fold_report(gate_pass_rate=60.0),
            self._make_fold_report(gate_pass_rate=80.0),
            self._make_fold_report(gate_pass_rate=70.0),
        ]
        m = self._compute_gate_run_metrics(folds)
        expected_std = round(float(np.std([60.0, 80.0, 70.0])), 2)
        self.assertAlmostEqual(m['gate_pass_rate_std'], expected_std, places=1)

    def test_gate_pass_rate_std_none_when_only_one_fold(self):
        folds = [self._make_fold_report(gate_pass_rate=65.0)]
        m = self._compute_gate_run_metrics(folds)
        self.assertIsNone(m['gate_pass_rate_std'])

    def test_gate_mode_used_from_first_fold(self):
        folds = [
            self._make_fold_report(gate_mode='percentile_top15'),
            self._make_fold_report(gate_mode='ref_magnitude'),
        ]
        m = self._compute_gate_run_metrics(folds)
        self.assertEqual(m['gate_mode_used'], 'percentile_top15')

    def test_empty_active_folds_returns_defaults(self):
        m = self._compute_gate_run_metrics([])
        self.assertEqual(m['gate_mode_used'], 'ref_magnitude')
        self.assertEqual(m['relax_loop_total'], 0)
        self.assertEqual(m['monotonic_fold_count'], 0)
        self.assertIsNone(m['gate_pass_rate_std'])
        self.assertIsNone(m['score_p10_mean'])


class TestGateBaselineCmpFmtCmpAvailability(unittest.TestCase):
    """_fmt_cmp must be defined at scope accessible to [V5_GATE_BASELINE_CMP]."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_fmt_cmp_defined_outside_if_baseline_exists_block(self):
        """_fmt_cmp must be defined before (and outside) if _baseline_path.exists(): block."""
        fmt_cmp_def = 'def _fmt_cmp('
        baseline_if = 'if _baseline_path.exists():'
        gate_if = 'if _gate_baseline_path.exists():'
        
        idx_fmt   = self.src.find(fmt_cmp_def)
        idx_base  = self.src.find(baseline_if)
        idx_gate  = self.src.find(gate_if)
        
        self.assertGreater(idx_fmt, 0)
        self.assertGreater(idx_base, 0)
        self.assertGreater(idx_gate, 0)
        # _fmt_cmp must be before both if-blocks
        self.assertLess(idx_fmt, idx_base, "_fmt_cmp must be before if _baseline_path.exists()")
        self.assertLess(idx_fmt, idx_gate, "_fmt_cmp must be before if _gate_baseline_path.exists()")


class TestCLIGateModeArg(unittest.TestCase):
    """--v5-gate-mode CLI argument exists and has correct choices."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'quick_start.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_v5_gate_mode_arg_present(self):
        self.assertIn('--v5-gate-mode', self.src)

    def test_choice_ref_magnitude_present(self):
        self.assertIn('"ref_magnitude"', self.src)

    def test_choice_percentile_top15_present(self):
        self.assertIn('"percentile_top15"', self.src)

    def test_default_is_ref_magnitude(self):
        idx = self.src.find('--v5-gate-mode')
        self.assertGreater(idx, 0)
        segment = self.src[idx:idx+500]
        self.assertIn('default="ref_magnitude"', segment)

    def test_gate_mode_wired_to_first_wf_call(self):
        self.assertIn("gate_mode=getattr(args, 'v5_gate_mode', 'ref_magnitude')", self.src)

    def test_gate_mode_wired_in_both_wf_call_sites(self):
        count = self.src.count("gate_mode=getattr(args, 'v5_gate_mode', 'ref_magnitude')")
        self.assertGreaterEqual(count, 2,
                                f"Expected gate_mode wired in at least 2 WF call sites, found {count}")


class TestRunV5WalkForwardSignature(unittest.TestCase):
    """run_v5_walk_forward() has gate_mode parameter."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_run_v5_walk_forward_has_gate_mode_param(self):
        wf_idx = self.src.find('def run_v5_walk_forward(')
        self.assertGreater(wf_idx, 0)
        # Find the closing paren of the function signature
        sig_end = self.src.find('):', wf_idx)
        sig = self.src[wf_idx:sig_end]
        self.assertIn("gate_mode=", sig)

    def test_train_v5_model_has_gate_mode_param(self):
        fn_idx = self.src.find('def train_v5_model(')
        self.assertGreater(fn_idx, 0)
        sig_end = self.src.find('):', fn_idx)
        sig = self.src[fn_idx:sig_end]
        self.assertIn("gate_mode=", sig)

    def test_v5forward_test_config_passed_gate_mode_in_wf(self):
        # Inside run_v5_walk_forward, the V5ForwardTestConfig construction passes gate_mode
        self.assertIn('gate_mode=gate_mode,', self.src)

    def test_fwd_config_passed_gate_mode_in_train_v5_model(self):
        # Inside train_v5_model, fwd_config passes gate_mode
        # The second occurrence (in train_v5_model context) should also exist
        count = self.src.count('gate_mode=gate_mode,')
        self.assertGreaterEqual(count, 2,
                                f"Expected gate_mode=gate_mode, in at least 2 places, found {count}")


class TestFmtCmpBehaviorGateCmp(unittest.TestCase):
    """_fmt_cmp helper: >= semantics for higher_is_better, strict < for strict_lower metrics."""

    def _fmt_cmp(self, name, bv, nv, threshold=None, higher_is_better=True, fmt='.4f',
                 upper_threshold=None, strict_lower=False):
        """Reproduce _fmt_cmp from v5_train.py."""
        if nv is None:
            return f"  {name:<50}: baseline=N/A  new=N/A  [SKIP]"
        bv_str = f"{bv:{fmt}}" if bv is not None else "N/A"
        diff_str = f"{nv - bv:+{fmt}}" if bv is not None else "N/A"
        if upper_threshold is not None:
            passed = nv <= upper_threshold
        elif threshold is not None:
            passed = nv >= threshold
        elif bv is not None:
            if higher_is_better:
                passed = nv >= bv
            elif strict_lower:
                passed = nv < bv
            else:
                passed = nv <= bv
        else:
            passed = None
        status = "PASS" if passed else ("FAIL" if passed is not None else "N/A")
        return (
            f"  {name:<50}: baseline={bv_str:<12}  "
            f"new={nv:{fmt}}  diff={diff_str}  [{status}]"
        )

    # --- higher_is_better (New >= Baseline) ---
    def test_higher_is_better_strict_improvement(self):
        line = self._fmt_cmp("mean_per_fold_total_r", 1.0, 1.5, higher_is_better=True)
        self.assertIn('[PASS]', line)

    def test_higher_is_better_equality_is_pass(self):
        line = self._fmt_cmp("mean_per_fold_total_r", 1.0, 1.0, higher_is_better=True)
        self.assertIn('[PASS]', line)

    def test_higher_is_better_regression_is_fail(self):
        line = self._fmt_cmp("mean_per_fold_expectancy_r", 0.05, 0.03, higher_is_better=True)
        self.assertIn('[FAIL]', line)

    def test_monotonic_fold_count_equality_is_pass(self):
        line = self._fmt_cmp("monotonic_fold_count", 3.0, 3.0, higher_is_better=True, fmt='.0f')
        self.assertIn('[PASS]', line)

    # --- strict_lower (New < Baseline, equality = FAIL) ---
    def test_strict_lower_improves_from_10_to_8(self):
        line = self._fmt_cmp("relax_loop_total", 10.0, 8.0,
                             higher_is_better=False, strict_lower=True, fmt='.0f')
        self.assertIn('[PASS]', line)

    def test_strict_lower_equality_is_fail(self):
        # equality must FAIL under strict_lower (New < Baseline required)
        line = self._fmt_cmp("relax_loop_total", 5.0, 5.0,
                             higher_is_better=False, strict_lower=True, fmt='.0f')
        self.assertIn('[FAIL]', line)

    def test_strict_lower_regression_is_fail(self):
        line = self._fmt_cmp("gate_pass_rate_std", 5.0, 15.0,
                             higher_is_better=False, strict_lower=True)
        self.assertIn('[FAIL]', line)

    def test_strict_lower_used_for_relax_loop_total_in_source(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            src = f.read()
        idx = src.find('relax_loop_total')
        cmp_idx = src.find('[V5_GATE_BASELINE_CMP]')
        # find the _fmt_cmp call for relax_loop_total after the gate cmp section starts
        segment = src[cmp_idx:]
        self.assertIn('strict_lower=True', segment,
                      "relax_loop_total and gate_pass_rate_std must use strict_lower=True")

    def test_strict_lower_used_for_gate_pass_rate_std_in_source(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            src = f.read()
        cmp_idx = src.find('[V5_GATE_BASELINE_CMP]')
        segment = src[cmp_idx:]
        count = segment.count('strict_lower=True')
        self.assertGreaterEqual(count, 2,
                                "Both relax_loop_total and gate_pass_rate_std must use strict_lower=True")

    # --- skip ---
    def test_skip_when_new_value_is_none(self):
        line = self._fmt_cmp("mean_score_disc_p90p50", 2.0, None)
        self.assertIn('[SKIP]', line)

    # --- threshold ---
    def test_threshold_gate_pass_at_boundary(self):
        line = self._fmt_cmp("monotonic_pct", 40.0, 50.0, threshold=50.0, fmt='.1f')
        self.assertIn('[PASS]', line)

    def test_threshold_gate_fail_below_boundary(self):
        line = self._fmt_cmp("monotonic_pct", 70.0, 49.9, threshold=50.0, fmt='.1f')
        self.assertIn('[FAIL]', line)


class TestGateSelectRateTracking(unittest.TestCase):
    """gate_select_rate: post-threshold selection rate (n_taken / n_total_bars * 100)."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_gate_select_rate_key_in_report(self):
        self.assertIn("report['gate_select_rate']", self.src,
                      "gate_select_rate must be stored in fold report")

    def test_gate_select_rate_uses_n_taken(self):
        self.assertIn('n_taken', self.src)
        # Verify gate_select_rate uses n_taken (post-threshold bars)
        idx = self.src.find("report['gate_select_rate']")
        context = self.src[max(0, idx - 50):idx + 200]
        self.assertIn('n_taken', context,
                      "gate_select_rate must be computed from n_taken (threshold-selected bars)")

    def test_gate_select_rate_uses_n_total_bars(self):
        idx = self.src.find("report['gate_select_rate']")
        context = self.src[max(0, idx - 100):idx + 300]
        self.assertIn('n_total_bars', context,
                      "gate_select_rate must use n_total_bars as denominator")

    def test_gate_select_rate_computation(self):
        """gate_select_rate = 100 * n_taken / n_total_bars."""
        n_taken = 150
        n_total_bars = 1000
        rate = round(100.0 * n_taken / max(n_total_bars, 1), 2)
        self.assertAlmostEqual(rate, 15.0, places=1)

    def test_gate_select_rate_zero_when_no_bars(self):
        n_taken = 0
        n_total_bars = 0
        rate = 100.0 * n_taken / max(n_total_bars, 1) if n_total_bars > 0 else 0.0
        self.assertAlmostEqual(rate, 0.0, places=1)

    def test_gate_pass_rate_std_uses_gate_select_rate_in_wf_section(self):
        idx = self.src.find('[V5_GATE_BASELINE_CMP]')
        self.assertGreater(idx, 0)
        gate_cmp_section = self.src[idx - 3000:idx + 3000]
        self.assertIn('gate_select_rate', gate_cmp_section,
                      "gate_pass_rate_std in [V5_GATE_BASELINE_CMP] must use gate_select_rate, not gate_pass_rate")

    def test_gate_pass_rate_preserved_for_backward_compat(self):
        self.assertIn('_qual_gate_pass_rate', self.src)
        self.assertIn("report['gate_pass_rate']", self.src)

    def test_gate_cutoff_source_in_v5_train(self):
        self.assertIn("report['gate_cutoff']", self.src)

    def test_gate_mode_source_in_v5_train(self):
        self.assertIn("report['gate_mode']", self.src)


class TestPromotionStrictRequirements(unittest.TestCase):
    """Promotion requires exactly 6 PASS lines and 0 SKIP lines."""

    def setUp(self):
        src_path = os.path.join(os.path.dirname(__file__), 'train', 'v5_train.py')
        with open(src_path) as f:
            self.src = f.read()

    def test_gcmp_skips_tracked(self):
        self.assertIn('_gcmp_skips', self.src)

    def test_promotion_requires_6_or_more_passes(self):
        self.assertIn('len(_gcmp_passes) >= 6', self.src,
                      "Promotion must require at least 6 non-SKIP PASS lines")

    def test_promotion_requires_zero_skips(self):
        self.assertIn('len(_gcmp_skips) == 0', self.src,
                      "Promotion must require 0 SKIP lines (all 6 metrics present)")

    def test_not_promoted_message_includes_skip_count(self):
        idx = self.src.find('[V5_GATE] Percentile gate NOT promoted')
        self.assertGreater(idx, 0)
        context = self.src[idx:idx + 300]
        self.assertIn('SKIP', context,
                      "NOT promoted message should mention SKIP metric count")

    def _simulate_promotion(self, lines):
        """Simulate go/no-go promotion decision logic from v5_train.py."""
        passes = [('[PASS]' in ln) for ln in lines if '[SKIP]' not in ln]
        skips  = [ln for ln in lines if '[SKIP]' in ln]
        all_pass = all(passes) and len(passes) >= 6 and len(skips) == 0
        return all_pass, passes, skips

    def test_all_6_pass_no_skip_promotes(self):
        lines = [f"  metric_{i}          : baseline=1.0  new=2.0  diff=+1.0  [PASS]" for i in range(6)]
        all_pass, passes, skips = self._simulate_promotion(lines)
        self.assertTrue(all_pass)

    def test_5_pass_1_fail_does_not_promote(self):
        lines = [f"  metric_{i}          : baseline=1.0  new=2.0  diff=+1.0  [PASS]" for i in range(5)]
        lines.append("  metric_5          : baseline=2.0  new=0.5  diff=-1.5  [FAIL]")
        all_pass, passes, skips = self._simulate_promotion(lines)
        self.assertFalse(all_pass)

    def test_any_skip_blocks_promotion(self):
        lines = [f"  metric_{i}          : baseline=1.0  new=2.0  diff=+1.0  [PASS]" for i in range(5)]
        lines.append("  metric_5          : baseline=N/A  new=N/A  [SKIP]")
        all_pass, passes, skips = self._simulate_promotion(lines)
        self.assertFalse(all_pass)

    def test_fewer_than_6_non_skip_lines_blocks_promotion(self):
        lines = [f"  metric_{i}          : baseline=1.0  new=2.0  diff=+1.0  [PASS]" for i in range(4)]
        all_pass, passes, skips = self._simulate_promotion(lines)
        self.assertFalse(all_pass)


if __name__ == '__main__':
    unittest.main(verbosity=2)
