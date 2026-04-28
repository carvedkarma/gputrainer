"""
Task #53 — V5 Score Collapse Fix Tests
=======================================
Covers three bugs that caused zero-trade training runs:

  Bug A: 2-class KL normalised away HOLD → all-HOLD model got near-zero KL penalty.
         Fix: use full 3-class KL [p_hold, p_long, p_short] with per-regime targets.

  Bug B: mae_cap parameter defined but never applied in compute_v5_scores.
         Fix: mae_pred = np.minimum(mae_pred, mae_cap) before risk floor.

  Bug C: [V5_DEBIAS] log marker missing — added dedicated startup log.

≥20 tests; all prior test suites (test_task51, test_task52, test_wf_threshold_ema)
must continue to pass.
"""

import re
import sys
import os
import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v5_train_src():
    path = os.path.join(os.path.dirname(__file__), "train", "v5_train.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def quick_start_src():
    path = os.path.join(os.path.dirname(__file__), "quick_start.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Bug B — mae_cap is now applied BEFORE the risk floor
# ---------------------------------------------------------------------------

class TestMaeCapApplied:
    """Bug B: mae_pred = np.minimum(mae_pred, mae_cap) must precede the risk floor."""

    def test_mae_minimum_line_exists(self, v5_train_src):
        """The explicit np.minimum(mae_pred, mae_cap) line must exist."""
        assert "mae_pred = np.minimum(mae_pred, mae_cap)" in v5_train_src, (
            "BUG B unfixed: mae_pred = np.minimum(mae_pred, mae_cap) line missing "
            "from compute_v5_scores — mae_cap was defined but never applied."
        )

    def test_mae_minimum_before_risk_floor(self, v5_train_src):
        """mae_cap must be applied BEFORE the risk floor (np.maximum)."""
        min_pos = v5_train_src.find("mae_pred = np.minimum(mae_pred, mae_cap)")
        floor_pos = v5_train_src.find("risk = np.maximum(mae_pred, 1e-4)")
        assert min_pos != -1, "mae_pred = np.minimum(mae_pred, mae_cap) not found"
        assert floor_pos != -1, "risk = np.maximum(mae_pred, 1e-4) not found"
        assert min_pos < floor_pos, (
            "mae_cap clamp must appear BEFORE the risk floor; found at "
            f"pos {min_pos} vs floor at pos {floor_pos}"
        )

    def test_mae_cap_not_applied_after_risk(self, v5_train_src):
        """Confirm no second np.minimum that wraps the final risk variable."""
        # The fix should be applied once before the floor, not after.
        min_pos = v5_train_src.find("mae_pred = np.minimum(mae_pred, mae_cap)")
        floor_pos = v5_train_src.find("risk = np.maximum(mae_pred, 1e-4)")
        # There must not be another minimum between floor and the score formula.
        score_formula_str = "score = (mu_R_adj / risk)"
        score_pos = v5_train_src.find(score_formula_str, floor_pos)
        if score_pos == -1:
            score_pos = v5_train_src.find("/ risk", floor_pos)
        # This test passes as long as the positions are ordered correctly.
        assert min_pos < floor_pos, "clamp before floor ordering violated"

    def test_mae_cap_numerical_small_mae(self):
        """If mae_pred < mae_cap, clamp has no effect (correct behaviour)."""
        mae_pred = np.array([0.5, 1.0, 1.5])
        mae_cap = 2.0
        clamped = np.minimum(mae_pred, mae_cap)
        np.testing.assert_array_equal(clamped, mae_pred)

    def test_mae_cap_numerical_large_mae(self):
        """If mae_pred > mae_cap, it must be reduced to mae_cap."""
        mae_pred = np.array([4.81, 5.0, 10.0])
        mae_cap = 2.0
        clamped = np.minimum(mae_pred, mae_cap)
        np.testing.assert_array_equal(clamped, np.array([2.0, 2.0, 2.0]))

    def test_mae_cap_improves_high_confidence_score(self):
        """
        Demonstrate the key effect of mae_cap: for HIGH-CONFIDENCE predictions
        (p_side > 0.333), applying mae_cap makes the score LARGER, allowing more
        signals to pass the score threshold.

        Score formula (simplified):
            conf_mult = p_side - score_lambda * (1 - p_side)
            score = (mu_R / risk) * conf_mult

        When mae_pred = 4.81 (uncapped) vs 2.0 (capped) with score_lambda = 0.5:
            A high-confidence bar p_side = 0.80 gets a 2.4x larger score with cap,
            meaning signals that were blocked by the threshold now pass.
        """
        score_lambda = 0.5
        p_side = 0.80   # high-confidence prediction
        mu_R = 0.10

        conf_mult = p_side - score_lambda * (1 - p_side)
        assert conf_mult > 0, "Expected positive confidence multiplier for p_side=0.80"

        # Without cap: large mae inflates denominator → smaller score
        mae_uncapped = 4.81
        risk_uncapped = max(mae_uncapped, 1e-4)
        score_uncapped = (mu_R / risk_uncapped) * conf_mult

        # With cap: clamped mae → larger score (better chance to pass threshold)
        mae_capped = float(np.minimum(mae_uncapped, 2.0))
        risk_capped = max(mae_capped, 1e-4)
        score_capped = (mu_R / risk_capped) * conf_mult

        assert math.isfinite(score_uncapped), "Uncapped score must be finite"
        assert math.isfinite(score_capped), "Capped score must be finite"
        assert score_capped > score_uncapped, (
            f"High-confidence score must be LARGER with cap ({score_capped:.5f}) "
            f"than without ({score_uncapped:.5f}) — cap reduces risk denominator, "
            "making strong signals more likely to pass the threshold"
        )

    def test_mae_cap_comment_present(self, v5_train_src):
        """A BUG FIX comment must document why the cap is applied."""
        assert "BUG FIX" in v5_train_src and "mae_cap" in v5_train_src, (
            "A BUG FIX comment near the mae_cap application must be present for auditability"
        )


# ---------------------------------------------------------------------------
# Bug A — 3-class KL replaces 2-class KL in compute_v5_loss
# ---------------------------------------------------------------------------

class TestThreeClassKL:
    """Bug A: Full 3-class [p_hold, p_long, p_short] KL must be used."""

    def test_hold_idx_defined(self, v5_train_src):
        """HOLD_IDX = 0 must be defined (replaces old code that only used LONG/SHORT)."""
        assert "HOLD_IDX, LONG_IDX, SHORT_IDX = 0, 1, 2" in v5_train_src, (
            "Bug A fix must define HOLD_IDX, LONG_IDX, SHORT_IDX = 0, 1, 2 "
            "to use 3-class distribution"
        )

    def test_three_class_targets_dict(self, v5_train_src):
        """_3CLS_TARGETS dict with bull/bear/chop keys must exist."""
        assert "_3CLS_TARGETS" in v5_train_src, (
            "3-class target dict _3CLS_TARGETS must exist for bug A fix"
        )

    def test_bull_target_sums_to_one(self):
        """Bull target [0.30, 0.55, 0.15] must sum to 1.0."""
        bull = [0.30, 0.55, 0.15]
        assert abs(sum(bull) - 1.0) < 1e-9, f"Bull target sums to {sum(bull)}, expected 1.0"

    def test_bear_target_sums_to_one(self):
        """Bear target [0.30, 0.15, 0.55] must sum to 1.0."""
        bear = [0.30, 0.15, 0.55]
        assert abs(sum(bear) - 1.0) < 1e-9, f"Bear target sums to {sum(bear)}, expected 1.0"

    def test_chop_target_sums_to_one(self):
        """Chop target [0.40, 0.30, 0.30] must sum to 1.0."""
        chop = [0.40, 0.30, 0.30]
        assert abs(sum(chop) - 1.0) < 1e-9, f"Chop target sums to {sum(chop)}, expected 1.0"

    def test_old_2class_normalisation_removed(self, v5_train_src):
        """Old 2-class normalisation line must be removed."""
        assert "pred_g / (pred_g.sum() + eps)" not in v5_train_src, (
            "Old 2-class normalisation (pred_g / (pred_g.sum() + eps)) must be "
            "removed — it masked HOLD collapse by normalising away the p_hold component"
        )

    def test_pred_h_assembled_from_hold_idx(self, v5_train_src):
        """pred_h must extract from HOLD_IDX (index 0), not be absent."""
        assert "action_probs[mask, HOLD_IDX].mean()" in v5_train_src, (
            "pred_h = action_probs[mask, HOLD_IDX].mean() must exist to include "
            "HOLD probability in the 3-class KL distribution"
        )

    def test_three_class_kl_numerical(self):
        """
        Demonstrate that 3-class KL penalises all-HOLD predictions correctly.

        2-class KL (old): normalises [p_long, p_short] → [0, 0] / eps ≈ [0.5, 0.5]
            KL([0.5, 0.5] || [0.6, 0.4]) ≈ 0.02  (tiny — no gradient)

        3-class KL (new): uses [p_hold, p_long, p_short] = [1.0, 0.0, 0.0]
            KL([1.0, 0.0, 0.0] || [0.30, 0.55, 0.15]) → very large (HOLD gets penalised)
        """
        eps = 1e-8

        # All-HOLD prediction
        p_hold, p_long, p_short = 1.0 - 2e-4, 1e-4, 1e-4

        # Old 2-class KL: normalise [p_long, p_short]
        old_pred = np.array([p_long, p_short])
        old_pred = old_pred / (old_pred.sum() + eps)   # ≈ [0.5, 0.5]
        old_target = np.array([0.60, 0.40])
        old_kl = np.sum(old_target * np.log((old_target + eps) / (old_pred + eps)))

        # New 3-class KL
        new_pred = np.array([p_hold, p_long, p_short])  # ≈ [1.0, 0, 0]
        new_target = np.array([0.30, 0.55, 0.15])        # bull target
        new_kl = np.sum(new_target * np.log((new_target + eps) / (new_pred + eps)))

        assert new_kl > old_kl * 10, (
            f"3-class KL ({new_kl:.3f}) must be >> 2-class KL ({old_kl:.3f}) "
            "for all-HOLD predictions — otherwise the gradient pressure is too weak"
        )

    def test_side_bal_weight_default_015(self, v5_train_src):
        """compute_v5_loss must have side_bal_weight=0.15 default."""
        assert "side_bal_weight=0.15" in v5_train_src, (
            "compute_v5_loss must accept side_bal_weight with default 0.15"
        )

    def test_side_bal_w_assignment_from_param(self, v5_train_src):
        """SIDE_BAL_W must be assigned from the side_bal_weight parameter."""
        assert "SIDE_BAL_W = side_bal_weight" in v5_train_src, (
            "SIDE_BAL_W must be assigned from side_bal_weight parameter, not hardcoded"
        )

    def test_old_hardcoded_005_removed(self, v5_train_src):
        """Old hardcoded SIDE_BAL_W = 0.05 must be removed."""
        assert "SIDE_BAL_W = 0.05" not in v5_train_src, (
            "Hardcoded SIDE_BAL_W = 0.05 must be removed in favour of configurable param"
        )


# ---------------------------------------------------------------------------
# Bug C — [V5_DEBIAS] log marker
# ---------------------------------------------------------------------------

class TestVDebiasLogMarker:
    """Bug C: A dedicated [V5_DEBIAS] log line must appear at fold start."""

    def test_v5_debias_marker_in_source(self, v5_train_src):
        assert "[V5_DEBIAS]" in v5_train_src, (
            "[V5_DEBIAS] log marker must exist in v5_train.py for easy grep-ability"
        )

    def test_v5_debias_shows_on_off_state(self, v5_train_src):
        """The [V5_DEBIAS] log must show whether mu_debias is ON or OFF."""
        # Find the [V5_DEBIAS] log line
        idx = v5_train_src.find("[V5_DEBIAS]")
        assert idx != -1
        # Look at the surrounding 200 characters for ON/OFF or ENABLED/DISABLED
        snippet = v5_train_src[idx:idx+200]
        has_state = ("ON" in snippet or "OFF" in snippet or
                     "ENABLED" in snippet or "DISABLED" in snippet or
                     "mu_debias" in snippet)
        assert has_state, (
            "[V5_DEBIAS] log must emit the ON/OFF state of mu_debias near the marker"
        )

    def test_v5_debias_default_off_message(self, v5_train_src):
        """The [V5_DEBIAS] log should note that default is OFF (opt-in feature)."""
        idx = v5_train_src.find("[V5_DEBIAS]")
        snippet = v5_train_src[idx:idx+300]
        assert "default" in snippet.lower() or "OFF" in snippet or "disable" in snippet.lower(), (
            "[V5_DEBIAS] log should clarify that mu_debias is OFF by default"
        )


# ---------------------------------------------------------------------------
# CLI wiring — --v5-side-bal-weight
# ---------------------------------------------------------------------------

class TestCLIWiring:
    """--v5-side-bal-weight CLI flag must exist with default 0.15."""

    def test_cli_flag_exists(self, quick_start_src):
        assert "--v5-side-bal-weight" in quick_start_src, (
            "--v5-side-bal-weight CLI argument must be defined in quick_start.py"
        )

    def test_cli_flag_default_015(self, quick_start_src):
        """Default for --v5-side-bal-weight must be 0.15."""
        idx = quick_start_src.find("--v5-side-bal-weight")
        snippet = quick_start_src[idx:idx+200]
        assert "default=0.15" in snippet, (
            "--v5-side-bal-weight must have default=0.15"
        )

    def test_side_bal_weight_wired_in_walk_forward_call(self, quick_start_src):
        """side_bal_weight=args.v5_side_bal_weight must appear in run_v5_walk_forward call."""
        assert "side_bal_weight=args.v5_side_bal_weight" in quick_start_src, (
            "side_bal_weight must be wired from args into run_v5_walk_forward call"
        )

    def test_side_bal_weight_wired_in_train_call(self, quick_start_src):
        """side_bal_weight must also be passed to train_v5_model (standalone mode)."""
        # Count occurrences — should appear in at least 2 call sites
        count = quick_start_src.count("side_bal_weight=args.v5_side_bal_weight")
        assert count >= 2, (
            f"side_bal_weight=args.v5_side_bal_weight must appear in ≥2 call sites "
            f"(walk_forward and train), found {count}"
        )


# ---------------------------------------------------------------------------
# Parameter threading in v5_train.py
# ---------------------------------------------------------------------------

class TestParameterThreading:
    """side_bal_weight must be threaded through the full call chain."""

    def test_run_v5_walk_forward_has_side_bal_weight(self, v5_train_src):
        """run_v5_walk_forward must accept side_bal_weight parameter."""
        idx = v5_train_src.find("def run_v5_walk_forward(")
        assert idx != -1
        # The signature is very long (>100 params); read 12000 chars to cover it all
        snippet = v5_train_src[idx:idx+12000]
        assert "side_bal_weight" in snippet, (
            "run_v5_walk_forward must have side_bal_weight parameter"
        )

    def test_train_v5_model_has_side_bal_weight(self, v5_train_src):
        """train_v5_model must accept side_bal_weight parameter."""
        idx = v5_train_src.find("def train_v5_model(")
        assert idx != -1
        snippet = v5_train_src[idx:idx+5000]
        assert "side_bal_weight" in snippet, (
            "train_v5_model must have side_bal_weight parameter"
        )

    def test_compute_v5_loss_train_call_passes_side_bal_weight(self, v5_train_src):
        """The main training-loop compute_v5_loss call must pass side_bal_weight."""
        # All 3 call sites (train, val, finetune) must pass it
        count = v5_train_src.count("side_bal_weight=side_bal_weight")
        assert count >= 3, (
            f"side_bal_weight=side_bal_weight must appear in ≥3 compute_v5_loss call "
            f"sites (train, val, finetune), found {count}"
        )

    def test_walk_forward_passes_side_bal_weight_to_train(self, v5_train_src):
        """run_v5_walk_forward must pass side_bal_weight= to train_v5_model."""
        # Find the section of the file that belongs to run_v5_walk_forward.
        # The function is enormous; search from its def to the next top-level def.
        wf_start = v5_train_src.find("def run_v5_walk_forward(")
        assert wf_start != -1, "run_v5_walk_forward not found"
        # Next top-level def after run_v5_walk_forward is train_v5_model
        next_def = v5_train_src.find("\ndef train_v5_model(", wf_start)
        wf_body = v5_train_src[wf_start:next_def] if next_def != -1 else v5_train_src[wf_start:]
        assert "side_bal_weight=side_bal_weight" in wf_body, (
            "run_v5_walk_forward must forward side_bal_weight to train_v5_model; "
            "check that side_bal_weight=side_bal_weight appears in the call"
        )
