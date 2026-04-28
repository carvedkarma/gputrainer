"""
Task #54 — V5 Forward Test Crash + Action Head Direction Collapse Fix Tests
============================================================================
Covers five bugs / improvements:

  Bug A: UnboundLocalError — `per_bar_threshold` used before assignment when
         config.per_symbol_thresholds=False.  Fix: initialise to None before block.

  Bug B: Action head direction collapse (100% LONG) due to asymmetric 3-class KL
         targets and insufficient entropy.  Fix: symmetric targets + entropy reg.

  Bug C: Hardcoded 'SIDE_BAL_W=0.05' string in log at line ~7056 misleads reviewers.
         Actual computation uses side_bal_weight (default=0.15).
         Fix: replace literal with f-string variable.

  Feature D: [V5_DIR_COLLAPSE] per-epoch warning when >90% of val predictions
             are one direction — makes collapse immediately visible in logs.

  Feature E: --v5-action-entropy-weight CLI flag (default 0.10) wired through
             quick_start → run_v5_walk_forward → train_v5_model → compute_v5_loss.

≥15 tests; source-only tests require no GPU/torch.
Torch-dependent tests are skipped when torch is unavailable (CI / Replit env).
"""

import re
import sys
import os
import math
import pytest

# ── try to import torch — skip compute tests if unavailable ──────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── make gpu_trainer importable for torch tests ───────────────────────────────
TRAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

# ---------------------------------------------------------------------------
# Source fixtures — no torch needed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v5_train_src():
    path = os.path.join(TRAINER_DIR, "train", "v5_train.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def quick_start_src():
    path = os.path.join(TRAINER_DIR, "quick_start.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Torch helpers (only used in GPU tests)
# ---------------------------------------------------------------------------

def _dummy_batch(n=64):
    """Minimal batch matching compute_v5_loss contract.

    Required keys (from v5_train.py ~line 707-882):
      valid, ret_R, mfe_R, mae_R, action_label, regime_label
      barrier_label and barrier_soft only needed in oracle/learnable modes.
    """
    import torch as _torch
    return {
        "valid": _torch.ones(n, dtype=_torch.bool),   # all samples valid
        "ret_R": _torch.randn(n, 1),
        "mfe_R": _torch.rand(n, 1).abs(),
        "mae_R": _torch.rand(n, 1).abs(),
        "barrier_R": _torch.randn(n, 1),
        "action_label": _torch.randint(0, 3, (n,)),
        "regime_label": _torch.randint(0, 3, (n,)),
    }


def _dummy_outputs(n=64):
    """Model outputs matching compute_v5_loss contract.

    Required keys (from v5_train.py ~line 715-881):
      ret_mu, ret_log_sigma, mfe, mae, action_logits, regime_logits
      barrier_logits only used in oracle/learnable barrier modes.
    """
    import torch as _torch
    return {
        "ret_mu": _torch.randn(n, 1),
        "ret_log_sigma": _torch.zeros(n, 1),
        "mfe": _torch.rand(n, 1).abs(),       # correct key: 'mfe' not 'mfe_mu'
        "mfe_log_sigma": _torch.zeros(n, 1),
        "mae": _torch.rand(n, 1).abs(),       # correct key: 'mae' not 'mae_mu'
        "mae_log_sigma": _torch.zeros(n, 1),
        "action_logits": _torch.randn(n, 3),  # correct key: 'action_logits'
        "regime_logits": _torch.randn(n, 3),  # correct key: 'regime_logits'
    }


# ═══════════════════════════════════════════════════════════════════════════
# BUG A — per_bar_threshold crash fix
# ═══════════════════════════════════════════════════════════════════════════

class TestPerBarThresholdCrashFix:
    """Bug A: per_bar_threshold must be initialised to None before the
    conditional block so the forward test does not crash when
    config.per_symbol_thresholds=False."""

    def test_none_initialisation_present(self, v5_train_src):
        """'per_bar_threshold = None' must exist in source."""
        assert "per_bar_threshold = None" in v5_train_src, (
            "BUG A unfixed: 'per_bar_threshold = None' not found — "
            "UnboundLocalError will fire when per_symbol_thresholds=False"
        )

    def test_none_before_conditional(self, v5_train_src):
        """The None init must appear BEFORE the conditional block."""
        idx_none = v5_train_src.find("per_bar_threshold = None")
        idx_cond = v5_train_src.find("if config.per_symbol_thresholds")
        assert idx_none != -1, "'per_bar_threshold = None' not found"
        assert idx_cond != -1, "'if config.per_symbol_thresholds' not found"
        assert idx_none < idx_cond, (
            "per_bar_threshold = None must appear BEFORE 'if config.per_symbol_thresholds'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BUG B — Symmetric 3-class KL targets
# ═══════════════════════════════════════════════════════════════════════════

class TestSymmetric3ClsTargets:
    """Bug B: 3-class targets must be symmetric so bull/bear folds retain
    meaningful LONG AND SHORT representation (max ratio 2:1 not 3.7:1)."""

    def test_bull_target_symmetric(self, v5_train_src):
        """Bull target must be [0.25, 0.50, 0.25]."""
        assert "'bull': [0.25, 0.50, 0.25]" in v5_train_src, (
            "Bull 3-class target should be [0.25, 0.50, 0.25]; "
            "found old asymmetric value"
        )

    def test_bear_target_symmetric(self, v5_train_src):
        """Bear target must be [0.25, 0.25, 0.50]."""
        assert "'bear': [0.25, 0.25, 0.50]" in v5_train_src, (
            "Bear 3-class target should be [0.25, 0.25, 0.50]; "
            "found old value"
        )

    def test_chop_target_near_symmetric(self, v5_train_src):
        """Chop target must be [0.35, 0.325, 0.325]."""
        assert "'chop': [0.35, 0.325, 0.325]" in v5_train_src, (
            "Chop 3-class target should be [0.35, 0.325, 0.325]"
        )

    def test_old_asymmetric_bull_removed(self, v5_train_src):
        """Old bull target [0.30, 0.55, 0.15] must be gone."""
        assert "'bull': [0.30, 0.55, 0.15]" not in v5_train_src, (
            "Old asymmetric bull target still present — collapse risk remains"
        )

    def test_old_asymmetric_bear_removed(self, v5_train_src):
        """Old bear target [0.30, 0.15, 0.55] must be gone."""
        assert "'bear': [0.30, 0.15, 0.55]" not in v5_train_src, (
            "Old asymmetric bear target still present"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BUG B (cont.) — Entropy regularisation in compute_v5_loss signature
# ═══════════════════════════════════════════════════════════════════════════

class TestEntropyRegularisation:
    """compute_v5_loss must accept and apply action_entropy_weight."""

    def test_entropy_param_in_signature(self, v5_train_src):
        """action_entropy_weight=0.0 must appear in compute_v5_loss signature."""
        assert "action_entropy_weight=0.0" in v5_train_src, (
            "compute_v5_loss must have action_entropy_weight=0.0 default in signature"
        )

    def test_l_entropy_key_added_to_losses(self, v5_train_src):
        """losses dict must contain 'L_entropy' key."""
        assert "'L_entropy'" in v5_train_src, (
            "'L_entropy' key must be added to losses dict in compute_v5_loss"
        )

    def test_entropy_formula_present(self, v5_train_src):
        """Mean-batch entropy formula must be present."""
        assert "action_probs.mean(dim=0)" in v5_train_src, (
            "Mean batch action probs computation missing from entropy regularisation"
        )

    def test_entropy_guarded_by_weight_check(self, v5_train_src):
        """Entropy regularisation must be guarded by 'if action_entropy_weight > 0.0'."""
        assert "if action_entropy_weight > 0.0" in v5_train_src, (
            "Entropy computation must be skipped when weight=0.0 for backward compat"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available in this environment")
    def test_entropy_zero_weight_gives_zero_l_entropy(self):
        """action_entropy_weight=0.0 → L_entropy=0 in losses dict."""
        from train.v5_train import compute_v5_loss
        torch.manual_seed(42)
        loss, ld = compute_v5_loss(_dummy_outputs(), _dummy_batch(),
                                   action_entropy_weight=0.0)
        assert ld.get("L_entropy") == 0.0, (
            f"L_entropy should be 0.0 at weight=0, got {ld.get('L_entropy')}"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available in this environment")
    def test_entropy_nonzero_weight_gives_nonzero_l_entropy(self):
        """action_entropy_weight=0.10 → non-zero L_entropy in losses."""
        from train.v5_train import compute_v5_loss
        torch.manual_seed(1)
        _, ld = compute_v5_loss(_dummy_outputs(), _dummy_batch(),
                                action_entropy_weight=0.10)
        assert ld.get("L_entropy") != 0.0, "L_entropy should be non-zero at weight=0.10"

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available in this environment")
    def test_entropy_is_nonpositive(self):
        """L_entropy = sum(p*log(p)) ≤ 0 (negentropy)."""
        from train.v5_train import compute_v5_loss
        torch.manual_seed(2)
        _, ld = compute_v5_loss(_dummy_outputs(), _dummy_batch(),
                                action_entropy_weight=0.10)
        assert ld["L_entropy"] <= 0.0, (
            f"L_entropy must be ≤ 0, got {ld['L_entropy']:.4f}"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available in this environment")
    def test_uniform_logits_max_entropy(self):
        """Uniform action logits → L_entropy ≈ -log(3) (maximum entropy)."""
        from train.v5_train import compute_v5_loss
        torch.manual_seed(3)
        n = 128
        outputs = _dummy_outputs(n)
        outputs["action_logits"] = torch.zeros(n, 3)  # correct key
        batch = _dummy_batch(n)
        _, ld = compute_v5_loss(outputs, batch, action_entropy_weight=0.10)
        expected = -math.log(3)
        assert abs(ld["L_entropy"] - expected) < 0.05, (
            f"Expected L_entropy ≈ {expected:.4f}, got {ld['L_entropy']:.4f}"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available in this environment")
    def test_entropy_gradient_pushes_toward_uniform(self):
        """Gradient of entropy loss w.r.t. collapsed logits should push
        the dominant class DOWN and minority classes UP (toward uniform).

        This verifies the gradient direction is correct, not just sign of loss.
        A logit of +10 for LONG → softmax ≈ [ε, 1, ε].
        Gradient of L_entropy = Σ p*log(p) w.r.t. logits should be negative
        for the dominant class (LONG, index 1) so that updating logits with
        -lr * grad DECREASES the LONG logit (pushes toward uniform).
        """
        from train.v5_train import compute_v5_loss
        torch.manual_seed(77)
        n = 32
        outputs = _dummy_outputs(n)
        logits = torch.full((n, 3), -10.0, requires_grad=True)
        # Manually set dominant logit via a separate leaf
        _base = torch.full((n, 3), -10.0)
        _base[:, 1] = 10.0  # LONG collapsed
        logits_leaf = _base.clone().detach().requires_grad_(True)
        outputs["action_logits"] = logits_leaf
        batch = _dummy_batch(n)

        loss, ld = compute_v5_loss(outputs, batch, action_entropy_weight=1.0)
        loss.backward()

        # Gradient for LONG class (index 1) should be negative:
        # updating with -lr * grad → logit decreases → less collapsed
        grad_long = logits_leaf.grad[:, 1].mean().item()
        assert grad_long < 0, (
            f"Entropy gradient for dominant LONG class should be negative "
            f"(push toward uniform), got {grad_long:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BUG C — Hardcoded SIDE_BAL_W=0.05 log fix
# ═══════════════════════════════════════════════════════════════════════════

class TestSideBalLogFix:
    """Bug C: SIDE_BAL_W log must use the variable, not hardcode '0.05'."""

    def test_hardcoded_0_05_removed(self, v5_train_src):
        """'SIDE_BAL_W=0.05' literal must not appear in source."""
        assert "SIDE_BAL_W=0.05" not in v5_train_src, (
            "Hardcoded 'SIDE_BAL_W=0.05' still present — misleads reviewers; "
            "actual weight is 0.15"
        )

    def test_fstring_variable_used(self, v5_train_src):
        """Log must use '{side_bal_weight:.2f}' f-string."""
        assert "SIDE_BAL_W={side_bal_weight:.2f}" in v5_train_src, (
            "SIDE_BAL_W log must use f-string variable 'side_bal_weight:.2f'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE D — [V5_DIR_COLLAPSE] warning
# ═══════════════════════════════════════════════════════════════════════════

class TestDirCollapseWarning:
    """Feature D: [V5_DIR_COLLAPSE] warning fires at BOTH batch and epoch level
    when >90% of predictions are one direction."""

    def test_warning_tag_in_source(self, v5_train_src):
        """[V5_DIR_COLLAPSE] warning must be present in v5_train.py."""
        assert "[V5_DIR_COLLAPSE]" in v5_train_src, (
            "[V5_DIR_COLLAPSE] warning tag not found in v5_train.py"
        )

    def test_epoch_level_collapse_threshold_90_pct(self, v5_train_src):
        """Epoch-level collapse threshold must be 0.90 (90%)."""
        assert "_collapse_threshold = 0.90" in v5_train_src, (
            "Epoch-level collapse threshold should be 0.90 (90%)"
        )

    def test_all_three_directions_covered(self, v5_train_src):
        """Warning must cover LONG, SHORT, and HOLD collapse."""
        assert "collapsed to LONG" in v5_train_src, "LONG collapse branch missing"
        assert "collapsed to SHORT" in v5_train_src, "SHORT collapse branch missing"
        assert "collapsed to HOLD" in v5_train_src, "HOLD collapse branch missing"

    def test_batch_level_detection_present(self, v5_train_src):
        """Batch-level [V5_DIR_COLLAPSE] detection must exist in training loop."""
        assert "_batch_collapse_warned" in v5_train_src, (
            "_batch_collapse_warned flag missing — batch-level detection not implemented"
        )

    def test_batch_collapse_warned_reset_each_epoch(self, v5_train_src):
        """_batch_collapse_warned must be reset to False at start of each epoch."""
        assert "_batch_collapse_warned = False" in v5_train_src, (
            "_batch_collapse_warned = False reset not found; must be reset per epoch"
        )

    def test_batch_detection_checks_valid_rows(self, v5_train_src):
        """Batch-level detection must filter on valid rows (not all rows)."""
        assert "_batch_valid" in v5_train_src, (
            "Batch detection must use batch_gpu.get('valid') to check valid rows only"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE E — CLI flag + wiring
# ═══════════════════════════════════════════════════════════════════════════

class TestCLIFlagWiring:
    """Feature E: --v5-action-entropy-weight must be wired from CLI through to
    compute_v5_loss."""

    def test_cli_flag_in_quick_start(self, quick_start_src):
        """--v5-action-entropy-weight must exist in quick_start.py."""
        assert "--v5-action-entropy-weight" in quick_start_src, (
            "--v5-action-entropy-weight CLI flag not found in quick_start.py"
        )

    def test_cli_default_is_0_10(self, quick_start_src):
        """Default must be 0.10 in the CLI add_argument call."""
        assert "default=0.10" in quick_start_src or "default=0.1" in quick_start_src, (
            "--v5-action-entropy-weight default must be 0.10 in quick_start.py"
        )

    def test_wired_to_walk_forward(self, quick_start_src):
        """action_entropy_weight=args.v5_action_entropy_weight must appear in call sites."""
        assert "action_entropy_weight=args.v5_action_entropy_weight" in quick_start_src, (
            "action_entropy_weight not wired from CLI args to walk_forward/train call"
        )

    def test_walk_forward_signature(self, v5_train_src):
        """run_v5_walk_forward must have action_entropy_weight parameter."""
        # Look for the parameter in the function definition context
        assert "action_entropy_weight=0.10" in v5_train_src, (
            "run_v5_walk_forward or train_v5_model missing action_entropy_weight=0.10 default"
        )

    def test_entropy_weight_passed_to_compute_loss(self, v5_train_src):
        """action_entropy_weight must be forwarded to compute_v5_loss call."""
        assert "action_entropy_weight=action_entropy_weight" in v5_train_src, (
            "action_entropy_weight not forwarded to compute_v5_loss call sites"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
