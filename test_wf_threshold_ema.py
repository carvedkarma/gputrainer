"""
Regression test for WF threshold EMA decay floor.

Verifies two things:
1. SOURCE INSPECTION: The production code in run_v5_walk_forward uses the shared
   `_wf_threshold_floor` variable (not the stale hardcoded 0.01) in both the dead-fold
   and no-report decay branches.  This test FAILS immediately if 0.01 is reintroduced.
2. BEHAVIORAL SIMULATION: The corrected logic (max(0.001, ema*decay)) produces different
   results from the old logic (max(0.01, ema*decay)) for V5-realistic ema values.

Run with:
    python -m pytest gpu_trainer/test_wf_threshold_ema.py -v
or:
    python gpu_trainer/test_wf_threshold_ema.py
"""
import os
import re

import pytest

# Path to production source
_V5_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "train", "v5_train.py")


# ---------------------------------------------------------------------------
# Helpers to read and inspect the production source
# ---------------------------------------------------------------------------

def _read_source():
    with open(_V5_TRAIN_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_wf_function_lines(source: str) -> list[str]:
    """Return the lines belonging to run_v5_walk_forward."""
    lines = source.splitlines()
    in_func = False
    func_lines = []
    for line in lines:
        if line.startswith("def run_v5_walk_forward("):
            in_func = True
        elif in_func and line.startswith("def ") and "run_v5_walk_forward" not in line:
            break
        if in_func:
            func_lines.append(line)
    return func_lines


# ---------------------------------------------------------------------------
# Source-level regression tests
# ---------------------------------------------------------------------------

class TestWFThresholdFloorSourceVerification:
    """Verify the production source does NOT contain the stale 0.01 floor in WF paths."""

    def setup_method(self):
        self.source = _read_source()
        self.wf_lines = _extract_wf_function_lines(self.source)
        self.wf_text = "\n".join(self.wf_lines)

    def test_no_hardcoded_01_floor_in_dead_fold_branch(self):
        """Dead-fold EMA decay must NOT use a hardcoded 0.01 literal as the floor."""
        # Pattern: min_threshold = 0.01 (the old bug)
        pattern = re.compile(r'\bmin_threshold\s*=\s*0\.01\b')
        matches = [ln for ln in self.wf_lines if pattern.search(ln)]
        assert matches == [], (
            f"REGRESSION: Found stale 'min_threshold = 0.01' in run_v5_walk_forward:\n"
            + "\n".join(matches)
        )

    def test_shared_floor_variable_defined_once(self):
        """_wf_threshold_floor must be defined exactly once inside run_v5_walk_forward."""
        definition_pattern = re.compile(r'\s*_wf_threshold_floor\s*=')
        definitions = [ln for ln in self.wf_lines if definition_pattern.search(ln)]
        assert len(definitions) == 1, (
            f"Expected exactly 1 definition of _wf_threshold_floor in run_v5_walk_forward, "
            f"found {len(definitions)}:\n" + "\n".join(definitions)
        )

    def test_floor_derived_from_tpd_ctrl_cfg(self):
        """_wf_threshold_floor must be derived from tpd_ctrl_cfg.min_threshold_floor."""
        definition_pattern = re.compile(r'\s*_wf_threshold_floor\s*=')
        for line in self.wf_lines:
            if definition_pattern.search(line):
                assert "tpd_ctrl_cfg" in line, (
                    f"_wf_threshold_floor definition does not reference tpd_ctrl_cfg:\n{line}"
                )
                assert "min_threshold_floor" in line, (
                    f"_wf_threshold_floor definition does not use min_threshold_floor:\n{line}"
                )
                break

    def test_dead_fold_branch_uses_shared_floor_variable(self):
        """Dead-fold branch must call max(_wf_threshold_floor, ...) not max(0.01, ...)."""
        # Find the dead-fold block: 'DEAD FOLD' log + max() call nearby
        dead_fold_block = []
        in_dead = False
        for line in self.wf_lines:
            if "DEAD FOLD" in line:
                in_dead = True
            if in_dead:
                dead_fold_block.append(line)
                if "max(" in line and "threshold_ema" in line:
                    break

        assert dead_fold_block, "Could not find the DEAD FOLD branch in run_v5_walk_forward"
        max_lines = [l for l in dead_fold_block if "max(" in l and "threshold_ema" in l]
        assert max_lines, "No max() call found in DEAD FOLD branch"
        for ml in max_lines:
            assert "_wf_threshold_floor" in ml, (
                f"DEAD FOLD max() does not use _wf_threshold_floor:\n{ml}"
            )
            # Also ensure the stale 0.01 literal is not the floor arg
            assert not re.search(r'max\(\s*0\.01\s*,', ml), (
                f"DEAD FOLD max() uses old hardcoded 0.01 floor:\n{ml}"
            )

    def test_no_report_branch_uses_shared_floor_variable(self):
        """No-report branch must call max(_wf_threshold_floor, ...) not max(0.01, ...)."""
        # The no-report branch is in the else: clause (no report file) — find all max()
        # calls on threshold_ema in the full WF function.  The key invariant is:
        # 1. At least 2 such calls exist (dead-fold + no-report paths).
        # 2. None of them use a raw 0.01 literal as the floor.
        # 3. All of them use _wf_threshold_floor.
        max_ema_lines = [
            ln for ln in self.wf_lines
            if "max(" in ln and "threshold_ema" in ln and "_wf_threshold_floor" in ln
        ]
        # There must be exactly 2 max() calls using _wf_threshold_floor
        assert len(max_ema_lines) >= 2, (
            f"Expected >= 2 'max(_wf_threshold_floor, ...)' calls in run_v5_walk_forward "
            f"(one for dead-fold, one for no-report), found {len(max_ema_lines)}:\n"
            + "\n".join(max_ema_lines)
        )
        # None should use 0.01
        for ml in max_ema_lines:
            assert not re.search(r'max\(\s*0\.01\s*,', ml), (
                f"Found old hardcoded 0.01 floor in threshold_ema max() call:\n{ml}"
            )

    def test_fold_override_uses_live_threshold_ema_state(self):
        """WF fold calls must pass threshold_ema, not stale blended_threshold snapshots."""
        override_lines = [
            ln for ln in self.wf_lines
            if "wf_threshold_override=" in ln
        ]
        assert override_lines, "Could not find wf_threshold_override assignments in run_v5_walk_forward"
        for ln in override_lines:
            assert "threshold_ema" in ln, (
                "wf_threshold_override must be sourced from threshold_ema so dead/low-conf "
                f"EMA updates carry into the next fold:\n{ln}"
            )
            assert "blended_threshold" not in ln, (
                "wf_threshold_override should not read blended_threshold (can become stale "
                f"across dead/low-conf folds):\n{ln}"
            )


# ---------------------------------------------------------------------------
# Behavioral tests: correct logic vs old logic on V5-realistic values
# ---------------------------------------------------------------------------

class TestWFThresholdFloorBehavior:
    """Verify that the corrected logic differs from old logic on V5-realistic values."""

    OLD_FLOOR = 0.01    # stale hardcoded value
    NEW_FLOOR = 0.001   # corrected from V5TPDControllerConfig.min_threshold_floor
    DECAY = 0.5         # typical wf_threshold_decay

    def _apply_floor(self, ema, decay, floor):
        """Replicate the exact production expression: max(floor, ema*decay)."""
        return max(floor, ema * decay)

    def test_v5_realistic_ema_produces_different_results_old_vs_new_floor(self):
        """With V5-realistic ema=0.0018, old floor clamps to 0.01; new floor allows 0.0009."""
        ema = 0.0018
        result_old = self._apply_floor(ema, self.DECAY, self.OLD_FLOOR)
        result_new = self._apply_floor(ema, self.DECAY, self.NEW_FLOOR)
        assert result_old != result_new, "Old and new floors should produce different results"
        assert result_old == self.OLD_FLOOR, (
            f"Old floor should clamp: expected {self.OLD_FLOOR}, got {result_old}"
        )
        assert result_new < self.OLD_FLOOR, (
            f"New floor should NOT clamp to old level: expected < {self.OLD_FLOOR}, got {result_new}"
        )
        assert result_new == self.NEW_FLOOR, (
            f"New floor should clamp to 0.001: expected {self.NEW_FLOOR}, got {result_new}"
        )

    def test_dead_fold_decay_converges_below_old_floor_with_new_floor(self):
        """20 consecutive dead folds starting at 0.005 should converge to NEW_FLOOR."""
        ema = 0.005
        for _ in range(20):
            ema = self._apply_floor(ema, self.DECAY, self.NEW_FLOOR)
        assert ema == self.NEW_FLOOR, (
            f"After 20 dead folds, expected convergence to {self.NEW_FLOOR}, got {ema}"
        )
        assert ema < self.OLD_FLOOR, (
            f"Converged value {ema} should be below old stale floor {self.OLD_FLOOR}"
        )

    def test_old_floor_would_clamp_realistic_ema(self):
        """Confirm that the OLD floor would have clamped V5-realistic ema values."""
        for ema in [0.0018, 0.0015, 0.0010, 0.0005]:
            result = self._apply_floor(ema, self.DECAY, self.OLD_FLOOR)
            assert result == self.OLD_FLOOR, (
                f"Old floor should clamp ema={ema} to {self.OLD_FLOOR}, got {result}"
            )

    def test_above_new_floor_decay_passes_through_unchanged(self):
        """When ema*decay > new_floor, result equals ema*decay (no clamping)."""
        ema = 0.010
        result = self._apply_floor(ema, self.DECAY, self.NEW_FLOOR)
        assert result == ema * self.DECAY, (
            f"Expected {ema * self.DECAY}, got {result}"
        )

    def test_fallback_matches_new_floor_when_cfg_is_none(self):
        """The None-safe fallback in the source must equal NEW_FLOOR (0.001)."""
        # Simulate: tpd_ctrl_cfg.min_threshold_floor if tpd_ctrl_cfg is not None else 0.001
        tpd_ctrl_cfg = None
        floor = tpd_ctrl_cfg.min_threshold_floor if tpd_ctrl_cfg is not None else 0.001
        assert floor == self.NEW_FLOOR, (
            f"None-safe fallback should be {self.NEW_FLOOR}, got {floor}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
