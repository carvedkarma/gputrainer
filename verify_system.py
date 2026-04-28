#!/usr/bin/env python3
"""
System Verification Module (v4.5.2)
====================================
Runs N cycles of assertions to prove that the Triple-Lane Aggression Engine,
CROSS correlation blocking, quota controller, payload completeness, HTF warmup,
OI integration, trade recording gating, and PR-AUC pack wiring are all
wired correctly end-to-end.

Usage:
    python quick_start.py --live --verify-system --cycles 30 --url <URL>
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

log = logging.getLogger("Verify")


@dataclass
class VerifyFailure:
    cycle: int
    category: str
    message: str


@dataclass
class VerifyStats:
    total_cycles: int = 0
    lane_decisions: Dict[str, int] = field(default_factory=lambda: {"CORE": 0, "FLOW": 0, "SCALP": 0, "HOLD": 0, "COOLDOWN": 0})
    cross_blocks: int = 0
    cross_allows_opposite: int = 0
    cross_checks_total: int = 0
    quota_steps_seen: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    flow_thr_values: List[float] = field(default_factory=list)
    core_thr_values: List[float] = field(default_factory=list)
    payload_checks: int = 0
    payload_passes: int = 0
    net_check_passes: int = 0
    net_check_violations: int = 0
    failures: List[VerifyFailure] = field(default_factory=list)


REQUIRED_CYCLE_PAYLOAD_KEYS = [
    "lane_selected", "htf_score", "hold_reason",
    "quota_step", "core_thr", "flow_thr", "scalp_thr",
    "lane_size_mult", "lane_budget_remaining_r",
]


class SystemVerifier:
    """Hooks into the LiveRunner to verify system invariants."""

    def __init__(self, max_cycles: int = 30):
        self.max_cycles = max_cycles
        self.stats = VerifyStats()
        self.cycle_payloads: List[dict] = []

    def verify_lane_routing(self, cycle: int, lane_result: dict, p_enter: float,
                            htf_score: int, htf: dict) -> List[VerifyFailure]:
        """Assert lane routing priority: CORE > FLOW > SCALP > HOLD."""
        failures = []
        lane = lane_result.get('lane_selected', 'UNKNOWN')
        self.stats.lane_decisions[lane] = self.stats.lane_decisions.get(lane, 0) + 1

        range_ok = htf.get('range_ok', False)
        momentum_ok = lane_result.get('momentum_ok', False)
        volatility_ok = lane_result.get('volatility_ok', False)
        core_thr = lane_result.get('core_thr')
        flow_thr = lane_result.get('flow_thr')
        scalp_thr = lane_result.get('scalp_thr')

        if core_thr is not None:
            self.stats.core_thr_values.append(core_thr)
        if flow_thr is not None:
            self.stats.flow_thr_values.append(flow_thr)

        if lane == 'HOLD':
            hold_reason = lane_result.get('hold_reason', '')
            if not hold_reason:
                failures.append(VerifyFailure(
                    cycle, "LANE_ROUTING",
                    f"HOLD decision has empty hold_reason"
                ))

        if lane == 'FLOW':
            if htf_score >= 3 and range_ok and core_thr is not None and p_enter >= core_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"FLOW selected but CORE conditions met: htf={htf_score} range_ok={range_ok} "
                    f"p_enter={p_enter:.4f} >= core_thr={core_thr:.4f}"
                ))

        if lane == 'SCALP':
            if htf_score >= 3 and range_ok and core_thr is not None and p_enter >= core_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"SCALP selected but CORE conditions met: htf={htf_score} range_ok={range_ok} "
                    f"p_enter={p_enter:.4f} >= core_thr={core_thr:.4f}"
                ))
            if htf_score >= 2 and flow_thr is not None and p_enter >= flow_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"SCALP selected but FLOW conditions met: htf={htf_score} "
                    f"p_enter={p_enter:.4f} >= flow_thr={flow_thr:.4f}"
                ))

        return failures

    def verify_cross_blocking(self, cycle: int, symbol: str, side: str,
                              blocked: bool, blocked_by: Optional[str],
                              existing_positions: Dict) -> List[VerifyFailure]:
        """Assert CROSS blocks only same-direction correlated exposure."""
        failures = []
        self.stats.cross_checks_total += 1

        if blocked and blocked_by:
            self.stats.cross_blocks += 1
            if blocked_by in existing_positions:
                existing_side = existing_positions[blocked_by].side
                if existing_side != side:
                    failures.append(VerifyFailure(
                        cycle, "CROSS_BLOCK",
                        f"CROSS blocked {symbol} {side} with {blocked_by} {existing_side} -- "
                        f"opposite-direction should NOT be blocked!"
                    ))

        if not blocked:
            for pos_sym, pos in existing_positions.items():
                if pos.side != side:
                    self.stats.cross_allows_opposite += 1

        return failures

    def verify_payload(self, cycle: int, payload: dict) -> List[VerifyFailure]:
        """Assert all required keys exist in cycle payload (the actual POST payload dict).
        
        For HOLD decisions, thresholds/size_mult/budget fields may legitimately be None
        (e.g., NEUTRAL_SIDE hold). For ENTER decisions, all fields must be populated.
        """
        failures = []
        self.stats.payload_checks += 1

        lane = payload.get('lane_selected')
        non_hold_keys = ["core_thr", "flow_thr", "scalp_thr", "lane_size_mult", "lane_budget_remaining_r"]
        always_required = [k for k in REQUIRED_CYCLE_PAYLOAD_KEYS if k not in non_hold_keys]

        if lane in ('HOLD', None):
            actual_missing = [k for k in always_required if k not in payload or payload[k] is None]
        else:
            actual_missing = [k for k in REQUIRED_CYCLE_PAYLOAD_KEYS if k not in payload or payload[k] is None]

        if actual_missing:
            failures.append(VerifyFailure(
                cycle, "PAYLOAD_MISSING",
                f"Cycle payload missing keys: {actual_missing} (lane={payload.get('lane_selected')})"
            ))
        else:
            self.stats.payload_passes += 1

        self.cycle_payloads.append(payload)
        return failures

    def verify_net_r(self, cycle: int, gross_r: float, cost_r: float, net_r: float) -> List[VerifyFailure]:
        """Assert net_r == gross_r - cost_r within tolerance."""
        failures = []
        expected = gross_r - cost_r
        diff = abs(net_r - expected)
        if diff > 1e-6:
            self.stats.net_check_violations += 1
            failures.append(VerifyFailure(
                cycle, "NET_CHECK",
                f"net_r={net_r:.6f} != gross_r={gross_r:.6f} - cost_r={cost_r:.6f} (diff={diff:.8f})"
            ))
        else:
            self.stats.net_check_passes += 1
        return failures

    def verify_flow_threshold_stepping(self) -> List[VerifyFailure]:
        """Assert FLOW threshold differs from CORE threshold at least once."""
        failures = []
        if len(self.stats.core_thr_values) > 0 and len(self.stats.flow_thr_values) > 0:
            min_len = min(len(self.stats.core_thr_values), len(self.stats.flow_thr_values))
            any_different = any(
                abs(self.stats.flow_thr_values[i] - self.stats.core_thr_values[i]) > 1e-6
                for i in range(min_len)
            )
            if not any_different:
                failures.append(VerifyFailure(
                    0, "FLOW_STEPPING",
                    f"flow_thr NEVER differed from core_thr across {min_len} matched cycles. "
                    f"FLOW quota stepping may not be working."
                ))
        return failures

    def record_quota_step(self, quota_step: int):
        """Track quota step distribution."""
        self.stats.quota_steps_seen[quota_step] = self.stats.quota_steps_seen.get(quota_step, 0) + 1

    def add_failure(self, failure: VerifyFailure):
        self.stats.failures.append(failure)

    def add_failures(self, failures: List[VerifyFailure]):
        self.stats.failures.extend(failures)

    def generate_report(self) -> str:
        """Generate the final verification report."""
        s = self.stats
        all_failures = s.failures.copy()
        all_failures.extend(self.verify_flow_threshold_stepping())

        passed = len(all_failures) == 0
        status = "PASSED" if passed else "FAILED"

        lines = []
        lines.append("# System Verification Report (v4.5.2)")
        lines.append(f"")
        lines.append(f"## Overall Status: **VERIFICATION {status}**")
        lines.append(f"")
        lines.append(f"Cycles run: {s.total_cycles}")
        lines.append(f"Total failures: {len(all_failures)}")
        lines.append(f"")

        lines.append("## Summary Table")
        lines.append("")
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")

        lane_ok = not any(f.category == "LANE_PRIORITY" for f in all_failures)
        lines.append(f"| Lane Router Priority (CORE>FLOW>SCALP>HOLD) | {'PASS' if lane_ok else 'FAIL'} |")

        hold_ok = not any(f.category == "LANE_ROUTING" for f in all_failures)
        lines.append(f"| HOLD includes hold_reason | {'PASS' if hold_ok else 'FAIL'} |")

        cross_ok = not any(f.category == "CROSS_BLOCK" for f in all_failures)
        lines.append(f"| CROSS blocks same-dir only | {'PASS' if cross_ok else 'FAIL'} |")

        flow_step_ok = not any(f.category == "FLOW_STEPPING" for f in all_failures)
        lines.append(f"| FLOW threshold stepping | {'PASS' if flow_step_ok else 'FAIL'} |")

        payload_ok = not any(f.category == "PAYLOAD_MISSING" for f in all_failures)
        lines.append(f"| Cycle payload completeness | {'PASS' if payload_ok else 'FAIL'} |")

        net_ok = not any(f.category == "NET_CHECK" for f in all_failures)
        lines.append(f"| net_r accounting invariant | {'PASS' if net_ok else 'FAIL'} |")

        lines.append(f"")
        lines.append("## Lane Decision Distribution")
        lines.append(f"")
        for lane, count in sorted(s.lane_decisions.items()):
            lines.append(f"- {lane}: {count}")
        lines.append(f"")

        lines.append("## CROSS Blocking Stats")
        lines.append(f"- Total CROSS checks: {s.cross_checks_total}")
        lines.append(f"- Same-direction blocks: {s.cross_blocks}")
        lines.append(f"- Opposite-direction allows: {s.cross_allows_opposite}")
        lines.append(f"")

        lines.append("## Quota Step Distribution")
        lines.append(f"")
        for step, count in sorted(s.quota_steps_seen.items()):
            lines.append(f"- Step {step}: {count} cycles")
        lines.append(f"")

        lines.append("## Payload Checks")
        lines.append(f"- Total checks: {s.payload_checks}")
        lines.append(f"- Passes: {s.payload_passes}")
        lines.append(f"- Required keys: {REQUIRED_CYCLE_PAYLOAD_KEYS}")
        lines.append(f"")

        lines.append("## Net R Accounting")
        lines.append(f"- Passes: {s.net_check_passes}")
        lines.append(f"- Violations: {s.net_check_violations}")
        lines.append(f"")

        if s.flow_thr_values and s.core_thr_values:
            lines.append("## Threshold Samples (first 5)")
            lines.append(f"")
            for i in range(min(5, len(s.flow_thr_values))):
                ct = s.core_thr_values[i] if i < len(s.core_thr_values) else "?"
                ft = s.flow_thr_values[i]
                lines.append(f"- Cycle ~{i}: core_thr={ct} flow_thr={ft}")
            lines.append(f"")

        if all_failures:
            lines.append("## Failures (first 10)")
            lines.append(f"")
            for f in all_failures[:10]:
                lines.append(f"- [Cycle {f.cycle}] **{f.category}**: {f.message}")
            lines.append(f"")

        lines.append("---")
        lines.append(f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

        return "\n".join(lines)

    def save_report(self, path: str = "verify_report.md"):
        report = self.generate_report()
        with open(path, 'w') as f:
            f.write(report)
        log.info(f"Verification report saved to {path}")
        print("\n" + report)
        return report


def run_static_audit() -> str:
    """Phase 1: Static code audit -- check key functions exist and are wired correctly."""
    lines = []
    lines.append("## Phase 1: Static Code Audit")
    lines.append("")

    import inspect
    from live_runner import LiveRunner, _compute_htf_score, _compute_momentum_ok, _compute_scalp_gates
    from live_runner import LANE_BUDGET, FLOW_QUOTA_STEPS, SCALP_HORIZON, SCALP_TP_R, SCALP_SL_R, SCALP_SIZE_MULT
    from portfolio import PortfolioManager, Position

    lines.append("### Lane Router (`_select_lane` in live_runner.py)")
    src = inspect.getsource(LiveRunner._select_lane)
    has_core_check = "htf_score >= 3" in src and "range_ok" in src
    has_flow_check = "htf_score >= 2" in src
    has_scalp_check = "htf_score >= 1" in src and ("volatility_ok" in src or "vol_expansion_ok" in src) and "momentum_ok" in src
    has_ordered_routing = src.index("CORE") < src.index("FLOW") < src.index("SCALP")
    lines.append(f"- CORE check (htf>=3 + range_ok): {'FOUND' if has_core_check else 'MISSING'}")
    lines.append(f"- FLOW check (htf>=2): {'FOUND' if has_flow_check else 'MISSING'}")
    lines.append(f"- SCALP check (htf>=1 + vol + mom): {'FOUND' if has_scalp_check else 'MISSING'}")
    lines.append(f"- Ordered routing CORE>FLOW>SCALP: {'CORRECT' if has_ordered_routing else 'WRONG'}")
    lines.append(f"- Uses distinct thresholds per lane: {'YES' if 'core_thr' in src and 'flow_thr' in src and 'scalp_thr' in src else 'NO'}")
    lines.append("")

    lines.append("### HTF Score (`_compute_htf_score` in live_runner.py)")
    htf_src = inspect.getsource(_compute_htf_score)
    lines.append(f"- Scores h1_trend match: {'YES' if 'h1' in htf_src else 'NO'}")
    lines.append(f"- Scores h4_trend match: {'YES' if 'h4' in htf_src else 'NO'}")
    lines.append(f"- Scores slope_ok: {'YES' if 'slope_ok' in htf_src else 'NO'}")
    lines.append("")

    lines.append("### CROSS Blocking (`_correlated_block` + `can_enter` in portfolio.py)")
    cross_src = inspect.getsource(PortfolioManager._correlated_block)
    can_enter_src = inspect.getsource(PortfolioManager.can_enter)
    lines.append(f"- Checks same-direction only: {'YES' if 'self.open_positions[partner].side == side' in cross_src else 'NO'}")
    lines.append(f"- Does NOT block opposite: {'CORRECT' if 'side == side' in cross_src else 'NEEDS CHECK'}")
    lines.append(f"- Checks BEFORE placing: {'YES' if 'blocked_by' in can_enter_src else 'NO'}")
    lines.append(f"- Logs specific reason: {'YES' if 'CROSS_BLOCK' in can_enter_src else 'NO'}")
    lines.append("")

    lines.append("### Quota Controller (in `_select_lane`)")
    lines.append(f"- quota_step computed: {'YES' if 'quota_step' in src else 'NO'}")
    lines.append(f"- FLOW_QUOTA_STEPS defined: {json.dumps({k: v for k, v in FLOW_QUOTA_STEPS.items()})}")
    lines.append(f"- Flow percentile stepping: {'YES' if 'flow_pct' in src else 'NO'}")
    lines.append(f"- Daily budget reset: {'YES' if '_reset_daily_budget_if_needed' in inspect.getsource(LiveRunner._process_symbol) else 'NO'}")
    lines.append("")

    lines.append("### SCALP Geometry Constants")
    lines.append(f"- SCALP_HORIZON: {SCALP_HORIZON}")
    lines.append(f"- SCALP_TP_R: {SCALP_TP_R}")
    lines.append(f"- SCALP_SL_R: {SCALP_SL_R}")
    lines.append(f"- SCALP_SIZE_MULT: {SCALP_SIZE_MULT}")
    lines.append("")

    lines.append("### Daily R Budgets")
    lines.append(f"- LANE_BUDGET: {json.dumps(LANE_BUDGET)}")
    lines.append("")

    lines.append("### Position Close Callback")
    close_src = inspect.getsource(PortfolioManager.close_position)
    lines.append(f"- on_close_callback called: {'YES' if 'on_close_callback' in close_src else 'NO'}")
    lines.append(f"- dashboard_trade_id checked: {'YES' if 'dashboard_trade_id' in close_src else 'NO'}")
    lines.append("")

    lines.append("### SCALP Time-Stop")
    exit_src = inspect.getsource(Position.check_exit)
    lines.append(f"- TIME_EXIT for SCALP: {'YES' if 'TIME_EXIT' in exit_src else 'NO'}")
    lines.append(f"- horizon check: {'YES' if 'self.horizon' in exit_src else 'NO'}")
    lines.append("")

    lines.append("### SCALP v4.5 Separation Gates (`_compute_scalp_gates`)")
    scalp_src = inspect.getsource(_compute_scalp_gates)
    lines.append(f"- ATR14/ATR50 ratio check: {'YES' if 'atr_ratio' in scalp_src else 'NO'}")
    lines.append(f"- true_range_z check: {'YES' if 'true_range_z' in scalp_src else 'NO'}")
    lines.append(f"- bb_width_z check: {'YES' if 'bb_width_z' in scalp_src else 'NO'}")
    lines.append(f"- ema20_slope check: {'YES' if 'ema20_slope' in scalp_src else 'NO'}")
    lines.append(f"- volume_ratio check: {'YES' if 'volume_ratio' in scalp_src else 'NO'}")
    lines.append(f"- macd_hist check: {'YES' if 'macd_hist' in scalp_src else 'NO'}")
    lines.append("")

    lines.append("### HTF Warmup Gate")
    process_src = inspect.getsource(LiveRunner._process_symbol)
    has_warmup = "WARMUP" in process_src and ("MIN_H1_BARS" in process_src or "min_h1" in process_src.lower())
    lines.append(f"- WARMUP gate in _process_symbol: {'FOUND' if has_warmup else 'MISSING'}")
    from live_runner import MIN_H1_BARS, MIN_H4_BARS
    lines.append(f"- MIN_H1_BARS={MIN_H1_BARS} MIN_H4_BARS={MIN_H4_BARS}")
    lines.append("")

    lines.append("### Trade Recording Gate (--record-trades)")
    exec_src = inspect.getsource(LiveRunner._execute_candidate)
    has_record_gate = "record_trades" in exec_src
    has_no_exec_reason = "RECORD_TRADES_OFF" in exec_src
    lines.append(f"- record_trades gate in _execute_candidate: {'FOUND' if has_record_gate else 'MISSING'}")
    lines.append(f"- RECORD_TRADES_OFF decision logged: {'YES' if has_no_exec_reason else 'NO'}")
    cycle_src = inspect.getsource(LiveRunner._run_cycle)
    has_cycle_gate = "record_trades" in cycle_src
    lines.append(f"- record_trades gate in _run_cycle (exits): {'FOUND' if has_cycle_gate else 'MISSING'}")
    lines.append("")

    lines.append("### OI Integration (v2)")
    from quick_start import fetch_open_interest_hist, compute_oi_features, _oi_sanity_check
    oi_fetch_src = inspect.getsource(fetch_open_interest_hist)
    oi_compute_src = inspect.getsource(compute_oi_features)
    has_oi_fetch_log = "[OI_FETCH]" in oi_fetch_src
    has_oi_cache_log = "[OI_CACHE]" in oi_fetch_src
    has_oi_align_log = "[OI_ALIGN]" in oi_compute_src
    has_tolerance = "tolerance" in oi_compute_src
    has_nonzero_filter = "oi_nonzero" in oi_compute_src or "sumOpenInterest'] > 0" in oi_compute_src
    has_pagination = "page" in oi_fetch_src and "current_start" in oi_fetch_src
    lines.append(f"- [OI_FETCH] log: {'FOUND' if has_oi_fetch_log else 'MISSING'}")
    lines.append(f"- [OI_CACHE] log: {'FOUND' if has_oi_cache_log else 'MISSING'}")
    lines.append(f"- [OI_ALIGN] log: {'FOUND' if has_oi_align_log else 'MISSING'}")
    lines.append(f"- merge_asof tolerance (2x period): {'YES' if has_tolerance else 'NO'}")
    lines.append(f"- Zero OI rows filtered: {'YES' if has_nonzero_filter else 'NO'}")
    lines.append(f"- Pagination over 30d window: {'YES' if has_pagination else 'NO'}")
    oi_sanity_src = inspect.getsource(_oi_sanity_check)
    has_coverage_threshold = "coverage_threshold" in oi_sanity_src
    lines.append(f"- OI auto-disable with coverage threshold: {'YES' if has_coverage_threshold else 'NO'}")
    lines.append("")

    lines.append("### PR-AUC Pack Wiring")
    from quick_start import FEATURE_VERSION
    lines.append(f"- FEATURE_VERSION: {FEATURE_VERSION}")
    try:
        from quick_start import train_enter_model
        train_src = inspect.getsource(train_enter_model)
        has_focal = "focal_bce_with_logits" in train_src or "focal_loss" in train_src
        has_sep_loss = "sep_loss" in train_src
        has_flip_penalty = "flip_penalty" in train_src
        has_safety_log = "[SAFETY]" in train_src
        has_sep_check = "[SEP_CHECK]" in train_src
        lines.append(f"- Focal BCE loss: {'ACTIVE' if has_focal else 'MISSING'}")
        lines.append(f"- Separation regularizer: {'ACTIVE' if has_sep_loss else 'MISSING'}")
        lines.append(f"- Flip penalty (v4.5.2): {'ACTIVE' if has_flip_penalty else 'MISSING'}")
        lines.append(f"- [SAFETY] per-epoch log: {'ACTIVE' if has_safety_log else 'MISSING'}")
        lines.append(f"- [SEP_CHECK] per-epoch log: {'ACTIVE' if has_sep_check else 'MISSING'}")
    except Exception as e:
        lines.append(f"- ERROR inspecting train_enter_model: {e}")
    lines.append("")

    lines.append("### Version Consistency")
    try:
        from quick_start import FEATURE_VERSION, SYSTEM_VERSION
        version_match = FEATURE_VERSION == SYSTEM_VERSION
        lines.append(f"- FEATURE_VERSION: {FEATURE_VERSION}")
        lines.append(f"- SYSTEM_VERSION: {SYSTEM_VERSION}")
        lines.append(f"- Versions match: {'YES' if version_match else 'NO'}")
    except Exception as e:
        lines.append(f"- ERROR checking versions: {e}")
    lines.append("")

    return "\n".join(lines)


@dataclass
class SeparationStats:
    total_cycles: int = 0
    scalp_entries: int = 0
    scalp_gate_pass: int = 0
    scalp_gate_fail: int = 0
    core_entries: int = 0
    flow_entries: int = 0
    hold_entries: int = 0
    scalp_with_range_ok: int = 0
    scalp_without_range_ok: int = 0
    scalp_size_halved: int = 0
    budget_violations: int = 0
    budget_checks: int = 0
    exit_resolve_logs: int = 0
    priority_violations: int = 0
    vol_expansion_rates: List[bool] = field(default_factory=list)
    momentum_rates: List[bool] = field(default_factory=list)
    atr_ratios: List[float] = field(default_factory=list)
    vol_ratios: List[float] = field(default_factory=list)
    failures: List[VerifyFailure] = field(default_factory=list)


class SeparationVerifier:
    """v4.5 Separation Verification -- checks SCALP gate enforcement,
    router priority, budget bounds, and exit resolve logs."""

    def __init__(self, max_cycles: int = 200):
        self.max_cycles = max_cycles
        self.stats = SeparationStats()

    def verify_cycle(self, cycle: int, lane_result: dict, p_enter: float,
                     htf_score: int, htf: dict, scalp_gates: dict,
                     budgets: dict) -> List[VerifyFailure]:
        failures = []
        self.stats.total_cycles += 1
        lane = lane_result.get('lane_selected', 'HOLD')

        if lane == 'CORE':
            self.stats.core_entries += 1
        elif lane == 'FLOW':
            self.stats.flow_entries += 1
        elif lane == 'SCALP':
            self.stats.scalp_entries += 1
        else:
            self.stats.hold_entries += 1

        vol_expansion_ok = scalp_gates.get('vol_expansion_ok', False)
        momentum_ok = scalp_gates.get('momentum_ok', False)
        self.stats.vol_expansion_rates.append(vol_expansion_ok)
        self.stats.momentum_rates.append(momentum_ok)
        if scalp_gates.get('atr_ratio') is not None:
            self.stats.atr_ratios.append(scalp_gates['atr_ratio'])
        if scalp_gates.get('vol_ratio') is not None:
            self.stats.vol_ratios.append(scalp_gates['vol_ratio'])

        if lane == 'SCALP':
            if not vol_expansion_ok:
                self.stats.scalp_gate_fail += 1
                failures.append(VerifyFailure(
                    cycle, "SCALP_GATE",
                    f"SCALP entered without vol_expansion_ok=True "
                    f"(atr_ratio={scalp_gates.get('atr_ratio', 0):.4f})"
                ))
            if not momentum_ok:
                self.stats.scalp_gate_fail += 1
                failures.append(VerifyFailure(
                    cycle, "SCALP_GATE",
                    f"SCALP entered without momentum_ok=True "
                    f"(ema20_slope={scalp_gates.get('ema20_slope')}, "
                    f"vol_ratio={scalp_gates.get('vol_ratio')})"
                ))

            if vol_expansion_ok and momentum_ok:
                self.stats.scalp_gate_pass += 1

            range_ok = htf.get('range_ok', False)
            if range_ok:
                self.stats.scalp_with_range_ok += 1
            else:
                self.stats.scalp_without_range_ok += 1

            size_mult = lane_result.get('lane_size_mult', 0)
            if not range_ok and size_mult > 0.126:
                failures.append(VerifyFailure(
                    cycle, "SCALP_SIZE_HALVING",
                    f"SCALP size_mult={size_mult} should be <=0.125 when range_ok=False"
                ))
            elif not range_ok and size_mult <= 0.126:
                self.stats.scalp_size_halved += 1

        core_thr = lane_result.get('core_thr')
        flow_thr = lane_result.get('flow_thr')
        range_ok = htf.get('range_ok', False)

        if lane == 'SCALP':
            if htf_score >= 3 and range_ok and core_thr and p_enter >= core_thr:
                self.stats.priority_violations += 1
                failures.append(VerifyFailure(
                    cycle, "PRIORITY",
                    f"SCALP selected but CORE eligible: htf={htf_score} range_ok p_enter={p_enter:.4f}>={core_thr:.4f}"
                ))
            if htf_score >= 2 and flow_thr and p_enter >= flow_thr:
                self.stats.priority_violations += 1
                failures.append(VerifyFailure(
                    cycle, "PRIORITY",
                    f"SCALP selected but FLOW eligible: htf={htf_score} p_enter={p_enter:.4f}>={flow_thr:.4f}"
                ))

        self.stats.budget_checks += 1
        for lane_name in ['CORE', 'FLOW', 'SCALP']:
            remaining = budgets.get(f'{lane_name.lower()}_remaining', None)
            if remaining is not None and remaining < -0.01:
                self.stats.budget_violations += 1
                failures.append(VerifyFailure(
                    cycle, "BUDGET",
                    f"{lane_name} budget negative: {remaining:.4f}R"
                ))

        self.stats.failures.extend(failures)
        return failures

    def record_exit_resolve(self):
        self.stats.exit_resolve_logs += 1

    def generate_report(self) -> str:
        s = self.stats
        passed = len(s.failures) == 0
        status = "PASSED" if passed else "FAILED"

        vol_rate = sum(1 for v in s.vol_expansion_rates if v) / max(len(s.vol_expansion_rates), 1) * 100
        mom_rate = sum(1 for v in s.momentum_rates if v) / max(len(s.momentum_rates), 1) * 100
        avg_atr = sum(s.atr_ratios) / max(len(s.atr_ratios), 1)
        avg_vol = sum(s.vol_ratios) / max(len(s.vol_ratios), 1)

        lines = [
            "# Separation Verification Report (v4.5)",
            "",
            f"## Overall Status: **VERIFICATION {status}**",
            "",
            f"Cycles run: {s.total_cycles}",
            f"Total failures: {len(s.failures)}",
            "",
            "## Summary Table",
            "",
            "| Check | Status |",
            "|-------|--------|",
            f"| SCALP gate enforcement (vol_expansion + momentum) | {'PASS' if not any(f.category == 'SCALP_GATE' for f in s.failures) else 'FAIL'} |",
            f"| Router priority (CORE > FLOW > SCALP) | {'PASS' if s.priority_violations == 0 else 'FAIL'} |",
            f"| Budget bounds (no negative) | {'PASS' if s.budget_violations == 0 else 'FAIL'} |",
            f"| SCALP size halving (range_ok=False -> 0.125x) | {'PASS' if not any(f.category == 'SCALP_SIZE_HALVING' for f in s.failures) else 'FAIL'} |",
            f"| Exit resolve logs present | {'PASS' if s.exit_resolve_logs > 0 else 'WARN (none seen)'} |",
            "",
            "## Lane Distribution",
            "",
            f"- CORE: {s.core_entries}",
            f"- FLOW: {s.flow_entries}",
            f"- SCALP: {s.scalp_entries} (gate_pass={s.scalp_gate_pass}, gate_fail={s.scalp_gate_fail})",
            f"- HOLD: {s.hold_entries}",
            "",
            "## SCALP Gate Metrics",
            "",
            f"- Vol Expansion OK rate: {vol_rate:.1f}%",
            f"- Momentum OK rate: {mom_rate:.1f}%",
            f"- Avg ATR ratio: {avg_atr:.3f} (min required: 1.20)",
            f"- Avg Volume ratio: {avg_vol:.3f} (min required: 1.20)",
            f"- SCALP with range_ok: {s.scalp_with_range_ok}",
            f"- SCALP without range_ok (size halved): {s.scalp_size_halved}",
            "",
            "## Budget & Exit",
            "",
            f"- Budget checks: {s.budget_checks}",
            f"- Budget violations: {s.budget_violations}",
            f"- Exit resolve logs captured: {s.exit_resolve_logs}",
            "",
        ]

        if s.failures:
            lines.append("## Failures (first 15)")
            lines.append("")
            for f in s.failures[:15]:
                lines.append(f"- [Cycle {f.cycle}] **{f.category}**: {f.message}")
            lines.append("")

        lines.append("---")
        lines.append(f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        return "\n".join(lines)

    def save_report(self, path: str = "verify_report_v4.5.md"):
        report = self.generate_report()
        with open(path, 'w') as f:
            f.write(report)
        log.info(f"Separation verification report saved to {path}")
        print("\n" + report)
        return report
