"""v5_correlation.py — Cross-asset correlation tracking & reliability-aware size penalty.

v5.0.8: Provides RollingDailyCorr for computing rolling daily R-unit correlations
across symbols and CorrBlocker for gating correlated same-direction entries.

Fix (Task #30): The old binary block (corr >= threshold → block) never fired in 23
walk-forward folds because there was insufficient aligned daily-R history to build a
reliable estimate.  Instead of a hard block the module now applies a multiplicative
size penalty only when correlation *and* overlap reliability conditions are met:
  - aligned_days >= min_aligned_days (data reliability gate)
  - overlap_ratio >= min_overlap_ratio (concurrent-position gate)
  - |corr| >= threshold         → strong penalty  (corr_penalty_strong, default 0.35)
  - |corr| >= threshold * 0.70  → moderate penalty (corr_penalty_moderate, default 0.65)
  - |corr| < threshold * 0.70   → no penalty (multiplier 1.0)
  - Unavailable / unreliable data → mild default conservative multiplier (default 0.90)
All decisions are logged with aligned_days, overlap_ratio, corr_value, reliability_flag,
and action_taken so they appear in fold logs and the corr report.
"""

import logging
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CorrConfig:
    enabled: bool = True
    window_days: int = 30
    threshold: float = 0.70
    same_side_only: bool = True
    log_matrix: bool = True
    min_aligned_days: int = 10
    max_block: int = 5                    # kept for back-compat; no longer used as hard cap
    min_overlap_ratio: float = 0.05       # min concurrent-position overlap to apply penalty
    corr_penalty_strong: float = 0.35     # size multiplier when |corr| >= threshold
    corr_penalty_moderate: float = 0.65   # size multiplier when |corr| >= threshold * 0.70
    corr_unavailable_mult: float = 0.90   # mild conservative default when data unreliable


class RollingDailyCorr:
    """Tracks daily R per symbol and computes rolling pairwise Pearson correlation."""

    def __init__(self, symbols: List[str], window_days: int = 30,
                 min_aligned_days: int = 10):
        self.symbols = sorted(symbols)
        self.window_days = window_days
        self.min_aligned_days = min_aligned_days
        self.daily_r: Dict[str, Dict[str, float]] = {s: {} for s in self.symbols}

    def record_trade(self, symbol: str, date_str: str, r_value: float):
        if symbol not in self.daily_r:
            return
        if np.isnan(r_value):
            return
        self.daily_r[symbol][date_str] = self.daily_r[symbol].get(date_str, 0.0) + r_value

    def get_aligned_daily_series(self, sym_a: str, sym_b: str,
                                 max_days: Optional[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        dates_a = set(self.daily_r.get(sym_a, {}).keys())
        dates_b = set(self.daily_r.get(sym_b, {}).keys())
        common = sorted(dates_a & dates_b)
        if max_days is not None and len(common) > max_days:
            common = common[-max_days:]
        if len(common) == 0:
            return np.array([]), np.array([])
        series_a = np.array([self.daily_r[sym_a][d] for d in common])
        series_b = np.array([self.daily_r[sym_b][d] for d in common])
        return series_a, series_b

    def pairwise_corr(self, sym_a: str, sym_b: str,
                      max_days: Optional[int] = None) -> Optional[float]:
        sa, sb = self.get_aligned_daily_series(sym_a, sym_b, max_days)
        if len(sa) < self.min_aligned_days:
            return None
        std_a, std_b = np.std(sa), np.std(sb)
        if std_a < 1e-12 or std_b < 1e-12:
            return 0.0
        corr = float(np.corrcoef(sa, sb)[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr

    def correlation_matrix(self, max_days: Optional[int] = None
                           ) -> Tuple[np.ndarray, List[str]]:
        n = len(self.symbols)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                c = self.pairwise_corr(self.symbols[i], self.symbols[j], max_days)
                if c is not None:
                    mat[i, j] = c
                    mat[j, i] = c
        return mat, self.symbols

    def get_daily_r_compact(self) -> Dict[str, Dict[str, float]]:
        return {s: dict(sorted(d.items())) for s, d in self.daily_r.items()}


class CorrBlocker:
    """Applies a reliability-aware size penalty when cross-asset correlation is high.

    Old behaviour (hard block, Task #30):
        corr >= threshold  →  block trade entirely

    New behaviour (size penalty):
        Penalty is only applied when BOTH reliability conditions are met:
          1. aligned_days >= config.min_aligned_days  (enough history)
          2. overlap_ratio >= config.min_overlap_ratio (concurrent positions exist)
        Then the worst (lowest) multiplier found across all open-position pairs is used:
          |corr| >= threshold        →  corr_penalty_strong  (default 0.35)
          |corr| >= threshold * 0.70 →  corr_penalty_moderate (default 0.65)
          |corr| < threshold * 0.70  →  1.0  (no penalty)
        When data is unavailable / unreliable: corr_unavailable_mult (default 0.90).

    should_block() is kept for back-compat but always returns False; callers should
    use compute_size_penalty() instead and multiply into their size multiplier.
    """

    def __init__(self, corr_tracker: RollingDailyCorr, config: CorrConfig,
                 overlap_ratio: float = 0.0):
        self.tracker = corr_tracker
        self.config = config
        self.overlap_ratio = overlap_ratio          # updated externally per fold
        self.blocked_count = 0                     # hard-block counter (should_block path)
        self.penalty_count = 0
        self.block_log: List[Dict] = []

    def on_position_closed(self, symbol: str):
        pass

    def should_block(self, symbol: str, side: int,
                     open_positions: Dict[str, int]) -> bool:
        """Hard-block gate: returns True when any open position is same-side (if
        same_side_only=True) with |corr| >= threshold AND there are enough aligned days.

        This is the classic binary block used in training-time sweep filtering.
        For live/fold-level sizing, use compute_size_penalty() which applies a
        proportional multiplier instead of a full block.
        """
        if not self.config.enabled:
            return False
        for other_sym, other_side in open_positions.items():
            if other_sym == symbol:
                continue
            if self.config.same_side_only and side != other_side:
                continue
            corr = self.tracker.pairwise_corr(symbol, other_sym, self.config.window_days)
            if corr is None:
                # Insufficient aligned data — no block, not reliable enough
                continue
            if abs(corr) >= self.config.threshold:
                self.blocked_count += 1
                self.block_log.append({
                    'symbol': symbol, 'other': other_sym,
                    'corr': corr, 'side': side,
                })
                return True
        return False

    def compute_size_penalty(self, symbol: str, side: int,
                             open_positions: Dict[str, int]) -> float:
        """Return the worst-case size multiplier for `symbol` given open positions.

        Returns 1.0 when no penalty applies (no positions, no reliable correlation).
        Caller multiplies this into their size multiplier *before* the floor clamp.
        """
        if not self.config.enabled:
            return 1.0
        if len(self.tracker.symbols) < 2 or not open_positions:
            return 1.0

        worst_mult = 1.0

        for other_sym, other_side in open_positions.items():
            if other_sym == symbol:
                continue
            if self.config.same_side_only and side != other_side:
                continue

            sa, sb = self.tracker.get_aligned_daily_series(symbol, other_sym,
                                                           self.config.window_days)
            aligned_days = len(sa)
            overlap_ok = self.overlap_ratio >= self.config.min_overlap_ratio
            reliable = (aligned_days >= self.config.min_aligned_days and overlap_ok)

            corr = self.tracker.pairwise_corr(symbol, other_sym, self.config.window_days)

            if not reliable or corr is None:
                mult = self.config.corr_unavailable_mult
                reliability_flag = "UNRELIABLE"
                action_taken = f"default_conservative mult={mult:.2f}"
            else:
                abs_corr = abs(corr)
                moderate_thresh = self.config.threshold * 0.70
                if abs_corr >= self.config.threshold:
                    mult = self.config.corr_penalty_strong
                    action_taken = f"STRONG_PENALTY mult={mult:.2f}"
                elif abs_corr >= moderate_thresh:
                    mult = self.config.corr_penalty_moderate
                    action_taken = f"MODERATE_PENALTY mult={mult:.2f}"
                else:
                    mult = 1.0
                    action_taken = "NO_PENALTY"
                reliability_flag = "RELIABLE"

            if mult < worst_mult:
                worst_mult = mult
                entry = {
                    'symbol': symbol, 'side': side,
                    'other_sym': other_sym, 'other_side': other_side,
                    'aligned_days': aligned_days,
                    'overlap_ratio': round(self.overlap_ratio, 4),
                    'corr_value': round(corr, 4) if corr is not None else None,
                    'reliability_flag': reliability_flag,
                    'action_taken': action_taken,
                    'size_mult': round(mult, 4),
                }
                self.block_log.append(entry)
                if mult < 1.0:
                    self.penalty_count += 1
                    log.info(
                        "[V5_CORR_PENALTY] %s side=%+d vs %s: aligned_days=%d "
                        "overlap=%.3f corr=%.3f reliability=%s action=%s",
                        symbol, side, other_sym, aligned_days, self.overlap_ratio,
                        corr if corr is not None else float('nan'),
                        reliability_flag, action_taken,
                    )

        return worst_mult


def compute_overlap_ratio(trade_bars: Dict[str, List[Tuple[int, int]]],
                          total_bars: int) -> float:
    if total_bars <= 0 or len(trade_bars) < 2:
        return 0.0

    occupied = np.zeros(total_bars, dtype=np.int32)
    for sym, spans in trade_bars.items():
        for start, end in spans:
            s = max(0, start)
            e = min(total_bars, end)
            if s < e:
                occupied[s:e] += 1

    overlap_bars = int(np.sum(occupied >= 2))
    return overlap_bars / total_bars


def build_fold_corr_report(fold_id: int, window_train: str, window_test: str,
                           corr_tracker: RollingDailyCorr,
                           overlap_ratio: float,
                           blocker: Optional[CorrBlocker] = None) -> Dict:
    mat, syms = corr_tracker.correlation_matrix(max_days=corr_tracker.window_days)

    upper = []
    max_abs = 0.0
    max_pair = ("", "")
    n = len(syms)
    for i in range(n):
        for j in range(i + 1, n):
            v = mat[i, j]
            if not np.isnan(v):
                upper.append(abs(v))
                if abs(v) > max_abs:
                    max_abs = abs(v)
                    max_pair = (syms[i], syms[j])

    mean_abs = float(np.mean(upper)) if upper else 0.0

    report = {
        'fold_id': fold_id,
        'window_train': window_train,
        'window_test': window_test,
        'symbols': syms,
        'corr_matrix': [[round(float(v), 4) if not np.isnan(v) else None
                          for v in row] for row in mat],
        'mean_abs_corr': round(mean_abs, 4),
        'max_abs_corr': round(max_abs, 4),
        'max_abs_corr_pair': list(max_pair),
        'overlap_ratio': round(overlap_ratio, 4),
        'daily_r_series': corr_tracker.get_daily_r_compact(),
    }
    if blocker is not None:
        report['corr_blocked_trades'] = blocker.blocked_count
        report['corr_penalty_trades'] = blocker.penalty_count
        report['corr_penalty_log'] = blocker.block_log[:50]
        report['corr_block_log'] = blocker.block_log[:50]

    return report


def log_corr_report(report: Dict, fold_id: int):
    log.info("=" * 60)
    log.info("[V5_CORR] Fold %d — Cross-Asset Correlation Report", fold_id)
    log.info("-" * 60)

    syms = report['symbols']
    mat = report['corr_matrix']
    header = f"{'':>12}" + "".join(f"{s:>12}" for s in syms)
    log.info(header)
    for i, s in enumerate(syms):
        row_str = f"{s:>12}"
        for j in range(len(syms)):
            v = mat[i][j]
            if v is None:
                row_str += f"{'N/A':>12}"
            else:
                row_str += f"{v:>12.4f}"
        log.info(row_str)

    log.info("-" * 60)
    log.info("[V5_CORR] mean_abs_corr=%.4f  max_abs_corr=%.4f (%s/%s)",
             report['mean_abs_corr'], report['max_abs_corr'],
             report['max_abs_corr_pair'][0], report['max_abs_corr_pair'][1])
    log.info("[V5_CORR] overlap_ratio=%.4f", report['overlap_ratio'])
    if 'corr_blocked_trades' in report:
        log.info("[V5_CORR] legacy_blocked_trades=%d  penalty_trades=%d",
                 report.get('corr_blocked_trades', 0), report.get('corr_penalty_trades', 0))
    log.info("=" * 60)


def save_corr_report(report: Dict, fold_id: int, output_dir: str = "checkpoints"):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"v5_corr_report_fold_{fold_id}.json"
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info("[V5_CORR] Report saved to %s", out)
