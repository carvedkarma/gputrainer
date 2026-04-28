"""Drawdown-Adaptive Throttle (DDT) for V5 Forward Test.

Keeps model weights + base threshold frozen but adapts entry strictness
and sizing gradually based on recent realized performance.

When recent realized R is negative (chop/regime shift), progressively:
  1) Raises the effective entry threshold (fewer trades)
  2) Reduces position size (lower risk)
When performance recovers, relaxes slowly (hysteresis).

Deterministic, testable, OFF by default. No online learning / no weight updates.
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class DDTConfig:
    enabled: bool = False
    lookback_trades: int = 60
    bad_rollr: float = 6.0
    thr_k: float = 0.60
    thr_min: float = 0.08
    thr_max: float = 0.25
    size_k: float = 0.70
    min_size_mult: float = 0.25
    alpha_down: float = 0.30
    alpha_up: float = 0.05
    warmup_trades: int = 20


class DrawdownAdaptiveThrottle:
    """Gradually tightens entry threshold and reduces position size during
    drawdowns, then slowly relaxes as performance recovers.

    State:
      - closed_r_history: deque of realized R values (maxlen=lookback_trades)
      - throttle_state: smoothed throttle value in [0, 1]

    Integration:
      - After all existing gates pass, before final score >= threshold check:
          thr_used = ddt.effective_threshold(thr_base)
      - After base sizing is computed:
          size_final = size_base * ddt.size_multiplier()
      - On each closed trade:
          ddt.update_on_trade_close(realized_r)
    """

    def __init__(self, config: DDTConfig):
        self.config = config
        self.closed_r_history: deque = deque(maxlen=config.lookback_trades)
        self.throttle_state: float = 0.0
        self.ddt_blocked: int = 0
        self.thr_used_history: List[float] = []
        self.size_mult_history: List[float] = []
        self._trade_log: List[dict] = []

    def update_on_trade_close(self, realized_r: float) -> None:
        if not np.isnan(realized_r):
            self.closed_r_history.append(realized_r)
            raw = self.compute_raw_throttle()
            self.throttle_state = self._step_smooth(raw)

    def compute_raw_throttle(self) -> float:
        if len(self.closed_r_history) < self.config.warmup_trades:
            return 0.0
        rolling_sum = sum(self.closed_r_history)
        if rolling_sum >= 0:
            return 0.0
        if rolling_sum <= -self.config.bad_rollr:
            return 1.0
        return (-rolling_sum) / self.config.bad_rollr

    def _step_smooth(self, raw: float) -> float:
        if raw > self.throttle_state:
            return (1 - self.config.alpha_down) * self.throttle_state + self.config.alpha_down * raw
        else:
            return (1 - self.config.alpha_up) * self.throttle_state + self.config.alpha_up * raw

    def effective_threshold(self, thr_base: float) -> float:
        if not self.config.enabled:
            return thr_base
        thr_eff = thr_base * (1 + self.config.thr_k * self.throttle_state)
        thr_eff = max(self.config.thr_min, min(thr_eff, self.config.thr_max))
        self.thr_used_history.append(thr_eff)
        return thr_eff

    def size_multiplier(self) -> float:
        if not self.config.enabled:
            return 1.0
        m = 1.0 - self.config.size_k * self.throttle_state
        m = max(self.config.min_size_mult, min(m, 1.0))
        self.size_mult_history.append(m)
        return m

    def record_block(self):
        self.ddt_blocked += 1

    def log_trade_close(self, realized_r: float, thr_used: float, size_mult: float):
        rolling_sum = sum(self.closed_r_history)
        raw = self.compute_raw_throttle()
        entry = {
            'rolling_sum_R': round(rolling_sum, 4),
            'raw_throttle': round(raw, 4),
            'throttle_state': round(self.throttle_state, 4),
            'thr_used': round(thr_used, 4),
            'size_mult': round(size_mult, 4),
        }
        self._trade_log.append(entry)
        log.debug(f"[V5_DDT] trade_close: realized_r={realized_r:+.4f} "
                  f"rolling_sum_R={rolling_sum:+.4f} raw_throttle={raw:.4f} "
                  f"throttle_state={self.throttle_state:.4f} "
                  f"thr_used={thr_used:.4f} size_mult={size_mult:.4f}")

    def diagnostics(self) -> dict:
        """Compact diagnostics for report/logging."""
        return {
            'throttle_level': round(self.throttle_state, 4),
            'rolling_sum_r': round(sum(self.closed_r_history), 4) if self.closed_r_history else 0.0,
            'total_trades_seen': len(self.closed_r_history),
            'max_throttle_seen': round(max(
                [0.0] + [self._compute_peak_throttle()]), 4),
            'lookback_trades': self.config.lookback_trades,
            'ddt_blocked': self.ddt_blocked,
        }

    def _compute_peak_throttle(self) -> float:
        """Approximate peak throttle from trade log."""
        if not self._trade_log:
            return self.throttle_state
        return max(e.get('throttle_state', 0.0) for e in self._trade_log)

    def get_diagnostics(self) -> dict:
        thr_arr = np.array(self.thr_used_history) if self.thr_used_history else np.array([0.0])
        size_arr = np.array(self.size_mult_history) if self.size_mult_history else np.array([1.0])
        return {
            'ddt_enabled': self.config.enabled,
            'params': {
                'lookback_trades': self.config.lookback_trades,
                'bad_rollr': self.config.bad_rollr,
                'thr_k': self.config.thr_k,
                'thr_min': self.config.thr_min,
                'thr_max': self.config.thr_max,
                'size_k': self.config.size_k,
                'min_size_mult': self.config.min_size_mult,
                'alpha_down': self.config.alpha_down,
                'alpha_up': self.config.alpha_up,
                'warmup_trades': self.config.warmup_trades,
            },
            'ddt_blocked_trades': self.ddt_blocked,
            'thr_used_avg': float(np.mean(thr_arr)),
            'thr_used_min': float(np.min(thr_arr)),
            'thr_used_max': float(np.max(thr_arr)),
            'size_mult_avg': float(np.mean(size_arr)),
            'size_mult_min': float(np.min(size_arr)),
            'final_throttle_state': round(self.throttle_state, 4),
            'final_rolling_sum_R': round(sum(self.closed_r_history), 4) if self.closed_r_history else 0.0,
            'total_closed_trades': len(self.closed_r_history),
        }
