from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import MythosConfig


@dataclass
class RiskState:
    daily_r: float = 0.0
    weekly_r: float = 0.0
    equity_r: float = 0.0
    peak_equity_r: float = 0.0
    day_idx: int = -1
    week_idx: int = -1


class RiskConstitution:
    """
    Non-bypassable risk layer for Mythos execution.
    """

    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self.state = RiskState()
        self.realized_r_history: List[float] = []

    def _update_calendar(self, bar_index: int) -> None:
        # 15m bars -> 96 bars/day, 672 bars/week
        day_idx = int(bar_index // 96)
        week_idx = int(bar_index // 672)
        if day_idx != self.state.day_idx:
            self.state.day_idx = day_idx
            self.state.daily_r = 0.0
        if week_idx != self.state.week_idx:
            self.state.week_idx = week_idx
            self.state.weekly_r = 0.0

    def allow_trade(self, bar_index: int) -> bool:
        self._update_calendar(bar_index)
        if self.state.daily_r <= self.cfg.daily_loss_cap_r:
            return False
        if self.state.weekly_r <= self.cfg.weekly_loss_cap_r:
            return False
        if (self.state.equity_r - self.state.peak_equity_r) <= self.cfg.trailing_stop_r:
            return False
        return True

    def sized_r(self, expected_r: float, uncertainty: float) -> float:
        """
        Apply uncertainty-aware fractional sizing.
        """
        edge = max(expected_r, 0.0)
        conf = 1.0 / max(1.0, 1.0 + uncertainty)
        size_mult = np.clip(conf * (1.0 + edge), self.cfg.min_size_mult, self.cfg.max_size_mult)
        return float(size_mult)

    def record_trade(self, realized_r: float) -> None:
        self.realized_r_history.append(float(realized_r))
        self.state.daily_r += float(realized_r)
        self.state.weekly_r += float(realized_r)
        self.state.equity_r += float(realized_r)
        self.state.peak_equity_r = max(self.state.peak_equity_r, self.state.equity_r)

