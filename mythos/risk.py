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
    day_trade_count: int = 0
    last_trade_bar: int = -10**9


class RiskConstitution:
    """
    Non-bypassable risk layer for Mythos execution.
    """

    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self.state = RiskState()
        self.realized_r_history: List[float] = []
        self._recent_high_conv_rr: List[float] = []
        self._recent_high_conv_max = int(max(getattr(cfg, "high_conviction_expectancy_window", 64), 8))
        self._equity_curve: List[float] = []

    def current_drawdown_r(self) -> float:
        return float(max(self.state.peak_equity_r - self.state.equity_r, 0.0))

    def should_disable_leverage(self) -> bool:
        dd = self.current_drawdown_r()
        return bool(dd >= float(max(getattr(self.cfg, "disable_leverage_drawdown_r", 6.0), 0.0)))

    def should_disable_conviction_boost(self) -> bool:
        dd = self.current_drawdown_r()
        return bool(dd >= float(max(getattr(self.cfg, "disable_conviction_boost_drawdown_r", 6.0), 0.0)))

    def should_stop_trading(self) -> bool:
        if not bool(getattr(self.cfg, "emergency_stop_enable", True)):
            return False
        dd = self.current_drawdown_r()
        dd_stop = float(max(getattr(self.cfg, "emergency_max_drawdown_r", 12.0), 0.0))
        eq_floor = float(getattr(self.cfg, "emergency_equity_floor_r", -18.0))
        return bool(dd >= dd_stop or self.state.equity_r <= eq_floor)

    def _to_bar_index(self, ts_ms: int | None = None, bar_index: int | None = None) -> int:
        if bar_index is not None:
            return int(bar_index)
        if ts_ms is None:
            raise ValueError("Either ts_ms or bar_index must be provided")
        # 15m bars
        return int(ts_ms // (15 * 60 * 1000))

    def _update_calendar(self, bar_index: int) -> None:
        # 15m bars -> 96 bars/day, 672 bars/week
        day_idx = int(bar_index // 96)
        week_idx = int(bar_index // 672)
        if day_idx != self.state.day_idx:
            self.state.day_idx = day_idx
            self.state.daily_r = 0.0
            self.state.day_trade_count = 0
        if week_idx != self.state.week_idx:
            self.state.week_idx = week_idx
            self.state.weekly_r = 0.0

    def allow_trade(
        self,
        bar_index: int | None = None,
        *,
        ts_ms: int | None = None,
        side: int = 0,
        edge: float = 0.0,
        uncertainty: float = 1.0,
    ) -> bool:
        _ = uncertainty
        if int(side) == 0:
            return False
        if float(edge) < float(self.cfg.min_edge_threshold):
            return False
        bar_index = self._to_bar_index(ts_ms=ts_ms, bar_index=bar_index)
        self._update_calendar(bar_index)
        if self.state.day_trade_count >= int(self.cfg.max_trades_per_day):
            return False
        if (bar_index - self.state.last_trade_bar) < int(self.cfg.cooldown_bars):
            return False
        if self.state.daily_r <= self.cfg.daily_loss_cap_r:
            return False
        if self.state.weekly_r <= self.cfg.weekly_loss_cap_r:
            return False
        if (self.state.equity_r - self.state.peak_equity_r) <= self.cfg.trailing_stop_r:
            return False
        return True

    def position_size_multiplier(
        self,
        edge: float,
        uncertainty: float,
        regime: int | None = None,
        conviction: float | None = None,
        high_conviction_recent: List[float] | None = None,
        allow_conviction_boost: bool = True,
    ) -> float:
        """
        Apply uncertainty-aware fractional sizing.
        """
        _ = regime
        safe_edge = max(float(edge), 0.0)
        safe_unc = max(float(uncertainty), 0.0)
        conf = 1.0 / (1.0 + safe_unc)
        size_mult = conf * (1.0 + safe_edge)
        dd = self.current_drawdown_r()
        dd_start = float(max(getattr(self.cfg, "drawdown_size_start_r", 6.0), 0.0))
        dd_max = float(max(getattr(self.cfg, "drawdown_size_full_r", 14.0), dd_start + 1e-6))
        min_frac = float(np.clip(getattr(self.cfg, "drawdown_size_min_scale", 0.35), 0.05, 1.0))
        if dd > dd_start:
            span = max(dd_max - dd_start, 1e-6)
            severity = float(np.clip((dd - dd_start) / span, 0.0, 1.0))
            throttle = float(1.0 - (1.0 - min_frac) * severity)
            size_mult *= throttle
        conv = float(np.clip(conviction if conviction is not None else conf, 0.0, 1.0))
        score_thr = float(np.clip(getattr(self.cfg, "conviction_score_threshold", 0.62), 0.0, 1.0))
        guard_window = int(max(getattr(self.cfg, "conviction_guard_window", 32), 1))
        guard_min_trades = int(max(getattr(self.cfg, "conviction_guard_min_trades", 8), 1))
        guard_min_expectancy = float(getattr(self.cfg, "conviction_guard_min_expectancy_r", 0.03))
        hist_src = high_conviction_recent if high_conviction_recent is not None else self._recent_high_conv_rr
        hist = [float(v) for v in hist_src[-guard_window:]] if hist_src else []
        hist_expectancy = float(np.mean(np.asarray(hist, dtype=np.float64))) if hist else 0.0
        hist_ready = len(hist) >= guard_min_trades and hist_expectancy >= guard_min_expectancy
        if allow_conviction_boost and (not self.should_disable_conviction_boost()) and conv >= score_thr and hist_ready:
            boost = float(np.clip(getattr(self.cfg, "conviction_boost", 0.35), 0.0, 2.0))
            span = max(1.0 - score_thr, 1e-6)
            gain = 1.0 + boost * ((conv - score_thr) / span)
            size_mult *= gain
        max_mult = float(max(getattr(self.cfg, "conviction_max_size_mult", self.cfg.max_size_mult), self.cfg.max_size_mult))
        size_mult = np.clip(size_mult, self.cfg.min_size_mult, max_mult)
        return float(size_mult)

    def high_conviction_expectancy(self) -> float:
        if not self._recent_high_conv_rr:
            return 0.0
        return float(np.mean(np.asarray(self._recent_high_conv_rr, dtype=np.float64)))

    def high_conviction_trade_count(self) -> int:
        return int(len(self._recent_high_conv_rr))

    def sized_r(self, expected_r: float, uncertainty: float) -> float:
        # Legacy alias retained for compatibility.
        return self.position_size_multiplier(edge=expected_r, uncertainty=uncertainty, regime=None)

    def record_trade(
        self,
        realized_r: float,
        ts_ms: int | None = None,
        *,
        edge: float | None = None,
        uncertainty: float | None = None,
        bar_index: int | None = None,
        conviction: float | None = None,
    ) -> None:
        _ = (edge, uncertainty)
        bar_index = self._to_bar_index(ts_ms=ts_ms, bar_index=bar_index)
        self._update_calendar(bar_index)
        self.realized_r_history.append(float(realized_r))
        if conviction is not None:
            conv = float(np.clip(conviction, 0.0, 1.0))
            thr = float(np.clip(getattr(self.cfg, "precision_high_conviction", 0.72), 0.0, 1.0))
            if conv >= thr:
                self._recent_high_conv_rr.append(float(realized_r))
                if len(self._recent_high_conv_rr) > self._recent_high_conv_max:
                    del self._recent_high_conv_rr[0 : len(self._recent_high_conv_rr) - self._recent_high_conv_max]
        self.state.daily_r += float(realized_r)
        self.state.weekly_r += float(realized_r)
        self.state.equity_r += float(realized_r)
        self.state.peak_equity_r = max(self.state.peak_equity_r, self.state.equity_r)
        self.state.day_trade_count += 1
        self.state.last_trade_bar = int(bar_index)

