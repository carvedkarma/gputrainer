"""v5.0.8+ Adaptive Position Sizing & Dynamic Risk Scaling.

Provides:
  - AdaptivePositionSizer: Score-percentile-normalized sizing (Kelly used as quality modifier)
  - RegimeScaler:          ATR-ratio + EMA-trend + rolling-equity regime detection → risk multiplier
  - DailyLossTracker:      Per-day and per-symbol daily R budget enforcement
  - TrailingEquityStop:    Pauses trading when equity drops too far from peak
  - SizingDiagnostics:     Collects stats for fold-level reporting

Fix (Task #30): The old fractional-Kelly formula was mathematically pinned to min_size_mult
for all realistic WR/RR combinations (e.g. WR=30%, b=3 → kelly_f=0.067 → frac_kelly=0.017
→ clamped to 0.25).  The new approach uses score-percentile-normalized sizing:
  - Maintain a rolling buffer of observed scores (configurable window).
  - Compute the current score's percentile within that buffer.
  - Map percentile → [min_size_mult, max_size_mult] linearly.
  - Apply Kelly as a quality modifier: positive Kelly → mild boost (capped at 1.5×);
    negative Kelly → mild penalty (floored at 0.7×).
  - Log per-trade diagnostics including p_win, raw_kelly_f, score_pct, pre_floor_mult,
    and final_size_mult.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AdaptiveSizingConfig:
    enabled: bool = False
    kelly_fraction: float = 0.25       # kept for backward-compat; used only as quality hint
    max_size_mult: float = 2.5
    min_size_mult: float = 0.25
    score_buffer_size: int = 200       # rolling window for score-percentile estimation


@dataclass
class RegimeScalingConfig:
    enabled: bool = False
    bull_mult: float = 1.5
    bear_mult: float = 0.5
    lookback_trades: int = 20
    atr_lookback: int = 96
    atr_bull_ratio: float = 0.8
    atr_bear_ratio: float = 1.5
    min_equity_trades: int = 15
    low_confidence_dampen: float = 0.5


@dataclass
class LossManagementConfig:
    daily_loss_cap: Optional[float] = None
    trailing_equity_stop: Optional[float] = None
    per_symbol_daily_r_budget: Optional[float] = None


class AdaptivePositionSizer:
    """Score-percentile-normalized position sizing from model predictions.

    OLD FORMULA (broken — pinned to min_size_mult for all realistic WR values):
      kelly_f = (p * b - q) / b
      mult = kelly_fraction * kelly_f   →  ≈ 0.017 for WR=30%, b=3  →  clamped to 0.25

    NEW FORMULA (Task #30 fix):
      1. Maintain a rolling score buffer of the last `score_buffer_size` trades.
      2. Compute score_pct = percentile rank of the current score in that buffer.
         (A high-conviction trade with a top-decile score gets score_pct ≈ 0.90.)
      3. Map score_pct → raw size:
             pct_mult = min_size_mult + score_pct * (max_size_mult - min_size_mult)
      4. Compute raw Kelly for a quality modifier:
             raw_kelly_f = (p * b - q) / b
             kelly_quality = clip(1.0 + raw_kelly_f * 0.5, 0.70, 1.50)
      5. pre_floor_mult = pct_mult * kelly_quality
      6. Clamp to [min_size_mult, max_size_mult].

    Per-trade diagnostics are logged at DEBUG level; a fold summary is emitted
    by get_diagnostics() with min/mean/max/std of final_size_mult.
    """

    def __init__(self, config: AdaptiveSizingConfig):
        self.config = config
        self.sizing_history: List[float] = []
        self._score_buffer: List[float] = []
        self._trade_count: int = 0

    def compute_size_multiplier(self, score: float, p_win: float,
                                 mu_r: float, mfe: float, mae: float) -> float:
        if not self.config.enabled:
            return 1.0

        self._trade_count += 1

        # Conservative guard: a negative expected return means the trade has negative
        # expected value — no score percentile or Kelly quality modifier should amplify
        # it above the minimum. Return min_size_mult immediately so the history and
        # score buffer still reflect the call (trade_count already incremented above).
        if mu_r < 0:
            mult = self.config.min_size_mult
            self.sizing_history.append(mult)
            return mult

        p = max(min(p_win, 0.99), 0.01)
        q = 1.0 - p

        safe_mae = max(mae, 0.01)
        b = max(mfe / safe_mae, 0.01)

        raw_kelly_f = (p * b - q) / b

        self._score_buffer.append(score)
        if len(self._score_buffer) > self.config.score_buffer_size:
            self._score_buffer.pop(0)

        if len(self._score_buffer) >= 5:
            score_pct = float(np.mean([s <= score for s in self._score_buffer]))
        else:
            score_pct = 0.5

        pct_mult = (self.config.min_size_mult
                    + score_pct * (self.config.max_size_mult - self.config.min_size_mult))

        kelly_quality = float(np.clip(1.0 + raw_kelly_f * 0.5, 0.70, 1.50))

        pre_floor_mult = pct_mult * kelly_quality

        mult = float(np.clip(pre_floor_mult,
                             self.config.min_size_mult, self.config.max_size_mult))

        self.sizing_history.append(mult)

        if self._trade_count % 10 == 1 or mult > self.config.min_size_mult * 1.5:
            log.debug(
                "[SIZE_DIAG] trade=%d p_win=%.3f raw_kelly_f=%.4f score=%.4f "
                "score_pct=%.3f pct_mult=%.3f kelly_quality=%.3f "
                "pre_floor_mult=%.3f final_size_mult=%.3f",
                self._trade_count, p_win, raw_kelly_f, score,
                score_pct, pct_mult, kelly_quality, pre_floor_mult, mult,
            )

        return mult

    def log_fold_summary(self):
        """Emit a fold-level sizing summary to the INFO log."""
        if not self.sizing_history:
            log.info("[SIZE_FOLD] No sized trades this fold.")
            return
        arr = np.array(self.sizing_history)
        log.info(
            "[SIZE_FOLD] final_size_mult — trades=%d  min=%.3f  mean=%.3f  "
            "max=%.3f  std=%.3f  p10=%.3f  p90=%.3f  pct_above_1x=%.1f%%",
            len(arr), float(np.min(arr)), float(np.mean(arr)),
            float(np.max(arr)), float(np.std(arr)),
            float(np.percentile(arr, 10)), float(np.percentile(arr, 90)),
            float(np.mean(arr > 1.0) * 100),
        )

    def get_diagnostics(self) -> dict:
        if not self.sizing_history:
            return {
                'adaptive_sizing_enabled': self.config.enabled,
                'total_sized_trades': 0,
            }
        arr = np.array(self.sizing_history)
        return {
            'adaptive_sizing_enabled': self.config.enabled,
            'total_sized_trades': len(arr),
            'avg_size_mult': float(np.mean(arr)),
            'median_size_mult': float(np.median(arr)),
            'min_size_mult': float(np.min(arr)),
            'max_size_mult': float(np.max(arr)),
            'std_size_mult': float(np.std(arr)),
            'pct_above_1x': float(np.mean(arr > 1.0) * 100),
            'pct_below_1x': float(np.mean(arr < 1.0) * 100),
            'size_p10': float(np.percentile(arr, 10)),
            'size_p90': float(np.percentile(arr, 90)),
        }


class RegimeScaler:
    """Dynamic risk scaling based on market regime.

    Three signals combined (all normalized to [-0.5, +0.5]):
      1. ATR ratio:          current ATR vs rolling average ATR (volatility regime)
      2. EMA200 alignment:   is price trending with or against EMA? (trend regime)
      3. Rolling equity:     recent trade performance (model regime)

    Regime score ∈ [-0.5, +0.5]:  -0.5 = hostile, 0 = neutral, +0.5 = favorable
    Mapped to [bear_mult, bull_mult] via linear interpolation.

    Safety features:
      - ATR signal gated until full lookback window is available (no zero-padding)
      - Equity signal requires min_equity_trades (default 15) before activating
      - When fewer than 2 signals are active, multiplier deviation from 1.0 is
        dampened by low_confidence_dampen (default 0.5) to prevent single-signal extremes
    """

    def __init__(self, config: RegimeScalingConfig):
        self.config = config
        self.recent_r: List[float] = []
        self.regime_history: List[dict] = []

    def compute_regime_multiplier(self, idx: int, side: int,
                                   close_prices: Optional[np.ndarray] = None,
                                   ema200: Optional[np.ndarray] = None,
                                   atr_values: Optional[np.ndarray] = None) -> float:
        if not self.config.enabled:
            return 1.0

        signals = []

        if atr_values is not None and idx >= self.config.atr_lookback:
            window = atr_values[idx - self.config.atr_lookback:idx]
            if len(window) == self.config.atr_lookback:
                rolling_atr = float(np.mean(window))
                current_atr = float(atr_values[idx])
                if rolling_atr > 0 and not np.isnan(current_atr) and not np.isnan(rolling_atr):
                    atr_ratio = current_atr / rolling_atr
                    if atr_ratio <= self.config.atr_bull_ratio:
                        signals.append(0.5)
                    elif atr_ratio >= self.config.atr_bear_ratio:
                        signals.append(-0.5)
                    else:
                        mid = (self.config.atr_bull_ratio + self.config.atr_bear_ratio) / 2
                        rng = (self.config.atr_bear_ratio - self.config.atr_bull_ratio) / 2
                        signals.append(-0.5 * (atr_ratio - mid) / max(rng, 0.01))

        if close_prices is not None and ema200 is not None and idx < len(close_prices):
            close_val = close_prices[idx]
            ema_val = ema200[idx]
            if ema_val > 0:
                trend_strength = (close_val - ema_val) / ema_val
                trend_strength = max(-0.1, min(0.1, trend_strength))
                if side == 1:
                    signals.append(trend_strength * 5)
                else:
                    signals.append(-trend_strength * 5)

        if len(self.recent_r) >= self.config.min_equity_trades:
            recent = self.recent_r[-self.config.lookback_trades:]
            recent_arr = np.array(recent)
            mean_r = np.mean(recent_arr)
            std_r = np.std(recent_arr) + 1e-6
            rolling_sharpe = mean_r / std_r
            equity_signal = max(-0.5, min(0.5, rolling_sharpe * 0.5))
            signals.append(equity_signal)

        if not signals:
            return 1.0

        n_signals = len(signals)
        regime_score = float(np.mean(signals))
        regime_score = max(-0.5, min(0.5, regime_score))

        norm_score = regime_score * 2.0

        if norm_score >= 0:
            mult = 1.0 + norm_score * (self.config.bull_mult - 1.0)
        else:
            mult = 1.0 + norm_score * (1.0 - self.config.bear_mult)

        if n_signals < 2:
            deviation = mult - 1.0
            mult = 1.0 + deviation * self.config.low_confidence_dampen

        self.regime_history.append({
            'idx': idx,
            'regime_score': float(regime_score),
            'multiplier': float(mult),
            'n_signals': n_signals,
        })

        return float(mult)

    def record_trade_result(self, r_value: float):
        if not np.isnan(r_value):
            self.recent_r.append(r_value)

    def get_diagnostics(self) -> dict:
        if not self.regime_history:
            return {
                'regime_scaling_enabled': self.config.enabled,
                'total_regime_trades': 0,
            }
        scores = [h['regime_score'] for h in self.regime_history]
        mults = [h['multiplier'] for h in self.regime_history]
        scores_arr = np.array(scores)
        mults_arr = np.array(mults)
        return {
            'regime_scaling_enabled': self.config.enabled,
            'total_regime_trades': len(self.regime_history),
            'avg_regime_score': float(np.mean(scores_arr)),
            'avg_regime_mult': float(np.mean(mults_arr)),
            'pct_bull': float(np.mean(scores_arr > 0.2) * 100),
            'pct_bear': float(np.mean(scores_arr < -0.2) * 100),
            'pct_neutral': float(np.mean(np.abs(scores_arr) <= 0.2) * 100),
            'regime_mult_p10': float(np.percentile(mults_arr, 10)),
            'regime_mult_p90': float(np.percentile(mults_arr, 90)),
        }


class DailyLossTracker:
    """Enforces daily loss cap and per-symbol daily R budgets.

    - daily_loss_cap: if cumulative R for the day drops below this, skip rest of day
    - per_symbol_daily_r_budget: if any symbol's daily R drops below this, skip that symbol rest of day
    """

    def __init__(self, config: LossManagementConfig):
        self.config = config
        self.current_date: Optional[str] = None
        self.daily_r: float = 0.0
        self.daily_killed: bool = False
        self.symbol_daily_r: Dict[str, float] = {}
        self.symbol_daily_killed: Dict[str, bool] = {}

        self.days_killed: int = 0
        self.trades_blocked_daily: int = 0
        self.trades_blocked_symbol: int = 0
        self.symbol_kills: Dict[str, int] = {}

    def new_bar(self, date_str: str):
        if date_str != self.current_date:
            self.current_date = date_str
            self.daily_r = 0.0
            self.daily_killed = False
            self.symbol_daily_r.clear()
            self.symbol_daily_killed.clear()

    def should_block(self, symbol: Optional[str] = None) -> bool:
        if self.config.daily_loss_cap is not None and self.daily_killed:
            self.trades_blocked_daily += 1
            return True

        if (self.config.per_symbol_daily_r_budget is not None
                and symbol is not None
                and self.symbol_daily_killed.get(symbol, False)):
            self.trades_blocked_symbol += 1
            return True

        return False

    def record_trade(self, r_value: float, symbol: Optional[str] = None):
        if np.isnan(r_value):
            return

        self.daily_r += r_value

        if self.config.daily_loss_cap is not None:
            if self.daily_r <= self.config.daily_loss_cap:
                if not self.daily_killed:
                    self.daily_killed = True
                    self.days_killed += 1
                    log.info("[V5_GATE] daily_loss_cap hit: date=%s cumR=%.2f cap=%.2f",
                             self.current_date, self.daily_r, self.config.daily_loss_cap)

        if symbol is not None:
            self.symbol_daily_r[symbol] = self.symbol_daily_r.get(symbol, 0.0) + r_value
            if self.config.per_symbol_daily_r_budget is not None:
                if self.symbol_daily_r[symbol] <= self.config.per_symbol_daily_r_budget:
                    if not self.symbol_daily_killed.get(symbol, False):
                        self.symbol_daily_killed[symbol] = True
                        self.symbol_kills[symbol] = self.symbol_kills.get(symbol, 0) + 1
                        log.info("[V5_GATE] per_symbol_cap hit: date=%s sym=%s cumR=%.2f cap=%.2f",
                                 self.current_date, symbol, self.symbol_daily_r[symbol],
                                 self.config.per_symbol_daily_r_budget)

    def get_diagnostics(self) -> dict:
        return {
            'daily_loss_cap': self.config.daily_loss_cap,
            'per_symbol_daily_r_budget': self.config.per_symbol_daily_r_budget,
            'days_killed': self.days_killed,
            'trades_blocked_daily_cap': self.trades_blocked_daily,
            'trades_blocked_symbol_cap': self.trades_blocked_symbol,
            'symbol_kill_counts': dict(self.symbol_kills),
        }


class TrailingEquityStop:
    """Pauses trading when equity drawdown exceeds threshold.

    Tracks cumulative R (equity curve). When equity drops more than
    `stop_distance` R from peak, blocks all trades until equity recovers
    to within `recovery_pct` of the stop distance from peak.
    """

    def __init__(self, stop_distance: float, recovery_pct: float = 0.5):
        self.stop_distance = stop_distance
        self.recovery_pct = recovery_pct
        self.cumulative_r: float = 0.0
        self.peak_r: float = 0.0
        self.stopped: bool = False

        self.stop_triggers: int = 0
        self.trades_blocked: int = 0
        self.max_drawdown_r: float = 0.0
        self.drawdown_history: List[float] = []

    def update(self, trade_r: float):
        if np.isnan(trade_r):
            return

        self.cumulative_r += trade_r

        if self.cumulative_r > self.peak_r:
            self.peak_r = self.cumulative_r

        drawdown = self.peak_r - self.cumulative_r
        self.drawdown_history.append(drawdown)
        self.max_drawdown_r = max(self.max_drawdown_r, drawdown)

        if not self.stopped and drawdown >= self.stop_distance:
            self.stopped = True
            self.stop_triggers += 1
            log.info("[V5_GATE] trailing_equity_stop TRIGGERED: peak=%.2f current=%.2f DD=%.2f stop=%.2f",
                     self.peak_r, self.cumulative_r, drawdown, self.stop_distance)

        recovery_threshold = self.stop_distance * self.recovery_pct
        if self.stopped and drawdown <= recovery_threshold:
            self.stopped = False
            log.info("[V5_GATE] trailing_equity_stop RECOVERED: peak=%.2f current=%.2f DD=%.2f",
                     self.peak_r, self.cumulative_r, drawdown)

    def should_block(self) -> bool:
        if self.stopped:
            self.trades_blocked += 1
            return True
        return False

    def get_diagnostics(self) -> dict:
        dd_arr = np.array(self.drawdown_history) if self.drawdown_history else np.array([0.0])
        return {
            'trailing_equity_stop': self.stop_distance,
            'stop_triggers': self.stop_triggers,
            'trades_blocked_equity_stop': self.trades_blocked,
            'max_drawdown_r': float(self.max_drawdown_r),
            'avg_drawdown_r': float(np.mean(dd_arr)),
            'final_equity_r': float(self.cumulative_r),
            'peak_equity_r': float(self.peak_r),
        }


@dataclass
class ConvictionSizingConfig:
    enabled: bool = False
    tier_top_pct: float = 5.0
    tier_top_mult: float = 2.5
    tier_high_pct: float = 20.0
    tier_high_mult: float = 1.5
    tier_mid_mult: float = 1.0
    tier_low_pct: float = 50.0
    tier_low_mult: float = 0.5
    confidence_boost_threshold: float = 0.65
    confidence_boost_mult: float = 1.3
    max_combined_mult: float = 3.5
    window_size: int = 500


class ConvictionSizer:
    """Score-tiered position sizing with directional confidence boost.

    Assigns size multipliers based on where the trade's score falls
    in the distribution of recent scores (percentile-based tiers):
      - Top tier_top_pct%:  tier_top_mult  (e.g., top 5% → 2.5x)
      - Top tier_high_pct%: tier_high_mult (e.g., top 20% → 1.5x)
      - Middle:             tier_mid_mult  (e.g., 1.0x)
      - Bottom tier_low_pct%: tier_low_mult (e.g., bottom 50% → 0.5x)

    Additionally, if directional confidence (p_long for LONG, p_short for SHORT)
    exceeds confidence_boost_threshold, the multiplier gets a confidence_boost_mult.

    Score percentiles are computed from a rolling window of the most recent
    window_size scores to adapt to changing model output distributions.
    """

    def __init__(self, config: ConvictionSizingConfig):
        self.config = config
        from collections import deque
        self.score_window: deque = deque(maxlen=config.window_size)
        self.sizing_history: List[dict] = []
        self._min_scores_for_tiers = 20

    def compute_size_multiplier(self, score: float, p_directional: float,
                                 side: int) -> float:
        if not self.config.enabled:
            return 1.0

        self.score_window.append(score)

        if len(self.score_window) < self._min_scores_for_tiers:
            tier_mult = self.config.tier_mid_mult
            tier_name = "warmup"
        else:
            scores_arr = np.array(self.score_window)
            pct = float(np.sum(scores_arr < score) / len(scores_arr) * 100)

            if pct >= (100 - self.config.tier_top_pct):
                tier_mult = self.config.tier_top_mult
                tier_name = "top"
            elif pct >= (100 - self.config.tier_high_pct):
                tier_mult = self.config.tier_high_mult
                tier_name = "high"
            elif pct < self.config.tier_low_pct:
                tier_mult = self.config.tier_low_mult
                tier_name = "low"
            else:
                tier_mult = self.config.tier_mid_mult
                tier_name = "mid"

        confidence_boost = 1.0
        if p_directional >= self.config.confidence_boost_threshold:
            confidence_boost = self.config.confidence_boost_mult

        combined = tier_mult * confidence_boost
        combined = min(combined, self.config.max_combined_mult)

        self.sizing_history.append({
            'score': score,
            'tier': tier_name,
            'tier_mult': tier_mult,
            'confidence_boost': confidence_boost,
            'combined_mult': combined,
            'p_directional': p_directional,
            'side': side,
        })

        return combined

    def get_diagnostics(self) -> dict:
        if not self.sizing_history:
            return {
                'conviction_sizing_enabled': self.config.enabled,
                'total_conviction_trades': 0,
            }
        mults = np.array([h['combined_mult'] for h in self.sizing_history])
        tiers = [h['tier'] for h in self.sizing_history]
        boosts = [h['confidence_boost'] for h in self.sizing_history]
        return {
            'conviction_sizing_enabled': self.config.enabled,
            'total_conviction_trades': len(self.sizing_history),
            'avg_conviction_mult': float(np.mean(mults)),
            'median_conviction_mult': float(np.median(mults)),
            'min_conviction_mult': float(np.min(mults)),
            'max_conviction_mult': float(np.max(mults)),
            'tier_distribution': {
                'top': tiers.count('top'),
                'high': tiers.count('high'),
                'mid': tiers.count('mid'),
                'low': tiers.count('low'),
                'warmup': tiers.count('warmup'),
            },
            'pct_confidence_boosted': float(np.mean([b > 1.0 for b in boosts]) * 100),
            'conviction_mult_p10': float(np.percentile(mults, 10)),
            'conviction_mult_p90': float(np.percentile(mults, 90)),
        }


@dataclass
class MultiRegimeConfig:
    enabled: bool = False
    adx_trending_threshold: float = 25.0
    adx_choppy_threshold: float = 20.0
    atr_high_vol_ratio: float = 1.3
    atr_low_vol_ratio: float = 0.7
    atr_rolling_window: int = 96
    ema_slope_window: int = 10
    ema_price_buffer: float = 0.005


class MultiRegimeClassifier:
    """Multi-regime market state classifier using ADX + ATR ratio + EMA200 slope.

    Classifies each bar into one of 5 regimes:
      - trending_up:   ADX >= adx_trending_threshold AND price > EMA200 AND EMA slope > 0
      - trending_down: ADX >= adx_trending_threshold AND price < EMA200 AND EMA slope < 0
      - choppy:        ADX < adx_choppy_threshold (no clear trend)
      - high_vol:      ATR ratio > atr_high_vol_ratio (danger zone, volatile)
      - low_vol:       ATR ratio < atr_low_vol_ratio AND not trending (compression)

    Priority order (for overlapping conditions):
      1. high_vol  (overrides everything — volatile markets are dangerous)
      2. choppy    (ADX says no trend — avoid directional trades)
      3. trending_up / trending_down (clear directional trend)
      4. low_vol   (quiet compression, potential breakout)

    Ultra-Conviction alignment:
      - LONG allowed:  trending_up, low_vol
      - SHORT allowed: trending_down, low_vol
      - BLOCKED:       choppy, high_vol
    """

    REGIMES = ("trending_up", "trending_down", "choppy", "high_vol", "low_vol")
    LONG_ALLOWED = {"trending_up", "low_vol"}
    SHORT_ALLOWED = {"trending_down", "low_vol"}
    BLOCKED = {"choppy", "high_vol"}

    def __init__(self, config: MultiRegimeConfig):
        self.config = config
        self.regime_counts: Dict[str, int] = {r: 0 for r in self.REGIMES}
        self.regime_counts["unknown"] = 0
        self.total_classified: int = 0

    def classify(self, adx_val: float, atr_current: float,
                 atr_rolling: float, close_price: float,
                 ema200_val: float, ema200_prev: float) -> str:
        self.total_classified += 1

        has_adx = not np.isnan(adx_val)
        has_atr = (not np.isnan(atr_current) and not np.isnan(atr_rolling)
                   and atr_rolling > 1e-10)
        has_ema = (not np.isnan(ema200_val) and not np.isnan(ema200_prev)
                   and ema200_val > 0)

        atr_ratio = (atr_current / atr_rolling) if has_atr else 1.0

        if has_atr and atr_ratio > self.config.atr_high_vol_ratio:
            self.regime_counts["high_vol"] += 1
            return "high_vol"

        if has_adx and adx_val < self.config.adx_choppy_threshold:
            if has_atr and atr_ratio < self.config.atr_low_vol_ratio:
                self.regime_counts["low_vol"] += 1
                return "low_vol"
            self.regime_counts["choppy"] += 1
            return "choppy"

        if has_adx and adx_val >= self.config.adx_trending_threshold and has_ema:
            ema_slope = (ema200_val - ema200_prev) / ema200_prev
            price_above = close_price > ema200_val * (1.0 + self.config.ema_price_buffer)
            price_below = close_price < ema200_val * (1.0 - self.config.ema_price_buffer)

            if price_above and ema_slope > 0:
                self.regime_counts["trending_up"] += 1
                return "trending_up"
            elif price_below and ema_slope < 0:
                self.regime_counts["trending_down"] += 1
                return "trending_down"

        if has_atr and atr_ratio < self.config.atr_low_vol_ratio:
            self.regime_counts["low_vol"] += 1
            return "low_vol"

        if has_adx and adx_val >= self.config.adx_choppy_threshold:
            if has_ema:
                if close_price > ema200_val:
                    self.regime_counts["trending_up"] += 1
                    return "trending_up"
                else:
                    self.regime_counts["trending_down"] += 1
                    return "trending_down"

        self.regime_counts["unknown"] += 1
        return "choppy"

    def is_long_allowed(self, regime: str) -> bool:
        return regime in self.LONG_ALLOWED

    def is_short_allowed(self, regime: str) -> bool:
        return regime in self.SHORT_ALLOWED

    def is_blocked(self, regime: str) -> bool:
        return regime in self.BLOCKED

    def get_diagnostics(self) -> dict:
        total = max(self.total_classified, 1)
        return {
            'multi_regime_enabled': self.config.enabled,
            'total_classified': self.total_classified,
            'regime_counts': dict(self.regime_counts),
            'regime_pct': {k: round(v / total * 100, 1)
                           for k, v in self.regime_counts.items()},
            'config': {
                'adx_trending': self.config.adx_trending_threshold,
                'adx_choppy': self.config.adx_choppy_threshold,
                'atr_high_vol': self.config.atr_high_vol_ratio,
                'atr_low_vol': self.config.atr_low_vol_ratio,
            }
        }


@dataclass
class UltraConvictionConfig:
    enabled: bool = False
    risk_cap: float = 0.05
    score_pct: float = 0.95
    adx_min: float = 25.0
    edge_min: float = 0.03
    dd_max: float = 0.10
    max_per_day: int = 1
    mult: float = 3.0
    score_window: int = 500


class UltraConvictionSizer:
    """Ultra-Conviction Risk Tier: allows rare, very-high-conviction trades
    to use up to risk_cap (default 5%) risk per trade.

    Applied AFTER conviction sizing but BEFORE final clamp.
    All 8 trigger conditions must be true for ultra to activate:
      1) Feature enabled
      2) Score >= rolling percentile(score_history, score_pct)
      3) ADX >= adx_min (strong trend)
      4) Edge for the trade side >= edge_min
      5) Regime aligned (bull→LONG, bear→SHORT; neutral disables)
      6) Equity drawdown <= dd_max
      7) Not blocked by any existing capital protection gate
      8) Ultra trades today < max_per_day
    """

    def __init__(self, config: UltraConvictionConfig):
        self.config = config
        from collections import deque
        self.score_window: deque = deque(maxlen=config.score_window)
        self.daily_ultra_counts: Dict[str, int] = {}
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.ultra_applied: int = 0
        self.ultra_skipped: int = 0
        self.skip_reasons: Dict[str, int] = {}
        self.ultra_history: List[dict] = []
        self._min_scores_for_pct = 20

    def update_equity(self, trade_r: float):
        if not np.isnan(trade_r):
            self.current_equity += trade_r
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity

    def _get_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def _get_score_percentile(self, score: float) -> Optional[float]:
        if len(self.score_window) < self._min_scores_for_pct:
            return None
        scores_arr = np.array(self.score_window)
        pct = self.config.score_pct
        if pct <= 1.0:
            pct = pct * 100
        pct = float(np.clip(pct, 0.0, 100.0))
        pct_val = float(np.percentile(scores_arr, pct))
        return pct_val

    def record_score(self, score: float):
        self.score_window.append(score)

    def _skip(self, reason: str, **log_kwargs) -> bool:
        self.ultra_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        parts = [f"reason={reason}"]
        for k, v in log_kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        log.debug("[V5_ULTRA] SKIP %s", " ".join(parts))
        return False

    def evaluate(self, symbol: str, side: int, score: float,
                 edge_l: float, edge_s: float, adx_val: float,
                 regime: str, date_str: str,
                 capital_blocked: bool = False) -> bool:
        if not self.config.enabled:
            return False

        pct_threshold = self._get_score_percentile(score)
        if pct_threshold is None:
            return self._skip("warmup", scores=len(self.score_window))
        if score < pct_threshold:
            return self._skip("score_low", score=score, threshold=pct_threshold)

        if not np.isnan(adx_val) and adx_val < self.config.adx_min:
            return self._skip("adx_low", adx=adx_val, min=self.config.adx_min)

        edge = edge_l if side == 1 else edge_s
        if edge < self.config.edge_min:
            return self._skip("edge_low", edge=edge, min=self.config.edge_min,
                              side="LONG" if side == 1 else "SHORT")

        _BLOCKED_REGIMES = {"choppy", "high_vol", "neutral", "chop", "unknown", ""}
        if regime in _BLOCKED_REGIMES:
            return self._skip("regime_blocked", regime=regime)
        _LONG_REGIMES = {"trending_up", "low_vol", "bull"}
        _SHORT_REGIMES = {"trending_down", "low_vol", "bear"}
        if side == 1 and regime not in _LONG_REGIMES:
            return self._skip("regime_mismatch", regime=regime,
                              side="LONG")
        if side == -1 and regime not in _SHORT_REGIMES:
            return self._skip("regime_mismatch", regime=regime,
                              side="SHORT")

        dd = self._get_drawdown()
        if dd > self.config.dd_max:
            return self._skip("drawdown_high", dd=dd, max=self.config.dd_max)

        if capital_blocked:
            return self._skip("capital_blocked")

        day_count = self.daily_ultra_counts.get(date_str, 0)
        if day_count >= self.config.max_per_day:
            return self._skip("max_per_day", count=day_count, max=self.config.max_per_day)

        self.daily_ultra_counts[date_str] = day_count + 1
        self.ultra_applied += 1
        side_str = "LONG" if side == 1 else "SHORT"
        display_pct = self.config.score_pct * 100 if self.config.score_pct <= 1.0 else self.config.score_pct
        log.info("[V5_ULTRA] APPLY symbol=%s side=%s score=%.2f p%.0f=%.2f adx=%.1f "
                 "edge_%s=%.3f regime=%s dd=%.2f mult=%.1f",
                 symbol, side_str, score, display_pct,
                 pct_threshold, adx_val,
                 "L" if side == 1 else "S", edge,
                 regime, dd, self.config.mult)
        return True

    def apply_ultra_sizing(self, current_mult: float, stop_distance_pct: float) -> float:
        sized_mult = current_mult * self.config.mult
        if stop_distance_pct > 0:
            risk_at_size = sized_mult * stop_distance_pct
            if risk_at_size > self.config.risk_cap:
                sized_mult = self.config.risk_cap / stop_distance_pct
        self.ultra_history.append({
            'mult_before': current_mult,
            'mult_after': sized_mult,
            'risk_cap': self.config.risk_cap,
        })
        return sized_mult

    def get_diagnostics(self) -> dict:
        result = {
            'ultra_conviction_enabled': self.config.enabled,
            'ultra_applied': self.ultra_applied,
            'ultra_skipped': self.ultra_skipped,
            'skip_reasons': dict(self.skip_reasons),
            'risk_cap': self.config.risk_cap,
            'mult': self.config.mult,
            'score_pct': self.config.score_pct,
            'adx_min': self.config.adx_min,
            'edge_min': self.config.edge_min,
            'dd_max': self.config.dd_max,
            'max_per_day': self.config.max_per_day,
        }
        if self.ultra_history:
            mults_before = np.array([h['mult_before'] for h in self.ultra_history])
            mults_after = np.array([h['mult_after'] for h in self.ultra_history])
            result['avg_mult_before'] = float(np.mean(mults_before))
            result['avg_mult_after'] = float(np.mean(mults_after))
            result['max_mult_after'] = float(np.max(mults_after))
        return result


@dataclass
class CalibrationMonitorConfig:
    enabled: bool = True
    window_size: int = 200
    n_bins: int = 10
    warn_ece: float = 0.10
    block_ece: float = 0.15
    min_trades: int = 50


class CalibrationMonitor:
    def __init__(self, config: CalibrationMonitorConfig = None):
        self.config = config or CalibrationMonitorConfig()
        self.pred_probs: List[float] = []
        self.actual_outcomes: List[int] = []
        self.rolling_ece: List[float] = []
        self.warnings_issued = 0
        self.blocks_issued = 0

    def record(self, predicted_prob: float, won: bool):
        self.pred_probs.append(float(predicted_prob))
        self.actual_outcomes.append(1 if won else 0)

    def compute_ece(self) -> Optional[float]:
        n = len(self.pred_probs)
        if n < self.config.min_trades:
            return None
        window = self.config.window_size
        probs = np.array(self.pred_probs[-window:])
        actuals = np.array(self.actual_outcomes[-window:])
        n_w = len(probs)
        if n_w < self.config.min_trades:
            return None
        bin_edges = np.linspace(0.0, 1.0, self.config.n_bins + 1)
        ece = 0.0
        for i in range(self.config.n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == self.config.n_bins - 1:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            n_bin = int(np.sum(mask))
            if n_bin == 0:
                continue
            avg_pred = float(np.mean(probs[mask]))
            avg_actual = float(np.mean(actuals[mask]))
            ece += (n_bin / n_w) * abs(avg_pred - avg_actual)
        self.rolling_ece.append(ece)
        return ece

    def should_warn(self) -> bool:
        ece = self.compute_ece()
        if ece is not None and ece > self.config.warn_ece:
            self.warnings_issued += 1
            return True
        return False

    def should_block(self) -> bool:
        ece = self.compute_ece()
        if ece is not None and ece > self.config.block_ece:
            self.blocks_issued += 1
            return True
        return False

    def get_diagnostics(self) -> dict:
        ece = self.compute_ece()
        return {
            'total_recorded': len(self.pred_probs),
            'current_ece': float(ece) if ece is not None else None,
            'warnings_issued': self.warnings_issued,
            'blocks_issued': self.blocks_issued,
            'window_size': self.config.window_size,
            'warn_threshold': self.config.warn_ece,
            'block_threshold': self.config.block_ece,
        }


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]
    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0
    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0].astype(float)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    eps = 1e-4
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_statistical_edge(t_r: np.ndarray, trade_timestamps: Optional[np.ndarray] = None,
                              bars_per_day: float = 96.0) -> dict:
    n = len(t_r)
    if n < 2:
        return {
            'sharpe_annualized': 0.0, 'sortino_annualized': 0.0,
            'expectancy_ci_lower': 0.0, 'expectancy_ci_upper': 0.0,
            'expectancy_t_stat': 0.0, 'expectancy_p_value': 1.0,
            'n_trades': n,
        }
    mean_r = float(np.mean(t_r))
    std_r = float(np.std(t_r, ddof=1))
    downside = t_r[t_r < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else max(std_r, 1e-6)

    if trade_timestamps is not None and n > 0:
        from datetime import datetime
        trade_days = np.array([datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                               for ts in trade_timestamps])
        unique_days = np.unique(trade_days)
        daily_pnl = np.array([float(np.sum(t_r[trade_days == d])) for d in unique_days])
        n_days = len(unique_days)
        if n_days > 1:
            d_mean = float(np.mean(daily_pnl))
            d_std = float(np.std(daily_pnl, ddof=1))
            d_down = daily_pnl[daily_pnl < 0]
            d_down_std = float(np.std(d_down, ddof=1)) if len(d_down) > 1 else max(d_std, 1e-6)
            sharpe = d_mean / max(d_std, 1e-6) * np.sqrt(252)
            sortino = d_mean / max(d_down_std, 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
            sortino = 0.0
    else:
        trades_per_year = n * 252 / max(n / bars_per_day, 1)
        ann_factor = np.sqrt(max(trades_per_year, 1))
        sharpe = mean_r / max(std_r, 1e-6) * ann_factor
        sortino = mean_r / max(downside_std, 1e-6) * ann_factor

    t_stat = mean_r / max(std_r / np.sqrt(n), 1e-8)
    from scipy import stats as sp_stats
    p_value = float(1.0 - sp_stats.t.cdf(t_stat, df=n - 1)) if n > 2 else 1.0

    rng = np.random.RandomState(42)
    n_bootstrap = 1000
    boot_means = np.array([
        float(np.mean(rng.choice(t_r, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        'sharpe_annualized': float(sharpe),
        'sortino_annualized': float(sortino),
        'expectancy_ci_lower': ci_lower,
        'expectancy_ci_upper': ci_upper,
        'expectancy_t_stat': float(t_stat),
        'expectancy_p_value': float(p_value),
        'n_trades': n,
    }


def build_sizing_diagnostics(sizer: Optional[AdaptivePositionSizer],
                              regime: Optional[RegimeScaler],
                              daily_tracker: Optional[DailyLossTracker],
                              equity_stop: Optional[TrailingEquityStop],
                              sized_r: Optional[np.ndarray] = None,
                              unsized_r: Optional[np.ndarray] = None,
                              conviction: Optional[ConvictionSizer] = None,
                              ultra: Optional['UltraConvictionSizer'] = None) -> dict:
    report = {}
    if sizer:
        report['adaptive_sizing'] = sizer.get_diagnostics()
    if regime:
        report['regime_scaling'] = regime.get_diagnostics()
    if daily_tracker:
        report['daily_loss_management'] = daily_tracker.get_diagnostics()
    if equity_stop:
        report['trailing_equity_stop'] = equity_stop.get_diagnostics()
    if conviction:
        report['conviction_sizing'] = conviction.get_diagnostics()
    if ultra:
        report['ultra_conviction'] = ultra.get_diagnostics()

    if sized_r is not None and unsized_r is not None and len(sized_r) > 0:
        report['sizing_comparison'] = {
            'unsized_total_r': float(np.sum(unsized_r)),
            'sized_total_r': float(np.sum(sized_r)),
            'sizing_impact_r': float(np.sum(sized_r) - np.sum(unsized_r)),
            'sizing_impact_pct': float((np.sum(sized_r) / max(abs(np.sum(unsized_r)), 0.01) - 1.0) * 100),
        }

    return report
