"""
Shared utility functions for data processing.

Single-source-of-truth for ATR computation and other common operations
used across v5_target_generator, candidate_generator, triple_barrier, pipeline.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute Average True Range (ATR) using Wilder's smoothing.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period (default 14)

    Returns:
        np.ndarray of ATR values, length = len(df).
        First `period` values are 0 (insufficient data).
    """
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(df)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    atr = np.zeros(n, dtype=np.float64)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR as a percentage of close price.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period

    Returns:
        np.ndarray of ATR/close ratios.
    """
    atr = compute_atr(df, period)
    closes = df['close'].values.astype(np.float64)
    atr_pct = np.where(closes > 0, atr / closes, 0.0)
    return atr_pct


def compute_adaptive_horizons(
    atr: np.ndarray,
    base_horizon: int = 16,
    median_window: int = 50,
    min_horizon: int = 8,
    max_horizon: int = 48,
) -> np.ndarray:
    """Compute per-bar adaptive horizon based on volatility.

    effective_horizon = base_horizon * (median_ATR_50 / current_ATR_14)
    Low vol -> horizon extends (more bars needed), high vol -> contracts.
    Clamped to [min_horizon, max_horizon].

    No lookahead: rolling median uses only data up to current bar.

    Args:
        atr: ATR array (len N), computed from compute_atr
        base_horizon: default horizon in bars
        median_window: rolling window for median ATR
        min_horizon: minimum horizon (default 8 = 2h on 15m)
        max_horizon: maximum horizon (default 48 = 12h on 15m)

    Returns:
        np.ndarray of int effective horizons, length = len(atr)
    """
    n = len(atr)
    horizons = np.full(n, base_horizon, dtype=np.int64)

    for i in range(n):
        if atr[i] <= 0:
            continue
        lookback_start = max(0, i - median_window + 1)
        atr_window = atr[lookback_start:i + 1]
        valid_atr = atr_window[atr_window > 0]
        if len(valid_atr) < 2:
            continue
        median_atr = np.median(valid_atr)
        if median_atr <= 0:
            continue
        ratio = median_atr / atr[i]
        eff = base_horizon * ratio
        horizons[i] = int(np.clip(round(eff), min_horizon, max_horizon))

    return horizons


def compute_vol_adjusted_sl(
    atr: np.ndarray,
    sl_mult: float,
    median_window: int = 50,
    high_vol_threshold: float = 1.3,
    sl_boost: float = 1.15,
) -> np.ndarray:
    """Compute per-bar volatility-adjusted SL multiplier.

    In high-vol (ATR > high_vol_threshold * median_ATR), SL_mult *= sl_boost.
    No lookahead: rolling median uses only data up to current bar.

    Args:
        atr: ATR array
        sl_mult: base SL multiplier
        median_window: rolling window for median ATR
        high_vol_threshold: threshold ratio for high-vol detection
        sl_boost: multiplicative boost for SL in high-vol

    Returns:
        np.ndarray of float adjusted SL multipliers, length = len(atr)
    """
    n = len(atr)
    sl_mults = np.full(n, sl_mult, dtype=np.float64)

    for i in range(n):
        if atr[i] <= 0:
            continue
        lookback_start = max(0, i - median_window + 1)
        atr_window = atr[lookback_start:i + 1]
        valid_atr = atr_window[atr_window > 0]
        if len(valid_atr) < 2:
            continue
        median_atr = np.median(valid_atr)
        if median_atr <= 0:
            continue
        if atr[i] > high_vol_threshold * median_atr:
            sl_mults[i] = sl_mult * sl_boost

    return sl_mults


def log_adaptive_horizon_diagnostics(
    horizons: np.ndarray,
    sl_mults: np.ndarray,
    base_horizon: int,
    base_sl_mult: float,
) -> None:
    """Log histogram of effective horizons and vol-adjusted SL multipliers."""
    valid_h = horizons[horizons > 0]
    if len(valid_h) == 0:
        logger.info("[ADAPTIVE_HORIZON] No valid horizons to log")
        return

    h_min, h_max = int(np.min(valid_h)), int(np.max(valid_h))
    h_mean = float(np.mean(valid_h))
    h_median = float(np.median(valid_h))

    bins = [8, 12, 16, 20, 24, 32, 48, 49]
    hist, _ = np.histogram(valid_h, bins=bins)
    hist_str = " | ".join(f"{bins[j]}-{bins[j+1]-1}:{hist[j]}" for j in range(len(hist)))

    logger.info(f"[ADAPTIVE_HORIZON] base={base_horizon} effective: "
                f"min={h_min} max={h_max} mean={h_mean:.1f} median={h_median:.0f}")
    logger.info(f"[ADAPTIVE_HORIZON] horizon histogram: {hist_str}")

    boosted = np.sum(sl_mults > base_sl_mult)
    total = len(sl_mults)
    logger.info(f"[ADAPTIVE_HORIZON] SL boost: {boosted}/{total} bars "
                f"({boosted/max(total,1):.1%}) have vol-adjusted SL "
                f"(base={base_sl_mult:.2f}, boosted={base_sl_mult*1.15:.2f})")


def generate_v5_sweep_outcomes(
    df: pd.DataFrame,
    horizon: int = 16,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    atr_period: int = 14,
    adaptive_horizon: bool = True,
    horizon_min: int = 8,
    horizon_max: int = 48,
    median_window: int = 50,
    high_vol_threshold: float = 1.3,
    sl_boost: float = 1.15,
    entry_lag_atr_fraction: float = 0.0,
) -> dict:
    """Generate side-conditional trade outcomes using v5-consistent barrier logic.

    Uses the same ATR computation as v5_target_generator for consistency.
    Simulates BOTH LONG and SHORT trades independently per bar.
    Supports volatility-adaptive horizons and SL adjustments.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        horizon: forward-looking window in bars (base horizon when adaptive)
        tp_mult: ATR multiplier for take-profit
        sl_mult: ATR multiplier for stop-loss
        atr_period: ATR lookback period
        adaptive_horizon: if True, compute per-bar adaptive horizon
        horizon_min: minimum adaptive horizon
        horizon_max: maximum adaptive horizon
        median_window: rolling window for median ATR
        high_vol_threshold: ATR ratio threshold for high-vol SL adjustment
        sl_boost: SL multiplier boost in high-vol
        entry_lag_atr_fraction: additional entry cost as fraction of ATR, modelling
            the gap between signal-bar close and next-bar open in live execution.
            For LONG: entry price += lag * ATR (pays more).
            For SHORT: entry price -= lag * ATR (sells for less).
            Default 0.0 = backward-compatible (no lag adjustment).
            Recommended: 0.25 for realistic live-entry modelling.

    Returns:
        dict with:
            r_long: (N,) float32 array -- realized R if LONG at this bar
            r_short: (N,) float32 array -- realized R if SHORT at this bar
            out_long: (N,) object array -- outcome type if LONG
            out_short: (N,) object array -- outcome type if SHORT
            realized_r: (N,) float32 -- DEPRECATED best-side oracle R (DO NOT USE FOR EVAL)
            outcome: (N,) object -- DEPRECATED best-side oracle outcome (DO NOT USE FOR EVAL)
            effective_horizons: (N,) int64 -- per-bar adaptive horizons used
            effective_sl_mults: (N,) float64 -- per-bar vol-adjusted SL multipliers
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    if adaptive_horizon:
        effective_horizons = compute_adaptive_horizons(
            atr, base_horizon=horizon, median_window=median_window,
            min_horizon=horizon_min, max_horizon=horizon_max,
        )
        effective_sl_mults = compute_vol_adjusted_sl(
            atr, sl_mult=sl_mult, median_window=median_window,
            high_vol_threshold=high_vol_threshold, sl_boost=sl_boost,
        )
        log_adaptive_horizon_diagnostics(effective_horizons, effective_sl_mults, horizon, sl_mult)
    else:
        effective_horizons = np.full(n, horizon, dtype=np.int64)
        effective_sl_mults = np.full(n, sl_mult, dtype=np.float64)

    r_long = np.full(n, np.nan, dtype=np.float64)
    r_short = np.full(n, np.nan, dtype=np.float64)
    out_long = np.full(n, "NO_CANDIDATE", dtype=object)
    out_short = np.full(n, "NO_CANDIDATE", dtype=object)
    realized_r_best = np.full(n, np.nan, dtype=np.float64)
    outcomes_best = np.full(n, "NO_CANDIDATE", dtype=object)

    for i in range(n):
        h_i = int(effective_horizons[i])
        sl_i = float(effective_sl_mults[i])
        if i + h_i >= n:
            continue
        if atr[i] <= 0 or closes[i] <= 0:
            continue

        base_entry = closes[i]
        lag = entry_lag_atr_fraction * atr[i]
        long_entry = base_entry + lag   # LONG fills at a worse (higher) price
        short_entry = base_entry - lag  # SHORT fills at a worse (lower) price
        tp_dist = atr[i] * tp_mult
        sl_dist = atr[i] * sl_i

        long_r_val, long_out_val = _simulate_trade(
            highs, lows, closes, i, h_i, n,
            long_entry, tp_dist, sl_dist, tp_mult, sl_i, atr[i], side=1
        )
        short_r_val, short_out_val = _simulate_trade(
            highs, lows, closes, i, h_i, n,
            short_entry, tp_dist, sl_dist, tp_mult, sl_i, atr[i], side=-1
        )

        r_long[i] = long_r_val
        r_short[i] = short_r_val
        out_long[i] = long_out_val
        out_short[i] = short_out_val

        if long_r_val >= short_r_val:
            realized_r_best[i] = long_r_val
            outcomes_best[i] = long_out_val
        else:
            realized_r_best[i] = short_r_val
            outcomes_best[i] = short_out_val

    return {
        'r_long': r_long.astype(np.float32),
        'r_short': r_short.astype(np.float32),
        'out_long': out_long,
        'out_short': out_short,
        'realized_r': realized_r_best.astype(np.float32),
        'outcome': outcomes_best,
        'effective_horizons': effective_horizons,
        'effective_sl_mults': effective_sl_mults,
    }


def _simulate_trade(
    highs, lows, closes, i, horizon, n,
    entry, tp_dist, sl_dist, tp_mult, sl_mult, atr_val, side=1
):
    """Simulate a single-direction trade through the barrier.

    Args:
        side: 1 for LONG, -1 for SHORT

    Returns:
        (realized_r, outcome_str)
    """
    if side == 1:
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist

    for j in range(i + 1, min(i + 1 + horizon, n)):
        if side == 1:
            if highs[j] >= tp_price:
                return tp_mult / sl_mult, "TP"
            if lows[j] <= sl_price:
                return -1.0, "SL"
        else:
            if lows[j] <= tp_price:
                return tp_mult / sl_mult, "TP"
            if highs[j] >= sl_price:
                return -1.0, "SL"

    exit_price = closes[min(i + horizon, n - 1)]
    if side == 1:
        expiry_r = (exit_price - entry) / (atr_val * sl_mult)
    else:
        expiry_r = (entry - exit_price) / (atr_val * sl_mult)

    if expiry_r >= 0:
        return expiry_r, "EXP_WIN"
    return expiry_r, "EXP_LOSS"


def _simulate_trade_trailing(
    highs, lows, closes, i, horizon, n,
    entry, tp_dist, sl_dist, tp_mult, sl_mult, atr_val, side=1,
    trail_activation=1.5, trail_distance=1.0, allow_runner=False,
    min_trail_profit_r=0.15,
):
    """Simulate a trade with trailing stop-loss.

    The trailing stop works in phases:
      Phase 1 (initial): Fixed SL at entry -/+ sl_dist (same as normal).
      Phase 2 (activated): Once price moves trail_activation * ATR in favor,
                           SL moves to entry + min_trail_profit floor, then trails
                           at trail_distance * ATR behind the best price seen.
      Phase 3 (runner, optional): If allow_runner=True, after TP level is reached,
                                  the trade stays open with a tight trail (0.5 * trail_distance * ATR)
                                  to capture extended moves.

    Args:
        trail_activation: ATR multiples of favorable move before trailing activates
        trail_distance: ATR multiples behind best price for trailing stop
        allow_runner: if True, don't exit at TP, let it run with tighter trail
        min_trail_profit_r: minimum R profit floor for trailing SL (prevents
            breakeven/tiny-win exits). Trail SL cannot go below entry + min_trail_profit_r * ATR * sl_mult.

    Returns:
        (realized_r, outcome_str)
    """
    if side == 1:
        tp_price = entry + tp_dist
        initial_sl = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        initial_sl = entry + sl_dist

    activation_dist = atr_val * trail_activation
    trail_dist_abs = atr_val * trail_distance
    min_profit_dist = min_trail_profit_r * atr_val * sl_mult

    trailing_active = False
    best_price = entry
    current_sl = initial_sl

    for j in range(i + 1, min(i + 1 + horizon, n)):
        bar_high = highs[j]
        bar_low = lows[j]
        bar_close = closes[j]

        if side == 1:
            if bar_high > best_price:
                best_price = bar_high

            if not allow_runner and bar_high >= tp_price:
                return tp_mult / sl_mult, "TP"

            if allow_runner and bar_high >= tp_price:
                trailing_active = True
                trail_dist_abs = atr_val * trail_distance * 0.5
                new_sl = best_price - trail_dist_abs
                new_sl = max(new_sl, entry + min_profit_dist)
                current_sl = max(current_sl, new_sl)

            if not trailing_active:
                if best_price - entry >= activation_dist:
                    trailing_active = True
                    new_sl = max(entry + min_profit_dist, best_price - trail_dist_abs)
                    current_sl = max(current_sl, new_sl)
            else:
                new_sl = best_price - trail_dist_abs
                new_sl = max(new_sl, entry + min_profit_dist)
                current_sl = max(current_sl, new_sl)

            if bar_low <= current_sl:
                realized_r = (current_sl - entry) / (atr_val * sl_mult)
                if trailing_active and realized_r > min_trail_profit_r:
                    return realized_r, "TRAIL_WIN"
                elif trailing_active:
                    return max(realized_r, 0.0), "TRAIL_BE"
                else:
                    return -1.0, "SL"

        else:
            if bar_low < best_price:
                best_price = bar_low

            if not allow_runner and bar_low <= tp_price:
                return tp_mult / sl_mult, "TP"

            if allow_runner and bar_low <= tp_price:
                trailing_active = True
                trail_dist_abs = atr_val * trail_distance * 0.5
                new_sl = best_price + trail_dist_abs
                new_sl = min(new_sl, entry - min_profit_dist)
                current_sl = min(current_sl, new_sl)

            if not trailing_active:
                if entry - best_price >= activation_dist:
                    trailing_active = True
                    new_sl = min(entry - min_profit_dist, best_price + trail_dist_abs)
                    current_sl = min(current_sl, new_sl)
            else:
                new_sl = best_price + trail_dist_abs
                new_sl = min(new_sl, entry - min_profit_dist)
                current_sl = min(current_sl, new_sl)

            if bar_high >= current_sl:
                realized_r = (entry - current_sl) / (atr_val * sl_mult)
                if trailing_active and realized_r > min_trail_profit_r:
                    return realized_r, "TRAIL_WIN"
                elif trailing_active:
                    return max(realized_r, 0.0), "TRAIL_BE"
                else:
                    return -1.0, "SL"

    exit_price = closes[min(i + horizon, n - 1)]
    if side == 1:
        expiry_r = (exit_price - entry) / (atr_val * sl_mult)
    else:
        expiry_r = (entry - exit_price) / (atr_val * sl_mult)

    if expiry_r >= 0:
        return expiry_r, "EXP_WIN"
    else:
        return expiry_r, "EXP_LOSS"


def generate_v5_sweep_outcomes_trailing(
    df: pd.DataFrame,
    horizon: int = 16,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    atr_period: int = 14,
    trail_activation: float = 1.5,
    trail_distance: float = 1.0,
    allow_runner: bool = False,
    min_trail_profit_r: float = 0.15,
    adaptive_horizon: bool = True,
    horizon_min: int = 8,
    horizon_max: int = 48,
    median_window: int = 50,
    high_vol_threshold: float = 1.3,
    sl_boost: float = 1.15,
) -> dict:
    """Generate side-conditional trade outcomes using trailing stop logic.

    Same interface as generate_v5_sweep_outcomes but uses _simulate_trade_trailing
    for dynamic exits instead of fixed TP/SL barriers.
    Supports volatility-adaptive horizons and SL adjustments.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        horizon: forward-looking window in bars (base horizon when adaptive)
        tp_mult: ATR multiplier for take-profit level
        sl_mult: ATR multiplier for initial stop-loss
        atr_period: ATR lookback period
        trail_activation: ATR multiples of favorable move before trailing activates
        trail_distance: ATR multiples behind best price for trailing stop
        allow_runner: if True, don't exit at TP, let it run with tighter trail
        adaptive_horizon: if True, compute per-bar adaptive horizon
        horizon_min: minimum adaptive horizon
        horizon_max: maximum adaptive horizon
        median_window: rolling window for median ATR
        high_vol_threshold: ATR ratio threshold for high-vol SL adjustment
        sl_boost: SL multiplier boost in high-vol

    Returns:
        dict with r_long, r_short, out_long, out_short, realized_r, outcome,
        effective_horizons, effective_sl_mults
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    if adaptive_horizon:
        effective_horizons = compute_adaptive_horizons(
            atr, base_horizon=horizon, median_window=median_window,
            min_horizon=horizon_min, max_horizon=horizon_max,
        )
        effective_sl_mults = compute_vol_adjusted_sl(
            atr, sl_mult=sl_mult, median_window=median_window,
            high_vol_threshold=high_vol_threshold, sl_boost=sl_boost,
        )
    else:
        effective_horizons = np.full(n, horizon, dtype=np.int64)
        effective_sl_mults = np.full(n, sl_mult, dtype=np.float64)

    r_long = np.full(n, np.nan, dtype=np.float64)
    r_short = np.full(n, np.nan, dtype=np.float64)
    out_long = np.full(n, "NO_CANDIDATE", dtype=object)
    out_short = np.full(n, "NO_CANDIDATE", dtype=object)
    realized_r_best = np.full(n, np.nan, dtype=np.float64)
    outcomes_best = np.full(n, "NO_CANDIDATE", dtype=object)

    for i in range(n):
        h_i = int(effective_horizons[i])
        sl_i = float(effective_sl_mults[i])
        if i + h_i >= n:
            continue
        if atr[i] <= 0 or closes[i] <= 0:
            continue

        entry = closes[i]
        tp_dist = atr[i] * tp_mult
        sl_dist = atr[i] * sl_i

        long_r_val, long_out_val = _simulate_trade_trailing(
            highs, lows, closes, i, h_i, n,
            entry, tp_dist, sl_dist, tp_mult, sl_i, atr[i], side=1,
            trail_activation=trail_activation, trail_distance=trail_distance,
            allow_runner=allow_runner, min_trail_profit_r=min_trail_profit_r,
        )
        short_r_val, short_out_val = _simulate_trade_trailing(
            highs, lows, closes, i, h_i, n,
            entry, tp_dist, sl_dist, tp_mult, sl_i, atr[i], side=-1,
            trail_activation=trail_activation, trail_distance=trail_distance,
            allow_runner=allow_runner, min_trail_profit_r=min_trail_profit_r,
        )

        r_long[i] = long_r_val
        r_short[i] = short_r_val
        out_long[i] = long_out_val
        out_short[i] = short_out_val

        if long_r_val >= short_r_val:
            realized_r_best[i] = long_r_val
            outcomes_best[i] = long_out_val
        else:
            realized_r_best[i] = short_r_val
            outcomes_best[i] = short_out_val

    return {
        'r_long': r_long.astype(np.float32),
        'r_short': r_short.astype(np.float32),
        'out_long': out_long,
        'out_short': out_short,
        'realized_r': realized_r_best.astype(np.float32),
        'outcome': outcomes_best,
        'effective_horizons': effective_horizons,
        'effective_sl_mults': effective_sl_mults,
    }
