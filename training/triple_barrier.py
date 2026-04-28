"""
Triple Barrier Simulator - Single Source of Truth
==================================================
Used by BOTH labeling and sweep evaluation to ensure parity.

compute_atr_14(df) -> ATR(14) using true range (OHLC)
triple_barrier_outcome_for_index(df, i, side, atr_i, ...) -> (outcome, r)
triple_barrier_batch(df, indices, sides, ...) -> (outcomes[], r_values[])
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_atr_14(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR using true range from OHLC data.
    
    Returns numpy array aligned to df rows.
    Matches the ATR method used in labeling (RegressionTargetGenerator._compute_atr).
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    n = len(df)
    
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    
    atr = np.empty(n, dtype=np.float64)
    atr[:] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    for i in range(min(period - 1, n)):
        if np.isnan(atr[i]):
            atr[i] = np.mean(tr[: i + 1]) if i > 0 else tr[0]
    
    return atr


def triple_barrier_outcome_for_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    side: int,
    atr_i: float,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 16,
    r_min_expiry: float = 1.0,
):
    """Simulate a single triple-barrier trade at index i.
    
    Args:
        highs/lows/closes: Price arrays from OHLC
        i: Entry bar index
        side: +1 (LONG) or -1 (SHORT)
        atr_i: ATR value at bar i
        tp_mult/sl_mult: ATR multipliers for TP/SL
        horizon: Max bars to hold
        r_min_expiry: Min R-multiple at expiry to count as win
    
    Returns:
        (outcome, realized_r) where outcome is one of:
        "TP", "SL", "EXP_WIN", "EXP_LOSS"
    """
    n = len(closes)
    entry = closes[i]
    
    a = atr_i
    if np.isnan(a) or a <= 0:
        a = entry * 0.005
    
    tp_dist = tp_mult * a
    sl_dist = sl_mult * a
    
    if side > 0:
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist
    
    for j in range(1, horizon + 1):
        idx = i + j
        if idx >= n:
            break
        
        if side > 0:
            tp_hit = highs[idx] >= tp_price
            sl_hit = lows[idx] <= sl_price
        else:
            tp_hit = lows[idx] <= tp_price
            sl_hit = highs[idx] >= sl_price
        
        if tp_hit and sl_hit:
            if side > 0:
                tp_excursion = highs[idx] - entry
                sl_excursion = entry - lows[idx]
            else:
                tp_excursion = entry - lows[idx]
                sl_excursion = highs[idx] - entry
            
            if tp_excursion >= sl_excursion:
                return ("TP", tp_mult / sl_mult)
            else:
                return ("SL", -1.0)
        elif tp_hit:
            return ("TP", tp_mult / sl_mult)
        elif sl_hit:
            return ("SL", -1.0)
    
    end_idx = min(i + horizon, n - 1)
    exit_price = closes[end_idx]
    if side > 0:
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price
    
    r_at_expiry = pnl / sl_dist if sl_dist > 0 else 0.0
    
    if r_at_expiry >= r_min_expiry:
        return ("EXP_WIN", r_at_expiry)
    else:
        return ("EXP_LOSS", r_at_expiry)


def compute_trade_cost_r(
    entry_price: float,
    atr_i: float,
    sl_mult: float,
    fees_bps_entry: float = 5.0,
    fees_bps_exit: float = 5.0,
    spread_bps: float = 1.0,
    slip_k: float = 0.10,
) -> float:
    """Compute round-trip trading cost in R-units.

    Cost components (all as fraction of entry_price):
      - fees:     (fees_bps_entry + fees_bps_exit) / 10_000
      - spread:   spread_bps / 10_000  (half at entry + half at exit)
      - slippage: 2 * slip_k * (atr / entry_price)  (entry + exit)

    R-unit conversion:
      sl_pct = sl_mult * atr / entry_price
      cost_R = total_cost_pct / sl_pct

    Returns cost_R (always >= 0).  net_R = gross_R - cost_R
    """
    if entry_price <= 0 or sl_mult <= 0:
        return 0.0
    if np.isnan(atr_i) or atr_i <= 0:
        return 0.0

    atr_pct = atr_i / entry_price

    fee_cost = (fees_bps_entry + fees_bps_exit) / 10_000.0
    spread_cost = spread_bps / 10_000.0
    slippage_cost = 2.0 * slip_k * atr_pct

    total_cost_pct = fee_cost + spread_cost + slippage_cost

    sl_pct = sl_mult * atr_pct
    if sl_pct <= 0:
        return 0.0

    return total_cost_pct / sl_pct


def compute_mfe_mae_for_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    side: int,
    atr_i: float,
    sl_mult: float = 1.5,
    horizon: int = 16,
) -> tuple:
    """Compute MFE and MAE in R-units for a trade at index i.

    MFE = max favorable excursion in R-units (positive)
    MAE = max adverse excursion in R-units (positive = how far against)

    R-unit = price distance / (sl_mult * ATR).

    Returns (mfe_r, mae_r).  Both >= 0.
    """
    n = len(closes)
    entry = closes[i]
    a = atr_i
    if np.isnan(a) or a <= 0:
        a = entry * 0.005
    sl_dist = sl_mult * a
    if sl_dist <= 0:
        return (0.0, 0.0)

    max_favorable = 0.0
    max_adverse = 0.0

    for j in range(1, horizon + 1):
        idx = i + j
        if idx >= n:
            break
        if side > 0:
            fav = highs[idx] - entry
            adv = entry - lows[idx]
        else:
            fav = entry - lows[idx]
            adv = highs[idx] - entry
        max_favorable = max(max_favorable, fav)
        max_adverse = max(max_adverse, adv)

    mfe_r = max_favorable / sl_dist
    mae_r = max_adverse / sl_dist
    return (mfe_r, mae_r)


def compute_soft_quality(
    mfe_r: float,
    mae_r: float,
    cost_r: float,
    temperature: float = 2.0,
) -> float:
    """Compute soft quality score and y_soft label.

    quality = net_mfe_r - mae_r  where net_mfe_r = mfe_r - cost_r
    y_soft  = sigmoid(quality / temperature)

    Higher quality => y_soft closer to 1.
    """
    net_mfe_r = mfe_r - cost_r
    quality = net_mfe_r - mae_r
    y_soft = 1.0 / (1.0 + np.exp(-quality / max(temperature, 0.01)))
    return float(y_soft)


def bidirectional_outcome_for_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    atr_i: float,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 16,
    r_min_expiry: float = 1.0,
):
    """Compute BOTH long and short triple-barrier outcomes at index i.

    No HTF gating — evaluates both directions unconditionally.

    Returns:
        dict with keys:
            long_outcome, long_r, long_hit_type,
            short_outcome, short_r, short_hit_type,
            y_quality (int 0/1), y_dir (int 0=SHORT, 1=LONG),
            y_dir_conf (float in [0,1]),
            best_outcome_r (float)
    """
    long_outcome, long_r = triple_barrier_outcome_for_index(
        highs, lows, closes, i, +1, atr_i,
        tp_mult, sl_mult, horizon, r_min_expiry,
    )
    short_outcome, short_r = triple_barrier_outcome_for_index(
        highs, lows, closes, i, -1, atr_i,
        tp_mult, sl_mult, horizon, r_min_expiry,
    )

    long_hit = long_outcome in ("TP", "EXP_WIN")
    short_hit = short_outcome in ("TP", "EXP_WIN")
    best_r = max(long_r, short_r)

    y_quality = 1 if (long_hit or short_hit or best_r >= 0.0) else 0
    if best_r < 0 and not long_hit and not short_hit:
        y_quality = 0

    if long_r > short_r:
        y_dir = 1
    elif short_r > long_r:
        y_dir = 0
    else:
        y_dir = 1

    margin = float(long_r - short_r)
    margin_clamped = max(-2.0, min(2.0, margin))
    y_dir_conf = 1.0 / (1.0 + np.exp(-margin_clamped))

    return {
        'long_outcome': long_outcome,
        'long_r': float(long_r),
        'short_outcome': short_outcome,
        'short_r': float(short_r),
        'y_quality': int(y_quality),
        'y_dir': int(y_dir),
        'y_dir_conf': float(y_dir_conf),
        'best_outcome_r': float(best_r),
    }


def triple_barrier_outcome_detailed(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    side: int,
    atr_i: float,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 16,
    r_min_expiry: float = 1.0,
):
    """Detailed triple-barrier outcome with t_hit, MFE, MAE diagnostics.

    Returns dict:
        hit_type: 'TP', 'SL', 'EXP_WIN', 'EXP_LOSS'
        r: realized R-multiple
        t_hit: bars until barrier hit (1..horizon)
        mfe: max favorable excursion in price units over full horizon
        mae: max adverse excursion in price units over full horizon
    """
    n = len(closes)
    entry = closes[i]
    a = atr_i
    if np.isnan(a) or a <= 0:
        a = entry * 0.005

    tp_dist = tp_mult * a
    sl_dist = sl_mult * a

    if side > 0:
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist

    hit_type = None
    t_hit = horizon
    r_val = 0.0
    max_favorable = 0.0
    max_adverse = 0.0

    for j in range(1, horizon + 1):
        idx = i + j
        if idx >= n:
            break

        if side > 0:
            fav = highs[idx] - entry
            adv = entry - lows[idx]
            tp_hit = highs[idx] >= tp_price
            sl_hit = lows[idx] <= sl_price
        else:
            fav = entry - lows[idx]
            adv = highs[idx] - entry
            tp_hit = lows[idx] <= tp_price
            sl_hit = highs[idx] >= sl_price

        max_favorable = max(max_favorable, fav)
        max_adverse = max(max_adverse, adv)

        if hit_type is None:
            if tp_hit and sl_hit:
                if side > 0:
                    tp_exc = highs[idx] - entry
                    sl_exc = entry - lows[idx]
                else:
                    tp_exc = entry - lows[idx]
                    sl_exc = highs[idx] - entry
                if tp_exc >= sl_exc:
                    hit_type = 'TP'
                    r_val = tp_mult / sl_mult
                    t_hit = j
                else:
                    hit_type = 'SL'
                    r_val = -1.0
                    t_hit = j
            elif tp_hit:
                hit_type = 'TP'
                r_val = tp_mult / sl_mult
                t_hit = j
            elif sl_hit:
                hit_type = 'SL'
                r_val = -1.0
                t_hit = j

    if hit_type is None:
        end_idx = min(i + horizon, n - 1)
        exit_price = closes[end_idx]
        if side > 0:
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price
        r_at_expiry = pnl / sl_dist if sl_dist > 0 else 0.0
        if r_at_expiry >= r_min_expiry:
            hit_type = 'EXP_WIN'
        else:
            hit_type = 'EXP_LOSS'
        r_val = r_at_expiry
        t_hit = horizon

    return {
        'hit_type': hit_type,
        'r': float(r_val),
        't_hit': int(t_hit),
        'mfe': float(max_favorable),
        'mae': float(max_adverse),
    }


def bidirectional_outcome_v47_for_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    atr_i: float,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 16,
    r_min_expiry_strict: float = 1.0,
):
    """v4.7.1 bidirectional labeling with TP quality score.

    For each direction (LONG, SHORT), evaluates detailed outcomes
    including t_hit, MFE, MAE. Computes a continuous TP quality score
    for TP-first bars to enable meaningful auto-balance.

    TP quality score (for TP-first bars only):
        q = 0.50*(1 - t_hit/horizon) + 0.25*clamp(mfe/tp_dist, 0, 1) + 0.25*(1 - clamp(mae/sl_dist, 0, 1))

    Returns dict with all prior keys plus:
      - long_tp_first, short_tp_first (bool)
      - tp_quality: float in [0,1] for the best-side TP (NaN if not TP-first)
      - long_t_hit, short_t_hit, long_mfe, short_mfe, long_mae, short_mae
    """
    long_d = triple_barrier_outcome_detailed(
        highs, lows, closes, i, +1, atr_i,
        tp_mult, sl_mult, horizon, r_min_expiry_strict,
    )
    short_d = triple_barrier_outcome_detailed(
        highs, lows, closes, i, -1, atr_i,
        tp_mult, sl_mult, horizon, r_min_expiry_strict,
    )

    long_tp_first = long_d['hit_type'] == 'TP'
    short_tp_first = short_d['hit_type'] == 'TP'
    long_exp_win = long_d['hit_type'] == 'EXP_WIN' and long_d['r'] >= r_min_expiry_strict
    short_exp_win = short_d['hit_type'] == 'EXP_WIN' and short_d['r'] >= r_min_expiry_strict

    best_r = max(long_d['r'], short_d['r'])

    if long_d['r'] >= short_d['r']:
        best_side_tp_first = long_tp_first
        best_side_exp_win = long_exp_win
        best_d = long_d
        y_dir = 1
    else:
        best_side_tp_first = short_tp_first
        best_side_exp_win = short_exp_win
        best_d = short_d
        y_dir = 0

    tp_dist = tp_mult * atr_i if atr_i > 0 else 1.0
    sl_dist = sl_mult * atr_i if atr_i > 0 else 1.0

    tp_quality = float('nan')
    if best_side_tp_first:
        ttp = best_d['t_hit']
        mfe_frac = min(max(best_d['mfe'] / tp_dist, 0.0), 1.5)
        mae_frac = min(max(best_d['mae'] / sl_dist, 0.0), 1.5)
        q = 0.50 * (1.0 - ttp / horizon) + 0.25 * min(mfe_frac, 1.0) + 0.25 * (1.0 - min(mae_frac, 1.0))

        if best_d['mae'] > 0.6 * sl_dist:
            q *= 0.5

        n_bars = len(closes)
        if i + 2 < n_bars and atr_i > 0:
            entry = closes[i]
            best_side = +1 if y_dir == 1 else -1
            early_move = 0.0
            for jj in range(1, min(3, n_bars - i)):
                if best_side > 0:
                    early_move = max(early_move, highs[i + jj] - entry)
                else:
                    early_move = max(early_move, entry - lows[i + jj])
            if early_move > 0.3 * atr_i:
                q *= 1.3

        tp_quality = max(0.0, min(1.0, q))

    y_quality = 0
    best_outcome_type = 'LOSS'

    if best_side_tp_first:
        y_quality = 1
        best_outcome_type = 'TP_FIRST'
    elif best_side_exp_win:
        y_quality = 1
        best_outcome_type = 'EXPIRY_STRONG'

    if y_quality == 0:
        best_outcome_type = 'WEAK' if best_r < 0 else 'LOSS'

    margin = float(long_d['r'] - short_d['r'])
    margin_clamped = max(-2.0, min(2.0, margin))
    y_dir_conf = 1.0 / (1.0 + np.exp(-margin_clamped))

    return {
        'long_outcome': long_d['hit_type'],
        'long_r': float(long_d['r']),
        'long_tp_first': long_tp_first,
        'long_t_hit': long_d['t_hit'],
        'long_mfe': long_d['mfe'],
        'long_mae': long_d['mae'],
        'short_outcome': short_d['hit_type'],
        'short_r': float(short_d['r']),
        'short_tp_first': short_tp_first,
        'short_t_hit': short_d['t_hit'],
        'short_mfe': short_d['mfe'],
        'short_mae': short_d['mae'],
        'y_quality': int(y_quality),
        'y_dir': int(y_dir),
        'y_dir_conf': float(y_dir_conf),
        'best_outcome_r': float(best_r),
        'best_outcome_type': best_outcome_type,
        'tp_quality': float(tp_quality),
    }


def compute_htf_score_target(
    h1_trend_sign: float,
    h4_trend_sign: float,
) -> int:
    """Compute HTF score classification target (0-3) from past HTF features.

    Scoring:
        0 = no trend (both flat)
        1 = weak (one timeframe has trend)
        2 = moderate (both have trend, disagree)
        3 = strong (both agree on direction)
    """
    h1_active = abs(h1_trend_sign) > 0
    h4_active = abs(h4_trend_sign) > 0

    if not h1_active and not h4_active:
        return 0
    if h1_active != h4_active:
        return 1
    if h1_trend_sign != h4_trend_sign:
        return 2
    return 3


def triple_barrier_batch(
    df: pd.DataFrame,
    indices: np.ndarray,
    sides: np.ndarray,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 16,
    r_min_expiry: float = 1.0,
    atr: np.ndarray = None,
):
    """Run triple-barrier simulation for a batch of trade entries.
    
    Args:
        df: OHLCV DataFrame
        indices: Array of entry bar indices
        sides: Array of trade directions (+1 LONG, -1 SHORT)
        tp_mult/sl_mult: ATR multipliers
        horizon: Max bars to hold
        r_min_expiry: Min R at expiry for win
        atr: Pre-computed ATR array (if None, computed internally)
    
    Returns:
        (outcomes, r_values) - arrays of same length as indices
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    
    if atr is None:
        atr = compute_atr_14(df)
    
    n_trades = len(indices)
    outcomes = np.empty(n_trades, dtype=object)
    r_values = np.empty(n_trades, dtype=np.float64)
    
    for k in range(n_trades):
        i = int(indices[k])
        side = int(sides[k])
        atr_i = float(atr[i])
        
        outcome, r = triple_barrier_outcome_for_index(
            highs, lows, closes, i, side, atr_i,
            tp_mult, sl_mult, horizon, r_min_expiry,
        )
        outcomes[k] = outcome
        r_values[k] = r
    
    return outcomes, r_values
