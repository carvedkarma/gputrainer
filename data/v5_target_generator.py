"""
V5.0.1 Target Generator: All Continuous Targets in R-Units

Computes for each bar (given a horizon):
- ret_R: close-to-close return at horizon, in R-units (normalized by ATR)
- mfe_R: max favorable excursion within horizon (in R-units)
- mae_R: max adverse excursion within horizon (in R-units)
- vol_h: realized volatility within horizon (std of bar-to-bar returns)

All targets use ONLY future bars within [i+1 .. i+horizon] -- no leakage.
All continuous targets (ret_R, mfe_R, mae_R) are in R-units for unit consistency.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from data.common import compute_atr, compute_adaptive_horizons, compute_vol_adjusted_sl, log_adaptive_horizon_diagnostics
from data.candidate_generator import compute_adx

logger = logging.getLogger(__name__)


@dataclass
class V5TargetConfig:
    horizon: int = 16
    atr_period: int = 14
    hold_target: float = 0.30
    mfe_min: float = 0.05
    adaptive_horizon: bool = True
    horizon_min: int = 8
    horizon_max: int = 48
    median_window: int = 50
    high_vol_threshold: float = 1.3
    sl_boost: float = 1.15


def build_v5_targets(
    df: pd.DataFrame,
    horizon: int = 16,
    atr_period: int = 14,
    hold_target: float = 0.30,
    mfe_min_r: float = 0.05,
    barrier_outcomes: Optional[Dict[str, np.ndarray]] = None,
    adaptive_horizon: bool = True,
    horizon_min: int = 8,
    horizon_max: int = 48,
    median_window: int = 50,
    high_vol_threshold: float = 1.3,
    sl_boost: float = 1.15,
) -> Dict[str, np.ndarray]:
    """Build v5 continuous targets from OHLCV data -- ALL in R-units.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        horizon: forward-looking window in bars (base horizon when adaptive)
        atr_period: ATR lookback for R-unit normalization
        hold_target: target fraction of HOLD labels (adaptive deadzone)
        mfe_min_r: minimum MFE in R-units required to classify as non-HOLD
        barrier_outcomes: optional dict from generate_v5_sweep_outcomes with
            r_long, r_short, out_long, out_short. When provided, action labels
            are derived from barrier outcomes instead of ret_R sign, aligning
            training targets with evaluation.
        adaptive_horizon: if True, compute per-bar volatility-adaptive horizon
        horizon_min: minimum adaptive horizon (default 8 = 2h on 15m)
        horizon_max: maximum adaptive horizon (default 48 = 12h on 15m)
        median_window: rolling window for median ATR computation
        high_vol_threshold: ATR ratio threshold for high-vol SL adjustment
        sl_boost: SL multiplier boost factor in high-vol conditions

    Returns:
        Dict with keys: ret_R, mfe_R, mae_R, vol_h, action_label, valid_mask, atr,
        effective_horizons, effective_sl_mults
        ret_R, mfe_R, mae_R are ALL in R-units (price_change / ATR).
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)
    eps = 1e-10

    if adaptive_horizon:
        effective_horizons = compute_adaptive_horizons(
            atr, base_horizon=horizon, median_window=median_window,
            min_horizon=horizon_min, max_horizon=horizon_max,
        )
    else:
        effective_horizons = np.full(n, horizon, dtype=np.int64)

    ret_R = np.full(n, np.nan, dtype=np.float64)
    mfe_R = np.full(n, np.nan, dtype=np.float64)
    mae_R = np.full(n, np.nan, dtype=np.float64)
    mfe_R_long = np.full(n, np.nan, dtype=np.float64)
    mae_R_long = np.full(n, np.nan, dtype=np.float64)
    mfe_R_short = np.full(n, np.nan, dtype=np.float64)
    mae_R_short = np.full(n, np.nan, dtype=np.float64)
    vol_h = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        h_i = int(effective_horizons[i])
        if i + h_i >= n:
            continue
        entry_price = closes[i]
        if entry_price <= 0 or atr[i] <= 0:
            continue

        future_closes = closes[i + 1: i + 1 + h_i]
        future_highs = highs[i + 1: i + 1 + h_i]
        future_lows = lows[i + 1: i + 1 + h_i]

        if len(future_closes) == 0:
            continue

        exit_price = future_closes[-1]
        ret_R[i] = (exit_price - entry_price) / (atr[i] + eps)

        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        long_mfe = (max_high - entry_price) / (atr[i] + eps)
        long_mae = (entry_price - min_low) / (atr[i] + eps)
        short_mfe = (entry_price - min_low) / (atr[i] + eps)
        short_mae = (max_high - entry_price) / (atr[i] + eps)

        mfe_R_long[i] = long_mfe
        mae_R_long[i] = long_mae
        mfe_R_short[i] = short_mfe
        mae_R_short[i] = short_mae

        if long_mfe >= short_mfe:
            mfe_R[i] = long_mfe
            mae_R[i] = long_mae
        else:
            mfe_R[i] = short_mfe
            mae_R[i] = short_mae

        bar_returns = np.diff(np.log(np.maximum(future_closes, eps)))
        if len(bar_returns) > 1:
            vol_h[i] = np.std(bar_returns, ddof=1)
        else:
            vol_h[i] = 0.0

    if adaptive_horizon:
        dummy_sl_mults = compute_vol_adjusted_sl(
            atr, sl_mult=1.5, median_window=median_window,
            high_vol_threshold=high_vol_threshold, sl_boost=sl_boost,
        )
        log_adaptive_horizon_diagnostics(effective_horizons, dummy_sl_mults, horizon, 1.5)

    valid_mask = (np.isfinite(ret_R) & np.isfinite(mfe_R) & np.isfinite(mae_R)
                  & np.isfinite(vol_h) & (atr > 0))

    adx = compute_adx(df)

    abs_ret_valid = np.abs(ret_R[valid_mask])
    if len(abs_ret_valid) > 0:
        deadzone_R_default = float(np.percentile(abs_ret_valid, hold_target * 100))
        deadzone_R_trending = float(np.percentile(abs_ret_valid, 20))
        deadzone_R_choppy = float(np.percentile(abs_ret_valid, 50))
    else:
        deadzone_R_default = 0.1
        deadzone_R_trending = 0.05
        deadzone_R_choppy = 0.2

    deadzone_per_bar = np.full(n, deadzone_R_default, dtype=np.float64)
    n_trending = 0
    n_choppy = 0
    n_normal = 0
    for i in range(n):
        if not valid_mask[i]:
            continue
        if adx[i] > 25:
            deadzone_per_bar[i] = deadzone_R_trending
            n_trending += 1
        elif adx[i] < 18:
            deadzone_per_bar[i] = deadzone_R_choppy
            n_choppy += 1
        else:
            n_normal += 1

    logger.info(f"[V5_TARGETS] Regime-adaptive deadzone: "
                f"trending(ADX>25)={deadzone_R_trending:.4f} n={n_trending}, "
                f"normal={deadzone_R_default:.4f} n={n_normal}, "
                f"choppy(ADX<18)={deadzone_R_choppy:.4f} n={n_choppy}")

    action_label = np.full(n, 0, dtype=np.int64)
    sample_weight = np.ones(n, dtype=np.float32)

    if barrier_outcomes is not None:
        b_r_long = barrier_outcomes['r_long']
        b_r_short = barrier_outcomes['r_short']
        n_barrier_long = 0
        n_barrier_short = 0
        n_barrier_hold = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            rl = b_r_long[i] if np.isfinite(b_r_long[i]) else -999.0
            rs = b_r_short[i] if np.isfinite(b_r_short[i]) else -999.0
            dz = deadzone_per_bar[i]

            side_conf = min(1.0, abs(rl - rs) / 1.0)
            sample_weight[i] = float(side_conf)

            long_positive = rl > 0
            short_positive = rs > 0
            if not long_positive and not short_positive:
                action_label[i] = 0
                n_barrier_hold += 1
            elif long_positive and not short_positive:
                if rl >= dz:
                    action_label[i] = 1
                    n_barrier_long += 1
                else:
                    action_label[i] = 0
                    n_barrier_hold += 1
            elif short_positive and not long_positive:
                if rs >= dz:
                    action_label[i] = 2
                    n_barrier_short += 1
                else:
                    action_label[i] = 0
                    n_barrier_hold += 1
            else:
                BOTH_POS_MARGIN = 1.05
                if rl > rs * BOTH_POS_MARGIN:
                    if rl >= dz:
                        action_label[i] = 1
                        n_barrier_long += 1
                    else:
                        action_label[i] = 0
                        n_barrier_hold += 1
                elif rs > rl * BOTH_POS_MARGIN:
                    if rs >= dz:
                        action_label[i] = 2
                        n_barrier_short += 1
                    else:
                        action_label[i] = 0
                        n_barrier_hold += 1
                else:
                    action_label[i] = 0
                    n_barrier_hold += 1
        logger.info(f"[V5_TARGETS] BARRIER-BASED labels: LONG={n_barrier_long} SHORT={n_barrier_short} HOLD={n_barrier_hold}")

        # Align mu_R regression target (ret_R) with barrier action labels.
        # Without this alignment, ret_R = close-to-close at horizon end, which often
        # disagrees with the barrier outcome label (TP/SL simulation).  Contradictory
        # gradients on the same bar corrupt the NLL loss head and cause:
        #   1. mu_r_correlation → -0.11 (opposite to true signal)
        #   2. sigma collapse → 0.001-0.002 score range
        #   3. Side-aware scoring gate fires even on good signals.
        #
        # Sign convention for barrier-aligned ret_R:
        #   LONG  → ret_R = +r_long   (positive: price rose, long TP hit)
        #   SHORT → ret_R = -r_short  (negative: price fell, short TP hit)
        #   HOLD  → ret_R = 0.0
        #
        # Why -r_short not +r_short: r_short is the profit of the short trade
        # (positive when short wins). Negating it makes mu_R negative for
        # bearish signals, consistent with side_aware_scoring (mu_R<0 required
        # for shorts) and producing aligned gradients from both CE and NLL.
        n_aligned_long = 0
        n_aligned_short = 0
        n_aligned_hold = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            lbl = action_label[i]
            if lbl == 1:  # LONG
                rl_val = b_r_long[i] if np.isfinite(b_r_long[i]) else 0.0
                ret_R[i] = float(rl_val)
                n_aligned_long += 1
            elif lbl == 2:  # SHORT
                rs_val = b_r_short[i] if np.isfinite(b_r_short[i]) else 0.0
                ret_R[i] = -float(rs_val)
                n_aligned_short += 1
            else:  # HOLD
                ret_R[i] = 0.0
                n_aligned_hold += 1
        _valid_ret = ret_R[valid_mask]
        _ret_mean = float(np.nanmean(_valid_ret)) if len(_valid_ret) > 0 else 0.0
        _ret_std  = float(np.nanstd(_valid_ret))  if len(_valid_ret) > 0 else 0.0
        logger.info(
            f"[V5_TARGETS] barrier_aligned=True  "
            f"LONG={n_aligned_long} SHORT={n_aligned_short} HOLD={n_aligned_hold} "
            f"ret_R_mean={_ret_mean:.4f} ret_R_std={_ret_std:.4f} "
            f"(LONG→+r_long, SHORT→-r_short, HOLD→0.0)"
        )
    else:
        for i in range(n):
            if not valid_mask[i]:
                continue
            dz = deadzone_per_bar[i]
            if ret_R[i] > 0:
                side_mfe = mfe_R_long[i]
            else:
                side_mfe = mfe_R_short[i]
            if np.abs(ret_R[i]) < dz or side_mfe < mfe_min_r:
                action_label[i] = 0
            elif ret_R[i] > 0:
                action_label[i] = 1
            else:
                action_label[i] = 2

    valid_weights = sample_weight[valid_mask]
    logger.info(f"[V5_TARGETS] Side-confidence weights: mean={np.mean(valid_weights):.3f} "
                f"median={np.median(valid_weights):.3f} min={np.min(valid_weights):.3f}")

    n_valid = int(np.sum(valid_mask))
    n_hold = int(np.sum(action_label[valid_mask] == 0))
    n_long = int(np.sum(action_label[valid_mask] == 1))
    n_short = int(np.sum(action_label[valid_mask] == 2))

    logger.info(f"[V5_TARGETS] horizon={horizon} valid={n_valid}/{n} "
                f"action: HOLD={n_hold} ({n_hold/max(n_valid,1):.1%}) "
                f"LONG={n_long} ({n_long/max(n_valid,1):.1%}) "
                f"SHORT={n_short} ({n_short/max(n_valid,1):.1%})")

    if n_valid > 0:
        ret_valid = ret_R[valid_mask]
        mfe_valid = mfe_R[valid_mask]
        mae_valid = mae_R[valid_mask]
        vol_valid = vol_h[valid_mask]
        logger.info(f"[V5_TARGETS] ret_R: mean={np.mean(ret_valid):.4f} std={np.std(ret_valid):.4f} "
                     f"p5={np.percentile(ret_valid,5):.4f} p95={np.percentile(ret_valid,95):.4f}")
        logger.info(f"[V5_TARGETS] mfe_R: mean={np.mean(mfe_valid):.3f} mae_R: mean={np.mean(mae_valid):.3f} "
                     f"vol_h: mean={np.mean(vol_valid):.6f}")

    assert np.all(np.isfinite(ret_R[valid_mask])), "ret_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mfe_R[valid_mask])), "mfe_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mae_R[valid_mask])), "mae_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(vol_h[valid_mask])), "vol_h contains NaN/Inf in valid region"

    # UNIT INVARIANT: all *_R arrays (ret_R, mfe_R, mae_R, etc.) are in R-units,
    # meaning each value = price_delta / atr[i]. Dividing by 'atr' a second time
    # (e.g. in the loss function) would produce price_delta / atr^2 — a unit error.
    # Consumers (compute_v5_loss, etc.) MUST treat *_R arrays as already ATR-normalized.
    return {
        'ret_R': ret_R.astype(np.float32),
        'mfe_R': mfe_R.astype(np.float32),
        'mae_R': mae_R.astype(np.float32),
        'mfe_R_long': mfe_R_long.astype(np.float32),
        'mae_R_long': mae_R_long.astype(np.float32),
        'mfe_R_short': mfe_R_short.astype(np.float32),
        'mae_R_short': mae_R_short.astype(np.float32),
        'vol_h': vol_h.astype(np.float32),
        'action_label': action_label,
        'valid_mask': valid_mask,
        'atr': atr.astype(np.float32),
        'deadzone_R': deadzone_R_default,
        'deadzone_R_trending': deadzone_R_trending,
        'deadzone_R_choppy': deadzone_R_choppy,
        'sample_weight': sample_weight,
        'effective_horizons': effective_horizons,
    }


def build_barrier_preset_labels(
    df: pd.DataFrame,
    presets: list,
    horizon: int = 16,
    atr_period: int = 14,
    temperature: float = 1.0,
    adaptive_horizon: bool = True,
    horizon_min: int = 8,
    horizon_max: int = 48,
    median_window: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build soft barrier selection labels from realized outcomes.

    For each bar, compute realized R under each preset, then produce:
    - oracle_idx: argmax preset (best realized R) -- research only
    - soft_target: softmax(R_preset / temperature) -- for learnable mode

    Supports volatility-adaptive horizons per bar.

    Returns:
        oracle_idx: (N,) int64 array of best preset index
        soft_target: (N, n_presets) float32 array of soft probabilities
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
    else:
        effective_horizons = np.full(n, horizon, dtype=np.int64)

    n_presets = len(presets)
    realized_r = np.full((n, n_presets), np.nan, dtype=np.float64)

    for pi, preset in enumerate(presets):
        tp_mult = preset['tp_mult']
        sl_mult = preset['sl_mult']

        for i in range(n):
            h_i = int(effective_horizons[i])
            if i + h_i >= n:
                continue
            if atr[i] <= 0 or closes[i] <= 0:
                continue

            entry = closes[i]
            tp_dist = atr[i] * tp_mult
            sl_dist = atr[i] * sl_mult

            long_tp = entry + tp_dist
            long_sl = entry - sl_dist
            short_tp = entry - tp_dist
            short_sl = entry + sl_dist

            long_r = np.nan
            short_r = np.nan

            for j in range(i + 1, min(i + 1 + h_i, n)):
                if highs[j] >= long_tp:
                    long_r = tp_mult / sl_mult
                    break
                if lows[j] <= long_sl:
                    long_r = -1.0
                    break
            if np.isnan(long_r):
                long_r = (closes[min(i + h_i, n - 1)] - entry) / (atr[i] * sl_mult)

            for j in range(i + 1, min(i + 1 + h_i, n)):
                if lows[j] <= short_tp:
                    short_r = tp_mult / sl_mult
                    break
                if highs[j] >= short_sl:
                    short_r = -1.0
                    break
            if np.isnan(short_r):
                short_r = (entry - closes[min(i + h_i, n - 1)]) / (atr[i] * sl_mult)

            realized_r[i, pi] = max(long_r, short_r)

    realized_r_clean = np.nan_to_num(realized_r, nan=-999.0)
    oracle_idx = np.argmax(realized_r_clean, axis=1).astype(np.int64)

    shifted = realized_r_clean / max(temperature, 1e-6)
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    exp_r = np.exp(shifted)
    soft_target = exp_r / (np.sum(exp_r, axis=1, keepdims=True) + 1e-8)
    soft_target = soft_target.astype(np.float32)

    logger.info(f"[V5_BARRIER_LABELS] {n_presets} presets, oracle distribution: "
                + ", ".join(f"{presets[p]['label']}={np.mean(oracle_idx==p):.1%}" for p in range(n_presets)))

    return oracle_idx, soft_target
