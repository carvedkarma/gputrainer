"""
Candidate Engine for Distributional Trade Forecaster (v4.9.1)

Filters bars to identify tradeable candidates using lightweight rules:
- ATR / realized volatility above minimum threshold
- Ultra-low-vol chop exclusion (unless regime allows)
- Optional breakout trigger (close > n-bar high)
- Optional mean-reversion trigger (z-score extremes)
- Fee/spread gate: expected move must exceed round-trip cost

The candidate mask is used by training + validation sweeps so that
ranking/scoring happens ONLY on candidate bars, not all bars.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CandidateConfig:
    enabled: bool = True

    min_atr_pct: float = 0.0015
    min_realized_vol_pct: float = 0.001
    realized_vol_lookback: int = 20

    chop_filter_enabled: bool = True
    chop_adx_threshold: float = 18.0
    chop_atr_rank_threshold: float = 0.20

    breakout_enabled: bool = True
    breakout_lookback: int = 20

    mean_reversion_enabled: bool = True
    mr_zscore_threshold: float = 2.0
    mr_lookback: int = 30

    fee_gate_enabled: bool = True
    round_trip_cost: float = 0.0009
    fee_gate_atr_mult: float = 1.5

    min_candidate_rate: float = 0.10
    max_candidate_rate: float = 0.80

    target_candidate_rate: float = 0.40
    auto_relax_min_rate: float = 0.25

    @classmethod
    def from_cli_args(cls, args) -> "CandidateConfig":
        return cls(
            enabled=getattr(args, 'use_candidates', True),
            min_atr_pct=getattr(args, 'cand_min_atr_pct', 0.0015),
            breakout_enabled=getattr(args, 'cand_breakout', True),
            mean_reversion_enabled=getattr(args, 'cand_mean_reversion', True),
            fee_gate_enabled=getattr(args, 'cand_fee_gate', True),
            round_trip_cost=getattr(args, 'cand_round_trip_cost', 0.0009),
            target_candidate_rate=getattr(args, 'cand_target_rate', 0.40),
            auto_relax_min_rate=getattr(args, 'cand_min_rate', 0.25),
        )


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    from data.common import compute_atr_pct as _compute_atr_pct
    return _compute_atr_pct(df, period)


def compute_realized_vol(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    closes = df['close'].values.astype(np.float64)
    n = len(closes)
    log_returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_returns[i] = np.log(closes[i] / closes[i - 1])

    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        vol[i] = np.std(log_returns[i - lookback + 1:i + 1])
    return vol


def compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(df)

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0

    atr_s = np.zeros(n, dtype=np.float64)
    pdm_s = np.zeros(n, dtype=np.float64)
    mdm_s = np.zeros(n, dtype=np.float64)

    if n > period:
        atr_s[period] = np.sum(tr[1:period + 1])
        pdm_s[period] = np.sum(plus_dm[1:period + 1])
        mdm_s[period] = np.sum(minus_dm[1:period + 1])
        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
            mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]

    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    np.divide(pdm_s, atr_s, out=plus_di, where=atr_s > 0)
    plus_di *= 100.0
    np.divide(mdm_s, atr_s, out=minus_di, where=atr_s > 0)
    minus_di *= 100.0
    di_sum = plus_di + minus_di
    dx = np.zeros(n, dtype=np.float64)
    np.divide(np.abs(plus_di - minus_di), di_sum, out=dx, where=di_sum > 0)
    dx *= 100.0
    assert np.all(np.isfinite(plus_di)), "plus_di contains NaN/Inf"
    assert np.all(np.isfinite(minus_di)), "minus_di contains NaN/Inf"
    assert np.all(np.isfinite(dx)), "dx contains NaN/Inf"

    adx = np.zeros(n, dtype=np.float64)
    start = 2 * period
    if n > start:
        adx[start] = np.mean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def compute_atr_rank(atr_pct: np.ndarray, lookback: int = 100) -> np.ndarray:
    n = len(atr_pct)
    rank = np.full(n, 0.5, dtype=np.float64)
    for i in range(lookback, n):
        window = atr_pct[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank[i] = np.sum(valid < atr_pct[i]) / len(valid)
    return rank


def compute_breakout_mask(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    n = len(closes)
    mask = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        period_high = np.max(highs[i - lookback:i])
        period_low = np.min(lows[i - lookback:i])
        if closes[i] > period_high or closes[i] < period_low:
            mask[i] = True
    return mask


def compute_zscore(df: pd.DataFrame, lookback: int = 30) -> np.ndarray:
    closes = df['close'].values.astype(np.float64)
    n = len(closes)
    zscores = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        window = closes[i - lookback:i]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 1e-10:
            zscores[i] = (closes[i] - mu) / sigma
    return zscores


def generate_candidate_mask(
    df: pd.DataFrame,
    config: CandidateConfig,
    atr_vals: Optional[np.ndarray] = None,
    symbol: str = "UNKNOWN",
) -> Tuple[np.ndarray, Dict]:
    """Generate a boolean candidate mask for all bars.

    Returns:
        candidate_mask: bool array of shape (n,), True = candidate
        diagnostics: dict with rates and filter stats
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)

    if not config.enabled:
        mask = np.ones(n, dtype=bool)
        return mask, {'candidate_rate': 1.0, 'filters_applied': 'none'}

    vol_pass = np.ones(n, dtype=bool)
    atr_pct = compute_atr_pct(df)

    vol_pass &= (atr_pct >= config.min_atr_pct) | np.isnan(atr_pct)

    realized_vol = compute_realized_vol(df, config.realized_vol_lookback)
    vol_pass &= (realized_vol >= config.min_realized_vol_pct) | np.isnan(realized_vol)

    chop_reject = np.zeros(n, dtype=bool)
    if config.chop_filter_enabled:
        adx = compute_adx(df)
        atr_rank = compute_atr_rank(atr_pct)
        chop_reject = (adx < config.chop_adx_threshold) & (atr_rank < config.chop_atr_rank_threshold)
        chop_reject &= ~np.isnan(adx)

    trigger_pass = np.zeros(n, dtype=bool)
    has_any_trigger = False

    if config.breakout_enabled:
        breakout = compute_breakout_mask(df, config.breakout_lookback)
        trigger_pass |= breakout
        has_any_trigger = True

    if config.mean_reversion_enabled:
        zscores = compute_zscore(df, config.mr_lookback)
        mr_trigger = np.abs(zscores) >= config.mr_zscore_threshold
        trigger_pass |= mr_trigger
        has_any_trigger = True

    if not has_any_trigger:
        trigger_pass = np.ones(n, dtype=bool)

    fee_pass = np.ones(n, dtype=bool)
    if config.fee_gate_enabled:
        if atr_vals is not None:
            atr_dollar = atr_vals
        else:
            atr_dollar = atr_pct * closes
        expected_move = atr_dollar * config.fee_gate_atr_mult
        cost_dollar = closes * config.round_trip_cost
        fee_pass = expected_move > cost_dollar
        fee_pass |= np.isnan(expected_move)

    candidate_mask = vol_pass & (~chop_reject) & trigger_pass & fee_pass

    non_nan = ~np.isnan(atr_pct)
    candidate_rate = float(np.nanmean(candidate_mask[non_nan])) if np.any(non_nan) else 0.0
    relaxation_steps = []

    target_rate = config.auto_relax_min_rate

    if candidate_rate < target_rate:
        atr_steps = [0.0012, 0.0010, 0.0007, 0.0005]
        for step_atr in atr_steps:
            if candidate_rate >= target_rate:
                break
            if step_atr < config.min_atr_pct:
                vol_pass_relaxed = (atr_pct >= step_atr) | np.isnan(atr_pct)
                vol_pass_relaxed &= (realized_vol >= config.min_realized_vol_pct) | np.isnan(realized_vol)
                candidate_mask = vol_pass_relaxed & (~chop_reject) & trigger_pass & fee_pass
                candidate_rate = float(np.nanmean(candidate_mask[non_nan])) if np.any(non_nan) else 0.0
                relaxation_steps.append(f"lower_atr_pct={step_atr}")
                logger.info(f"[CANDIDATE_RELAX] {symbol}: lowered min_atr_pct to {step_atr} -> rate={candidate_rate:.3f}")

    if candidate_rate < target_rate and config.chop_filter_enabled:
        candidate_mask = vol_pass & trigger_pass & fee_pass
        if not np.isnan(atr_pct).all():
            candidate_rate = float(np.nanmean(candidate_mask[non_nan]))
        relaxation_steps.append("disable_chop")
        logger.info(f"[CANDIDATE_RELAX] {symbol}: disabled chop filter -> rate={candidate_rate:.3f}")

    if candidate_rate < target_rate and has_any_trigger:
        candidate_mask = vol_pass & fee_pass
        if not np.isnan(atr_pct).all():
            candidate_rate = float(np.nanmean(candidate_mask[non_nan]))
        relaxation_steps.append("disable_triggers")
        logger.info(f"[CANDIDATE_RELAX] {symbol}: disabled trigger gating -> rate={candidate_rate:.3f}")

    if relaxation_steps:
        logger.info(f"[CANDIDATE_RELAX] {symbol}: relaxation path: {' -> '.join(relaxation_steps)} final_rate={candidate_rate:.3f}")

    if candidate_rate > config.max_candidate_rate:
        pass

    diagnostics = {
        'candidate_rate': candidate_rate,
        'total_bars': n,
        'n_candidates': int(candidate_mask.sum()),
        'vol_pass_rate': float(np.nanmean(vol_pass)),
        'chop_reject_rate': float(np.nanmean(chop_reject)) if config.chop_filter_enabled else 0.0,
        'trigger_pass_rate': float(np.nanmean(trigger_pass)) if has_any_trigger else 1.0,
        'fee_pass_rate': float(np.nanmean(fee_pass)) if config.fee_gate_enabled else 1.0,
        'symbol': symbol,
    }

    logger.info(f"[CANDIDATE] {symbol}: rate={candidate_rate:.1%} "
                f"({diagnostics['n_candidates']}/{n} bars) | "
                f"vol_pass={diagnostics['vol_pass_rate']:.1%} "
                f"chop_rej={diagnostics['chop_reject_rate']:.1%} "
                f"trigger={diagnostics['trigger_pass_rate']:.1%} "
                f"fee={diagnostics['fee_pass_rate']:.1%}")

    return candidate_mask, diagnostics


BARRIER_PRESETS = {
    'tight': {'tp_mult': 1.5, 'sl_mult': 1.0, 'label': 'tight'},
    'standard': {'tp_mult': 2.0, 'sl_mult': 1.5, 'label': 'standard'},
    'wide': {'tp_mult': 3.0, 'sl_mult': 1.5, 'label': 'wide'},
    'asymmetric': {'tp_mult': 3.5, 'sl_mult': 1.25, 'label': 'asymmetric'},
    'swing': {'tp_mult': 4.0, 'sl_mult': 2.0, 'label': 'swing'},
    'scalp': {'tp_mult': 1.2, 'sl_mult': 0.8, 'label': 'scalp'},
}

DEFAULT_PRESETS = ['tight', 'standard', 'wide', 'asymmetric']

MULTI_HORIZONS = [8, 16, 32]
DEFAULT_HORIZON = 16


@dataclass
class MultiHorizonConfig:
    horizons: list = field(default_factory=lambda: [8, 16, 32])
    cooldown_per_horizon: bool = False
    unified_cooldown: int = 4

    @classmethod
    def from_cli_args(cls, args) -> "MultiHorizonConfig":
        horizons_str = getattr(args, 'multi_horizons', '8,16,32')
        horizons = [int(h.strip()) for h in horizons_str.split(',')]
        return cls(
            horizons=horizons,
            cooldown_per_horizon=getattr(args, 'cooldown_per_horizon', False),
            unified_cooldown=getattr(args, 'cooldown', 4),
        )


@dataclass
class PresetConfig:
    presets: list = field(default_factory=lambda: DEFAULT_PRESETS)
    preset_defs: dict = field(default_factory=lambda: BARRIER_PRESETS)
    mode: str = "fixed:standard"

    @classmethod
    def from_cli_args(cls, args) -> "PresetConfig":
        presets_str = getattr(args, 'barrier_presets', 'tight,standard,wide,asymmetric')
        presets = [p.strip() for p in presets_str.split(',')]
        mode = getattr(args, 'multi_preset_mode', 'fixed:standard')
        return cls(presets=presets, mode=mode)

    def get_preset_params(self, name: str) -> Dict:
        return self.preset_defs.get(name, self.preset_defs['standard'])

    @property
    def is_oracle(self) -> bool:
        return self.mode == "oracle"

    @property
    def is_fixed(self) -> bool:
        return self.mode.startswith("fixed:")

    @property
    def fixed_preset_name(self) -> str:
        if self.is_fixed:
            return self.mode.split(":", 1)[1]
        return "standard"

    @property
    def is_learnable(self) -> bool:
        return self.mode == "learnable"


def compute_money_score(
    win_probs: np.ndarray,
    e_r: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    horizon_bars: int = 16,
    score_lambda: float = 0.5,
    regime_weights: Optional[np.ndarray] = None,
    use_efficiency: bool = True,
) -> np.ndarray:
    """Enhanced money-score formula.

    base = p_win * q50 - lambda * max(0, -q10)
    efficiency = base / (horizon_bars + 1)
    final = efficiency * regime_weight

    Falls back to original score when use_efficiency=False.
    """
    downside = np.maximum(0.0, -q10)
    base = win_probs * q50 - score_lambda * downside

    if use_efficiency:
        efficiency = base / (horizon_bars + 1)
    else:
        efficiency = base

    if regime_weights is not None:
        final = efficiency * regime_weights
    else:
        final = efficiency

    return final


def compute_kelly_size(
    win_probs: np.ndarray,
    avg_win_r: float,
    avg_loss_r: float,
    min_size: float = 0.0005,
    max_size: float = 0.003,
    kelly_fraction: float = 0.25,
) -> np.ndarray:
    """Kelly-like position sizing from win probability and win/loss magnitudes.

    kelly_f = p_win * avg_win / |avg_loss| - (1 - p_win)
    size = clip(kelly_fraction * kelly_f, min_size, max_size)
    """
    if abs(avg_loss_r) < 1e-8:
        return np.full_like(win_probs, min_size)

    win_loss_ratio = abs(avg_win_r / avg_loss_r)
    kelly_f = win_probs * win_loss_ratio - (1.0 - win_probs)
    kelly_f = np.maximum(kelly_f, 0.0)
    sizes = kelly_fraction * kelly_f
    sizes = np.clip(sizes, min_size, max_size)
    return sizes


@dataclass
class RiskControls:
    daily_loss_limit_r: float = -3.0
    max_concurrent_trades: int = 6
    max_symbol_exposure: int = 3
    skip_high_funding: bool = False
    funding_threshold: float = 0.001

    @classmethod
    def from_cli_args(cls, args) -> "RiskControls":
        return cls(
            daily_loss_limit_r=getattr(args, 'daily_loss_limit', -3.0),
            max_concurrent_trades=getattr(args, 'max_concurrent', 6),
            max_symbol_exposure=getattr(args, 'max_symbol_exposure', 3),
        )


def apply_risk_controls(
    selected_indices: np.ndarray,
    selected_r: np.ndarray,
    selected_symbols: Optional[np.ndarray],
    risk: RiskControls,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply risk controls to selected trades, returning filtered indices and R values."""
    if len(selected_indices) == 0:
        return selected_indices, selected_r

    kept_indices = []
    kept_r = []
    cumulative_r = 0.0
    concurrent = 0
    symbol_counts: Dict[int, int] = {}
    daily_bar_count = 96

    for i, (idx, r_val) in enumerate(zip(selected_indices, selected_r)):
        if i > 0 and (idx - selected_indices[0]) >= daily_bar_count:
            cumulative_r = sum(kept_r[max(0, len(kept_r) - daily_bar_count):])

        if cumulative_r <= risk.daily_loss_limit_r:
            continue

        if concurrent >= risk.max_concurrent_trades:
            continue

        if selected_symbols is not None:
            sym = int(selected_symbols[i])
            if symbol_counts.get(sym, 0) >= risk.max_symbol_exposure:
                continue
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        kept_indices.append(idx)
        kept_r.append(float(r_val))
        cumulative_r += float(r_val)
        concurrent += 1

    if len(kept_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(kept_indices), np.array(kept_r)
