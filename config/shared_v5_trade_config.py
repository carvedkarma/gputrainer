"""shared_v5_trade_config.py — Single source of truth for V5 trade parameters.

Both v5_train.py (walk-forward simulation) and live_runner.py (live inference)
load defaults from this module.  CLI arguments always take priority over these
defaults; the config is the authoritative baseline that prevents train/live
parameter divergence.

Usage
-----
from config.shared_v5_trade_config import V5TradeDefaults, load_shared_defaults

defaults = load_shared_defaults()
# defaults.score_threshold, defaults.min_p_side, … etc.
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class V5TradeDefaults:
    # ── Scoring / signal gate ────────────────────────────────────────────────
    score_threshold: float = 0.001         # Minimum composite V5 score to enter (post NaN-fix: scores are 0.001-0.002)
    score_lambda: float = 0.30            # Task #56 A1: Lambda lowered 0.50→0.30; break-even p_side 0.333→0.231
    min_mu_r_score: float = 0.0           # Minimum expected-return component (0.0 = disabled; 0.005 caused NaN cascade after mu_debias EMA converged)
    min_p_side: float = 0.0               # Minimum directional probability
    min_p_short: float = 0.0              # Separate floor for short p_short head

    # ── Costs / slippage ────────────────────────────────────────────────────
    slippage_base_bps: float = 6.0        # Bitget taker fee ~3 bps each side = 6 bps round-trip
    slippage_impact_mult: float = 0.0     # Impact multiplier on top of base

    # ── Correlation penalty ─────────────────────────────────────────────────
    corr_thresh: float = 0.70             # Correlation above which penalty starts
    corr_window_days: int = 30            # Rolling window for daily-R correlation
    corr_min_aligned_days: int = 10       # Minimum aligned days for reliable corr

    # ── Cooldown ─────────────────────────────────────────────────────────────
    cooldown_bars: int = 4                # Per-symbol cooldown after trade close

    # ── Symbol kill / recovery ───────────────────────────────────────────────
    per_symbol_r_kill: Optional[float] = None   # Kill threshold (R units, negative)
    kill_recovery_bars: int = 48          # Cooldown bars before recovery check
    kill_recovery_r_threshold: float = 2.0  # R improvement needed above kill floor
    kill_hysteresis_r: float = 1.0        # Extra buffer to prevent rapid re-kill

    # ── Position sizing ──────────────────────────────────────────────────────
    min_size_mult: float = 0.25           # Minimum size multiplier
    max_size_mult: float = 2.5            # Maximum size multiplier
    size_floor: float = 0.5              # Floor applied after all modifiers
    kelly_fraction: float = 0.25         # Fractional Kelly (legacy, kept for compat)
    score_buffer_size: int = 200          # Rolling buffer for score-percentile sizing

    # ── Live halt switches ───────────────────────────────────────────────────
    halt_on_data_staleness: bool = True
    max_data_staleness_seconds: float = 300.0   # 5 minutes
    halt_on_api_errors: bool = True
    max_consecutive_api_errors: int = 5
    max_daily_loss_r: Optional[float] = None    # Hard daily R loss limit


def load_shared_defaults() -> V5TradeDefaults:
    """Return the shared defaults and log a config summary."""
    cfg = V5TradeDefaults()
    log.info("[SHARED_CFG] V5 shared trade defaults loaded:")
    log.info(f"  score_threshold={cfg.score_threshold}  score_lambda={cfg.score_lambda}  "
             f"min_mu_r={cfg.min_mu_r_score}  min_p_side={cfg.min_p_side}  "
             f"min_p_short={cfg.min_p_short}")
    log.info(f"  slippage_base_bps={cfg.slippage_base_bps}  "
             f"corr_thresh={cfg.corr_thresh}  cooldown={cfg.cooldown_bars}")
    log.info(f"  per_symbol_r_kill={cfg.per_symbol_r_kill}  "
             f"kill_recovery_bars={cfg.kill_recovery_bars}  "
             f"kill_recovery_r_threshold={cfg.kill_recovery_r_threshold}")
    log.info(f"  sizing: min={cfg.min_size_mult}  max={cfg.max_size_mult}  "
             f"floor={cfg.size_floor}  score_buffer={cfg.score_buffer_size}")
    log.info(f"  halt: data_staleness={cfg.halt_on_data_staleness}/"
             f"{cfg.max_data_staleness_seconds}s  "
             f"api_errors={cfg.halt_on_api_errors}/{cfg.max_consecutive_api_errors}  "
             f"max_daily_loss_r={cfg.max_daily_loss_r}")
    return cfg
