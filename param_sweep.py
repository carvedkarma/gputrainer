#!/usr/bin/env python3
"""
V5 Parameter Sweep — Real walk-forward results using your actual GPU model.

Runs your real run_v5_walk_forward function with multiple configs and prints
a ranked table of WR%, E[R]/trade, Sharpe, and total R.

Usage (on your GPU machine):
    python gpu_trainer/param_sweep.py
    python gpu_trainer/param_sweep.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python gpu_trainer/param_sweep.py --folds 2 --epochs 40 --fast
    python gpu_trainer/param_sweep.py --baseline-only
    python gpu_trainer/param_sweep.py --output results.json

Runtime estimates (RTX 3090, 6 symbols, 3 folds):
    Fast mode (--fast):  ~30 min/config  × 5 configs  = ~2.5 hrs
    Full sweep:          ~40 min/config  × 16 configs = ~11 hrs (run overnight)

Bug fixes applied in this sweep vs your current training command:
    REMOVED:  --v5-side-aware-scoring   (causes NaN cascade via min_mu_r_score)
    REMOVED:  --v5-regime-side-map      (forces 100% SHORT in bear-market folds)
    CHANGED:  --v5-min-threshold 0.02   (was 0.04, now matches model output range)
    ADDED:    min_mu_r_score=0.0        (in v5_train.py compute_v5_scores calls)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ParamSweep")

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    label: str
    tp_mult: float = 3.0
    sl_mult: float = 1.0
    adx_min: float = 18.0
    adx_gate: bool = True
    ema200_soft_mult: Optional[float] = 1.0
    min_threshold: Optional[float] = None  # None = let per-symbol thresholds control (post NaN-fix, scores are 0.001-0.002)
    corr_thresh: float = 0.90
    short_oversample: bool = True
    short_min_fraction: float = 0.35
    per_symbol_threshold: bool = True
    per_side_threshold: bool = True
    trailing_sl: bool = True
    trail_activation: float = 1.5
    trail_distance: float = 1.0
    # Bug-fixed: side_aware_scoring=False and regime_side_map=None are always off
    description: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Sweep grid — baseline + key variations
# ---------------------------------------------------------------------------

def build_sweep_configs(fast: bool = False) -> list:
    """
    Build the list of configs to sweep.

    Fast mode: only 5 configs (baseline + 4 critical variations).
    Full mode: 16 configs covering TP/SL, ADX, EMA soft mult.
    """

    # BASELINE — current recommended command with 3 bug fixes applied
    baseline = SweepConfig(
        label="BASELINE",
        tp_mult=3.0, sl_mult=1.0,
        adx_min=18.0, adx_gate=True,
        ema200_soft_mult=1.0,
        min_threshold=None,
        description="Bug-fixed baseline: no side_aware_scoring, no regime_side_map, min_thr=None (per-symbol thresholds control)",
    )

    if fast:
        # 5 configs: baseline + the 4 most impactful single-variable changes
        return [
            baseline,
            SweepConfig("TP3_SL075", tp_mult=3.0, sl_mult=0.75,
                        description="Tighter SL — cuts losers faster"),
            SweepConfig("TP2_SL10", tp_mult=2.0, sl_mult=1.0,
                        description="Lower TP — more frequent winners"),
            SweepConfig("ADX22", adx_min=22.0,
                        description="Stricter ADX — only strong trends"),
            SweepConfig("EMA_SOFT_05", ema200_soft_mult=0.5,
                        description="Tighter EMA soft gate — 50% size against trend"),
        ]

    # Full grid: TP/SL × ADX × EMA_soft
    configs = [baseline]

    tp_sl_pairs = [
        (3.0, 1.0),   # baseline TP/SL
        (3.0, 0.75),  # tighter SL
        (2.0, 1.0),   # lower TP
        (2.0, 0.75),  # both tighter
    ]
    adx_values = [15.0, 18.0, 22.0]
    ema_soft_values = [0.5, 1.0]

    seen = {baseline.label}
    for tp, sl in tp_sl_pairs:
        for adx in adx_values:
            for ema in ema_soft_values:
                label = f"TP{tp:.0f}_SL{sl:.2f}_ADX{adx:.0f}_EMA{ema:.1f}".replace(".", "p")
                if label in seen:
                    continue
                seen.add(label)
                configs.append(SweepConfig(
                    label=label,
                    tp_mult=tp, sl_mult=sl,
                    adx_min=adx, adx_gate=True,
                    ema200_soft_mult=ema,
                    min_threshold=None,
                    description=f"TP={tp}x SL={sl}x ADX≥{adx} EMA_soft={ema}x",
                ))

    return configs


# ---------------------------------------------------------------------------
# Result collection — parses run_v5_walk_forward return value
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    label: str
    config: SweepConfig
    total_r: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    sharpe: float = 0.0
    n_folds: int = 0
    n_live_folds: int = 0
    max_dd: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    fold_details: list = field(default_factory=list)
    avg_debias_spread_ratio: Optional[float] = None
    n_collapsed_debias: int = 0

    def to_dict(self):
        return {
            "label": self.label,
            "config": self.config.to_dict(),
            "total_r": round(self.total_r, 4),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy": round(self.expectancy, 4),
            "sharpe": round(self.sharpe, 4),
            "n_folds": self.n_folds,
            "n_live_folds": self.n_live_folds,
            "max_dd": round(self.max_dd, 4),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "error": self.error,
            "avg_debias_spread_ratio": self.avg_debias_spread_ratio,
            "n_collapsed_debias": self.n_collapsed_debias,
        }


def _extract_results(wf_result, label: str, cfg: SweepConfig) -> SweepResult:
    """
    Extract metrics from run_v5_walk_forward's return value.

    run_v5_walk_forward returns:
        {
          'folds': [list of fold reports],
          'aggregate': {
              'total_trades': int,
              'total_r': float,
              'avg_expectancy_r': float,
              'n_folds': int,
              'active_folds': int,
              ...
          },
          'per_symbol_summary': {...}
        }

    Each fold report contains:
        total_trades, total_r, win_rate, expectancy_r, sharpe, max_drawdown_r, ...
    """
    result = SweepResult(label=label, config=cfg)

    if wf_result is None:
        result.error = "run_v5_walk_forward returned None (no data or early exit)"
        return result

    if not isinstance(wf_result, dict):
        result.error = f"Unexpected return type: {type(wf_result)}"
        return result

    # Pull from aggregate summary
    agg = wf_result.get("aggregate", {})
    folds = wf_result.get("folds", [])

    result.total_r = float(agg.get("total_r", 0.0))
    result.n_trades = int(agg.get("total_trades", 0))
    result.expectancy = float(agg.get("avg_expectancy_r", 0.0))
    result.n_folds = int(agg.get("n_folds", len(folds)))
    result.n_live_folds = int(agg.get("active_folds", 0))

    # Derive WR and Sharpe from fold reports (not in aggregate)
    live_folds = [f for f in folds if isinstance(f, dict) and f.get("total_trades", 0) > 0]
    if live_folds:
        # Weighted WR by trade count
        total_wins = sum(
            f.get("win_rate", 0.0) * f.get("total_trades", 0) for f in live_folds
        )
        result.win_rate = total_wins / max(result.n_trades, 1)
        # Average Sharpe across live folds
        sharpes = [f.get("sharpe", 0.0) for f in live_folds if "sharpe" in f]
        result.sharpe = float(sum(sharpes) / len(sharpes)) if sharpes else 0.0
        # Max drawdown (worst fold)
        dds = [f.get("max_drawdown_r", 0.0) for f in live_folds if "max_drawdown_r" in f]
        result.max_dd = float(max(dds)) if dds else 0.0

    result.fold_details = [
        {
            "fold": i + 1,
            "n_trades": f.get("total_trades", 0),
            "total_r": round(float(f.get("total_r", 0.0)), 4),
            "win_rate": round(float(f.get("win_rate", 0.0)), 4),
            "expectancy_r": round(float(f.get("expectancy_r", 0.0)), 4),
            "sharpe": round(float(f.get("sharpe", 0.0)), 4),
            "debias_spread_ratio": f.get("debias_spread_ratio"),
            "score_monotonic": f.get("score_monotonic"),
        }
        for i, f in enumerate(folds)
        if isinstance(f, dict)
    ]

    debias_ratios = [
        f["debias_spread_ratio"]
        for f in result.fold_details
        if f.get("debias_spread_ratio") is not None
    ]
    result.avg_debias_spread_ratio = round(
        sum(debias_ratios) / len(debias_ratios), 2
    ) if debias_ratios else None
    result.n_collapsed_debias = sum(
        1 for r in debias_ratios if r < 5.0
    )

    return result


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------

def run_single_config(
    cfg: SweepConfig,
    data_dir: Path,
    device: str,
    symbols: list,
    epochs: int,
    batch_size: int,
    max_folds: int,
    max_trades_per_day: Optional[int] = None,
    target_tpd: float = 6.5,
    replit_url: Optional[str] = None,
) -> SweepResult:
    """Run one config through real V5 walk-forward and return metrics."""
    from train.v5_train import run_v5_walk_forward, V5TPDControllerConfig

    log.info("=" * 70)
    log.info("[SWEEP] Running: %s", cfg.label)
    log.info("[SWEEP]   tp=%.2f  sl=%.2f  adx_min=%.1f  ema_soft=%s  min_thr=%s",
             cfg.tp_mult, cfg.sl_mult, cfg.adx_min,
             f"{cfg.ema200_soft_mult:.2f}" if cfg.ema200_soft_mult is not None else "OFF",
             f"{cfg.min_threshold:.3f}" if cfg.min_threshold is not None else "None(per-sym)")
    log.info("[SWEEP]   short_oversample=%s  per_sym_thr=%s  per_side_thr=%s  trailing_sl=%s",
             cfg.short_oversample, cfg.per_symbol_threshold, cfg.per_side_threshold, cfg.trailing_sl)
    log.info("[SWEEP]   target_tpd=%.1f/sym/day  max_trades_per_day=%s",
             target_tpd, str(max_trades_per_day) if max_trades_per_day else "uncapped")
    log.info("=" * 70)

    t0 = time.time()
    try:
        tpd_cfg = V5TPDControllerConfig(target_tpd=target_tpd)
        wf_result = run_v5_walk_forward(
            data_dir=data_dir,
            device=device,
            symbols=symbols,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            train_months=12,
            test_months=1,
            # TP / SL
            tp_mult=cfg.tp_mult,
            sl_mult=cfg.sl_mult,
            # ADX gate
            adx_gate=cfg.adx_gate,
            adx_min=cfg.adx_min,
            # EMA200 soft gate (None = disabled)
            ema200_soft_mult=cfg.ema200_soft_mult,
            # Threshold
            min_threshold=cfg.min_threshold,
            # Correlation block
            corr_block=True,
            corr_thresh=cfg.corr_thresh,
            corr_same_side_only=True,
            # Short balance
            short_oversample=cfg.short_oversample,
            short_min_fraction=cfg.short_min_fraction,
            # Per-symbol & per-side thresholds
            per_symbol_threshold=cfg.per_symbol_threshold,
            per_side_threshold=cfg.per_side_threshold,
            # Trailing SL
            trailing_sl=cfg.trailing_sl,
            trail_activation=cfg.trail_activation,
            trail_distance=cfg.trail_distance,
            # BUG FIX: side_aware_scoring REMOVED — caused NaN cascade via min_mu_r_score
            side_aware_scoring=False,
            # BUG FIX: regime_side_map REMOVED — caused 100% SHORT in bear folds
            regime_side_map=None,
            # Recency weight (matches recommended command)
            recency_weight=True,
            # Standard settings — adaptive_sizing/conviction_sizing/warm_start intentionally
            # omitted (use defaults=False) to match the known-working training command exactly.
            # CRITICAL FIX: w_action=2.5 (was 2.0 default, matched train_v5_model default bug fix)
            w_action=2.5,
            mu_debias=True,
            per_symbol_cooldown=True,
            cooldown=4,
            wf_threshold_ema=True,
            balanced_sampling=True,
            balanced_sampling_mode="cap",
            per_sym_no_edge_fallback=True,
            # Limit folds for speed
            max_folds=max_folds,
            model_version="v5",
            # Live-trading constraints — apply same cap as live system so results are representative
            max_trades_per_day=max_trades_per_day,
            # Custom tpd target (builds V5TPDControllerConfig with the requested target)
            tpd_ctrl_cfg=tpd_cfg,
            # Live training monitor push (optional)
            replit_url=replit_url,
        )
        elapsed = time.time() - t0
        result = _extract_results(wf_result, cfg.label, cfg)
        result.elapsed_sec = elapsed
        log.info("[SWEEP] %s done in %.1f min — trades=%d  WR=%.1f%%  E[R]=%.4f  totalR=%.2f",
                 cfg.label, elapsed / 60, result.n_trades,
                 result.win_rate * 100, result.expectancy, result.total_r)
        return result
    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        log.error("[SWEEP] %s FAILED after %.1f min:", cfg.label, elapsed / 60)
        log.error("[SWEEP] Full traceback:\n%s", traceback.format_exc())
        result = SweepResult(label=cfg.label, config=cfg, elapsed_sec=elapsed)
        result.error = f"{type(exc).__name__}: {exc}"
        return result


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_results_table(results: list, highlight_top: int = 3):
    """Print a ranked results table sorted by E[R]/trade descending."""
    valid = [r for r in results if r.error is None and r.n_trades > 0]
    failed = [r for r in results if r.error is not None or r.n_trades == 0]

    valid.sort(key=lambda r: r.expectancy, reverse=True)

    bar = "=" * 130
    print(f"\n{bar}")
    print("V5 PARAMETER SWEEP RESULTS — Ranked by E[R]/trade")
    print(bar)
    header = (
        f"{'Rank':<5} {'Label':<40} {'E[R]/trade':>10} {'WR%':>7} {'TotalR':>8} "
        f"{'Trades':>7} {'Sharpe':>7} {'Folds':>6} {'DebiasRatio':>12} {'Collapsed':>10} {'Time':>8}"
    )
    print(header)
    print("-" * 130)

    for rank, r in enumerate(valid, 1):
        prefix = ">>>" if rank <= highlight_top else "   "
        debias_str = f"{r.avg_debias_spread_ratio:.1f}x" if r.avg_debias_spread_ratio is not None else "N/A"
        collapsed_str = f"{r.n_collapsed_debias} folds" if r.n_collapsed_debias > 0 else "none"
        row = (
            f"{prefix}{rank:<3} "
            f"{r.label:<40} "
            f"{r.expectancy:>+10.4f} "
            f"{r.win_rate * 100:>7.1f} "
            f"{r.total_r:>+8.2f} "
            f"{r.n_trades:>7} "
            f"{r.sharpe:>7.2f} "
            f"{r.n_live_folds}/{r.n_folds:>2} "
            f"{debias_str:>12} "
            f"{collapsed_str:>10} "
            f"{r.elapsed_sec / 60:>7.1f}m"
        )
        print(row)

    if failed:
        print(f"\n{'FAILED / ZERO TRADES':^110}")
        print("-" * 110)
        for r in failed:
            reason = r.error if r.error else "0 trades produced"
            print(f"   {r.label:<40} {reason}")

    print(bar)

    if valid:
        best = valid[0]
        print(f"\nBEST CONFIG: {best.label}")
        print(f"  tp_mult={best.config.tp_mult}  sl_mult={best.config.sl_mult}")
        print(f"  adx_min={best.config.adx_min}  ema200_soft_mult={best.config.ema200_soft_mult}")
        print(f"  min_threshold={best.config.min_threshold}")
        print(f"\nRECOMMENDED TRAINING COMMAND:")
        ema_arg = (
            f"--v5-ema200-soft-mult {best.config.ema200_soft_mult}"
            if best.config.ema200_soft_mult is not None
            else ""
        )
        min_thr_arg = (
            f"--v5-min-threshold {best.config.min_threshold}"
            if best.config.min_threshold is not None
            else ""
        )
        print(f"""
python quick_start.py --train-v5 --v5-walk-forward \\
    --tp-mult {best.config.tp_mult} --sl-mult {best.config.sl_mult} \\
    --v5-adx-gate --v5-adx-min {best.config.adx_min} \\
    {ema_arg} \\
    {min_thr_arg} \\
    --v5-trailing-sl --v5-trail-activation {best.config.trail_activation} --v5-trail-distance {best.config.trail_distance} \\
    --v5-corr-thresh {best.config.corr_thresh} \\
    --v5-short-oversample --v5-short-min-fraction {best.config.short_min_fraction} \\
    --v5-per-symbol-threshold --v5-per-side-threshold \\
    --v5-recency-weight --v5-wf-warm-start
""")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V5 Parameter Sweep — real walk-forward")
    parser.add_argument("--data-dir", default="data_cache",
                        help="Directory containing parquet data files (default: data_cache)")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"],
                        help="Symbols to use (fewer = faster)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per fold (default: 100 — minimum needed for action head to escape mean-prediction plateau)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--folds", type=int, default=3,
                        help="Max folds per config (default: 3, use 2 for faster sweeps)")
    parser.add_argument("--max-tpd", type=int, default=None,
                        help="Max trades per day cap (default: None = uncapped). "
                             "Set to 8 to match live trading constraints so sweep results are representative.")
    parser.add_argument("--target-tpd", type=float, default=6.5,
                        help="Target trades per symbol per day for threshold calibration (default: 6.5). "
                             "To target N total trades/day across S symbols, pass N/S here.")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: only 5 configs instead of full 16-config grid")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the bug-fixed baseline config")
    parser.add_argument("--output", default="sweep_results.json",
                        help="Output JSON file for results (default: sweep_results.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in output JSON (resume interrupted sweep)")
    parser.add_argument("--configs", nargs="+",
                        help="Run only specific config labels from the grid")
    parser.add_argument("--replit-url", default=None,
                        help="Replit app URL for live training monitor push "
                             "(e.g. https://your-app.replit.app). Enables real-time "
                             "epoch-by-epoch updates on the Training Monitor page.")
    args = parser.parse_args()

    # Validate data directory — try several candidate paths
    _candidates = [
        Path(args.data_dir),                          # as given (absolute or relative to CWD)
        Path(__file__).parent / args.data_dir,        # relative to this script
        Path(__file__).parent / "data_cache",         # script sibling data_cache
        Path(args.data_dir).expanduser().resolve(),   # fully resolved
    ]
    data_dir = None
    for _c in _candidates:
        if _c.exists() and list(_c.glob("*.parquet")):
            data_dir = _c
            break

    if data_dir is None:
        log.error("Could not find data directory with .parquet files.")
        log.error("Tried:")
        for _c in _candidates:
            log.error("  %s  (exists=%s)", _c.resolve(), _c.exists())
        log.error("Fix: pass --data-dir with the full path, e.g.:")
        log.error('  python param_sweep.py --data-dir "C:/Users/you/Downloads/gpu_trainer/data_cache"')
        sys.exit(1)

    parquet_files = list(data_dir.glob("*.parquet"))
    log.info("[DATA] Using data directory: %s", data_dir.resolve())
    log.info("[DATA] Found %d parquet files", len(parquet_files))

    # Verify each requested symbol has a matching file (SYMBOL_15m.parquet)
    missing_syms = [s for s in args.symbols if not (data_dir / f"{s}_15m.parquet").exists()]
    if missing_syms:
        log.warning("[DATA] Missing _15m.parquet files for: %s", missing_syms)
        log.warning("[DATA] Available files: %s", sorted(f.name for f in parquet_files))
        args.symbols = [s for s in args.symbols if s not in missing_syms]
        if not args.symbols:
            log.error("[DATA] No symbols have matching data files. Exiting.")
            sys.exit(1)
        log.info("[DATA] Continuing with: %s", args.symbols)
    else:
        log.info("[DATA] All %d requested symbols have data files", len(args.symbols))

    # Device detection
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1024**3)
        else:
            log.warning("No GPU detected — sweep will be very slow on CPU")
    except ImportError:
        log.error("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    # Add gpu_trainer to path if needed
    gpu_trainer_dir = Path(__file__).parent
    if str(gpu_trainer_dir) not in sys.path:
        sys.path.insert(0, str(gpu_trainer_dir))

    # Build configs
    if args.baseline_only:
        configs = [SweepConfig("BASELINE", description="Bug-fixed baseline")]
    else:
        configs = build_sweep_configs(fast=args.fast)
        if args.configs:
            configs = [c for c in configs if c.label in args.configs]
            if not configs:
                log.error("No matching configs found for labels: %s", args.configs)
                sys.exit(1)

    # Load existing results for resume
    existing_labels = set()
    all_results = []
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        try:
            with open(output_path) as f:
                existing_data = json.load(f)
            for entry in existing_data.get("results", []):
                existing_labels.add(entry.get("label", ""))
            log.info("[SWEEP] Resume mode: skipping %d already-completed configs", len(existing_labels))
        except Exception as e:
            log.warning("[SWEEP] Could not load existing results: %s", e)

    log.info("[SWEEP] Starting sweep: %d configs × %d folds × %d epochs on %d symbols",
             len(configs), args.folds, args.epochs, len(args.symbols))
    log.info("[SWEEP] Symbols: %s", ", ".join(args.symbols))
    log.info("[SWEEP] Output: %s", output_path.resolve())

    start_time = time.time()

    for i, cfg in enumerate(configs):
        if cfg.label in existing_labels:
            log.info("[SWEEP] Skipping %s (already in %s)", cfg.label, args.output)
            continue

        log.info("[SWEEP] Config %d/%d: %s", i + 1, len(configs), cfg.label)
        result = run_single_config(
            cfg=cfg,
            data_dir=data_dir,
            device=device,
            symbols=args.symbols,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_folds=args.folds,
            max_trades_per_day=args.max_tpd,
            target_tpd=args.target_tpd,
            replit_url=args.replit_url,
        )
        all_results.append(result)

        # Save checkpoint after each config so sweep can be resumed
        checkpoint = {
            "sweep_start": datetime.now().isoformat(),
            "total_elapsed_min": round((time.time() - start_time) / 60, 1),
            "configs_done": len(all_results),
            "configs_total": len(configs),
            "symbols": args.symbols,
            "epochs": args.epochs,
            "folds": args.folds,
            "max_tpd": args.max_tpd,
            "target_tpd": args.target_tpd,
            "results": [r.to_dict() for r in all_results],
        }
        try:
            with open(output_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            log.info("[SWEEP] Checkpoint saved → %s", output_path)
        except Exception as e:
            log.warning("[SWEEP] Could not save checkpoint: %s", e)

    total_elapsed = time.time() - start_time
    log.info("[SWEEP] All configs done in %.1f min", total_elapsed / 60)

    # Print results table
    print_results_table(all_results)

    # Final save
    final_output = {
        "sweep_completed": datetime.now().isoformat(),
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "configs_total": len(configs),
        "symbols": args.symbols,
        "epochs": args.epochs,
        "folds": args.folds,
        "max_tpd": args.max_tpd,
        "target_tpd": args.target_tpd,
        "results": [r.to_dict() for r in all_results],
    }
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    log.info("[SWEEP] Final results saved → %s", output_path.resolve())


if __name__ == "__main__":
    main()
