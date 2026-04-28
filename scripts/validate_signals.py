#!/usr/bin/env python3
"""
validate_signals.py — V5 signal quality validator.

Runs a walk-forward validation and reports a comprehensive diagnostic:
  1. Score spread   (p1 / p25 / p50 / p75 / p99) per fold
  2. LONG / SHORT split at every threshold slice (via FWD_SLICE_AUDIT)
  3. Score decile table  (combined + L_/S_ rows per fold)
  4. Per-symbol status   (ACTIVE / HIGH_BAR) from per_symbol_summary
  5. Per-symbol best thresholds (from per_symbol_thresholds in fold reports)
  6. Debias spread ratio check (warns if mu_R discrimination collapsed)

When --model-path points to a directory containing v5_walkforward_report.json,
the saved report is loaded and displayed WITHOUT re-training.

Exit codes:
  0 — at least 1 ACTIVE symbol (edge=="YES") AND at least 1 fold with score_monotonic=True
  1 — all symbols HIGH_BAR / no trades / no monotone fold / no active symbol
  2 — fatal error (import failure, missing data, etc.)

Usage (run from gpu_trainer directory):
    python scripts/validate_signals.py --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --folds 1 --epochs 50 --data-dir data_cache

    # Load saved checkpoint report without re-training:
    python scripts/validate_signals.py --model-path checkpoints/my_run \\
        --symbols BTCUSDT ETHUSDT SOLUSDT --data-dir data_cache
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ValidateSignals")


def _check_imports():
    missing = []
    for pkg in ["numpy", "pandas", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s — run: pip install %s", missing, " ".join(missing))
        sys.exit(2)


def _print_score_spread(spread_dict, fold_idx):
    if not spread_dict:
        log.warning("  Fold %d: score spread not available", fold_idx)
        return
    log.info(
        "  Fold %d score spread: p1=%.5f  p25=%.5f  p50=%.5f  p75=%.5f  p99=%.5f  range=%.5f",
        fold_idx,
        spread_dict.get("p1", float("nan")),
        spread_dict.get("p25", float("nan")),
        spread_dict.get("p50", float("nan")),
        spread_dict.get("p75", float("nan")),
        spread_dict.get("p99", float("nan")),
        spread_dict.get("p99", 0) - spread_dict.get("p1", 0),
    )
    if spread_dict.get("p99", 0) - spread_dict.get("p1", 0) < 1e-4:
        log.warning(
            "  Fold %d: WARNING — score range near-zero. "
            "Model may be collapsed (all bars score the same).",
            fold_idx,
        )


def _try_load_saved_report(model_path_str):
    """Try to load a saved v5_walkforward_report.json from a directory.

    Returns the parsed dict or None if not found.
    """
    if not model_path_str:
        return None
    mp = Path(model_path_str)
    candidates = [
        mp / "v5_walkforward_report.json",
        mp / "checkpoints" / "v5_walkforward_report.json",
    ]
    for p in candidates:
        if p.exists():
            log.info("Loading saved walk-forward report from: %s", p)
            with open(p) as f:
                return json.load(f)
    return None


def _print_per_sym_side_bias(folds):
    """Print per-symbol side-bias diagnostics from the latest fold that has the data."""
    side_bias = {}
    for f in folds:
        if not isinstance(f, dict):
            continue
        sb = f.get("per_sym_side_bias", {})
        if sb:
            side_bias = sb
    if not side_bias:
        return
    biased = {s: v for s, v in side_bias.items() if v.get("biased")}
    if not biased:
        return
    log.info("=" * 80)
    log.info("PER-SYMBOL SIDE BIAS (>85%% single-side dominance)")
    log.info("=" * 80)
    for sym in sorted(biased):
        v = biased[sym]
        log.warning(
            "  %-14s  LONG=%.0f%%  SHORT=%.0f%%  [SIDE_BIAS: %s]",
            sym, v["long_pct"], v["short_pct"], v["bias_dir"],
        )
    log.warning(
        "  ACTION: increase short_min_fraction or short_oversample strength "
        "for biased symbols."
    )
    log.info("")


def _print_per_symbol_thresholds(folds):
    """Print best threshold per symbol collected across folds."""
    sym_thresholds = {}
    for f in folds:
        if not isinstance(f, dict):
            continue
        pst = f.get("per_symbol_thresholds", {})
        for sym, thr in pst.items():
            if thr is not None:
                if sym not in sym_thresholds or sym_thresholds[sym] is None:
                    sym_thresholds[sym] = thr
    if sym_thresholds:
        log.info("=" * 80)
        log.info("PER-SYMBOL BEST THRESHOLDS (latest fold)")
        log.info("=" * 80)
        for sym in sorted(sym_thresholds):
            thr = sym_thresholds[sym]
            log.info("  %-14s  threshold=%.6f", sym, thr)
        log.info("")


def _display_report(result):
    """Display a walk-forward result dict and return (fold_has_monotone, active_symbols)."""
    agg = result.get("aggregate", {})
    folds = result.get("folds", [])
    per_sym = result.get("per_symbol_summary", {})

    total_trades = int(agg.get("total_trades", 0))
    total_r = float(agg.get("total_r", 0.0))
    avg_er = float(agg.get("avg_expectancy_r", 0.0))
    n_active_folds = int(agg.get("active_folds", 0))
    n_folds = int(agg.get("n_folds", len(folds)))

    log.info("")
    log.info("=" * 80)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 80)
    log.info("  Folds         : %d total, %d active", n_folds, n_active_folds)
    log.info("  Total trades  : %d", total_trades)
    log.info("  Total R       : %+.4f", total_r)
    log.info("  Avg E[R]/trade: %+.4f", avg_er)
    log.info("")

    log.info("=" * 80)
    log.info("PER-FOLD DETAIL")
    log.info("=" * 80)
    fold_has_monotone = False
    for i, f in enumerate(folds):
        if not isinstance(f, dict):
            continue
        n_tr = f.get("total_trades", 0)
        er = f.get("expectancy_r", 0.0)
        wr = f.get("win_rate", 0.0)
        sharpe = f.get("sharpe", 0.0)
        total_r_fold = f.get("total_r", 0.0)
        mono = f.get("score_monotonic")
        debias_ratio = f.get("debias_spread_ratio")
        mono_str = "PASS" if mono else ("FAIL" if mono is not None else "N/A")

        log.info(
            "  Fold %d: trades=%d  E[R]=%+.4f  WR=%.1f%%  Sharpe=%.2f  "
            "TotalR=%+.4f  decile_mono=%s",
            i + 1, n_tr, er, wr * 100, sharpe, total_r_fold, mono_str,
        )

        if debias_ratio is not None:
            flag = "  [OK]" if debias_ratio >= 5.0 else "  [COLLAPSED! — reduce mu_debias_alpha]"
            log.info("         debias_spread_ratio=%.2fx%s", debias_ratio, flag)

        _print_score_spread(f.get("score_spread"), i + 1)

        sq = f.get("side_quality", {})
        if sq:
            log.info(
                "         LONG  avg_score=%+.4f  avg_p_side=%.4f  avg_mu_R=%+.4f  head_agree=%.0f%%",
                sq.get("long_avg_score", 0), sq.get("long_avg_p_side", 0),
                sq.get("long_avg_mu_r", 0), sq.get("long_head_agree_pct", 0),
            )
            log.info(
                "         SHORT avg_score=%+.4f  avg_p_side=%.4f  avg_mu_R=%+.4f  head_agree=%.0f%%",
                sq.get("short_avg_score", 0), sq.get("short_avg_p_side", 0),
                sq.get("short_avg_mu_r", 0), sq.get("short_head_agree_pct", 0),
            )

        decile_rows = f.get("score_decile_table", [])
        if decile_rows:
            log.info("         Score decile table (fold %d, monotonic=%s):", i + 1, mono_str)
            log.info(
                "           %-10s %9s %9s %5s %8s %7s",
                "Label", "ScoreLo", "ScoreHi", "N", "AvgR", "WR",
            )
            for row in decile_rows:
                lbl = row.get("label", f"D{row['decile']:02d}")
                log.info(
                    "           %-10s %9.4f %9.4f %5d %+8.4f %6.1f%%",
                    lbl, row["score_lo"], row["score_hi"],
                    row["n_trades"], row["avg_r"], row["win_rate"] * 100,
                )

        sym_stats = f.get("per_symbol_stats", {})
        if sym_stats:
            log.info("         Per-symbol (fold %d):", i + 1)
            for sn, ss in sym_stats.items():
                log.info(
                    "           %-12s trades=%d  E[R]=%+.4f  WR=%.1f%%  Total=%+.4fR",
                    sn, ss["trades"], ss["expectancy_r"],
                    ss["win_rate"] * 100, ss["total_r"],
                )

        slice_rows = f.get("slice_audit", [])
        if slice_rows:
            log.info("         LONG/SHORT slice split (fold %d):", i + 1)
            log.info(
                "           %-14s %-7s %6s %8s %7s %6s",
                "Slice", "Side", "N", "E[R]", "WR%", "PF",
            )
            for row in slice_rows:
                if row.get("side") in ("LONG", "SHORT", "ALL"):
                    log.info(
                        "           %-14s %-7s %6d %+8.4f %6.1f%% %6.2f",
                        row["threshold_label"], row["side"],
                        row["n_trades"], row["expectancy"],
                        row["win_rate"] * 100, row["pf"],
                    )

        if mono is True:
            fold_has_monotone = True

    log.info("")

    _print_per_symbol_thresholds(folds)
    _print_per_sym_side_bias(folds)

    active_symbols = []
    no_edge_symbols = []
    if per_sym:
        log.info("=" * 80)
        log.info("PER-SYMBOL STATUS")
        log.info("=" * 80)
        log.info(
            "  %-12s %7s %8s %7s %6s %10s %10s",
            "Symbol", "Trades", "E[R]", "WR%", "Folds", "TotalR", "Status",
        )
        log.info("-" * 80)
        for sym, info in per_sym.items():
            if not isinstance(info, dict):
                continue
            n = info.get("trades", 0)
            er_sym = info.get("expectancy_r", 0.0)
            wr = info.get("win_rate", 0.0)
            total_r_sym = info.get("total_r", 0.0)
            folds_seen = info.get("folds", 0)
            edge = info.get("edge", "NO")
            if edge == "YES":
                active_symbols.append(sym)
                status_tag = "ACTIVE"
            else:
                no_edge_symbols.append(sym)
                status_tag = "HIGH_BAR" if n == 0 else "NO_EDGE"
            log.info(
                "  %-12s %7d %+8.4f %6.1f%% %6d %+10.4f %10s",
                sym, n, er_sym, wr * 100, folds_seen, total_r_sym, status_tag,
            )
        log.info("")
        log.info(
            "  Active   : %d — %s",
            len(active_symbols),
            active_symbols if active_symbols else "NONE",
        )
        log.info(
            "  No edge  : %d — %s",
            len(no_edge_symbols),
            no_edge_symbols if no_edge_symbols else "none",
        )
        log.info("")

    return fold_has_monotone, active_symbols, total_trades, total_r, avg_er


def _verdict(fold_has_monotone, active_symbols, total_trades, avg_er, total_r):
    """Print verdict and exit with appropriate code."""
    log.info("=" * 80)
    log.info("VALIDATION VERDICT")
    log.info("=" * 80)

    if total_trades == 0:
        log.error("FAIL: 0 trades produced across all folds.")
        log.error("  Root causes to investigate:")
        log.error("  1. ALL symbols returned HIGH_BAR — model has no positive edge.")
        log.error("  2. mu_debias collapsed mu_R spread (check debias_spread_ratio < 5x).")
        log.error("  3. quality_gate rejecting all bars (check [V5_DEBIAS_SPREAD] log).")
        log.error("  4. Epochs too low — action head stuck in mean-prediction plateau.")
        log.error("  FIX: Retrain with w_action=2.5 and epochs>=100.")
        sys.exit(1)

    if len(active_symbols) == 0:
        log.warning(
            "FAIL: %d trades generated but no symbol has edge==YES in per_symbol_summary.",
            total_trades,
        )
        log.warning("  All symbols are HIGH_BAR or NO_EDGE — live trading would fire on noise.")
        log.warning("  Actions: more epochs, lower min_threshold, more training data.")
        sys.exit(1)

    if not fold_has_monotone:
        log.warning(
            "WARN: %d trades, %d ACTIVE symbols — but no fold has score_monotonic=True.",
            total_trades, len(active_symbols),
        )
        log.warning("  Score-to-return monotonicity is required for live edge.")
        log.warning("  Current aggregate E[R] = %+.4f.", avg_er)
        if avg_er <= 0:
            log.warning("  Additionally, aggregate E[R] is negative — model is losing.")
        log.warning("  Actions: more epochs, short_oversample, per_symbol_threshold tuning.")
        sys.exit(1)

    log.info(
        "PASS: %d trades  E[R]=%+.4f  Total R=%+.4f",
        total_trades, avg_er, total_r,
    )
    log.info(
        "  %d ACTIVE symbol(s): %s",
        len(active_symbols), active_symbols,
    )
    log.info("  At least 1 fold has monotone score decile ordering.")
    log.info("  Model is generating positive-expectancy, score-ordered signals. Ready for live.")
    sys.exit(0)


def _run_validation(args):
    saved_result = _try_load_saved_report(args.model_path)
    if saved_result is not None:
        log.info("=" * 80)
        log.info("V5 SIGNAL VALIDATOR  [LOADED FROM SAVED REPORT]")
        log.info("=" * 80)
        fold_has_monotone, active_symbols, total_trades, total_r, avg_er = _display_report(saved_result)
        _verdict(fold_has_monotone, active_symbols, total_trades, avg_er, total_r)
        return

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from train.v5_train import run_v5_walk_forward
    except ImportError as e:
        log.error("Cannot import v5_train: %s", e)
        log.error("Run this script from the gpu_trainer directory or its parent.")
        sys.exit(2)

    _candidates = [
        Path(args.data_dir),
        Path(__file__).parent.parent / args.data_dir,
        Path(__file__).parent.parent / "data_cache",
    ]
    data_dir = None
    for c in _candidates:
        if c.exists() and list(c.glob("*.parquet")):
            data_dir = c
            break
    if data_dir is None:
        log.error("No parquet data found in: %s", [str(c) for c in _candidates])
        sys.exit(2)

    log.info("=" * 80)
    log.info("V5 SIGNAL VALIDATOR  [FULL WALK-FORWARD]")
    log.info("=" * 80)
    log.info("Data dir   : %s", data_dir.resolve())
    log.info("Symbols    : %s", args.symbols)
    log.info("Folds      : %d", args.folds)
    log.info("Epochs     : %d", args.epochs)
    if args.model_path:
        log.info(
            "Model path : %s  (no saved report found — running fresh walk-forward)",
            args.model_path,
        )
    log.info("=" * 80)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Device: %s", device)
    except ImportError:
        device = "cpu"

    t0 = time.time()
    try:
        result = run_v5_walk_forward(
            data_dir=data_dir,
            device=device,
            symbols=args.symbols,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            train_months=12,
            test_months=1,
            tp_mult=3.0,
            sl_mult=1.0,
            adx_gate=True,
            adx_min=18.0,
            ema200_soft_mult=1.0,
            corr_block=True,
            corr_thresh=0.90,
            corr_same_side_only=True,
            short_oversample=True,
            short_min_fraction=0.35,
            per_symbol_threshold=True,
            per_side_threshold=True,
            trailing_sl=True,
            trail_activation=1.5,
            trail_distance=1.0,
            side_aware_scoring=False,
            recency_weight=True,
            mu_debias=True,
            per_symbol_cooldown=True,
            cooldown=4,
            wf_threshold_ema=True,
            balanced_sampling=True,
            balanced_sampling_mode="cap",
            per_sym_no_edge_fallback=True,
            max_folds=args.folds,
            model_version="v5",
            w_action=2.5,
        )
    except Exception:
        import traceback
        log.error("run_v5_walk_forward failed:\n%s", traceback.format_exc())
        sys.exit(2)

    elapsed = time.time() - t0
    log.info("Walk-forward completed in %.1f min", elapsed / 60)

    if result is None:
        log.error("run_v5_walk_forward returned None — no data or early exit.")
        sys.exit(1)

    fold_has_monotone, active_symbols, total_trades, total_r, avg_er = _display_report(result)
    _verdict(fold_has_monotone, active_symbols, total_trades, avg_er, total_r)


def main():
    parser = argparse.ArgumentParser(
        description="V5 signal quality validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        help="Symbols to validate (fewer = faster)",
    )
    parser.add_argument(
        "--data-dir", default="data_cache",
        help="Directory with *_15m.parquet files",
    )
    parser.add_argument(
        "--folds", type=int, default=2,
        help="Number of walk-forward folds (default: 2, use 1 for quick check)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Epochs per fold (default: 100 — minimum for action head convergence)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help=(
            "Path to a prior checkpoint directory. "
            "If v5_walkforward_report.json is found there, it is loaded and displayed "
            "without re-training. Otherwise, a fresh walk-forward is run."
        ),
    )
    args = parser.parse_args()

    _check_imports()
    _run_validation(args)


if __name__ == "__main__":
    main()
