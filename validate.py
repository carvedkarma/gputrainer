"""validate.py — V5 fast validation harness.

Modes
-----
unit_audit    : Pytest precision-audit suite + direct torch-free config checks.
smoke_wf      : 2-symbol, 2-fold walk-forward smoke test (BTC + ETH).
canary_wf     : 4-symbol, 3-fold walk-forward (BTC/ETH/SOL/BNB).
candidate_diff: Run WF on all 20 symbols, dump per-candidate CSV.
                --run run_a.json saves the run JSON to that path.
                --vs baseline.csv prints top-20 decision changes vs a prior CSV.
compare       : Load two validate_runs JSON files and diff all metric tables
                including gate block %.

Usage
-----
python validate.py unit_audit
python validate.py smoke_wf
python validate.py canary_wf [--folds 3]
python validate.py candidate_diff [--folds 2] [--run run_a.json] [--csv out.csv]
python validate.py compare runs/smoke_wf_A.json runs/smoke_wf_B.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

RUNS_DIR = Path(__file__).parent / "validate_runs"
DATA_DIR = Path(__file__).parent / "data_cache"

SMOKE_SYMBOLS  = ["BTCUSDT", "ETHUSDT"]
CANARY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "PEPEUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT",
    "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
]

CSV_COLUMNS = [
    "fold_idx", "bar_idx", "timestamp", "symbol", "side",
    "raw_score", "final_score", "threshold",
    "taken", "block_reason",
    "mu_R", "p_trade", "adx_val", "regime_label", "corr_blocked",
    "oracle_r",
]


# ─────────────────────────────────────────────
# Metric computation helpers
# ─────────────────────────────────────────────

def _oracle_r(r: Dict) -> float:
    return float(r.get("oracle_r", 0.0) or 0.0)


def _compute_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Compute all required trade-level metrics from a list of candidate records."""
    taken = [r for r in records if r.get("taken")]
    blocked = [r for r in records if not r.get("taken")]
    n_trades = len(taken)

    gate_counts: Dict[str, int] = defaultdict(int)
    for r in blocked:
        gate = r.get("block_reason") or "unknown"
        gate_counts[gate] += 1
    n_blocked = len(blocked)
    gate_pct = {
        g: round(c / max(n_blocked, 1) * 100, 1)
        for g, c in sorted(gate_counts.items(), key=lambda x: -x[1])
    }

    if n_trades == 0:
        return {
            "trades": 0, "oracle_r_total": 0.0, "expectancy": 0.0,
            "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
            "long_trades": 0, "short_trades": 0,
            "long_expectancy": 0.0, "short_expectancy": 0.0,
            "top_decile_avg_r": 0.0, "bottom_decile_avg_r": 0.0,
            "monotonic_score_r": "N/A",
            "gate_block_pct": gate_pct,
            "n_candidates": len(records), "n_blocked": n_blocked,
        }

    rs = [_oracle_r(r) for r in taken]
    wins   = [v for v in rs if v > 0]
    losses = [v for v in rs if v <= 0]
    long_rs  = [_oracle_r(r) for r in taken if r.get("side") == 1]
    short_rs = [_oracle_r(r) for r in taken if r.get("side") == -1]

    sorted_by_score = sorted(taken, key=lambda r: r.get("final_score", r.get("raw_score", 0.0)), reverse=True)
    decile = max(1, n_trades // 10)
    top_decile = [_oracle_r(r) for r in sorted_by_score[:decile]]
    bot_decile = [_oracle_r(r) for r in sorted_by_score[-decile:]]

    return {
        "trades": n_trades,
        "oracle_r_total": round(sum(rs), 3),
        "expectancy": round(sum(rs) / n_trades, 4),
        "win_rate": round(len(wins) / n_trades, 4),
        "avg_win_r": round(sum(wins) / max(len(wins), 1), 4),
        "avg_loss_r": round(sum(losses) / max(len(losses), 1), 4),
        "long_trades": len(long_rs),
        "short_trades": len(short_rs),
        "long_expectancy": round(sum(long_rs) / max(len(long_rs), 1), 4),
        "short_expectancy": round(sum(short_rs) / max(len(short_rs), 1), 4),
        "top_decile_avg_r": round(sum(top_decile) / max(len(top_decile), 1), 4),
        "bottom_decile_avg_r": round(sum(bot_decile) / max(len(bot_decile), 1), 4),
        "monotonic_score_r": "PASS" if _check_monotonic(taken) else "FAIL",
        "gate_block_pct": gate_pct,
        "n_candidates": len(records),
        "n_blocked": n_blocked,
    }


def _check_monotonic(taken: List[Dict], n_buckets: int = 5) -> bool:
    if len(taken) < n_buckets * 2:
        return True
    sorted_t = sorted(taken, key=lambda r: r.get("final_score", r.get("raw_score", 0.0)))
    bsz = len(sorted_t) // n_buckets
    means = [sum(_oracle_r(r) for r in sorted_t[i*bsz:(i+1)*bsz]) / bsz
             for i in range(n_buckets) if sorted_t[i*bsz:(i+1)*bsz]]
    return all(means[i] <= means[i+1] for i in range(len(means) - 1))


# ─────────────────────────────────────────────
# ASCII tables
# ─────────────────────────────────────────────

CORE_METRIC_ROWS = [
    ("trades",              "Trades"),
    ("n_candidates",        "Candidates"),
    ("n_blocked",           "Blocked"),
    ("oracle_r_total",      "Total oracle R"),
    ("expectancy",          "Expectancy (R/trade)"),
    ("win_rate",            "Win rate"),
    ("avg_win_r",           "Avg win (R)"),
    ("avg_loss_r",          "Avg loss (R)"),
    ("long_trades",         "Long trades"),
    ("short_trades",        "Short trades"),
    ("long_expectancy",     "Long E[R]"),
    ("short_expectancy",    "Short E[R]"),
    ("top_decile_avg_r",    "Top-decile avg R"),
    ("bottom_decile_avg_r", "Bottom-decile avg R"),
    ("monotonic_score_r",   "Monotonic score→R"),
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_metrics_table(title: str, m: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    for key, label in CORE_METRIC_ROWS:
        print(f"  {label:<28} {_fmt(m.get(key, '—'))}")
    gate_pct = m.get("gate_block_pct", {})
    if gate_pct:
        print(f"\n  Gate block breakdown (% of blocked):")
        for gate, pct in list(gate_pct.items())[:12]:
            print(f"    {gate:<24} {pct:>5.1f}%")
    print()


def _print_compare_table(label_a: str, label_b: str,
                         m_a: Dict[str, Any], m_b: Dict[str, Any]) -> None:
    numeric_keys = [
        ("trades",              "Trades"),
        ("oracle_r_total",      "Total oracle R"),
        ("expectancy",          "Expectancy (R/trade)"),
        ("win_rate",            "Win rate"),
        ("avg_win_r",           "Avg win (R)"),
        ("avg_loss_r",          "Avg loss (R)"),
        ("long_expectancy",     "Long E[R]"),
        ("short_expectancy",    "Short E[R]"),
        ("top_decile_avg_r",    "Top-decile avg R"),
        ("bottom_decile_avg_r", "Bottom-decile avg R"),
        ("n_candidates",        "Candidates"),
        ("n_blocked",           "Blocked"),
    ]
    print(f"\n{'=' * 72}")
    print(f"  COMPARE  {label_a}  vs  {label_b}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<28} {'A':>12} {'B':>12} {'Delta':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*14}")
    for key, label in numeric_keys:
        va = m_a.get(key, 0) or 0
        vb = m_b.get(key, 0) or 0
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {label:<28} {_fmt(va):>12} {_fmt(vb):>12} {f'{delta:+.4f}{arrow}':>14}")
        else:
            print(f"  {label:<28} {str(va):>12} {str(vb):>12} {'—':>14}")

    print(f"\n  Monotonic  A: {m_a.get('monotonic_score_r', '—')}   "
          f"B: {m_b.get('monotonic_score_r', '—')}")

    gp_a = m_a.get("gate_block_pct", {})
    gp_b = m_b.get("gate_block_pct", {})
    all_gates = sorted(set(list(gp_a.keys()) + list(gp_b.keys())))
    if all_gates:
        print(f"\n  Gate block % comparison:")
        print(f"  {'Gate':<24} {'A%':>8} {'B%':>8} {'Delta%':>10}")
        print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*10}")
        for g in all_gates:
            pa = gp_a.get(g, 0.0)
            pb = gp_b.get(g, 0.0)
            dg = pb - pa
            arrow = "↑" if dg > 0 else ("↓" if dg < 0 else "=")
            print(f"  {g:<24} {pa:>7.1f}% {pb:>7.1f}% {f'{dg:+.1f}%{arrow}':>10}")
    print()


# ─────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────

def _save_csv(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("[validate] CSV saved → %s  (%d rows)", path, len(records))


def _load_csv(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: Dict[str, Any] = {}
            for col in CSV_COLUMNS:
                v = row.get(col, "")
                if col in ("fold_idx", "bar_idx", "timestamp", "side"):
                    try:
                        rec[col] = int(v) if v not in ("", "nan", "None") else None
                    except ValueError:
                        rec[col] = None
                elif col in ("raw_score", "final_score", "threshold",
                             "mu_R", "p_trade", "adx_val", "oracle_r"):
                    try:
                        rec[col] = float(v) if v not in ("", "nan") else float("nan")
                    except ValueError:
                        rec[col] = float("nan")
                elif col in ("taken", "corr_blocked"):
                    rec[col] = str(v).lower() in ("true", "1", "yes")
                else:
                    rec[col] = v
            records.append(rec)
    log.info("[validate] Loaded %d records from %s", len(records), path)
    return records


# ─────────────────────────────────────────────
# JSON run helpers
# ─────────────────────────────────────────────

def _save_run(mode: str, payload: Dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"{mode}_{ts}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("[validate] Results saved → %s", out)
    return out


def _load_run(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Walk-forward runner
# ─────────────────────────────────────────────

def _run_wf(
    symbols: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    train_months: int,
    test_months: int = 1,
    max_folds: Optional[int] = None,
    seed: int = 42,
    test_weeks: Optional[int] = None,
) -> Tuple[List[Dict], Optional[Dict]]:
    """Run walk-forward and collect all candidate records via candidate_logger.

    Returns
    -------
    (records, wf_report)
        records   : list of per-candidate dicts from candidate_logger
        wf_report : full walk-forward report dict (folds + aggregate),
                    or None if the WF produced no output
    """
    try:
        import torch
    except ImportError:
        log.error("[validate] torch not available — WF modes require the GPU machine.")
        sys.exit(1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[validate] Device: %s", dev)

    sys.path.insert(0, str(Path(__file__).parent))
    from train.v5_train import run_v5_walk_forward

    records: List[Dict] = []

    def _logger(rec: Dict) -> None:
        records.append(rec)

    torch.manual_seed(seed)
    wf_report = run_v5_walk_forward(
        data_dir=DATA_DIR,
        device=dev,
        symbols=symbols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_months=train_months,
        test_months=test_months,
        test_weeks=test_weeks,
        max_folds=max_folds,
        candidate_logger=_logger,
        slippage_base_bps=6.0,
        score_lambda=0.5,
        balanced_sampling=True,
        calibration_monitor=True,
    )
    return records, wf_report


# ─────────────────────────────────────────────
# Mode: unit_audit
# ─────────────────────────────────────────────

def _torch_free_config_checks() -> List[Tuple[str, str, bool]]:
    """Direct torch-free checks of V5 config defaults.

    Loads shared_v5_trade_config.py directly via importlib to avoid the
    config/__init__.py which imports torch.

    Returns list of (check_name, detail, passed) tuples.
    """
    results: List[Tuple[str, str, bool]] = []
    try:
        import importlib.util
        cfg_path = Path(__file__).parent / "config" / "shared_v5_trade_config.py"
        spec = importlib.util.spec_from_file_location("_shared_cfg_direct", str(cfg_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        cfg = mod.V5TradeDefaults()

        checks = [
            ("score_threshold == 0.02",   cfg.score_threshold == 0.02,   f"got {cfg.score_threshold}"),
            ("slippage_base_bps == 6.0",  cfg.slippage_base_bps == 6.0,  f"got {cfg.slippage_base_bps}"),
            ("score_lambda == 0.5",       cfg.score_lambda == 0.5,        f"got {cfg.score_lambda}"),
            ("size_floor == 0.5",         cfg.size_floor == 0.5,          f"got {cfg.size_floor}"),
            ("cooldown_bars >= 4",        cfg.cooldown_bars >= 4,         f"got {cfg.cooldown_bars}"),
        ]
        for name, passed, detail in checks:
            results.append((name, detail, passed))
    except Exception as e:
        results.append(("config_import", str(e), False))
    return results


def mode_unit_audit(args: argparse.Namespace) -> None:
    log.info("[validate] mode=unit_audit")
    import subprocess
    tests_path = Path(__file__).parent / "tests" / "test_v5_precision_audit.py"
    if not tests_path.exists():
        log.error("Test file not found: %s", tests_path)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_path), "-v", "--tb=short"],
        cwd=str(Path(__file__).parent),
    )
    pytest_ok = result.returncode == 0

    config_checks = _torch_free_config_checks()
    config_ok = all(passed for _, _, passed in config_checks)

    print(f"\n{'=' * 60}")
    print("  unit_audit — torch-free config checks")
    print('=' * 60)
    for name, detail, passed in config_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}  ({detail})")
    print()

    overall_ok = pytest_ok and config_ok
    status = "PASSED" if overall_ok else "FAILED"
    log.info("[validate] unit_audit %s", status)

    payload = {
        "mode": "unit_audit",
        "pytest_exit_code": result.returncode,
        "pytest_ok": pytest_ok,
        "config_checks": [{"name": n, "detail": d, "passed": p} for n, d, p in config_checks],
        "config_ok": config_ok,
        "status": status,
    }
    _save_run("unit_audit", payload)
    if not overall_ok:
        sys.exit(1)


# ─────────────────────────────────────────────
# Fold-level breakdown helper
# ─────────────────────────────────────────────

def _fold_breakdown(records: List[Dict]) -> List[Dict]:
    """Group taken-trade records by fold_idx field embedded in each record.

    The candidate_logger emits 'fold_idx' on every record so we can group
    accurately.  Falls back to a single fold when fold_idx is absent/None.
    """
    taken = [r for r in records if r.get("taken")]
    if not taken:
        return []

    fold_buckets: Dict[Any, List[Dict]] = defaultdict(list)
    for r in taken:
        fid = r.get("fold_idx")
        fold_buckets[fid].append(r)

    fold_summaries = []
    for fid in sorted(fold_buckets.keys(), key=lambda x: (x is None, x)):
        fold_records = fold_buckets[fid]
        rs = [_oracle_r(r) for r in fold_records]
        wins = [v for v in rs if v > 0]
        fold_summaries.append({
            "fold": fid if fid is not None else "?",
            "trades": len(fold_records),
            "total_r": round(sum(rs), 3),
            "expectancy": round(sum(rs) / max(len(rs), 1), 4),
            "win_rate": round(len(wins) / max(len(rs), 1), 4),
        })
    return fold_summaries


def _print_fold_table(fold_summaries: List[Dict]) -> None:
    if not fold_summaries:
        print("  (no fold breakdown available)")
        return
    print(f"\n  {'Fold':<6} {'Trades':>8} {'Total R':>10} {'Exp':>10} {'WR':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for f in fold_summaries:
        print(f"  {f['fold']:<6} {f['trades']:>8} {f['total_r']:>10.3f} "
              f"{f['expectancy']:>10.4f} {f['win_rate']:>7.1%}")
    print()


# ─────────────────────────────────────────────
# Canonical report display helpers
# ─────────────────────────────────────────────

def _print_fold_reports(mode: str, fold_reports: List[Dict]) -> None:
    """Print canonical per-fold metrics from wf_report['folds'] dicts."""
    if not fold_reports:
        print("  (no canonical fold reports)")
        return
    print(f"\n{'=' * 80}")
    print(f"  {mode.upper()} — Canonical fold metrics (from forward-test report dicts)")
    print('=' * 80)
    print(f"  {'Fold':<6} {'Trades':>8} {'Total R':>10} {'E[R]':>10} "
          f"{'WR':>7} {'Long':>7} {'Short':>7} {'Threshold':>11}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*11}")
    for fd in fold_reports:
        fold_id   = fd.get('fold', '?')
        n         = fd.get('total_trades', 0)
        total_r   = fd.get('total_r', 0.0)
        expect    = fd.get('expectancy_r', 0.0)
        wr        = fd.get('win_rate', 0.0)
        n_long    = fd.get('n_long', fd.get('direction_stats', {}).get('long_trades', '—'))
        n_short   = fd.get('n_short', fd.get('direction_stats', {}).get('short_trades', '—'))
        thr       = fd.get('score_threshold', fd.get('threshold_ema', '—'))
        thr_str   = f"{thr:.4f}" if isinstance(thr, float) else str(thr)
        print(f"  {fold_id:<6} {n:>8} {total_r:>10.3f} {expect:>10.4f} "
              f"{wr:>6.1%} {str(n_long):>7} {str(n_short):>7} {thr_str:>11}")
    print()

    all_n     = sum(fd.get('total_trades', 0) for fd in fold_reports)
    all_r     = sum(fd.get('total_r', 0.0)     for fd in fold_reports)
    exp_vals  = [fd.get('expectancy_r', 0.0)   for fd in fold_reports if fd.get('total_trades', 0) > 0]
    wr_vals   = [fd.get('win_rate', 0.0)        for fd in fold_reports if fd.get('total_trades', 0) > 0]
    avg_exp   = sum(exp_vals) / max(len(exp_vals), 1)
    avg_wr    = sum(wr_vals)  / max(len(wr_vals),  1)
    tot_long  = sum(fd.get('n_long',  fd.get('direction_stats', {}).get('long_trades',  0)) for fd in fold_reports)
    tot_short = sum(fd.get('n_short', fd.get('direction_stats', {}).get('short_trades', 0)) for fd in fold_reports)

    print(f"  {'TOTAL/AVG':<6} {all_n:>8} {all_r:>10.3f} {avg_exp:>10.4f} "
          f"{avg_wr:>6.1%} {tot_long:>7} {tot_short:>7} {'':>11}")
    print()


def _print_direction_split(fold_reports: List[Dict]) -> None:
    """Print long vs short expectancy from canonical direction_stats in fold reports."""
    longs_r, shorts_r = [], []
    for fd in fold_reports:
        ds = fd.get('direction_stats', {})
        n_l = ds.get('long_trades',  0)
        n_s = ds.get('short_trades', 0)
        if n_l > 0:
            longs_r.extend([ds.get('long_expectancy_r',  0.0)] * n_l)
        if n_s > 0:
            shorts_r.extend([ds.get('short_expectancy_r', 0.0)] * n_s)
    if not longs_r and not shorts_r:
        return
    avg_l = sum(longs_r)  / max(len(longs_r),  1)
    avg_s = sum(shorts_r) / max(len(shorts_r), 1)
    print(f"  Long  trades: {len(longs_r):5d}   avg E[R]={avg_l:+.4f}")
    print(f"  Short trades: {len(shorts_r):5d}   avg E[R]={avg_s:+.4f}")
    print()


def _print_score_decile_table(records: List[Dict]) -> None:
    """Print 10-bucket score-decile R table from candidate_logger records."""
    taken = [r for r in records if r.get("taken")]
    if len(taken) < 20:
        print("  (insufficient trades for score-decile analysis)")
        return
    taken_sorted = sorted(taken, key=lambda r: r.get("final_score", r.get("raw_score", 0.0)))
    n = len(taken_sorted)
    n_buckets = 10
    bsz = n // n_buckets
    print(f"\n  Score decile table  (10 buckets, lowest → highest score)")
    print(f"  {'Decile':>7} {'Score lo':>10} {'Score hi':>10} "
          f"{'Trades':>8} {'Avg R':>10} {'WR':>8}")
    print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    monotonic = True
    prev_avg = None
    for d in range(n_buckets):
        s = d * bsz
        e = (d + 1) * bsz if d < n_buckets - 1 else n
        bucket = taken_sorted[s:e]
        if not bucket:
            continue
        scores = [r.get("final_score", r.get("raw_score", 0.0)) for r in bucket]
        rs = [_oracle_r(r) for r in bucket]
        avg_r = sum(rs) / len(rs)
        wr = sum(1 for v in rs if v > 0) / len(rs)
        lo, hi = scores[0], scores[-1]
        mono_ok = prev_avg is None or avg_r >= prev_avg - 0.02
        if not mono_ok:
            monotonic = False
        mark = "" if mono_ok else "↓"
        print(f"  {d+1:>7} {lo:>10.4f} {hi:>10.4f} "
              f"{len(bucket):>8} {avg_r:>10.4f} {wr:>7.1%} {mark}")
        prev_avg = avg_r
    status = "PASS" if monotonic else "FAIL"
    print(f"  Monotonic score→R: {status}")
    print()


def _print_gate_block_table(records: List[Dict]) -> None:
    """Print gate block count and oracle_R table from candidate_logger records."""
    blocked = [r for r in records if not r.get("taken")]
    if not blocked:
        print("  (no blocked candidates)")
        return
    gate_cnt: Dict[str, int]   = defaultdict(int)
    gate_r:   Dict[str, float] = defaultdict(float)
    for r in blocked:
        g = r.get("block_reason") or "unknown"
        gate_cnt[g] += 1
        gate_r[g]   += _oracle_r(r)
    total_blocked = len(blocked)
    print(f"\n  Gate block breakdown  (n_blocked={total_blocked})")
    print(f"  {'Gate':<28} {'Count':>8} {'% of blk':>10} {'Oracle R':>10} {'Avg R':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for gate, cnt in sorted(gate_cnt.items(), key=lambda x: -x[1]):
        pct = cnt / total_blocked * 100
        total_r = gate_r[gate]
        avg_r   = total_r / cnt if cnt else 0.0
        print(f"  {gate:<28} {cnt:>8} {pct:>9.1f}% {total_r:>10.3f} {avg_r:>10.4f}")
    print()


# ─────────────────────────────────────────────
# Mode: smoke_wf
# ─────────────────────────────────────────────

def mode_smoke_wf(args: argparse.Namespace) -> None:
    log.info("[validate] mode=smoke_wf  symbols=%s  epochs=15  folds=2  train_months=3  test_weeks=3",
             SMOKE_SYMBOLS)
    t0 = time.time()
    records, wf_report = _run_wf(
        symbols=SMOKE_SYMBOLS, epochs=15, batch_size=128, lr=3e-4,
        train_months=3, test_months=1, max_folds=2, seed=42, test_weeks=3,
    )
    elapsed = time.time() - t0
    fold_reports = (wf_report or {}).get("folds", [])
    _print_fold_reports("smoke_wf", fold_reports)
    _print_direction_split(fold_reports)
    _print_gate_block_table(records)
    _print_score_decile_table(records)
    candidate_metrics = _compute_metrics(records)
    _print_metrics_table("smoke_wf — compact summary (candidate log)", candidate_metrics)
    fold_summary_from_records = _fold_breakdown(records)
    payload = {
        "mode": "smoke_wf", "elapsed_s": round(elapsed, 1),
        "symbols": SMOKE_SYMBOLS,
        "metrics": candidate_metrics,
        "fold_summary": fold_summary_from_records,
        "n_candidate_records": len(records),
        "wf_report": wf_report,
    }
    out = _save_run("smoke_wf", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: canary_wf
# ─────────────────────────────────────────────

def mode_canary_wf(args: argparse.Namespace) -> None:
    folds = getattr(args, "folds", 3)
    log.info("[validate] mode=canary_wf  symbols=%s  epochs=30  folds=%d", CANARY_SYMBOLS, folds)
    t0 = time.time()
    records, wf_report = _run_wf(
        symbols=CANARY_SYMBOLS, epochs=30, batch_size=128, lr=3e-4,
        train_months=6, test_months=1, max_folds=folds, seed=42,
    )
    elapsed = time.time() - t0
    fold_reports = (wf_report or {}).get("folds", [])
    _print_fold_reports("canary_wf", fold_reports)
    _print_direction_split(fold_reports)
    _print_gate_block_table(records)
    _print_score_decile_table(records)
    candidate_metrics = _compute_metrics(records)
    _print_metrics_table("canary_wf — compact summary (candidate log)", candidate_metrics)
    fold_summary_from_records = _fold_breakdown(records)
    payload = {
        "mode": "canary_wf", "elapsed_s": round(elapsed, 1),
        "symbols": CANARY_SYMBOLS, "max_folds": folds,
        "metrics": candidate_metrics,
        "fold_summary": fold_summary_from_records,
        "n_candidate_records": len(records),
        "wf_report": wf_report,
    }
    out = _save_run("canary_wf", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: candidate_diff
# ─────────────────────────────────────────────

def _top20_decision_changes(base: List[Dict], new: List[Dict]) -> None:
    """Print top-20 bars where taken/block_reason changed between two runs."""
    base_map = {(r.get("bar_idx", 0), r.get("symbol", ""), r.get("timestamp", 0)): r
                for r in base}
    new_map  = {(r.get("bar_idx", 0), r.get("symbol", ""), r.get("timestamp", 0)): r
                for r in new}
    changes = []
    for key, r_b in base_map.items():
        r_n = new_map.get(key)
        if r_n is None:
            continue
        taken_b = bool(r_b.get("taken"))
        taken_n = bool(r_n.get("taken"))
        reason_b = r_b.get("block_reason", "")
        reason_n = r_n.get("block_reason", "")
        if taken_b != taken_n or reason_b != reason_n:
            oracle = _oracle_r(r_b)
            changes.append({
                "bar_idx": key[0], "symbol": key[1],
                "old_taken": taken_b, "new_taken": taken_n,
                "old_reason": reason_b or "taken",
                "new_reason": reason_n or "taken",
                "oracle_r": oracle,
                "impact": abs(oracle),
            })
    changes.sort(key=lambda x: -x["impact"])
    top = changes[:20]
    print(f"\n{'=' * 72}")
    print(f"  TOP-20 DECISION CHANGES  (baseline → new)  total_changes={len(changes)}")
    print('=' * 72)
    print(f"  {'bar_idx':>8} {'sym':<12} {'old':>20} {'new':>20} {'oracle_R':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*20} {'-'*20} {'-'*10}")
    for c in top:
        old_str = "TAKEN" if c["old_taken"] else f"BLOCKED({c['old_reason']})"
        new_str = "TAKEN" if c["new_taken"] else f"BLOCKED({c['new_reason']})"
        print(f"  {c['bar_idx']:>8} {c['symbol']:<12} {old_str:>20} {new_str:>20} {c['oracle_r']:>10.4f}")
    print()


def mode_candidate_diff(args: argparse.Namespace) -> None:
    """Run WF, dump per-candidate CSV, and save run JSON.

    Flags
    -----
    --folds N        : number of WF folds to run (default 2).
    --run JSON_PATH  : save the run JSON output to this path (default: auto-named
                       in validate_runs/).  Pass this to 'compare' for A/B diffing.
    --csv PATH       : save the candidate CSV to PATH (default: auto-named).
    --vs CSV_PATH    : after running, load this baseline CSV and print top-20
                       decision changes sorted by |oracle_R|.
    """
    folds    = getattr(args, "folds", 2)
    vs_path  = getattr(args, "vs", None)
    run_out  = getattr(args, "run", None)
    csv_out  = getattr(args, "csv", None)

    log.info("[validate] mode=candidate_diff  symbols=ALL20  folds=%d", folds)
    t0 = time.time()
    records, wf_report = _run_wf(
        symbols=ALL_SYMBOLS, epochs=30, batch_size=128, lr=3e-4,
        train_months=6, test_months=1, max_folds=folds, seed=42,
    )
    elapsed = time.time() - t0
    log.info("[validate] WF complete in %.1fs  total_records=%d", elapsed, len(records))

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(csv_out) if csv_out else (RUNS_DIR / f"candidate_diff_{ts_str}.csv")
    _save_csv(records, csv_path)
    print(f"CSV saved: {csv_path}")

    taken   = [r for r in records if r.get("taken")]
    blocked = [r for r in records if not r.get("taken")]
    all_r   = [_oracle_r(r) for r in records]
    taken_r = [_oracle_r(r) for r in taken]

    oracle_total  = sum(all_r)
    taken_total   = sum(taken_r)
    blocked_total = sum(_oracle_r(r) for r in blocked)
    capture       = taken_total / oracle_total if oracle_total != 0 else 0.0

    gate_oracle: Dict[str, float] = defaultdict(float)
    gate_count:  Dict[str, int]   = defaultdict(int)
    for r in blocked:
        g = r.get("block_reason") or "unknown"
        gate_oracle[g] += _oracle_r(r)
        gate_count[g]  += 1

    print(f"\n{'=' * 60}")
    print("  CANDIDATE DIFF — Gate oracle R left on the table")
    print('=' * 60)
    print(f"  All candidates : {len(records):5d}   oracle_R={oracle_total:.2f}")
    print(f"  Taken          : {len(taken):5d}   oracle_R={taken_total:.2f}")
    print(f"  Blocked        : {len(blocked):5d}   oracle_R={blocked_total:.2f}")
    print(f"  Capture rate   : {capture:.1%}")
    print(f"\n  {'Gate':<24} {'Blocked':>8} {'Oracle R':>10} {'Avg R':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10}")
    for gate, cnt in sorted(gate_count.items(), key=lambda x: -abs(gate_oracle[x[0]])):
        avg = gate_oracle[gate] / cnt if cnt else 0
        print(f"  {gate:<24} {cnt:>8} {gate_oracle[gate]:>10.3f} {avg:>10.4f}")
    print()

    metrics = _compute_metrics(records)
    _print_metrics_table("candidate_diff trade metrics", metrics)

    if vs_path:
        log.info("[validate] Comparing new run vs baseline: %s", vs_path)
        baseline_records = _load_csv(vs_path)
        _top20_decision_changes(baseline_records, records)

    payload = {
        "mode": "candidate_diff",
        "symbols": "ALL20",
        "source": "wf_run",
        "max_folds": folds,
        "elapsed_s": round(elapsed, 1),
        "csv_path": str(csv_path),
        "metrics": metrics,
        "oracle_total_all": round(oracle_total, 3),
        "oracle_total_taken": round(taken_total, 3),
        "capture_rate": round(capture, 4),
        "n_candidates": len(records),
        "gate_oracle_r": {g: round(v, 3) for g, v in gate_oracle.items()},
        "gate_block_count": dict(gate_count),
        "wf_report": wf_report,
    }
    if run_out:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(run_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("[validate] Results saved → %s", out)
    else:
        out = _save_run("candidate_diff", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: compare
# ─────────────────────────────────────────────

def mode_compare(args: argparse.Namespace) -> None:
    path_a, path_b = args.files
    log.info("[validate] mode=compare  A=%s  B=%s", path_a, path_b)
    run_a = _load_run(path_a)
    run_b = _load_run(path_b)
    m_a = run_a.get("metrics", run_a)
    m_b = run_b.get("metrics", run_b)
    label_a = f"{run_a.get('mode', '?')} [{Path(path_a).stem}]"
    label_b = f"{run_b.get('mode', '?')} [{Path(path_b).stem}]"
    _print_compare_table(label_a, label_b, m_a, m_b)
    out = _save_run("compare", {
        "mode": "compare", "file_a": path_a, "file_b": path_b,
        "metrics_a": m_a, "metrics_b": m_b,
    })
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Modes:\n"
        "  unit_audit     -- Pytest precision audit + torch-free config checks\n"
        "  smoke_wf       -- 2-symbol 2-fold walk-forward (BTC/ETH)\n"
        "  canary_wf      -- 4-symbol N-fold walk-forward (BTC/ETH/SOL/BNB)\n"
        "  candidate_diff -- Per-candidate gate oracle R breakdown (CSV output)\n"
        "  compare        -- Diff two validate_runs JSON files\n"
    )
    p = argparse.ArgumentParser(
        description="V5 fast validation harness",
        epilog=epilog,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("unit_audit",
                   help="Pytest precision audit + torch-free config checks")

    sub.add_parser("smoke_wf",
                   help="2-symbol 2-fold walk-forward smoke test (BTC/ETH)")

    canary = sub.add_parser("canary_wf",
                            help="4-symbol N-fold walk-forward (BTC/ETH/SOL/BNB)")
    canary.add_argument("--folds", type=int, default=3,
                        help="Number of folds (default: 3)")

    diff = sub.add_parser("candidate_diff",
                          help="Run WF + dump per-candidate CSV and save run JSON")
    diff.add_argument("--folds", type=int, default=2,
                      help="Number of WF folds to run (default: 2)")
    diff.add_argument("--run", metavar="JSON_PATH",
                      help="Save run JSON output to this path (default: auto-named in validate_runs/). "
                           "Use this path with 'compare' for A/B diffing.")
    diff.add_argument("--csv", metavar="PATH",
                      help="Save candidate CSV to PATH (default: auto-named in validate_runs/)")
    diff.add_argument("--vs", metavar="CSV_PATH",
                      help="Load this baseline candidate CSV and print top-20 decision changes")

    cmp = sub.add_parser("compare",
                         help="Diff two validate_runs JSON files (including gate block pct)")
    cmp.add_argument("files", nargs=2, metavar="FILE",
                     help="Two JSON run files to compare")

    return p


def main() -> None:
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    {
        "unit_audit":     mode_unit_audit,
        "smoke_wf":       mode_smoke_wf,
        "canary_wf":      mode_canary_wf,
        "candidate_diff": mode_candidate_diff,
        "compare":        mode_compare,
    }[args.mode](args)


if __name__ == "__main__":
    main()
