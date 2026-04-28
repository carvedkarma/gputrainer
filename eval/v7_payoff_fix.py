"""V7 Asymmetric Payoff Fix — simulate stops + sizing + tighter threshold.

Compares 4 configurations on the same 7-symbol, 5-fold walk-forward dataset:

  A. Baseline             — top 10% |pred|, fixed 60-min hold, equal size,
                            no filter
  B. Cell filter          — same as A but only cells with prior-fold mean > 0
  C. Full bundle (top 10) — cell filter + adaptive exits (1.5×vol_16 stop +
                            30-min underwater time exit) + confidence sizing
  D. Full bundle (top 2.5)— same as C but only the top quintile within decile
                            (effectively top ~2% of all predictions)

Metrics per config: trade count, gross WR, mean gross / net per trade (bps,
weight-adjusted), Sharpe-like, cumulative net, max drawdown. Cost gate at
2/4/6/8 bps round-trip. Pre-committed verdict.

Run:  python -m gpu_trainer.eval.v7_payoff_fix
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer.eval.v7_signal_audit_augmented import (
    build_features, build_targets, load_symbol, walk_forward_indices)
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("payoff_fix")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
           "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
TARGET = "sign_60m"
TOP_DECILE_Q = 0.90      # top 10% within fold
STOP_VOL_MULT = 1.5      # hard stop at 1.5 × vol_16 adverse move
HOLD_BARS = 4            # 60-minute hold horizon (4 × 15-min)
TIME_STOP_BAR = 2        # if underwater after 2 bars (30 min), exit
SIZING_WEIGHTS = {0: 0.25, 1: 0.50, 2: 0.75, 3: 0.90, 4: 1.00}
OUT_MD = Path(".local/reports/v7_payoff_fix.md")
OUT_JSON = Path(".local/reports/v7_payoff_fix.json")


def _hgbr():
    return HistGradientBoostingRegressor(
        max_iter=120, max_depth=4, learning_rate=0.05,
        min_samples_leaf=300, max_bins=128, random_state=0,
        early_stopping=False)


def hour_session(h: int) -> str:
    if 0 <= h < 7: return "Asia"
    if 7 <= h < 14: return "EU"
    if 14 <= h < 21: return "US"
    return "Late"


def quintile_bins(x: np.ndarray, n: int = 5) -> np.ndarray:
    out = np.full(len(x), -1, dtype=int)
    finite = np.isfinite(x)
    if finite.sum() < n:
        return out
    ranks = pd.Series(x[finite]).rank(pct=True, method="average").to_numpy()
    out[finite] = np.clip((ranks * n).astype(int), 0, n - 1)
    return out


def simulate_exits_vec(entries: np.ndarray, H: np.ndarray, L: np.ndarray,
                        C: np.ndarray, dirs: np.ndarray, stop_pct: np.ndarray,
                        time_stop_bar: int = TIME_STOP_BAR
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised exit simulator.

    Inputs are all length-N (or NxHOLD_BARS for the OHLC matrices).
    Returns (gross_log_return, exit_bar_idx, reason_code) where reason_code
    is 0=stop, 1=time_stop, 2=fixed, 3=nan.

    Conservative tie-break: stop hits before any target/time-stop in the same
    bar. If `stop_pct` is NaN/<=0 for a trade, that trade falls through to the
    fixed exit (no stop applied).
    """
    N, B = H.shape
    assert B == HOLD_BARS

    # Effective stop levels; NaN for trades with bad stop_pct → no stop hits
    sp = np.where(np.isfinite(stop_pct) & (stop_pct > 0), stop_pct, np.nan)
    stop_long = entries * (1.0 - sp)     # (N,)
    stop_short = entries * (1.0 + sp)

    # Per-bar hit masks
    is_long = (dirs > 0)[:, None]                     # (N,1)
    is_short = (dirs < 0)[:, None]
    long_hit = is_long & (L <= stop_long[:, None])    # (N,B)  NaN comparisons → False
    short_hit = is_short & (H >= stop_short[:, None])
    hit = long_hit | short_hit                        # (N,B)
    any_hit = hit.any(axis=1)
    first_hit = np.where(any_hit, hit.argmax(axis=1), B)  # B if no hit

    # Time-stop trigger: bar `time_stop_bar` close is adverse (and no earlier stop)
    ts_close = C[:, time_stop_bar]
    ts_ret = np.log(ts_close / entries) * dirs
    time_stop_active = (ts_ret < 0) & np.isfinite(ts_ret) & (first_hit > time_stop_bar)

    # Exit bar selection
    exit_bar = np.where(first_hit < B, first_hit,
                        np.where(time_stop_active, time_stop_bar, B - 1))
    reason = np.where(first_hit < B, 0,
                      np.where(time_stop_active, 1, 2))  # 0=stop,1=time,2=fixed

    # Compute returns
    # stop returns
    stop_ret_long = np.log(stop_long / entries)             # ≈ −sp
    stop_ret_short = -np.log(stop_short / entries)          # ≈ −sp
    stop_ret = np.where(dirs > 0, stop_ret_long, stop_ret_short)
    # close-based returns at exit_bar
    rng = np.arange(N)
    close_at_exit = C[rng, exit_bar.clip(0, B - 1)]
    close_ret = np.log(close_at_exit / entries) * dirs

    gross = np.where(reason == 0, stop_ret, close_ret)
    # NaN check on chosen close
    bad = ~np.isfinite(gross)
    gross = np.where(bad, 0.0, gross)
    reason = np.where(bad, 3, reason)
    return gross.astype(np.float64), exit_bar.astype(np.int64), reason.astype(np.int64)


_REASON_NAMES = {0: "stop", 1: "time_stop", 2: "fixed", 3: "nan"}


def collect_symbol_trades(symbol: str) -> pd.DataFrame:
    """For each top-decile signal, record both the fixed-hold and the
    adaptive-exit gross return, plus diagnostic context for cell filtering."""
    df = load_symbol(symbol)
    if df.empty or len(df) < 20000:
        log.warning("%s: insufficient bars (%d)", symbol, len(df))
        return pd.DataFrame()
    feats = build_features(df)
    targs = build_targets(df)
    y = targs[TARGET].to_numpy()
    fwd = targs["ret_60m"].to_numpy()

    feat_warmup = feats.notna().sum(axis=1)
    valid_from = max(int((feat_warmup > 8).idxmax()), 200)

    feats_arr = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y = y[valid_from:]
    fwd = fwd[valid_from:]
    timestamps = df["timestamp"].to_numpy()[valid_from:]

    # Slice OHLC arrays for the same window
    close_arr = df["close"].astype("float64").to_numpy()[valid_from:]
    high_arr = df["high"].astype("float64").to_numpy()[valid_from:]
    low_arr = df["low"].astype("float64").to_numpy()[valid_from:]

    close_s = pd.Series(close_arr)
    logret = np.log(close_s).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(close_s) - np.log(close_s.shift(16))).to_numpy()
    dt_idx = pd.to_datetime(timestamps, unit="ms", utc=True)
    hour_utc = dt_idx.hour.to_numpy()

    folds = walk_forward_indices(timestamps)
    if not folds:
        log.warning("%s: not enough history", symbol)
        return pd.DataFrame()

    rows = []
    for fold_i, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr = feats_arr[tlo:thi]; y_tr = y[tlo:thi]
        X_te = feats_arr[slo:shi]
        m_tr = np.isfinite(y_tr)
        if m_tr.sum() < 500:
            continue
        m = _hgbr()
        m.fit(X_tr[m_tr], y_tr[m_tr])
        pred = m.predict(X_te)

        ok = np.isfinite(pred)
        if ok.sum() < 100:
            continue
        thresh = np.quantile(np.abs(pred[ok]), TOP_DECILE_Q)
        sel = ok & (np.abs(pred) >= thresh)
        if sel.sum() < 10:
            continue

        sel_idx = np.where(sel)[0]
        global_idx = slo + sel_idx
        # Drop trades whose exit window runs off the end of the array
        keep = (global_idx + HOLD_BARS) < len(close_arr)
        sel_idx = sel_idx[keep]
        global_idx = global_idx[keep]
        if len(global_idx) < 10:
            continue

        abs_pred_sel = np.abs(pred[slo + sel_idx - slo])  # = |pred|[sel_idx]
        ap_q = quintile_bins(abs_pred_sel, 5)
        v_q = quintile_bins(vol_16[global_idx], 5)
        direction = np.sign(pred[sel_idx]).astype(int)
        # Sign of trend at entry vs trade direction
        sign_tr = np.sign(trend_16[global_idx])
        regime = np.where(sign_tr == 0, "FLAT",
                          np.where(sign_tr == direction, "WITH", "COUNTER"))
        session = np.array([hour_session(int(h))
                            for h in hour_utc[global_idx]])

        # --- vectorised per-fold exit simulation ---
        gi = global_idx
        entries = close_arr[gi]
        valid_entry = np.isfinite(entries) & (entries > 0)
        if not valid_entry.any():
            continue
        # Build (N, HOLD_BARS) OHLC matrices for next bars
        offsets = np.arange(1, 1 + HOLD_BARS)
        idx_mat = gi[:, None] + offsets[None, :]
        H = high_arr[idx_mat]
        L = low_arr[idx_mat]
        C = close_arr[idx_mat]
        sp = STOP_VOL_MULT * vol_16[gi]
        gross_adapt, exit_bar, reason_code = simulate_exits_vec(
            entries, H, L, C, direction.astype(np.float64), sp)
        gross_fixed = fwd[global_idx]

        # Mask: valid entry + finite fixed return (so baseline is comparable)
        finite_fixed = np.isfinite(gross_fixed)
        keep_mask = valid_entry & finite_fixed
        if not keep_mask.any():
            continue
        n_kept = int(keep_mask.sum())

        weights = np.array([SIZING_WEIGHTS.get(int(q), 0.0) for q in ap_q])
        ts_kept = timestamps[gi]
        risk_pct = sp  # = STOP_VOL_MULT * vol_16[gi]; planned R per trade
        for j in np.where(keep_mask)[0]:
            rows.append({
                "symbol": symbol,
                "fold": int(fold_i),
                "ts": int(ts_kept[j]),
                "abs_pred_q": int(ap_q[j]),
                "vol_q": int(v_q[j]),
                "regime": str(regime[j]),
                "session": str(session[j]),
                "direction": int(direction[j]),
                "gross_fixed": float(gross_fixed[j]),
                "gross_adapt": float(gross_adapt[j]),
                "exit_reason": _REASON_NAMES[int(reason_code[j])],
                "exit_bar": int(exit_bar[j]),
                "weight": float(weights[j]),
                "risk_pct": float(risk_pct[j]),
            })
    out = pd.DataFrame(rows)
    log.info("%s: %d trades collected", symbol, len(out))
    return out


def apply_walk_forward_filter(trades: pd.DataFrame,
                              gross_col: str,
                              cost_frac: float,
                              min_prior_n: int = 30) -> pd.Series:
    """Build per-fold mask: trade only if prior-fold cell-mean net > 0."""
    if trades.empty:
        return pd.Series(dtype=bool)
    keep = np.zeros(len(trades), dtype=bool)
    cells = list(zip(trades["abs_pred_q"], trades["vol_q"],
                     trades["regime"], trades["session"]))
    nets = (trades[gross_col].to_numpy() - cost_frac)
    folds = trades["fold"].to_numpy()
    for fold in sorted(np.unique(folds)):
        in_fold = folds == fold
        if fold == 0:
            keep[in_fold] = True
            continue
        prior = folds < fold
        cell_sum, cell_cnt = {}, {}
        for c, n in zip([cells[i] for i in np.where(prior)[0]], nets[prior]):
            cell_sum[c] = cell_sum.get(c, 0.0) + n
            cell_cnt[c] = cell_cnt.get(c, 0) + 1
        for i in np.where(in_fold)[0]:
            c = cells[i]
            cnt = cell_cnt.get(c, 0)
            if cnt < min_prior_n:
                keep[i] = True
            else:
                keep[i] = (cell_sum[c] / cnt) > 0
    return pd.Series(keep, index=trades.index)


def cumulative_metrics(net_per_trade: np.ndarray,
                       weights: np.ndarray | None = None) -> dict:
    """Sharpe-like, max drawdown over weighted cumulative bps curve."""
    if len(net_per_trade) == 0:
        return {"n": 0, "mean_bps": 0.0, "std_bps": 0.0, "sharpe": 0.0,
                "cum_bps": 0.0, "max_dd_bps": 0.0, "wr": 0.0,
                "n_eff_capital": 0.0}
    bps = net_per_trade * 1e4
    if weights is None:
        weights = np.ones_like(bps)
    w_sum = weights.sum()
    if w_sum <= 0:
        return {"n": int(len(bps)), "mean_bps": 0.0, "std_bps": 0.0,
                "sharpe": 0.0, "cum_bps": 0.0, "max_dd_bps": 0.0, "wr": 0.0,
                "n_eff_capital": 0.0}
    weighted_bps = bps * weights
    mu = float(weighted_bps.sum() / w_sum)
    if len(bps) > 1 and weights.sum() > 0:
        # weighted variance
        var = float(((weights * (bps - mu) ** 2).sum() / w_sum))
        sd = float(np.sqrt(var))
    else:
        sd = 0.0
    n_eff = float(w_sum)
    sharpe = (mu / sd * np.sqrt(n_eff)) if sd > 0 else 0.0
    cum = np.cumsum(weighted_bps)
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    wr = float((bps > 0).mean() * 100)
    return {"n": int(len(bps)), "mean_bps": mu, "std_bps": sd,
            "sharpe": sharpe, "cum_bps": float(cum[-1]) if len(cum) else 0.0,
            "max_dd_bps": max_dd, "wr": wr, "n_eff_capital": n_eff}


def cost_gate(gross: np.ndarray, weights: np.ndarray, cost_bps: float) -> dict:
    if len(gross) == 0 or weights.sum() == 0:
        return {"n": 0, "mean_net_bps": 0.0, "passes": False}
    net_bps = gross * 1e4 - cost_bps
    weighted = (net_bps * weights).sum() / weights.sum()
    return {"n": int(len(gross)), "mean_net_bps": float(weighted),
            "passes": bool(weighted > 0)}


def evaluate_config(trades: pd.DataFrame, gross_col: str,
                    use_filter: bool, weighted: bool,
                    threshold_p: int | None,
                    cost_bps: float = 8.0) -> dict:
    """Compute book-wide and per-symbol metrics for one configuration."""
    df = trades.copy()
    if threshold_p is not None:
        df = df[df["abs_pred_q"] >= threshold_p].copy()
    if df.empty:
        return {"by_symbol": {}, "book": cumulative_metrics(np.array([]))}
    if use_filter:
        # Per symbol filter (cell stats are per-symbol-meaningful)
        keep_pieces = []
        for sym in df["symbol"].unique():
            s = df[df["symbol"] == sym]
            mask = apply_walk_forward_filter(s, gross_col, cost_bps * 1e-4)
            keep_pieces.append(mask)
        keep_full = pd.concat(keep_pieces).sort_index()
        df = df[keep_full.values]
    if df.empty:
        return {"by_symbol": {}, "book": cumulative_metrics(np.array([]))}
    weights = df["weight"].to_numpy() if weighted else np.ones(len(df))
    gross = df[gross_col].to_numpy()
    net = gross - cost_bps * 1e-4

    book = cumulative_metrics(net, weights)
    book["gross_wr"] = float((gross > 0).mean() * 100)
    book["mean_gross_bps"] = float((gross * 1e4 * weights).sum() / weights.sum()) \
        if weights.sum() > 0 else 0.0
    book["cost_gate"] = {f"{c}bps": cost_gate(gross, weights, c)
                         for c in (2.0, 4.0, 6.0, 8.0)}

    sym_metrics = {}
    for sym in df["symbol"].unique():
        s = df[df["symbol"] == sym]
        sg = s[gross_col].to_numpy()
        sw = s["weight"].to_numpy() if weighted else np.ones(len(s))
        sn = sg - cost_bps * 1e-4
        m = cumulative_metrics(sn, sw)
        m["gross_wr"] = float((sg > 0).mean() * 100)
        m["mean_gross_bps"] = float((sg * 1e4 * sw).sum() / sw.sum()) \
            if sw.sum() > 0 else 0.0
        m["cost_gate"] = {f"{c}bps": cost_gate(sg, sw, c)
                          for c in (2.0, 4.0, 6.0, 8.0)}
        sym_metrics[sym] = m
    return {"by_symbol": sym_metrics, "book": book}


def render_block(L: list, label: str, summary: dict) -> None:
    L.append(f"### {label}")
    L.append("")
    b = summary["book"]
    L.append(f"Book-wide: n={b['n']:,}, gross WR={b['gross_wr']:.2f}%, "
             f"mean gross={b['mean_gross_bps']:+.2f} bps, "
             f"mean net (8 bps)={b['mean_bps']:+.2f} bps, "
             f"net WR={b['wr']:.2f}%, "
             f"Sharpe-like={b['sharpe']:+.2f}, "
             f"cum net={b['cum_bps']:+,.0f} bps, "
             f"max DD={b['max_dd_bps']:+,.0f} bps.")
    L.append("")
    L.append("Cost gate:")
    L.append("")
    L.append("| cost | mean net | passes |")
    L.append("|---:|---:|:---:|")
    for c in ("2.0bps", "4.0bps", "6.0bps", "8.0bps"):
        cg = b["cost_gate"][c]
        L.append(f"| {c} | {cg['mean_net_bps']:+.2f} bps | "
                 f"{'✓' if cg['passes'] else '✗'} |")
    L.append("")
    L.append("| symbol | n | gross WR% | mean gross bps | mean net bps "
             "| net WR% | Sharpe | cum bps | maxDD bps | "
             "passes 8/6/4/2 bps |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for sym in sorted(summary["by_symbol"].keys()):
        s = summary["by_symbol"][sym]
        passes = "/".join(
            "✓" if s["cost_gate"][c]["passes"] else "✗"
            for c in ("8.0bps", "6.0bps", "4.0bps", "2.0bps"))
        L.append(f"| {sym} | {s['n']:,} | {s['gross_wr']:.2f} | "
                 f"{s['mean_gross_bps']:+.2f} | {s['mean_bps']:+.2f} | "
                 f"{s['wr']:.2f} | {s['sharpe']:+.2f} | "
                 f"{s['cum_bps']:+,.0f} | {s['max_dd_bps']:+,.0f} | "
                 f"{passes} |")
    L.append("")


def render_compare(L: list, configs: dict) -> None:
    L.append("## Side-by-side comparison")
    L.append("")
    L.append("| metric | A: Baseline | B: + Filter | C: + Stops+Sizing "
             "(top10) | D: + Tighter (top2) |")
    L.append("|---|---:|---:|---:|---:|")
    A, B, C, D = (configs[k]["book"] for k in ("A", "B", "C", "D"))

    def row(name, key, fmt="{:+.2f}", suf=""):
        return (f"| {name} | {fmt.format(A[key])}{suf} | "
                f"{fmt.format(B[key])}{suf} | {fmt.format(C[key])}{suf} | "
                f"{fmt.format(D[key])}{suf} |")

    L.append(f"| n trades | {A['n']:,} | {B['n']:,} | {C['n']:,} | {D['n']:,} |")
    L.append(row("gross WR %", "gross_wr"))
    L.append(row("net WR %", "wr"))
    L.append(row("mean gross / trade", "mean_gross_bps", suf=" bps"))
    L.append(row("mean NET (8bps) / trade", "mean_bps", suf=" bps"))
    L.append(row("Sharpe-like", "sharpe"))
    L.append(f"| cum net bps | {A['cum_bps']:+,.0f} | {B['cum_bps']:+,.0f} "
             f"| {C['cum_bps']:+,.0f} | {D['cum_bps']:+,.0f} |")
    L.append(f"| max DD bps | {A['max_dd_bps']:+,.0f} | {B['max_dd_bps']:+,.0f} "
             f"| {C['max_dd_bps']:+,.0f} | {D['max_dd_bps']:+,.0f} |")
    L.append("")
    L.append("Cost-gate pass count (book-wide):")
    L.append("")
    L.append("| cost | A | B | C | D |")
    L.append("|---:|:---:|:---:|:---:|:---:|")
    for c in ("2.0bps", "4.0bps", "6.0bps", "8.0bps"):
        L.append(f"| {c} | "
                 f"{'✓' if A['cost_gate'][c]['passes'] else '✗'} | "
                 f"{'✓' if B['cost_gate'][c]['passes'] else '✗'} | "
                 f"{'✓' if C['cost_gate'][c]['passes'] else '✗'} | "
                 f"{'✓' if D['cost_gate'][c]['passes'] else '✗'} |")
    L.append("")


def verdict(configs: dict) -> str:
    """Apply pre-committed verdict rule to the BEST of C and D."""
    def per_sym_pass(cfg, cost_key):
        return sum(1 for s in cfg["by_symbol"].values()
                   if s["cost_gate"][cost_key]["passes"])
    best_label, best_score = None, -1
    for k in ("C", "D"):
        cfg = configs[k]
        s8 = per_sym_pass(cfg, "8.0bps")
        s6 = per_sym_pass(cfg, "6.0bps")
        s4 = per_sym_pass(cfg, "4.0bps")
        score = s8 * 100 + s6 * 10 + s4
        if score > best_score:
            best_label, best_score = k, score
    cfg = configs[best_label]
    s8 = per_sym_pass(cfg, "8.0bps")
    s6 = per_sym_pass(cfg, "6.0bps")
    s4 = per_sym_pass(cfg, "4.0bps")
    if s8 >= 3:
        v = (f"**GO** — config {best_label} passes the 8 bps cost gate on "
             f"{s8}/7 symbols. Task #104 (V7 organism) is authorised, built "
             "around the filtered + adaptive-exit signal (not raw decile).")
    elif s6 >= 4:
        v = (f"**CONDITIONAL GO** — config {best_label} passes 6 bps on "
             f"{s6}/7 symbols. Authorised contingent on maker-only / "
             "rebate execution venue.")
    elif s4 >= 4:
        v = (f"**REDESIGN** — config {best_label} passes 4 bps on "
             f"{s4}/7 symbols. Narrow-scope build only; do NOT authorise "
             "the full V7 organism on this evidence.")
    else:
        v = (f"**PIVOT confirmed** — best config {best_label} passes 8/6/4 "
             f"bps on only {s8}/{s6}/{s4} symbols out of 7. Stop adding "
             "offline levers. Pivot to V5 paper trading on a real venue "
             "feed for ground-truth on execution friction before any "
             "further signal work.")
    return v + f"\n\n_Per-symbol pass counts (best config = {best_label}): " \
               f"{s8}/7 at 8 bps, {s6}/7 at 6 bps, {s4}/7 at 4 bps._"


CACHE_DIR = Path(".local/cache/v7_payoff_fix")


def collect_or_load(symbol: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{symbol}.parquet"
    if p.exists():
        log.info("%s: loading cached trades from %s", symbol, p)
        return pd.read_parquet(p)
    df = collect_symbol_trades(symbol)
    if not df.empty:
        df.to_parquet(p, index=False)
        log.info("%s: cached %d trades to %s", symbol, len(df), p)
    return df


def main():
    import sys
    args = sys.argv[1:]
    # `--collect SYM1 SYM2` → only collect those symbols (skip analysis)
    # `--analyze-only`     → require all caches present, run analysis
    # default              → collect all + analyze
    if args and args[0] == "--collect":
        for sym in (args[1:] or SYMBOLS):
            try:
                collect_or_load(sym)
            except Exception as e:
                log.warning("%s failed: %s", sym, str(e)[:200])
        log.info("collection step done")
        return
    if args and args[0] == "--analyze-only":
        pieces = []
        for sym in SYMBOLS:
            p = CACHE_DIR / f"{sym}.parquet"
            if p.exists():
                pieces.append(pd.read_parquet(p))
            else:
                log.warning("%s: no cache, skipping", sym)
        if not pieces:
            raise RuntimeError("no cached trades")
        raw = pd.concat(pieces, ignore_index=True)
    else:
        log.info("collecting trades for %d symbols ...", len(SYMBOLS))
        pieces = []
        for sym in SYMBOLS:
            try:
                t = collect_or_load(sym)
                if not t.empty:
                    pieces.append(t)
            except Exception as e:
                log.warning("%s failed: %s", sym, str(e)[:200])
        if not pieces:
            raise RuntimeError("no trades collected")
        raw = pd.concat(pieces, ignore_index=True)
    log.info("total raw trades: %d (mean exit_bar=%.2f, "
             "stop %.1f%% / time_stop %.1f%% / fixed %.1f%%)",
             len(raw), raw["exit_bar"].mean() + 1,
             (raw["exit_reason"] == "stop").mean() * 100,
             (raw["exit_reason"] == "time_stop").mean() * 100,
             (raw["exit_reason"] == "fixed").mean() * 100)

    # Configurations
    configs = {
        "A": evaluate_config(raw, "gross_fixed",
                             use_filter=False, weighted=False, threshold_p=None),
        "B": evaluate_config(raw, "gross_fixed",
                             use_filter=True, weighted=False, threshold_p=None),
        "C": evaluate_config(raw, "gross_adapt",
                             use_filter=True, weighted=True, threshold_p=None),
        "D": evaluate_config(raw, "gross_adapt",
                             use_filter=True, weighted=True, threshold_p=4),
    }

    L = []
    L.append("# V7 Asymmetric Payoff Fix — Before vs After")
    L.append("")
    L.append(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"Same dataset ({', '.join(SYMBOLS)}, ~180k 15-min bars each, "
             "5+ years). Same model (HGBR on `sign_60m`), same 5-fold "
             "walk-forward (24 m train / 6 m test, 16-bar embargo). Same "
             "top-decile selection. The only thing that changes is exit logic, "
             "sizing, threshold, and whether the cell filter is applied.")
    L.append("")
    L.append("**Adaptive exit per trade:** hard stop at "
             f"{STOP_VOL_MULT}× trailing 4-h vol_16 adverse excursion "
             "(walked through 15-min bar high/low; conservative tie-break "
             "= stop hits first); time stop at "
             f"{TIME_STOP_BAR * 15} min if underwater on that bar's close; "
             f"otherwise hold to {HOLD_BARS * 15} min.")
    L.append("")
    L.append("**Confidence sizing:** weight scales with within-decile "
             "|pred| quintile: " +
             ", ".join(f"P{k+1}={v}" for k, v in SIZING_WEIGHTS.items()))
    L.append("")
    L.append("**Tighter threshold (config D):** keep only top quintile P5 "
             "of the top decile = top ~2% of all predictions, then weighted.")
    L.append("")
    L.append("**Cell filter (B/C/D):** for fold N≥1, keep a trade only if "
             "its (|pred|-q × vol-q × regime × session) cell had positive "
             "mean net on folds 0..N−1 (≥30 prior obs; otherwise keep).")
    L.append("")
    L.append("## Results")
    L.append("")
    render_block(L, "A. Baseline — top 10%, fixed 60-min, equal size, no filter",
                 configs["A"])
    render_block(L, "B. + Cell filter (no other changes)", configs["B"])
    render_block(L, "C. + Adaptive exits + confidence sizing (top 10%)",
                 configs["C"])
    render_block(L, "D. + Tighter threshold (top ~2%)", configs["D"])
    render_compare(L, configs)
    L.append("## Verdict")
    L.append("")
    L.append(verdict(configs))
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT_MD)

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "symbols": SYMBOLS,
        "stop_vol_mult": STOP_VOL_MULT,
        "hold_bars": HOLD_BARS,
        "time_stop_bar": TIME_STOP_BAR,
        "sizing_weights": SIZING_WEIGHTS,
        "configs": configs,
        "exit_reason_share": {
            "stop": float((raw["exit_reason"] == "stop").mean()),
            "time_stop": float((raw["exit_reason"] == "time_stop").mean()),
            "fixed": float((raw["exit_reason"] == "fixed").mean()),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
