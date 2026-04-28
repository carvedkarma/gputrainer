"""V7 Filtered Simulation — apply ex-ante regime filter and compare vs unfiltered.

For each (symbol, fold), train HGBR on sign_60m, take top-decile of |pred|, and
record per-trade context: (|pred|-quintile within fold, vol-quintile within
fold, trend regime, session). Quintiling each fold within itself is fine —
that's just discretizing the test-set predictions, not leaking labels.

Then run the **out-of-sample** filter: for fold N, look up each trade's cell in
the cell-mean table built from folds 0..N-1. Trade only cells whose prior-fold
mean net is > 0. Fold 0 has no prior data so it trades unfiltered (baseline).

Compare filtered vs unfiltered on each symbol and book-wide:
  * trade count, gross WR, net WR
  * mean net per trade (bps), std, Sharpe-like
  * total realised net (bps cumulative), max drawdown of cumulative bps
  * cost-gate verdict at 8 / 6 / 4 / 2 bps round-trip

Run:  python -m gpu_trainer.eval.v7_filtered_simulation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gpu_trainer.eval.v7_signal_audit_augmented import (
    build_features, build_targets, load_symbol, walk_forward_indices)
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("filt_sim")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
           "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
TARGET = "sign_60m"
TOP_DECILE_Q = 0.90
COST_BPS = 8.0
COST_FRAC = COST_BPS * 1e-4
OUT_MD = Path(".local/reports/v7_filtered_simulation.md")
OUT_JSON = Path(".local/reports/v7_filtered_simulation.json")


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
    """Return integer bucket index 0..n-1 by percentile rank, NaN-safe (-1)."""
    out = np.full(len(x), -1, dtype=int)
    finite = np.isfinite(x)
    if finite.sum() < n:
        return out
    ranks = pd.Series(x[finite]).rank(pct=True, method="average").to_numpy()
    out[finite] = np.clip((ranks * n).astype(int), 0, n - 1)
    return out


def collect_symbol_trades(symbol: str) -> pd.DataFrame:
    """Walk forward on a symbol, return per-trade DataFrame with cell labels."""
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
    y = y[valid_from:]; fwd = fwd[valid_from:]
    timestamps = df["timestamp"].to_numpy()[valid_from:]

    close = df["close"].astype("float64").iloc[valid_from:].reset_index(drop=True)
    logret = np.log(close).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(close) - np.log(close.shift(16))).to_numpy()
    dt_idx = pd.to_datetime(timestamps, unit="ms", utc=True)
    hour_utc = dt_idx.hour.to_numpy()

    folds = walk_forward_indices(timestamps)
    if not folds:
        log.warning("%s: not enough history", symbol)
        return pd.DataFrame()

    rows = []
    for fold_i, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr = feats_arr[tlo:thi]; y_tr = y[tlo:thi]
        X_te = feats_arr[slo:shi]; fwd_te = fwd[slo:shi]
        m_tr = np.isfinite(y_tr)
        if m_tr.sum() < 500:
            continue
        m = _hgbr()
        m.fit(X_tr[m_tr], y_tr[m_tr])
        pred = m.predict(X_te)

        ok = np.isfinite(pred) & np.isfinite(fwd_te)
        if ok.sum() < 100:
            continue
        thresh = np.quantile(np.abs(pred[ok]), TOP_DECILE_Q)
        sel = ok & (np.abs(pred) >= thresh)
        if sel.sum() < 10:
            continue

        # Discretize WITHIN this fold's selected trades
        abs_pred_sel = np.abs(pred[sel])
        gi_local = np.arange(slo, shi)[sel]
        vol_sel = vol_16[gi_local]
        trend_sel = trend_16[gi_local]
        hour_sel = hour_utc[gi_local]

        ap_q = quintile_bins(abs_pred_sel, 5)
        v_q = quintile_bins(vol_sel, 5)

        direction = np.sign(pred[sel])
        gross = direction * fwd_te[sel]
        net = gross - COST_FRAC

        sign_trend = np.sign(trend_sel)
        regime = np.where(sign_trend == 0, "FLAT",
                          np.where(sign_trend == direction, "WITH", "COUNTER"))
        session = np.array([hour_session(int(h)) for h in hour_sel])

        for i in range(len(direction)):
            rows.append({
                "symbol": symbol,
                "fold": int(fold_i),
                "ts": int(timestamps[gi_local[i]]),
                "abs_pred_q": int(ap_q[i]),
                "vol_q": int(v_q[i]),
                "regime": str(regime[i]),
                "session": str(session[i]),
                "direction": int(direction[i]),
                "gross": float(gross[i]),
                "net": float(net[i]),
            })
    out = pd.DataFrame(rows)
    log.info("%s: %d trades across %d folds",
             symbol, len(out), out["fold"].nunique() if len(out) else 0)
    return out


def cumulative_metrics(net_bps_per_trade: np.ndarray) -> dict:
    """Sharpe-like, max drawdown over cumulative bps curve."""
    if len(net_bps_per_trade) == 0:
        return {"n": 0, "mean_bps": 0.0, "std_bps": 0.0, "sharpe": 0.0,
                "cum_bps": 0.0, "max_dd_bps": 0.0, "wr": 0.0}
    mu = float(np.mean(net_bps_per_trade))
    sd = float(np.std(net_bps_per_trade, ddof=1)) if len(net_bps_per_trade) > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(len(net_bps_per_trade))) if sd > 0 else 0.0
    cum = np.cumsum(net_bps_per_trade)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    wr = float((net_bps_per_trade > 0).mean() * 100)
    return {"n": int(len(net_bps_per_trade)),
            "mean_bps": mu, "std_bps": sd, "sharpe": sharpe,
            "cum_bps": float(cum[-1]), "max_dd_bps": max_dd,
            "wr": wr}


def cost_gate_at(gross_bps_per_trade: np.ndarray, cost_bps: float) -> dict:
    if len(gross_bps_per_trade) == 0:
        return {"n": 0, "mean_net_bps": 0.0, "passes": False}
    net = gross_bps_per_trade - cost_bps
    mu = float(net.mean())
    return {"n": int(len(gross_bps_per_trade)),
            "mean_net_bps": mu, "passes": bool(mu > 0)}


def apply_walk_forward_filter(trades: pd.DataFrame,
                              min_prior_n: int = 30) -> pd.DataFrame:
    """For each fold N>=1, keep trades whose (cell) mean net on folds 0..N-1 is > 0.

    Fold 0 has no prior history → we keep all of it (baseline behaviour). A cell
    needs at least `min_prior_n` prior trades in the lookup history to count;
    otherwise fall back to "trade it" (don't over-filter on noise)."""
    if trades.empty:
        return trades
    keep_mask = np.zeros(len(trades), dtype=bool)
    cells = list(zip(trades["abs_pred_q"], trades["vol_q"],
                     trades["regime"], trades["session"]))
    nets = trades["net"].to_numpy()
    folds = trades["fold"].to_numpy()

    for fold in sorted(np.unique(folds)):
        in_fold = folds == fold
        if fold == 0:
            keep_mask[in_fold] = True  # baseline for first fold
            continue
        prior = folds < fold
        # Build cell mean-net lookup from prior folds
        prior_cells = [c for c, p in zip(cells, prior) if p]
        prior_nets = nets[prior]
        cell_sums: dict = {}
        cell_counts: dict = {}
        for c, n in zip(prior_cells, prior_nets):
            cell_sums[c] = cell_sums.get(c, 0.0) + n
            cell_counts[c] = cell_counts.get(c, 0) + 1
        cell_mean = {c: cell_sums[c] / cell_counts[c] for c in cell_sums}

        for i in np.where(in_fold)[0]:
            c = cells[i]
            cnt = cell_counts.get(c, 0)
            if cnt < min_prior_n:
                keep_mask[i] = True  # not enough data — don't over-filter
            else:
                keep_mask[i] = cell_mean[c] > 0
    out = trades.copy()
    out["kept"] = keep_mask
    return out


def summarize(trades: pd.DataFrame, label: str) -> dict:
    """Compute per-symbol and book-wide metrics (gross + net @ multiple costs)."""
    sym_metrics = {}
    for sym in trades["symbol"].unique():
        s = trades[trades["symbol"] == sym]
        net_bps = s["net"].to_numpy() * 1e4
        gross_bps = s["gross"].to_numpy() * 1e4
        m = cumulative_metrics(net_bps)
        m["gross_wr"] = float((s["gross"] > 0).mean() * 100)
        m["mean_gross_bps"] = float(s["gross"].mean() * 1e4)
        m["cost_gate"] = {
            f"{c}bps": cost_gate_at(gross_bps, c) for c in (2.0, 4.0, 6.0, 8.0)
        }
        sym_metrics[sym] = m

    # book-wide
    net_bps_all = trades["net"].to_numpy() * 1e4
    gross_bps_all = trades["gross"].to_numpy() * 1e4
    book = cumulative_metrics(net_bps_all)
    book["gross_wr"] = float((trades["gross"] > 0).mean() * 100)
    book["mean_gross_bps"] = float(trades["gross"].mean() * 1e4)
    book["cost_gate"] = {
        f"{c}bps": cost_gate_at(gross_bps_all, c) for c in (2.0, 4.0, 6.0, 8.0)
    }
    return {"label": label, "by_symbol": sym_metrics, "book": book}


def render_section(L: list, title: str, summary: dict) -> None:
    L.append(f"## {title}")
    L.append("")
    book = summary["book"]
    L.append(f"**Book-wide** ({summary['label']}): "
             f"n={book['n']:,}, gross WR={book['gross_wr']:.2f}%, "
             f"mean gross={book['mean_gross_bps']:+.2f} bps, "
             f"mean net (8 bps cost)={book['mean_bps']:+.2f} bps, "
             f"net WR={book['wr']:.2f}%, "
             f"Sharpe-like={book['sharpe']:+.2f}, "
             f"cum net={book['cum_bps']:+,.0f} bps, "
             f"max DD={book['max_dd_bps']:+,.0f} bps.")
    L.append("")
    L.append("Cost gate (mean net per trade after round-trip cost):")
    L.append("")
    L.append("| cost | mean net | passes |")
    L.append("|---:|---:|:---:|")
    for c in ("2.0bps", "4.0bps", "6.0bps", "8.0bps"):
        cg = book["cost_gate"][c]
        L.append(f"| {c} | {cg['mean_net_bps']:+.2f} bps | "
                 f"{'✓' if cg['passes'] else '✗'} |")
    L.append("")
    L.append("Per-symbol:")
    L.append("")
    L.append("| symbol | n | gross WR% | mean gross bps | mean net bps "
             "| net WR% | Sharpe | cum bps | maxDD bps |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sym in sorted(summary["by_symbol"].keys()):
        s = summary["by_symbol"][sym]
        L.append(f"| {sym} | {s['n']:,} | {s['gross_wr']:.2f} | "
                 f"{s['mean_gross_bps']:+.2f} | "
                 f"{s['mean_bps']:+.2f} | {s['wr']:.2f} | "
                 f"{s['sharpe']:+.2f} | {s['cum_bps']:+,.0f} | "
                 f"{s['max_dd_bps']:+,.0f} |")
    L.append("")


def render_comparison(L: list, base: dict, filt: dict) -> None:
    L.append("## Side-by-side comparison: UNFILTERED vs FILTERED")
    L.append("")
    L.append("| metric | UNFILTERED | FILTERED | delta |")
    L.append("|---|---:|---:|---:|")
    bb, bf = base["book"], filt["book"]

    def fmt(a, b, suf=""):
        d = b - a
        return f"| {a:+.2f}{suf} | {b:+.2f}{suf} | {d:+.2f}{suf} |"

    L.append(f"| n trades | {bb['n']:,} | {bf['n']:,} | "
             f"{bf['n']-bb['n']:+,} ({(bf['n']-bb['n'])/bb['n']*100:+.1f}%) |")
    L.append(f"| gross WR % | {bb['gross_wr']:.2f} | {bf['gross_wr']:.2f} | "
             f"{bf['gross_wr']-bb['gross_wr']:+.2f} |")
    L.append(f"| mean gross bps {fmt(bb['mean_gross_bps'], bf['mean_gross_bps'])}")
    L.append(f"| mean net (8 bps) bps {fmt(bb['mean_bps'], bf['mean_bps'])}")
    L.append(f"| net WR % | {bb['wr']:.2f} | {bf['wr']:.2f} | "
             f"{bf['wr']-bb['wr']:+.2f} |")
    L.append(f"| Sharpe-like {fmt(bb['sharpe'], bf['sharpe'])}")
    L.append(f"| cum net bps | {bb['cum_bps']:+,.0f} | {bf['cum_bps']:+,.0f} "
             f"| {bf['cum_bps']-bb['cum_bps']:+,.0f} |")
    L.append(f"| max DD bps | {bb['max_dd_bps']:+,.0f} | {bf['max_dd_bps']:+,.0f} "
             f"| {bf['max_dd_bps']-bb['max_dd_bps']:+,.0f} |")
    L.append("")
    L.append("Cost gate flips (book-wide):")
    L.append("")
    L.append("| cost | UNFILTERED net | passes | FILTERED net | passes |")
    L.append("|---:|---:|:---:|---:|:---:|")
    for c in ("2.0bps", "4.0bps", "6.0bps", "8.0bps"):
        u = bb["cost_gate"][c]
        f = bf["cost_gate"][c]
        L.append(f"| {c} | {u['mean_net_bps']:+.2f} | "
                 f"{'✓' if u['passes'] else '✗'} | "
                 f"{f['mean_net_bps']:+.2f} | "
                 f"{'✓' if f['passes'] else '✗'} |")
    L.append("")


def main():
    log.info("collecting trades for %d symbols ...", len(SYMBOLS))
    all_trades = []
    for sym in SYMBOLS:
        try:
            t = collect_symbol_trades(sym)
            if not t.empty:
                all_trades.append(t)
        except Exception as e:
            log.warning("%s failed: %s", sym, str(e)[:200])
    if not all_trades:
        raise RuntimeError("no trades collected")
    raw = pd.concat(all_trades, ignore_index=True)
    log.info("total raw trades: %d", len(raw))

    # Apply walk-forward filter per symbol independently
    filtered_pieces = []
    for sym in raw["symbol"].unique():
        s = raw[raw["symbol"] == sym].copy()
        s = apply_walk_forward_filter(s, min_prior_n=30)
        filtered_pieces.append(s)
    raw_with_keep = pd.concat(filtered_pieces, ignore_index=True)
    kept = raw_with_keep[raw_with_keep["kept"]].copy()
    log.info("kept %d / %d trades after walk-forward filter (%.1f%%)",
             len(kept), len(raw_with_keep),
             len(kept) / len(raw_with_keep) * 100)

    base_summary = summarize(raw_with_keep, "UNFILTERED (baseline top-decile)")
    filt_summary = summarize(kept, "FILTERED (prior-fold cell-mean > 0)")

    L = []
    L.append("# V7 Filtered Simulation — Out-of-Sample Comparison")
    L.append("")
    L.append(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"For each of {len(SYMBOLS)} full-history symbols ({', '.join(SYMBOLS)}), "
             "we re-run the V7 best cell (`sign_60m` target, HGBR model) under the "
             "same 5-fold walk-forward (24 months train / 6 months test, 16-bar "
             "embargo). For each top-decile trade we record its cell label "
             "`(|pred|-quintile, vol-quintile, trend-regime, session)`.")
    L.append("")
    L.append("**Filter:** for fold N≥1, keep a trade only if its cell's mean net "
             "on folds 0..N−1 was > 0 (with at least 30 prior observations; "
             "otherwise keep, to avoid over-filtering on noise). Fold 0 has no "
             "prior data so it trades the baseline. **This is strictly "
             "out-of-sample.**")
    L.append("")
    L.append(f"Cost: {COST_BPS} bps round-trip on the headline; cost gate also "
             "checked at 6 / 4 / 2 bps.")
    L.append("")
    render_section(L, "UNFILTERED baseline (re-confirms previous audit)", base_summary)
    render_section(L, "FILTERED (ex-ante regime filter)", filt_summary)
    render_comparison(L, base_summary, filt_summary)

    L.append("## Verdict")
    L.append("")
    bb, bf = base_summary["book"], filt_summary["book"]
    delta = bf["mean_bps"] - bb["mean_bps"]
    if bf["cost_gate"]["8.0bps"]["passes"]:
        verdict = "**GO** — filter clears the 8 bps cost gate book-wide."
    elif bf["cost_gate"]["6.0bps"]["passes"]:
        verdict = ("**CONDITIONAL GO** — filter clears the 6 bps cost gate. "
                   "Maker-only or reduced-fee venue required to be tradeable.")
    elif bf["cost_gate"]["4.0bps"]["passes"]:
        verdict = ("**REDESIGN** — filter clears the 4 bps gate but not 6/8. "
                   "Add asymmetric exits + hold-horizon optimisation before "
                   "considering build authorisation.")
    else:
        verdict = ("**PIVOT confirmed** — filter improves net by "
                   f"{delta:+.2f} bps but doesn't clear any cost gate. "
                   "Either the cell-mean filter has insufficient predictive "
                   "stability across folds, or the residual signal is below "
                   "any reasonable cost floor.")
    L.append(verdict)
    L.append("")
    L.append(f"- Net per trade improved by **{delta:+.2f} bps** "
             f"({bb['mean_bps']:+.2f} → {bf['mean_bps']:+.2f}).")
    L.append(f"- Trade count changed by **{bf['n']-bb['n']:+,}** "
             f"({(bf['n']-bb['n'])/bb['n']*100:+.1f}%).")
    n_sym_pass_8 = sum(1 for s in filt_summary["by_symbol"].values()
                       if s["cost_gate"]["8.0bps"]["passes"])
    n_sym_pass_6 = sum(1 for s in filt_summary["by_symbol"].values()
                       if s["cost_gate"]["6.0bps"]["passes"])
    n_sym_pass_4 = sum(1 for s in filt_summary["by_symbol"].values()
                       if s["cost_gate"]["4.0bps"]["passes"])
    L.append(f"- Per-symbol cost-gate pass count: "
             f"**{n_sym_pass_8}/7** at 8 bps, "
             f"**{n_sym_pass_6}/7** at 6 bps, "
             f"**{n_sym_pass_4}/7** at 4 bps.")
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT_MD)

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "symbols": SYMBOLS,
        "cost_bps": COST_BPS,
        "unfiltered": base_summary,
        "filtered": filt_summary,
        "filter_keep_rate": len(kept) / len(raw_with_keep),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
