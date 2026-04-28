"""V7 Path A — Finalization checks before paper-trading.

Builds on v7_payoff_geometry's finding (winning geometry: top-X% |pred|,
no-stop, fixed 120-min hold). Adds:

  1. Selectivity sweep: top 0.5 / 1 / 1.5 / 2 / 2.5 / 3 % of |pred|
  2. Maker-only entry simulation (cost grid: 0 / 2 / 4 / 6 / 8 / 10 bps)
  3. Tail-risk reporting: worst 1/5/10% trades, MAE distribution,
     consecutive loser clusters, worst day/week, per-symbol max DD
  4. Rolling 30-day OUT-OF-SAMPLE thresholding with daily snapshots
     logged to .local/reports/v7_path_a_thresholds.csv

Whitelist (per user): ADA, XRP, AVAX, SOL (probationary), ETH disabled.
SOL appears in per-symbol detail but is excluded from the "tradeable book"
roll-up.

Cache: re-collect at top 5% (vs prior 2%) so we can sub-select 1% / 2%
in post and emulate rolling thresholds. Stores actual `pred_abs` float.

Run:  python -m gpu_trainer.eval.v7_path_a_finalize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer.eval.v7_signal_audit_augmented import (
    build_features, build_targets, load_symbol, walk_forward_indices)
from gpu_trainer.eval.v7_payoff_fix import _hgbr, hour_session, quintile_bins

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("path_a_final")

WHITELIST = ["ADAUSDT", "XRPUSDT", "AVAXUSDT", "SOLUSDT"]   # ETH dropped
PROBATIONARY = {"SOLUSDT"}
TRADEABLE = [s for s in WHITELIST if s not in PROBATIONARY]
TARGET = "sign_60m"
TOP_Q_CACHE = 0.95           # collect top 5% so we can sub-select 1% / 2%
HORIZON = 16                 # 16 × 15min = 4h forward bars stored
HOLD_BARS = 8                # 120-min winning geometry
COSTS_BPS = [0, 2, 4, 6, 8, 10]    # cost grid in bps round-trip
SELECTIVITY_PCTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # top-X%
ROLLING_DAYS = 30
CACHE_DIR = Path(".local/cache/v7_path_a")
OUT_MD = Path(".local/reports/v7_path_a_finalize.md")
OUT_JSON = Path(".local/reports/v7_path_a_finalize.json")
OUT_THRESH_CSV = Path(".local/reports/v7_path_a_thresholds.csv")


# --------------------------------------------------------------------------
# COLLECTION (top 5%, stores pred_abs)
# --------------------------------------------------------------------------

def collect(symbol: str) -> pd.DataFrame:
    df = load_symbol(symbol)
    if df.empty or len(df) < 20000:
        log.warning("%s: insufficient bars (%d)", symbol, len(df))
        return pd.DataFrame()
    feats = build_features(df)
    targs = build_targets(df)
    y = targs[TARGET].to_numpy()
    feat_warmup = feats.notna().sum(axis=1)
    valid_from = max(int((feat_warmup > 8).idxmax()), 200)
    feats_arr = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y = y[valid_from:]
    timestamps = df["timestamp"].to_numpy()[valid_from:]
    close_arr = df["close"].astype("float64").to_numpy()[valid_from:]
    high_arr = df["high"].astype("float64").to_numpy()[valid_from:]
    low_arr = df["low"].astype("float64").to_numpy()[valid_from:]
    close_s = pd.Series(close_arr)
    logret = np.log(close_s).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    folds = walk_forward_indices(timestamps)
    if not folds:
        return pd.DataFrame()

    rows = []
    Hcols = [f"H{i+1}" for i in range(HORIZON)]
    Lcols = [f"L{i+1}" for i in range(HORIZON)]
    Ccols = [f"C{i+1}" for i in range(HORIZON)]
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
        thresh = np.quantile(np.abs(pred[ok]), TOP_Q_CACHE)
        sel = ok & (np.abs(pred) >= thresh)
        sel_idx = np.where(sel)[0]
        global_idx = slo + sel_idx
        keep = (global_idx + HORIZON) < len(close_arr)
        sel_idx = sel_idx[keep]; global_idx = global_idx[keep]
        if len(global_idx) < 5:
            continue
        direction = np.sign(pred[sel_idx]).astype(int)
        gi = global_idx
        entries = close_arr[gi]
        valid_entry = np.isfinite(entries) & (entries > 0)
        if not valid_entry.any():
            continue
        offsets = np.arange(1, 1 + HORIZON)
        idx_mat = gi[:, None] + offsets[None, :]
        H = high_arr[idx_mat]; L = low_arr[idx_mat]; C = close_arr[idx_mat]
        for j in np.where(valid_entry)[0]:
            row = {"symbol": symbol, "fold": int(fold_i),
                   "ts": int(timestamps[gi[j]]),
                   "pred_abs": float(np.abs(pred[sel_idx[j]])),
                   "direction": int(direction[j]),
                   "entry": float(entries[j]),
                   "vol_16": float(vol_16[gi[j]])}
            for k in range(HORIZON):
                row[Hcols[k]] = float(H[j, k])
                row[Lcols[k]] = float(L[j, k])
                row[Ccols[k]] = float(C[j, k])
            rows.append(row)
    out = pd.DataFrame(rows)
    log.info("%s: %d top-5%% trades collected", symbol, len(out))
    return out


def ensure_cache() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for sym in WHITELIST:
        p = CACHE_DIR / f"{sym}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            log.info("%s: %d cached", sym, len(df))
        else:
            df = collect(sym)
            if not df.empty:
                df.to_parquet(p, index=False)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# WINNING GEOMETRY: time-exit, no-stop, hold=120m (8 bars)
# --------------------------------------------------------------------------

def gross_time_exit(t: pd.DataFrame, hb: int = HOLD_BARS) -> np.ndarray:
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    cl = t[f"C{hb}"].to_numpy()
    g = np.log(cl / e) * d
    return np.where(np.isfinite(g), g, 0.0)


def mae_per_trade(t: pd.DataFrame, hb: int = HOLD_BARS) -> np.ndarray:
    """Maximum Adverse Excursion over the held window, in fractional return.
    For longs: min low / entry - 1. For shorts: 1 - max high / entry.
    Returns NEGATIVE values (mae <= 0)."""
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    H = t[[f"H{i+1}" for i in range(hb)]].to_numpy()
    L = t[[f"L{i+1}" for i in range(hb)]].to_numpy()
    long_mae = np.log(L.min(axis=1) / e)              # negative
    short_mae = -np.log(H.max(axis=1) / e)            # negative
    mae = np.where(d > 0, long_mae, short_mae)
    return np.where(np.isfinite(mae), mae, 0.0)


# --------------------------------------------------------------------------
# SELECTORS
# --------------------------------------------------------------------------

def select_top_p_per_fold(t: pd.DataFrame, p_pct: float) -> pd.Series:
    """Within each (symbol, fold), keep the top-p% by pred_abs. p_pct in %."""
    keep = pd.Series(False, index=t.index)
    for (s, f), sub in t.groupby(["symbol", "fold"]):
        # Cache is top 5%. To get top p%, we need quantile q within the cache:
        # p% of all bars = (p/5)*100% of cache → take quantile (1 - p/5).
        q = max(0.0, 1.0 - (p_pct / 5.0))
        thresh = sub["pred_abs"].quantile(q)
        keep.loc[sub.index] = sub["pred_abs"] >= thresh
    return keep


def select_rolling_threshold(t: pd.DataFrame, p_pct: float,
                             prior_days: int = ROLLING_DAYS,
                             snapshot_path: Path | None = None
                             ) -> pd.Series:
    """At each entry, keep iff pred_abs >= rolling-prior-days quantile of
    cached pred_abs values (within same symbol). This emulates live OOS
    thresholding using only past information.

    Logs daily threshold snapshots if snapshot_path is given.
    """
    keep = pd.Series(False, index=t.index)
    snapshots = []  # (symbol, date, p_pct, threshold, n_above)
    # Cache is top 5%; rolling threshold within cache = quantile (1 - p/5)
    q = max(0.0, 1.0 - (p_pct / 5.0))
    t_sorted = t.sort_values(["symbol", "ts"]).copy()
    t_sorted["dt"] = pd.to_datetime(t_sorted["ts"], unit="ms", utc=True)
    window_ms = prior_days * 86400 * 1000
    for sym, sub in t_sorted.groupby("symbol", sort=False):
        ts = sub["ts"].to_numpy()
        pa = sub["pred_abs"].to_numpy()
        idx = sub.index.to_numpy()
        # For each row i, find prior window [ts[i]-window, ts[i]) and take qtile
        # Use two pointers: lo advances as ts[i] advances.
        N = len(ts)
        lo = 0
        last_snap_day = None
        for i in range(N):
            while lo < i and ts[lo] < ts[i] - window_ms:
                lo += 1
            window = pa[lo:i]
            if len(window) >= 50:    # min sample for stable quantile
                thresh = float(np.quantile(window, q))
            else:
                thresh = np.inf      # don't trade until enough history
            keep.loc[idx[i]] = pa[i] >= thresh
            day = pd.Timestamp(ts[i], unit="ms", tz="UTC").normalize()
            if day != last_snap_day:
                snapshots.append({
                    "symbol": sym, "date": day.isoformat(),
                    "p_pct": p_pct, "threshold": thresh,
                    "prior_window_n": int(len(window))})
                last_snap_day = day
    if snapshot_path:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(snapshots).to_csv(snapshot_path, index=False)
        log.info("wrote %d threshold snapshots → %s",
                 len(snapshots), snapshot_path)
    return keep.reindex(t.index, fill_value=False)


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------

def book_stats(gross: np.ndarray, costs_bps=COSTS_BPS) -> dict:
    if len(gross) == 0:
        return {"n": 0}
    out = {"n": int(len(gross)),
           "gross_mean_bps": float(gross.mean() * 1e4),
           "gross_wr_pct": float((gross > 0).mean() * 100)}
    for c_bps in costs_bps:
        c = c_bps / 1e4
        net = gross - c
        bps_net = net * 1e4
        cum = np.cumsum(bps_net)
        peak = np.maximum.accumulate(cum)
        dd = float((cum - peak).min()) if len(cum) > 0 else 0.0
        out[f"net_mean_bps_{c_bps}"] = float(bps_net.mean())
        out[f"wr_{c_bps}"] = float((bps_net > 0).mean() * 100)
        out[f"max_dd_bps_{c_bps}"] = dd
        out[f"cum_bps_{c_bps}"] = float(cum[-1])
    return out


def per_symbol_table(t: pd.DataFrame, gross: np.ndarray,
                     cost_bps: int) -> pd.DataFrame:
    rows = []
    syms = t["symbol"].to_numpy()
    c = cost_bps / 1e4
    for s in sorted(np.unique(syms)):
        m = syms == s
        g = gross[m]
        if len(g) == 0:
            continue
        net_bps = (g - c) * 1e4
        cum = np.cumsum(net_bps)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak).min() if len(cum) else 0.0
        rows.append({
            "symbol": s,
            "n": int(len(g)),
            "gross_mean_bps": float(g.mean() * 1e4),
            "net_mean_bps": float(net_bps.mean()),
            "wr_pct": float((net_bps > 0).mean() * 100),
            "max_dd_bps": float(dd),
            "cum_bps": float(cum[-1]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# TAIL-RISK REPORT
# --------------------------------------------------------------------------

def tail_risk(t: pd.DataFrame, gross: np.ndarray, cost_bps: int) -> dict:
    """Tail-risk diagnostics on the chosen geometry."""
    if len(gross) == 0:
        return {"n": 0}
    c = cost_bps / 1e4
    net = gross - c
    net_bps = net * 1e4
    out = {"n": int(len(gross)), "cost_bps": int(cost_bps)}

    # 1) Worst N% of trades
    for q_pct in (1, 5, 10):
        k = max(1, int(len(net_bps) * q_pct / 100))
        worst = np.sort(net_bps)[:k]
        out[f"worst_{q_pct}pct_mean_bps"] = float(worst.mean())
        out[f"worst_{q_pct}pct_min_bps"] = float(worst.min())
        out[f"worst_{q_pct}pct_count"] = int(k)

    # 2) MAE distribution (in vol_16 units, i.e. × the natural risk unit)
    mae = mae_per_trade(t, HOLD_BARS)               # negative log returns
    vol = t["vol_16"].to_numpy()
    valid = np.isfinite(vol) & (vol > 1e-8) & np.isfinite(mae)
    mae_units = -mae[valid] / vol[valid]            # positive: × vol units
    out["mae_units_mean"] = float(mae_units.mean())
    out["mae_units_median"] = float(np.median(mae_units))
    out["mae_units_p90"] = float(np.quantile(mae_units, 0.90))
    out["mae_units_p95"] = float(np.quantile(mae_units, 0.95))
    out["mae_units_p99"] = float(np.quantile(mae_units, 0.99))
    out["mae_units_max"] = float(mae_units.max())
    # frac of trades where MAE > 1.5x vol_16 (the old E2 stop level)
    out["frac_mae_gt_1p5_vol"] = float((mae_units > 1.5).mean())
    out["frac_mae_gt_3_vol"] = float((mae_units > 3.0).mean())

    # 3) Consecutive loser clusters (chronological per symbol then concat)
    t_ord = t.sort_values(["symbol", "ts"])
    perm = t_ord.index.to_numpy()
    pos_to_idx = {idx: i for i, idx in enumerate(t.index)}
    nb_chrono = np.array([net_bps[pos_to_idx[i]] for i in perm])
    is_loss = nb_chrono <= 0
    streaks = []
    cur = 0
    for x in is_loss:
        if x:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    streaks = np.array(streaks) if streaks else np.array([0])
    out["loser_streak_max"] = int(streaks.max())
    out["loser_streak_p95"] = float(np.quantile(streaks, 0.95))
    out["loser_streak_p99"] = float(np.quantile(streaks, 0.99))
    out["loser_streak_count_ge5"] = int((streaks >= 5).sum())
    out["loser_streak_count_ge10"] = int((streaks >= 10).sum())

    # 4) Worst day & worst week (book-wide)
    df = t.copy()
    df["net_bps"] = net_bps
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    by_day = df.groupby(df["dt"].dt.floor("D"))["net_bps"].sum()
    by_week = df.groupby(df["dt"].dt.to_period("W"))["net_bps"].sum()
    out["worst_day_bps"] = float(by_day.min())
    out["worst_day_date"] = str(by_day.idxmin().date())
    out["worst_week_bps"] = float(by_week.min())
    out["worst_week_period"] = str(by_week.idxmin())
    out["best_day_bps"] = float(by_day.max())
    out["best_week_bps"] = float(by_week.max())
    out["days_active"] = int(len(by_day))

    # 5) Per-symbol max DD
    per_sym = {}
    syms = t["symbol"].to_numpy()
    for s in sorted(np.unique(syms)):
        m = syms == s
        nb = net_bps[m]
        cum = np.cumsum(nb)
        peak = np.maximum.accumulate(cum)
        dd = float((cum - peak).min()) if len(cum) else 0.0
        per_sym[s] = {
            "n": int(m.sum()),
            "max_dd_bps": dd,
            "cum_bps": float(cum[-1]) if len(cum) else 0.0,
            "worst_trade_bps": float(nb.min()) if len(nb) else 0.0}
    out["per_symbol"] = per_sym
    return out


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------

def render_selectivity_table(L: list, raw: pd.DataFrame) -> dict:
    L.append("## 1. Selectivity sweep (per-fold top-p% by |pred|)")
    L.append("")
    L.append("Cost held at 8 bps round-trip. Geometry: no-stop, hold=120m.")
    L.append("Tradeable book = ADA + XRP + AVAX (ETH disabled, "
             "SOL probationary shown separately).")
    L.append("")
    L.append("| top-p% | n book | gross WR % | gross bps | "
             "net bps @4 | net bps @6 | net bps @8 | DD bps @8 | "
             "ADA net @8 | XRP net @8 | AVAX net @8 | SOL net @8 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out = {}
    for p in SELECTIVITY_PCTS:
        keep = select_top_p_per_fold(raw, p)
        sub = raw[keep].copy()
        sub_book = sub[sub["symbol"].isin(TRADEABLE)]
        g_book = gross_time_exit(sub_book)
        bs = book_stats(g_book)
        per = per_symbol_table(sub, gross_time_exit(sub), 8)
        per_lookup = {r["symbol"]: r for _, r in per.iterrows()}
        def cell(s):
            return f"{per_lookup[s]['net_mean_bps']:+.2f}" \
                if s in per_lookup else "—"
        L.append(f"| {p:.1f}% | {bs.get('n',0):,} | "
                 f"{bs.get('gross_wr_pct',0):.1f} | "
                 f"{bs.get('gross_mean_bps',0):+.2f} | "
                 f"{bs.get('net_mean_bps_4',0):+.2f} | "
                 f"{bs.get('net_mean_bps_6',0):+.2f} | "
                 f"{bs.get('net_mean_bps_8',0):+.2f} | "
                 f"{bs.get('max_dd_bps_8',0):+.0f} | "
                 f"{cell('ADAUSDT')} | {cell('XRPUSDT')} | "
                 f"{cell('AVAXUSDT')} | {cell('SOLUSDT')} |")
        out[f"top_{p}pct"] = {"book": bs,
                              "per_symbol": per.to_dict("records")}
    L.append("")
    return out


def render_cost_grid(L: list, raw: pd.DataFrame, p_pct: float) -> dict:
    L.append(f"## 2. Maker-only / cost-grid sweep at top {p_pct}%")
    L.append("")
    L.append("Same geometry; varying round-trip cost from 0 (full maker "
             "rebate) to 10 bps (worst-case retail taker).")
    L.append("")
    keep = select_top_p_per_fold(raw, p_pct)
    sub = raw[keep].copy()
    book = sub[sub["symbol"].isin(TRADEABLE)]
    g_book = gross_time_exit(book)
    bs = book_stats(g_book)

    L.append("| cost (bps) | mean net bps | win rate % | "
             "max DD bps | cum bps |")
    L.append("|---:|---:|---:|---:|---:|")
    for c in COSTS_BPS:
        L.append(f"| {c} | {bs[f'net_mean_bps_{c}']:+.2f} | "
                 f"{bs[f'wr_{c}']:.1f} | "
                 f"{bs[f'max_dd_bps_{c}']:+.0f} | "
                 f"{bs[f'cum_bps_{c}']:+.0f} |")
    L.append("")

    # Per-symbol at maker-realistic 4 bps
    L.append("**Per-symbol detail at 4 bps (maker-tier estimate):**")
    L.append("")
    per = per_symbol_table(sub, gross_time_exit(sub), 4)
    L.append("| symbol | n | gross bps | net bps | WR % | max DD bps | "
             "cum bps |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in per.iterrows():
        tag = " (probationary)" if r["symbol"] in PROBATIONARY else ""
        L.append(f"| {r['symbol']}{tag} | {r['n']:,} | "
                 f"{r['gross_mean_bps']:+.2f} | "
                 f"{r['net_mean_bps']:+.2f} | "
                 f"{r['wr_pct']:.1f} | "
                 f"{r['max_dd_bps']:+.0f} | "
                 f"{r['cum_bps']:+.0f} |")
    L.append("")
    return {"book": bs, "per_symbol": per.to_dict("records")}


def render_rolling_threshold(L: list, raw: pd.DataFrame, p_pct: float) -> dict:
    L.append(f"## 3. Rolling 30-day OOS threshold at top {p_pct}%")
    L.append("")
    L.append(f"Threshold recomputed at each trade as the rolling "
             f"{ROLLING_DAYS}-day quantile of prior cached |pred| values "
             f"(per symbol). Daily snapshots logged for audit.")
    L.append("")
    keep = select_rolling_threshold(raw, p_pct, ROLLING_DAYS, OUT_THRESH_CSV)
    sub = raw[keep].copy()
    book = sub[sub["symbol"].isin(TRADEABLE)]
    g_book = gross_time_exit(book)
    bs = book_stats(g_book)

    # Compare to per-fold thresholding at same p
    keep_fold = select_top_p_per_fold(raw, p_pct)
    book_fold = raw[keep_fold & raw["symbol"].isin(TRADEABLE)]
    g_fold = gross_time_exit(book_fold)
    bs_fold = book_stats(g_fold)

    L.append("| method | n book | gross bps | "
             "net bps @4 | net bps @6 | net bps @8 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| per-fold (in-sample for the test fold) | "
             f"{bs_fold['n']:,} | {bs_fold['gross_mean_bps']:+.2f} | "
             f"{bs_fold['net_mean_bps_4']:+.2f} | "
             f"{bs_fold['net_mean_bps_6']:+.2f} | "
             f"{bs_fold['net_mean_bps_8']:+.2f} |")
    L.append(f"| **rolling 30d (true OOS)** | "
             f"**{bs['n']:,}** | **{bs['gross_mean_bps']:+.2f}** | "
             f"**{bs['net_mean_bps_4']:+.2f}** | "
             f"**{bs['net_mean_bps_6']:+.2f}** | "
             f"**{bs['net_mean_bps_8']:+.2f}** |")
    L.append("")

    # Per-symbol with rolling threshold
    per = per_symbol_table(sub, gross_time_exit(sub), 6)
    L.append("**Per-symbol @ 6 bps (rolling threshold):**")
    L.append("")
    L.append("| symbol | n | gross bps | net bps | WR % | max DD bps |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in per.iterrows():
        tag = " (probationary)" if r["symbol"] in PROBATIONARY else ""
        L.append(f"| {r['symbol']}{tag} | {r['n']:,} | "
                 f"{r['gross_mean_bps']:+.2f} | "
                 f"{r['net_mean_bps']:+.2f} | "
                 f"{r['wr_pct']:.1f} | {r['max_dd_bps']:+.0f} |")
    L.append("")
    return {"rolling_book": bs, "per_fold_book": bs_fold,
            "rolling_per_symbol": per.to_dict("records")}


def render_tail_risk(L: list, raw: pd.DataFrame, p_pct: float,
                     cost_bps: int) -> dict:
    L.append(f"## 4. Tail-risk report — top {p_pct}%, cost = {cost_bps} bps")
    L.append("")
    keep = select_top_p_per_fold(raw, p_pct)
    sub = raw[keep & raw["symbol"].isin(TRADEABLE + list(PROBATIONARY))]\
        .copy().reset_index(drop=True)
    g = gross_time_exit(sub)
    tr = tail_risk(sub, g, cost_bps)

    L.append("**Worst-trade percentiles (book-wide, includes SOL):**")
    L.append("")
    L.append("| bucket | n | mean net bps | min net bps |")
    L.append("|---|---:|---:|---:|")
    for q in (1, 5, 10):
        L.append(f"| worst {q}% | {tr[f'worst_{q}pct_count']:,} | "
                 f"{tr[f'worst_{q}pct_mean_bps']:+.1f} | "
                 f"{tr[f'worst_{q}pct_min_bps']:+.1f} |")
    L.append("")

    L.append("**MAE distribution (Maximum Adverse Excursion in vol_16 "
             "units within the 120-min window):**")
    L.append("")
    L.append(f"- mean: {tr['mae_units_mean']:.2f}× vol")
    L.append(f"- median: {tr['mae_units_median']:.2f}× vol")
    L.append(f"- 90th pct: {tr['mae_units_p90']:.2f}× vol")
    L.append(f"- 95th pct: {tr['mae_units_p95']:.2f}× vol")
    L.append(f"- 99th pct: {tr['mae_units_p99']:.2f}× vol")
    L.append(f"- max: {tr['mae_units_max']:.2f}× vol")
    L.append(f"- frac > 1.5× vol (the old E2 stop level): "
             f"**{tr['frac_mae_gt_1p5_vol']*100:.1f}%**")
    L.append(f"- frac > 3× vol (catastrophic): "
             f"**{tr['frac_mae_gt_3_vol']*100:.2f}%**")
    L.append("")

    L.append("**Consecutive losing-trade clusters:**")
    L.append("")
    L.append(f"- max streak: **{tr['loser_streak_max']} trades**")
    L.append(f"- 95th pct streak length: {tr['loser_streak_p95']:.0f} trades")
    L.append(f"- 99th pct streak length: {tr['loser_streak_p99']:.0f} trades")
    L.append(f"- # streaks ≥ 5: {tr['loser_streak_count_ge5']}")
    L.append(f"- # streaks ≥ 10: {tr['loser_streak_count_ge10']}")
    L.append("")

    L.append("**Worst day / week / drawdown:**")
    L.append("")
    L.append(f"- worst day: **{tr['worst_day_bps']:+.0f} bps** "
             f"on {tr['worst_day_date']}")
    L.append(f"- best day: {tr['best_day_bps']:+.0f} bps")
    L.append(f"- worst week: **{tr['worst_week_bps']:+.0f} bps** "
             f"({tr['worst_week_period']})")
    L.append(f"- best week: {tr['best_week_bps']:+.0f} bps")
    L.append(f"- active days: {tr['days_active']:,}")
    L.append("")

    L.append("**Per-symbol cumulative & drawdown:**")
    L.append("")
    L.append("| symbol | n | cum bps | max DD bps | worst trade bps |")
    L.append("|---|---:|---:|---:|---:|")
    for s in sorted(tr["per_symbol"]):
        ps = tr["per_symbol"][s]
        tag = " (probationary)" if s in PROBATIONARY else ""
        L.append(f"| {s}{tag} | {ps['n']:,} | {ps['cum_bps']:+.0f} | "
                 f"{ps['max_dd_bps']:+.0f} | {ps['worst_trade_bps']:+.1f} |")
    L.append("")
    return tr


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main() -> None:
    raw = ensure_cache()
    if raw.empty:
        log.error("no cache built"); return
    log.info("loaded %d trades across %d symbols (top 5%% per fold)",
             len(raw), raw["symbol"].nunique())

    L = ["# V7 Path A — Finalization checks", "",
         f"_Generated: {pd.Timestamp.utcnow().isoformat(timespec='seconds')}_",
         "",
         f"**Whitelist (per user instruction):** {', '.join(WHITELIST)} "
         f"(ETH disabled).",
         f"**Tradeable book** = {', '.join(TRADEABLE)}.",
         f"**Probationary** = {', '.join(PROBATIONARY)} (shown but not in "
         f"book roll-up).",
         "",
         "Cache: top 5% per (symbol, fold) by |pred|, with full forward "
         "16×15-min OHLC paths.  Sub-selection (top 1% / 2% / etc) is done "
         "in post-processing from this cache.",
         ""]

    sweep = render_selectivity_table(L, raw)

    # Pick the best-by-net@8 selectivity for downstream sections
    best_p = max(SELECTIVITY_PCTS,
                 key=lambda p: sweep[f"top_{p}pct"]["book"].get(
                     "net_mean_bps_8", -9))
    L.append(f"**Best top-p% by net bps @ 8: top {best_p}%.** "
             f"Using this selectivity for downstream sections.")
    L.append("")

    cost = render_cost_grid(L, raw, best_p)
    rolling = render_rolling_threshold(L, raw, best_p)
    tail = render_tail_risk(L, raw, best_p, 6)

    # Final go/no-go summary
    L.append("## 5. Go / no-go summary")
    L.append("")
    book6 = cost["book"]["net_mean_bps_6"]
    book8 = cost["book"]["net_mean_bps_8"]
    rolling6 = rolling["rolling_book"]["net_mean_bps_6"]
    rolling8 = rolling["rolling_book"]["net_mean_bps_8"]
    L.append(f"- **In-sample-fold per-fold thresholding @ 6 bps:** "
             f"{book6:+.2f} bps/trade")
    L.append(f"- **True-OOS rolling 30d thresholding @ 6 bps:** "
             f"{rolling6:+.2f} bps/trade")
    L.append(f"- **Degradation from per-fold → rolling OOS:** "
             f"{rolling6 - book6:+.2f} bps")
    L.append(f"- **Worst single day (book):** "
             f"{tail['worst_day_bps']:+.0f} bps")
    L.append(f"- **Max consecutive losers:** {tail['loser_streak_max']}")
    L.append(f"- **Frac trades with MAE > 1.5×vol:** "
             f"{tail['frac_mae_gt_1p5_vol']*100:.1f}%")
    L.append("")
    if rolling6 > 0 and rolling8 > -2:
        verdict = ("✅ **GO for paper-trading** — rolling-OOS threshold "
                   "geometry survives at maker fees.")
    elif rolling6 > 0:
        verdict = ("🟡 **CONDITIONAL GO for paper-trading** — only "
                   "tradeable at maker fees (≤ 6 bps).")
    elif rolling6 > -2:
        verdict = ("🟠 **MARGINAL** — rolling OOS thinner than per-fold; "
                   "paper at smallest size only.")
    else:
        verdict = ("❌ **HOLD** — rolling OOS degradation kills the "
                   "edge. Do not paper.")
    L.append(f"**Verdict: {verdict}**")
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L))
    OUT_JSON.write_text(json.dumps({
        "selectivity_sweep": sweep,
        "cost_grid": cost,
        "rolling_threshold": rolling,
        "tail_risk": tail,
        "best_top_p_pct": best_p,
    }, default=float, indent=2))
    log.info("wrote %s", OUT_MD)
    log.info("wrote %s", OUT_JSON)
    log.info("wrote %s", OUT_THRESH_CSV)


if __name__ == "__main__":
    main()
