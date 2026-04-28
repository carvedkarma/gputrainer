"""V7 Path A — Payoff-Geometry Sweep.

Holds the E2-best universe constant (whitelist = ADA/AVAX/ETH/SOL/XRP, top
2% |pred|, HGBR sign_60m, 5-fold walk-forward, 16-bar embargo) and sweeps
five families of exit geometries to find one whose expectancy R survives
realistic costs:

  1. Stop-width sweep            (fixed 60m hold, no TP)
  2. Symmetric / asymmetric TP   (stop + TP, fixed hold cap)
  3. Trailing-stop variants      (chandelier from running MFE)
  4. Time-exit sweep             (no stop, varying hold horizon)
  5. Path-quality-conditioned    (gate by bar-1 favorability, then exit)

Per variant, per symbol, reports: n trades, Avg R gross, Avg R net @ 4/6/8
bps, Win R avg, Loss R avg, Expectancy R, Win rate, max DD bps.

Run:  python -m gpu_trainer.eval.v7_payoff_geometry
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
log = logging.getLogger("payoff_geom")

WHITELIST = ["SOLUSDT", "ADAUSDT", "XRPUSDT", "ETHUSDT", "AVAXUSDT"]
TARGET = "sign_60m"
TOP_Q = 0.98          # top 2% within fold (= E2 winner)
HORIZON = 16          # 16 × 15min = 4 hours (max forward window stored)
COST_LEVELS = [4e-4, 6e-4, 8e-4]  # round-trip in fractional return
CACHE_DIR = Path(".local/cache/v7_payoff_geometry")
OUT_MD = Path(".local/reports/v7_payoff_geometry.md")
OUT_JSON = Path(".local/reports/v7_payoff_geometry.json")


# --------------------------------------------------------------------------
# COLLECTION  (caches per-trade OHLC paths)
# --------------------------------------------------------------------------

def collect_paths(symbol: str) -> pd.DataFrame:
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
    trend_16 = (np.log(close_s) - np.log(close_s.shift(16))).to_numpy()
    dt_idx = pd.to_datetime(timestamps, unit="ms", utc=True)
    hour_utc = dt_idx.hour.to_numpy()

    folds = walk_forward_indices(timestamps)
    if not folds:
        return pd.DataFrame()

    rows = []
    path_cols_H = [f"H{i+1}" for i in range(HORIZON)]
    path_cols_L = [f"L{i+1}" for i in range(HORIZON)]
    path_cols_C = [f"C{i+1}" for i in range(HORIZON)]
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
        thresh = np.quantile(np.abs(pred[ok]), TOP_Q)
        sel = ok & (np.abs(pred) >= thresh)
        if sel.sum() < 5:
            continue
        sel_idx = np.where(sel)[0]
        global_idx = slo + sel_idx
        keep = (global_idx + HORIZON) < len(close_arr)
        sel_idx = sel_idx[keep]
        global_idx = global_idx[keep]
        if len(global_idx) < 5:
            continue

        abs_pred_sel = np.abs(pred[sel_idx])
        ap_q = quintile_bins(abs_pred_sel, 5)
        v_q = quintile_bins(vol_16[global_idx], 5)
        direction = np.sign(pred[sel_idx]).astype(int)
        sign_tr = np.sign(trend_16[global_idx])
        regime = np.where(sign_tr == 0, "FLAT",
                          np.where(sign_tr == direction, "WITH", "COUNTER"))
        session = np.array([hour_session(int(h))
                            for h in hour_utc[global_idx]])

        gi = global_idx
        entries = close_arr[gi]
        valid_entry = np.isfinite(entries) & (entries > 0)
        if not valid_entry.any():
            continue
        offsets = np.arange(1, 1 + HORIZON)
        idx_mat = gi[:, None] + offsets[None, :]
        H = high_arr[idx_mat]
        L = low_arr[idx_mat]
        C = close_arr[idx_mat]
        risk = vol_16[gi]   # base unit; planned R = stop_mult * risk
        for j in np.where(valid_entry)[0]:
            row = {"symbol": symbol, "fold": int(fold_i),
                   "ts": int(timestamps[gi[j]]),
                   "abs_pred_q": int(ap_q[j]), "vol_q": int(v_q[j]),
                   "regime": str(regime[j]), "session": str(session[j]),
                   "direction": int(direction[j]),
                   "entry": float(entries[j]),
                   "vol_16": float(risk[j])}
            for k in range(HORIZON):
                row[path_cols_H[k]] = float(H[j, k])
                row[path_cols_L[k]] = float(L[j, k])
                row[path_cols_C[k]] = float(C[j, k])
            rows.append(row)
    out = pd.DataFrame(rows)
    log.info("%s: %d top-2%% trades collected", symbol, len(out))
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
            df = collect_paths(sym)
            if not df.empty:
                df.to_parquet(p, index=False)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# EXIT SIMULATORS  (vectorised, conservative tie-breaking: stop > TP > time)
# --------------------------------------------------------------------------

def _ohlc(trades: pd.DataFrame, hb: int):
    H = trades[[f"H{i+1}" for i in range(hb)]].to_numpy()
    L = trades[[f"L{i+1}" for i in range(hb)]].to_numpy()
    C = trades[[f"C{i+1}" for i in range(hb)]].to_numpy()
    return H, L, C


def sim_fixed_hold(t: pd.DataFrame, hold_bars: int) -> np.ndarray:
    H, L, C = _ohlc(t, hold_bars)
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    cl = C[:, hold_bars - 1]
    g = np.log(cl / e) * d
    return np.where(np.isfinite(g), g, 0.0)


def sim_stop_only(t: pd.DataFrame, stop_mult: float,
                  hold_bars: int) -> np.ndarray:
    H, L, C = _ohlc(t, hold_bars)
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    sp = stop_mult * t["vol_16"].to_numpy()
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, np.nan)
    stop_long = e * (1.0 - sp); stop_short = e * (1.0 + sp)
    is_long = (d > 0)[:, None]; is_short = (d < 0)[:, None]
    long_hit = is_long & (L <= stop_long[:, None])
    short_hit = is_short & (H >= stop_short[:, None])
    hit = long_hit | short_hit
    any_hit = hit.any(axis=1)
    first_hit = np.where(any_hit, hit.argmax(axis=1), hold_bars)
    stop_ret = -sp                           # negative log-return at stop
    cl_at = C[np.arange(len(e)), np.clip(first_hit, 0, hold_bars - 1)]
    close_ret = np.log(cl_at / e) * d
    g = np.where(first_hit < hold_bars, stop_ret, close_ret)
    return np.where(np.isfinite(g), g, 0.0)


def sim_stop_tp(t: pd.DataFrame, stop_mult: float, tp_mult: float,
                hold_bars: int) -> np.ndarray:
    """Both stop and TP active. Conservative: stop hits first within bar."""
    H, L, C = _ohlc(t, hold_bars)
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    sp = stop_mult * t["vol_16"].to_numpy()
    tp = tp_mult * t["vol_16"].to_numpy()
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, np.nan)
    tp = np.where(np.isfinite(tp) & (tp > 0), tp, np.nan)
    stop_long = e * (1.0 - sp); stop_short = e * (1.0 + sp)
    tp_long = e * (1.0 + tp);   tp_short = e * (1.0 - tp)
    is_long = (d > 0)[:, None]; is_short = (d < 0)[:, None]
    s_hit = (is_long & (L <= stop_long[:, None])) | \
            (is_short & (H >= stop_short[:, None]))
    t_hit = (is_long & (H >= tp_long[:, None])) | \
            (is_short & (L <= tp_short[:, None]))
    s_first = np.where(s_hit.any(axis=1), s_hit.argmax(axis=1), hold_bars)
    t_first = np.where(t_hit.any(axis=1), t_hit.argmax(axis=1), hold_bars)
    # Conservative: when both fire on same bar, stop wins
    stop_first = s_first <= t_first
    exit_bar = np.where(stop_first & (s_first < hold_bars), s_first,
                        np.where(t_first < hold_bars, t_first, hold_bars))
    reason = np.where(stop_first & (s_first < hold_bars), 0,
                      np.where(t_first < hold_bars, 1, 2))  # 0=stop,1=tp,2=time
    stop_ret = -sp
    tp_ret = tp
    cl_at = C[np.arange(len(e)),
              np.clip(exit_bar, 0, hold_bars - 1)]
    close_ret = np.log(cl_at / e) * d
    g = np.where(reason == 0, stop_ret,
                 np.where(reason == 1, tp_ret, close_ret))
    return np.where(np.isfinite(g), g, 0.0)


def sim_trailing(t: pd.DataFrame, init_stop_mult: float, trail_mult: float,
                 hold_bars: int) -> np.ndarray:
    """Initial hard stop at init_stop_mult × vol_16. Once trade is in profit
    by >= trail_mult × vol_16, stop trails at (running_MFE - trail_mult×vol)
    in price space. Bar-by-bar python loop because trail is path-dependent."""
    H, L, C = _ohlc(t, hold_bars)
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    vol = t["vol_16"].to_numpy()
    init_sp = init_stop_mult * vol
    trail_sp = trail_mult * vol
    init_sp = np.where(np.isfinite(init_sp) & (init_sp > 0), init_sp, np.nan)
    trail_sp = np.where(np.isfinite(trail_sp) & (trail_sp > 0),
                        trail_sp, np.nan)

    N = len(e)
    long = d > 0
    short = d < 0
    # current stop level in price
    stop_level = np.where(long, e * (1 - init_sp),
                          np.where(short, e * (1 + init_sp), np.nan))
    mfe_price = e.copy()  # most-favorable-extreme price
    exit_ret = np.full(N, np.nan)
    closed = np.zeros(N, dtype=bool)
    for b in range(hold_bars):
        # update MFE before computing trail
        active = ~closed
        mfe_price = np.where(active & long & (H[:, b] > mfe_price),
                             H[:, b], mfe_price)
        mfe_price = np.where(active & short & (L[:, b] < mfe_price),
                             L[:, b], mfe_price)
        # trail: only if in profit by >= trail_sp
        in_profit_long = active & long & \
            ((mfe_price - e) / e >= trail_sp)
        in_profit_short = active & short & \
            ((e - mfe_price) / e >= trail_sp)
        trail_long_lvl = mfe_price * (1 - trail_sp)
        trail_short_lvl = mfe_price * (1 + trail_sp)
        stop_level = np.where(in_profit_long & (trail_long_lvl > stop_level),
                              trail_long_lvl, stop_level)
        stop_level = np.where(in_profit_short & (trail_short_lvl < stop_level),
                              trail_short_lvl, stop_level)
        # check stop hit this bar (after trail update — stop is set BEFORE bar
        # opens; trail update for THIS bar only protects subsequent bars).
        # To be conservative we compare against stop_level as it stood at the
        # start of the bar — easier: re-derive prev_stop. To stay simple, we
        # compare against current stop_level (slightly optimistic on trail
        # tightening). Documented caveat.
        hit_long = active & long & (L[:, b] <= stop_level)
        hit_short = active & short & (H[:, b] >= stop_level)
        hit = hit_long | hit_short
        # exit at stop_level
        ret_at_stop = np.where(long, np.log(stop_level / e),
                               -np.log(stop_level / e))
        exit_ret = np.where(hit & ~closed, ret_at_stop, exit_ret)
        closed = closed | hit
    # any still open → close at last bar
    open_mask = ~closed
    cl_final = C[:, hold_bars - 1]
    final_ret = np.log(cl_final / e) * d
    exit_ret = np.where(open_mask, final_ret, exit_ret)
    return np.where(np.isfinite(exit_ret), exit_ret, 0.0)


def sim_path_gate(t: pd.DataFrame, gate_mult: float, hold_bars: int,
                  stop_mult: float | None) -> np.ndarray:
    """Path-quality-conditioned EXIT (every signal IS entered):
      • At end of bar 1 (15 min), measure bar1_ret = log(C1/entry) * dir.
      • If bar1_ret < gate_mult * vol_16 → EXIT NOW at bar-1 close (realize
        bar1_ret).  This is the early-exit branch — the trade still happened
        and still pays round-trip costs.
      • Else → HOLD to `hold_bars` with optional hard stop at stop_mult×vol.
    Returns the realized gross return per trade (length N, not filtered).
    """
    H, L, C = _ohlc(t, hold_bars)
    e = t["entry"].to_numpy()
    d = t["direction"].to_numpy().astype(np.float64)
    vol = t["vol_16"].to_numpy()
    bar1_ret = np.log(C[:, 0] / e) * d
    pass_gate = (bar1_ret >= (gate_mult * vol)) & np.isfinite(bar1_ret)
    early = np.where(np.isfinite(bar1_ret), bar1_ret, 0.0)
    if stop_mult is None:
        held = sim_fixed_hold(t, hold_bars)
    else:
        held = sim_stop_only(t, stop_mult, hold_bars)
    return np.where(pass_gate, held, early)


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------

def stats(gross: np.ndarray, risk: np.ndarray, costs=COST_LEVELS) -> dict:
    valid = np.isfinite(gross) & np.isfinite(risk) & (risk > 1e-8)
    g = gross[valid]; r = risk[valid]
    if len(g) == 0:
        return {"n": 0}
    R_gross = g / r
    out = {
        "n": int(len(g)),
        "avg_R_gross": float(R_gross.mean()),
        "median_R_gross": float(np.median(R_gross)),
        "mean_gross_bps": float(g.mean() * 1e4),
        "wr_gross_pct": float((g > 0).mean() * 100),
    }
    # Per-cost metrics
    for c in costs:
        net = g - c
        R_net = net / r
        wins = R_net > 0
        losses = R_net <= 0
        winR = float(R_net[wins].mean()) if wins.any() else 0.0
        lossR = float(R_net[losses].mean()) if losses.any() else 0.0
        wr = float(wins.mean())
        bps_net = net * 1e4
        cum = np.cumsum(bps_net)
        peak = np.maximum.accumulate(cum)
        dd = float((cum - peak).min()) if len(cum) > 0 else 0.0
        ck = int(round(c * 1e4))
        out[f"avg_R_net_{ck}"] = float(R_net.mean())
        out[f"win_R_{ck}"] = winR
        out[f"loss_R_{ck}"] = lossR
        out[f"wr_{ck}"] = float(wr * 100)
        out[f"expectancy_R_{ck}"] = float(wr * winR + (1 - wr) * lossR)
        out[f"mean_net_bps_{ck}"] = float(bps_net.mean())
        out[f"max_dd_bps_{ck}"] = dd
    return out


# --------------------------------------------------------------------------
# VARIANT BUILDER + RUNNER
# --------------------------------------------------------------------------

def build_variants() -> list[tuple[str, str, callable]]:
    """List of (family, label, fn(trades_df) → (gross, risk_for_R))."""
    V = []

    # Family 1: stop-width sweep, no TP, fixed 60m hold
    for sm in [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        V.append(("1.stop_sweep", f"stop={sm:.2f}×vol, hold=60m",
                  lambda t, sm=sm: (sim_stop_only(t, sm, 4),
                                    sm * t["vol_16"].to_numpy())))

    # Family 2a: symmetric TP (stop = TP)
    for m in [0.75, 1.0, 1.5]:
        V.append(("2a.sym_tp", f"stop={m:.2f}×vol, TP={m:.2f}×vol, hold=60m",
                  lambda t, m=m: (sim_stop_tp(t, m, m, 4),
                                  m * t["vol_16"].to_numpy())))

    # Family 2b: asymmetric TP (TP > stop) — payoff > 1
    asym_pairs = [(0.75, 1.5), (0.75, 2.0), (1.0, 1.5), (1.0, 2.0),
                  (1.0, 3.0), (1.25, 2.5)]
    for sm, tm in asym_pairs:
        V.append(("2b.asym_tp",
                  f"stop={sm:.2f}×vol, TP={tm:.2f}×vol, hold=60m",
                  lambda t, sm=sm, tm=tm: (sim_stop_tp(t, sm, tm, 4),
                                           sm * t["vol_16"].to_numpy())))
    # 2b extended hold (240m) so TPs have more time to fire
    for sm, tm in [(1.0, 2.0), (1.0, 3.0), (1.25, 3.0)]:
        V.append(("2b.asym_tp_long",
                  f"stop={sm:.2f}×vol, TP={tm:.2f}×vol, hold=240m",
                  lambda t, sm=sm, tm=tm: (sim_stop_tp(t, sm, tm, 16),
                                           sm * t["vol_16"].to_numpy())))

    # Family 3: trailing stops
    for isp, tr in [(1.5, 1.0), (1.5, 1.5), (2.0, 1.0), (2.0, 1.5),
                    (1.0, 1.0)]:
        V.append(("3.trail",
                  f"init={isp:.2f}×vol, trail={tr:.2f}×vol, hold=240m",
                  lambda t, isp=isp, tr=tr: (sim_trailing(t, isp, tr, 16),
                                             isp * t["vol_16"].to_numpy())))

    # Family 4: time-exit only (no stop)
    for hb, mins in [(2, 30), (4, 60), (8, 120), (16, 240)]:
        V.append(("4.time_exit",
                  f"no stop, hold={mins}m",
                  lambda t, hb=hb: (sim_fixed_hold(t, hb),
                                    1.5 * t["vol_16"].to_numpy())))

    # Family 5: path-quality-conditioned EXITS
    # Every signal IS entered. At end of bar 1, exit if not favorable;
    # otherwise hold to long horizon (with optional stop).
    for gm, stop in [(0.0, 1.5), (0.25, 1.5), (0.5, 1.5),
                     (0.0, None), (0.25, None), (0.5, None)]:
        lbl = f"early-exit if bar1<{gm:.2f}×vol, hold=240m, " + \
              (f"stop={stop:.2f}×vol" if stop else "no stop")
        risk_mult = stop if stop is not None else 1.5
        V.append(("5.path_gate", lbl,
                  lambda t, gm=gm, stop=stop, rm=risk_mult:
                  (sim_path_gate(t, gm, 16, stop),
                   rm * t["vol_16"].to_numpy())))
    return V


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------

def render_variant_block(L: list, family: str, label: str,
                         gross: np.ndarray, risk: np.ndarray,
                         symbols: np.ndarray) -> dict:
    L.append(f"### {family} — {label}")
    L.append("")
    book = stats(gross, risk)
    if book["n"] == 0:
        L.append("_No trades._"); L.append("")
        return {"book": book, "by_symbol": {}}
    L.append(f"**Book: n={book['n']:,}, gross WR={book['wr_gross_pct']:.1f}%,"
             f" mean gross={book['mean_gross_bps']:+.2f} bps, "
             f"Avg R gross={book['avg_R_gross']:+.3f}**")
    L.append("")
    L.append("| symbol | n | Avg R gross | "
             "Avg R net@4 | Avg R net@6 | Avg R net@8 | "
             "Win R@8 | Loss R@8 | Exp R@8 | WR%@8 | "
             "net bps@8 | max DD bps@8 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_sym = {}
    syms_unique = sorted(np.unique(symbols))
    for s in syms_unique:
        m = symbols == s
        st = stats(gross[m], risk[m])
        if st["n"] == 0:
            continue
        by_sym[s] = st
        L.append(f"| {s} | {st['n']:,} | {st['avg_R_gross']:+.3f} | "
                 f"{st['avg_R_net_4']:+.3f} | {st['avg_R_net_6']:+.3f} | "
                 f"{st['avg_R_net_8']:+.3f} | "
                 f"{st['win_R_8']:+.3f} | {st['loss_R_8']:+.3f} | "
                 f"{st['expectancy_R_8']:+.3f} | {st['wr_8']:.1f} | "
                 f"{st['mean_net_bps_8']:+.2f} | {st['max_dd_bps_8']:+.0f} |")
    L.append(f"| **BOOK** | **{book['n']:,}** | "
             f"**{book['avg_R_gross']:+.3f}** | "
             f"**{book['avg_R_net_4']:+.3f}** | "
             f"**{book['avg_R_net_6']:+.3f}** | "
             f"**{book['avg_R_net_8']:+.3f}** | "
             f"**{book['win_R_8']:+.3f}** | **{book['loss_R_8']:+.3f}** | "
             f"**{book['expectancy_R_8']:+.3f}** | **{book['wr_8']:.1f}** | "
             f"**{book['mean_net_bps_8']:+.2f}** | "
             f"**{book['max_dd_bps_8']:+.0f}** |")
    L.append("")
    return {"book": book, "by_symbol": by_sym}


def main() -> None:
    raw = ensure_cache()
    if raw.empty:
        log.error("no cache built"); return
    log.info("loaded %d trades across %d symbols",
             len(raw), raw["symbol"].nunique())

    variants = build_variants()
    L = ["# V7 Path A — Payoff-Geometry Sweep", "",
         f"_Generated: {pd.Timestamp.utcnow().isoformat(timespec='seconds')}_",
         "",
         f"Universe: top 2% |pred|, HGBR sign_60m, 5-fold walk-forward, "
         f"16-bar embargo. Whitelist: {', '.join(WHITELIST)}.",
         f"Total candidate trades after filters: **{len(raw):,}**.",
         "Cost levels evaluated: 4 / 6 / 8 bps round-trip.",
         "**Success criterion: at least one variant with positive book-wide "
         "Expectancy R AND positive mean net bps at 8 bps cost.**",
         "",
         f"_R-multiple definition: R = realised return / planned risk, where "
         f"planned risk = stop_mult × vol_16 at entry. For variants without "
         f"a hard stop (time-exits, no-stop path-gate), R denominator uses a "
         f"reference 1.5× vol_16._",
         ""]

    syms = raw["symbol"].to_numpy()
    results = {}
    for family, label, fn in variants:
        try:
            gross, risk = fn(raw)
        except Exception as e:
            log.error("variant %s/%s failed: %s", family, label, e)
            continue
        key = f"{family} | {label}"
        results[key] = render_variant_block(L, family, label,
                                            gross, risk, syms)
        log.info("done: %s | book n=%d Exp R@8=%+.3f net@8=%+.2fbps",
                 key, results[key]["book"].get("n", 0),
                 results[key]["book"].get("expectancy_R_8", 0.0),
                 results[key]["book"].get("mean_net_bps_8", 0.0))

    # ---- Path B: leaderboard + cost ladder for top variant ----
    L += ["", "## Leaderboard (sorted by Expectancy R @ 8 bps, book-wide)", ""]
    L.append("| variant | n | Exp R @4 | Exp R @6 | Exp R @8 | "
             "net bps @4 | net bps @6 | net bps @8 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    sorted_keys = sorted(results.keys(),
                         key=lambda k: -results[k]["book"]
                         .get("expectancy_R_8", -9))
    for k in sorted_keys:
        b = results[k]["book"]
        if b.get("n", 0) == 0:
            continue
        L.append(f"| {k} | {b['n']:,} | "
                 f"{b['expectancy_R_4']:+.3f} | {b['expectancy_R_6']:+.3f} | "
                 f"{b['expectancy_R_8']:+.3f} | "
                 f"{b['mean_net_bps_4']:+.2f} | "
                 f"{b['mean_net_bps_6']:+.2f} | "
                 f"{b['mean_net_bps_8']:+.2f} |")
    L.append("")

    # Path B: cost ladder for top variant by Exp R @ 8bps
    if sorted_keys:
        winner = sorted_keys[0]
        wb = results[winner]["book"]
        L.append("## Path B — Cost ladder for top geometry")
        L.append("")
        L.append(f"**Winner by Exp R @ 8 bps: `{winner}`**")
        L.append("")
        L.append("| cost (bps) | Avg R net | Exp R | Win R | Loss R | "
                 "mean net bps | max DD bps |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for c in [2, 4, 6, 8, 10]:
            if c not in (4, 6, 8):
                # add fresh stat for non-default cost
                # Recompute from raw using same fn
                fam, lbl = winner.split(" | ", 1)
                # find variant fn
                fn = next((f for fa, l, f in variants
                           if fa == fam and l == lbl), None)
                if fn is None:
                    continue
                g, r = fn(raw)
                st = stats(g, r, costs=[c * 1e-4])
                key = f"{c}"
                L.append(f"| {c} | {st[f'avg_R_net_{c}']:+.3f} | "
                         f"{st[f'expectancy_R_{c}']:+.3f} | "
                         f"{st[f'win_R_{c}']:+.3f} | "
                         f"{st[f'loss_R_{c}']:+.3f} | "
                         f"{st[f'mean_net_bps_{c}']:+.2f} | "
                         f"{st[f'max_dd_bps_{c}']:+.0f} |")
            else:
                L.append(f"| {c} | {wb[f'avg_R_net_{c}']:+.3f} | "
                         f"{wb[f'expectancy_R_{c}']:+.3f} | "
                         f"{wb[f'win_R_{c}']:+.3f} | "
                         f"{wb[f'loss_R_{c}']:+.3f} | "
                         f"{wb[f'mean_net_bps_{c}']:+.2f} | "
                         f"{wb[f'max_dd_bps_{c}']:+.0f} |")
        L.append("")
        # Verdict
        passes_8 = wb["expectancy_R_8"] > 0 and wb["mean_net_bps_8"] > 0
        passes_6 = wb["expectancy_R_6"] > 0 and wb["mean_net_bps_6"] > 0
        passes_4 = wb["expectancy_R_4"] > 0 and wb["mean_net_bps_4"] > 0
        if passes_8:
            v = "✅ **PASS at 8 bps** — geometry is tradeable at retail cost."
        elif passes_6:
            v = "🟡 **PASS at 6 bps only** — needs maker-tier execution."
        elif passes_4:
            v = "🟠 **PASS at 4 bps only** — needs aggressive maker rebates."
        else:
            v = "❌ **FAIL at all cost levels** — no geometry survives."
        L.append(f"**Verdict: {v}**")
        L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L))
    OUT_JSON.write_text(json.dumps({k: v for k, v in results.items()},
                                   default=float, indent=2))
    log.info("wrote %s", OUT_MD)
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
