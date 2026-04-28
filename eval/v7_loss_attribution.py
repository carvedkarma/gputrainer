"""V7 Loss Attribution — why does a 55%-WR system lose money?

Re-runs the best cell of the V7 audit (BTCUSDT / sign_60m / HGBR) but, instead of
collapsing each fold to a single mean, captures every top-decile trade with its
diagnostic context, then slices losses across:

  * predicted-confidence bucket (top 1% / 2.5% / 5% / 10% of |pred|)
  * realised-vol regime at entry (quintiles of trailing 16-bar return std)
  * trend regime at entry (sign of trailing 16-bar return vs sign of prediction:
    "with-trend" vs "counter-trend")
  * hour-of-day (UTC; Asia / EU / US sessions)
  * day-of-week (Mon..Sun)
  * win-vs-loss size distribution

Goal: find loss concentrations that are either avoidable (filter out by regime)
or convertible (e.g. counter-trend trades that need a tighter exit). Writes
`.local/reports/v7_loss_attribution_btc.md`.

Run:  python -m gpu_trainer.eval.v7_loss_attribution
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer.eval.v7_signal_audit_augmented import (
    EMBARGO_BARS,
    build_features,
    build_targets,
    load_symbol,
    walk_forward_indices,
)
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loss_attr")

SYMBOL = "BTCUSDT"
TARGET = "sign_60m"
COST_BPS = 8.0
COST_FRAC = COST_BPS * 1e-4
TOP_DECILE_Q = 0.90
OUT_MD = Path(".local/reports/v7_loss_attribution_btc.md")
OUT_JSON = Path(".local/reports/v7_loss_attribution_btc.json")


def _hgbr():
    return HistGradientBoostingRegressor(
        max_iter=120, max_depth=4, learning_rate=0.05,
        min_samples_leaf=300, max_bins=128, random_state=0,
        early_stopping=False)


def collect_trades() -> pd.DataFrame:
    """Walk-forward over BTC, return a long DataFrame of every top-decile trade."""
    log.info("loading %s ...", SYMBOL)
    df = load_symbol(SYMBOL)
    if df.empty or len(df) < 20000:
        raise RuntimeError(f"not enough bars for {SYMBOL}: {len(df)}")
    feats = build_features(df)
    targs = build_targets(df)

    y_full = targs[TARGET].to_numpy()
    fwd_ret_full = targs["ret_60m"].to_numpy()

    feat_warmup = feats.notna().sum(axis=1)
    valid_from = int((feat_warmup > 8).idxmax())
    valid_from = max(valid_from, 200)

    feats_arr = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y = y_full[valid_from:]
    fwd = fwd_ret_full[valid_from:]
    timestamps = df["timestamp"].to_numpy()[valid_from:]  # ms epoch

    # Regime context aligned to the same valid_from-trimmed slice
    close = df["close"].astype("float64").iloc[valid_from:].reset_index(drop=True)
    logret = np.log(close).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(close) - np.log(close.shift(16))).to_numpy()
    trend_96 = (np.log(close) - np.log(close.shift(96))).to_numpy()
    dt_idx = pd.to_datetime(timestamps, unit="ms", utc=True)
    hour_utc = dt_idx.hour.to_numpy()
    dow = dt_idx.dayofweek.to_numpy()

    folds = walk_forward_indices(timestamps)
    log.info("got %d folds, valid_from=%d, total bars=%d",
             len(folds), valid_from, len(timestamps))

    rows = []
    for fold_i, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr = feats_arr[tlo:thi]; y_tr = y[tlo:thi]
        X_te = feats_arr[slo:shi]; y_te = y[slo:shi]
        fwd_te = fwd[slo:shi]

        m_tr = np.isfinite(y_tr)
        m_te = np.isfinite(fwd_te)
        if m_tr.sum() < 500 or m_te.sum() < 200:
            log.warning("fold %d skipped: insufficient rows", fold_i)
            continue
        X_tr2, y_tr2 = X_tr[m_tr], y_tr[m_tr]
        X_te2 = X_te
        fwd_te2 = fwd_te
        # We keep all test rows (don't drop NaN target in test) so we can score
        # every prediction; only NaN forward returns disqualify a trade later.

        m = _hgbr()
        m.fit(X_tr2, y_tr2)
        pred = m.predict(X_te2)

        ok = np.isfinite(pred) & np.isfinite(fwd_te2)
        if ok.sum() < 100:
            continue
        thresh = np.quantile(np.abs(pred[ok]), TOP_DECILE_Q)
        sel_local = ok & (np.abs(pred) >= thresh)
        if sel_local.sum() < 10:
            continue

        direction = np.sign(pred[sel_local])
        gross = direction * fwd_te2[sel_local]
        net = gross - COST_FRAC
        global_idx = np.arange(slo, shi)[sel_local]

        for i, gi in enumerate(global_idx):
            rows.append({
                "fold": int(fold_i),
                "ts": int(timestamps[gi]),
                "pred": float(pred[sel_local][i]),
                "abs_pred": float(abs(pred[sel_local][i])),
                "direction": int(direction[i]),
                "fwd_ret_60m": float(fwd_te2[sel_local][i]),
                "gross": float(gross[i]),
                "net": float(net[i]),
                "is_win": int(gross[i] > 0),
                "is_win_net": int(net[i] > 0),
                "vol_16": float(vol_16[gi]) if np.isfinite(vol_16[gi]) else np.nan,
                "trend_16": float(trend_16[gi]) if np.isfinite(trend_16[gi]) else np.nan,
                "trend_96": float(trend_96[gi]) if np.isfinite(trend_96[gi]) else np.nan,
                "hour_utc": int(hour_utc[gi]),
                "dow": int(dow[gi]),
            })
        log.info("fold %d: %d trades selected (gross WR=%.2f%%, mean net=%.2f bps)",
                 fold_i, int(sel_local.sum()),
                 float((gross > 0).mean() * 100),
                 float(net.mean() * 1e4))

    df_trades = pd.DataFrame(rows)
    log.info("collected %d trades total", len(df_trades))
    return df_trades


# ---------- slicing helpers ----------

def slice_summary(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    g = df.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "wr_gross": g["is_win"].mean() * 100,
        "wr_net": g["is_win_net"].mean() * 100,
        "mean_gross_bps": g["gross"].mean() * 1e4,
        "mean_net_bps": g["net"].mean() * 1e4,
        "median_gross_bps": g["gross"].median() * 1e4,
        "sum_net_bps": g["net"].sum() * 1e4,
    })
    out.index.name = label
    return out


def quintile_label(s: pd.Series, n: int = 5, prefix: str = "Q") -> pd.Series:
    """Quintile-rank a series, ignoring NaN."""
    ranks = s.rank(pct=True, method="average")
    bins = np.where(np.isnan(ranks), -1,
                    np.clip((ranks * n).astype(int), 0, n - 1))
    out = pd.Series([f"{prefix}{b+1}" if b >= 0 else "NA" for b in bins],
                    index=s.index)
    return out


def hour_session(h: int) -> str:
    # UTC: Asia 00-07, EU 07-14, US 14-21, OFF 21-24
    if 0 <= h < 7:
        return "Asia (00-07 UTC)"
    if 7 <= h < 14:
        return "EU (07-14 UTC)"
    if 14 <= h < 21:
        return "US (14-21 UTC)"
    return "Late (21-24 UTC)"


def dow_label(d: int) -> str:
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]


# ---------- report ----------

def write_report(trades: pd.DataFrame) -> dict:
    if trades.empty:
        raise RuntimeError("no trades collected")

    n = len(trades)
    gross_wr = trades["is_win"].mean() * 100
    net_wr = trades["is_win_net"].mean() * 100
    mean_gross = trades["gross"].mean() * 1e4
    mean_net = trades["net"].mean() * 1e4
    avg_win = trades.loc[trades["gross"] > 0, "gross"].mean() * 1e4
    avg_loss = -trades.loc[trades["gross"] <= 0, "gross"].mean() * 1e4
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

    # Slices
    trades = trades.copy()
    trades["abs_pred_q"] = quintile_label(trades["abs_pred"], 5, "P")
    trades["vol_q"] = quintile_label(trades["vol_16"], 5, "V")
    trades["trend_q"] = quintile_label(trades["trend_16"], 5, "T")

    # With-trend = sign of prior 16-bar return matches direction
    sign_trend = np.sign(trades["trend_16"])
    trades["regime_trend"] = np.where(
        sign_trend == 0, "FLAT",
        np.where(sign_trend == trades["direction"], "WITH-trend", "COUNTER-trend"))
    trades["session"] = trades["hour_utc"].map(hour_session)
    trades["dow_name"] = trades["dow"].map(dow_label)

    # Top decile of |pred| split by quintile of |pred| within that top decile
    s_conf = slice_summary(trades, "abs_pred_q", "abs_pred quintile")
    s_vol = slice_summary(trades, "vol_q", "trailing-vol quintile")
    s_trend = slice_summary(trades, "trend_q", "trailing-trend quintile")
    s_regime = slice_summary(trades, "regime_trend", "trend regime")
    s_session = slice_summary(trades, "session", "session")
    s_dow = slice_summary(trades, "dow_name", "day-of-week")

    # If we kept only profitable buckets, what would PnL be?
    def cell_keep_positive(slc: pd.DataFrame) -> dict:
        winners = slc[slc["mean_net_bps"] > 0]
        kept = trades[trades.set_index(slc.index.name).index.isin(winners.index)] \
            if False else None
        return {"buckets_positive": int(len(winners)),
                "buckets_total": int(len(slc)),
                "trades_kept": int(winners["n"].sum()),
                "trades_dropped": int(slc["n"].sum() - winners["n"].sum())}

    keep_summary = {
        "by_session": cell_keep_positive(s_session),
        "by_vol": cell_keep_positive(s_vol),
        "by_regime": cell_keep_positive(s_regime),
        "by_conf": cell_keep_positive(s_conf),
    }

    # Aggressive filter: keep ONLY trades where (regime, session, conf, vol)
    # quadruple's mean_net is positive across the training (we approximate by
    # using the same trades; this overstates, just shows ceiling)
    grouped = trades.groupby(
        ["abs_pred_q", "vol_q", "regime_trend", "session"])
    cell_means = grouped["net"].mean() * 1e4
    cell_n = grouped.size()
    pos_cells = cell_means[cell_means > 0]
    keep_mask = grouped.ngroup().isin(
        cell_means[cell_means > 0].index.map(
            lambda idx: grouped.indices[idx]).values
    ) if False else None
    # Simpler: build per-trade mask via groupby filter
    cell_lookup = (cell_means.to_dict())

    def lookup_net(row):
        return cell_lookup.get(
            (row["abs_pred_q"], row["vol_q"], row["regime_trend"], row["session"]),
            np.nan)
    trades["cell_mean_net_bps"] = trades.apply(lookup_net, axis=1)
    keep = trades[trades["cell_mean_net_bps"] > 0]
    drop = trades[trades["cell_mean_net_bps"] <= 0]
    ceiling = {
        "if_kept_only_positive_quadruples": {
            "n_kept": int(len(keep)),
            "n_dropped": int(len(drop)),
            "wr_gross_kept": float(keep["is_win"].mean() * 100) if len(keep) else 0.0,
            "wr_net_kept": float(keep["is_win_net"].mean() * 100) if len(keep) else 0.0,
            "mean_net_bps_kept": float(keep["net"].mean() * 1e4) if len(keep) else 0.0,
            "sum_net_bps_kept": float(keep["net"].sum() * 1e4) if len(keep) else 0.0,
            "sum_net_bps_dropped_avoided_loss": -float(drop["net"].sum() * 1e4) if len(drop) else 0.0,
        }
    }

    L = []
    L.append("# V7 Loss Attribution — BTCUSDT / sign_60m / HGBR")
    L.append("")
    L.append(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    L.append("")
    L.append("## What this is")
    L.append("")
    L.append("Re-ran the best cell of the V7 audit, but kept every top-decile trade "
             "instead of collapsing per fold. Then sliced the trade set to find "
             "**where the losses concentrate** — so we know what to filter "
             "(avoid losses) or what to redesign (convert losses).")
    L.append("")
    L.append("## Headline numbers")
    L.append("")
    L.append(f"- Total top-decile trades: **{n:,}**")
    L.append(f"- Gross win rate (before cost): **{gross_wr:.2f}%**")
    L.append(f"- Net win rate (after {COST_BPS} bps): **{net_wr:.2f}%**")
    L.append(f"- Average winner: **+{avg_win:.2f} bps**")
    L.append(f"- Average loser:  **−{avg_loss:.2f} bps**")
    L.append(f"- Win/Loss size ratio: **{win_loss_ratio:.3f}** "
             f"(needs to be > {(100 - gross_wr)/gross_wr:.3f} just to break even gross)")
    L.append(f"- Mean **gross** per trade: **{mean_gross:+.2f} bps**")
    L.append(f"- Mean **net** per trade:   **{mean_net:+.2f} bps**")
    L.append("")
    L.append("**Diagnosis from headline:** WR is real but the average loser is "
             f"~{avg_loss/avg_win:.2f}x the average winner. To turn this into a "
             "money-maker we either need (a) bigger winners (let them run / "
             "scale by confidence), (b) smaller losers (tighter stops / earlier "
             "exits on reversals), or (c) trade fewer but better cells.")
    L.append("")

    def render_table(slc: pd.DataFrame, title: str) -> None:
        L.append(f"### {title}")
        L.append("")
        L.append(f"| {slc.index.name} | n | gross WR% | net WR% | "
                 "mean gross bps | mean net bps | sum net bps |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        slc_sorted = slc.sort_values("mean_net_bps", ascending=False)
        for ix, r in slc_sorted.iterrows():
            L.append(f"| {ix} | {int(r['n']):,} | "
                     f"{r['wr_gross']:.2f} | {r['wr_net']:.2f} | "
                     f"{r['mean_gross_bps']:+.2f} | "
                     f"{r['mean_net_bps']:+.2f} | "
                     f"{r['sum_net_bps']:+.0f} |")
        L.append("")

    L.append("## Loss slicing — where the bleed is")
    L.append("")
    render_table(s_conf,    "By |prediction| quintile (P5 = strongest convictions)")
    render_table(s_vol,     "By trailing 4-h realised vol quintile (V5 = highest vol)")
    render_table(s_trend,   "By trailing 4-h return quintile (T5 = strongest up-trend)")
    render_table(s_regime,  "By trend regime (with-trend vs counter-trend)")
    render_table(s_session, "By session (UTC hour grouping)")
    render_table(s_dow,     "By day-of-week")

    L.append("## Ceiling estimate — if we filtered out the loss-making quadruples")
    L.append("")
    c = ceiling["if_kept_only_positive_quadruples"]
    L.append("Group every trade by `(|pred|-quintile, vol-quintile, trend-regime, "
             "session)` (5 × 5 × 3 × 4 = 300 cells). Keep only cells whose "
             "in-sample mean net is > 0. **Note: this is in-sample so it "
             "overstates — it's a *ceiling*, not a forecast.** A real "
             "implementation must learn the per-cell filter on prior folds and "
             "apply it forward.")
    L.append("")
    L.append(f"- Trades kept: **{c['n_kept']:,}** of {n:,} ({c['n_kept']/n*100:.1f}%)")
    L.append(f"- Trades dropped: **{c['n_dropped']:,}** ({c['n_dropped']/n*100:.1f}%)")
    L.append(f"- Mean net per trade on kept set: **{c['mean_net_bps_kept']:+.2f} bps**")
    L.append(f"- Total net (kept set):  **{c['sum_net_bps_kept']:+.0f} bps**")
    L.append(f"- Total loss avoided (dropped set): "
             f"**{c['sum_net_bps_dropped_avoided_loss']:+.0f} bps**")
    L.append("")

    L.append("## What the slices say (interpretation)")
    L.append("")
    # Identify top / bottom buckets for each slice
    def top_bot(slc: pd.DataFrame, what: str) -> str:
        srt = slc.sort_values("mean_net_bps", ascending=False)
        best = srt.iloc[0]
        worst = srt.iloc[-1]
        return (f"- **{what}**: best bucket `{srt.index[0]}` "
                f"({best['mean_net_bps']:+.2f} bps net, "
                f"WR {best['wr_gross']:.1f}%, n={int(best['n']):,}); "
                f"worst bucket `{srt.index[-1]}` "
                f"({worst['mean_net_bps']:+.2f} bps net, "
                f"WR {worst['wr_gross']:.1f}%, n={int(worst['n']):,}).")
    L.append(top_bot(s_conf,    "Confidence"))
    L.append(top_bot(s_vol,     "Volatility regime"))
    L.append(top_bot(s_trend,   "Trend strength"))
    L.append(top_bot(s_regime,  "Trend alignment"))
    L.append(top_bot(s_session, "Session"))
    L.append(top_bot(s_dow,     "Day-of-week"))
    L.append("")

    L.append("## Concrete next steps")
    L.append("")
    L.append("1. **Build an ex-ante filter** on the buckets that lose money in "
             "the *prior* fold (so it's strictly out-of-sample). The ceiling "
             "above shows how much room there is if such a filter were "
             "perfect — actual delivery will be a fraction of it.")
    L.append("2. **Replace fixed 60-min hold** with an adaptive exit: time-stop "
             "when adverse excursion exceeds a multiple of vol_16, target "
             "exit when favourable excursion reaches a multiple of vol_16. "
             "Asymmetric stops directly attack the avg_loss > avg_win problem.")
    L.append("3. **Cost-tier the trades**: for low-confidence cells, only enter "
             "if maker fills are achievable; pay taker only on the highest-"
             "quintile predictions. The sweep at 2/4/6 bps is the right tool "
             "(Task #105).")
    L.append("4. **Retire low-edge slices**: any quintile/session that "
             "permanently shows negative IC across folds should never trade.")
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT_MD)

    payload = {
        "symbol": SYMBOL,
        "target": TARGET,
        "cost_bps": COST_BPS,
        "n_trades": n,
        "gross_wr": gross_wr,
        "net_wr": net_wr,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "mean_gross_bps": mean_gross,
        "mean_net_bps": mean_net,
        "slices": {
            "by_confidence": s_conf.reset_index().to_dict(orient="records"),
            "by_vol": s_vol.reset_index().to_dict(orient="records"),
            "by_trend": s_trend.reset_index().to_dict(orient="records"),
            "by_regime": s_regime.reset_index().to_dict(orient="records"),
            "by_session": s_session.reset_index().to_dict(orient="records"),
            "by_dow": s_dow.reset_index().to_dict(orient="records"),
        },
        "keep_summary": keep_summary,
        "ceiling": ceiling,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", OUT_JSON)
    return payload


def main():
    trades = collect_trades()
    write_report(trades)


if __name__ == "__main__":
    main()
