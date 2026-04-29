from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MythosConfig
from .experts import build_experts
from .features import build_feature_frame
from .promotion import evaluate_promotion
from .risk import RiskConstitution
from .router import MetaRouter
from .world_model import WorldModel

log = logging.getLogger("mythos")


def _monthly_folds(
    first_ts_ms: int,
    last_ts_ms: int,
    train_months: int,
    test_months: int,
) -> List[Tuple[str, str, str, str]]:
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError as exc:
        raise RuntimeError("python-dateutil is required for mythos walk-forward") from exc

    data_start = datetime.utcfromtimestamp(first_ts_ms / 1000)
    data_end = datetime.utcfromtimestamp(last_ts_ms / 1000)
    first_test_start = data_start + relativedelta(months=train_months)
    folds: List[Tuple[str, str, str, str]] = []
    cur = first_test_start
    while cur < data_end:
        train_end = cur
        test_end = cur + relativedelta(months=test_months)
        if test_end > data_end:
            test_end = data_end
        train_start = train_end - relativedelta(months=train_months)
        if train_start < data_start:
            train_start = data_start
        folds.append(
            (
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                cur.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )
        )
        cur = test_end
    return folds


def _slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # [start, end)
    st = pd.Timestamp(start, tz="UTC")
    et = pd.Timestamp(end, tz="UTC")
    t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[(t >= st) & (t < et)].copy()


def _compute_atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return atr


def _barrier_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    idx: int,
    side: int,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    atr: np.ndarray,
) -> float:
    # returns realized R in [-sl_mult,+tp_mult] clipped
    entry = close[idx]
    atr_i = max(float(atr[idx]), 1e-9)
    if side == 1:
        tp = entry + tp_mult * atr_i
        sl = entry - sl_mult * atr_i
        for j in range(idx + 1, min(idx + horizon + 1, len(close))):
            if low[j] <= sl:
                return -sl_mult
            if high[j] >= tp:
                return tp_mult
        fin = close[min(idx + horizon, len(close) - 1)]
        return float(np.clip((fin - entry) / atr_i, -sl_mult, tp_mult))
    # short
    tp = entry - tp_mult * atr_i
    sl = entry + sl_mult * atr_i
    for j in range(idx + 1, min(idx + horizon + 1, len(close))):
        if high[j] >= sl:
            return -sl_mult
        if low[j] <= tp:
            return tp_mult
    fin = close[min(idx + horizon, len(close) - 1)]
    return float(np.clip((entry - fin) / atr_i, -sl_mult, tp_mult))


def _run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: MythosConfig,
) -> Dict[str, object]:
    train_feat = build_feature_frame(train_df)
    test_feat = build_feature_frame(test_df)
    if len(train_feat) < 500 or len(test_feat) < 200:
        return {
            "total_trades": 0,
            "total_r": 0.0,
            "expectancy_r": 0.0,
            "win_rate": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "status": "DEAD",
            "promotion": {"should_promote": False, "reasons": ["insufficient_data"]},
        }

    wm = WorldModel(cfg.random_state)
    wm.fit(train_feat)
    train_regime = wm.predict_regime(train_feat)
    test_regime = wm.predict_regime(test_feat)

    experts = build_experts(cfg.random_state)
    X_tr = train_feat[["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"]].to_numpy(dtype=np.float64)
    y1 = train_feat["fwd_ret_1"].to_numpy(dtype=np.float64)
    y4 = train_feat["fwd_ret_4"].to_numpy(dtype=np.float64)
    y16 = train_feat["fwd_ret_16"].to_numpy(dtype=np.float64)
    for ex in experts:
        ex.fit(X_tr, y1, y4, y16, train_regime)

    router = MetaRouter(cfg.random_state)
    router.fit(train_feat, train_regime, experts)

    risk = RiskConstitution(cfg)
    close = test_feat["close"].to_numpy(dtype=np.float64)
    high = test_feat["high"].to_numpy(dtype=np.float64)
    low = test_feat["low"].to_numpy(dtype=np.float64)
    timestamps = test_feat["timestamp"].to_numpy(dtype=np.int64)
    atr = _compute_atr(close, high, low, period=14)
    X_te = test_feat[["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"]].to_numpy(dtype=np.float64)

    trades: List[float] = []
    n_long = 0
    n_short = 0
    for i in range(len(test_feat) - cfg.horizon):
        regime = int(test_regime[i])
        x = X_te[i]
        routed = router.route_one(
            x=x,
            regime=regime,
            experts=experts,
            vol_16=float(test_feat.iloc[i]["vol_16"]),
            trend_ema=float(test_feat.iloc[i]["trend_ema"]),
        )
        side = int(routed.get("side", 0))
        edge = float(routed.get("edge", 0.0))
        uncertainty = float(routed.get("uncertainty", 0.2))
        if not risk.allow_trade(ts_ms=int(timestamps[i]), side=side, edge=edge, uncertainty=uncertainty):
            continue
        realized = _barrier_outcome(
            close=close,
            high=high,
            low=low,
            idx=i,
            side=side,
            tp_mult=cfg.tp_mult,
            sl_mult=cfg.sl_mult,
            horizon=cfg.horizon,
            atr=atr,
        )
        size = risk.position_size_multiplier(edge=edge, uncertainty=uncertainty, regime=regime)
        rr = float(realized * size)
        risk.record_trade(rr, int(timestamps[i]), edge=edge, uncertainty=uncertainty)
        trades.append(rr)
        if side == 1:
            n_long += 1
        else:
            n_short += 1

    n = len(trades)
    total_r = float(np.sum(trades)) if n else 0.0
    expect = float(np.mean(trades)) if n else 0.0
    wr = float(np.mean(np.array(trades) > 0)) if n else 0.0
    status = "ACTIVE" if n >= cfg.min_trades_per_fold else ("LOW_CONF" if n > 0 else "DEAD")
    promotion = evaluate_promotion(
        expectancy=expect,
        win_rate=wr,
        trades=n,
        min_trades=cfg.min_trades_per_fold,
        edge_drift=float(np.nanstd(np.array(trades)) if n else 1.0),
        score_monotonic=expect > 0.0,
        side_balance=(min(n_long, n_short) / max(n_long + n_short, 1)),
    )
    return {
        "total_trades": n,
        "total_r": round(total_r, 4),
        "expectancy_r": round(expect, 4),
        "win_rate": round(wr, 4),
        "long_trades": n_long,
        "short_trades": n_short,
        "status": status,
        "promotion": asdict(promotion),
    }


def run_mythos_walk_forward(
    data_dir: Path,
    symbols: List[str],
    train_months: int,
    test_months: int,
    config: Optional[MythosConfig] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    if not symbols:
        raise ValueError("symbols must not be empty")
    cfg = config or MythosConfig()
    sym = symbols[0]
    data_path = data_dir / f"{sym}_15m.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    raw = pd.read_parquet(data_path)
    if "timestamp" not in raw.columns:
        raise ValueError("Input parquet must include 'timestamp' column")

    first_ts = int(raw["timestamp"].min())
    last_ts = int(raw["timestamp"].max())
    folds = _monthly_folds(first_ts, last_ts, train_months=train_months, test_months=test_months)
    log.info("[MYTHOS] Generated %d folds", len(folds))
    reports: List[Dict[str, object]] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        train_df = _slice_by_dates(raw, tr_s, tr_e)
        test_df = _slice_by_dates(raw, te_s, te_e)
        fold = _run_fold(train_df, test_df, cfg)
        fold.update(
            {
                "fold": i,
                "window_start": te_s,
                "window_end": te_e,
            }
        )
        reports.append(fold)
        log.info(
            "[MYTHOS][FOLD %d] trades=%d totalR=%+.2f status=%s long/short=%d/%d",
            i,
            fold["total_trades"],
            fold["total_r"],
            fold["status"],
            fold["long_trades"],
            fold["short_trades"],
        )

    total_trades = int(sum(r["total_trades"] for r in reports))
    total_r = float(sum(r["total_r"] for r in reports))
    act = sum(1 for r in reports if r["status"] == "ACTIVE")
    low = sum(1 for r in reports if r["status"] == "LOW_CONF")
    dead = sum(1 for r in reports if r["status"] == "DEAD")
    aggregate = {
        "symbol": sym,
        "folds": len(reports),
        "total_trades": total_trades,
        "total_r": round(total_r, 4),
        "expectancy_r": round((total_r / total_trades) if total_trades else 0.0, 4),
        "active_folds": act,
        "low_conf_folds": low,
        "dead_folds": dead,
    }
    result = {"folds": reports, "aggregate": aggregate, "config": asdict(cfg)}
    out = output_path or (Path("checkpoints") / "mythos_walkforward_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("[MYTHOS] Report saved to %s", out)
    return result
