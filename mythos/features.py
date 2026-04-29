from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"]
    l = df["low"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(df, period=period)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    di_sum = plus_di.abs() + minus_di.abs()
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


@dataclass
class MythosFrame:
    X: np.ndarray
    y: np.ndarray
    returns: np.ndarray
    timestamps: np.ndarray
    feature_names: List[str]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature builder used by the walk-forward engine.
    """
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = data["close"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    ret4 = close.pct_change(4).fillna(0.0)
    ret16 = close.pct_change(16).fillna(0.0)
    fwd1 = close.shift(-1) / close - 1.0
    fwd4 = close.shift(-4) / close - 1.0
    fwd16 = close.shift(-16) / close - 1.0
    vol16 = ret1.rolling(16).std().fillna(0.0)
    vol64 = ret1.rolling(64).std().fillna(0.0)
    z64 = ((close - close.rolling(64).mean()) / close.rolling(64).std().replace(0.0, np.nan)).fillna(0.0)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    trend = (ema20 / ema100 - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    range_break = ((data["high"] - data["low"]) / close.replace(0.0, np.nan)).fillna(0.0)
    out = pd.DataFrame(
        {
            "timestamp": data["timestamp"].astype(np.int64),
            "open": data["open"].astype(float),
            "high": data["high"].astype(float),
            "low": data["low"].astype(float),
            "close": close,
            "ret_1": ret1,
            "ret_4": ret4,
            "ret_16": ret16,
            "fwd_ret_1": fwd1.fillna(0.0),
            "fwd_ret_4": fwd4.fillna(0.0),
            "fwd_ret_16": fwd16.fillna(0.0),
            "vol_16": vol16,
            "vol_64": vol64,
            "zscore_64": z64,
            "trend_ema": trend,
            "range_break_48": range_break,
        }
    )
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_mythos_features(ohlcv: np.ndarray) -> np.ndarray:
    """
    Compatibility helper for tests: expects [open, high, low, close, volume].
    """
    arr = np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("ohlcv must be a 2D array with at least 5 columns")
    df = pd.DataFrame(
        {
            "timestamp": np.arange(len(arr), dtype=np.int64) * 60_000,
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": arr[:, 4],
        }
    )
    feat = build_feature_frame(df)
    cols = ["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema", "range_break_48"]
    return feat[cols].to_numpy(dtype=np.float64)


def build_mythos_frame(df: pd.DataFrame, horizon: int = 48, score_quantile: float = 0.75) -> MythosFrame:
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = data["close"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    ret4 = close.pct_change(4).fillna(0.0)
    ret16 = close.pct_change(16).fillna(0.0)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()
    trend = (ema_fast / ema_slow - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    vol = ret1.rolling(32).std().fillna(0.0)
    atr = _atr(data, period=14).fillna(method="bfill").fillna(0.0)
    atr_pct = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rsi = _rsi(close, period=14).fillna(50.0) / 100.0
    adx = _adx(data, period=14).fillna(20.0) / 100.0
    candle_spread = ((data["high"] - data["low"]) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    volume_z = (
        (data["volume"] - data["volume"].rolling(64).mean())
        / data["volume"].rolling(64).std().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feature_cols: List[Tuple[str, pd.Series]] = [
        ("ret1", ret1),
        ("ret4", ret4),
        ("ret16", ret16),
        ("trend", trend),
        ("vol", vol),
        ("atr_pct", atr_pct),
        ("rsi", rsi),
        ("adx", adx),
        ("spread", candle_spread),
        ("volume_z", volume_z),
    ]
    X_df = pd.concat([series.rename(name) for name, series in feature_cols], axis=1)
    X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X = X_df.values.astype(np.float32)

    fwd_ret = close.shift(-horizon) / close - 1.0
    fwd_ret = fwd_ret.fillna(0.0).astype(float)
    # Binary edge label: top quantile forward returns in this split.
    thr = float(np.quantile(fwd_ret.values, score_quantile))
    y = (fwd_ret.values >= thr).astype(np.int64)
    y[fwd_ret.values <= 0.0] = 0

    max_idx = len(data) - horizon
    if max_idx <= 256:
        raise ValueError("Not enough bars to build mythos frame")

    return MythosFrame(
        X=X[:max_idx],
        y=y[:max_idx],
        returns=fwd_ret.values[:max_idx].astype(np.float32),
        timestamps=data["timestamp"].values[:max_idx],
        feature_names=[name for name, _ in feature_cols],
    )

