from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from math import isfinite
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


class AnalogMemory:
    def __init__(self, X: np.ndarray, realized_r: np.ndarray, sides: np.ndarray, k: int = 48):
        self.X = np.asarray(X, dtype=np.float64)
        self.realized_r = np.asarray(realized_r, dtype=np.float64)
        self.sides = np.asarray(sides, dtype=np.int64)
        self.k = int(max(k, 4))
        self._mu = np.mean(self.X, axis=0) if self.X.size else np.zeros(1, dtype=np.float64)
        self._sigma = np.std(self.X, axis=0) + 1e-6 if self.X.size else np.ones(1, dtype=np.float64)
        self.Xn = (self.X - self._mu) / self._sigma if self.X.size else self.X

    def query(self, x: np.ndarray, side: int) -> Dict[str, float]:
        if self.Xn.size == 0:
            return {"analog_edge": 0.0, "analog_conf": 0.5, "analog_hits": 0.0}
        xn = (np.asarray(x, dtype=np.float64) - self._mu) / self._sigma
        d2 = np.sum((self.Xn - xn) ** 2, axis=1)
        order = np.argsort(d2)
        k = min(self.k, len(order))
        idx = order[:k]
        mask = self.sides[idx] == int(side)
        if np.any(mask):
            idx = idx[mask]
        rr = self.realized_r[idx]
        if rr.size == 0:
            return {"analog_edge": 0.0, "analog_conf": 0.5, "analog_hits": 0.0}
        analog_edge = float(np.mean(rr))
        analog_conf = float(np.clip(np.mean(rr > 0.0), 0.0, 1.0))
        return {"analog_edge": analog_edge, "analog_conf": analog_conf, "analog_hits": float(rr.size)}


def _build_analog_memory(train_feat: pd.DataFrame, cfg: MythosConfig) -> AnalogMemory:
    X = train_feat[["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"]].to_numpy(dtype=np.float64)
    # Fast proxy target (vectorized): avoids expensive per-bar barrier simulation
    # while still giving a useful analog retrieval memory.
    raw = (
        0.55 * train_feat["fwd_ret_4"].to_numpy(dtype=np.float64)
        + 0.45 * train_feat["fwd_ret_16"].to_numpy(dtype=np.float64)
    )
    vol = np.maximum(
        0.5
        * (
            train_feat["vol_16"].to_numpy(dtype=np.float64)
            + train_feat["vol_64"].to_numpy(dtype=np.float64)
        ),
        1e-6,
    )
    long_r = np.clip(raw / (vol * np.sqrt(8.0)), -cfg.sl_mult, cfg.tp_mult)
    short_r = -long_r
    usable = len(train_feat)
    X2 = np.concatenate([X[:usable], X[:usable]], axis=0)
    rr = np.concatenate([long_r[:usable], short_r[:usable]], axis=0)
    sides = np.concatenate(
        [np.ones(usable, dtype=np.int64), -np.ones(usable, dtype=np.int64)], axis=0
    )
    return AnalogMemory(X2, rr, sides, k=int(getattr(cfg, "analog_k", 48)))


class V3ExecutionGovernor:
    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self.side_hist: List[int] = []
        self.loss_streak = 0
        self.pause_until_bar = -1

    def adjusted_edge_floor(self, equity_r: float) -> float:
        base = float(self.cfg.abstain_edge_floor)
        dd = max(-float(equity_r), 0.0)
        start = float(max(getattr(self.cfg, "drawdown_edge_start_r", 8.0), 0.0))
        step = float(max(getattr(self.cfg, "drawdown_edge_step_r", 4.0), 1e-6))
        boost = float(max(getattr(self.cfg, "drawdown_edge_boost", 0.0025), 0.0))
        if dd <= start:
            return base
        increments = int((dd - start) // step) + 1
        return base + increments * boost

    def side_penalty(self, side: int) -> float:
        if side == 0:
            return 0.0
        w = int(max(getattr(self.cfg, "side_balance_window", 160), 8))
        hist = self.side_hist[-w:]
        if not hist:
            return 0.0
        long_frac = float(np.mean(np.array(hist) == 1))
        short_frac = float(np.mean(np.array(hist) == -1))
        dominant = max(long_frac, short_frac)
        soft_cap = float(np.clip(getattr(self.cfg, "side_imbalance_soft_cap", 0.82), 0.5, 1.0))
        if dominant <= soft_cap:
            return 0.0
        is_dominant_side = (side == 1 and long_frac >= short_frac) or (side == -1 and short_frac > long_frac)
        if not is_dominant_side:
            return 0.0
        penalty = float(max(getattr(self.cfg, "side_imbalance_edge_penalty", 0.015), 0.0))
        return penalty * (dominant - soft_cap) / max(1e-6, 1.0 - soft_cap)

    def allow_by_streak(self, bar_idx: int) -> bool:
        if bar_idx < self.pause_until_bar:
            return False
        return True

    def record_trade(self, side: int, realized_r: float, bar_idx: int) -> None:
        self.side_hist.append(int(side))
        if float(realized_r) < 0.0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        trig = int(max(getattr(self.cfg, "loss_streak_trigger", 4), 1))
        cd = int(max(getattr(self.cfg, "loss_streak_cooldown_bars", 12), 1))
        if self.loss_streak >= trig:
            self.pause_until_bar = int(bar_idx + cd)
            self.loss_streak = 0


class V4AdaptiveBrain:
    """
    Online adaptation layer:
    - detects abrupt local state changes (feature-space shock)
    - reweights experts with regime-conditional EWMA utility
    - hardens/relaxes confidence and edge based on detected instability
    """

    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self._recent_x: List[np.ndarray] = []
        self._recent_rr: List[float] = []
        self._recent_side: List[int] = []
        self._recent_regime: List[int] = []
        self._exp_global: Dict[str, float] = {}
        self._exp_regime: Dict[Tuple[int, str], float] = {}
        self._regime_streak = 0
        self._prev_regime: Optional[int] = None
        self._shock_streak = 0
        self._change_cooldown_until = -1

    def _shock_level(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        self._recent_x.append(x)
        max_keep = int(max(getattr(self.cfg, "side_balance_window", 160), 32))
        if len(self._recent_x) > max_keep:
            del self._recent_x[0 : len(self._recent_x) - max_keep]
        if len(self._recent_x) < 12:
            return 0.0
        recent = np.stack(self._recent_x, axis=0)
        mu = np.mean(recent, axis=0)
        sigma = np.std(recent, axis=0) + 1e-6
        z = np.mean(np.abs((x - mu) / sigma))
        trigger = float(max(getattr(self.cfg, "change_detect_z_thresh", 2.6), 0.5))
        return float(max((z - trigger) / max(trigger, 1e-6), 0.0))

    def _regime_flip_intensity(self, regime: int) -> float:
        reg = int(regime)
        if self._prev_regime is None:
            self._prev_regime = reg
            self._regime_streak = 1
            return 0.0
        if reg == self._prev_regime:
            self._regime_streak += 1
            self._prev_regime = reg
            return 0.0
        # Regime changed: short streak before flip indicates unstable transition.
        streak = max(self._regime_streak, 1)
        self._prev_regime = reg
        self._regime_streak = 1
        return float(np.clip(1.0 / streak, 0.0, 1.0))

    def adapt_signal(
        self,
        bar_idx: int,
        x: np.ndarray,
        regime: int,
        expert_name: str,
        edge: float,
        confidence: float,
        uncertainty: float,
    ) -> Dict[str, float]:
        shock = self._shock_level(x)
        flip = self._regime_flip_intensity(regime)
        # Regime-aware expert utility weighting.
        g = float(self._exp_global.get(expert_name, 0.0))
        r = float(self._exp_regime.get((int(regime), expert_name), 0.0))
        utility = 0.6 * g + 0.4 * r
        utility_w = float(np.tanh(utility))  # bounded in [-1,1]
        util_gain = float(max(getattr(self.cfg, "online_allocator_lr", 0.06), 0.0))
        min_mult = float(max(getattr(self.cfg, "online_allocator_min_mult", 0.75), 0.1))
        max_mult = float(max(getattr(self.cfg, "online_allocator_max_mult", 1.55), min_mult))
        alloc_mult = float(np.clip(1.0 + util_gain * utility_w, min_mult, max_mult))
        edge = float(edge * alloc_mult)
        confidence = float(np.clip(confidence + 0.07 * utility_w, 0.0, 1.0))
        # Instability hardening: tighten when shock/flip rises.
        shock_w = 0.18
        instability = np.clip(shock + flip, 0.0, 2.0)
        harden = shock_w * instability
        confidence = float(np.clip(confidence - 0.35 * harden, 0.0, 1.0))
        edge = float(edge * (1.0 - 0.45 * harden))
        uncertainty = float(np.clip(uncertainty * (1.0 + 0.60 * harden), 0.005, 1.5))
        confirm_need = int(max(getattr(self.cfg, "change_detect_confirm_bars", 2), 1))
        if shock > 0.0:
            self._shock_streak += 1
        else:
            self._shock_streak = max(self._shock_streak - 1, 0)
        cooldown = int(max(getattr(self.cfg, "change_detect_cooldown_bars", 24), 1))
        change_mode = False
        if self._shock_streak >= confirm_need and int(bar_idx) >= self._change_cooldown_until:
            change_mode = True
            self._change_cooldown_until = int(bar_idx + cooldown)
            self._shock_streak = 0
        if change_mode:
            edge += float(max(getattr(self.cfg, "change_edge_floor_boost", 0.004), 0.0))
            confidence = float(np.clip(confidence + float(getattr(self.cfg, "change_confidence_boost", 0.03)), 0.0, 1.0))
            uncertainty = float(
                np.clip(
                    uncertainty * float(max(getattr(self.cfg, "change_uncertainty_mult", 1.15), 1.0)),
                    0.005,
                    2.0,
                )
            )
        return {
            "edge": edge,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "shock": float(shock),
            "flip": float(flip),
            "change_mode": float(1.0 if change_mode else 0.0),
        }

    def update_after_trade(self, expert_name: str, regime: int, realized_r: float, side: int) -> None:
        rr = float(realized_r)
        alpha = float(np.clip(getattr(self.cfg, "online_allocator_lr", 0.06), 0.001, 0.8))
        prev_g = float(self._exp_global.get(expert_name, 0.0))
        self._exp_global[expert_name] = (1.0 - alpha) * prev_g + alpha * rr
        rk = (int(regime), expert_name)
        prev_r = float(self._exp_regime.get(rk, 0.0))
        self._exp_regime[rk] = (1.0 - alpha) * prev_r + alpha * rr
        self._recent_rr.append(rr)
        self._recent_side.append(int(side))
        self._recent_regime.append(int(regime))
        max_keep = int(max(getattr(self.cfg, "side_balance_window", 160), 32))
        if len(self._recent_rr) > max_keep:
            del self._recent_rr[0 : len(self._recent_rr) - max_keep]
            del self._recent_side[0 : len(self._recent_side) - max_keep]
            del self._recent_regime[0 : len(self._recent_regime) - max_keep]

    def state_dict(self) -> Dict[str, object]:
        return {
            "exp_global": {k: float(v) for k, v in self._exp_global.items()},
            "exp_regime": {f"{k[0]}::{k[1]}": float(v) for k, v in self._exp_regime.items()},
            "recent_rr": [float(v) for v in self._recent_rr],
            "recent_side": [int(v) for v in self._recent_side],
            "recent_regime": [int(v) for v in self._recent_regime],
        }


def _select_fold_metric(fold: Dict[str, object], metric: str) -> float:
    metric = str(metric or "total_r").lower()
    if metric == "expectancy_r":
        return float(fold.get("expectancy_r", 0.0))
    if metric == "win_rate":
        return float(fold.get("win_rate", 0.0))
    if metric == "robust_score":
        return float(fold.get("robust_score", 0.0))
    return float(fold.get("total_r", 0.0))


def _save_model_artifact(
    output_dir: Path,
    symbol: str,
    fold_idx: int,
    window_start: str,
    window_end: str,
    metric_name: str,
    metric_value: float,
    cfg: MythosConfig,
    wm: WorldModel,
    experts,
    router: Optional[MetaRouter] = None,
    brain: Optional[V4AdaptiveBrain] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"mythos_best_{symbol}.json"
    payload = {
        "kind": "mythos_best_fold_model",
        "symbol": symbol,
        "fold": int(fold_idx),
        "window_start": window_start,
        "window_end": window_end,
        "selection_metric": metric_name,
        "selection_value": float(metric_value),
        "config": asdict(cfg),
        "world_model": wm.to_state_dict(),
        "experts": [ex.to_state_dict() for ex in experts],
        "router": router.to_state_dict() if router is not None else None,
        "adaptive_brain": brain.state_dict() if brain is not None else None,
    }
    model_path.write_text(json.dumps(payload, indent=2))
    return model_path


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
            "wins": 0,
            "losses": 0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "status": "DEAD",
            "promotion": {"promote": False, "reasons": ["insufficient_data"], "checks": {}},
            "max_drawdown_r": 0.0,
            "robust_score": 0.0,
            "_model_state": None,
        }

    wm = WorldModel(n_states=cfg.n_regimes, random_state=cfg.random_state)
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

    router = MetaRouter(cfg)
    router.fit(train_feat, train_regime, experts)
    analog_mem = _build_analog_memory(train_feat, cfg)
    governor = V3ExecutionGovernor(cfg)
    adaptive = V4AdaptiveBrain(cfg)

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
    change_mode_bars = 0
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
        expert_name = str(routed.get("expert_name", "none"))
        side = int(routed.get("side", 0))
        edge = float(routed.get("edge", 0.0))
        confidence = float(routed.get("confidence", 0.0))
        uncertainty = float(routed.get("uncertainty", 0.2))
        analog = analog_mem.query(x=x, side=side) if side != 0 else {"analog_edge": 0.0, "analog_conf": 0.5, "analog_hits": 0.0}
        blend = float(np.clip(getattr(cfg, "analog_blend", 0.35), 0.0, 0.95))
        edge = (1.0 - blend) * edge + blend * float(analog["analog_edge"])
        confidence = (1.0 - blend) * confidence + blend * float(analog["analog_conf"])
        hit_ratio = min(float(analog["analog_hits"]) / max(float(getattr(cfg, "analog_k", 48)), 1.0), 1.0)
        conf_lift = max(confidence - 0.5, 0.0)
        uncertainty = float(np.clip(uncertainty * (1.0 - 0.25 * conf_lift * hit_ratio), 0.005, 1.0))
        adapted = adaptive.adapt_signal(
            x=x,
            regime=regime,
            expert_name=expert_name,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
        )
        edge = float(adapted["edge"])
        confidence = float(adapted["confidence"])
        uncertainty = float(adapted["uncertainty"])
        if float(adapted.get("change_mode", 0.0)) > 0.5:
            change_mode_bars += 1
        edge_floor = governor.adjusted_edge_floor(risk.state.equity_r)
        edge -= governor.side_penalty(side)
        if confidence < cfg.min_confidence:
            continue
        if edge < edge_floor:
            continue
        if not governor.allow_by_streak(i):
            continue
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
        governor.record_trade(side=side, realized_r=rr, bar_idx=i)
        router.update_reliability(expert_name=expert_name, realized_r=rr, regime=regime)
        adaptive.update_after_trade(expert_name=expert_name, regime=regime, realized_r=rr, side=side)
        trades.append(rr)
        if side == 1:
            n_long += 1
        else:
            n_short += 1

    n = len(trades)
    wins = int(np.sum(np.array(trades) > 0.0)) if n else 0
    losses = int(np.sum(np.array(trades) < 0.0)) if n else 0
    gross_profit = float(np.sum(np.array([t for t in trades if t > 0.0], dtype=np.float64))) if n else 0.0
    gross_loss = float(-np.sum(np.array([t for t in trades if t < 0.0], dtype=np.float64))) if n else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0.0 else 0.0)
    total_r = float(np.sum(trades)) if n else 0.0
    expect = float(np.mean(trades)) if n else 0.0
    wr = float(np.mean(np.array(trades) > 0)) if n else 0.0
    eq = np.cumsum(np.array(trades, dtype=np.float64)) if n else np.array([0.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_drawdown_r = float(np.min(dd)) if dd.size else 0.0
    dd_pen = float(np.clip(getattr(cfg, "robust_score_dd_penalty", 0.35), 0.0, 5.0))
    pf_for_score = profit_factor if isfinite(profit_factor) else 3.0
    robust_score = float(expect * 100.0 + (pf_for_score - 1.0) * 5.0 - dd_pen * abs(max_drawdown_r))
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
        "wins": wins,
        "losses": losses,
        "gross_profit_r": round(gross_profit, 4),
        "gross_loss_r": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4) if isfinite(profit_factor) else "inf",
        "max_drawdown_r": round(max_drawdown_r, 4),
        "robust_score": round(robust_score, 6),
        "change_mode_bars": int(change_mode_bars),
        "long_trades": n_long,
        "short_trades": n_short,
        "status": status,
        "promotion": asdict(promotion),
        "_model_state": {"world_model": wm, "experts": experts, "router": router, "adaptive": adaptive},
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
    if cfg.max_folds is not None:
        folds = folds[: max(int(cfg.max_folds), 0)]
    log.info("[MYTHOS] Generated %d folds", len(folds))
    reports: List[Dict[str, object]] = []
    best_metric = float("-inf")
    best_fold_artifact: Optional[Path] = None
    metric_name = str(getattr(cfg, "best_model_metric", "total_r")).lower()
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        train_df = _slice_by_dates(raw, tr_s, tr_e)
        test_df = _slice_by_dates(raw, te_s, te_e)
        fold = _run_fold(train_df, test_df, cfg)
        model_state = fold.pop("_model_state", None)
        fold.update(
            {
                "fold": i,
                "window_start": te_s,
                "window_end": te_e,
            }
        )
        reports.append(fold)
        if cfg.save_best_model and model_state is not None:
            fold_metric = _select_fold_metric(fold, metric_name)
            if fold_metric > best_metric:
                best_metric = fold_metric
                best_fold_artifact = _save_model_artifact(
                    output_dir=Path(cfg.model_output_dir),
                    symbol=sym,
                    fold_idx=i,
                    window_start=te_s,
                    window_end=te_e,
                    metric_name=metric_name,
                    metric_value=fold_metric,
                    cfg=cfg,
                    wm=model_state["world_model"],
                    experts=model_state["experts"],
                    router=model_state.get("router"),
                    brain=model_state.get("adaptive"),
                )
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
    total_wins = int(sum(r.get("wins", 0) for r in reports))
    total_losses = int(sum(r.get("losses", 0) for r in reports))
    gross_profit = float(sum(r.get("gross_profit_r", 0.0) for r in reports))
    gross_loss = float(sum(r.get("gross_loss_r", 0.0) for r in reports))
    agg_win_rate = (total_wins / total_trades) if total_trades else 0.0
    agg_pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0.0 else 0.0)
    total_r = float(sum(r["total_r"] for r in reports))
    avg_max_drawdown = float(np.mean([r.get("max_drawdown_r", 0.0) for r in reports])) if reports else 0.0
    avg_robust_score = float(np.mean([r.get("robust_score", 0.0) for r in reports])) if reports else 0.0
    total_change_mode_bars = int(sum(r.get("change_mode_bars", 0) for r in reports))
    change_mode_rate = float(total_change_mode_bars / max(total_trades, 1))
    act = sum(1 for r in reports if r["status"] == "ACTIVE")
    low = sum(1 for r in reports if r["status"] == "LOW_CONF")
    dead = sum(1 for r in reports if r["status"] == "DEAD")
    aggregate = {
        "symbol": sym,
        "folds": len(reports),
        "n_folds": len(reports),
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": round(agg_win_rate, 4),
        "gross_profit_r": round(gross_profit, 4),
        "gross_loss_r": round(gross_loss, 4),
        "profit_factor": round(agg_pf, 4) if isfinite(agg_pf) else "inf",
        "total_r": round(total_r, 4),
        "expectancy_r": round((total_r / total_trades) if total_trades else 0.0, 4),
        "avg_max_drawdown_r": round(avg_max_drawdown, 4),
        "avg_robust_score": round(avg_robust_score, 6),
        "change_mode_bars": total_change_mode_bars,
        "change_mode_rate": round(change_mode_rate, 4),
        "active_folds": act,
        "low_conf_folds": low,
        "dead_folds": dead,
        "best_model_metric": metric_name,
        "best_model_metric_value": round(best_metric, 6) if best_metric > float("-inf") else None,
        "best_model_path": str(best_fold_artifact) if best_fold_artifact is not None else None,
    }
    result = {"folds": reports, "aggregate": aggregate, "config": asdict(cfg)}
    out = output_path or (Path("checkpoints") / "mythos_walkforward_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("[MYTHOS] Report saved to %s", out)
    return result
