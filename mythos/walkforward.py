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
from .features import MYTHOS_FEATURE_COLUMNS, build_feature_frame
from .promotion import evaluate_promotion
from .risk import RiskConstitution
from .router import MetaRouter
from .world_model import WorldModel

log = logging.getLogger("mythos")


MYTHOS_STATE_COLS = list(MYTHOS_FEATURE_COLUMNS)


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
    cols = [c for c in MYTHOS_STATE_COLS if c in train_feat.columns]
    X = train_feat[cols].to_numpy(dtype=np.float64)
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
        self.side_recent_rr: Dict[int, List[float]] = {1: [], -1: []}
        self.side_pause_until: Dict[int, int] = {1: -1, -1: -1}
        self.side_health: Dict[int, float] = {1: 0.0, -1: 0.0}
        self.side_health_ema: Dict[int, float] = {1: 0.0, -1: 0.0}

    def _adaptive_long_target(self) -> float:
        strength = float(np.clip(getattr(self.cfg, "adaptive_side_target_strength", 0.22), 0.0, 1.0))
        target_min = float(np.clip(getattr(self.cfg, "adaptive_side_target_min", 0.35), 0.05, 0.95))
        target_max = float(np.clip(getattr(self.cfg, "adaptive_side_target_max", 0.65), 0.05, 0.95))
        if target_min > target_max:
            target_min, target_max = target_max, target_min
        long_h = float(np.tanh(self.side_health.get(1, 0.0)))
        short_h = float(np.tanh(self.side_health.get(-1, 0.0)))
        target = 0.5 + 0.5 * strength * (long_h - short_h)
        return float(np.clip(target, target_min, target_max))

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
            concentration_penalty = 0.0
        else:
            is_dominant_side = (side == 1 and long_frac >= short_frac) or (side == -1 and short_frac > long_frac)
            if not is_dominant_side:
                concentration_penalty = 0.0
            else:
                penalty = float(max(getattr(self.cfg, "side_imbalance_edge_penalty", 0.015), 0.0))
                concentration_penalty = penalty * (dominant - soft_cap) / max(1e-6, 1.0 - soft_cap)

        # Adaptive side allocator: keeps trading active while nudging side mix
        # toward whichever side is currently healthier.
        target_long = self._adaptive_long_target()
        target_side = target_long if int(side) == 1 else (1.0 - target_long)
        side_frac = long_frac if int(side) == 1 else short_frac
        frac_gap = float(side_frac - target_side)
        own_health = float(np.tanh(self.side_health.get(int(side), 0.0)))
        opp_health = float(np.tanh(self.side_health.get(-int(side), 0.0)))
        health_gap = own_health - opp_health
        dyn_penalty_scale = float(max(getattr(self.cfg, "side_health_penalty", 0.02), 0.0))
        dyn_boost_scale = float(max(getattr(self.cfg, "side_health_boost", 0.008), 0.0))
        if frac_gap > 0.0:
            adaptive_term = dyn_penalty_scale * frac_gap * (1.0 + max(-health_gap, 0.0))
        else:
            adaptive_term = -dyn_boost_scale * (-frac_gap) * (1.0 + max(-health_gap, 0.0))
        return float(concentration_penalty + adaptive_term)

    def allow_by_streak(self, bar_idx: int, side: int = 0) -> bool:
        if bar_idx < self.pause_until_bar:
            return False
        if bool(getattr(self.cfg, "side_fail_hard_pause", False)) and int(side) in (-1, 1) and bar_idx < int(
            self.side_pause_until.get(int(side), -1)
        ):
            return False
        return True

    def _update_side_health(self, side: int, bar_idx: int) -> None:
        side = int(side)
        if side not in (-1, 1):
            return
        hist = self.side_recent_rr.setdefault(side, [])
        w = int(max(getattr(self.cfg, "side_fail_window", 48), 8))
        if len(hist) > w:
            del hist[0 : len(hist) - w]
        min_n = int(max(getattr(self.cfg, "side_fail_min_trades", 10), 1))
        if len(hist) < min_n:
            return
        fail_expect = float(getattr(self.cfg, "side_fail_expectancy_r", -0.12))
        side_expect = float(np.mean(np.array(hist, dtype=np.float64)))
        if side_expect > fail_expect:
            return
        shortfall = float(max(fail_expect - side_expect, 0.0))
        self.side_health[side] = float(self.side_health.get(side, 0.0) - shortfall)
        if bool(getattr(self.cfg, "side_fail_hard_pause", False)):
            cd = int(max(getattr(self.cfg, "side_fail_cooldown_bars", 24), 1))
            self.side_pause_until[side] = max(int(self.side_pause_until.get(side, -1)), int(bar_idx + cd))
        # Reset side-local buffer after triggering to avoid repetitive stale signals.
        self.side_recent_rr[side] = []

    def record_trade(self, side: int, realized_r: float, bar_idx: int) -> None:
        self.side_hist.append(int(side))
        s = int(side)
        if s in (-1, 1):
            self.side_recent_rr.setdefault(s, []).append(float(realized_r))
            decay = float(np.clip(getattr(self.cfg, "side_health_decay", 0.97), 0.7, 0.999))
            rr = float(np.clip(realized_r, -2.0, 2.0))
            prev_h = float(self.side_health.get(s, 0.0))
            self.side_health[s] = decay * prev_h + (1.0 - decay) * rr
            ema_alpha = float(np.clip(getattr(self.cfg, "side_fail_ema_alpha", 0.25), 0.01, 1.0))
            prev_ema = float(self.side_health_ema.get(s, 0.0))
            self.side_health_ema[s] = (1.0 - ema_alpha) * prev_ema + ema_alpha * rr
        if float(realized_r) < 0.0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        trig = int(max(getattr(self.cfg, "loss_streak_trigger", 4), 1))
        cd = int(max(getattr(self.cfg, "loss_streak_cooldown_bars", 12), 1))
        if self.loss_streak >= trig:
            self.pause_until_bar = int(bar_idx + cd)
            self.loss_streak = 0
        self._update_side_health(side=s, bar_idx=bar_idx)


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
        self._transition_outcome: Dict[Tuple[int, int], float] = {}
        self._transition_count: Dict[Tuple[int, int], int] = {}
        self._regime_streak = 0
        self._prev_regime: Optional[int] = None
        self._last_transition: Tuple[int, int] | None = None
        self._shock_streak = 0
        self._change_cooldown_until = -1
        self._flip_pressure_until = -1

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
            self._last_transition = None
            return 0.0
        # Regime changed: short streak before flip indicates unstable transition.
        prev_reg = int(self._prev_regime)
        streak = max(self._regime_streak, 1)
        self._prev_regime = reg
        self._regime_streak = 1
        self._last_transition = (prev_reg, reg)
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
        # Transition learner: learn online which regime transitions are favorable/unfavorable.
        trans_min_n = int(max(getattr(self.cfg, "transition_min_samples", 6), 1))
        trans_edge_scale = float(np.clip(getattr(self.cfg, "transition_edge_gain", 0.35), 0.0, 2.0))
        trans_conf_scale = float(np.clip(getattr(self.cfg, "transition_confidence_gain", 0.06), 0.0, 1.0))
        trans_unc_scale = float(np.clip(getattr(self.cfg, "transition_uncertainty_gain", 0.30), 0.0, 2.0))
        trans = self._last_transition
        if trans is not None:
            trans_mean = float(self._transition_outcome.get(trans, 0.0))
            trans_n = int(self._transition_count.get(trans, 0))
            if trans_n >= trans_min_n:
                trans_score = float(np.clip(trans_mean, -1.0, 1.0))
                edge = float(edge * (1.0 + trans_edge_scale * trans_score))
                confidence = float(np.clip(confidence + trans_conf_scale * trans_score, 0.0, 1.0))
                uncertainty = float(np.clip(uncertainty * (1.0 - trans_unc_scale * trans_score), 0.005, 1.5))
            else:
                trans_score = 0.0
        else:
            trans_score = 0.0
        # Regime flip pressure: aggressively harden for a short window after abrupt flips.
        flip_trig = float(max(getattr(self.cfg, "flip_intensity_trigger", 0.35), 0.0))
        if flip >= flip_trig:
            hold = int(max(getattr(self.cfg, "flip_harden_hold_bars", 24), 1))
            self._flip_pressure_until = max(self._flip_pressure_until, int(bar_idx + hold))
        flip_pressure = 1.0 if int(bar_idx) < int(self._flip_pressure_until) else 0.0
        # Instability hardening: tighten when shock/flip rises.
        shock_w = 0.18
        instability = np.clip(shock + flip + 0.75 * flip_pressure, 0.0, 2.5)
        harden = shock_w * instability
        inst_edge_mult = float(np.clip(getattr(self.cfg, "instability_edge_mult", 0.70), 0.1, 1.0))
        inst_conf_drop = float(np.clip(getattr(self.cfg, "instability_confidence_drop", 0.08), 0.0, 0.5))
        inst_unc_mult = float(max(getattr(self.cfg, "instability_uncertainty_mult", 1.35), 1.0))
        confidence = float(np.clip(confidence - inst_conf_drop * harden, 0.0, 1.0))
        edge = float(edge * (1.0 - (1.0 - inst_edge_mult) * harden))
        uncertainty = float(np.clip(uncertainty * (1.0 + (inst_unc_mult - 1.0) * harden), 0.005, 1.5))
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
            "flip_pressure": float(flip_pressure),
            "transition_score": float(trans_score),
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
        trans = self._last_transition
        if trans is not None:
            t_alpha = float(np.clip(getattr(self.cfg, "transition_learn_rate", 0.12), 0.001, 0.95))
            prev_t = float(self._transition_outcome.get(trans, 0.0))
            self._transition_outcome[trans] = (1.0 - t_alpha) * prev_t + t_alpha * rr
            self._transition_count[trans] = int(self._transition_count.get(trans, 0)) + 1
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
            "transition_outcome": {f"{k[0]}->{k[1]}": float(v) for k, v in self._transition_outcome.items()},
            "transition_count": {f"{k[0]}->{k[1]}": int(v) for k, v in self._transition_count.items()},
            "recent_rr": [float(v) for v in self._recent_rr],
            "recent_side": [int(v) for v in self._recent_side],
            "recent_regime": [int(v) for v in self._recent_regime],
        }


class NeuralMetaLearner:
    """
    Online neural quality scorer for final trade gating.
    Learns whether routed decisions convert into positive R in current context.
    """

    def __init__(self, cfg: MythosConfig, n_features: int):
        self.cfg = cfg
        self.n_features = int(max(n_features, 1))
        self.enabled = bool(getattr(cfg, "use_meta_learner", True))
        self.device = "cpu"
        self._active = False
        self._torch = None
        self._nn = None
        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._fallback_active = False
        self._fb_w: Optional[np.ndarray] = None
        self._fb_b = 0.0
        self._buffer_x: List[np.ndarray] = []
        self._buffer_y: List[float] = []
        self._max_buffer = int(max(getattr(cfg, "meta_learner_buffer_size", 6000), 256))
        self._trained_steps = 0
        self._ready = False
        self._init_model()

    def _enable_fallback(self) -> None:
        if not bool(getattr(self.cfg, "meta_learner_fallback", True)):
            return
        self.device = "cpu"
        self._fallback_active = True
        self._fb_w = np.zeros(self.n_features + 5, dtype=np.float64)
        self._fb_b = 0.0
        self._active = True

    def _effective_min_samples(self) -> int:
        requested = int(max(getattr(self.cfg, "meta_learner_min_train_samples", 256), 16))
        # Fold-local training may have far fewer trades than the static default.
        # Keep a floor, but allow earlier adaptation inside each fold.
        fold_adaptive = int(max(min(getattr(self.cfg, "side_balance_window", 160), 192) * 0.6, 64))
        return int(min(requested, fold_adaptive))

    def _init_model(self) -> None:
        if not self.enabled:
            return
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self._enable_fallback()
            return
        requested = str(getattr(self.cfg, "meta_learner_device", "auto")).strip().lower()
        has_cuda = bool(torch.cuda.is_available())
        if requested == "cuda":
            if not has_cuda:
                self._enable_fallback()
                return
            self.device = "cuda"
        elif requested == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if has_cuda else "cpu"
        hidden = int(max(getattr(self.cfg, "meta_learner_hidden", 96), 8))
        dropout = float(np.clip(getattr(self.cfg, "meta_learner_dropout", 0.10), 0.0, 0.9))
        self._model = nn.Sequential(
            nn.Linear(self.n_features + 5, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        ).to(self.device)
        lr = float(max(getattr(self.cfg, "meta_learner_lr", 7e-4), 1e-6))
        wd = float(max(getattr(self.cfg, "meta_learner_weight_decay", 1e-6), 0.0))
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        self._loss_fn = nn.BCELoss()
        self._torch = torch
        self._nn = nn
        self._active = True

    def is_active(self) -> bool:
        torch_active = bool(self._model is not None and self._torch is not None)
        return bool(self._active and (torch_active or self._fallback_active))

    def is_ready(self) -> bool:
        return bool(self.is_active() and self._ready)

    def _meta_vector(
        self,
        x: np.ndarray,
        edge: float,
        confidence: float,
        uncertainty: float,
        regime: int,
        side: int,
    ) -> np.ndarray:
        core = np.asarray(x, dtype=np.float64).reshape(-1)
        aux = np.array(
            [
                float(edge),
                float(confidence),
                float(uncertainty),
                float(regime),
                float(side),
            ],
            dtype=np.float64,
        )
        return np.concatenate([core, aux], axis=0)

    def score(
        self,
        x: np.ndarray,
        edge: float,
        confidence: float,
        uncertainty: float,
        regime: int,
        side: int,
    ) -> float:
        if not self.is_ready():
            return 0.5
        vec = self._meta_vector(x=x, edge=edge, confidence=confidence, uncertainty=uncertainty, regime=regime, side=side)
        if self._fallback_active and self._fb_w is not None:
            vv = vec / (np.linalg.norm(vec) + 1e-6)
            z = float(np.dot(self._fb_w, vv) + self._fb_b)
            p = 1.0 / (1.0 + float(np.exp(-np.clip(z, -40.0, 40.0))))
            return float(np.clip(p, 0.0, 1.0))
        t = self._torch.tensor(vec.reshape(1, -1), dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            p = float(self._model(t).item())
        return float(np.clip(p, 0.0, 1.0))

    def apply_adjustments(
        self,
        edge: float,
        confidence: float,
        uncertainty: float,
        meta_p: float,
    ) -> Dict[str, float]:
        signed = float((float(meta_p) - 0.5) * 2.0)  # in [-1, 1]
        edge_gain = float(np.clip(getattr(self.cfg, "meta_learner_edge_blend", 0.30), 0.0, 2.0))
        conf_gain = float(np.clip(getattr(self.cfg, "meta_learner_conf_blend", 0.20), 0.0, 1.0))
        unc_gain = float(np.clip(getattr(self.cfg, "meta_learner_uncertainty_penalty", 0.80), 0.0, 2.0))
        out_edge = float(edge * (1.0 + edge_gain * signed))
        out_conf = float(np.clip(confidence + conf_gain * signed, 0.0, 1.0))
        # Raise uncertainty when meta confidence is poor; lower when strong.
        out_unc = float(np.clip(uncertainty * (1.0 - 0.35 * unc_gain * signed), 0.005, 2.0))
        return {"edge": out_edge, "confidence": out_conf, "uncertainty": out_unc}

    def pass_filter(self, side: int, meta_p: float) -> bool:
        if int(side) == 0:
            return False
        min_side = float(np.clip(getattr(self.cfg, "meta_learner_min_side_prob", 0.50), 0.0, 1.0))
        p = float(np.clip(meta_p, 0.0, 1.0))
        if int(side) > 0:
            return p >= min_side
        return (1.0 - p) >= min_side

    def update(
        self,
        x: np.ndarray,
        edge: float,
        confidence: float,
        uncertainty: float,
        regime: int,
        side: int,
        realized_r: float,
    ) -> None:
        if not self.is_active():
            return
        vec = self._meta_vector(x=x, edge=edge, confidence=confidence, uncertainty=uncertainty, regime=regime, side=side)
        y = 1.0 if float(realized_r) > 0.0 else 0.0
        self._buffer_x.append(vec)
        self._buffer_y.append(y)
        if len(self._buffer_x) > self._max_buffer:
            trim = len(self._buffer_x) - self._max_buffer
            del self._buffer_x[:trim]
            del self._buffer_y[:trim]
        min_n = self._effective_min_samples()
        if len(self._buffer_x) < min_n:
            return
        batch_size = int(max(getattr(self.cfg, "meta_learner_batch_size", 1024), 16))
        train_steps = int(max(getattr(self.cfg, "meta_learner_train_steps", 2), 1))
        x_np = np.asarray(self._buffer_x, dtype=np.float64)
        y_np = np.asarray(self._buffer_y, dtype=np.float64)
        n = len(x_np)
        if self._fallback_active and self._fb_w is not None:
            lr = float(np.clip(getattr(self.cfg, "meta_learner_fallback_lr", 0.03), 1e-5, 0.5))
            reg = float(np.clip(getattr(self.cfg, "meta_learner_reg_weight", 0.25), 0.0, 2.0))
            for _ in range(train_steps):
                idx = np.random.choice(n, size=min(batch_size, n), replace=False)
                xb = x_np[idx]
                yb = y_np[idx]
                xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-6)
                z = np.clip(xb @ self._fb_w + self._fb_b, -40.0, 40.0)
                pred = 1.0 / (1.0 + np.exp(-z))
                err = pred - yb
                grad_w = (xb.T @ err) / max(len(idx), 1) + reg * self._fb_w
                grad_b = float(np.mean(err))
                self._fb_w = self._fb_w - lr * grad_w
                self._fb_b = float(self._fb_b - lr * grad_b)
            self._trained_steps += int(train_steps)
            self._ready = True
            return
        for _ in range(train_steps):
            idx = np.random.choice(n, size=min(batch_size, n), replace=False)
            xb = self._torch.tensor(x_np[idx], dtype=self._torch.float32, device=self.device)
            yb = self._torch.tensor(y_np[idx].reshape(-1, 1), dtype=self._torch.float32, device=self.device)
            pred = self._model(xb)
            loss = self._loss_fn(pred, yb)
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
        self._trained_steps += int(train_steps)
        self._ready = True

    def state_dict(self) -> Dict[str, object]:
        return {
            "active": bool(self.is_active()),
            "ready": bool(self.is_ready()),
            "device": str(self.device),
            "fallback_active": bool(self._fallback_active),
            "buffer_size": int(len(self._buffer_x)),
            "max_buffer": int(self._max_buffer),
            "trained_steps": int(self._trained_steps),
        }


def _bootstrap_meta(meta: NeuralMetaLearner, train_feat: pd.DataFrame, train_regime: np.ndarray) -> None:
    if not meta.is_active() or train_feat.empty:
        return
    cols = [c for c in MYTHOS_STATE_COLS if c in train_feat.columns]
    if not cols:
        return
    X = train_feat[cols].to_numpy(dtype=np.float64)
    n = len(X)
    if n <= 8:
        return
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
    norm = raw / (vol * np.sqrt(8.0))
    norm = np.clip(norm, -2.0, 2.0)
    q_hi = float(np.quantile(norm, 0.55))
    q_lo = float(np.quantile(norm, 0.45))
    # Use directional proxy labels to warm start meta learner so it doesn't stay inactive
    # in folds where realized trades are sparse.
    for i in range(n):
        rr = float(norm[i])
        if rr >= q_hi:
            side = 1
            target_rr = abs(rr) + 0.05
        elif rr <= q_lo:
            side = -1
            target_rr = abs(rr) + 0.05
        else:
            # neutral points are still useful for negative examples
            side = 1 if rr >= 0.0 else -1
            target_rr = -abs(rr)
        edge = float(max(abs(rr) * 0.6 + 0.004, 0.0))
        conf = float(np.clip(0.5 + 0.3 * abs(rr), 0.35, 0.95))
        unc = float(np.clip(0.4 - 0.2 * abs(rr), 0.05, 0.8))
        reg = int(train_regime[i]) if i < len(train_regime) else 0
        meta.update(
            x=X[i],
            edge=edge,
            confidence=conf,
            uncertainty=unc,
            regime=reg,
            side=side,
            realized_r=target_rr,
        )


def _conviction_score(
    edge: float,
    confidence: float,
    uncertainty: float,
    meta_p: float,
    cfg: MythosConfig,
) -> float:
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.005), 1e-6))
    edge_n = float(np.clip(max(edge, 0.0) / edge_unit, 0.0, 4.0) / 4.0)
    conf_n = float(np.clip(confidence, 0.0, 1.0))
    unc_n = float(np.clip(1.0 / (1.0 + max(float(uncertainty), 0.0)), 0.0, 1.0))
    meta_n = float(np.clip(abs(float(meta_p) - 0.5) * 2.0, 0.0, 1.0))
    w_edge = float(max(getattr(cfg, "conviction_weight_edge", 0.30), 0.0))
    w_conf = float(max(getattr(cfg, "conviction_weight_confidence", 0.30), 0.0))
    w_unc = float(max(getattr(cfg, "conviction_weight_uncertainty", 0.25), 0.0))
    w_meta = float(max(getattr(cfg, "conviction_weight_meta", 0.15), 0.0))
    w_sum = max(w_edge + w_conf + w_unc + w_meta, 1e-6)
    score = (w_edge * edge_n + w_conf * conf_n + w_unc * unc_n + w_meta * meta_n) / w_sum
    return float(np.clip(score, 0.0, 1.0))


def _recent_quality_stats(
    rr_hist: List[float],
    *,
    window: int,
    min_trades: int,
) -> Dict[str, float]:
    w = int(max(window, 1))
    m = int(max(min_trades, 1))
    recent = rr_hist[-w:] if rr_hist else []
    n = int(len(recent))
    if n == 0:
        return {"ready": 0.0, "n": 0.0, "hit_rate": 0.0, "expectancy": 0.0}
    arr = np.asarray(recent, dtype=np.float64)
    hit_rate = float(np.mean(arr > 0.0))
    expectancy = float(np.mean(arr))
    ready = float(n >= m)
    return {"ready": ready, "n": float(n), "hit_rate": hit_rate, "expectancy": expectancy}


def _bayes_bucket(prior_alpha: float, prior_beta: float) -> Dict[str, float]:
    return {
        "alpha": float(max(prior_alpha, 1e-6)),
        "beta": float(max(prior_beta, 1e-6)),
        "n": 0.0,
        "sum_r": 0.0,
    }


def _update_bayes_quality_state(
    *,
    side: int,
    regime: int,
    realized_r: float,
    side_stats: Dict[int, Dict[str, float]],
    regime_side_stats: Dict[Tuple[int, int], Dict[str, float]],
    cfg: MythosConfig,
) -> None:
    s = int(side)
    if s not in (-1, 1):
        return
    prior_alpha = float(np.clip(getattr(cfg, "bayes_quality_prior_alpha", 2.0), 0.10, 100.0))
    prior_beta = float(np.clip(getattr(cfg, "bayes_quality_prior_beta", 2.0), 0.10, 100.0))
    decay = float(np.clip(getattr(cfg, "bayes_quality_decay", 0.995), 0.90, 1.0))
    hit = 1.0 if float(realized_r) > 0.0 else 0.0

    def _touch(bucket: Dict[str, float]) -> None:
        alpha = float(bucket.get("alpha", prior_alpha))
        beta = float(bucket.get("beta", prior_beta))
        n = float(bucket.get("n", 0.0))
        s_r = float(bucket.get("sum_r", 0.0))
        alpha = prior_alpha + (alpha - prior_alpha) * decay
        beta = prior_beta + (beta - prior_beta) * decay
        n *= decay
        s_r *= decay
        alpha += hit
        beta += (1.0 - hit)
        n += 1.0
        s_r += float(realized_r)
        bucket["alpha"] = float(alpha)
        bucket["beta"] = float(beta)
        bucket["n"] = float(n)
        bucket["sum_r"] = float(s_r)

    s_bucket = side_stats.setdefault(s, _bayes_bucket(prior_alpha, prior_beta))
    r_bucket = regime_side_stats.setdefault((int(regime), s), _bayes_bucket(prior_alpha, prior_beta))
    _touch(s_bucket)
    _touch(r_bucket)


def _bayes_quality_gate(
    *,
    side: int,
    regime: int,
    edge: float,
    confidence: float,
    uncertainty: float,
    total_trades: int,
    side_stats: Dict[int, Dict[str, float]],
    regime_side_stats: Dict[Tuple[int, int], Dict[str, float]],
    cfg: MythosConfig,
) -> Dict[str, float]:
    if not bool(getattr(cfg, "bayes_quality_enable", True)):
        return {"pass": 1.0, "ready": 0.0}
    s = int(side)
    if s not in (-1, 1):
        return {"pass": 1.0, "ready": 0.0}
    prior_alpha = float(np.clip(getattr(cfg, "bayes_quality_prior_alpha", 2.0), 0.10, 100.0))
    prior_beta = float(np.clip(getattr(cfg, "bayes_quality_prior_beta", 2.0), 0.10, 100.0))
    s_bucket = side_stats.setdefault(s, _bayes_bucket(prior_alpha, prior_beta))
    r_bucket = regime_side_stats.setdefault((int(regime), s), _bayes_bucket(prior_alpha, prior_beta))
    s_n = float(max(s_bucket.get("n", 0.0), 0.0))
    r_n = float(max(r_bucket.get("n", 0.0), 0.0))
    warmup = int(max(getattr(cfg, "bayes_quality_warmup_trades", 20), 1))
    if int(total_trades) < warmup or (s_n + r_n) < max(float(warmup) * 0.6, 4.0):
        return {"pass": 1.0, "ready": 0.0}

    def _stats(bucket: Dict[str, float]) -> Tuple[float, float]:
        a = float(max(bucket.get("alpha", prior_alpha), 1e-6))
        b = float(max(bucket.get("beta", prior_beta), 1e-6))
        n = float(max(bucket.get("n", 0.0), 1e-6))
        p = float(a / max(a + b, 1e-6))
        e = float(bucket.get("sum_r", 0.0) / n)
        return p, e

    s_p, s_e = _stats(s_bucket)
    r_p, r_e = _stats(r_bucket)
    reg_w = float(np.clip(getattr(cfg, "bayes_quality_regime_weight", 0.45), 0.0, 1.0))
    p_hist = float((1.0 - reg_w) * s_p + reg_w * r_p)
    e_hist = float((1.0 - reg_w) * s_e + reg_w * r_e)
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    edge_n = float(np.clip(max(edge, 0.0) / max(edge_unit * 2.5, 1e-6), 0.0, 1.0))
    conf_n = float(np.clip(confidence, 0.0, 1.0))
    unc_n = float(np.clip(uncertainty, 0.0, 2.0) / 2.0)
    p_adj = float(
        p_hist
        + float(np.clip(getattr(cfg, "bayes_quality_edge_scale", 0.22), 0.0, 2.0)) * edge_n
        + float(np.clip(getattr(cfg, "bayes_quality_confidence_scale", 0.10), 0.0, 1.0)) * max(conf_n - 0.5, 0.0)
        - float(np.clip(getattr(cfg, "bayes_quality_uncertainty_scale", 0.18), 0.0, 2.0)) * unc_n
    )
    p_adj = float(np.clip(p_adj, 0.0, 1.0))
    e_adj = float(e_hist + (0.30 * edge_n + 0.10 * (conf_n - 0.5) - 0.20 * unc_n) * edge_unit)
    min_p = float(np.clip(getattr(cfg, "bayes_quality_min_win_prob", 0.50), 0.0, 1.0))
    min_e = float(getattr(cfg, "bayes_quality_min_expectancy", -0.01))
    margin = float(np.clip(getattr(cfg, "bayes_quality_reject_margin", 0.05), 0.0, 0.5))
    bad_prob = bool(p_adj < (min_p - margin))
    bad_exp = bool(e_adj < (min_e - margin * edge_unit))
    allowed = not (bad_prob and bad_exp)
    return {
        "pass": float(1.0 if allowed else 0.0),
        "ready": 1.0,
        "p_adj": float(p_adj),
        "e_adj": float(e_adj),
    }


def _nonconformity_score(
    *,
    edge: float,
    confidence: float,
    uncertainty: float,
    meta_p: float,
    analog_hits: float,
    cfg: MythosConfig,
) -> float:
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    edge_n = float(np.clip(max(edge, 0.0) / max(edge_unit * 2.5, 1e-6), 0.0, 1.0))
    conf_n = float(np.clip(confidence, 0.0, 1.0))
    unc_n = float(np.clip(uncertainty, 0.0, 2.0) / 2.0)
    meta_n = float(np.clip(abs(float(meta_p) - 0.5) * 2.0, 0.0, 1.0))
    analog_n = float(
        np.clip(float(analog_hits) / max(float(getattr(cfg, "analog_k", 48)), 1.0), 0.0, 1.0)
    )
    w_unc = float(max(getattr(cfg, "nonconformity_weight_uncertainty", 0.36), 0.0))
    w_conf = float(max(getattr(cfg, "nonconformity_weight_confidence", 0.22), 0.0))
    w_edge = float(max(getattr(cfg, "nonconformity_weight_edge", 0.20), 0.0))
    w_meta = float(max(getattr(cfg, "nonconformity_weight_meta", 0.14), 0.0))
    w_analog = float(max(getattr(cfg, "nonconformity_weight_analog", 0.08), 0.0))
    denom = float(max(w_unc + w_conf + w_edge + w_meta + w_analog, 1e-6))
    raw = (
        w_unc * unc_n
        + w_conf * (1.0 - conf_n)
        + w_edge * (1.0 - edge_n)
        + w_meta * (1.0 - meta_n)
        + w_analog * (1.0 - analog_n)
    )
    return float(np.clip(raw / denom, 0.0, 1.0))


def _nonconformity_gate(
    *,
    score: float,
    conviction: float,
    edge: float,
    confidence: float,
    total_trades: int,
    winner_scores: List[float],
    ready_checks: int = 0,
    rejects: int = 0,
    cfg: MythosConfig,
) -> Dict[str, float]:
    if not bool(getattr(cfg, "nonconformity_enable", True)):
        return {"pass": 1.0, "ready": 0.0, "override": 0.0}
    warmup = int(max(getattr(cfg, "nonconformity_warmup_trades", 24), 1))
    min_winners = int(max(getattr(cfg, "nonconformity_min_winners", 16), 1))
    if int(total_trades) < warmup or len(winner_scores) < min_winners:
        return {"pass": 1.0, "ready": 0.0, "override": 0.0}
    q = float(np.clip(getattr(cfg, "nonconformity_quantile", 0.86), 0.50, 0.99))
    margin = float(np.clip(getattr(cfg, "nonconformity_margin", 0.03), 0.0, 1.0))
    w = int(max(getattr(cfg, "nonconformity_window", 160), 8))
    sample = np.asarray(winner_scores[-w:], dtype=np.float64)
    threshold = float(np.clip(np.quantile(sample, q) + margin, 0.0, 1.0))
    target_reject = float(np.clip(getattr(cfg, "nonconformity_target_reject_rate", 0.48), 0.0, 0.99))
    tol = float(np.clip(getattr(cfg, "nonconformity_reject_tolerance", 0.12), 0.0, 0.5))
    relax_gain = float(np.clip(getattr(cfg, "nonconformity_adaptive_relax", 0.16), 0.0, 1.0))
    max_relax = float(np.clip(getattr(cfg, "nonconformity_adaptive_max_relax", 0.18), 0.0, 0.5))
    observed_reject = float(rejects / max(ready_checks, 1))
    overshoot = float(max(observed_reject - (target_reject + tol), 0.0))
    adaptive_relax = float(np.clip(relax_gain * overshoot, 0.0, max_relax))
    threshold = float(np.clip(threshold + adaptive_relax, 0.0, 1.0))
    passed = bool(float(score) <= threshold)
    override = False
    if not passed:
        conv_floor = float(np.clip(getattr(cfg, "nonconformity_override_conviction", 0.88), 0.0, 1.0))
        edge_floor = float(getattr(cfg, "min_edge_threshold", 0.02)) + float(
            max(getattr(cfg, "nonconformity_override_edge_buffer", 0.003), 0.0)
        )
        conf_floor = float(np.clip(getattr(cfg, "min_confidence", 0.55), 0.0, 1.0)) + float(
            np.clip(getattr(cfg, "nonconformity_override_confidence_buffer", 0.04), 0.0, 1.0)
        )
        soft_margin = float(np.clip(getattr(cfg, "nonconformity_soft_override_margin", 0.04), 0.0, 0.5))
        near_threshold = bool(float(score) <= float(np.clip(threshold + soft_margin, 0.0, 1.0)))
        soft_ok = (
            near_threshold
            and float(conviction) >= max(conv_floor - 0.06, 0.0)
            and float(edge) >= max(edge_floor - 0.001, 0.0)
            and float(confidence) >= min(max(conf_floor - 0.03, 0.0), 1.0)
        )
        if soft_ok:
            passed = True
            override = True
        if (
            (not passed)
            and float(conviction) >= conv_floor
            and float(edge) >= edge_floor
            and float(confidence) >= min(conf_floor, 1.0)
        ):
            passed = True
            override = True
    return {
        "pass": float(1.0 if passed else 0.0),
        "ready": 1.0,
        "threshold": float(threshold),
        "adaptive_relax": float(adaptive_relax),
        "override": float(1.0 if override else 0.0),
    }


def _adaptive_rebalance_adjustment(
    *,
    side: int,
    long_trades: List[float],
    short_trades: List[float],
    cfg: MythosConfig,
) -> Dict[str, float]:
    s = int(side)
    if s not in (-1, 1):
        return {"edge_adjust": 0.0, "conf_adjust": 0.0}
    if not bool(getattr(cfg, "side_rebalance_enable", True)):
        return {"edge_adjust": 0.0, "conf_adjust": 0.0}
    warmup = int(max(getattr(cfg, "side_rebalance_warmup_trades", 40), 1))
    total = int(len(long_trades) + len(short_trades))
    if total < warmup:
        return {"edge_adjust": 0.0, "conf_adjust": 0.0}
    w = int(max(getattr(cfg, "side_rebalance_window", 96), 8))
    long_recent = list(long_trades[-w:]) if long_trades else []
    short_recent = list(short_trades[-w:]) if short_trades else []
    total_recent = int(len(long_recent) + len(short_recent))
    if total_recent < max(8, int(0.5 * w)):
        return {"edge_adjust": 0.0, "conf_adjust": 0.0}
    short_frac = float(len(short_recent) / max(total_recent, 1))
    short_target = float(np.clip(getattr(cfg, "side_rebalance_short_target", 0.32), 0.05, 0.50))
    gap = float(short_target - short_frac)
    if abs(gap) <= 1e-9:
        return {"edge_adjust": 0.0, "conf_adjust": 0.0}
    quality_guard = float(np.clip(getattr(cfg, "side_rebalance_quality_guard", 0.06), 0.0, 0.50))
    long_exp = float(np.mean(np.asarray(long_recent, dtype=np.float64))) if long_recent else 0.0
    short_exp = float(np.mean(np.asarray(short_recent, dtype=np.float64))) if short_recent else 0.0
    max_adj = float(np.clip(getattr(cfg, "side_rebalance_max_adjust", 0.012), 0.0, 0.10))
    short_boost = float(np.clip(getattr(cfg, "side_rebalance_short_boost", 0.0035), 0.0, 0.05))
    long_pen = float(np.clip(getattr(cfg, "side_rebalance_long_penalty", 0.0030), 0.0, 0.05))
    conf_boost = float(np.clip(getattr(cfg, "side_rebalance_conf_boost", 0.02), 0.0, 0.20))
    if s == -1 and gap > 0.0:
        # Boost shorts only when short side is underrepresented and not clearly degraded.
        if short_exp + quality_guard < long_exp:
            return {"edge_adjust": 0.0, "conf_adjust": 0.0}
        scale = float(min(gap / max(short_target, 1e-6), 1.0))
        return {
            "edge_adjust": float(np.clip(short_boost * (0.5 + scale), 0.0, max_adj)),
            "conf_adjust": float(np.clip(conf_boost * (0.5 + scale), 0.0, 0.25)),
        }
    if s == 1 and gap > 0.0:
        # Penalize longs a bit when short allocation is too low.
        scale = float(min(gap / max(short_target, 1e-6), 1.0))
        return {"edge_adjust": float(-np.clip(long_pen * scale, 0.0, max_adj)), "conf_adjust": 0.0}
    return {"edge_adjust": 0.0, "conf_adjust": 0.0}


def _intelligence_bucket() -> Dict[str, float]:
    return {"n": 0.0, "hit_ema": 0.5, "exp_ema": 0.0, "var_ema": 0.0}


def _update_intelligence_state(
    *,
    side: int,
    regime: int,
    expert_name: str,
    realized_r: float,
    side_stats: Dict[int, Dict[str, float]],
    regime_side_stats: Dict[Tuple[int, int], Dict[str, float]],
    expert_stats: Dict[str, Dict[str, float]],
    cfg: MythosConfig,
) -> None:
    s = int(side)
    if s not in (-1, 1):
        return
    alpha = float(np.clip(getattr(cfg, "intelligence_ema_alpha", 0.08), 0.01, 1.0))
    rr = float(realized_r)
    hit = 1.0 if rr > 0.0 else 0.0

    def _touch(bucket: Dict[str, float]) -> None:
        n = float(max(bucket.get("n", 0.0), 0.0) + 1.0)
        hit_ema = float(bucket.get("hit_ema", 0.5))
        exp_ema = float(bucket.get("exp_ema", 0.0))
        var_ema = float(max(bucket.get("var_ema", 0.0), 0.0))
        hit_ema = (1.0 - alpha) * hit_ema + alpha * hit
        exp_ema = (1.0 - alpha) * exp_ema + alpha * rr
        resid = rr - exp_ema
        var_ema = (1.0 - alpha) * var_ema + alpha * (resid * resid)
        bucket["n"] = n
        bucket["hit_ema"] = float(np.clip(hit_ema, 0.0, 1.0))
        bucket["exp_ema"] = exp_ema
        bucket["var_ema"] = float(max(var_ema, 0.0))

    _touch(side_stats.setdefault(s, _intelligence_bucket()))
    _touch(regime_side_stats.setdefault((int(regime), s), _intelligence_bucket()))
    if expert_name:
        _touch(expert_stats.setdefault(str(expert_name), _intelligence_bucket()))


def _intelligence_bucket_score(bucket: Dict[str, float], cfg: MythosConfig) -> float:
    min_n = int(max(getattr(cfg, "intelligence_min_samples", 24), 1))
    n = int(max(bucket.get("n", 0.0), 0.0))
    if n < min_n:
        return 0.0
    hit_w = float(np.clip(getattr(cfg, "intelligence_hit_weight", 0.55), 0.0, 2.0))
    exp_w = float(np.clip(getattr(cfg, "intelligence_expectancy_weight", 0.45), 0.0, 2.0))
    var_w = float(np.clip(getattr(cfg, "intelligence_variance_penalty", 0.18), 0.0, 2.0))
    unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    hit_term = float((float(bucket.get("hit_ema", 0.5)) - 0.5) * 2.0)
    exp_term = float(np.tanh(float(bucket.get("exp_ema", 0.0)) / max(2.0 * unit, 1e-6)))
    var_term = float(np.tanh(float(np.sqrt(max(bucket.get("var_ema", 0.0), 0.0))) / max(3.0 * unit, 1e-6)))
    score = hit_w * hit_term + exp_w * exp_term - var_w * var_term
    return float(np.clip(score, -2.0, 2.0))


def _apply_intelligence_adjustment(
    *,
    side: int,
    regime: int,
    expert_name: str,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    analog_edge: float,
    side_stats: Dict[int, Dict[str, float]],
    regime_side_stats: Dict[Tuple[int, int], Dict[str, float]],
    expert_stats: Dict[str, Dict[str, float]],
    cfg: MythosConfig,
) -> Dict[str, float]:
    s = int(side)
    if not bool(getattr(cfg, "intelligence_enable", True)) or s not in (-1, 1):
        return {
            "side": float(s),
            "edge": float(edge),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "uncertainty": float(np.clip(uncertainty, 0.005, 2.0)),
            "score": 0.0,
            "switched": 0.0,
        }
    side_score = _intelligence_bucket_score(side_stats.setdefault(s, _intelligence_bucket()), cfg)
    reg_score = _intelligence_bucket_score(regime_side_stats.setdefault((int(regime), s), _intelligence_bucket()), cfg)
    exp_score = _intelligence_bucket_score(expert_stats.setdefault(str(expert_name), _intelligence_bucket()), cfg)
    score = float(0.45 * reg_score + 0.35 * side_score + 0.20 * exp_score)

    max_adj = float(np.clip(getattr(cfg, "intelligence_max_edge_adjust", 0.020), 0.0, 0.50))
    pos_scale = float(np.clip(getattr(cfg, "intelligence_edge_scale", 0.010), 0.0, 0.20))
    neg_scale = float(np.clip(getattr(cfg, "intelligence_negative_edge_scale", 0.012), 0.0, 0.20))
    conf_scale = float(np.clip(getattr(cfg, "intelligence_conf_scale", 0.06), 0.0, 0.50))
    unc_scale = float(np.clip(getattr(cfg, "intelligence_uncertainty_scale", 0.30), 0.0, 1.50))

    if score >= 0.0:
        edge_adj = float(np.clip(score * pos_scale, 0.0, max_adj))
    else:
        edge_adj = float(-np.clip(abs(score) * neg_scale, 0.0, max_adj))
    edge2 = float(max(edge + edge_adj, 0.0))
    conf2 = float(np.clip(confidence + score * conf_scale, 0.0, 1.0))
    if score >= 0.0:
        unc_mult = float(max(1.0 - unc_scale * min(score, 1.0), 0.5))
    else:
        unc_mult = float(1.0 + unc_scale * min(abs(score), 1.0))
    unc2 = float(np.clip(uncertainty * unc_mult, 0.005, 2.0))

    switched = False
    if bool(getattr(cfg, "intelligence_side_switch_enable", True)):
        other = -s
        other_side = _intelligence_bucket_score(side_stats.setdefault(other, _intelligence_bucket()), cfg)
        other_reg = _intelligence_bucket_score(regime_side_stats.setdefault((int(regime), other), _intelligence_bucket()), cfg)
        other_score = float(0.55 * other_reg + 0.45 * other_side)
        gap = float(other_score - score)
        min_gap = float(np.clip(getattr(cfg, "intelligence_side_switch_min_gap", 0.30), 0.0, 2.0))
        min_adv = float(np.clip(getattr(cfg, "intelligence_side_switch_min_analog_adv", 0.0015), 0.0, 0.50))
        conv_guard = float(np.clip(getattr(cfg, "intelligence_side_switch_conviction_guard", 0.58), 0.0, 1.0))
        # Switch only when live conviction is weak and historical evidence is clearly better opposite side.
        if float(conviction) < conv_guard and gap >= min_gap and float(analog_edge) < min_adv:
            s = other
            switched = True
            edge2 = float(max(edge2 + 0.25 * min(max_adj, pos_scale * min(gap, 1.0)), 0.0))
            conf2 = float(np.clip(conf2 + 0.03, 0.0, 1.0))
            unc2 = float(np.clip(unc2 * 1.05, 0.005, 2.0))

    return {
        "side": float(s),
        "edge": edge2,
        "confidence": conf2,
        "uncertainty": unc2,
        "score": score,
        "switched": float(1.0 if switched else 0.0),
    }


def _can_intelligence_switch_side(
    *,
    bar_idx: int,
    switched_so_far: int,
    intelligence_mode_bars: int,
    last_switch_bar: int,
    cfg: MythosConfig,
) -> bool:
    if not bool(getattr(cfg, "intelligence_side_switch_enable", True)):
        return False
    cooldown = int(max(getattr(cfg, "intelligence_side_switch_cooldown_bars", 96), 1))
    if int(bar_idx) - int(last_switch_bar) < cooldown:
        return False
    warmup = int(max(getattr(cfg, "intelligence_side_switch_warmup_bars", 256), 1))
    if int(bar_idx) < warmup:
        return False
    if int(intelligence_mode_bars) <= 0:
        return True
    max_rate = float(np.clip(getattr(cfg, "intelligence_side_switch_max_rate", 0.10), 0.0, 1.0))
    obs_rate = float(switched_so_far / max(intelligence_mode_bars, 1))
    return bool(obs_rate <= max_rate)


def _is_sure_signal(
    *,
    side: int,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    meta_p: float,
    meta_ready: bool,
    analog_hits: float,
    sure_recent_rr: List[float],
    cfg: MythosConfig,
) -> bool:
    _ = uncertainty
    if int(side) == 0:
        return False
    high_conv = float(np.clip(getattr(cfg, "precision_high_conviction", 0.72), 0.0, 1.0))
    if float(conviction) < high_conv:
        return False
    if bool(getattr(cfg, "conviction_requires_meta_ready", True)) and not bool(meta_ready):
        return False
    meta_strength = float(np.clip(abs(float(meta_p) - 0.5) * 2.0, 0.0, 1.0))
    if meta_strength < float(np.clip(getattr(cfg, "sure_meta_strength_min", 0.10), 0.0, 1.0)):
        return False
    edge_floor = float(getattr(cfg, "min_edge_threshold", 0.02)) + float(
        max(getattr(cfg, "sure_edge_buffer", 0.001), 0.0)
    )
    if float(edge) < edge_floor:
        return False
    conf_floor = float(np.clip(getattr(cfg, "min_confidence", 0.55), 0.0, 1.0)) + float(
        np.clip(getattr(cfg, "sure_confidence_buffer", 0.03), 0.0, 0.5)
    )
    if float(confidence) < min(conf_floor, 1.0):
        return False
    analog_floor_abs = int(max(getattr(cfg, "sure_min_analog_hits", 12), 0))
    analog_floor_ratio = float(np.clip(getattr(cfg, "sure_min_analog_ratio", 0.20), 0.0, 1.0))
    analog_floor = max(analog_floor_abs, int(np.ceil(float(getattr(cfg, "analog_k", 48)) * analog_floor_ratio)))
    if float(analog_hits) < float(max(analog_floor, 0)):
        return False

    stats = _recent_quality_stats(
        sure_recent_rr,
        window=int(max(getattr(cfg, "sure_recent_window", 96), 8)),
        min_trades=int(max(getattr(cfg, "sure_recent_min_trades", 24), 1)),
    )
    hit_floor = float(np.clip(getattr(cfg, "sure_recent_min_hit_rate", 0.52), 0.0, 1.0))
    exp_floor = float(getattr(cfg, "sure_recent_min_expectancy", 0.04))
    if bool(stats["ready"]):
        hit_gap = float(max(hit_floor - float(stats["hit_rate"]), 0.0))
        exp_gap = float(max(exp_floor - float(stats["expectancy"]), 0.0))
        # Avoid binary lockout after a weak streak: increase required conviction/meta
        # instead of permanently disabling sure classification.
        exp_scale = max(abs(exp_floor), 1e-6)
        conv_penalty = 0.45 * hit_gap + 0.25 * min(exp_gap / exp_scale, 1.0)
        meta_penalty = 0.50 * hit_gap + 0.20 * min(exp_gap / exp_scale, 1.0)
        dyn_conv = float(np.clip(high_conv + conv_penalty, 0.0, 1.0))
        dyn_meta = float(
            np.clip(getattr(cfg, "sure_meta_strength_min", 0.10) + meta_penalty, 0.0, 1.0)
        )
        if float(conviction) < dyn_conv:
            return False
        if float(meta_strength) < dyn_meta:
            return False
    else:
        cold_conv = high_conv + float(np.clip(getattr(cfg, "sure_cold_start_conviction_extra", 0.04), 0.0, 0.5))
        cold_meta = float(np.clip(getattr(cfg, "sure_meta_strength_min", 0.10), 0.0, 1.0)) + float(
            np.clip(getattr(cfg, "sure_cold_start_meta_extra", 0.06), 0.0, 0.5)
        )
        if float(conviction) < min(cold_conv, 1.0):
            return False
        if meta_strength < min(cold_meta, 1.0):
            return False
    return True


def _allow_conviction_leverage(
    *,
    is_sure_signal: bool,
    edge: float,
    confidence: float,
    conviction: float,
    context_score: float,
    leveraged_recent_rr: List[float],
    leveraged_recent_ctx: List[float],
    cfg: MythosConfig,
) -> bool:
    if not bool(is_sure_signal):
        return False
    score_thr = float(np.clip(getattr(cfg, "conviction_score_threshold", 0.62), 0.0, 1.0))
    conv_floor = score_thr + float(np.clip(getattr(cfg, "leverage_conviction_buffer", 0.04), 0.0, 0.5))
    if float(conviction) < min(conv_floor, 1.0):
        return False
    edge_floor = float(getattr(cfg, "min_edge_threshold", 0.02)) + float(
        max(getattr(cfg, "leverage_edge_buffer", 0.003), 0.0)
    )
    if float(edge) < edge_floor:
        return False
    conf_floor = float(np.clip(getattr(cfg, "min_confidence", 0.55), 0.0, 1.0)) + float(
        np.clip(getattr(cfg, "leverage_confidence_buffer", 0.04), 0.0, 0.5)
    )
    if float(confidence) < min(conf_floor, 1.0):
        return False

    stats = _recent_quality_stats(
        leveraged_recent_rr,
        window=int(max(getattr(cfg, "leverage_recent_window", 120), 8)),
        min_trades=int(max(getattr(cfg, "leverage_recent_min_trades", 20), 1)),
    )
    hit_floor = float(np.clip(getattr(cfg, "leverage_recent_min_hit_rate", 0.53), 0.0, 1.0))
    exp_floor = float(getattr(cfg, "leverage_recent_min_expectancy", 0.05))
    if bool(stats["ready"]):
        if float(stats["hit_rate"]) < hit_floor or float(stats["expectancy"]) < exp_floor:
            return False
    else:
        # Cold start: require exceptionally strong conviction before any boost.
        if float(conviction) < min(conv_floor + 0.05, 1.0):
            return False
    policy_stats = _recent_quality_stats(
        leveraged_recent_rr,
        window=int(max(getattr(cfg, "leverage_policy_window", 160), 8)),
        min_trades=int(max(getattr(cfg, "leverage_policy_min_trades", 24), 1)),
    )
    policy_hit_floor = float(np.clip(getattr(cfg, "leverage_policy_min_hit_rate", 0.54), 0.0, 1.0))
    policy_exp_floor = float(getattr(cfg, "leverage_policy_min_expectancy", 0.06))
    ctx_w = float(np.clip(getattr(cfg, "leverage_policy_context_weight", 0.60), 0.0, 1.0))
    p_window = int(max(getattr(cfg, "leverage_policy_window", 160), 8))
    recent_ctx = leveraged_recent_ctx[-p_window:] if leveraged_recent_ctx else []
    hist_ctx = float(np.mean(np.asarray(recent_ctx, dtype=np.float64))) if recent_ctx else 0.5
    blended_hit = float((1.0 - ctx_w) * float(policy_stats["hit_rate"]) + ctx_w * float(context_score))
    blended_exp = float((1.0 - ctx_w) * float(policy_stats["expectancy"]) + ctx_w * max(hist_ctx - 0.5, 0.0))
    if bool(policy_stats["ready"]):
        if blended_hit < policy_hit_floor or blended_exp < policy_exp_floor:
            return False
    else:
        cold_extra = float(
            np.clip(getattr(cfg, "leverage_policy_cold_start_conviction_extra", 0.08), 0.0, 0.5)
        )
        if float(conviction) < min(conv_floor + cold_extra, 1.0):
            return False
        if float(context_score) < max(policy_hit_floor - 0.08, 0.50):
            return False
    return True


def _net_edge_ok(
    *,
    edge: float,
    uncertainty: float,
    cfg: MythosConfig,
) -> bool:
    cost_r = _estimate_execution_cost_r(edge=edge, uncertainty=uncertainty, cfg=cfg)
    net_edge = float(edge - cost_r)
    floor = float(max(getattr(cfg, "leverage_net_edge_floor", 0.002), 0.0))
    return bool(net_edge >= floor)


def _side_policy_ok(
    *,
    side: int,
    side_rr_hist: Dict[int, List[float]],
    cfg: MythosConfig,
) -> bool:
    if not bool(getattr(cfg, "leverage_side_policy_enable", True)):
        return True
    s = int(side)
    if s not in (-1, 1):
        return False
    hist = list(side_rr_hist.get(s, []))
    min_n = int(max(getattr(cfg, "leverage_side_min_trades", 12), 1))
    if len(hist) < min_n:
        return True
    arr = np.asarray(hist[-max(min_n, 64):], dtype=np.float64)
    hit_rate = float(np.mean(arr > 0.0)) if arr.size else 0.0
    expectancy = float(np.mean(arr)) if arr.size else 0.0
    min_hit = float(np.clip(getattr(cfg, "leverage_side_min_hit_rate", 0.52), 0.0, 1.0))
    min_exp = float(getattr(cfg, "leverage_side_min_expectancy", 0.03))
    return bool(hit_rate >= min_hit and expectancy >= min_exp)


def _estimate_execution_cost_r(
    *,
    edge: float,
    uncertainty: float,
    cfg: MythosConfig,
) -> float:
    unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    fee_bps = float(max(getattr(cfg, "execution_fee_bps", 4.0), 0.0))
    slippage_bps = float(max(getattr(cfg, "execution_slippage_bps", 2.0), 0.0))
    raw_frac = float((fee_bps + slippage_bps) / 10000.0)
    base_r = float(raw_frac / unit)
    unc_mult = float(1.0 + 0.35 * np.clip(uncertainty, 0.0, 2.0))
    edge_relief = float(1.0 - 0.15 * np.clip(edge / max(2.0 * unit, 1e-6), 0.0, 1.0))
    cost_r = float(base_r * unc_mult * edge_relief)
    cap_r = float(np.clip(getattr(cfg, "execution_cost_cap_r", 0.35), 0.0, 5.0))
    return float(np.clip(cost_r, 0.0, cap_r))


def _leverage_context_score(
    *,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    meta_p: float,
    analog_hits: float,
    cfg: MythosConfig,
) -> float:
    unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    edge_n = float(np.clip(max(edge, 0.0) / max(unit * 2.5, 1e-6), 0.0, 1.0))
    conf_n = float(np.clip(confidence, 0.0, 1.0))
    unc_n = float(np.clip(1.0 / (1.0 + max(uncertainty, 0.0)), 0.0, 1.0))
    conv_n = float(np.clip(conviction, 0.0, 1.0))
    meta_n = float(np.clip(abs(meta_p - 0.5) * 2.0, 0.0, 1.0))
    analog_n = float(
        np.clip(float(analog_hits) / max(float(getattr(cfg, "analog_k", 48)), 1.0), 0.0, 1.0)
    )
    score = (
        0.24 * conv_n
        + 0.20 * edge_n
        + 0.16 * conf_n
        + 0.10 * unc_n
        + 0.15 * meta_n
        + 0.15 * analog_n
    )
    return float(np.clip(score, 0.0, 1.0))


def _counterfactual_pass(
    analog_mem: AnalogMemory,
    x: np.ndarray,
    side: int,
    edge: float,
    uncertainty: float,
    cfg: MythosConfig,
) -> bool:
    if side == 0:
        return False
    choose = analog_mem.query(x=x, side=side)
    alt_side = -1 if side == 1 else 1
    alt = analog_mem.query(x=x, side=alt_side)
    min_adv = float(max(getattr(cfg, "counterfactual_min_advantage_r", 0.006), 0.0))
    risk_pen = float(max(getattr(cfg, "counterfactual_risk_penalty", 0.6), 0.0))
    margin = float(max(getattr(cfg, "counterfactual_margin", 0.006), 0.0))
    unc_w = float(np.clip(getattr(cfg, "counterfactual_uncertainty_weight", 0.50), 0.0, 2.0))
    min_alt_hits = int(max(getattr(cfg, "counterfactual_min_alt_hits", 8), 0))
    choose_hits = float(choose.get("analog_hits", 0.0))
    alt_hits = float(alt.get("analog_hits", 0.0))
    # Keep uncertainty drag proportional to edge/min-adv scale to avoid over-pruning.
    unc = float(np.clip(uncertainty, 0.0, 2.0))
    adjusted_edge = float(edge * (1.0 - np.clip(risk_pen * unc, 0.0, 0.95)))
    unc_drag = float(unc_w * unc * max(min_adv, 1e-6))
    if choose_hits < min_alt_hits or alt_hits < min_alt_hits:
        # If analog support is sparse, rely on live edge instead of hard rejecting.
        return adjusted_edge >= (0.5 * min_adv)
    choose_score = float(
        choose["analog_edge"] + margin * (float(choose.get("analog_conf", 0.5)) - 0.5)
    )
    alt_score = float(
        alt["analog_edge"]
        + margin * (float(alt.get("analog_conf", 0.5)) - 0.5)
        + unc_drag
    )
    analog_adv = float(choose_score - alt_score)
    combo = float(adjusted_edge + analog_adv)
    # Allow strong live-edge trades to pass unless counterfactual evidence is decisively negative.
    if adjusted_edge >= (0.75 * min_adv) and analog_adv >= (-0.5 * min_adv):
        return True
    return combo >= min_adv


def _adaptive_counterfactual_pass(
    *,
    analog_mem: AnalogMemory,
    x: np.ndarray,
    side: int,
    edge: float,
    uncertainty: float,
    cfg: MythosConfig,
    accepted_trades: int,
    cf_rejects: int,
) -> bool:
    target = float(np.clip(getattr(cfg, "counterfactual_target_reject_rate", 0.70), 0.0, 0.99))
    tol = float(np.clip(getattr(cfg, "counterfactual_reject_tolerance", 0.10), 0.0, 0.5))
    relax_gain = float(np.clip(getattr(cfg, "counterfactual_adaptive_relax", 0.35), 0.0, 1.0))
    min_floor = float(np.clip(getattr(cfg, "counterfactual_adaptive_min_adv_floor", 0.25), 0.05, 1.0))
    obs = float(cf_rejects / max(cf_rejects + accepted_trades, 1))
    overshoot = float(max(obs - (target + tol), 0.0))
    if overshoot <= 0.0:
        return _counterfactual_pass(
            analog_mem=analog_mem,
            x=x,
            side=side,
            edge=edge,
            uncertainty=uncertainty,
            cfg=cfg,
        )
    orig_min_adv = float(max(getattr(cfg, "counterfactual_min_advantage_r", 0.006), 0.0))
    relax = float(np.clip(relax_gain * overshoot, 0.0, 0.95))
    adj_min_adv = float(max(orig_min_adv * (1.0 - relax), orig_min_adv * min_floor))
    adj_margin = float(max(getattr(cfg, "counterfactual_margin", 0.006), 0.0) * (1.0 - 0.6 * relax))
    adj_risk_pen = float(max(getattr(cfg, "counterfactual_risk_penalty", 0.6), 0.0) * (1.0 - 0.5 * relax))
    orig = (
        getattr(cfg, "counterfactual_min_advantage_r", orig_min_adv),
        getattr(cfg, "counterfactual_margin", 0.006),
        getattr(cfg, "counterfactual_risk_penalty", 0.6),
    )
    try:
        setattr(cfg, "counterfactual_min_advantage_r", adj_min_adv)
        setattr(cfg, "counterfactual_margin", adj_margin)
        setattr(cfg, "counterfactual_risk_penalty", adj_risk_pen)
        return _counterfactual_pass(
            analog_mem=analog_mem,
            x=x,
            side=side,
            edge=edge,
            uncertainty=uncertainty,
            cfg=cfg,
        )
    finally:
        setattr(cfg, "counterfactual_min_advantage_r", orig[0])
        setattr(cfg, "counterfactual_margin", orig[1])
        setattr(cfg, "counterfactual_risk_penalty", orig[2])


def _adaptive_nonconformity_gate(
    *,
    score: float,
    conviction: float,
    edge: float,
    confidence: float,
    total_trades: int,
    winner_scores: List[float],
    nonconformity_rejects: int,
    cfg: MythosConfig,
) -> Dict[str, float]:
    target = float(np.clip(getattr(cfg, "nonconformity_target_reject_rate", 0.48), 0.0, 0.99))
    tol = float(np.clip(getattr(cfg, "nonconformity_reject_tolerance", 0.12), 0.0, 0.5))
    relax_gain = float(np.clip(getattr(cfg, "nonconformity_adaptive_relax", 0.16), 0.0, 1.0))
    relax_cap = float(np.clip(getattr(cfg, "nonconformity_adaptive_max_relax", 0.18), 0.0, 0.5))
    obs = float(nonconformity_rejects / max(nonconformity_rejects + total_trades, 1))
    overshoot = float(max(obs - (target + tol), 0.0))
    gate = _nonconformity_gate(
        score=score,
        conviction=conviction,
        edge=edge,
        confidence=confidence,
        total_trades=total_trades,
        winner_scores=winner_scores,
        cfg=cfg,
    )
    if overshoot <= 0.0:
        return gate
    relax = float(np.clip(relax_gain * overshoot, 0.0, relax_cap))
    # Relax threshold first, then allow a soft override when signal is close.
    if float(gate.get("ready", 0.0)) > 0.5:
        threshold = float(np.clip(gate.get("threshold", 0.5) + relax, 0.0, 1.0))
        if float(score) <= threshold:
            return {"pass": 1.0, "ready": 1.0, "threshold": threshold, "override": 1.0}
        soft_margin = float(np.clip(getattr(cfg, "nonconformity_soft_override_margin", 0.04), 0.0, 0.5))
        if float(score) <= (threshold + soft_margin * relax):
            conv_floor = float(np.clip(getattr(cfg, "nonconformity_override_conviction", 0.88), 0.0, 1.0))
            edge_floor = float(getattr(cfg, "min_edge_threshold", 0.02)) + float(
                max(getattr(cfg, "nonconformity_override_edge_buffer", 0.003), 0.0)
            )
            conf_floor = float(np.clip(getattr(cfg, "min_confidence", 0.55), 0.0, 1.0)) + float(
                np.clip(getattr(cfg, "nonconformity_override_confidence_buffer", 0.04), 0.0, 1.0)
            )
            if (
                float(conviction) >= max(conv_floor - relax, 0.0)
                and float(edge) >= max(edge_floor - relax * 0.01, 0.0)
                and float(confidence) >= min(max(conf_floor - 0.2 * relax, 0.0), 1.0)
            ):
                return {"pass": 1.0, "ready": 1.0, "threshold": threshold, "override": 1.0}
    return gate


def _short_aggression_boost(
    side: int,
    long_recent: List[float],
    short_recent: List[float],
    cfg: MythosConfig,
) -> float:
    if int(side) != -1:
        return 0.0
    min_n = int(max(getattr(cfg, "short_aggr_min_trades", 12), 2))
    if len(short_recent) < min_n:
        return 0.0
    if len(long_recent) < min_n:
        return 0.0
    long_exp = float(np.mean(np.array(long_recent, dtype=np.float64)))
    short_exp = float(np.mean(np.array(short_recent, dtype=np.float64)))
    edge = short_exp - long_exp
    trig = float(max(getattr(cfg, "short_aggr_expectancy_trigger", 0.04), 0.0))
    if edge <= trig:
        return 0.0
    gain = float(max(getattr(cfg, "short_aggr_edge_boost", 0.0035), 0.0))
    cap = float(max(getattr(cfg, "short_aggr_max_boost", 0.03), 0.0))
    scale = min((edge - trig) / max(trig, 1e-6), 1.0)
    return float(min(gain * (1.0 + scale), cap))


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

    experts = build_experts(cfg.random_state, cfg=cfg)
    xcols = [c for c in MYTHOS_STATE_COLS if c in train_feat.columns]
    X_tr = train_feat[xcols].to_numpy(dtype=np.float64)
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
    meta = NeuralMetaLearner(cfg=cfg, n_features=X_tr.shape[1])
    meta_bootstrap_n = int(max(getattr(cfg, "meta_bootstrap_trades", 0), 0))
    if meta_bootstrap_n > 0:
        usable = min(meta_bootstrap_n, max(len(train_feat) - cfg.horizon - 1, 0))
        if usable > 0:
            close_tr = train_feat["close"].to_numpy(dtype=np.float64)
            high_tr = train_feat["high"].to_numpy(dtype=np.float64)
            low_tr = train_feat["low"].to_numpy(dtype=np.float64)
            atr_tr = _compute_atr(close_tr, high_tr, low_tr, period=14)
            for j in range(usable):
                s = 1 if float(train_feat.iloc[j]["fwd_ret_4"]) >= 0.0 else -1
                rr = _barrier_outcome(
                    close=close_tr,
                    high=high_tr,
                    low=low_tr,
                    idx=j,
                    side=s,
                    tp_mult=cfg.tp_mult,
                    sl_mult=cfg.sl_mult,
                    horizon=cfg.horizon,
                    atr=atr_tr,
                )
                meta.update(
                    x=X_tr[j],
                    edge=float(abs(train_feat.iloc[j]["fwd_ret_4"])),
                    confidence=float(np.clip(0.5 + 0.5 * np.tanh(abs(train_feat.iloc[j]["fwd_ret_4"]) / 0.01), 0.0, 1.0)),
                    uncertainty=float(np.clip(train_feat.iloc[j]["vol_16"] * 2.0, 0.01, 2.0)),
                    regime=int(train_regime[j]),
                    side=int(s),
                    realized_r=float(rr),
                )

    risk = RiskConstitution(cfg)
    close = test_feat["close"].to_numpy(dtype=np.float64)
    high = test_feat["high"].to_numpy(dtype=np.float64)
    low = test_feat["low"].to_numpy(dtype=np.float64)
    timestamps = test_feat["timestamp"].to_numpy(dtype=np.int64)
    atr = _compute_atr(close, high, low, period=14)
    X_te = test_feat[xcols].to_numpy(dtype=np.float64)

    trades: List[float] = []
    gross_trades: List[float] = []
    skip_counts = {
        "low_confidence": 0,
        "edge_below_floor": 0,
        "streak_pause": 0,
        "counterfactual_reject": 0,
        "bayes_quality_reject": 0,
        "nonconformity_reject": 0,
        "risk_reject": 0,
        "meta_reject": 0,
        "low_conviction": 0,
    }
    cf_rejects = 0
    n_long = 0
    n_short = 0
    change_mode_bars = 0
    flip_pressure_bars = 0
    transition_mode_bars = 0
    meta_mode_bars = 0
    recent_trade_rr: List[float] = []
    long_trades: List[float] = []
    short_trades: List[float] = []
    high_conv_trades = 0
    high_conv_wins = 0
    high_conv_r_sum = 0.0
    sure_trades = 0
    sure_hits = 0
    sure_total_r = 0.0
    leveraged_trades = 0
    leveraged_hits = 0
    leveraged_total_r = 0.0
    sure_leveraged_trades = 0
    sure_leveraged_hits = 0
    sure_leveraged_total_r = 0.0
    leverage_blocked_candidates = 0
    leverage_boost_approved = 0
    execution_cost_total_r = 0.0
    bayes_quality_rejects = 0
    bayes_quality_ready_checks = 0
    nonconformity_rejects = 0
    nonconformity_overrides = 0
    nonconformity_ready_checks = 0
    intelligence_active_bars = 0
    intelligence_switched_sides = 0
    intelligence_score_acc = 0.0
    sure_recent_rr: List[float] = []
    sure_lev_recent_rr: List[float] = []
    sure_lev_recent_ctx: List[float] = []
    winner_nonconformity_scores: List[float] = []
    bayes_side_stats: Dict[int, Dict[str, float]] = {
        1: _bayes_bucket(
            float(np.clip(getattr(cfg, "bayes_quality_prior_alpha", 2.0), 0.10, 100.0)),
            float(np.clip(getattr(cfg, "bayes_quality_prior_beta", 2.0), 0.10, 100.0)),
        ),
        -1: _bayes_bucket(
            float(np.clip(getattr(cfg, "bayes_quality_prior_alpha", 2.0), 0.10, 100.0)),
            float(np.clip(getattr(cfg, "bayes_quality_prior_beta", 2.0), 0.10, 100.0)),
        ),
    }
    bayes_regime_side_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    intelligence_side_stats: Dict[int, Dict[str, float]] = {1: _intelligence_bucket(), -1: _intelligence_bucket()}
    intelligence_regime_side_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    intelligence_expert_stats: Dict[str, Dict[str, float]] = {}
    side_quality_rr: Dict[int, List[float]] = {1: [], -1: []}
    intelligence_score_sum = 0.0
    intelligence_side_switches = 0
    intelligence_mode_bars = 0
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
            bar_idx=i,
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
        if float(adapted.get("flip_pressure", 0.0)) > 0.5:
            flip_pressure_bars += 1
        if abs(float(adapted.get("transition_score", 0.0))) > 1e-9:
            transition_mode_bars += 1
        meta_p = meta.score(
            x=x,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            regime=regime,
            side=side,
        )
        meta_gain = float(np.clip(getattr(cfg, "meta_learner_edge_blend", 0.30), 0.0, 2.0))
        meta_conf_gain = float(np.clip(getattr(cfg, "meta_learner_conf_blend", 0.20), 0.0, 1.0))
        meta_unc_pen = float(np.clip(getattr(cfg, "meta_learner_uncertainty_penalty", 0.80), 0.0, 2.0))
        meta_min_side_prob = float(np.clip(getattr(cfg, "meta_learner_min_side_prob", 0.50), 0.0, 1.0))
        meta_ready_floor = float(np.clip(getattr(cfg, "meta_learner_ready_prob_floor", 0.42), 0.0, 1.0))
        meta_ready_ceiling = float(np.clip(getattr(cfg, "meta_learner_ready_prob_ceiling", 0.58), 0.0, 1.0))
        meta_warmup = int(max(getattr(cfg, "meta_learner_warmup_samples", 1024), 0))
        meta_buffer_n = int(getattr(meta, "_buffer_x", []) and len(getattr(meta, "_buffer_x", [])) or 0)
        runtime_warmup = int(meta_warmup)
        if hasattr(meta, "_effective_min_samples"):
            try:
                # Keep minimum warmup sufficiently high to avoid unstable early-fold
                # leverage/sure gating from under-trained meta outputs.
                adaptive_floor = int(max(int(meta._effective_min_samples()) * 2, 128))
                runtime_warmup = int(max(min(runtime_warmup, adaptive_floor), 128))
            except Exception:
                runtime_warmup = int(meta_warmup)
        meta_ready = (
            bool(meta.is_ready())
            and meta_buffer_n >= runtime_warmup
            and (meta_p <= meta_ready_floor or meta_p >= meta_ready_ceiling)
        )
        if abs(meta_p - 0.5) > 1e-9:
            meta_mode_bars += 1
        if side != 0 and meta_ready and meta_p < meta_min_side_prob:
            skip_counts["meta_reject"] += 1
            continue
        signed = float((meta_p - 0.5) * 2.0)
        edge = float(edge * (1.0 + meta_gain * signed))
        confidence = float(np.clip(confidence + meta_conf_gain * signed, 0.0, 1.0))
        uncertainty = float(np.clip(uncertainty * (1.0 + meta_unc_pen * max(0.5 - meta_p, 0.0)), 0.005, 2.0))
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        high_conv_thresh = float(np.clip(getattr(cfg, "precision_high_conviction", 0.72), 0.0, 1.0))
        analog_hits = float(analog.get("analog_hits", 0.0))
        intel = _apply_intelligence_adjustment(
            side=side,
            regime=regime,
            expert_name=expert_name,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            conviction=conviction,
            analog_edge=float(analog.get("analog_edge", 0.0)),
            side_stats=intelligence_side_stats,
            regime_side_stats=intelligence_regime_side_stats,
            expert_stats=intelligence_expert_stats,
            cfg=cfg,
        )
        side = int(round(float(intel.get("side", side))))
        edge = float(intel.get("edge", edge))
        confidence = float(np.clip(intel.get("confidence", confidence), 0.0, 1.0))
        uncertainty = float(np.clip(intel.get("uncertainty", uncertainty), 0.005, 2.0))
        intel_score = float(intel.get("score", 0.0))
        intelligence_score_sum += intel_score
        if abs(intel_score) > 1e-9:
            intelligence_mode_bars += 1
        if float(intel.get("switched", 0.0)) > 0.5:
            intelligence_side_switches += 1
        is_sure_signal = _is_sure_signal(
            side=side,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            conviction=conviction,
            meta_p=meta_p,
            meta_ready=meta_ready,
            analog_hits=analog_hits,
            sure_recent_rr=sure_recent_rr,
            cfg=cfg,
        )
        sure_recent_window = int(max(getattr(cfg, "sure_recent_window", 96), 8))
        min_conv = float(np.clip(getattr(cfg, "precision_min_conviction", 0.52), 0.0, 1.0))
        if side != 0 and conviction < min_conv:
            skip_counts["low_conviction"] += 1
            continue
        edge_floor = governor.adjusted_edge_floor(risk.state.equity_r)
        edge -= governor.side_penalty(side)
        if confidence < cfg.min_confidence:
            skip_counts["low_confidence"] += 1
            continue
        if edge < edge_floor:
            skip_counts["edge_below_floor"] += 1
            continue
        if not governor.allow_by_streak(i, side=side):
            skip_counts["streak_pause"] += 1
            continue
        if not _adaptive_counterfactual_pass(
            analog_mem=analog_mem,
            x=x,
            side=side,
            edge=edge,
            uncertainty=uncertainty,
            cfg=cfg,
            accepted_trades=len(trades),
            cf_rejects=cf_rejects,
        ):
            cf_rejects += 1
            skip_counts["counterfactual_reject"] += 1
            continue
        bayes_gate = _bayes_quality_gate(
            side=side,
            regime=regime,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            total_trades=len(trades),
            side_stats=bayes_side_stats,
            regime_side_stats=bayes_regime_side_stats,
            cfg=cfg,
        )
        if float(bayes_gate.get("ready", 0.0)) > 0.5:
            bayes_quality_ready_checks += 1
            if float(bayes_gate.get("pass", 1.0)) < 0.5:
                bayes_quality_rejects += 1
                skip_counts["bayes_quality_reject"] += 1
                continue
        nonconf_score = _nonconformity_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            analog_hits=analog_hits,
            cfg=cfg,
        )
        nonconf_gate = _adaptive_nonconformity_gate(
            score=nonconf_score,
            conviction=conviction,
            edge=edge,
            confidence=confidence,
            total_trades=len(trades),
            winner_scores=winner_nonconformity_scores,
            nonconformity_rejects=nonconformity_rejects,
            cfg=cfg,
        )
        if float(nonconf_gate.get("ready", 0.0)) > 0.5:
            nonconformity_ready_checks += 1
            if float(nonconf_gate.get("pass", 1.0)) < 0.5:
                nonconformity_rejects += 1
                skip_counts["nonconformity_reject"] += 1
                continue
            if float(nonconf_gate.get("override", 0.0)) > 0.5:
                nonconformity_overrides += 1
        if not risk.allow_trade(ts_ms=int(timestamps[i]), side=side, edge=edge, uncertainty=uncertainty):
            skip_counts["risk_reject"] += 1
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
        base_size = risk.position_size_multiplier(
            edge=edge,
            uncertainty=uncertainty,
            regime=regime,
            conviction=conviction,
            allow_conviction_boost=False,
        )
        context_score = _leverage_context_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            conviction=conviction,
            meta_p=meta_p,
            analog_hits=analog_hits,
            cfg=cfg,
        )
        lev_recent_window = int(max(getattr(cfg, "leverage_recent_window", 120), 8))
        lev_policy_window = int(max(getattr(cfg, "leverage_policy_window", 160), 8))
        leverage_allowed = _allow_conviction_leverage(
            is_sure_signal=is_sure_signal,
            edge=edge,
            confidence=confidence,
            conviction=conviction,
            context_score=context_score,
            leveraged_recent_rr=sure_lev_recent_rr,
            leveraged_recent_ctx=sure_lev_recent_ctx,
            cfg=cfg,
        )
        if risk.should_disable_leverage():
            leverage_allowed = False
        net_edge_est = float(edge - _estimate_execution_cost_r(edge=edge, uncertainty=uncertainty, cfg=cfg))
        if net_edge_est < float(max(getattr(cfg, "leverage_net_edge_floor", 0.002), 0.0)):
            leverage_allowed = False
        if leverage_allowed and not _side_policy_ok(side=side, side_rr_hist=side_quality_rr, cfg=cfg):
            leverage_allowed = False
        if is_sure_signal and not leverage_allowed:
            leverage_blocked_candidates += 1
        if leverage_allowed:
            leverage_boost_approved += 1
        size = risk.position_size_multiplier(
            edge=edge,
            uncertainty=uncertainty,
            regime=regime,
            conviction=conviction,
            allow_conviction_boost=leverage_allowed,
        )
        leveraged = bool(size > (base_size + 1e-9))
        rr_gross = float(realized * size)
        execution_cost_r = _estimate_execution_cost_r(
            edge=edge,
            uncertainty=uncertainty,
            cfg=cfg,
        )
        rr = float(rr_gross - execution_cost_r)
        execution_cost_total_r += float(execution_cost_r)
        risk.record_trade(rr, int(timestamps[i]), edge=edge, uncertainty=uncertainty, conviction=conviction)
        governor.record_trade(side=side, realized_r=rr, bar_idx=i)
        router.update_reliability(expert_name=expert_name, realized_r=rr, regime=regime)
        adaptive.update_after_trade(expert_name=expert_name, regime=regime, realized_r=rr, side=side)
        _update_bayes_quality_state(
            side=side,
            regime=regime,
            realized_r=rr,
            side_stats=bayes_side_stats,
            regime_side_stats=bayes_regime_side_stats,
            cfg=cfg,
        )
        _update_intelligence_state(
            side=side,
            regime=regime,
            expert_name=expert_name,
            realized_r=rr,
            side_stats=intelligence_side_stats,
            regime_side_stats=intelligence_regime_side_stats,
            expert_stats=intelligence_expert_stats,
            cfg=cfg,
        )
        meta.update(
            x=x,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            regime=regime,
            side=side,
            realized_r=rr,
        )
        trades.append(rr)
        gross_trades.append(rr_gross)
        if rr > 0.0:
            winner_nonconformity_scores.append(float(nonconf_score))
            nc_window = int(max(getattr(cfg, "nonconformity_window", 160), 8))
            if len(winner_nonconformity_scores) > nc_window:
                del winner_nonconformity_scores[0 : len(winner_nonconformity_scores) - nc_window]
        if is_sure_signal:
            sure_trades += 1
            sure_total_r += rr
            sure_recent_rr.append(rr)
            if len(sure_recent_rr) > sure_recent_window:
                del sure_recent_rr[0 : len(sure_recent_rr) - sure_recent_window]
            if rr > 0.0:
                sure_hits += 1
        if leveraged:
            leveraged_trades += 1
            leveraged_total_r += rr
            if rr > 0.0:
                leveraged_hits += 1
        if is_sure_signal and leveraged:
            sure_leveraged_trades += 1
            sure_leveraged_total_r += rr
            sure_lev_recent_rr.append(rr)
            if len(sure_lev_recent_rr) > lev_recent_window:
                del sure_lev_recent_rr[0 : len(sure_lev_recent_rr) - lev_recent_window]
            sure_lev_recent_ctx.append(float(context_score))
            if len(sure_lev_recent_ctx) > lev_policy_window:
                del sure_lev_recent_ctx[0 : len(sure_lev_recent_ctx) - lev_policy_window]
            if rr > 0.0:
                sure_leveraged_hits += 1
        recent_trade_rr.append(rr)
        max_recent = int(max(getattr(cfg, "transition_min_samples", 6) * 4, 16))
        if len(recent_trade_rr) > max_recent:
            del recent_trade_rr[0 : len(recent_trade_rr) - max_recent]
        if side == 1:
            n_long += 1
            long_trades.append(rr)
        else:
            n_short += 1
            short_trades.append(rr)
        side_hist = side_quality_rr.setdefault(int(side), [])
        side_hist.append(rr)
        if len(side_hist) > 128:
            del side_hist[0 : len(side_hist) - 128]
        if conviction >= high_conv_thresh:
            high_conv_trades += 1
            high_conv_r_sum += rr
            if rr > 0.0:
                high_conv_wins += 1

    n = len(trades)
    wins = int(np.sum(np.array(trades) > 0.0)) if n else 0
    losses = int(np.sum(np.array(trades) < 0.0)) if n else 0
    gross_profit = float(np.sum(np.array([t for t in trades if t > 0.0], dtype=np.float64))) if n else 0.0
    gross_loss = float(-np.sum(np.array([t for t in trades if t < 0.0], dtype=np.float64))) if n else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0.0 else 0.0)
    total_r = float(np.sum(trades)) if n else 0.0
    gross_total_r = float(np.sum(gross_trades)) if n else 0.0
    expect = float(np.mean(trades)) if n else 0.0
    wr = float(np.mean(np.array(trades) > 0)) if n else 0.0
    long_n = len(long_trades)
    short_n = len(short_trades)
    long_wr = float(np.mean(np.array(long_trades) > 0.0)) if long_n else 0.0
    short_wr = float(np.mean(np.array(short_trades) > 0.0)) if short_n else 0.0
    long_r_sum = float(np.sum(long_trades)) if long_n else 0.0
    short_r_sum = float(np.sum(short_trades)) if short_n else 0.0
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
        "flip_pressure_bars": int(flip_pressure_bars),
        "transition_mode_bars": int(transition_mode_bars),
        "meta_mode_bars": int(meta_mode_bars),
        "counterfactual_rejects": int(cf_rejects),
        "bayes_quality_rejects": int(bayes_quality_rejects),
        "nonconformity_rejects": int(nonconformity_rejects),
        "nonconformity_overrides": int(nonconformity_overrides),
        "bayes_quality_ready_checks": int(bayes_quality_ready_checks),
        "nonconformity_ready_checks": int(nonconformity_ready_checks),
        "nonconformity_winner_ref_count": int(len(winner_nonconformity_scores)),
        "intelligence_mode_bars": int(intelligence_mode_bars),
        "intelligence_side_switches": int(intelligence_side_switches),
        "intelligence_avg_score": round(float(intelligence_score_sum / max(len(trades), 1)), 6),
        "skip_reasons": {k: int(v) for k, v in skip_counts.items()},
        "long_trades": n_long,
        "short_trades": n_short,
        "long_win_rate": round(long_wr, 4),
        "short_win_rate": round(short_wr, 4),
        "long_total_r": round(long_r_sum, 4),
        "short_total_r": round(short_r_sum, 4),
        "high_conviction_trades": int(high_conv_trades),
        "high_conviction_win_rate": round(float(high_conv_wins / max(high_conv_trades, 1)), 4),
        "high_conviction_total_r": round(float(high_conv_r_sum), 4),
        "gross_total_r": round(float(gross_total_r), 4),
        "net_total_r": round(float(total_r), 4),
        "execution_cost_total_r": round(float(execution_cost_total_r), 4),
        "net_expectancy_r": round(float(expect), 4),
        "net_win_rate": round(float(wr), 4),
        "sure_trades": int(sure_trades),
        "sure_hits": int(sure_hits),
        "sure_win_rate": round(float(sure_hits / max(sure_trades, 1)), 4),
        "sure_total_r": round(float(sure_total_r), 4),
        "leveraged_trades": int(leveraged_trades),
        "leveraged_hits": int(leveraged_hits),
        "leveraged_hit_rate": round(float(leveraged_hits / max(leveraged_trades, 1)), 4),
        "leveraged_total_r": round(float(leveraged_total_r), 4),
        "sure_leveraged_trades": int(sure_leveraged_trades),
        "sure_leveraged_hits": int(sure_leveraged_hits),
        "sure_leveraged_hit_rate": round(float(sure_leveraged_hits / max(sure_leveraged_trades, 1)), 4),
        "sure_leveraged_total_r": round(float(sure_leveraged_total_r), 4),
        "net_sure_leveraged_hit_rate": round(float(sure_leveraged_hits / max(sure_leveraged_trades, 1)), 4),
        "leverage_boost_approved": int(leverage_boost_approved),
        "leverage_blocked_candidates": int(leverage_blocked_candidates),
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
            "[MYTHOS][FOLD %d] trades=%d totalR=%+.2f status=%s long/short=%d/%d sure=%d sure_win=%s sure_lev=%d sure_lev_hits=%d lev_boost=%d lev_blocked=%d bayes_rej=%d nonconf_rej=%d nonconf_ovr=%d intel_mode=%d intel_switch=%d intel_avg=%s",
            i,
            fold["total_trades"],
            fold["total_r"],
            fold["status"],
            fold["long_trades"],
            fold["short_trades"],
            fold.get("sure_trades", 0),
            fold.get("sure_win_rate", 0.0),
            fold.get("sure_leveraged_trades", 0),
            fold.get("sure_leveraged_hits", 0),
            fold.get("leverage_boost_approved", 0),
            fold.get("leverage_blocked_candidates", 0),
            fold.get("bayes_quality_rejects", 0),
            fold.get("nonconformity_rejects", 0),
            fold.get("nonconformity_overrides", 0),
            fold.get("intelligence_mode_bars", 0),
            fold.get("intelligence_side_switches", 0),
            fold.get("intelligence_avg_score", 0.0),
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
    total_flip_pressure_bars = int(sum(r.get("flip_pressure_bars", 0) for r in reports))
    total_cf_rejects = int(sum(r.get("counterfactual_rejects", 0) for r in reports))
    total_skip_reasons = {
        "low_confidence": int(sum(int(r.get("skip_reasons", {}).get("low_confidence", 0)) for r in reports)),
        "edge_below_floor": int(sum(int(r.get("skip_reasons", {}).get("edge_below_floor", 0)) for r in reports)),
        "streak_pause": int(sum(int(r.get("skip_reasons", {}).get("streak_pause", 0)) for r in reports)),
        "counterfactual_reject": int(sum(int(r.get("skip_reasons", {}).get("counterfactual_reject", 0)) for r in reports)),
        "bayes_quality_reject": int(sum(int(r.get("skip_reasons", {}).get("bayes_quality_reject", 0)) for r in reports)),
        "nonconformity_reject": int(sum(int(r.get("skip_reasons", {}).get("nonconformity_reject", 0)) for r in reports)),
        "risk_reject": int(sum(int(r.get("skip_reasons", {}).get("risk_reject", 0)) for r in reports)),
        "meta_reject": int(sum(int(r.get("skip_reasons", {}).get("meta_reject", 0)) for r in reports)),
        "low_conviction": int(sum(int(r.get("skip_reasons", {}).get("low_conviction", 0)) for r in reports)),
    }
    change_mode_rate = float(total_change_mode_bars / max(total_trades, 1))
    total_transition_mode_bars = int(sum(r.get("transition_mode_bars", 0) for r in reports))
    total_meta_mode_bars = int(sum(r.get("meta_mode_bars", 0) for r in reports))
    total_long_trades = int(sum(r.get("long_trades", 0) for r in reports))
    total_short_trades = int(sum(r.get("short_trades", 0) for r in reports))
    total_long_wins = int(
        sum(int(round(float(r.get("long_win_rate", 0.0)) * int(r.get("long_trades", 0)))) for r in reports)
    )
    total_short_wins = int(
        sum(int(round(float(r.get("short_win_rate", 0.0)) * int(r.get("short_trades", 0)))) for r in reports)
    )
    total_long_r = float(sum(float(r.get("long_total_r", 0.0)) for r in reports))
    total_short_r = float(sum(float(r.get("short_total_r", 0.0)) for r in reports))
    total_high_conv_trades = int(sum(int(r.get("high_conviction_trades", 0)) for r in reports))
    total_high_conv_r = float(sum(float(r.get("high_conviction_total_r", 0.0)) for r in reports))
    total_high_conv_wins = int(
        sum(
            int(
                round(
                    float(r.get("high_conviction_win_rate", 0.0))
                    * int(r.get("high_conviction_trades", 0))
                )
            )
            for r in reports
        )
    )
    total_sure_trades = int(sum(int(r.get("sure_trades", 0)) for r in reports))
    total_sure_hits = int(sum(int(r.get("sure_hits", 0)) for r in reports))
    total_sure_r = float(sum(float(r.get("sure_total_r", 0.0)) for r in reports))
    total_leveraged_trades = int(sum(int(r.get("leveraged_trades", 0)) for r in reports))
    total_leveraged_hits = int(sum(int(r.get("leveraged_hits", 0)) for r in reports))
    total_leveraged_r = float(sum(float(r.get("leveraged_total_r", 0.0)) for r in reports))
    total_sure_lev_trades = int(sum(int(r.get("sure_leveraged_trades", 0)) for r in reports))
    total_sure_lev_hits = int(sum(int(r.get("sure_leveraged_hits", 0)) for r in reports))
    total_sure_lev_r = float(sum(float(r.get("sure_leveraged_total_r", 0.0)) for r in reports))
    total_execution_cost_r = float(sum(float(r.get("execution_cost_total_r", 0.0)) for r in reports))
    total_gross_r = float(sum(float(r.get("gross_total_r", r.get("total_r", 0.0))) for r in reports))
    total_lev_boost_approved = int(sum(int(r.get("leverage_boost_approved", 0)) for r in reports))
    total_lev_blocked = int(sum(int(r.get("leverage_blocked_candidates", 0)) for r in reports))
    total_bayes_quality_rejects = int(sum(int(r.get("bayes_quality_rejects", 0)) for r in reports))
    total_nonconformity_rejects = int(sum(int(r.get("nonconformity_rejects", 0)) for r in reports))
    total_nonconformity_overrides = int(sum(int(r.get("nonconformity_overrides", 0)) for r in reports))
    total_bayes_quality_ready_checks = int(sum(int(r.get("bayes_quality_ready_checks", 0)) for r in reports))
    total_nonconformity_ready_checks = int(sum(int(r.get("nonconformity_ready_checks", 0)) for r in reports))
    total_nonconformity_winner_ref_count = int(
        sum(int(r.get("nonconformity_winner_ref_count", 0)) for r in reports)
    )
    total_intelligence_mode_bars = int(sum(int(r.get("intelligence_mode_bars", 0)) for r in reports))
    total_intelligence_side_switches = int(sum(int(r.get("intelligence_side_switches", 0)) for r in reports))
    weighted_intelligence_score_num = float(
        sum(float(r.get("intelligence_avg_score", 0.0)) * int(r.get("total_trades", 0)) for r in reports)
    )
    aggregate_intelligence_avg_score = float(weighted_intelligence_score_num / max(total_trades, 1))
    cf_reject_rate = float(total_cf_rejects / max(total_cf_rejects + total_trades, 1))
    bayes_quality_reject_rate = float(
        total_bayes_quality_rejects / max(total_bayes_quality_rejects + total_trades, 1)
    )
    nonconformity_reject_rate = float(
        total_nonconformity_rejects / max(total_nonconformity_rejects + total_trades, 1)
    )
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
        "transition_mode_bars": total_transition_mode_bars,
        "transition_mode_rate": round(float(total_transition_mode_bars / max(total_trades, 1)), 4),
        "meta_mode_bars": total_meta_mode_bars,
        "meta_mode_rate": round(float(total_meta_mode_bars / max(total_trades, 1)), 4),
        "long_trades": total_long_trades,
        "short_trades": total_short_trades,
        "long_win_rate": round(float(total_long_wins / max(total_long_trades, 1)), 4),
        "short_win_rate": round(float(total_short_wins / max(total_short_trades, 1)), 4),
        "long_total_r": round(total_long_r, 4),
        "short_total_r": round(total_short_r, 4),
        "high_conviction_trades": total_high_conv_trades,
        "high_conviction_win_rate": round(float(total_high_conv_wins / max(total_high_conv_trades, 1)), 4),
        "high_conviction_total_r": round(total_high_conv_r, 4),
        "gross_total_r": round(total_gross_r, 4),
        "net_total_r": round(total_r, 4),
        "execution_cost_total_r": round(total_execution_cost_r, 4),
        "net_expectancy_r": round((total_r / total_trades) if total_trades else 0.0, 4),
        "net_win_rate": round(agg_win_rate, 4),
        "sure_trades": total_sure_trades,
        "sure_hits": total_sure_hits,
        "sure_win_rate": round(float(total_sure_hits / max(total_sure_trades, 1)), 4),
        "sure_total_r": round(total_sure_r, 4),
        "leveraged_trades": total_leveraged_trades,
        "leveraged_hits": total_leveraged_hits,
        "leveraged_hit_rate": round(float(total_leveraged_hits / max(total_leveraged_trades, 1)), 4),
        "leveraged_total_r": round(total_leveraged_r, 4),
        "sure_leveraged_trades": total_sure_lev_trades,
        "sure_leveraged_hits": total_sure_lev_hits,
        "sure_leveraged_hit_rate": round(float(total_sure_lev_hits / max(total_sure_lev_trades, 1)), 4),
        "sure_leveraged_total_r": round(total_sure_lev_r, 4),
        "net_sure_leveraged_hit_rate": round(float(total_sure_lev_hits / max(total_sure_lev_trades, 1)), 4),
        "leverage_boost_approved": total_lev_boost_approved,
        "leverage_blocked_candidates": total_lev_blocked,
        "flip_pressure_bars": total_flip_pressure_bars,
        "flip_pressure_rate": round(float(total_flip_pressure_bars / max(total_trades, 1)), 4),
        "counterfactual_rejects": total_cf_rejects,
        "counterfactual_reject_rate": round(cf_reject_rate, 4),
        "bayes_quality_rejects": total_bayes_quality_rejects,
        "bayes_quality_reject_rate": round(bayes_quality_reject_rate, 4),
        "bayes_quality_ready_checks": total_bayes_quality_ready_checks,
        "nonconformity_rejects": total_nonconformity_rejects,
        "nonconformity_reject_rate": round(nonconformity_reject_rate, 4),
        "nonconformity_overrides": total_nonconformity_overrides,
        "nonconformity_ready_checks": total_nonconformity_ready_checks,
        "nonconformity_winner_ref_count": total_nonconformity_winner_ref_count,
        "intelligence_mode_bars": total_intelligence_mode_bars,
        "intelligence_mode_rate": round(float(total_intelligence_mode_bars / max(total_trades, 1)), 4),
        "intelligence_side_switches": total_intelligence_side_switches,
        "intelligence_avg_score": round(aggregate_intelligence_avg_score, 6),
        "skip_reasons": total_skip_reasons,
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
