from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
from math import comb, erf, isfinite, sqrt
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


def _recent_calibration_stats(
    confidence_hist: List[float],
    outcome_hist: List[float],
    rr_hist: List[float],
    *,
    window: int,
    min_samples: int,
) -> Dict[str, float]:
    w = int(max(window, 1))
    m = int(max(min_samples, 1))
    n = int(min(len(confidence_hist), len(outcome_hist), len(rr_hist)))
    if n <= 0:
        return {
            "ready": 0.0,
            "n": 0.0,
            "abs_error": 0.0,
            "brier": 0.0,
            "hit_rate": 0.0,
            "expectancy": 0.0,
            "avg_confidence": 0.0,
            "confidence_gap": 0.0,
        }
    k = int(min(w, n))
    conf = np.asarray(confidence_hist[-k:], dtype=np.float64)
    out = np.asarray(outcome_hist[-k:], dtype=np.float64)
    rr = np.asarray(rr_hist[-k:], dtype=np.float64)
    conf = np.clip(conf, 0.0, 1.0)
    out = np.clip(out, 0.0, 1.0)
    err = conf - out
    abs_error = float(np.mean(np.abs(err))) if err.size else 0.0
    brier = float(np.mean(err ** 2)) if err.size else 0.0
    hit_rate = float(np.mean(out > 0.5)) if out.size else 0.0
    expectancy = float(np.mean(rr)) if rr.size else 0.0
    avg_conf = float(np.mean(conf)) if conf.size else 0.0
    return {
        "ready": float(k >= m),
        "n": float(k),
        "abs_error": abs_error,
        "brier": brier,
        "hit_rate": hit_rate,
        "expectancy": expectancy,
        "avg_confidence": avg_conf,
        "confidence_gap": float(avg_conf - hit_rate),
    }


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
    readiness = 1.0
    if n < min_n:
        cold_min = int(max(min_n // 4, 4))
        if n < cold_min:
            return 0.0
        # Sparse-data warmup: allow a scaled score instead of hard lockout.
        readiness = float(np.clip(n / max(min_n, 1), 0.0, 1.0))
    hit_w = float(np.clip(getattr(cfg, "intelligence_hit_weight", 0.55), 0.0, 2.0))
    exp_w = float(np.clip(getattr(cfg, "intelligence_expectancy_weight", 0.45), 0.0, 2.0))
    var_w = float(np.clip(getattr(cfg, "intelligence_variance_penalty", 0.18), 0.0, 2.0))
    unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    hit_term = float((float(bucket.get("hit_ema", 0.5)) - 0.5) * 2.0)
    exp_term = float(np.tanh(float(bucket.get("exp_ema", 0.0)) / max(2.0 * unit, 1e-6)))
    var_term = float(np.tanh(float(np.sqrt(max(bucket.get("var_ema", 0.0), 0.0))) / max(3.0 * unit, 1e-6)))
    score = (hit_w * hit_term + exp_w * exp_term - var_w * var_term) * readiness
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
    bar_idx: int = 0,
    switched_so_far: int = 0,
    intelligence_mode_bars: int = 0,
    last_switch_bar: int = -10_000_000,
    instability: float = 0.0,
    flip_pressure: float = 0.0,
    change_mode: float = 0.0,
    long_trade_count: int = 0,
    short_trade_count: int = 0,
    stress_mode: bool = False,
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
        own_bucket = side_stats.setdefault(s, _intelligence_bucket())
        other_bucket = side_stats.setdefault(other, _intelligence_bucket())
        own_reg_bucket = regime_side_stats.setdefault((int(regime), s), _intelligence_bucket())
        other_reg_bucket = regime_side_stats.setdefault((int(regime), other), _intelligence_bucket())
        other_side = _intelligence_bucket_score(other_bucket, cfg)
        other_reg = _intelligence_bucket_score(other_reg_bucket, cfg)
        other_score = float(0.55 * other_reg + 0.45 * other_side)
        gap = float(other_score - score)
        min_gap = float(np.clip(getattr(cfg, "intelligence_side_switch_min_gap", 0.30), 0.0, 2.0))
        min_adv = float(np.clip(getattr(cfg, "intelligence_side_switch_min_analog_adv", 0.0015), 0.0, 0.50))
        conv_guard = float(np.clip(getattr(cfg, "intelligence_side_switch_conviction_guard", 0.58), 0.0, 1.0))
        can_switch = _can_intelligence_switch_side(
            bar_idx=int(bar_idx),
            switched_so_far=int(switched_so_far),
            intelligence_mode_bars=int(intelligence_mode_bars),
            last_switch_bar=int(last_switch_bar),
            cfg=cfg,
        )
        if (not can_switch) and bool(stress_mode):
            stress_warmup = int(max(round(max(getattr(cfg, "intelligence_side_switch_warmup_trades", 40), 1) * 0.4), 12))
            can_switch = bool(intelligence_mode_bars >= stress_warmup)
        min_side_samples = int(max(getattr(cfg, "intelligence_side_switch_min_samples", 48), 1))
        total_side = int(max(int(long_trade_count) + int(short_trade_count), 0))
        short_share = float(int(short_trade_count) / max(total_side, 1)) if total_side > 0 else 0.5
        switch_into_underweight = bool(
            (other == -1 and short_share < 0.36) or (other == 1 and short_share > 0.64)
        )
        if switch_into_underweight and total_side >= 24:
            min_side_samples = int(max(12, round(min_side_samples * 0.60)))
            min_gap = float(max(min_gap * 0.70, 0.0))
            conv_guard = float(min(conv_guard + 0.08, 0.92))
        if bool(stress_mode) and total_side >= 20:
            min_side_samples = int(max(10, round(min_side_samples * 0.65)))
            min_gap = float(max(min_gap * 0.75, 0.0))
            conv_guard = float(min(conv_guard + 0.10, 0.95))
        own_n = int(max(own_bucket.get("n", 0.0), 0.0))
        other_n = int(max(other_bucket.get("n", 0.0), 0.0))
        own_reg_n = int(max(own_reg_bucket.get("n", 0.0), 0.0))
        other_reg_n = int(max(other_reg_bucket.get("n", 0.0), 0.0))
        own_min_samples = max(min_side_samples // 2, 8) if switch_into_underweight else min_side_samples
        reg_min_samples = max(min_side_samples // 3, 6) if switch_into_underweight else max(8, min_side_samples // 2)
        has_depth = bool(
            own_n >= own_min_samples
            and other_n >= min_side_samples
            and own_reg_n >= reg_min_samples
            and other_reg_n >= reg_min_samples
        )
        own_exp = float(own_bucket.get("exp_ema", 0.0))
        other_exp = float(other_bucket.get("exp_ema", 0.0))
        own_reg_exp = float(own_reg_bucket.get("exp_ema", 0.0))
        other_reg_exp = float(other_reg_bucket.get("exp_ema", 0.0))
        min_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-4))
        has_quality_advantage = bool(
            other_exp > own_exp + 0.35 * min_unit and other_reg_exp > own_reg_exp + 0.20 * min_unit
        )
        unstable = bool(float(flip_pressure) > 0.5 or float(change_mode) > 0.5 or float(instability) >= 1.05)
        if total_side >= 40 and other == 1:
            long_share = float(int(long_trade_count) / max(total_side, 1))
            # Resist switching into already-dominant longs unless opposite evidence is overwhelming.
            min_gap += float(np.clip(max(long_share - 0.58, 0.0) * 1.5, 0.0, 0.35))
        analog_switch_limit = float(min_adv * (1.8 if switch_into_underweight else 1.0))
        # Switch only when live conviction is weak, state is stable, and opposite-side quality is decisively better.
        if (
            can_switch
            and has_depth
            and has_quality_advantage
            and not unstable
            and float(conviction) < conv_guard
            and gap >= min_gap
            and float(analog_edge) < analog_switch_limit
        ):
            s = other
            switched = True
            edge2 = float(max(edge2 + 0.15 * min(max_adj, pos_scale * min(gap, 1.0)), 0.0))
            conf2 = float(np.clip(conf2 + 0.015, 0.0, 1.0))
            unc2 = float(np.clip(unc2 * 1.03, 0.005, 2.0))
        elif (
            can_switch
            and bool(stress_mode)
            and float(conviction) < min(conv_guard + 0.12, 0.97)
            and float(other_exp - own_exp) >= (0.15 * min_unit)
            and other_n >= max(8, own_n // 2)
            and other_reg_n >= max(6, own_reg_n // 2)
            and float(analog_edge) < (analog_switch_limit * 1.25)
        ):
            # Month-shield fallback: in stressed windows allow a guarded side flip
            # with relaxed depth requirements to avoid getting stuck on a weak side.
            s = other
            switched = True
            edge2 = float(max(edge2 + 0.10 * min(max_adj, pos_scale * min(max(gap, 0.0), 1.0)), 0.0))
            conf2 = float(np.clip(conf2 + 0.010, 0.0, 1.0))
            unc2 = float(np.clip(unc2 * 1.02, 0.005, 2.0))

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
    warmup = int(max(getattr(cfg, "intelligence_side_switch_warmup_trades", 40), 0))
    if int(intelligence_mode_bars) < warmup:
        return False
    if int(intelligence_mode_bars) <= 0:
        return True
    max_rate = float(np.clip(getattr(cfg, "intelligence_side_switch_max_rate", 0.10), 0.0, 1.0))
    obs_rate = float(switched_so_far / max(intelligence_mode_bars, 1))
    return bool(obs_rate <= max_rate)


_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _utc_day_hour_arrays(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # Support both seconds and milliseconds timestamps.
    unit = "ms" if int(np.nanmax(np.abs(ts))) >= 10_000_000_000 else "s"
    dt = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    day = pd.Series(dt.dayofweek).fillna(0).astype(np.int64).to_numpy()
    hour = pd.Series(dt.hour).fillna(0).astype(np.int64).to_numpy()
    return day, hour


def _time_adaptive_recent_stats(values: List[float], *, min_samples: int) -> Dict[str, float]:
    n = int(len(values))
    m = int(max(min_samples, 1))
    if n < m:
        return {"ready": 0.0, "trades": float(n), "expectancy": 0.0, "win_rate": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "ready": 1.0,
        "trades": float(n),
        "expectancy": float(np.mean(arr)),
        "win_rate": float(np.mean(arr > 0.0)),
    }


def _resolve_time_adaptive_side_stats(
    *,
    day_idx: int,
    hour_idx: int,
    side: int,
    day_hour_rr: Dict[Tuple[int, int, int], List[float]],
    day_rr: Dict[Tuple[int, int], List[float]],
    hour_rr: Dict[Tuple[int, int], List[float]],
    cfg: MythosConfig,
    min_samples_override: Optional[int] = None,
) -> Dict[str, float]:
    s = int(side)
    if s not in (-1, 1):
        return {"ready": 0.0, "trades": 0.0, "expectancy": 0.0, "win_rate": 0.0}
    base_min_samples = int(max(getattr(cfg, "time_adaptive_min_bucket_trades", 8), 1))
    min_samples = int(max(min_samples_override if min_samples_override is not None else base_min_samples, 1))
    weighted: List[Tuple[float, Dict[str, float]]] = []
    components = (
        (0.60, _time_adaptive_recent_stats(day_hour_rr.get((int(day_idx), int(hour_idx), s), []), min_samples=min_samples)),
        (0.25, _time_adaptive_recent_stats(day_rr.get((int(day_idx), s), []), min_samples=min_samples)),
        (0.15, _time_adaptive_recent_stats(hour_rr.get((int(hour_idx), s), []), min_samples=min_samples)),
    )
    for w, stats in components:
        if float(stats.get("ready", 0.0)) > 0.5:
            weighted.append((float(w), stats))
    if not weighted:
        return {"ready": 0.0, "trades": 0.0, "expectancy": 0.0, "win_rate": 0.0}
    denom = float(max(sum(w for w, _ in weighted), 1e-6))
    exp = float(sum(w * float(s0.get("expectancy", 0.0)) for w, s0 in weighted) / denom)
    wr = float(sum(w * float(s0.get("win_rate", 0.0)) for w, s0 in weighted) / denom)
    trades = float(sum(float(s0.get("trades", 0.0)) for _, s0 in weighted))
    return {"ready": 1.0, "trades": trades, "expectancy": exp, "win_rate": wr}


def _apply_time_adaptive_adjustment(
    *,
    side: int,
    edge: float,
    confidence: float,
    conviction: float,
    total_trades: int,
    day_idx: int,
    hour_idx: int,
    day_hour_rr: Dict[Tuple[int, int, int], List[float]],
    day_rr: Dict[Tuple[int, int], List[float]],
    hour_rr: Dict[Tuple[int, int], List[float]],
    cfg: MythosConfig,
    long_trade_count: int = 0,
    short_trade_count: int = 0,
    stress_mode: bool = False,
) -> Dict[str, float]:
    s = int(side)
    if not bool(getattr(cfg, "time_adaptive_enable", True)) or s not in (-1, 1):
        return {
            "side": float(s),
            "edge": float(max(edge, 0.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "edge_adjust": 0.0,
            "conf_adjust": 0.0,
            "switched": 0.0,
            "ready": 0.0,
            "expectancy": 0.0,
        }
    warmup = int(max(getattr(cfg, "time_adaptive_warmup_trades", 36), 0))
    if bool(stress_mode):
        warmup = int(max(round(warmup * 0.55), 18))
    if int(total_trades) < warmup:
        return {
            "side": float(s),
            "edge": float(max(edge, 0.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "edge_adjust": 0.0,
            "conf_adjust": 0.0,
            "switched": 0.0,
            "ready": 0.0,
            "expectancy": 0.0,
        }
    total_side = int(max(int(long_trade_count) + int(short_trade_count), 0))
    short_share = float(int(short_trade_count) / max(total_side, 1)) if total_side > 0 else 0.5
    underweight_side = -1 if short_share < 0.36 else (1 if short_share > 0.64 else 0)
    bucket_min = int(max(getattr(cfg, "time_adaptive_min_bucket_trades", 8), 1))
    if bool(stress_mode):
        bucket_min = int(max(round(bucket_min * 0.70), 4))
    own = _resolve_time_adaptive_side_stats(
        day_idx=day_idx,
        hour_idx=hour_idx,
        side=s,
        day_hour_rr=day_hour_rr,
        day_rr=day_rr,
        hour_rr=hour_rr,
        cfg=cfg,
        min_samples_override=bucket_min,
    )
    if float(own.get("ready", 0.0)) < 0.5:
        return {
            "side": float(s),
            "edge": float(max(edge, 0.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "edge_adjust": 0.0,
            "conf_adjust": 0.0,
            "switched": 0.0,
            "ready": 0.0,
            "expectancy": 0.0,
        }
    max_edge_adj = float(np.clip(getattr(cfg, "time_adaptive_max_edge_adjust", 0.018), 0.0, 0.50))
    edge_scale = float(np.clip(getattr(cfg, "time_adaptive_edge_scale", 0.010), 0.0, 0.25))
    conf_scale = float(np.clip(getattr(cfg, "time_adaptive_conf_scale", 0.05), 0.0, 1.0))
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    own_exp = float(own.get("expectancy", 0.0))
    own_signal = float(np.tanh(own_exp / max(2.0 * edge_unit, 1e-6)))
    edge_adj = float(np.clip(own_signal * edge_scale, -max_edge_adj, max_edge_adj))
    conf_adj = float(own_signal * conf_scale)
    switched = False

    if bool(getattr(cfg, "time_adaptive_switch_enable", True)):
        switch_into_underweight = bool(underweight_side in (-1, 1) and int(-s) == int(underweight_side))
        other_bucket_min = bucket_min
        if switch_into_underweight and total_side >= 24:
            other_bucket_min = int(max(3, round(bucket_min * 0.60)))
        other = _resolve_time_adaptive_side_stats(
            day_idx=day_idx,
            hour_idx=hour_idx,
            side=-s,
            day_hour_rr=day_hour_rr,
            day_rr=day_rr,
            hour_rr=hour_rr,
            cfg=cfg,
            min_samples_override=other_bucket_min,
        )
        min_samples = int(max(getattr(cfg, "time_adaptive_switch_min_samples", 10), 1))
        if switch_into_underweight and total_side >= 24:
            min_samples = int(max(6, round(min_samples * 0.60)))
        if bool(stress_mode):
            min_samples = int(max(4, round(min_samples * 0.65)))
        if (
            float(other.get("ready", 0.0)) > 0.5
            and int(other.get("trades", 0.0)) >= min_samples
            and int(own.get("trades", 0.0)) >= max(min_samples // 2, 4)
        ):
            other_exp = float(other.get("expectancy", 0.0))
            gap_r = float(other_exp - own_exp)
            min_gap_r = float(np.clip(getattr(cfg, "time_adaptive_switch_min_gap_r", 0.04), 0.0, 2.0))
            conviction_guard = float(
                np.clip(getattr(cfg, "time_adaptive_switch_conviction_guard", 0.62), 0.0, 1.0)
            )
            if switch_into_underweight and total_side >= 24:
                min_gap_r = float(max(min_gap_r * 0.65, 0.0))
                conviction_guard = float(min(conviction_guard + 0.10, 0.92))
            if bool(stress_mode):
                min_gap_r = float(max(min_gap_r * 0.75, 0.0))
                conviction_guard = float(min(conviction_guard + 0.08, 0.95))
            if float(conviction) < conviction_guard and gap_r >= min_gap_r:
                s = -s
                switched = True
                switched_signal = float(np.tanh(other_exp / max(2.0 * edge_unit, 1e-6)))
                edge_adj = float(np.clip(switched_signal * edge_scale, -max_edge_adj, max_edge_adj))
                conf_adj = float(switched_signal * conf_scale + min(0.015, conf_scale * 0.35))
                own_exp = other_exp

    edge2 = float(max(edge + edge_adj, 0.0))
    conf2 = float(np.clip(confidence + conf_adj, 0.0, 1.0))
    return {
        "side": float(s),
        "edge": edge2,
        "confidence": conf2,
        "edge_adjust": edge_adj,
        "conf_adjust": conf_adj,
        "switched": float(1.0 if switched else 0.0),
        "ready": 1.0,
        "expectancy": own_exp,
    }


def _apply_micro_change_intelligence(
    *,
    side: int,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    bar_idx: int,
    close: np.ndarray,
    feat_row: pd.Series,
    last_switch_bar: int,
    cfg: MythosConfig,
) -> Dict[str, float]:
    s = int(side)
    if not bool(getattr(cfg, "micro_change_enable", True)) or s not in (-1, 1) or int(bar_idx) < 4:
        return {
            "side": float(s),
            "edge": float(max(edge, 0.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "uncertainty": float(np.clip(uncertainty, 0.005, 2.0)),
            "score": 0.0,
            "alignment": 0.0,
            "switched": 0.0,
            "active": 0.0,
        }
    px_now = float(close[bar_idx])
    px_1 = float(close[bar_idx - 1])
    px_4 = float(close[bar_idx - 4])
    ret1 = float((px_now / max(px_1, 1e-12)) - 1.0)
    ret4 = float((px_now / max(px_4, 1e-12)) - 1.0)
    vol16 = float(max(feat_row.get("vol_16", 0.0), 1e-6))
    vol64 = float(max(feat_row.get("vol_64", vol16), 1e-6))
    trend_accel = float(feat_row.get("trend_accel_4", 0.0))
    trend_slope = float(feat_row.get("trend_slope_8", 0.0))
    volume_z = float(feat_row.get("volume_z_64", 0.0))
    adx = float(feat_row.get("adx_14", 0.0))
    ret1_n = float(np.tanh(ret1 / vol16))
    ret4_n = float(np.tanh(ret4 / vol64))
    trend_n = float(np.tanh((0.65 * trend_accel + 0.35 * trend_slope) / max(0.02, 2.0 * vol64)))
    volume_n = float(np.tanh(volume_z / 2.0))
    adx_n = float(np.clip((adx - 0.2) / 0.8, -1.0, 1.0))
    context_n = float(0.60 * volume_n + 0.40 * adx_n)
    w1 = float(max(getattr(cfg, "micro_change_ret1_weight", 0.42), 0.0))
    w4 = float(max(getattr(cfg, "micro_change_ret4_weight", 0.30), 0.0))
    wt = float(max(getattr(cfg, "micro_change_trend_weight", 0.18), 0.0))
    wc = float(max(getattr(cfg, "micro_change_volume_weight", 0.10), 0.0))
    denom = float(max(w1 + w4 + wt + wc, 1e-6))
    score = float(np.clip((w1 * ret1_n + w4 * ret4_n + wt * trend_n + wc * context_n) / denom, -1.0, 1.0))
    threshold = float(np.clip(getattr(cfg, "micro_change_score_threshold", 0.08), 0.0, 1.0))
    if float(abs(score)) < threshold:
        return {
            "side": float(s),
            "edge": float(max(edge, 0.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "uncertainty": float(np.clip(uncertainty, 0.005, 2.0)),
            "score": score,
            "alignment": float(s * score),
            "switched": 0.0,
            "active": 0.0,
        }
    micro_side = 1 if score > 0.0 else -1
    switched = False
    if bool(getattr(cfg, "micro_change_switch_enable", True)) and micro_side != s:
        switch_thr = float(np.clip(getattr(cfg, "micro_change_switch_threshold", 0.22), 0.0, 1.0))
        conv_guard = float(np.clip(getattr(cfg, "micro_change_switch_conviction_guard", 0.62), 0.0, 1.0))
        switch_cd = int(max(getattr(cfg, "micro_change_switch_cooldown_bars", 8), 1))
        if (
            float(abs(score)) >= switch_thr
            and float(conviction) < conv_guard
            and int(bar_idx) - int(last_switch_bar) >= switch_cd
        ):
            s = micro_side
            switched = True
    alignment = float(np.clip(s * score, -1.0, 1.0))
    edge_scale = float(np.clip(getattr(cfg, "micro_change_edge_scale", 0.004), 0.0, 0.10))
    conf_scale = float(np.clip(getattr(cfg, "micro_change_conf_scale", 0.022), 0.0, 0.50))
    unc_scale = float(np.clip(getattr(cfg, "micro_change_uncertainty_scale", 0.14), 0.0, 1.0))
    edge_adj = float(np.clip(alignment * edge_scale, -2.0 * edge_scale, 2.0 * edge_scale))
    conf_adj = float(alignment * conf_scale)
    unc_mult = float(np.clip(1.0 - unc_scale * alignment, 0.50, 1.60))
    edge2 = float(max(edge + edge_adj, 0.0))
    conf2 = float(np.clip(confidence + conf_adj, 0.0, 1.0))
    unc2 = float(np.clip(uncertainty * unc_mult, 0.005, 2.0))
    return {
        "side": float(s),
        "edge": edge2,
        "confidence": conf2,
        "uncertainty": unc2,
        "score": score,
        "alignment": alignment,
        "switched": float(1.0 if switched else 0.0),
        "active": 1.0,
    }


def _apply_calibration_intelligence(
    *,
    side: int,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    confidence_hist: List[float],
    outcome_hist: List[float],
    rr_hist: List[float],
    cfg: MythosConfig,
) -> Dict[str, float]:
    s = int(side)
    base = {
        "edge": float(max(edge, 0.0)),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "uncertainty": float(np.clip(uncertainty, 0.005, 2.0)),
        "ready": 0.0,
        "active": 0.0,
        "mode_score": 0.0,
        "mode_protective": 0.0,
        "mode_opportunistic": 0.0,
        "abs_error": 0.0,
        "brier": 0.0,
        "hit_rate": 0.0,
        "expectancy": 0.0,
        "confidence_gap": 0.0,
        "participation_scale": 1.0,
    }
    if s not in (-1, 1) or not bool(getattr(cfg, "calibration_intelligence_enable", True)):
        return base

    stats = _recent_calibration_stats(
        confidence_hist,
        outcome_hist,
        rr_hist,
        window=int(max(getattr(cfg, "calibration_window", 120), 8)),
        min_samples=int(max(getattr(cfg, "calibration_min_samples", 28), 4)),
    )
    base.update(
        {
            "abs_error": float(stats.get("abs_error", 0.0)),
            "brier": float(stats.get("brier", 0.0)),
            "hit_rate": float(stats.get("hit_rate", 0.0)),
            "expectancy": float(stats.get("expectancy", 0.0)),
            "confidence_gap": float(stats.get("confidence_gap", 0.0)),
        }
    )
    if float(stats.get("ready", 0.0)) < 0.5:
        return base

    abs_err = float(stats.get("abs_error", 0.0))
    brier = float(stats.get("brier", 0.0))
    hit_rate = float(stats.get("hit_rate", 0.0))
    expectancy = float(stats.get("expectancy", 0.0))
    conf_gap = float(stats.get("confidence_gap", 0.0))

    target_abs = float(np.clip(getattr(cfg, "calibration_target_abs_error", 0.16), 0.01, 0.60))
    target_brier = float(np.clip(getattr(cfg, "calibration_target_brier", 0.22), 0.01, 0.80))
    under_margin = float(np.clip(getattr(cfg, "calibration_underconfidence_margin", 0.04), 0.0, 0.30))
    min_hit_boost = float(np.clip(getattr(cfg, "calibration_min_hit_for_boost", 0.53), 0.0, 1.0))
    min_exp_boost = float(getattr(cfg, "calibration_min_expectancy_for_boost", 0.02))
    conf_scale = float(np.clip(getattr(cfg, "calibration_conf_scale", 0.028), 0.0, 0.50))
    edge_scale = float(np.clip(getattr(cfg, "calibration_edge_scale", 0.004), 0.0, 0.10))
    unc_scale = float(np.clip(getattr(cfg, "calibration_uncertainty_scale", 0.35), 0.0, 2.0))
    max_conf_adj = float(np.clip(getattr(cfg, "calibration_max_conf_adjust", 0.08), 0.0, 0.60))
    max_edge_adj = float(np.clip(getattr(cfg, "calibration_max_edge_adjust", 0.012), 0.0, 0.20))
    max_unc_mult = float(np.clip(getattr(cfg, "calibration_max_uncertainty_mult", 1.60), 1.0, 5.0))
    min_part_scale = float(np.clip(getattr(cfg, "calibration_participation_min_scale", 0.30), 0.0, 1.0))

    abs_overshoot = float(max(abs_err - target_abs, 0.0) / max(target_abs, 1e-6))
    brier_overshoot = float(max(brier - target_brier, 0.0) / max(target_brier, 1e-6))
    overconf = float(max(conf_gap, 0.0))
    stress = float(np.clip(0.55 * abs_overshoot + 0.35 * brier_overshoot + 0.70 * overconf, 0.0, 2.0))

    underconf = float(max(-conf_gap - under_margin, 0.0))
    boost_score = 0.0
    if hit_rate >= min_hit_boost and expectancy >= min_exp_boost:
        exp_scale = max(abs(min_exp_boost), 0.02)
        boost_score = float(np.clip(underconf + max(expectancy - min_exp_boost, 0.0) / exp_scale, 0.0, 1.5))

    mode_score = float(np.clip(boost_score - stress, -2.0, 2.0))
    conf_adj = float(np.clip(mode_score * conf_scale, -max_conf_adj, max_conf_adj))
    edge_adj = float(np.clip(mode_score * edge_scale, -max_edge_adj, max_edge_adj))
    if mode_score < 0.0:
        unc_mult = float(np.clip(1.0 + unc_scale * abs(mode_score), 1.0, max_unc_mult))
    elif mode_score > 0.0:
        unc_mult = float(np.clip(1.0 - 0.45 * unc_scale * mode_score, 0.55, 1.0))
    else:
        unc_mult = 1.0

    conf2 = float(np.clip(confidence + conf_adj, 0.0, 1.0))
    edge2 = float(max(edge + edge_adj, 0.0))
    stress_n = float(np.clip(stress / 2.0, 0.0, 1.0))
    # Reliability-aware shrink: when confidence is overestimating realized hit-rate,
    # cap confidence and edge so selection pressure favors truly repeatable setups.
    if conf_gap > 0.0:
        conf_ceiling = float(np.clip(hit_rate + max(0.06 - under_margin, 0.02), 0.30, 0.88))
        conf2 = float(min(conf2, conf_ceiling))
        edge2 = float(max(edge2 * (1.0 - 0.45 * stress_n), 0.0))
    # In protective mode, avoid oversizing from stale conviction bursts.
    if mode_score < -0.15 and float(conviction) > 0.75:
        edge2 = float(max(edge2 - 0.25 * max_edge_adj, 0.0))
    unc2 = float(np.clip(uncertainty * unc_mult, 0.005, 2.0))
    protective = float(1.0 if mode_score < -0.05 else 0.0)
    opportunistic = float(1.0 if mode_score > 0.05 else 0.0)
    participation_scale = 1.0
    if protective > 0.5:
        participation_scale = float(np.clip(1.0 - 0.75 * min(abs(mode_score), 1.0), min_part_scale, 1.0))

    return {
        "edge": edge2,
        "confidence": conf2,
        "uncertainty": unc2,
        "ready": 1.0,
        "active": float(1.0 if abs(mode_score) > 1e-9 else 0.0),
        "mode_score": mode_score,
        "mode_protective": protective,
        "mode_opportunistic": opportunistic,
        "abs_error": abs_err,
        "brier": brier,
        "hit_rate": hit_rate,
        "expectancy": expectancy,
        "confidence_gap": conf_gap,
        "participation_scale": participation_scale,
    }


def _update_time_adaptive_memory(
    *,
    day_idx: int,
    hour_idx: int,
    side: int,
    realized_r: float,
    day_hour_rr: Dict[Tuple[int, int, int], List[float]],
    day_rr: Dict[Tuple[int, int], List[float]],
    hour_rr: Dict[Tuple[int, int], List[float]],
    cfg: MythosConfig,
) -> None:
    s = int(side)
    if s not in (-1, 1):
        return
    w = int(max(getattr(cfg, "time_adaptive_window", 240), 8))
    keys = (
        (day_hour_rr, (int(day_idx), int(hour_idx), s)),
        (day_rr, (int(day_idx), s)),
        (hour_rr, (int(hour_idx), s)),
    )
    for store, key in keys:
        hist = store.setdefault(key, [])
        hist.append(float(realized_r))
        if len(hist) > w:
            del hist[0 : len(hist) - w]


def _new_time_bucket_report() -> Dict[str, Dict[str, float]]:
    return {
        "all": {"trades": 0.0, "wins": 0.0, "sum_r": 0.0},
        "long": {"trades": 0.0, "wins": 0.0, "sum_r": 0.0},
        "short": {"trades": 0.0, "wins": 0.0, "sum_r": 0.0},
    }


def _update_time_bucket_report(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    *,
    bucket_key: str,
    side: int,
    realized_r: float,
) -> None:
    s = int(side)
    if s not in (-1, 1):
        return
    bucket = stats.setdefault(str(bucket_key), _new_time_bucket_report())
    labels = ["all", "long" if s == 1 else "short"]
    for label in labels:
        node = bucket.setdefault(label, {"trades": 0.0, "wins": 0.0, "sum_r": 0.0})
        node["trades"] = float(node.get("trades", 0.0) + 1.0)
        node["wins"] = float(node.get("wins", 0.0) + (1.0 if float(realized_r) > 0.0 else 0.0))
        node["sum_r"] = float(node.get("sum_r", 0.0) + float(realized_r))


def _summarize_time_bucket_report(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    *,
    ordered_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    ordered = list(ordered_keys or [])
    tail = sorted(k for k in stats.keys() if k not in set(ordered))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in ordered + tail:
        if key not in stats:
            continue
        row = stats[key]
        out[key] = {}
        for label in ("all", "long", "short"):
            node = row.get(label, {})
            trades = int(round(float(node.get("trades", 0.0))))
            wins = int(round(float(node.get("wins", 0.0))))
            total_r = float(node.get("sum_r", 0.0))
            win_rate = float(wins / max(trades, 1)) if trades else 0.0
            expectancy = float(total_r / max(trades, 1)) if trades else 0.0
            out[key][label] = {
                "trades": trades,
                "wins": wins,
                "win_rate": round(win_rate, 4),
                "expectancy_r": round(expectancy, 4),
                "total_r": round(total_r, 4),
            }
    return out


def _merge_time_bucket_reports(
    reports: List[Dict[str, object]],
    *,
    field: str,
    ordered_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    acc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for report in reports:
        rows = report.get(field, {})
        if not isinstance(rows, dict):
            continue
        for bucket_key, side_map in rows.items():
            if not isinstance(side_map, dict):
                continue
            base = acc.setdefault(str(bucket_key), _new_time_bucket_report())
            for label in ("all", "long", "short"):
                node = side_map.get(label, {})
                if not isinstance(node, dict):
                    continue
                base[label]["trades"] = float(base[label].get("trades", 0.0) + float(node.get("trades", 0.0)))
                base[label]["wins"] = float(base[label].get("wins", 0.0) + float(node.get("wins", 0.0)))
                base[label]["sum_r"] = float(base[label].get("sum_r", 0.0) + float(node.get("total_r", 0.0)))
    return _summarize_time_bucket_report(acc, ordered_keys=ordered_keys)


def _rank_time_bucket_side(
    bucket_report: Dict[str, Dict[str, Dict[str, float]]],
    *,
    side_key: str,
    min_trades: int,
    top_n: int,
    reverse: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for bucket_key, payload in bucket_report.items():
        node = payload.get(side_key, {})
        trades = int(node.get("trades", 0))
        if trades < int(max(min_trades, 1)):
            continue
        rows.append(
            {
                "bucket": str(bucket_key),
                "trades": int(trades),
                "wins": int(node.get("wins", 0)),
                "win_rate": float(node.get("win_rate", 0.0)),
                "expectancy_r": float(node.get("expectancy_r", 0.0)),
                "total_r": float(node.get("total_r", 0.0)),
            }
        )
    rows.sort(
        key=lambda r: (
            float(r.get("expectancy_r", 0.0)),
            float(r.get("win_rate", 0.0)),
            int(r.get("trades", 0)),
        ),
        reverse=bool(reverse),
    )
    return rows[: int(max(top_n, 1))]


def _precision_selective_quality_score(
    *,
    edge: float,
    confidence: float,
    uncertainty: float,
    conviction: float,
    support_score: float,
    reliability_score: float,
    stress_score: float,
    cfg: MythosConfig,
) -> float:
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    edge_n = float(np.clip(max(edge, 0.0) / max(2.5 * edge_unit, 1e-6), 0.0, 1.0))
    conf_n = float(np.clip(confidence, 0.0, 1.0))
    unc_n = float(np.clip(uncertainty, 0.0, 2.0) / 2.0)
    conv_n = float(np.clip(conviction, 0.0, 1.0))
    w_edge = float(max(getattr(cfg, "precision_selective_edge_weight", 0.45), 0.0))
    w_conf = float(max(getattr(cfg, "precision_selective_conf_weight", 0.35), 0.0))
    w_unc = float(max(getattr(cfg, "precision_selective_uncertainty_weight", 0.20), 0.0))
    w_conv = float(max(getattr(cfg, "precision_selective_conviction_weight", 0.25), 0.0))
    denom = float(max(w_edge + w_conf + w_unc + w_conv, 1e-6))
    core_score = float(np.clip((w_edge * edge_n + w_conf * conf_n + w_conv * conv_n - w_unc * unc_n) / denom, 0.0, 1.0))
    support_n = float(np.clip(support_score, 0.0, 1.0))
    reliability_n = float(np.clip(reliability_score, 0.0, 1.0))
    stress_n = float(np.clip(stress_score, 0.0, 1.0))
    support_w = float(np.clip(getattr(cfg, "precision_selective_support_weight", 0.22), 0.0, 2.0))
    reliability_w = float(np.clip(getattr(cfg, "precision_selective_reliability_weight", 0.28), 0.0, 2.0))
    stress_w = float(np.clip(getattr(cfg, "precision_selective_stress_weight", 0.24), 0.0, 2.0))
    adjusted = float(
        core_score
        + support_w * (support_n - 0.5)
        + reliability_w * (reliability_n - 0.5)
        - stress_w * stress_n
    )
    adjusted = float(np.clip(adjusted, 0.0, 1.0))
    # map to roughly [-1, 1] for stable quantile gating
    return float(np.clip(2.0 * adjusted - 1.0, -1.0, 1.0))


def _precision_selective_support_score(
    *,
    meta_p: float,
    analog_hits: float,
    side_recent_stats: Dict[str, float],
    cfg: MythosConfig,
) -> float:
    meta_n = float(np.clip(abs(float(meta_p) - 0.5) * 2.0, 0.0, 1.0))
    analog_n = float(
        np.clip(float(analog_hits) / max(float(getattr(cfg, "analog_k", 48)), 1.0), 0.0, 1.0)
    )
    side_n = 0.5
    if float(side_recent_stats.get("ready", 0.0)) > 0.5:
        hit_rate = float(side_recent_stats.get("hit_rate", 0.0))
        expectancy = float(side_recent_stats.get("expectancy", 0.0))
        edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
        hit_signal = float(np.clip((hit_rate - 0.5) / 0.20, -1.0, 1.0))
        exp_signal = float(np.clip(expectancy / max(2.0 * edge_unit, 1e-6), -1.0, 1.0))
        side_n = float(np.clip(0.5 + 0.30 * hit_signal + 0.20 * exp_signal, 0.0, 1.0))
    w_meta = float(max(getattr(cfg, "precision_selective_meta_support_weight", 0.34), 0.0))
    w_analog = float(max(getattr(cfg, "precision_selective_analog_support_weight", 0.24), 0.0))
    w_side = float(max(getattr(cfg, "precision_selective_side_support_weight", 0.42), 0.0))
    denom = float(max(w_meta + w_analog + w_side, 1e-6))
    return float(np.clip((w_meta * meta_n + w_analog * analog_n + w_side * side_n) / denom, 0.0, 1.0))


def _precision_selective_reliability_score(
    *,
    calibration_adj: Dict[str, float],
    cfg: MythosConfig,
) -> float:
    if float(calibration_adj.get("ready", 0.0)) < 0.5:
        return 0.5
    abs_err = float(calibration_adj.get("abs_error", 0.0))
    brier = float(calibration_adj.get("brier", 0.0))
    conf_gap = float(calibration_adj.get("confidence_gap", 0.0))
    mode_score = float(calibration_adj.get("mode_score", 0.0))
    target_abs = float(np.clip(getattr(cfg, "calibration_target_abs_error", 0.16), 0.01, 0.60))
    target_brier = float(np.clip(getattr(cfg, "calibration_target_brier", 0.22), 0.01, 0.80))
    abs_quality = float(np.clip(1.0 - max(abs_err - target_abs, 0.0) / max(2.0 * target_abs, 1e-6), 0.0, 1.0))
    brier_quality = float(np.clip(1.0 - max(brier - target_brier, 0.0) / max(2.0 * target_brier, 1e-6), 0.0, 1.0))
    gap_quality = float(np.clip(1.0 - max(conf_gap, 0.0) / 0.35, 0.0, 1.0))
    mode_quality = float(np.clip(0.5 + 0.25 * mode_score, 0.0, 1.0))
    score = float(
        0.35 * abs_quality
        + 0.25 * brier_quality
        + 0.25 * gap_quality
        + 0.15 * mode_quality
    )
    return float(np.clip(score, 0.0, 1.0))


def _precision_selective_stress_score(
    *,
    uncertainty: float,
    calibration_adj: Dict[str, float],
    side_recent_stats: Dict[str, float],
    cfg: MythosConfig,
) -> float:
    unc_n = float(np.clip(float(uncertainty), 0.0, 2.0) / 2.0)
    conf_gap = float(calibration_adj.get("confidence_gap", 0.0))
    overconf_n = float(np.clip(max(conf_gap, 0.0) / 0.35, 0.0, 1.0))
    side_drag = 0.0
    if float(side_recent_stats.get("ready", 0.0)) > 0.5:
        hit_rate = float(side_recent_stats.get("hit_rate", 0.0))
        expectancy = float(side_recent_stats.get("expectancy", 0.0))
        edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
        hit_shortfall = float(np.clip(max(0.5 - hit_rate, 0.0) / 0.5, 0.0, 1.0))
        exp_shortfall = float(np.clip(max(-expectancy, 0.0) / max(2.0 * edge_unit, 1e-6), 0.0, 1.0))
        side_drag = float(0.55 * hit_shortfall + 0.45 * exp_shortfall)
    return float(np.clip(0.40 * unc_n + 0.35 * overconf_n + 0.25 * side_drag, 0.0, 1.0))


def _precision_selective_quality_lift(
    *,
    accepted_scores: List[float],
    recent_rr: List[float],
    cfg: MythosConfig,
) -> Dict[str, float]:
    n = int(min(len(accepted_scores), len(recent_rr)))
    min_eval = int(max(getattr(cfg, "precision_selective_quality_eval_min_trades", 32), 4))
    if n < min_eval:
        return {"ready": 0.0, "lift": 0.0, "high_win_rate": 0.0, "low_win_rate": 0.0}
    scores = np.asarray(accepted_scores[-n:], dtype=np.float64)
    rr = np.asarray(recent_rr[-n:], dtype=np.float64)
    split_q = float(np.clip(getattr(cfg, "precision_selective_quality_eval_quantile", 0.65), 0.50, 0.99))
    split = float(np.quantile(scores, split_q))
    high_mask = scores >= split
    low_mask = ~high_mask
    min_group = int(max(min_eval // 4, 4))
    if int(np.sum(high_mask)) < min_group or int(np.sum(low_mask)) < min_group:
        return {"ready": 0.0, "lift": 0.0, "high_win_rate": 0.0, "low_win_rate": 0.0}
    high_wr = float(np.mean(rr[high_mask] > 0.0))
    low_wr = float(np.mean(rr[low_mask] > 0.0))
    return {
        "ready": 1.0,
        "lift": float(high_wr - low_wr),
        "high_win_rate": high_wr,
        "low_win_rate": low_wr,
    }


def _precision_selective_gate(
    *,
    quality: float,
    candidate_scores: List[float],
    accepted_scores: List[float],
    recent_rr: List[float],
    total_trades: int,
    cfg: MythosConfig,
    stress_mode: bool = False,
) -> Dict[str, float]:
    if not bool(getattr(cfg, "precision_selective_enable", False)):
        return {"pass": 1.0, "ready": 0.0}
    min_trades = int(max(getattr(cfg, "precision_selective_min_trades", 48), 0))
    score_window = int(max(getattr(cfg, "precision_selective_score_window", 512), 32))
    score_min_samples = int(max(getattr(cfg, "precision_selective_score_min_samples", 128), 16))
    rr = list(recent_rr[-score_window:]) if recent_rr else []
    if bool(stress_mode):
        min_trades = int(max(round(min_trades * 0.65), 18))
        score_min_samples = int(max(round(score_min_samples * 0.70), 48))
    if int(total_trades) < min_trades:
        return {"pass": 1.0, "ready": 0.0}
    scores = list(candidate_scores[-score_window:]) if candidate_scores else []
    if len(scores) < score_min_samples:
        return {"pass": 1.0, "ready": 0.0}
    target_wr = float(np.clip(getattr(cfg, "precision_selective_target_win_rate", 0.52), 0.0, 1.0))
    base_q = float(np.clip(getattr(cfg, "precision_selective_base_quantile", 0.70), 0.50, 0.999))
    max_q = float(np.clip(getattr(cfg, "precision_selective_max_quantile", 0.95), base_q, 0.999))
    adapt_gain = float(np.clip(getattr(cfg, "precision_selective_adapt_gain", 0.40), 0.0, 2.0))
    recent_wr = float(np.mean(np.asarray(rr, dtype=np.float64) > 0.0)) if rr else target_wr
    dynamic_q = float(np.clip(base_q + adapt_gain * (target_wr - recent_wr), base_q, max_q))
    quality_lift = _precision_selective_quality_lift(
        accepted_scores=accepted_scores,
        recent_rr=rr,
        cfg=cfg,
    )
    quality_lift_ready = float(quality_lift.get("ready", 0.0))
    quality_lift_val = float(quality_lift.get("lift", 0.0))
    quality_lift_adjust = 0.0
    if quality_lift_ready > 0.5:
        target_lift = float(np.clip(getattr(cfg, "precision_selective_quality_min_lift", 0.015), 0.0, 0.5))
        relax_gain = float(np.clip(getattr(cfg, "precision_selective_quality_relax_gain", 0.35), 0.0, 3.0))
        tighten_gain = float(np.clip(getattr(cfg, "precision_selective_quality_tighten_gain", 0.20), 0.0, 3.0))
        relax_cap = float(np.clip(getattr(cfg, "precision_selective_quality_relax_cap", 0.08), 0.0, 0.5))
        tighten_cap = float(np.clip(getattr(cfg, "precision_selective_quality_tighten_cap", 0.04), 0.0, 0.5))
        if quality_lift_val < target_lift:
            lift_gap = float(target_lift - quality_lift_val)
            quality_lift_adjust = float(-np.clip(relax_gain * lift_gap, 0.0, relax_cap))
        else:
            lift_excess = float(quality_lift_val - target_lift)
            quality_lift_adjust = float(np.clip(tighten_gain * lift_excess, 0.0, tighten_cap))
        min_q = float(np.clip(base_q - relax_cap, 0.50, base_q))
        dynamic_q = float(np.clip(dynamic_q + quality_lift_adjust, min_q, max_q))
    precision_pressure = float(
        np.clip((target_wr - recent_wr) / max(target_wr, 1e-6), 0.0, 1.0)
    )
    if bool(stress_mode):
        # Preserve stricter behavior for high-precision targets (>=55%).
        if target_wr < 0.55:
            # For lower precision targets only, allow slight anti-lockout relaxation.
            stress_relax = float(np.clip((target_wr - recent_wr) - 0.06, 0.0, 0.14))
            if stress_relax > 0.0:
                dynamic_q = float(np.clip(dynamic_q - min(0.06, 0.55 * stress_relax), 0.60, max_q))
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), dynamic_q))
    allowed = bool(float(quality) >= threshold)
    conf_floor_boost = float(np.clip(0.14 * precision_pressure, 0.0, 0.16))
    edge_floor_boost = float(np.clip(0.015 * precision_pressure, 0.0, 0.020))
    unc_base = float(np.clip(getattr(cfg, "risk_cap_override_max_uncertainty", 0.70), 0.10, 2.0))
    uncertainty_cap = float(np.clip(unc_base - 0.28 * precision_pressure, 0.30, unc_base))
    if quality_lift_ready > 0.5 and quality_lift_val < 0.0:
        weak = float(np.clip(abs(quality_lift_val) / 0.20, 0.0, 1.0))
        conf_floor_boost *= float(1.0 - 0.60 * weak)
        edge_floor_boost *= float(1.0 - 0.60 * weak)
        uncertainty_cap = float(np.clip(uncertainty_cap + 0.20 * weak, 0.30, 2.0))
    return {
        "pass": float(1.0 if allowed else 0.0),
        "ready": 1.0,
        "threshold": threshold,
        "quality": float(quality),
        "dynamic_quantile": dynamic_q,
        "quality_lift": quality_lift_val,
        "quality_lift_ready": quality_lift_ready,
        "quality_lift_adjust": quality_lift_adjust,
        "recent_win_rate": recent_wr,
        "precision_pressure": precision_pressure,
        "conf_floor_boost": conf_floor_boost,
        "edge_floor_boost": edge_floor_boost,
        "uncertainty_cap": uncertainty_cap,
    }


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


def _legacy_stable_leverage_ok(
    *,
    is_sure_signal: bool,
    conviction: float,
    edge: float,
    confidence: float,
    cfg: MythosConfig,
) -> bool:
    """
    Legacy-stable permissive leverage gate:
    keep strong conviction boosts active to restore historical participation.
    """
    if not bool(is_sure_signal):
        return False
    conv_floor = float(np.clip(getattr(cfg, "precision_high_conviction", 0.72), 0.0, 1.0))
    edge_floor = float(max(getattr(cfg, "min_edge_threshold", 0.01), 0.0))
    conf_floor = float(np.clip(getattr(cfg, "min_confidence", 0.50), 0.0, 1.0))
    if float(conviction) < conv_floor:
        return False
    if float(edge) < edge_floor:
        return False
    if float(confidence) < conf_floor:
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
    confidence: float,
    conviction: float,
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
    quality = float(
        np.clip(0.55 * float(np.clip(confidence, 0.0, 1.0)) + 0.45 * float(np.clip(conviction, 0.0, 1.0)), 0.0, 1.0)
    )
    combo = float(adjusted_edge + analog_adv)
    # Allow strong live-edge trades to pass unless counterfactual evidence is decisively negative.
    if adjusted_edge >= (0.75 * min_adv) and analog_adv >= (-0.5 * min_adv):
        return True
    if quality >= 0.80 and adjusted_edge >= (0.90 * min_adv) and analog_adv >= (-0.90 * min_adv):
        return True
    combo_adj = float(combo + max(quality - 0.50, 0.0) * 0.50 * min_adv)
    return combo_adj >= min_adv


def _adaptive_counterfactual_pass(
    *,
    analog_mem: AnalogMemory,
    x: np.ndarray,
    side: int,
    edge: float,
    confidence: float,
    conviction: float,
    uncertainty: float,
    cfg: MythosConfig,
    accepted_trades: int,
    cf_rejects: int,
) -> bool:
    target = float(np.clip(getattr(cfg, "counterfactual_target_reject_rate", 0.70), 0.0, 0.99))
    if bool(getattr(cfg, "precision_selective_enable", False)):
        precision_target = float(np.clip(getattr(cfg, "precision_selective_target_win_rate", 0.52), 0.0, 1.0))
        if precision_target >= 0.55:
            target = min(target, 0.72)
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
            confidence=confidence,
            conviction=conviction,
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
            confidence=confidence,
            conviction=conviction,
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


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + erf(float(x) / sqrt(2.0))))


def _robust_metric_neutral_value(metric: str) -> float:
    if str(metric).lower() == "win_rate":
        return 0.5
    return 0.0


def _score_fold_slice(
    folds: List[Dict[str, object]],
    indices: List[int],
    metric: str,
) -> float:
    if not indices:
        return 0.0
    metric = str(metric or "total_r").lower()
    values = np.asarray(
        [_select_fold_metric(folds[i], metric) for i in indices],
        dtype=np.float64,
    )
    if metric == "total_r":
        return float(np.sum(values))
    weights = np.asarray(
        [max(float(folds[i].get("total_trades", 0)), 1.0) for i in indices],
        dtype=np.float64,
    )
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / denom)


def _build_cpcv_splits(
    *,
    n_folds: int,
    test_size: int,
    max_paths: int,
    seed: int,
    purge_folds: int = 0,
    embargo_folds: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    if n_folds < 2:
        return []
    test_size = int(np.clip(test_size, 1, n_folds - 1))
    max_paths = int(max(max_paths, 1))
    all_idx = list(range(n_folds))
    total_paths = int(comb(n_folds, test_size))
    rng = np.random.default_rng(int(seed))
    if total_paths <= max_paths:
        chosen = list(combinations(all_idx, test_size))
    elif total_paths <= max_paths * 4:
        combos = list(combinations(all_idx, test_size))
        pick = rng.choice(len(combos), size=max_paths, replace=False)
        chosen = [combos[int(i)] for i in sorted(pick.tolist())]
    else:
        uniq: set[Tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max_paths * 64
        while len(uniq) < max_paths and attempts < max_attempts:
            sample = tuple(sorted(int(i) for i in rng.choice(n_folds, size=test_size, replace=False)))
            uniq.add(sample)
            attempts += 1
        chosen = sorted(uniq)
    splits: List[Tuple[List[int], List[int]]] = []
    purge_folds = int(max(purge_folds, 0))
    embargo_folds = int(max(embargo_folds, 0))
    for test_idx in chosen:
        test_set = set(int(i) for i in test_idx)
        forbidden: set[int] = set(test_set)
        if purge_folds > 0:
            for t in test_set:
                lo = max(0, int(t) - purge_folds)
                hi = min(n_folds - 1, int(t) + purge_folds)
                forbidden.update(range(lo, hi + 1))
        if embargo_folds > 0 and test_set:
            post_start = max(test_set) + 1
            post_end = min(n_folds - 1, max(test_set) + embargo_folds)
            if post_start <= post_end:
                forbidden.update(range(post_start, post_end + 1))
        train_idx = [i for i in all_idx if i not in forbidden]
        test_list = [int(i) for i in test_idx]
        if train_idx and test_list:
            splits.append((train_idx, test_list))
    return splits


def _sharpe_robust_diagnostics(
    values: np.ndarray,
    *,
    benchmark_sr: float,
    trial_count: int,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out = {
        "sample_size": float(n),
        "mean": 0.0,
        "std": 0.0,
        "sharpe": 0.0,
        "psr": 0.0,
        "dsr": 0.0,
        "sharpe_deflator": 0.0,
    }
    if n < 2:
        return out
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    out["mean"] = mean
    out["std"] = std
    if std <= 1e-12:
        out["sharpe"] = 0.0
        if mean > 0.0 and float(benchmark_sr) <= 0.0:
            out["psr"] = 1.0
            out["dsr"] = 1.0
        return out
    sharpe = float((mean / std) * sqrt(float(n)))
    out["sharpe"] = sharpe
    centered = arr - mean
    z = centered / max(std, 1e-12)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    denom_term = float(1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * (sharpe ** 2))
    denom_term = float(max(denom_term, 1e-9))
    zscore = float((sharpe - float(benchmark_sr)) * sqrt(max(n - 1, 1)) / sqrt(denom_term))
    psr = _normal_cdf(zscore)
    n_trials = int(max(trial_count, 1))
    deflator = float(sqrt(max(2.0 * np.log(float(n_trials)) / max(n - 1, 1), 0.0))) if n_trials > 1 else 0.0
    zscore_deflated = float((sharpe - deflator - float(benchmark_sr)) * sqrt(max(n - 1, 1)) / sqrt(denom_term))
    dsr = _normal_cdf(zscore_deflated)
    out.update(
        {
            "psr": float(np.clip(psr, 0.0, 1.0)),
            "dsr": float(np.clip(dsr, 0.0, 1.0)),
            "sharpe_deflator": deflator,
        }
    )
    return out


def _spa_single_model(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
    block_size: int = 1,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out = {"t_stat": 0.0, "p_value": 1.0, "bootstrap_samples": float(int(max(bootstrap_samples, 0)))}
    if n < 3:
        return out
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 1e-12:
        out["t_stat"] = float(max(mu, 0.0))
        out["p_value"] = 0.0 if mu > 0.0 else 1.0
        return out
    t_obs = float(max(sqrt(float(n)) * mu / sigma, 0.0))
    centered = arr - mu
    reps = int(max(bootstrap_samples, 32))
    block_size = int(max(block_size, 1))
    rng = np.random.default_rng(int(seed))
    ge_count = 0
    for _ in range(reps):
        if block_size <= 1:
            idx = rng.integers(0, n, size=n)
        else:
            idx_parts: List[int] = []
            while len(idx_parts) < n:
                start = int(rng.integers(0, n))
                take = int(min(block_size, n - len(idx_parts)))
                idx_parts.extend(((start + j) % n) for j in range(take))
            idx = np.asarray(idx_parts[:n], dtype=np.int64)
        sample = centered[idx]
        s = float(np.std(sample, ddof=1))
        if s <= 1e-12:
            t_boot = 0.0
        else:
            t_boot = float(max(sqrt(float(n)) * float(np.mean(sample)) / s, 0.0))
        if t_boot >= t_obs:
            ge_count += 1
    p_value = float(ge_count / max(reps, 1))
    out["t_stat"] = t_obs
    out["p_value"] = float(np.clip(p_value, 0.0, 1.0))
    out["bootstrap_samples"] = float(reps)
    return out


def _robust_validation_report(
    *,
    folds: List[Dict[str, object]],
    cfg: MythosConfig,
) -> Dict[str, object]:
    enabled = bool(getattr(cfg, "robust_validation_enable", True))
    metric = str(getattr(cfg, "robust_validation_metric", "expectancy_r")).lower()
    report: Dict[str, object] = {
        "enabled": enabled,
        "ready": False,
        "metric": metric,
        "n_folds": int(len(folds)),
    }
    if not enabled:
        report["reason"] = "disabled"
        return report
    min_folds = int(max(getattr(cfg, "robust_validation_min_folds", 5), 3))
    if len(folds) < min_folds:
        report["reason"] = f"insufficient_folds({len(folds)}<{min_folds})"
        return report
    test_fraction = float(np.clip(getattr(cfg, "cpcv_test_fraction", 0.40), 0.10, 0.90))
    test_size = int(np.clip(round(len(folds) * test_fraction), 1, len(folds) - 1))
    max_paths = int(max(getattr(cfg, "cpcv_max_paths", 256), 1))
    seed = int(max(getattr(cfg, "cpcv_random_seed", 42), 0))
    purge_folds = int(max(getattr(cfg, "cpcv_purge_folds", 1), 0))
    embargo_folds = int(max(getattr(cfg, "cpcv_embargo_folds", 1), 0))
    splits = _build_cpcv_splits(
        n_folds=len(folds),
        test_size=test_size,
        max_paths=max_paths,
        seed=seed,
        purge_folds=purge_folds,
        embargo_folds=embargo_folds,
    )
    if not splits:
        report["reason"] = "no_valid_cpcv_splits"
        return report

    neutral = float(_robust_metric_neutral_value(metric))
    rows: List[Dict[str, object]] = []
    train_scores: List[float] = []
    test_scores: List[float] = []
    for train_idx, test_idx in splits:
        train_score = _score_fold_slice(folds, train_idx, metric)
        test_score = _score_fold_slice(folds, test_idx, metric)
        test_total_r = float(sum(float(folds[i].get("total_r", 0.0)) for i in test_idx))
        test_trades = int(sum(int(folds[i].get("total_trades", 0)) for i in test_idx))
        train_scores.append(train_score)
        test_scores.append(test_score)
        rows.append(
            {
                "train_folds": [int(i + 1) for i in train_idx],
                "test_folds": [int(i + 1) for i in test_idx],
                "train_score": round(float(train_score), 6),
                "test_score": round(float(test_score), 6),
                "test_total_r": round(test_total_r, 4),
                "test_trades": test_trades,
            }
        )
    tr = np.asarray(train_scores, dtype=np.float64)
    te = np.asarray(test_scores, dtype=np.float64)
    sign_flip = np.logical_and(tr > neutral, te <= neutral)
    overfit_prob = float(np.mean(sign_flip)) if sign_flip.size else 0.0
    oos_underperform_prob = float(np.mean(te <= neutral)) if te.size else 0.0
    path_mean = float(np.mean(te)) if te.size else 0.0
    path_median = float(np.median(te)) if te.size else 0.0
    path_p10 = float(np.quantile(te, 0.10)) if te.size else 0.0
    path_p90 = float(np.quantile(te, 0.90)) if te.size else 0.0

    sharpe_diag = _sharpe_robust_diagnostics(
        te,
        benchmark_sr=float(getattr(cfg, "robust_validation_sr_benchmark", 0.0)),
        trial_count=int(max(getattr(cfg, "robust_validation_trial_count", 8), 1)),
    )
    spa_diag = _spa_single_model(
        te,
        bootstrap_samples=int(max(getattr(cfg, "robust_validation_spa_bootstrap_samples", 400), 32)),
        seed=seed + 17,
        block_size=int(max(getattr(cfg, "robust_validation_spa_block_size", 3), 1)),
    )
    top_n = int(max(getattr(cfg, "robust_validation_report_top_paths", 5), 1))
    sorted_rows = sorted(rows, key=lambda x: float(x.get("test_score", 0.0)), reverse=True)
    top_paths = sorted_rows[:top_n]
    worst_paths = list(reversed(sorted_rows[-top_n:])) if sorted_rows else []
    significance_95 = bool(
        float(spa_diag.get("p_value", 1.0)) < 0.05 and float(sharpe_diag.get("dsr", 0.0)) > 0.50
    )
    report.update(
        {
            "ready": True,
            "neutral_score": round(neutral, 6),
            "test_fraction": round(test_fraction, 4),
            "test_fold_size": int(test_size),
            "cpcv_purge_folds": int(purge_folds),
            "cpcv_embargo_folds": int(embargo_folds),
            "paths_evaluated": int(len(rows)),
            "pbo_overfit_probability": round(overfit_prob, 6),
            "pbo_oos_underperform_probability": round(oos_underperform_prob, 6),
            "cpcv_path_score_mean": round(path_mean, 6),
            "cpcv_path_score_median": round(path_median, 6),
            "cpcv_path_score_p10": round(path_p10, 6),
            "cpcv_path_score_p90": round(path_p90, 6),
            "sharpe": round(float(sharpe_diag.get("sharpe", 0.0)), 6),
            "psr": round(float(sharpe_diag.get("psr", 0.0)), 6),
            "dsr": round(float(sharpe_diag.get("dsr", 0.0)), 6),
            "sharpe_deflator": round(float(sharpe_diag.get("sharpe_deflator", 0.0)), 6),
            "spa_t_stat": round(float(spa_diag.get("t_stat", 0.0)), 6),
            "spa_p_value": round(float(spa_diag.get("p_value", 1.0)), 6),
            "spa_block_size": int(max(getattr(cfg, "robust_validation_spa_block_size", 3), 1)),
            "significant_edge_95": significance_95,
            "top_paths": top_paths,
            "worst_paths": worst_paths,
        }
    )
    return report


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
        "artifact_schema_version": int(max(getattr(cfg, "artifact_schema_version", 2), 1)),
        "decision_graph_version": "mythos-institutional-v2",
        "runtime_strict_config": bool(getattr(cfg, "runtime_strict_config", True)),
        "runtime_require_adaptive_brain": bool(
            getattr(cfg, "runtime_require_adaptive_brain", False)
        ),
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


def _resolve_adaptive_tp_sl(
    *,
    cfg: MythosConfig,
    side: int,
    edge: float,
    confidence: float,
    uncertainty: float,
    trend_ema: float,
    vol_16: float,
    vol_64: float,
) -> Tuple[float, float]:
    base_tp = float(max(getattr(cfg, "tp_mult", 2.0), 0.10))
    base_sl = float(max(getattr(cfg, "sl_mult", 1.5), 0.10))
    if not bool(getattr(cfg, "adaptive_tp_sl_enable", True)):
        return base_tp, base_sl
    edge_unit = float(max(getattr(cfg, "min_expected_r", 0.01), 1e-6))
    edge_score = float(np.tanh(float(edge) / max(2.0 * edge_unit, 1e-6)))
    conf_score = float(np.clip((float(confidence) - 0.5) * 2.0, -1.0, 1.0))
    unc_norm = float(np.clip(float(uncertainty), 0.0, 2.0) / 2.0)
    vol_ratio = float(np.clip(float(vol_16) / max(float(vol_64), 1e-6), 0.25, 4.0))
    vol_stress = float(max(vol_ratio - 1.0, 0.0))
    trend_score = float(np.clip(np.tanh(abs(float(trend_ema)) * 4.0), 0.0, 1.0))
    quality = float(np.clip(0.45 * conf_score + 0.45 * edge_score - 0.30 * unc_norm, -1.0, 1.0))

    tp_gain = (
        float(np.clip(getattr(cfg, "adaptive_tp_quality_gain", 0.35), 0.0, 2.0)) * quality
        + float(np.clip(getattr(cfg, "adaptive_tp_trend_gain", 0.20), 0.0, 2.0)) * trend_score
        - float(np.clip(getattr(cfg, "adaptive_tp_vol_penalty", 0.18), 0.0, 2.0)) * vol_stress
    )
    sl_gain = (
        -float(np.clip(getattr(cfg, "adaptive_sl_quality_tighten", 0.25), 0.0, 2.0)) * max(quality, 0.0)
        + float(np.clip(getattr(cfg, "adaptive_sl_uncertainty_widen", 0.30), 0.0, 2.0)) * unc_norm
        + float(np.clip(getattr(cfg, "adaptive_sl_vol_widen", 0.20), 0.0, 2.0)) * vol_stress
    )
    if int(side) == -1:
        tp_gain += float(np.clip(getattr(cfg, "adaptive_short_tp_bias", 0.05), -1.0, 1.0))
        sl_gain += float(np.clip(getattr(cfg, "adaptive_short_sl_bias", 0.04), -1.0, 1.0))

    tp_mult = float(base_tp * (1.0 + tp_gain))
    sl_mult = float(base_sl * (1.0 + sl_gain))
    tp_mult = float(
        np.clip(
            tp_mult,
            float(np.clip(getattr(cfg, "adaptive_tp_min_mult", 1.2), 0.10, 20.0)),
            float(max(getattr(cfg, "adaptive_tp_max_mult", 3.6), getattr(cfg, "adaptive_tp_min_mult", 1.2))),
        )
    )
    sl_mult = float(
        np.clip(
            sl_mult,
            float(np.clip(getattr(cfg, "adaptive_sl_min_mult", 0.8), 0.10, 20.0)),
            float(max(getattr(cfg, "adaptive_sl_max_mult", 2.4), getattr(cfg, "adaptive_sl_min_mult", 0.8))),
        )
    )
    return tp_mult, sl_mult


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
            "decision_bars": 0,
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
            "meta_soft_blocks": 0,
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
                boot_edge = float(abs(train_feat.iloc[j]["fwd_ret_4"]))
                boot_conf = float(
                    np.clip(0.5 + 0.5 * np.tanh(abs(train_feat.iloc[j]["fwd_ret_4"]) / 0.01), 0.0, 1.0)
                )
                boot_unc = float(np.clip(train_feat.iloc[j]["vol_16"] * 2.0, 0.01, 2.0))
                tp_mult_i, sl_mult_i = _resolve_adaptive_tp_sl(
                    cfg=cfg,
                    side=s,
                    edge=boot_edge,
                    confidence=boot_conf,
                    uncertainty=boot_unc,
                    trend_ema=float(train_feat.iloc[j].get("trend_ema", 0.0)),
                    vol_16=float(train_feat.iloc[j].get("vol_16", 0.0)),
                    vol_64=float(train_feat.iloc[j].get("vol_64", max(train_feat.iloc[j].get("vol_16", 0.0), 1e-6))),
                )
                rr = _barrier_outcome(
                    close=close_tr,
                    high=high_tr,
                    low=low_tr,
                    idx=j,
                    side=s,
                    tp_mult=tp_mult_i,
                    sl_mult=sl_mult_i,
                    horizon=cfg.horizon,
                    atr=atr_tr,
                )
                meta.update(
                    x=X_tr[j],
                    edge=boot_edge,
                    confidence=boot_conf,
                    uncertainty=boot_unc,
                    regime=int(train_regime[j]),
                    side=int(s),
                    realized_r=float(rr),
                )

    risk = RiskConstitution(cfg)
    close = test_feat["close"].to_numpy(dtype=np.float64)
    high = test_feat["high"].to_numpy(dtype=np.float64)
    low = test_feat["low"].to_numpy(dtype=np.float64)
    timestamps = test_feat["timestamp"].to_numpy(dtype=np.int64)
    weekday_idx, hour_idx = _utc_day_hour_arrays(timestamps)
    atr = _compute_atr(close, high, low, period=14)
    X_te = test_feat[xcols].to_numpy(dtype=np.float64)

    trades: List[float] = []
    gross_trades: List[float] = []
    skip_counts = {
        "low_confidence": 0,
        "edge_below_floor": 0,
        "streak_pause": 0,
        "precision_selective_reject": 0,
        "precision_uncertainty_cap": 0,
        "counterfactual_reject": 0,
        "bayes_quality_reject": 0,
        "nonconformity_reject": 0,
        "risk_reject": 0,
        "meta_reject": 0,
        "low_conviction": 0,
    }
    meta_soft_blocks = 0
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
    used_tp_mults: List[float] = []
    used_sl_mults: List[float] = []
    long_tp_mults: List[float] = []
    short_tp_mults: List[float] = []
    long_sl_mults: List[float] = []
    short_sl_mults: List[float] = []
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
    time_adaptive_day_hour_rr: Dict[Tuple[int, int, int], List[float]] = {}
    time_adaptive_day_rr: Dict[Tuple[int, int], List[float]] = {}
    time_adaptive_hour_rr: Dict[Tuple[int, int], List[float]] = {}
    time_bucket_day_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    time_bucket_hour_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    side_quality_rr: Dict[int, List[float]] = {1: [], -1: []}
    legacy_stable_mode = bool(
        (not bool(getattr(cfg, "bayes_quality_enable", True)))
        and (not bool(getattr(cfg, "nonconformity_enable", True)))
        and (not bool(getattr(cfg, "side_rebalance_enable", True)))
        and (not bool(getattr(cfg, "intelligence_enable", True)))
        and float(max(getattr(cfg, "execution_fee_bps", 0.0), 0.0)) <= 1e-9
        and float(max(getattr(cfg, "execution_slippage_bps", 0.0), 0.0)) <= 1e-9
    )
    intelligence_score_sum = 0.0
    intelligence_side_switches = 0
    intelligence_mode_bars = 0
    intelligence_last_switch_bar = -10_000_000
    time_adaptive_mode_bars = 0
    time_adaptive_ready_bars = 0
    time_adaptive_side_switches = 0
    time_adaptive_edge_adjust_sum = 0.0
    time_adaptive_conf_adjust_sum = 0.0
    precision_selective_scores: List[float] = []
    precision_selective_recent_rr: List[float] = []
    precision_selective_recent_scores: List[float] = []
    precision_selective_mode_bars = 0
    precision_selective_ready_bars = 0
    precision_selective_rejects = 0
    precision_selective_threshold_sum = 0.0
    precision_selective_quality_sum = 0.0
    precision_selective_quantile_sum = 0.0
    precision_selective_quality_lift_sum = 0.0
    precision_selective_quality_lift_checks = 0
    month_shield_mode_bars = 0
    last_trade_bar = -1
    opportunity_rescue_bars = 0
    opportunity_override_counterfactual = 0
    opportunity_override_bayes = 0
    opportunity_override_nonconformity = 0
    opportunity_max_drought_ratio = 0.0
    decision_bars_total = int(max(len(test_feat) - cfg.horizon, 0))
    participation_relaxed_bars = 0
    participation_override_counterfactual = 0
    participation_pressure_sum = 0.0
    micro_change_mode_bars = 0
    micro_change_switches = 0
    micro_change_score_sum = 0.0
    micro_change_alignment_sum = 0.0
    micro_change_last_switch_bar = -10_000_000
    calibration_conf_hist: List[float] = []
    calibration_outcome_hist: List[float] = []
    calibration_rr_hist: List[float] = []
    calibration_mode_bars = 0
    calibration_ready_bars = 0
    calibration_protective_bars = 0
    calibration_opportunistic_bars = 0
    calibration_abs_error_sum = 0.0
    calibration_brier_sum = 0.0
    calibration_hit_rate_sum = 0.0
    calibration_gap_sum = 0.0
    calibration_mode_score_sum = 0.0
    calibration_participation_scale_sum = 0.0
    calibration_keep = int(max(getattr(cfg, "calibration_window", 120) * 2, 64))
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
        meta_soft_block = bool(side != 0 and meta_ready and meta_p < meta_min_side_prob)
        meta_shortfall = float(max(meta_min_side_prob - meta_p, 0.0)) if meta_soft_block else 0.0
        signed = float((meta_p - 0.5) * 2.0)
        edge = float(edge * (1.0 + meta_gain * signed))
        confidence = float(np.clip(confidence + meta_conf_gain * signed, 0.0, 1.0))
        uncertainty = float(np.clip(uncertainty * (1.0 + meta_unc_pen * max(0.5 - meta_p, 0.0)), 0.005, 2.0))
        if meta_soft_block:
            meta_soft_blocks += 1
            edge = float(max(edge * (1.0 - min(1.2 * meta_shortfall, 0.65)), 0.0))
            confidence = float(np.clip(confidence - min(0.75 * meta_shortfall, 0.20), 0.0, 1.0))
            uncertainty = float(np.clip(uncertainty * (1.0 + min(2.0 * meta_shortfall, 0.80)), 0.005, 2.0))
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        recent_global = _recent_quality_stats(
            trades,
            window=96,
            min_trades=24,
        )
        recent_long = _recent_quality_stats(
            long_trades,
            window=64,
            min_trades=12,
        )
        recent_short = _recent_quality_stats(
            short_trades,
            window=64,
            min_trades=12,
        )
        recent_exp = float(recent_global.get("expectancy", 0.0))
        recent_hit = float(recent_global.get("hit_rate", 0.5))
        side_gap = float(recent_short.get("expectancy", 0.0) - recent_long.get("expectancy", 0.0))
        month_shield_stress = bool(
            bool(recent_global.get("ready", 0.0))
            and (
                recent_exp < 0.0
                or recent_hit < 0.42
                or (
                    bool(recent_long.get("ready", 0.0))
                    and bool(recent_short.get("ready", 0.0))
                    and abs(side_gap) > 0.04
                )
            )
        )
        if month_shield_stress:
            month_shield_mode_bars += 1
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
            bar_idx=i,
            switched_so_far=intelligence_side_switches,
            intelligence_mode_bars=intelligence_mode_bars,
            last_switch_bar=intelligence_last_switch_bar,
            instability=float(adapted.get("shock", 0.0)) + float(adapted.get("flip", 0.0)),
            flip_pressure=float(adapted.get("flip_pressure", 0.0)),
            change_mode=float(adapted.get("change_mode", 0.0)),
            long_trade_count=len(long_trades),
            short_trade_count=len(short_trades),
            stress_mode=month_shield_stress,
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
            intelligence_last_switch_bar = int(i)
        day_i = int(weekday_idx[i]) if i < len(weekday_idx) else 0
        hour_i = int(hour_idx[i]) if i < len(hour_idx) else 0
        time_adj = _apply_time_adaptive_adjustment(
            side=side,
            edge=edge,
            confidence=confidence,
            conviction=conviction,
            total_trades=len(trades),
            day_idx=day_i,
            hour_idx=hour_i,
            day_hour_rr=time_adaptive_day_hour_rr,
            day_rr=time_adaptive_day_rr,
            hour_rr=time_adaptive_hour_rr,
            cfg=cfg,
            long_trade_count=len(long_trades),
            short_trade_count=len(short_trades),
            stress_mode=month_shield_stress,
        )
        side = int(round(float(time_adj.get("side", side))))
        edge = float(time_adj.get("edge", edge))
        confidence = float(np.clip(time_adj.get("confidence", confidence), 0.0, 1.0))
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        edge_adj = float(time_adj.get("edge_adjust", 0.0))
        conf_adj = float(time_adj.get("conf_adjust", 0.0))
        time_adaptive_edge_adjust_sum += edge_adj
        time_adaptive_conf_adjust_sum += conf_adj
        if float(time_adj.get("ready", 0.0)) > 0.5:
            time_adaptive_ready_bars += 1
        if abs(edge_adj) > 1e-9 or abs(conf_adj) > 1e-9 or float(time_adj.get("switched", 0.0)) > 0.5:
            time_adaptive_mode_bars += 1
        if float(time_adj.get("switched", 0.0)) > 0.5:
            time_adaptive_side_switches += 1
        micro_adj = _apply_micro_change_intelligence(
            side=side,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            conviction=conviction,
            bar_idx=i,
            close=close,
            feat_row=test_feat.iloc[i],
            last_switch_bar=micro_change_last_switch_bar,
            cfg=cfg,
        )
        side = int(round(float(micro_adj.get("side", side))))
        edge = float(micro_adj.get("edge", edge))
        confidence = float(np.clip(micro_adj.get("confidence", confidence), 0.0, 1.0))
        uncertainty = float(np.clip(micro_adj.get("uncertainty", uncertainty), 0.005, 2.0))
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        calibration_adj = _apply_calibration_intelligence(
            side=side,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            conviction=conviction,
            confidence_hist=calibration_conf_hist,
            outcome_hist=calibration_outcome_hist,
            rr_hist=calibration_rr_hist,
            cfg=cfg,
        )
        edge = float(calibration_adj.get("edge", edge))
        confidence = float(np.clip(calibration_adj.get("confidence", confidence), 0.0, 1.0))
        uncertainty = float(np.clip(calibration_adj.get("uncertainty", uncertainty), 0.005, 2.0))
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        if float(calibration_adj.get("ready", 0.0)) > 0.5:
            calibration_ready_bars += 1
            calibration_abs_error_sum += float(calibration_adj.get("abs_error", 0.0))
            calibration_brier_sum += float(calibration_adj.get("brier", 0.0))
            calibration_hit_rate_sum += float(calibration_adj.get("hit_rate", 0.0))
            calibration_gap_sum += float(calibration_adj.get("confidence_gap", 0.0))
            calibration_participation_scale_sum += float(calibration_adj.get("participation_scale", 1.0))
            if float(calibration_adj.get("active", 0.0)) > 0.5:
                calibration_mode_bars += 1
                calibration_mode_score_sum += float(calibration_adj.get("mode_score", 0.0))
            if float(calibration_adj.get("mode_protective", 0.0)) > 0.5:
                calibration_protective_bars += 1
            if float(calibration_adj.get("mode_opportunistic", 0.0)) > 0.5:
                calibration_opportunistic_bars += 1
        micro_score = float(micro_adj.get("score", 0.0))
        micro_alignment = float(micro_adj.get("alignment", 0.0))
        if float(micro_adj.get("active", 0.0)) > 0.5:
            micro_change_mode_bars += 1
            micro_change_score_sum += micro_score
            micro_change_alignment_sum += micro_alignment
        if float(micro_adj.get("switched", 0.0)) > 0.5:
            micro_change_switches += 1
            micro_change_last_switch_bar = int(i)
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
        participation_pressure = 0.0
        if bool(getattr(cfg, "participation_adapt_enable", True)) and decision_bars_total > 0:
            progress = float((i + 1) / max(decision_bars_total, 1))
            start_progress = float(np.clip(getattr(cfg, "participation_relax_start_progress", 0.25), 0.0, 1.0))
            if progress >= start_progress:
                target_fold_trades = int(max(getattr(cfg, "participation_target_trades_per_fold", 40), 1))
                target_now = float(target_fold_trades * progress)
                trade_gap = float(max(target_now - len(trades), 0.0))
                participation_pressure = float(np.clip(trade_gap / max(float(target_fold_trades), 1.0), 0.0, 1.0))
        if participation_pressure > 0.0:
            align_floor = float(np.clip(getattr(cfg, "micro_change_participation_align_min", 0.35), 0.0, 1.0))
            align_scale = float(np.clip(align_floor + (1.0 - align_floor) * max(micro_alignment, 0.0), align_floor, 1.0))
            participation_pressure *= align_scale
            calib_scale = float(
                np.clip(calibration_adj.get("participation_scale", 1.0), 0.0, 1.0)
            )
            participation_pressure *= calib_scale
        participation_pressure_sum += participation_pressure
        sure_recent_window = int(max(getattr(cfg, "sure_recent_window", 96), 8))
        min_conv = float(np.clip(getattr(cfg, "precision_min_conviction", 0.52), 0.0, 1.0))
        adaptive_min_conv = min_conv
        if participation_pressure > 0.0:
            conv_relax = float(
                np.clip(getattr(cfg, "participation_conviction_relax_max", 0.08), 0.0, 0.50)
                * participation_pressure
            )
            adaptive_min_conv = float(
                max(min_conv - conv_relax, np.clip(getattr(cfg, "participation_min_conviction", 0.46), 0.0, 1.0))
            )
        if side != 0 and conviction < adaptive_min_conv:
            skip_counts["low_conviction"] += 1
            continue
        precision_conf_floor_boost = 0.0
        precision_edge_floor_boost = 0.0
        precision_uncertainty_cap: Optional[float] = None
        precision_quality = 0.0
        if side != 0:
            score_window = int(max(getattr(cfg, "precision_selective_score_window", 512), 32))
            side_stats = _recent_quality_stats(
                side_quality_rr.get(int(side), []),
                window=int(max(getattr(cfg, "precision_selective_side_window", 96), 8)),
                min_trades=int(max(getattr(cfg, "precision_selective_side_min_trades", 12), 1)),
            )
            support_score = _precision_selective_support_score(
                meta_p=meta_p,
                analog_hits=analog_hits,
                side_recent_stats=side_stats,
                cfg=cfg,
            )
            reliability_score = _precision_selective_reliability_score(
                calibration_adj=calibration_adj,
                cfg=cfg,
            )
            stress_score = _precision_selective_stress_score(
                uncertainty=uncertainty,
                calibration_adj=calibration_adj,
                side_recent_stats=side_stats,
                cfg=cfg,
            )
            quality = _precision_selective_quality_score(
                edge=edge,
                confidence=confidence,
                uncertainty=uncertainty,
                conviction=conviction,
                support_score=support_score,
                reliability_score=reliability_score,
                stress_score=stress_score,
                cfg=cfg,
            )
            precision_quality = float(quality)
            precision_selective_scores.append(float(quality))
            if len(precision_selective_scores) > score_window:
                del precision_selective_scores[0 : len(precision_selective_scores) - score_window]
            precision_gate = _precision_selective_gate(
                quality=float(quality),
                candidate_scores=precision_selective_scores,
                accepted_scores=precision_selective_recent_scores,
                recent_rr=precision_selective_recent_rr,
                total_trades=len(trades),
                cfg=cfg,
                stress_mode=month_shield_stress,
            )
            if float(precision_gate.get("ready", 0.0)) > 0.5:
                precision_selective_mode_bars += 1
                precision_selective_ready_bars += 1
                precision_selective_threshold_sum += float(precision_gate.get("threshold", 0.0))
                precision_selective_quality_sum += float(precision_gate.get("quality", quality))
                precision_selective_quantile_sum += float(precision_gate.get("dynamic_quantile", 0.0))
                if float(precision_gate.get("quality_lift_ready", 0.0)) > 0.5:
                    precision_selective_quality_lift_checks += 1
                    precision_selective_quality_lift_sum += float(precision_gate.get("quality_lift", 0.0))
                precision_conf_floor_boost = float(
                    max(precision_gate.get("conf_floor_boost", 0.0), 0.0)
                )
                precision_edge_floor_boost = float(
                    max(precision_gate.get("edge_floor_boost", 0.0), 0.0)
                )
                precision_uncertainty_cap = float(
                    np.clip(precision_gate.get("uncertainty_cap", 2.0), 0.05, 2.0)
                )
                if float(precision_gate.get("pass", 1.0)) < 0.5:
                    precision_selective_rejects += 1
                    skip_counts["precision_selective_reject"] += 1
                    continue
        edge_floor = governor.adjusted_edge_floor(risk.state.equity_r)
        edge -= governor.side_penalty(side)
        conviction = _conviction_score(
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            meta_p=meta_p,
            cfg=cfg,
        )
        drought_ratio = 0.0
        dynamic_conf_floor = float(cfg.min_confidence)
        dynamic_edge_floor = float(edge_floor)
        opportunity_enabled = bool(getattr(cfg, "opportunity_rescue_enable", True))
        if opportunity_enabled:
            drought_bars = int(i - last_trade_bar) if last_trade_bar >= 0 else int(i + 1)
            drought_start = int(max(getattr(cfg, "opportunity_rescue_start_bars", 96), 1))
            drought_full = int(
                max(getattr(cfg, "opportunity_rescue_full_bars", 384), drought_start + 1)
            )
            if drought_bars > drought_start:
                drought_ratio = float(
                    np.clip((drought_bars - drought_start) / max(drought_full - drought_start, 1), 0.0, 1.0)
                )
                opportunity_rescue_bars += 1
                opportunity_max_drought_ratio = max(opportunity_max_drought_ratio, drought_ratio)
                conf_relax = float(np.clip(getattr(cfg, "opportunity_rescue_conf_relax", 0.08), 0.0, 0.50))
                edge_relax = float(max(getattr(cfg, "opportunity_rescue_edge_relax", 0.010), 0.0))
                min_conf_floor = float(
                    np.clip(getattr(cfg, "opportunity_rescue_min_confidence", 0.47), 0.0, 1.0)
                )
                min_edge_floor = float(max(getattr(cfg, "opportunity_rescue_min_edge", 0.004), 0.0))
                dynamic_conf_floor = max(float(cfg.min_confidence) - conf_relax * drought_ratio, min_conf_floor)
                dynamic_edge_floor = max(float(edge_floor) - edge_relax * drought_ratio, min_edge_floor)
        if participation_pressure > 0.0:
            conf_relax = float(
                np.clip(getattr(cfg, "participation_conf_relax_max", 0.10), 0.0, 0.60) * participation_pressure
            )
            edge_relax = float(
                np.clip(getattr(cfg, "participation_edge_relax_max", 0.008), 0.0, 0.20) * participation_pressure
            )
            min_conf_floor = float(np.clip(getattr(cfg, "participation_min_confidence", 0.44), 0.0, 1.0))
            min_edge_floor = float(max(getattr(cfg, "participation_min_edge", 0.003), 0.0))
            dynamic_conf_floor = max(dynamic_conf_floor - conf_relax, min_conf_floor)
            dynamic_edge_floor = max(dynamic_edge_floor - edge_relax, min_edge_floor)
            participation_relaxed_bars += 1
        if side != 0 and bool(getattr(cfg, "precision_selective_enable", False)):
            # Precision controller: when recent precision lags target, harden floors
            # after rescue/participation relaxations to prioritize hit quality.
            dynamic_conf_floor = min(dynamic_conf_floor + precision_conf_floor_boost, 0.92)
            dynamic_edge_floor = dynamic_edge_floor + precision_edge_floor_boost
            if precision_uncertainty_cap is not None and uncertainty > precision_uncertainty_cap:
                skip_counts["precision_uncertainty_cap"] += 1
                continue
        if meta_soft_block:
            meta_hard_conf_floor = min(dynamic_conf_floor + 0.04, 1.0)
            meta_hard_edge_floor = dynamic_edge_floor + 0.0015
            if (
                confidence < meta_hard_conf_floor
                and edge < meta_hard_edge_floor
                and conviction < min(min_conv + 0.08, 1.0)
            ):
                skip_counts["meta_reject"] += 1
                continue
        if confidence < dynamic_conf_floor:
            skip_counts["low_confidence"] += 1
            continue
        if edge < dynamic_edge_floor:
            skip_counts["edge_below_floor"] += 1
            continue
        if not governor.allow_by_streak(i, side=side):
            skip_counts["streak_pause"] += 1
            continue
        drought_override_ready = (
            drought_ratio >= float(np.clip(getattr(cfg, "opportunity_rescue_override_start", 0.55), 0.0, 1.0))
            and conviction >= float(np.clip(getattr(cfg, "opportunity_rescue_override_conviction", 0.70), 0.0, 1.0))
            and edge >= (
                dynamic_edge_floor + float(max(getattr(cfg, "opportunity_rescue_override_edge_buffer", 0.002), 0.0))
            )
            and confidence >= min(
                dynamic_conf_floor
                + float(np.clip(getattr(cfg, "opportunity_rescue_override_conf_buffer", 0.02), 0.0, 1.0)),
                1.0,
            )
        )
        participation_override_ready = bool(
            participation_pressure >= 0.30
            and conviction >= float(np.clip(getattr(cfg, "participation_override_conviction", 0.74), 0.0, 1.0))
            and edge >= (
                dynamic_edge_floor + float(max(getattr(cfg, "participation_override_edge_buffer", 0.0015), 0.0))
            )
            and confidence >= min(
                dynamic_conf_floor
                + float(np.clip(getattr(cfg, "participation_override_conf_buffer", 0.01), 0.0, 1.0)),
                1.0,
            )
        )
        can_override_reject = bool(drought_override_ready or participation_override_ready)
        cf_pass = _adaptive_counterfactual_pass(
            analog_mem=analog_mem,
            x=x,
            side=side,
            edge=edge,
            confidence=confidence,
            conviction=conviction,
            uncertainty=uncertainty,
            cfg=cfg,
            accepted_trades=len(trades),
            cf_rejects=cf_rejects,
        )
        if not cf_pass:
            if can_override_reject:
                if (not drought_override_ready) and participation_override_ready:
                    participation_override_counterfactual += 1
                opportunity_override_counterfactual += 1
                edge = float(max(edge - 0.0015 * drought_ratio, 0.0))
                confidence = float(np.clip(confidence - 0.01 * drought_ratio, 0.0, 1.0))
            else:
                cf_rejects += 1
                skip_counts["counterfactual_reject"] += 1
                continue
        if edge < dynamic_edge_floor or confidence < dynamic_conf_floor:
            skip_counts["edge_below_floor"] += 1
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
                if can_override_reject:
                    opportunity_override_bayes += 1
                    edge = float(max(edge - 0.001 * drought_ratio, 0.0))
                    confidence = float(np.clip(confidence - 0.006 * drought_ratio, 0.0, 1.0))
                else:
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
                if can_override_reject:
                    opportunity_override_nonconformity += 1
                    edge = float(max(edge - 0.001 * drought_ratio, 0.0))
                    confidence = float(np.clip(confidence - 0.006 * drought_ratio, 0.0, 1.0))
                else:
                    nonconformity_rejects += 1
                    skip_counts["nonconformity_reject"] += 1
                    continue
            if float(nonconf_gate.get("override", 0.0)) > 0.5:
                nonconformity_overrides += 1
        if not risk.allow_trade(
            ts_ms=int(timestamps[i]),
            side=side,
            edge=edge,
            uncertainty=uncertainty,
            conviction=conviction,
            edge_floor=dynamic_edge_floor,
        ):
            skip_counts["risk_reject"] += 1
            continue
        tp_mult_i, sl_mult_i = _resolve_adaptive_tp_sl(
            cfg=cfg,
            side=side,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            trend_ema=float(test_feat.iloc[i].get("trend_ema", 0.0)),
            vol_16=float(test_feat.iloc[i].get("vol_16", 0.0)),
            vol_64=float(test_feat.iloc[i].get("vol_64", max(test_feat.iloc[i].get("vol_16", 0.0), 1e-6))),
        )
        realized = _barrier_outcome(
            close=close,
            high=high,
            low=low,
            idx=i,
            side=side,
            tp_mult=tp_mult_i,
            sl_mult=sl_mult_i,
            horizon=cfg.horizon,
            atr=atr,
        )
        used_tp_mults.append(float(tp_mult_i))
        used_sl_mults.append(float(sl_mult_i))
        if int(side) == 1:
            long_tp_mults.append(float(tp_mult_i))
            long_sl_mults.append(float(sl_mult_i))
        elif int(side) == -1:
            short_tp_mults.append(float(tp_mult_i))
            short_sl_mults.append(float(sl_mult_i))
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
        if legacy_stable_mode:
            leverage_allowed = _legacy_stable_leverage_ok(
                is_sure_signal=is_sure_signal,
                conviction=conviction,
                edge=edge,
                confidence=confidence,
                cfg=cfg,
            )
        else:
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
        # Count approval only when it yields actual boosted size, so diagnostics
        # align with realized leveraged_trades and avoid false approval inflation.
        if leverage_allowed:
            projected_size = risk.position_size_multiplier(
                edge=edge,
                uncertainty=uncertainty,
                regime=regime,
                conviction=conviction,
                allow_conviction_boost=True,
            )
            if projected_size > (base_size + 1e-9):
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
        risk.record_trade(
            rr,
            int(timestamps[i]),
            edge=edge,
            uncertainty=uncertainty,
            conviction=conviction,
            size_mult=size,
        )
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
        calibration_conf_hist.append(float(np.clip(confidence, 0.0, 1.0)))
        calibration_outcome_hist.append(1.0 if rr > 0.0 else 0.0)
        calibration_rr_hist.append(float(rr))
        if len(calibration_conf_hist) > calibration_keep:
            del calibration_conf_hist[0 : len(calibration_conf_hist) - calibration_keep]
        if len(calibration_outcome_hist) > calibration_keep:
            del calibration_outcome_hist[0 : len(calibration_outcome_hist) - calibration_keep]
        if len(calibration_rr_hist) > calibration_keep:
            del calibration_rr_hist[0 : len(calibration_rr_hist) - calibration_keep]
        precision_selective_recent_rr.append(rr)
        ps_window = int(max(getattr(cfg, "precision_selective_score_window", 512), 32))
        if len(precision_selective_recent_rr) > ps_window:
            del precision_selective_recent_rr[0 : len(precision_selective_recent_rr) - ps_window]
        precision_selective_recent_scores.append(float(precision_quality))
        if len(precision_selective_recent_scores) > ps_window:
            del precision_selective_recent_scores[0 : len(precision_selective_recent_scores) - ps_window]
        last_trade_bar = int(i)
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
        _update_time_adaptive_memory(
            day_idx=day_i,
            hour_idx=hour_i,
            side=side,
            realized_r=rr,
            day_hour_rr=time_adaptive_day_hour_rr,
            day_rr=time_adaptive_day_rr,
            hour_rr=time_adaptive_hour_rr,
            cfg=cfg,
        )
        day_key = _DAY_NAMES[int(np.clip(day_i, 0, len(_DAY_NAMES) - 1))]
        hour_key = f"{int(np.clip(hour_i, 0, 23)):02d}"
        _update_time_bucket_report(
            time_bucket_day_raw,
            bucket_key=day_key,
            side=side,
            realized_r=rr,
        )
        _update_time_bucket_report(
            time_bucket_hour_raw,
            bucket_key=hour_key,
            side=side,
            realized_r=rr,
        )
        side_hist = side_quality_rr.setdefault(int(side), [])
        side_hist.append(rr)
        if len(side_hist) > 128:
            del side_hist[0 : len(side_hist) - 128]
        if conviction >= high_conv_thresh:
            high_conv_trades += 1
            high_conv_r_sum += rr
            if rr > 0.0:
                high_conv_wins += 1

    decision_bars = int(max(len(test_feat) - cfg.horizon, 0))
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
    day_report = _summarize_time_bucket_report(
        time_bucket_day_raw,
        ordered_keys=list(_DAY_NAMES),
    )
    hour_report = _summarize_time_bucket_report(
        time_bucket_hour_raw,
        ordered_keys=[f"{h:02d}" for h in range(24)],
    )
    bucket_min_trades = int(max(getattr(cfg, "time_adaptive_min_bucket_trades", 8), 1))
    bucket_top_n = int(max(getattr(cfg, "time_adaptive_report_top_n", 6), 1))
    best_short_hours = _rank_time_bucket_side(
        hour_report,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    worst_short_hours = _rank_time_bucket_side(
        hour_report,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    best_long_hours = _rank_time_bucket_side(
        hour_report,
        side_key="long",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    worst_long_hours = _rank_time_bucket_side(
        hour_report,
        side_key="long",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    best_short_days = _rank_time_bucket_side(
        day_report,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    worst_short_days = _rank_time_bucket_side(
        day_report,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    return {
        "total_trades": n,
        "decision_bars": decision_bars,
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
        "meta_soft_blocks": int(meta_soft_blocks),
        "counterfactual_rejects": int(cf_rejects),
        "bayes_quality_rejects": int(bayes_quality_rejects),
        "nonconformity_rejects": int(nonconformity_rejects),
        "nonconformity_overrides": int(nonconformity_overrides),
        "bayes_quality_ready_checks": int(bayes_quality_ready_checks),
        "nonconformity_ready_checks": int(nonconformity_ready_checks),
        "nonconformity_winner_ref_count": int(len(winner_nonconformity_scores)),
        "intelligence_mode_bars": int(intelligence_mode_bars),
        "intelligence_side_switches": int(intelligence_side_switches),
        "intelligence_avg_score": round(float(intelligence_score_sum / max(intelligence_mode_bars, 1)), 6),
        "time_adaptive_mode_bars": int(time_adaptive_mode_bars),
        "time_adaptive_ready_bars": int(time_adaptive_ready_bars),
        "time_adaptive_side_switches": int(time_adaptive_side_switches),
        "time_adaptive_avg_edge_adjust": round(
            float(time_adaptive_edge_adjust_sum / max(time_adaptive_mode_bars, 1)), 6
        ),
        "time_adaptive_avg_conf_adjust": round(
            float(time_adaptive_conf_adjust_sum / max(time_adaptive_mode_bars, 1)), 6
        ),
        "micro_change_mode_bars": int(micro_change_mode_bars),
        "micro_change_mode_rate": round(float(micro_change_mode_bars / max(decision_bars, 1)), 4),
        "micro_change_switches": int(micro_change_switches),
        "micro_change_avg_score": round(float(micro_change_score_sum / max(micro_change_mode_bars, 1)), 6),
        "micro_change_avg_alignment": round(float(micro_change_alignment_sum / max(micro_change_mode_bars, 1)), 6),
        "calibration_intelligence_enable": bool(getattr(cfg, "calibration_intelligence_enable", True)),
        "calibration_ready_bars": int(calibration_ready_bars),
        "calibration_mode_bars": int(calibration_mode_bars),
        "calibration_mode_rate": round(float(calibration_mode_bars / max(decision_bars, 1)), 4),
        "calibration_protective_bars": int(calibration_protective_bars),
        "calibration_opportunistic_bars": int(calibration_opportunistic_bars),
        "calibration_avg_abs_error": round(
            float(calibration_abs_error_sum / max(calibration_ready_bars, 1)), 6
        ),
        "calibration_avg_brier": round(
            float(calibration_brier_sum / max(calibration_ready_bars, 1)), 6
        ),
        "calibration_avg_hit_rate": round(
            float(calibration_hit_rate_sum / max(calibration_ready_bars, 1)), 6
        ),
        "calibration_avg_confidence_gap": round(
            float(calibration_gap_sum / max(calibration_ready_bars, 1)), 6
        ),
        "calibration_avg_mode_score": round(
            float(calibration_mode_score_sum / max(calibration_mode_bars, 1)), 6
        ),
        "calibration_avg_participation_scale": round(
            float(calibration_participation_scale_sum / max(calibration_ready_bars, 1)), 6
        ),
        "time_adaptive_enable": bool(getattr(cfg, "time_adaptive_enable", True)),
        "precision_selective_enable": bool(getattr(cfg, "precision_selective_enable", False)),
        "precision_selective_mode_bars": int(precision_selective_mode_bars),
        "precision_selective_ready_bars": int(precision_selective_ready_bars),
        "precision_selective_rejects": int(precision_selective_rejects),
        "precision_selective_avg_threshold": round(
            float(precision_selective_threshold_sum / max(precision_selective_ready_bars, 1)), 6
        ),
        "precision_selective_avg_quality": round(
            float(precision_selective_quality_sum / max(precision_selective_ready_bars, 1)), 6
        ),
        "precision_selective_avg_quantile": round(
            float(precision_selective_quantile_sum / max(precision_selective_ready_bars, 1)), 6
        ),
        "precision_selective_quality_lift_checks": int(precision_selective_quality_lift_checks),
        "precision_selective_avg_quality_lift": round(
            float(precision_selective_quality_lift_sum / max(precision_selective_quality_lift_checks, 1)), 6
        ),
        "month_shield_mode_bars": int(month_shield_mode_bars),
        "time_bucket_day_side": day_report,
        "time_bucket_hour_side": hour_report,
        "time_bucket_best_short_hours": best_short_hours,
        "time_bucket_worst_short_hours": worst_short_hours,
        "time_bucket_best_long_hours": best_long_hours,
        "time_bucket_worst_long_hours": worst_long_hours,
        "time_bucket_best_short_days": best_short_days,
        "time_bucket_worst_short_days": worst_short_days,
        "opportunity_rescue_bars": int(opportunity_rescue_bars),
        "opportunity_max_drought_ratio": round(float(opportunity_max_drought_ratio), 6),
        "opportunity_override_counterfactual": int(opportunity_override_counterfactual),
        "opportunity_override_bayes": int(opportunity_override_bayes),
        "opportunity_override_nonconformity": int(opportunity_override_nonconformity),
        "participation_relaxed_bars": int(participation_relaxed_bars),
        "participation_relax_rate": round(float(participation_relaxed_bars / max(decision_bars, 1)), 4),
        "participation_avg_pressure": round(float(participation_pressure_sum / max(decision_bars, 1)), 4),
        "participation_override_counterfactual": int(participation_override_counterfactual),
        "tp_mult_avg": round(float(np.mean(np.asarray(used_tp_mults, dtype=np.float64))) if used_tp_mults else float(cfg.tp_mult), 4),
        "tp_mult_min": round(float(np.min(np.asarray(used_tp_mults, dtype=np.float64))) if used_tp_mults else float(cfg.tp_mult), 4),
        "tp_mult_max": round(float(np.max(np.asarray(used_tp_mults, dtype=np.float64))) if used_tp_mults else float(cfg.tp_mult), 4),
        "sl_mult_avg": round(float(np.mean(np.asarray(used_sl_mults, dtype=np.float64))) if used_sl_mults else float(cfg.sl_mult), 4),
        "sl_mult_min": round(float(np.min(np.asarray(used_sl_mults, dtype=np.float64))) if used_sl_mults else float(cfg.sl_mult), 4),
        "sl_mult_max": round(float(np.max(np.asarray(used_sl_mults, dtype=np.float64))) if used_sl_mults else float(cfg.sl_mult), 4),
        "long_tp_mult_avg": round(float(np.mean(np.asarray(long_tp_mults, dtype=np.float64))) if long_tp_mults else float(cfg.tp_mult), 4),
        "short_tp_mult_avg": round(float(np.mean(np.asarray(short_tp_mults, dtype=np.float64))) if short_tp_mults else float(cfg.tp_mult), 4),
        "long_sl_mult_avg": round(float(np.mean(np.asarray(long_sl_mults, dtype=np.float64))) if long_sl_mults else float(cfg.sl_mult), 4),
        "short_sl_mult_avg": round(float(np.mean(np.asarray(short_sl_mults, dtype=np.float64))) if short_sl_mults else float(cfg.sl_mult), 4),
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
            "[MYTHOS][FOLD %d] trades=%d totalR=%+.2f status=%s long/short=%d/%d sure=%d sure_win=%s sure_lev=%d sure_lev_hits=%d lev_boost=%d lev_blocked=%d bayes_rej=%d nonconf_rej=%d nonconf_ovr=%d meta_soft=%d intel_mode=%d intel_switch=%d intel_avg=%s time_mode=%d time_switch=%d time_edge_adj=%s prec_mode=%d prec_rej=%d prec_q=%s opp_bars=%d opp_drought_max=%s opp_ovr(cf/bq/nc)=%d/%d/%d",
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
            fold.get("meta_soft_blocks", 0),
            fold.get("intelligence_mode_bars", 0),
            fold.get("intelligence_side_switches", 0),
            fold.get("intelligence_avg_score", 0.0),
            fold.get("time_adaptive_mode_bars", 0),
            fold.get("time_adaptive_side_switches", 0),
            fold.get("time_adaptive_avg_edge_adjust", 0.0),
            fold.get("precision_selective_mode_bars", 0),
            fold.get("precision_selective_rejects", 0),
            fold.get("precision_selective_avg_quantile", 0.0),
            fold.get("opportunity_rescue_bars", 0),
            fold.get("opportunity_max_drought_ratio", 0.0),
            fold.get("opportunity_override_counterfactual", 0),
            fold.get("opportunity_override_bayes", 0),
            fold.get("opportunity_override_nonconformity", 0),
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
        "precision_selective_reject": int(
            sum(int(r.get("skip_reasons", {}).get("precision_selective_reject", 0)) for r in reports)
        ),
        "precision_uncertainty_cap": int(
            sum(int(r.get("skip_reasons", {}).get("precision_uncertainty_cap", 0)) for r in reports)
        ),
        "counterfactual_reject": int(sum(int(r.get("skip_reasons", {}).get("counterfactual_reject", 0)) for r in reports)),
        "bayes_quality_reject": int(sum(int(r.get("skip_reasons", {}).get("bayes_quality_reject", 0)) for r in reports)),
        "nonconformity_reject": int(sum(int(r.get("skip_reasons", {}).get("nonconformity_reject", 0)) for r in reports)),
        "risk_reject": int(sum(int(r.get("skip_reasons", {}).get("risk_reject", 0)) for r in reports)),
        "meta_reject": int(sum(int(r.get("skip_reasons", {}).get("meta_reject", 0)) for r in reports)),
        "low_conviction": int(sum(int(r.get("skip_reasons", {}).get("low_conviction", 0)) for r in reports)),
    }
    total_decision_bars = int(sum(int(r.get("decision_bars", 0)) for r in reports))
    mode_denom = max(total_decision_bars, 1)
    change_mode_rate = float(total_change_mode_bars / mode_denom)
    total_transition_mode_bars = int(sum(r.get("transition_mode_bars", 0) for r in reports))
    total_meta_mode_bars = int(sum(r.get("meta_mode_bars", 0) for r in reports))
    total_meta_soft_blocks = int(sum(r.get("meta_soft_blocks", 0) for r in reports))
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
        sum(float(r.get("intelligence_avg_score", 0.0)) * int(r.get("intelligence_mode_bars", 0)) for r in reports)
    )
    aggregate_intelligence_avg_score = float(weighted_intelligence_score_num / max(total_intelligence_mode_bars, 1))
    total_time_adaptive_mode_bars = int(sum(int(r.get("time_adaptive_mode_bars", 0)) for r in reports))
    total_time_adaptive_ready_bars = int(sum(int(r.get("time_adaptive_ready_bars", 0)) for r in reports))
    total_time_adaptive_side_switches = int(sum(int(r.get("time_adaptive_side_switches", 0)) for r in reports))
    weighted_time_edge_adj = float(
        sum(
            float(r.get("time_adaptive_avg_edge_adjust", 0.0)) * int(r.get("time_adaptive_mode_bars", 0))
            for r in reports
        )
    )
    weighted_time_conf_adj = float(
        sum(
            float(r.get("time_adaptive_avg_conf_adjust", 0.0)) * int(r.get("time_adaptive_mode_bars", 0))
            for r in reports
        )
    )
    aggregate_time_edge_adj = float(weighted_time_edge_adj / max(total_time_adaptive_mode_bars, 1))
    aggregate_time_conf_adj = float(weighted_time_conf_adj / max(total_time_adaptive_mode_bars, 1))
    total_micro_change_mode_bars = int(sum(int(r.get("micro_change_mode_bars", 0)) for r in reports))
    total_micro_change_switches = int(sum(int(r.get("micro_change_switches", 0)) for r in reports))
    weighted_micro_score = float(
        sum(float(r.get("micro_change_avg_score", 0.0)) * int(r.get("micro_change_mode_bars", 0)) for r in reports)
    )
    weighted_micro_alignment = float(
        sum(float(r.get("micro_change_avg_alignment", 0.0)) * int(r.get("micro_change_mode_bars", 0)) for r in reports)
    )
    aggregate_micro_score = float(weighted_micro_score / max(total_micro_change_mode_bars, 1))
    aggregate_micro_alignment = float(weighted_micro_alignment / max(total_micro_change_mode_bars, 1))
    total_calibration_ready_bars = int(sum(int(r.get("calibration_ready_bars", 0)) for r in reports))
    total_calibration_mode_bars = int(sum(int(r.get("calibration_mode_bars", 0)) for r in reports))
    total_calibration_protective_bars = int(
        sum(int(r.get("calibration_protective_bars", 0)) for r in reports)
    )
    total_calibration_opportunistic_bars = int(
        sum(int(r.get("calibration_opportunistic_bars", 0)) for r in reports)
    )
    weighted_calibration_abs_error = float(
        sum(
            float(r.get("calibration_avg_abs_error", 0.0)) * int(r.get("calibration_ready_bars", 0))
            for r in reports
        )
    )
    weighted_calibration_brier = float(
        sum(float(r.get("calibration_avg_brier", 0.0)) * int(r.get("calibration_ready_bars", 0)) for r in reports)
    )
    weighted_calibration_hit_rate = float(
        sum(
            float(r.get("calibration_avg_hit_rate", 0.0)) * int(r.get("calibration_ready_bars", 0))
            for r in reports
        )
    )
    weighted_calibration_gap = float(
        sum(
            float(r.get("calibration_avg_confidence_gap", 0.0)) * int(r.get("calibration_ready_bars", 0))
            for r in reports
        )
    )
    weighted_calibration_mode_score = float(
        sum(
            float(r.get("calibration_avg_mode_score", 0.0)) * int(r.get("calibration_mode_bars", 0))
            for r in reports
        )
    )
    weighted_calibration_participation_scale = float(
        sum(
            float(r.get("calibration_avg_participation_scale", 1.0)) * int(r.get("calibration_ready_bars", 0))
            for r in reports
        )
    )
    aggregate_calibration_abs_error = float(
        weighted_calibration_abs_error / max(total_calibration_ready_bars, 1)
    )
    aggregate_calibration_brier = float(
        weighted_calibration_brier / max(total_calibration_ready_bars, 1)
    )
    aggregate_calibration_hit_rate = float(
        weighted_calibration_hit_rate / max(total_calibration_ready_bars, 1)
    )
    aggregate_calibration_gap = float(
        weighted_calibration_gap / max(total_calibration_ready_bars, 1)
    )
    aggregate_calibration_mode_score = float(
        weighted_calibration_mode_score / max(total_calibration_mode_bars, 1)
    )
    aggregate_calibration_participation_scale = float(
        weighted_calibration_participation_scale / max(total_calibration_ready_bars, 1)
    )
    total_precision_mode_bars = int(sum(int(r.get("precision_selective_mode_bars", 0)) for r in reports))
    total_precision_ready_bars = int(sum(int(r.get("precision_selective_ready_bars", 0)) for r in reports))
    total_precision_rejects = int(sum(int(r.get("precision_selective_rejects", 0)) for r in reports))
    total_month_shield_mode_bars = int(sum(int(r.get("month_shield_mode_bars", 0)) for r in reports))
    weighted_precision_threshold = float(
        sum(
            float(r.get("precision_selective_avg_threshold", 0.0)) * int(r.get("precision_selective_ready_bars", 0))
            for r in reports
        )
    )
    weighted_precision_quality = float(
        sum(
            float(r.get("precision_selective_avg_quality", 0.0)) * int(r.get("precision_selective_ready_bars", 0))
            for r in reports
        )
    )
    weighted_precision_quantile = float(
        sum(
            float(r.get("precision_selective_avg_quantile", 0.0)) * int(r.get("precision_selective_ready_bars", 0))
            for r in reports
        )
    )
    total_precision_quality_lift_checks = int(
        sum(int(r.get("precision_selective_quality_lift_checks", 0)) for r in reports)
    )
    weighted_precision_quality_lift = float(
        sum(
            float(r.get("precision_selective_avg_quality_lift", 0.0))
            * int(r.get("precision_selective_quality_lift_checks", 0))
            for r in reports
        )
    )
    aggregate_precision_threshold = float(weighted_precision_threshold / max(total_precision_ready_bars, 1))
    aggregate_precision_quality = float(weighted_precision_quality / max(total_precision_ready_bars, 1))
    aggregate_precision_quantile = float(weighted_precision_quantile / max(total_precision_ready_bars, 1))
    aggregate_precision_quality_lift = float(
        weighted_precision_quality_lift / max(total_precision_quality_lift_checks, 1)
    )
    total_opportunity_rescue_bars = int(sum(int(r.get("opportunity_rescue_bars", 0)) for r in reports))
    total_opportunity_override_counterfactual = int(
        sum(int(r.get("opportunity_override_counterfactual", 0)) for r in reports)
    )
    total_opportunity_override_bayes = int(sum(int(r.get("opportunity_override_bayes", 0)) for r in reports))
    total_opportunity_override_nonconformity = int(
        sum(int(r.get("opportunity_override_nonconformity", 0)) for r in reports)
    )
    total_participation_relaxed_bars = int(sum(int(r.get("participation_relaxed_bars", 0)) for r in reports))
    total_participation_override_counterfactual = int(
        sum(int(r.get("participation_override_counterfactual", 0)) for r in reports)
    )
    avg_participation_pressure = float(
        sum(float(r.get("participation_avg_pressure", 0.0)) * int(r.get("decision_bars", 0)) for r in reports)
        / max(total_decision_bars, 1)
    )
    max_opportunity_drought_ratio = float(
        max([float(r.get("opportunity_max_drought_ratio", 0.0)) for r in reports], default=0.0)
    )
    avg_tp_mult = float(
        sum(float(r.get("tp_mult_avg", cfg.tp_mult)) * int(r.get("total_trades", 0)) for r in reports)
        / max(total_trades, 1)
    )
    avg_sl_mult = float(
        sum(float(r.get("sl_mult_avg", cfg.sl_mult)) * int(r.get("total_trades", 0)) for r in reports)
        / max(total_trades, 1)
    )
    avg_long_tp_mult = float(
        sum(float(r.get("long_tp_mult_avg", cfg.tp_mult)) * int(r.get("long_trades", 0)) for r in reports)
        / max(total_long_trades, 1)
    )
    avg_short_tp_mult = float(
        sum(float(r.get("short_tp_mult_avg", cfg.tp_mult)) * int(r.get("short_trades", 0)) for r in reports)
        / max(total_short_trades, 1)
    )
    avg_long_sl_mult = float(
        sum(float(r.get("long_sl_mult_avg", cfg.sl_mult)) * int(r.get("long_trades", 0)) for r in reports)
        / max(total_long_trades, 1)
    )
    avg_short_sl_mult = float(
        sum(float(r.get("short_sl_mult_avg", cfg.sl_mult)) * int(r.get("short_trades", 0)) for r in reports)
        / max(total_short_trades, 1)
    )
    aggregate_day_buckets = _merge_time_bucket_reports(
        reports,
        field="time_bucket_day_side",
        ordered_keys=list(_DAY_NAMES),
    )
    aggregate_hour_buckets = _merge_time_bucket_reports(
        reports,
        field="time_bucket_hour_side",
        ordered_keys=[f"{h:02d}" for h in range(24)],
    )
    bucket_min_trades = int(max(getattr(cfg, "time_adaptive_min_bucket_trades", 8), 1))
    bucket_top_n = int(max(getattr(cfg, "time_adaptive_report_top_n", 6), 1))
    agg_best_short_hours = _rank_time_bucket_side(
        aggregate_hour_buckets,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    agg_worst_short_hours = _rank_time_bucket_side(
        aggregate_hour_buckets,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    agg_best_long_hours = _rank_time_bucket_side(
        aggregate_hour_buckets,
        side_key="long",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    agg_worst_long_hours = _rank_time_bucket_side(
        aggregate_hour_buckets,
        side_key="long",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    agg_best_short_days = _rank_time_bucket_side(
        aggregate_day_buckets,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=True,
    )
    agg_worst_short_days = _rank_time_bucket_side(
        aggregate_day_buckets,
        side_key="short",
        min_trades=bucket_min_trades,
        top_n=bucket_top_n,
        reverse=False,
    )
    cf_reject_rate = float(total_cf_rejects / max(total_cf_rejects + total_trades, 1))
    bayes_quality_reject_rate = float(
        total_bayes_quality_rejects / max(total_bayes_quality_rejects + total_trades, 1)
    )
    nonconformity_reject_rate = float(
        total_nonconformity_rejects / max(total_nonconformity_rejects + total_trades, 1)
    )
    robust_validation = _robust_validation_report(folds=reports, cfg=cfg)
    robust_ready = bool(robust_validation.get("ready", False))
    robust_pbo = float(robust_validation.get("pbo_overfit_probability", 0.0))
    robust_dsr = float(robust_validation.get("dsr", 0.0))
    robust_psr = float(robust_validation.get("psr", 0.0))
    robust_spa_p = float(robust_validation.get("spa_p_value", 1.0))
    act = sum(1 for r in reports if r["status"] == "ACTIVE")
    low = sum(1 for r in reports if r["status"] == "LOW_CONF")
    dead = sum(1 for r in reports if r["status"] == "DEAD")
    aggregate = {
        "symbol": sym,
        "folds": len(reports),
        "n_folds": len(reports),
        "decision_bars": total_decision_bars,
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
        "transition_mode_rate": round(float(total_transition_mode_bars / mode_denom), 4),
        "meta_mode_bars": total_meta_mode_bars,
        "meta_mode_rate": round(float(total_meta_mode_bars / mode_denom), 4),
        "meta_soft_blocks": total_meta_soft_blocks,
        "meta_soft_block_rate": round(float(total_meta_soft_blocks / mode_denom), 4),
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
        "flip_pressure_rate": round(float(total_flip_pressure_bars / mode_denom), 4),
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
        "intelligence_mode_rate": round(float(total_intelligence_mode_bars / mode_denom), 4),
        "intelligence_side_switches": total_intelligence_side_switches,
        "intelligence_avg_score": round(aggregate_intelligence_avg_score, 6),
        "time_adaptive_enable": bool(getattr(cfg, "time_adaptive_enable", True)),
        "time_adaptive_mode_bars": total_time_adaptive_mode_bars,
        "time_adaptive_mode_rate": round(float(total_time_adaptive_mode_bars / mode_denom), 4),
        "time_adaptive_ready_bars": total_time_adaptive_ready_bars,
        "time_adaptive_side_switches": total_time_adaptive_side_switches,
        "time_adaptive_avg_edge_adjust": round(aggregate_time_edge_adj, 6),
        "time_adaptive_avg_conf_adjust": round(aggregate_time_conf_adj, 6),
        "micro_change_mode_bars": total_micro_change_mode_bars,
        "micro_change_mode_rate": round(float(total_micro_change_mode_bars / mode_denom), 4),
        "micro_change_switches": total_micro_change_switches,
        "micro_change_avg_score": round(aggregate_micro_score, 6),
        "micro_change_avg_alignment": round(aggregate_micro_alignment, 6),
        "calibration_intelligence_enable": bool(getattr(cfg, "calibration_intelligence_enable", True)),
        "calibration_ready_bars": total_calibration_ready_bars,
        "calibration_mode_bars": total_calibration_mode_bars,
        "calibration_mode_rate": round(float(total_calibration_mode_bars / mode_denom), 4),
        "calibration_protective_bars": total_calibration_protective_bars,
        "calibration_opportunistic_bars": total_calibration_opportunistic_bars,
        "calibration_avg_abs_error": round(aggregate_calibration_abs_error, 6),
        "calibration_avg_brier": round(aggregate_calibration_brier, 6),
        "calibration_avg_hit_rate": round(aggregate_calibration_hit_rate, 6),
        "calibration_avg_confidence_gap": round(aggregate_calibration_gap, 6),
        "calibration_avg_mode_score": round(aggregate_calibration_mode_score, 6),
        "calibration_avg_participation_scale": round(aggregate_calibration_participation_scale, 6),
        "precision_selective_enable": bool(getattr(cfg, "precision_selective_enable", False)),
        "precision_selective_mode_bars": total_precision_mode_bars,
        "precision_selective_mode_rate": round(float(total_precision_mode_bars / mode_denom), 4),
        "precision_selective_ready_bars": total_precision_ready_bars,
        "precision_selective_rejects": total_precision_rejects,
        "precision_selective_reject_rate": round(
            float(total_precision_rejects / max(total_precision_rejects + total_trades, 1)), 4
        ),
        "month_shield_mode_bars": total_month_shield_mode_bars,
        "month_shield_mode_rate": round(float(total_month_shield_mode_bars / mode_denom), 4),
        "precision_selective_avg_threshold": round(aggregate_precision_threshold, 6),
        "precision_selective_avg_quality": round(aggregate_precision_quality, 6),
        "precision_selective_avg_quantile": round(aggregate_precision_quantile, 6),
        "precision_selective_quality_lift_checks": total_precision_quality_lift_checks,
        "precision_selective_avg_quality_lift": round(aggregate_precision_quality_lift, 6),
        "time_bucket_day_side": aggregate_day_buckets,
        "time_bucket_hour_side": aggregate_hour_buckets,
        "time_bucket_best_short_hours": agg_best_short_hours,
        "time_bucket_worst_short_hours": agg_worst_short_hours,
        "time_bucket_best_long_hours": agg_best_long_hours,
        "time_bucket_worst_long_hours": agg_worst_long_hours,
        "time_bucket_best_short_days": agg_best_short_days,
        "time_bucket_worst_short_days": agg_worst_short_days,
        "opportunity_rescue_bars": total_opportunity_rescue_bars,
        "opportunity_rescue_rate": round(float(total_opportunity_rescue_bars / mode_denom), 4),
        "opportunity_max_drought_ratio": round(max_opportunity_drought_ratio, 6),
        "opportunity_override_counterfactual": total_opportunity_override_counterfactual,
        "opportunity_override_bayes": total_opportunity_override_bayes,
        "opportunity_override_nonconformity": total_opportunity_override_nonconformity,
        "participation_relaxed_bars": total_participation_relaxed_bars,
        "participation_relax_rate": round(float(total_participation_relaxed_bars / mode_denom), 4),
        "participation_avg_pressure": round(avg_participation_pressure, 4),
        "participation_override_counterfactual": total_participation_override_counterfactual,
        "robust_validation_enable": bool(getattr(cfg, "robust_validation_enable", True)),
        "robust_validation_ready": robust_ready,
        "robust_validation_paths": int(robust_validation.get("paths_evaluated", 0) or 0),
        "robust_validation_pbo": round(robust_pbo, 6),
        "robust_validation_dsr": round(robust_dsr, 6),
        "robust_validation_psr": round(robust_psr, 6),
        "robust_validation_spa_p_value": round(robust_spa_p, 6),
        "robust_validation_significant_edge_95": bool(
            robust_validation.get("significant_edge_95", False)
        ),
        "robust_validation": robust_validation,
        "adaptive_tp_sl_enable": bool(getattr(cfg, "adaptive_tp_sl_enable", True)),
        "tp_mult_avg": round(avg_tp_mult, 4),
        "sl_mult_avg": round(avg_sl_mult, 4),
        "long_tp_mult_avg": round(avg_long_tp_mult, 4),
        "short_tp_mult_avg": round(avg_short_tp_mult, 4),
        "long_sl_mult_avg": round(avg_long_sl_mult, 4),
        "short_sl_mult_avg": round(avg_short_sl_mult, 4),
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
