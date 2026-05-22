from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from mythos.config import MythosConfig
from mythos.experts import ExpertPrediction


@dataclass
class RouterDecision:
    expert_name: str
    side: int
    expected_r: float
    uncertainty: float
    confidence: float
    abstain: bool
    reason: str


class MetaRouter:
    """
    Regime- and confidence-aware expert selector with abstention support.
    """

    def __init__(self, cfg: MythosConfig | int | None = None):
        # Backward compatibility: older call sites pass random_state (int).
        if isinstance(cfg, MythosConfig):
            self.cfg = cfg
        elif isinstance(cfg, int):
            self.cfg = MythosConfig(random_state=int(cfg))
        else:
            self.cfg = MythosConfig()
        self._reliability: Dict[str, List[float]] = {}
        self._regime_reliability: Dict[int, Dict[str, List[float]]] = {}
        self._ema_reliability: Dict[str, float] = {}
        self._fitted = False

    def update_reliability(self, expert_name: str, realized_r: float, regime: int | None = None) -> None:
        history = self._reliability.setdefault(expert_name, [])
        history.append(float(realized_r))
        if len(history) > self.cfg.reliability_window:
            del history[0 : len(history) - self.cfg.reliability_window]
        alpha = float(np.clip(getattr(self.cfg, "online_reliability_alpha", 0.08), 0.001, 0.95))
        prev_ema = float(self._ema_reliability.get(expert_name, 0.0))
        self._ema_reliability[expert_name] = (1.0 - alpha) * prev_ema + alpha * float(realized_r)
        if regime is not None:
            reg_key = int(regime)
            reg_map = self._regime_reliability.setdefault(reg_key, {})
            reg_hist = reg_map.setdefault(expert_name, [])
            reg_hist.append(float(realized_r))
            max_len = int(max(getattr(self.cfg, "reliability_regime_window", 80), 5))
            if len(reg_hist) > max_len:
                del reg_hist[0 : len(reg_hist) - max_len]

    def _reliability_score(self, expert_name: str, regime: int | None = None) -> float:
        history = self._reliability.get(expert_name, [])
        if len(history) < 5:
            base = 0.0
        else:
            arr = np.array(history, dtype=np.float64)
            base = float(np.mean(arr) / (np.std(arr) + 1e-6))
        ema = float(self._ema_reliability.get(expert_name, 0.0))
        reg_score = 0.0
        if regime is not None:
            reg_hist = self._regime_reliability.get(int(regime), {}).get(expert_name, [])
            if len(reg_hist) >= 4:
                reg_arr = np.array(reg_hist, dtype=np.float64)
                reg_score = float(np.mean(reg_arr) / (np.std(reg_arr) + 1e-6))
        return 0.55 * base + 0.25 * ema + 0.20 * reg_score

    def fit(self, train_feat, train_regime, experts) -> "MetaRouter":
        # Keep fit hook so walkforward/tests can share a sklearn-like flow.
        _ = (train_feat, train_regime, experts)
        self._fitted = True
        return self

    def route_one(
        self,
        x: np.ndarray,
        regime: int,
        experts,
        vol_16: float | None = None,
        trend_ema: float | None = None,
    ) -> Dict[str, float | int | str]:
        _ = (vol_16, trend_ema)
        if not self._fitted:
            self._fitted = True
        preds = [ex.predict_one(x, regime) for ex in experts]
        decision = self.select(regime=int(regime), predictions=preds)
        return {
            "expert_name": decision.expert_name,
            "side": int(decision.side),
            "edge": float(decision.expected_r),
            "confidence": float(decision.confidence),
            "uncertainty": float(decision.uncertainty),
            "abstain": bool(decision.abstain),
            "reason": decision.reason,
        }

    def select(self, regime: int, predictions: List[ExpertPrediction]) -> RouterDecision:
        if not predictions:
            return RouterDecision(
                expert_name="none",
                side=0,
                expected_r=0.0,
                uncertainty=1.0,
                confidence=0.0,
                abstain=True,
                reason="no_predictions",
            )
        scored: List[Tuple[float, ExpertPrediction]] = []
        for pred in predictions:
            regime_fit = 1.0 if regime in pred.regime_affinity else 0.7
            rel = self._reliability_score(pred.expert_name, regime=regime)
            score = (
                pred.expected_r
                - self.cfg.uncertainty_penalty * pred.uncertainty
                + self.cfg.router_reliability_weight * rel
            ) * regime_fit
            scored.append((score, pred))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        if best.expected_r < self.cfg.abstain_edge_floor:
            return RouterDecision(
                expert_name=best.expert_name,
                side=0,
                expected_r=best.expected_r,
                uncertainty=best.uncertainty,
                confidence=best.confidence,
                abstain=True,
                reason="edge_below_floor",
            )
        min_conf = float(np.clip(self.cfg.min_confidence, 0.0, 1.0))
        if best.confidence < min_conf:
            return RouterDecision(
                expert_name=best.expert_name,
                side=0,
                expected_r=best.expected_r,
                uncertainty=best.uncertainty,
                confidence=best.confidence,
                abstain=True,
                reason="low_confidence",
            )
        if best_score < 0:
            return RouterDecision(
                expert_name=best.expert_name,
                side=0,
                expected_r=best.expected_r,
                uncertainty=best.uncertainty,
                confidence=best.confidence,
                abstain=True,
                reason="negative_meta_score",
            )
        return RouterDecision(
            expert_name=best.expert_name,
            side=best.side,
            expected_r=best.expected_r,
            uncertainty=best.uncertainty,
            confidence=best.confidence,
            abstain=False,
            reason="selected",
        )

    def to_state_dict(self) -> Dict[str, object]:
        return {
            "reliability": {k: [float(v) for v in vals] for k, vals in self._reliability.items()},
            "ema_reliability": {k: float(v) for k, v in self._ema_reliability.items()},
            "regime_reliability": {
                int(reg): {k: [float(v) for v in vals] for k, vals in reg_map.items()}
                for reg, reg_map in self._regime_reliability.items()
            },
        }

