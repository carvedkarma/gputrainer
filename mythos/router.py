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

    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self._reliability: Dict[str, List[float]] = {}

    def update_reliability(self, expert_name: str, realized_r: float) -> None:
        history = self._reliability.setdefault(expert_name, [])
        history.append(float(realized_r))
        if len(history) > self.cfg.reliability_window:
            del history[0 : len(history) - self.cfg.reliability_window]

    def _reliability_score(self, expert_name: str) -> float:
        history = self._reliability.get(expert_name, [])
        if len(history) < 5:
            return 0.0
        arr = np.array(history, dtype=np.float64)
        return float(np.mean(arr) / (np.std(arr) + 1e-6))

    def select(self, regime: str, predictions: List[ExpertPrediction]) -> RouterDecision:
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
            rel = self._reliability_score(pred.expert_name)
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
        if best.confidence < self.cfg.min_confidence:
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

