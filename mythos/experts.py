from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from mythos.config import MythosConfig


@dataclass
class ExpertOutput:
    side: int
    expected_r: float
    confidence: float
    uncertainty: float
    score: float


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x))


class TrendLongExpert:
    name = "trend_long"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        mu = max(row.get("ret_96", 0.0), 0.0)
        mom = max(row.get("mom_16", 0.0), 0.0)
        vol_pen = max(row.get("vol_96", 0.0) - row.get("vol_384", 0.0), 0.0)
        exp_r = 12.0 * mu + 4.0 * mom - 2.5 * vol_pen
        conf = float(np.clip(0.45 + 20.0 * mu + 8.0 * mom, 0.0, 1.0))
        unc = float(np.clip(0.20 + 6.0 * vol_pen, 0.0, 1.0))
        score = exp_r * (0.5 + conf) - unc
        return ExpertOutput(side=1, expected_r=float(exp_r), confidence=conf, uncertainty=unc, score=float(score))


class TrendShortExpert:
    name = "trend_short"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        mu = max(-row.get("ret_96", 0.0), 0.0)
        mom = max(-row.get("mom_16", 0.0), 0.0)
        vol_pen = max(row.get("vol_96", 0.0) - row.get("vol_384", 0.0), 0.0)
        exp_r = 12.0 * mu + 4.0 * mom - 2.5 * vol_pen
        conf = float(np.clip(0.45 + 20.0 * mu + 8.0 * mom, 0.0, 1.0))
        unc = float(np.clip(0.20 + 6.0 * vol_pen, 0.0, 1.0))
        score = exp_r * (0.5 + conf) - unc
        return ExpertOutput(side=-1, expected_r=float(exp_r), confidence=conf, uncertainty=unc, score=float(score))


class MeanReversionExpert:
    name = "mean_reversion"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        z = row.get("zscore_64", 0.0)
        side = -1 if z > 0 else 1
        exp_r = 0.7 * abs(z) - 0.12
        conf = float(np.clip(0.35 + 0.18 * abs(z), 0.0, 1.0))
        unc = float(np.clip(0.30 - 0.04 * abs(z), 0.05, 1.0))
        score = exp_r * (0.4 + conf) - unc
        return ExpertOutput(side=side, expected_r=float(exp_r), confidence=conf, uncertainty=unc, score=float(score))


class BreakoutExpert:
    name = "breakout"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        br = row.get("range_break_48", 0.0)
        side = 1 if row.get("ret_16", 0.0) >= 0 else -1
        exp_r = 0.9 * br - 0.1
        conf = float(np.clip(0.30 + 0.60 * br, 0.0, 1.0))
        unc = float(np.clip(0.45 - 0.2 * br, 0.05, 1.0))
        score = exp_r * (0.4 + conf) - unc
        return ExpertOutput(side=side, expected_r=float(exp_r), confidence=conf, uncertainty=unc, score=float(score))


class DefensiveExpert:
    name = "defensive"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        return ExpertOutput(side=0, expected_r=0.0, confidence=1.0, uncertainty=0.1, score=0.0)


class ExpertCouncil:
    def __init__(self, cfg: MythosConfig):
        self.cfg = cfg
        self.experts = {
            "trend_long": TrendLongExpert(),
            "trend_short": TrendShortExpert(),
            "mean_reversion": MeanReversionExpert(),
            "breakout": BreakoutExpert(),
            "defensive": DefensiveExpert(),
        }
        self._recent_returns: Dict[str, list[float]] = {k: [] for k in self.experts.keys()}

    def update_expert_return(self, expert: str, realized_r: float) -> None:
        arr = self._recent_returns.get(expert)
        if arr is None:
            return
        arr.append(float(realized_r))
        if len(arr) > self.cfg.lookback_analogs:
            del arr[0]

    def reliability(self, expert: str) -> float:
        arr = np.asarray(self._recent_returns.get(expert, []), dtype=float)
        if arr.size < 8:
            return 0.5
        return float(np.clip(0.5 + 0.25 * _safe_mean(arr), 0.0, 1.0))

    def infer_all(self, row: Dict[str, float]) -> Dict[str, ExpertOutput]:
        outputs: Dict[str, ExpertOutput] = {}
        for name, expert in self.experts.items():
            out = expert.infer(row, self.cfg)
            rel = self.reliability(name)
            calibrated_score = out.score * (0.6 + 0.8 * rel)
            outputs[name] = ExpertOutput(
                side=out.side,
                expected_r=out.expected_r,
                confidence=float(np.clip(out.confidence * (0.75 + 0.5 * rel), 0.0, 1.0)),
                uncertainty=out.uncertainty,
                score=calibrated_score,
            )
        return outputs

