from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mythos.config import MythosConfig


@dataclass
class ExpertOutput:
    side: int
    expected_r: float
    confidence: float
    uncertainty: float
    score: float


@dataclass
class ExpertPrediction:
    expert_name: str
    side: int
    expected_r: float
    confidence: float
    uncertainty: float
    score: float
    regime_affinity: List[int]


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
        mu = max(row.get("ret_16", 0.0), 0.0)
        mom = max(row.get("ret_4", 0.0), 0.0)
        vol_pen = max(row.get("vol_16", 0.0) - row.get("vol_64", 0.0), 0.0)
        exp_r = 12.0 * mu + 4.0 * mom - 2.5 * vol_pen
        conf = float(np.clip(0.45 + 20.0 * mu + 8.0 * mom, 0.0, 1.0))
        unc = float(np.clip(0.20 + 6.0 * vol_pen, 0.0, 1.0))
        score = exp_r * (0.5 + conf) - unc
        return ExpertOutput(side=1, expected_r=float(exp_r), confidence=conf, uncertainty=unc, score=float(score))


class TrendShortExpert:
    name = "trend_short"

    def infer(self, row: Dict[str, float], cfg: MythosConfig) -> ExpertOutput:
        mu = max(-row.get("ret_16", 0.0), 0.0)
        mom = max(-row.get("ret_4", 0.0), 0.0)
        vol_pen = max(row.get("vol_16", 0.0) - row.get("vol_64", 0.0), 0.0)
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


class _SklearnLikeExpert:
    def __init__(self, name: str, side: int):
        self.name = name
        self.side = side
        self._coef = None
        self._bias = 0.0
        self._sigma = 1.0

    def fit(self, X: np.ndarray, y1: np.ndarray, y4: np.ndarray, y16: np.ndarray, regime_ids: np.ndarray) -> None:
        _ = (y1, regime_ids)
        # Convert raw forward returns into a volatility-normalized proxy so
        # expected_r lives in a practical "R-like" range for downstream gates.
        raw_target = 0.5 * y4 + 0.5 * y16
        vol_proxy = np.maximum(0.5 * (X[:, 3] + X[:, 4]), 1e-5)
        target = raw_target / (vol_proxy * np.sqrt(8.0))
        target = np.clip(target, -4.0, 4.0)
        if self.side < 0:
            target = -target
        if X.size == 0:
            self._coef = np.zeros(7, dtype=np.float64)
            self._bias = 0.0
            self._sigma = 1.0
            return
        XtX = X.T @ X + np.eye(X.shape[1]) * 1e-4
        self._coef = np.linalg.solve(XtX, X.T @ target)
        pred = X @ self._coef
        self._bias = float(np.mean(target - pred))
        self._sigma = float(np.std(target - pred) + 1e-4)

    def predict_one(self, x: np.ndarray, regime: int) -> ExpertPrediction:
        if self._coef is None:
            mu = 0.0
        else:
            mu = float(x @ self._coef + self._bias)
        expected = max(mu, 0.0)
        # Keep uncertainty calibrated near the edge scale used by the router.
        uncertainty = float(np.clip(self._sigma * 0.03, 0.005, 0.08))
        snr = expected / max(uncertainty, 1e-6)
        confidence = float(np.clip(0.55 + 0.30 * np.tanh(1.5 * snr), 0.0, 1.0))
        return ExpertPrediction(
            expert_name=self.name,
            side=self.side,
            expected_r=expected,
            confidence=confidence,
            uncertainty=uncertainty,
            score=expected * confidence - uncertainty * 0.15,
            regime_affinity=[max(regime - 1, 0), regime, regime + 1],
        )

    def to_state_dict(self) -> Dict[str, object]:
        coef = self._coef if self._coef is not None else np.zeros(7, dtype=np.float64)
        return {
            "name": self.name,
            "side": int(self.side),
            "coef": coef.tolist(),
            "bias": float(self._bias),
            "sigma": float(self._sigma),
        }


class _TorchNeuralExpert:
    def __init__(self, name: str, side: int, cfg: MythosConfig):
        self.name = name
        self.side = int(side)
        self.cfg = cfg
        self._model = None
        self._sigma = 0.03
        self._device = "cpu"
        self._active = False
        self._fallback = _SklearnLikeExpert(name=name, side=side)

    def fit(self, X: np.ndarray, y1: np.ndarray, y4: np.ndarray, y16: np.ndarray, regime_ids: np.ndarray) -> None:
        self._fallback.fit(X, y1, y4, y16, regime_ids)
        if X.size == 0:
            self._active = False
            return
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self._active = False
            return
        if not bool(getattr(self.cfg, "use_gpu_neural_expert", True)):
            self._active = False
            return
        has_cuda = bool(torch.cuda.is_available())
        if not has_cuda and not bool(getattr(self.cfg, "allow_cpu_neural_expert", False)):
            self._active = False
            return
        self._device = "cuda" if has_cuda else "cpu"
        x = torch.tensor(X, dtype=torch.float32, device=self._device)
        raw_target = 0.5 * y4 + 0.5 * y16
        vol_proxy = np.maximum(0.5 * (X[:, 3] + X[:, 4]), 1e-5)
        target = raw_target / (vol_proxy * np.sqrt(8.0))
        target = np.clip(target, -4.0, 4.0)
        if self.side < 0:
            target = -target
        y = torch.tensor(target.reshape(-1, 1), dtype=torch.float32, device=self._device)
        hidden = int(max(getattr(self.cfg, "neural_expert_hidden_dim", 64), 8))
        dropout = float(np.clip(getattr(self.cfg, "neural_expert_dropout", 0.1), 0.0, 0.8))
        self._model = nn.Sequential(
            nn.Linear(X.shape[1], hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        ).to(self._device)
        lr = float(max(getattr(self.cfg, "neural_expert_lr", 0.001), 1e-6))
        epochs = int(max(getattr(self.cfg, "neural_expert_epochs", 8), 1))
        weight_decay = float(max(getattr(self.cfg, "neural_expert_weight_decay", 1e-6), 0.0))
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.HuberLoss(delta=1.0)
        batch_size = int(max(getattr(self.cfg, "neural_expert_batch_size", 1024), 32))
        n = x.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=self._device)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xb = x[idx]
                yb = y[idx]
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = self._model(x).squeeze(-1)
            resid = y.squeeze(-1) - pred
            self._sigma = float(torch.std(resid).item() + 1e-4)
        self._active = True

    def predict_one(self, x: np.ndarray, regime: int) -> ExpertPrediction:
        if not self._active or self._model is None:
            return self._fallback.predict_one(x, regime)
        try:
            import torch
            xx = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self._device)
            with torch.no_grad():
                mu = float(self._model(xx).item())
        except Exception:
            return self._fallback.predict_one(x, regime)
        expected = max(mu, 0.0)
        uncertainty = float(np.clip(self._sigma * 0.03, 0.005, 0.12))
        snr = expected / max(uncertainty, 1e-6)
        confidence = float(np.clip(0.55 + 0.30 * np.tanh(1.5 * snr), 0.0, 1.0))
        return ExpertPrediction(
            expert_name=self.name,
            side=self.side,
            expected_r=expected,
            confidence=confidence,
            uncertainty=uncertainty,
            score=expected * confidence - uncertainty * 0.15,
            regime_affinity=[max(regime - 1, 0), regime, regime + 1],
        )

    def to_state_dict(self) -> Dict[str, object]:
        fallback = self._fallback.to_state_dict()
        return {
            "name": self.name,
            "side": int(self.side),
            "active": bool(self._active),
            "device": self._device,
            "sigma": float(self._sigma),
            "fallback": fallback,
        }


def _infer_feature_columns_from_fit_matrix(X: np.ndarray) -> List[str]:
    # Keep backward compatibility for older 7-factor layouts.
    if X.shape[1] <= 7:
        return ["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"][: X.shape[1]]
    # Rich multi-factor layout from mythos.features.MYTHOS_FEATURE_COLUMNS.
    return [
        "ret_1",
        "ret_4",
        "ret_16",
        "ret_64",
        "vol_16",
        "vol_64",
        "vol_ratio_16_64",
        "zscore_64",
        "zscore_128",
        "trend_ema",
        "trend_strength",
        "range_break_48",
        "atr_pct_14",
        "rsi_14",
        "adx_14",
    ][: X.shape[1]]


def build_experts(random_state: int | None = None, cfg: Optional[MythosConfig] = None):
    _ = random_state
    cfg = cfg or MythosConfig()
    ExpertCls = _TorchNeuralExpert if bool(getattr(cfg, "use_gpu_neural_expert", True)) else _SklearnLikeExpert
    if ExpertCls is _TorchNeuralExpert:
        return [
            ExpertCls("trend_long", side=1, cfg=cfg),
            ExpertCls("trend_short", side=-1, cfg=cfg),
            ExpertCls("mean_revert", side=1, cfg=cfg),
            ExpertCls("breakout", side=1, cfg=cfg),
        ]
    return [
        ExpertCls("trend_long", side=1),
        ExpertCls("trend_short", side=-1),
        ExpertCls("mean_revert", side=1),
        ExpertCls("breakout", side=1),
    ]

