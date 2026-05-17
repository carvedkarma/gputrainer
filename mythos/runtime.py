from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import MythosConfig
from .experts import ExpertPrediction
from .features import MYTHOS_FEATURE_COLUMNS
from .router import MetaRouter


def _coerce_config(payload: Dict[str, object], *, strict: bool = False) -> MythosConfig:
    cfg_data = payload if isinstance(payload, dict) else {}
    allowed = {f.name for f in fields(MythosConfig)}
    if strict:
        unknown = sorted(str(k) for k in cfg_data.keys() if k not in allowed)
        if unknown:
            raise ValueError(
                "Mythos artifact config has unknown fields in strict mode: "
                + ", ".join(unknown)
            )
    clean = {k: v for k, v in cfg_data.items() if k in allowed}
    return MythosConfig(**clean)


class RuntimeWorldModel:
    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        centers: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.scaler_scale = np.where(np.abs(self.scaler_scale) < 1e-9, 1.0, self.scaler_scale)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.transition_matrix = None if transition_matrix is None else np.asarray(transition_matrix, dtype=np.float64)
        self.n_states = int(self.centers.shape[0]) if self.centers.ndim == 2 else 1
        self.n_features = int(self.centers.shape[1]) if self.centers.ndim == 2 else int(self.scaler_mean.shape[0])

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "RuntimeWorldModel":
        return cls(
            scaler_mean=np.asarray(state.get("scaler_mean", []), dtype=np.float64),
            scaler_scale=np.asarray(state.get("scaler_scale", []), dtype=np.float64),
            centers=np.asarray(state.get("kmeans_centers", []), dtype=np.float64),
            transition_matrix=np.asarray(state.get("transition_matrix"), dtype=np.float64)
            if state.get("transition_matrix") is not None
            else None,
        )

    def predict_regime(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        d = min(arr.shape[1], self.n_features)
        xx = arr[:, :d]
        mean = self.scaler_mean[:d] if self.scaler_mean.size >= d else np.zeros(d, dtype=np.float64)
        scale = self.scaler_scale[:d] if self.scaler_scale.size >= d else np.ones(d, dtype=np.float64)
        cen = self.centers[:, :d] if self.centers.ndim == 2 else np.zeros((1, d), dtype=np.float64)
        x_std = (xx - mean) / np.where(np.abs(scale) < 1e-9, 1.0, scale)
        d2 = np.sum((x_std[:, None, :] - cen[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)


class RuntimeLinearExpert:
    def __init__(self, name: str, side: int, coef: np.ndarray, bias: float, sigma: float) -> None:
        self.name = str(name)
        self.side = int(side)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.bias = float(bias)
        self.sigma = float(max(sigma, 1e-4))

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "RuntimeLinearExpert":
        if "coef" in state:
            return cls(
                name=str(state.get("name", "expert")),
                side=int(state.get("side", 1)),
                coef=np.asarray(state.get("coef", []), dtype=np.float64),
                bias=float(state.get("bias", 0.0)),
                sigma=float(state.get("sigma", 0.03)),
            )
        fallback = state.get("fallback", {}) if isinstance(state.get("fallback"), dict) else {}
        return cls(
            name=str(state.get("name", fallback.get("name", "expert"))),
            side=int(state.get("side", fallback.get("side", 1))),
            coef=np.asarray(fallback.get("coef", []), dtype=np.float64),
            bias=float(fallback.get("bias", 0.0)),
            sigma=float(state.get("sigma", fallback.get("sigma", 0.03))),
        )

    def predict_one(self, x: np.ndarray, regime: int) -> ExpertPrediction:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        d = min(xx.shape[0], self.coef.shape[0]) if self.coef.size else 0
        mu = float(np.dot(xx[:d], self.coef[:d]) + self.bias) if d > 0 else 0.0
        expected = max(mu, 0.0)
        uncertainty = float(np.clip(self.sigma * 0.03, 0.005, 0.12))
        snr = expected / max(uncertainty, 1e-6)
        confidence = float(np.clip(0.55 + 0.30 * np.tanh(1.5 * snr), 0.0, 1.0))
        return ExpertPrediction(
            expert_name=self.name,
            side=self.side,
            expected_r=expected,
            confidence=confidence,
            uncertainty=uncertainty,
            score=expected * confidence - uncertainty * 0.15,
            regime_affinity=[max(int(regime) - 1, 0), int(regime), int(regime) + 1],
        )


class MythosRuntimeModel:
    def __init__(
        self,
        cfg: MythosConfig,
        world_model: RuntimeWorldModel,
        experts: List[RuntimeLinearExpert],
        router: MetaRouter,
        feature_columns: List[str],
        symbol: str,
        artifact_path: Path,
    ) -> None:
        self.cfg = cfg
        self.world_model = world_model
        self.experts = experts
        self.router = router
        self.feature_columns = feature_columns
        self.symbol = symbol
        self.artifact_path = artifact_path
        self._is_mythos = True
        self._is_v5 = False
        self._is_v6 = False
        self.live_edge_threshold = float(max(getattr(cfg, "abstain_edge_floor", 0.01), 0.0))
        self.live_min_confidence = float(np.clip(getattr(cfg, "min_confidence", 0.5), 0.0, 1.0))

    @classmethod
    def from_artifact(cls, artifact_path: Path) -> "MythosRuntimeModel":
        path = Path(artifact_path)
        payload = json.loads(path.read_text())
        schema_version = int(max(payload.get("artifact_schema_version", 1), 1))
        strict_mode = bool(payload.get("runtime_strict_config", schema_version >= 2))
        cfg = _coerce_config(payload.get("config", {}), strict=strict_mode)
        if bool(getattr(cfg, "runtime_require_adaptive_brain", False)) and payload.get("adaptive_brain") is None:
            raise ValueError(
                f"Mythos artifact requires adaptive_brain for runtime parity but it is missing: {path}"
            )
        if str(payload.get("kind", "mythos_best_fold_model")) != "mythos_best_fold_model":
            raise ValueError(f"Invalid Mythos artifact kind: {path}")
        wm = RuntimeWorldModel.from_state_dict(payload.get("world_model", {}))
        experts = [RuntimeLinearExpert.from_state_dict(ex) for ex in payload.get("experts", [])]
        if not experts:
            raise ValueError(f"Mythos artifact has no experts: {path}")

        dim = int(max(experts[0].coef.shape[0], wm.n_features))
        if dim <= len(MYTHOS_FEATURE_COLUMNS):
            feature_columns = list(MYTHOS_FEATURE_COLUMNS[:dim])
        else:
            feature_columns = list(MYTHOS_FEATURE_COLUMNS) + [f"feat_{i}" for i in range(len(MYTHOS_FEATURE_COLUMNS), dim)]

        router = MetaRouter(cfg)
        router_state = payload.get("router", {}) if isinstance(payload.get("router"), dict) else {}
        router._reliability = {
            str(k): [float(v) for v in vals]
            for k, vals in (router_state.get("reliability", {}) or {}).items()
        }
        router._ema_reliability = {
            str(k): float(v) for k, v in (router_state.get("ema_reliability", {}) or {}).items()
        }
        router._regime_reliability = {
            int(reg): {str(k): [float(v) for v in vals] for k, vals in reg_map.items()}
            for reg, reg_map in (router_state.get("regime_reliability", {}) or {}).items()
        }
        router._fitted = True

        return cls(
            cfg=cfg,
            world_model=wm,
            experts=experts,
            router=router,
            feature_columns=feature_columns,
            symbol=str(payload.get("symbol", "BTCUSDT")),
            artifact_path=path,
        )

    def predict_from_feature_row(self, feature_row: Dict[str, float]) -> Dict[str, object]:
        x = np.array([float(feature_row.get(col, 0.0)) for col in self.feature_columns], dtype=np.float64)
        regime = int(self.world_model.predict_regime(x.reshape(1, -1))[0])
        preds = [ex.predict_one(x, regime=regime) for ex in self.experts]
        decision = self.router.select(regime=regime, predictions=preds)
        side_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
        return {
            "side": side_map.get(int(decision.side), "NEUTRAL"),
            "edge": float(decision.expected_r),
            "confidence": float(decision.confidence),
            "uncertainty": float(decision.uncertainty),
            "abstain": bool(decision.abstain),
            "reason": str(decision.reason),
            "expert_name": str(decision.expert_name),
            "regime": regime,
        }
