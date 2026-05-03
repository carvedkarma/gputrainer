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


def _coerce_config(config_payload: Dict[str, object]) -> MythosConfig:
    if not isinstance(config_payload, dict):
        return MythosConfig()
    valid = {f.name for f in fields(MythosConfig)}
    filtered = {k: v for k, v in config_payload.items() if k in valid}
    return MythosConfig(**filtered)


class RuntimeWorldModel:
    def __init__(
        self,
        n_states: int,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        centers: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.n_states = int(n_states)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64).reshape(1, -1)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64).reshape(1, -1)
        self.scaler_scale = np.where(np.abs(self.scaler_scale) < 1e-12, 1.0, self.scaler_scale)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.transition_matrix = (
            np.asarray(transition_matrix, dtype=np.float64)
            if transition_matrix is not None
            else None
        )

    @classmethod
    def from_state_dict(cls, payload: Dict[str, object]) -> "RuntimeWorldModel":
        return cls(
            n_states=int(payload.get("n_states", 4)),
            scaler_mean=np.asarray(payload.get("scaler_mean", []), dtype=np.float64),
            scaler_scale=np.asarray(payload.get("scaler_scale", []), dtype=np.float64),
            centers=np.asarray(payload.get("kmeans_centers", []), dtype=np.float64),
            transition_matrix=payload.get("transition_matrix"),
        )

    def predict_regime(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx.reshape(1, -1)
        xx_scaled = (xx - self.scaler_mean) / self.scaler_scale
        diffs = xx_scaled[:, None, :] - self.centers[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)


class RuntimeLinearExpert:
    def __init__(self, name: str, side: int, coef: np.ndarray, bias: float, sigma: float):
        self.name = str(name)
        self.side = int(side)
        self._coef = np.asarray(coef, dtype=np.float64)
        self._bias = float(bias)
        self._sigma = float(max(sigma, 1e-4))

    @classmethod
    def from_state_dict(cls, payload: Dict[str, object]) -> "RuntimeLinearExpert":
        state = payload
        if isinstance(payload.get("fallback"), dict):
            state = payload["fallback"]
        return cls(
            name=str(payload.get("name", state.get("name", "expert"))),
            side=int(payload.get("side", state.get("side", 1))),
            coef=np.asarray(state.get("coef", []), dtype=np.float64),
            bias=float(state.get("bias", 0.0)),
            sigma=float(state.get("sigma", payload.get("sigma", 0.03))),
        )

    def predict_one(self, x: np.ndarray, regime: int) -> ExpertPrediction:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._coef.size == 0:
            mu = 0.0
        else:
            n = min(xx.shape[0], self._coef.shape[0])
            mu = float(np.dot(xx[:n], self._coef[:n]) + self._bias)
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
    def from_artifact(cls, model_path: Path) -> "MythosRuntimeModel":
        payload = json.loads(model_path.read_text())
        cfg = _coerce_config(payload.get("config", {}))
        wm = RuntimeWorldModel.from_state_dict(payload.get("world_model", {}))
        expert_states = payload.get("experts", [])
        experts = [RuntimeLinearExpert.from_state_dict(st) for st in expert_states if isinstance(st, dict)]
        if not experts:
            raise ValueError(f"No experts found in Mythos artifact: {model_path}")
        router = MetaRouter(cfg)
        router_state = payload.get("router") or {}
        router._reliability = {
            str(k): [float(vv) for vv in vals]
            for k, vals in (router_state.get("reliability") or {}).items()
            if isinstance(vals, list)
        }
        router._ema_reliability = {
            str(k): float(vv) for k, vv in (router_state.get("ema_reliability") or {}).items()
        }
        router._regime_reliability = {}
        for reg, reg_map in (router_state.get("regime_reliability") or {}).items():
            if not isinstance(reg_map, dict):
                continue
            rk = int(reg)
            router._regime_reliability[rk] = {}
            for ex_name, vals in reg_map.items():
                if isinstance(vals, list):
                    router._regime_reliability[rk][str(ex_name)] = [float(vv) for vv in vals]
        router._fitted = True

        feature_columns = payload.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns:
            n_features = int(wm.centers.shape[1]) if wm.centers.ndim == 2 and wm.centers.shape[1] > 0 else int(len(experts[0]._coef))
            # Legacy 7-factor artifacts used a different ordering than modern Mythos.
            if n_features <= 7:
                legacy_7 = ["ret_1", "ret_4", "ret_16", "vol_16", "vol_64", "zscore_64", "trend_ema"]
                feature_columns = legacy_7[:n_features]
            elif n_features <= len(MYTHOS_FEATURE_COLUMNS):
                feature_columns = list(MYTHOS_FEATURE_COLUMNS[:n_features])
            else:
                feature_columns = list(MYTHOS_FEATURE_COLUMNS) + [
                    f"feature_{i}" for i in range(len(MYTHOS_FEATURE_COLUMNS), n_features)
                ]
        else:
            feature_columns = [str(c) for c in feature_columns]

        return cls(
            cfg=cfg,
            world_model=wm,
            experts=experts,
            router=router,
            feature_columns=feature_columns,
            symbol=str(payload.get("symbol", "UNKNOWN")),
            artifact_path=Path(model_path),
        )

    def predict_from_feature_row(self, row: Dict[str, float]) -> Dict[str, object]:
        x = np.array([float(row.get(c, 0.0)) for c in self.feature_columns], dtype=np.float64)
        regime = int(self.world_model.predict_regime(x)[0])
        routed = self.router.route_one(
            x=x,
            regime=regime,
            experts=self.experts,
            vol_16=float(row.get("vol_16", 0.0)),
            trend_ema=float(row.get("trend_ema", 0.0)),
        )
        return {
            "side": int(routed.get("side", 0)),
            "edge": float(routed.get("edge", 0.0)),
            "confidence": float(np.clip(routed.get("confidence", 0.0), 0.0, 1.0)),
            "uncertainty": float(max(routed.get("uncertainty", 1.0), 0.0)),
            "abstain": bool(routed.get("abstain", False)),
            "reason": str(routed.get("reason", "selected")),
            "expert_name": str(routed.get("expert_name", "unknown")),
            "regime": regime,
        }
