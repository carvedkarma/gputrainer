from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class WorldStateSnapshot:
    state_id: int
    transition_confidence: float
    state_distribution: Dict[int, float]


class WorldModel:
    """
    Lightweight latent-state world model.

    It maps market features into discrete latent states and tracks transition
    probabilities between those states for short-horizon regime awareness.
    """

    def __init__(self, random_state: int = 42, n_states: int = 4) -> None:
        self.random_state = int(random_state)
        self.n_states = int(n_states)
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_states, n_init=20, random_state=self.random_state)
        self.transition_matrix_: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, x) -> "WorldModel":
        xx = self._to_array(x)
        x_s = self.scaler.fit_transform(xx)
        states = self.model.fit_predict(x_s)
        self.transition_matrix_ = self._build_transition_matrix(states)
        self.fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        self._check_fitted()
        xx = self._to_array(x)
        x_s = self.scaler.transform(xx)
        return self.model.predict(x_s)

    def predict_regime(self, x) -> np.ndarray:
        return self.transform(x)

    def snapshot(self, recent_states: np.ndarray) -> WorldStateSnapshot:
        self._check_fitted()
        recent_states = np.asarray(recent_states, dtype=np.int64)
        if len(recent_states) == 0:
            return WorldStateSnapshot(state_id=0, transition_confidence=0.0, state_distribution={})
        state_id = int(recent_states[-1])
        counts = np.bincount(recent_states, minlength=self.n_states).astype(float)
        probs = counts / max(counts.sum(), 1.0)
        dist = {i: float(p) for i, p in enumerate(probs) if p > 0}
        trans_conf = 0.0
        if self.transition_matrix_ is not None and 0 <= state_id < self.transition_matrix_.shape[0]:
            trans_conf = float(np.max(self.transition_matrix_[state_id]))
        return WorldStateSnapshot(
            state_id=state_id,
            transition_confidence=trans_conf,
            state_distribution=dist,
        )

    def _build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        mat = np.zeros((self.n_states, self.n_states), dtype=float)
        if len(states) <= 1:
            return mat
        for i in range(len(states) - 1):
            mat[states[i], states[i + 1]] += 1.0
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return mat / row_sums

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("WorldModel must be fitted before use.")

    def _to_array(self, x) -> np.ndarray:
        if hasattr(x, "columns"):
            # DataFrame path
            cols = [
                c
                for c in [
                    "ret_1",
                    "ret_4",
                    "ret_16",
                    "ret_64",
                    "vol_16",
                    "vol_64",
                    "vol_256",
                    "zscore_64",
                    "trend_ema",
                    "trend_slope_8",
                    "range_break_48",
                    "atr_pct",
                    "rsi_14",
                    "adx_14",
                    "vol_z_128",
                ]
                if c in x.columns
            ]
            if not cols:
                cols = list(x.select_dtypes(include=["number"]).columns)
            arr = x[cols].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(x, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("WorldModel input must be 2D")
        return arr

    def to_state_dict(self) -> Dict[str, object]:
        self._check_fitted()
        return {
            "random_state": int(self.random_state),
            "n_states": int(self.n_states),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "kmeans_centers": self.model.cluster_centers_.tolist(),
            "transition_matrix": self.transition_matrix_.tolist() if self.transition_matrix_ is not None else None,
        }
