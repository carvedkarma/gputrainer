from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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

    def __init__(self, n_states: int = 6, random_state: int = 42) -> None:
        self.n_states = int(n_states)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_states, n_init=20, random_state=self.random_state)
        self.transition_matrix_: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, x: np.ndarray) -> "WorldModel":
        x_s = self.scaler.fit_transform(x)
        states = self.model.fit_predict(x_s)
        self.transition_matrix_ = self._build_transition_matrix(states)
        self.fitted_ = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x_s = self.scaler.transform(x)
        return self.model.predict(x_s)

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
