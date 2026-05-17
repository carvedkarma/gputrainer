import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mythos.config import MythosConfig
from mythos.risk import RiskConstitution
from mythos.runtime import MythosRuntimeModel
from mythos.walkforward import _build_cpcv_splits, _spa_single_model


class MythosInstitutionalUpgradesTests(unittest.TestCase):
    def test_hard_caps_are_non_overridable_by_default(self):
        cfg = MythosConfig(
            daily_loss_cap_r=-1.0,
            weekly_loss_cap_r=-20.0,
            trailing_stop_r=-50.0,
            cooldown_bars=0,
            max_trades_per_day=100,
            risk_cap_override_enable=True,
            risk_hard_stops_non_overridable=True,
        )
        risk = RiskConstitution(cfg)
        risk.state.daily_r = -1.5
        risk.state.day_idx = 1
        risk.state.week_idx = 0
        allowed = risk.allow_trade(
            bar_index=100,
            side=1,
            edge=0.20,
            uncertainty=0.10,
            conviction=0.99,
        )
        self.assertFalse(allowed)

    def test_soft_override_path_only_when_hard_caps_disabled(self):
        cfg = MythosConfig(
            daily_loss_cap_r=-1.0,
            weekly_loss_cap_r=-20.0,
            trailing_stop_r=-50.0,
            cooldown_bars=0,
            max_trades_per_day=100,
            risk_cap_override_enable=True,
            risk_hard_stops_non_overridable=False,
            risk_cap_override_conviction=0.8,
            risk_cap_override_edge_buffer=0.002,
            risk_cap_override_max_uncertainty=0.7,
        )
        risk = RiskConstitution(cfg)
        risk.state.daily_r = -1.5
        risk.state.day_idx = 1
        risk.state.week_idx = 0
        allowed = risk.allow_trade(
            bar_index=100,
            side=1,
            edge=0.50,
            uncertainty=0.10,
            conviction=0.95,
        )
        self.assertTrue(allowed)

    def test_cpcv_splits_apply_purge_and_embargo(self):
        n_folds = 10
        splits = _build_cpcv_splits(
            n_folds=n_folds,
            test_size=2,
            max_paths=32,
            seed=11,
            purge_folds=1,
            embargo_folds=1,
        )
        self.assertGreater(len(splits), 0)
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            for t in test_set:
                for near in (t - 1, t, t + 1):
                    if 0 <= near < n_folds:
                        self.assertNotIn(near, train_set)
            post = max(test_set) + 1
            if post < n_folds:
                self.assertNotIn(post, train_set)

    def test_spa_block_bootstrap_runs_and_bounds_p_value(self):
        vals = np.array([0.12, -0.04, 0.03, 0.10, -0.02, 0.07, -0.01, 0.05], dtype=np.float64)
        out = _spa_single_model(vals, bootstrap_samples=64, seed=42, block_size=3)
        self.assertIn("p_value", out)
        self.assertGreaterEqual(float(out["p_value"]), 0.0)
        self.assertLessEqual(float(out["p_value"]), 1.0)

    def test_runtime_strict_config_rejects_unknown_keys(self):
        payload = {
            "kind": "mythos_best_fold_model",
            "artifact_schema_version": 2,
            "runtime_strict_config": True,
            "symbol": "BTCUSDT",
            "config": {
                "min_expected_r": 0.01,
                "unknown_runtime_field": 123,
            },
            "world_model": {
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "kmeans_centers": [[0.0]],
            },
            "experts": [
                {
                    "name": "expert_a",
                    "side": 1,
                    "coef": [0.1],
                    "bias": 0.0,
                    "sigma": 0.03,
                }
            ],
            "router": {"reliability": {}, "ema_reliability": {}, "regime_reliability": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict_artifact.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                MythosRuntimeModel.from_artifact(path)

    def test_runtime_backward_compat_still_loads_schema_v1(self):
        payload = {
            "kind": "mythos_best_fold_model",
            "artifact_schema_version": 1,
            "symbol": "BTCUSDT",
            "config": {
                "min_expected_r": 0.01,
                "unknown_runtime_field": 123,
            },
            "world_model": {
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "kmeans_centers": [[0.0]],
            },
            "experts": [
                {
                    "name": "expert_a",
                    "side": 1,
                    "coef": [0.1],
                    "bias": 0.0,
                    "sigma": 0.03,
                }
            ],
            "router": {"reliability": {}, "ema_reliability": {}, "regime_reliability": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_artifact.json"
            path.write_text(json.dumps(payload))
            model = MythosRuntimeModel.from_artifact(path)
            self.assertEqual(model.symbol, "BTCUSDT")


if __name__ == "__main__":
    unittest.main()
