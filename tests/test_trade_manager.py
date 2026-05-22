from dataclasses import dataclass
import unittest

from trade_manager import TradeManager


@dataclass
class _DummyPos:
    side: str = "LONG"
    entry_price: float = 100.0
    sl_price: float = 95.0
    tp_price: float = 115.0
    original_sl: float = 95.0
    bar_index: int = 1
    horizon: int = 24
    htf_score: int = 0
    threshold_used: float = 0.6
    p_enter: float = 0.7
    lane: str = "MYTHOS"

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"


class TradeManagerPolicyTests(unittest.TestCase):
    def test_max_policy_triggers_scale_out(self):
        tm = TradeManager(policy_mode="max", enable_scale_out=True)
        pos = _DummyPos()

        action = tm.update_position(
            symbol="BTCUSDT",
            pos=pos,
            current_price=107.0,  # +1.4R on a 5-point stop
            current_bar=5,
            current_htf_score=1,
            current_p_enter=0.72,
            candle_high=107.0,
            candle_low=100.0,
        )

        self.assertEqual(action.action, "CLOSE_PARTIAL")
        self.assertIsNotNone(action.close_pct)
        self.assertGreaterEqual(float(action.close_pct), 5.0)
        self.assertLessEqual(float(action.close_pct), 50.0)

    def test_regime_aware_horizon_extends_for_winner(self):
        tm = TradeManager(policy_mode="max", enable_regime_time_stop=True)
        pos = _DummyPos(horizon=20, htf_score=0)

        # First update seeds favorable excursion and adaptive horizon.
        _ = tm.update_position(
            symbol="ETHUSDT",
            pos=pos,
            current_price=108.0,
            current_bar=8,
            current_htf_score=3,
            current_p_enter=0.9,
            candle_high=108.0,
            candle_low=101.0,
        )
        state = tm.get_state("ETHUSDT")
        self.assertIsNotNone(state)
        self.assertGreater(state["effective_horizon"], pos.horizon)

    def test_regime_aware_horizon_shrinks_when_trade_degrades(self):
        tm = TradeManager(policy_mode="max", enable_regime_time_stop=True)
        pos = _DummyPos(horizon=30, htf_score=2)

        _ = tm.update_position(
            symbol="SOLUSDT",
            pos=pos,
            current_price=98.8,  # mild unrealized loss, avoid hard adverse flip path
            current_bar=10,
            current_htf_score=-1,
            current_p_enter=0.35,
            candle_high=100.2,
            candle_low=98.6,
        )
        state = tm.get_state("SOLUSDT")
        self.assertIsNotNone(state)
        self.assertLess(state["effective_horizon"], pos.horizon)


if __name__ == "__main__":
    unittest.main()
