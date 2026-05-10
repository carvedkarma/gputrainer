from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]
    checks: Dict[str, bool]

    @property
    def should_promote(self) -> bool:
        return self.promote


@dataclass
class PromotionGate:
    min_trades: int = 40
    min_expectancy: float = 0.03
    min_profit_factor: float = 1.10
    min_win_rate: float = 0.33
    max_drawdown_r: float = -20.0

    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, object]:
        checks = {
            "min_trades": metrics.get("trades", 0) >= self.min_trades,
            "min_expectancy": metrics.get("expectancy_r", -999.0) >= self.min_expectancy,
            "min_profit_factor": metrics.get("profit_factor", 0.0) >= self.min_profit_factor,
            "min_win_rate": metrics.get("win_rate", 0.0) >= self.min_win_rate,
            "max_drawdown": metrics.get("max_drawdown_r", -999.0) >= self.max_drawdown_r,
        }
        promoted = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        return PromotionDecision(promote=promoted, reasons=failed, checks=checks)


def evaluate_promotion(
    expectancy: float,
    win_rate: float,
    trades: int,
    min_trades: int,
    edge_drift: float,
    score_monotonic: bool,
    side_balance: float,
) -> PromotionDecision:
    """
    Lightweight promotion heuristic used by mythos walk-forward.
    """
    checks = {
        "min_trades": trades >= min_trades,
        "expectancy_positive": expectancy > 0.0,
        "win_rate_floor": win_rate >= 0.30,
        "edge_drift_ok": edge_drift < 3.0,
        "score_monotonic": bool(score_monotonic),
        "side_balance_ok": side_balance >= 0.05,
    }
    promote = all(checks.values())
    reasons = [k for k, ok in checks.items() if not ok]
    return PromotionDecision(promote=promote, reasons=reasons, checks=checks)
