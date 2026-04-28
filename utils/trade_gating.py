"""
Unified Trade Gating Logic

This module provides a SINGLE source of truth for trade gating decisions,
used by both training evaluation and inference. This prevents the
"trained on one policy, traded on another" problem.

All gating logic should go through these functions to ensure consistency.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class GateFailure(Enum):
    """Enum for identifying which gate failed."""
    NONE = "none"
    EDGE_GATE = "edge_gate"
    CONFIDENCE_GATE = "confidence_gate"
    SPREAD_GATE = "spread_gate"
    MARGIN_GATE = "margin_gate"


@dataclass
class TradeGateConfig:
    """Configuration for trade gating rules.
    
    These parameters control when a signal is allowed to become a trade.
    Use the same config for both training evaluation and inference.
    """
    # Cost parameters
    fixed_cost: float = 0.0009  # 0.09% round-trip cost
    
    # Edge gate: mu - cost >= min_net_edge
    min_net_edge: float = 0.0005  # Minimum edge after costs (0.05%)
    
    # Confidence gate: |mu| / sigma >= min_confidence
    min_confidence: float = 0.6  # Tuned for BTC 15m h=16
    
    # Spread gate: (q75 - q25) >= spread_multiplier * cost
    spread_multiplier: float = 3.0  # Require spread >= 3x cost
    
    # Optional margin gate: confidence - min_confidence >= min_margin
    use_margin_gate: bool = False
    min_margin: float = 0.2
    
    # Cooldown (handled separately in evaluation loop)
    cooldown_bars: int = 8  # horizon/2

    def __post_init__(self):
        """Validate configuration."""
        if self.fixed_cost < 0:
            raise ValueError("fixed_cost must be non-negative")
        if self.min_confidence < 0:
            raise ValueError("min_confidence must be non-negative")
        if self.spread_multiplier < 1:
            raise ValueError("spread_multiplier should be >= 1")


# Default configuration - used when no custom config provided
DEFAULT_GATE_CONFIG = TradeGateConfig()


@dataclass 
class GateResult:
    """Result of trade gating decision."""
    allowed: bool
    action: str  # "LONG", "SHORT", or "HOLD"
    original_action: str  # Original model prediction
    failed_gate: GateFailure
    edge: float
    confidence: float
    spread: Optional[float]
    details: Dict[str, float]


def compute_trade_gate(
    mu: float,
    sigma: float,
    direction: str,  # "LONG", "SHORT", "HOLD"
    q25: Optional[float] = None,
    q75: Optional[float] = None,
    config: Optional[TradeGateConfig] = None
) -> GateResult:
    """
    Unified trade gating function.
    
    This function determines whether a signal should be allowed to trade.
    Used by BOTH training evaluation AND inference to ensure consistency.
    
    Args:
        mu: Expected return (regression head output)
        sigma: Uncertainty (regression head output)
        direction: Model's predicted direction ("LONG", "SHORT", "HOLD")
        q25: 25th percentile quantile (optional, for spread gate)
        q75: 75th percentile quantile (optional, for spread gate)
        config: Gate configuration (uses DEFAULT_GATE_CONFIG if None)
    
    Returns:
        GateResult with decision and diagnostic info
    """
    if config is None:
        config = DEFAULT_GATE_CONFIG
    
    # === FIXED: Handle log_sigma if passed (log_sigma is typically negative) ===
    # If sigma looks like log_sigma (negative), convert to sigma = exp(log_sigma)
    import math
    actual_sigma = sigma
    if sigma < 0:  # Likely log_sigma
        actual_sigma = math.exp(max(sigma, -10))  # Convert log_sigma to sigma
        logger.debug(f"Converted log_sigma={sigma:.4f} to sigma={actual_sigma:.6f}")
    
    # Compute gate metrics
    edge = abs(mu) - config.fixed_cost
    confidence = abs(mu) / max(actual_sigma, 1e-8)
    spread = (q75 - q25) if (q25 is not None and q75 is not None) else None
    
    details = {
        "mu": mu,
        "sigma": actual_sigma,  # Use converted sigma, not raw log_sigma
        "raw_sigma": sigma,  # Keep original for debugging
        "edge": edge,
        "confidence": confidence,
        "cost": config.fixed_cost,
        "min_net_edge": config.min_net_edge,
        "min_confidence": config.min_confidence,
    }
    
    # If already HOLD, just return
    if direction == "HOLD":
        return GateResult(
            allowed=False,
            action="HOLD",
            original_action="HOLD",
            failed_gate=GateFailure.NONE,
            edge=edge,
            confidence=confidence,
            spread=spread,
            details=details
        )
    
    # Gate 1: Edge gate
    # Trade only if: edge = |mu| - cost >= min_net_edge
    if edge < config.min_net_edge:
        logger.debug(f"Edge gate failed: edge={edge:.4f} < min={config.min_net_edge:.4f}")
        return GateResult(
            allowed=False,
            action="HOLD",
            original_action=direction,
            failed_gate=GateFailure.EDGE_GATE,
            edge=edge,
            confidence=confidence,
            spread=spread,
            details=details
        )
    
    # Gate 2: Confidence gate
    # Trade only if: confidence = |mu|/sigma >= min_confidence
    if confidence < config.min_confidence:
        logger.debug(f"Confidence gate failed: conf={confidence:.4f} < min={config.min_confidence:.4f}")
        return GateResult(
            allowed=False,
            action="HOLD",
            original_action=direction,
            failed_gate=GateFailure.CONFIDENCE_GATE,
            edge=edge,
            confidence=confidence,
            spread=spread,
            details=details
        )
    
    # Gate 3: Spread gate (if quantiles available)
    if spread is not None:
        min_spread = config.spread_multiplier * config.fixed_cost
        details["spread"] = spread
        details["min_spread"] = min_spread
        
        if spread < min_spread:
            logger.debug(f"Spread gate failed: spread={spread:.4f} < min={min_spread:.4f}")
            return GateResult(
                allowed=False,
                action="HOLD",
                original_action=direction,
                failed_gate=GateFailure.SPREAD_GATE,
                edge=edge,
                confidence=confidence,
                spread=spread,
                details=details
            )
    
    # Gate 4: Margin gate (optional)
    if config.use_margin_gate:
        margin = confidence - config.min_confidence
        details["margin"] = margin
        details["min_margin"] = config.min_margin
        
        if margin < config.min_margin:
            logger.debug(f"Margin gate failed: margin={margin:.4f} < min={config.min_margin:.4f}")
            return GateResult(
                allowed=False,
                action="HOLD",
                original_action=direction,
                failed_gate=GateFailure.MARGIN_GATE,
                edge=edge,
                confidence=confidence,
                spread=spread,
                details=details
            )
    
    # All gates passed
    return GateResult(
        allowed=True,
        action=direction,
        original_action=direction,
        failed_gate=GateFailure.NONE,
        edge=edge,
        confidence=confidence,
        spread=spread,
        details=details
    )


def apply_cooldown(
    trade_signals: list,
    cooldown_bars: int
) -> list:
    """
    Apply cooldown to trade signals.
    
    After taking a trade, no new trades for `cooldown_bars` candles.
    
    Args:
        trade_signals: List of boolean trade signals
        cooldown_bars: Number of bars to wait after a trade
        
    Returns:
        Filtered list with cooldown applied
    """
    result = [False] * len(trade_signals)
    last_trade_idx = -cooldown_bars - 1
    
    for i in range(len(trade_signals)):
        if trade_signals[i] and (i - last_trade_idx) > cooldown_bars:
            result[i] = True
            last_trade_idx = i
    
    return result


def log_gate_statistics(
    gate_results: list,
    context: str = "evaluation"
) -> Dict[str, float]:
    """
    Log statistics about gate failures for debugging.
    
    Args:
        gate_results: List of GateResult objects
        context: String describing the context (e.g., "training", "inference")
    
    Returns:
        Dictionary of gate statistics
    """
    total = len(gate_results)
    if total == 0:
        return {}
    
    allowed = sum(1 for r in gate_results if r.allowed)
    edge_fails = sum(1 for r in gate_results if r.failed_gate == GateFailure.EDGE_GATE)
    conf_fails = sum(1 for r in gate_results if r.failed_gate == GateFailure.CONFIDENCE_GATE)
    spread_fails = sum(1 for r in gate_results if r.failed_gate == GateFailure.SPREAD_GATE)
    margin_fails = sum(1 for r in gate_results if r.failed_gate == GateFailure.MARGIN_GATE)
    holds = sum(1 for r in gate_results if r.original_action == "HOLD")
    
    stats = {
        "total_signals": total,
        "allowed_trades": allowed,
        "pass_rate": allowed / total,
        "original_holds": holds,
        "edge_gate_fails": edge_fails,
        "confidence_gate_fails": conf_fails,
        "spread_gate_fails": spread_fails,
        "margin_gate_fails": margin_fails,
    }
    
    logger.info(f"[{context}] Gate Statistics:")
    logger.info(f"  Total signals: {total}")
    logger.info(f"  Original HOLDs: {holds} ({100*holds/total:.1f}%)")
    logger.info(f"  Allowed trades: {allowed} ({100*allowed/total:.1f}%)")
    logger.info(f"  Edge gate fails: {edge_fails} ({100*edge_fails/total:.1f}%)")
    logger.info(f"  Confidence gate fails: {conf_fails} ({100*conf_fails/total:.1f}%)")
    logger.info(f"  Spread gate fails: {spread_fails} ({100*spread_fails/total:.1f}%)")
    if margin_fails > 0:
        logger.info(f"  Margin gate fails: {margin_fails} ({100*margin_fails/total:.1f}%)")
    
    return stats
