"""
Edge-Based Signal Generation

Institutional-grade signal generation using:
    edge = (μ - cost) / σ
    confidence = edge value
    enter_trade = confidence > threshold

This approach naturally produces "strong signals only" because:
- Weak signals have low μ (expected return)
- Uncertain signals have high σ (volatility)
- Both result in low edge

Output format includes:
- action: LONG, SHORT, or NO_TRADE
- confidence: Edge-based confidence score
- expected_move: μ prediction
- uncertainty: σ prediction
- cost_estimate: Transaction cost estimate
- suggested_order_type: maker vs taker
- risk_adjusted_size: Position size recommendation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Transaction costs for edge calculation."""
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    base_slippage: float = 0.0001
    vol_slippage_mult: float = 0.5
    avg_funding_8h: float = 0.0001
    
    def round_trip_cost(self, volatility: float, is_taker: bool = True, 
                        hold_hours: float = 4) -> float:
        """Calculate total round-trip cost."""
        fee = self.taker_fee if is_taker else self.maker_fee
        slippage = self.base_slippage + self.vol_slippage_mult * volatility
        funding = self.avg_funding_8h * (hold_hours / 8)
        return (fee * 2) + (slippage * 2) + abs(funding)


@dataclass
class EdgeSignal:
    """
    Complete trading signal output.
    
    All the information needed to make a trading decision.
    """
    timestamp: int
    
    action: str  # "LONG", "SHORT", "NO_TRADE"
    confidence: float  # Edge-based confidence (edge / σ)
    
    expected_move: float  # μ (expected return)
    uncertainty: float  # σ (volatility/uncertainty)
    edge: float  # μ - cost
    
    cost_estimate: float
    suggested_order_type: str  # "MAKER" or "TAKER"
    urgency: str  # "LOW", "MEDIUM", "HIGH"
    
    position_size_pct: float  # Recommended position size as % of capital
    stop_loss_pct: float  # Suggested stop loss distance
    take_profit_pct: float  # Suggested take profit distance
    
    regime: str  # Current market regime
    expert_weights: Dict[str, float]  # MoE expert weights if available
    
    p10_return: Optional[float] = None
    p50_return: Optional[float] = None
    p90_return: Optional[float] = None
    
    reasons: Optional[list] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return asdict(self)


class EdgeSignalGenerator:
    """
    Generates trading signals from model predictions.
    
    Uses edge-based confidence scoring:
    1. Get μ (expected return) and σ (uncertainty) from model
    2. Calculate cost based on current volatility
    3. Compute edge = μ - cost
    4. Compute confidence = edge / σ
    5. Generate signal only if confidence exceeds threshold
    """
    
    def __init__(self,
                 min_edge_threshold: float = 0.5,
                 max_position_pct: float = 0.1,
                 costs: Optional[TradingCosts] = None):
        """
        Args:
            min_edge_threshold: Minimum edge in σ units to trigger a trade.
                               Edge = (μ - cost) / σ, so 0.5 means expected profit
                               is at least 0.5 standard deviations above costs.
            max_position_pct: Maximum position size as fraction of capital
            costs: Transaction cost configuration
        """
        self.min_confidence = min_edge_threshold  # Kept for backward compatibility
        self.min_edge_threshold = min_edge_threshold
        self.max_position_pct = max_position_pct
        self.costs = costs or TradingCosts()
        
    def calculate_edge_metrics(self,
                                mu: float,
                                sigma: float,
                                current_volatility: float) -> Tuple[float, float, float, bool]:
        """
        Calculate edge metrics using the institutional formula:
            edge = (μ - cost) / σ
        
        This is the risk-adjusted expected profit that accounts for:
        - Direction and magnitude of expected move (μ)
        - Transaction costs (cost)
        - Uncertainty/volatility (σ)
        
        Returns:
            edge: Risk-adjusted edge = (|μ| - cost) / σ
            confidence: Edge value (same as edge in this formulation)
            cost: Transaction cost
            is_taker: Whether to use taker order
        """
        cost_maker = self.costs.round_trip_cost(current_volatility, is_taker=False)
        cost_taker = self.costs.round_trip_cost(current_volatility, is_taker=True)
        
        # Edge calculation per institutional spec: edge = (μ - cost) / σ
        # We use |μ| since direction is handled separately
        sigma_safe = max(sigma, 0.001)
        edge_maker = (abs(mu) - cost_maker) / sigma_safe
        edge_taker = (abs(mu) - cost_taker) / sigma_safe
        
        urgency_threshold = 1.5  # Edge ratio threshold for taker
        
        if edge_taker > edge_maker * urgency_threshold and edge_taker > 0:
            is_taker = True
            cost = cost_taker
            edge = edge_taker
        else:
            is_taker = False
            cost = cost_maker
            edge = edge_maker
        
        # In this formulation, edge IS the confidence (risk-adjusted score)
        confidence = edge
        
        return edge, confidence, cost, is_taker
    
    def calculate_position_size(self,
                                 edge: float,
                                 sigma: float,
                                 capital: float = 1.0) -> float:
        """
        Calculate position size using bounded Kelly criterion.
        
        Since edge = (μ - cost) / σ, we need:
            Kelly = (μ - cost) / σ² = edge / σ
        
        Bounded to prevent over-betting.
        """
        if sigma <= 0 or edge <= 0:
            return 0
        
        # Edge is already (μ - cost) / σ, so Kelly = edge / σ
        kelly = edge / sigma
        
        half_kelly = kelly * 0.5
        
        position_pct = min(half_kelly, self.max_position_pct)
        
        position_pct = max(0, position_pct)
        
        return position_pct
    
    def calculate_stops(self,
                        mu: float,
                        sigma: float,
                        direction: int) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Uses ATR-based approach with μ/σ information.
        Enforces minimum R:R ratio of 1.5.
        """
        stop_mult = 2.0
        stop_loss = sigma * stop_mult
        
        risk = stop_loss
        reward = abs(mu)
        rr_ratio = reward / max(risk, 0.001)
        
        if rr_ratio < 1.5:
            take_profit = risk * 1.5
        else:
            take_profit = abs(mu) * 1.2
        
        return stop_loss, take_profit
    
    def validate_entry_levels(self,
                              current_price: float,
                              entry_offset: float,
                              sl_distance: float,
                              tp_distance: float,
                              direction: int) -> Tuple[float, float, float, bool]:
        """
        Validate and compute actual price levels with directional enforcement.
        
        Ensures:
        - LONG: SL < Entry < TP
        - SHORT: TP < Entry < SL
        - R:R >= 1.5
        
        Args:
            current_price: Current market price
            entry_offset: Predicted entry offset (percentage)
            sl_distance: Predicted SL distance (percentage)
            tp_distance: Predicted TP distance (percentage)
            direction: 1 for LONG, -1 for SHORT
            
        Returns:
            (entry_price, sl_price, tp_price, is_valid)
        """
        MIN_RR_RATIO = 1.5
        
        if direction == 1:  # LONG
            entry_price = current_price * (1 + entry_offset)
            sl_price = entry_price * (1 - sl_distance)
            tp_price = entry_price * (1 + tp_distance)
            
            # Validate ordering: SL < Entry < TP
            if not (sl_price < entry_price < tp_price):
                # Fix the ordering
                sl_price = min(sl_price, entry_price * 0.99)
                tp_price = max(tp_price, entry_price * 1.01)
                
        elif direction == -1:  # SHORT
            entry_price = current_price * (1 - entry_offset)
            sl_price = entry_price * (1 + sl_distance)
            tp_price = entry_price * (1 - tp_distance)
            
            # Validate ordering: TP < Entry < SL
            if not (tp_price < entry_price < sl_price):
                # Fix the ordering
                tp_price = min(tp_price, entry_price * 0.99)
                sl_price = max(sl_price, entry_price * 1.01)
        else:
            return current_price, current_price, current_price, False
            
        # Validate R:R ratio
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr_ratio = reward / max(risk, current_price * 0.0001)
        
        if rr_ratio < MIN_RR_RATIO:
            # Adjust TP to meet minimum R:R
            if direction == 1:
                tp_price = entry_price + (risk * MIN_RR_RATIO)
            else:
                tp_price = entry_price - (risk * MIN_RR_RATIO)
        
        is_valid = True
        return entry_price, sl_price, tp_price, is_valid
    
    def calculate_real_confidence(self,
                                   edge: float,
                                   sigma: float,
                                   mu: float,
                                   sl_distance: float,
                                   tp_distance: float,
                                   ensemble_agreement: float = 1.0) -> float:
        """
        Calculate real confidence score using multiple factors.
        
        confidence = w1 * ensemble_agreement + 
                     w2 * (expected_return / volatility) +
                     w3 * (tp_distance / sl_distance)
        
        Args:
            edge: Risk-adjusted edge in σ units
            sigma: Predicted uncertainty
            mu: Predicted expected return
            sl_distance: Stop loss distance
            tp_distance: Take profit distance
            ensemble_agreement: Agreement between ensemble models (0-1)
            
        Returns:
            Real confidence score (0-1)
        """
        W1 = 0.4  # Ensemble agreement weight
        W2 = 0.3  # Return/volatility weight
        W3 = 0.3  # R:R ratio weight
        
        # Factor 1: Ensemble agreement (0-1)
        f1 = min(1.0, max(0.0, ensemble_agreement))
        
        # Factor 2: Return/volatility ratio (normalized to 0-1)
        # Higher |mu|/sigma is better
        return_vol_ratio = abs(mu) / max(sigma, 0.001)
        f2 = min(1.0, return_vol_ratio / 3.0)  # Normalize: 3σ move = 1.0
        
        # Factor 3: Risk/Reward ratio (normalized to 0-1)
        rr_ratio = tp_distance / max(sl_distance, 0.001)
        f3 = min(1.0, rr_ratio / 3.0)  # Normalize: 3:1 R:R = 1.0
        
        confidence = W1 * f1 + W2 * f2 + W3 * f3
        
        return max(0.0, min(1.0, confidence))
    
    def determine_urgency(self,
                          edge: float,
                          sigma: float,
                          mu: float) -> str:
        """Determine signal urgency based on edge characteristics."""
        confidence = edge / max(sigma, 0.001)
        
        if confidence > 2.0 and abs(mu) > 0.01:
            return "HIGH"
        elif confidence > 1.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_signal(self,
                        mu: float,
                        sigma: float,
                        current_price: float,
                        current_volatility: float,
                        timestamp: int,
                        regime: str = "UNKNOWN",
                        expert_weights: Optional[Dict[str, float]] = None,
                        quantiles: Optional[Tuple[float, float, float]] = None,
                        entry_offset: float = 0.0,
                        sl_distance: Optional[float] = None,
                        tp_distance: Optional[float] = None,
                        ensemble_agreement: float = 1.0) -> EdgeSignal:
        """
        Generate a complete trading signal with validated entry/SL/TP levels.
        
        Args:
            mu: Predicted expected return
            sigma: Predicted uncertainty
            current_price: Current asset price
            current_volatility: Current realized volatility
            timestamp: Signal timestamp
            regime: Current market regime
            expert_weights: MoE expert weights
            quantiles: (p10, p50, p90) return predictions
            entry_offset: Predicted entry offset (from trading head)
            sl_distance: Predicted SL distance (from trading head, or None to calculate)
            tp_distance: Predicted TP distance (from trading head, or None to calculate)
            ensemble_agreement: Agreement between ensemble models (0-1)
            
        Returns:
            EdgeSignal with complete trading recommendation
        """
        edge, edge_confidence, cost, is_taker = self.calculate_edge_metrics(
            mu, sigma, current_volatility
        )
        
        # Edge is now in σ units: edge >= threshold means trade
        should_trade = edge >= self.min_edge_threshold
        
        if should_trade:
            action = "LONG" if mu > 0 else "SHORT"
            direction = 1 if mu > 0 else -1
        else:
            action = "NO_TRADE"
            direction = 0
        
        position_size = self.calculate_position_size(edge, sigma) if should_trade else 0
        
        # Calculate SL/TP if not provided from trading head
        if should_trade:
            if sl_distance is None or tp_distance is None:
                stop_loss, take_profit = self.calculate_stops(mu, sigma, direction)
            else:
                stop_loss, take_profit = sl_distance, tp_distance
            
            # Validate and enforce directional constraints with R:R >= 1.5
            entry_price, sl_price, tp_price, is_valid = self.validate_entry_levels(
                current_price, entry_offset, stop_loss, take_profit, direction
            )
            
            # Convert back to percentages for output
            stop_loss = abs(entry_price - sl_price) / entry_price
            take_profit = abs(tp_price - entry_price) / entry_price
        else:
            stop_loss, take_profit = 0, 0
        
        urgency = self.determine_urgency(edge, sigma, mu) if should_trade else "LOW"
        
        # Calculate real confidence using multiple factors
        real_confidence = self.calculate_real_confidence(
            edge=edge,
            sigma=sigma,
            mu=mu,
            sl_distance=stop_loss if stop_loss > 0 else 0.01,
            tp_distance=take_profit if take_profit > 0 else 0.015,
            ensemble_agreement=ensemble_agreement
        )
        
        reasons = []
        if should_trade:
            reasons.append(f"Edge: {edge:.2f}σ (risk-adjusted)")
            reasons.append(f"Expected move: {mu*100:.3f}%")
            reasons.append(f"Uncertainty: {sigma*100:.3f}%")
            reasons.append(f"R:R ratio: {take_profit/max(stop_loss, 0.001):.2f}")
            if regime != "UNKNOWN":
                reasons.append(f"Regime: {regime}")
        else:
            if edge < self.min_confidence:
                reasons.append(f"Edge too low: {edge:.2f}σ < {self.min_confidence}σ threshold")
        
        signal = EdgeSignal(
            timestamp=timestamp,
            action=action,
            confidence=real_confidence,  # Use real confidence instead of edge-based
            expected_move=mu,
            uncertainty=sigma,
            edge=edge,
            cost_estimate=cost,
            suggested_order_type="TAKER" if is_taker else "MAKER",
            urgency=urgency,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            regime=regime,
            expert_weights=expert_weights or {},
            p10_return=quantiles[0] if quantiles else None,
            p50_return=quantiles[1] if quantiles else None,
            p90_return=quantiles[2] if quantiles else None,
            reasons=reasons
        )
        
        return signal


class ModelSignalInterface:
    """
    Interface between ML models and signal generation.
    
    Handles:
    - Getting predictions from models
    - Converting to edge signals
    - Aggregating multiple model predictions
    """
    
    def __init__(self,
                 signal_generator: EdgeSignalGenerator = None,
                 device: str = "cuda"):
        self.signal_generator = signal_generator or EdgeSignalGenerator()
        self.device = device
        
    def generate_from_model(self,
                            model: torch.nn.Module,
                            features: torch.Tensor,
                            current_price: float,
                            current_volatility: float,
                            timestamp: int,
                            regime: str = "UNKNOWN") -> EdgeSignal:
        """
        Generate signal from a single model.
        """
        model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            
            output = model(features)
            
            if isinstance(output, tuple):
                mu, sigma = output[0], output[1]
                if len(output) > 2:
                    gate_weights = output[2]
                else:
                    gate_weights = None
            else:
                mu = output[:, 0]
                sigma = torch.abs(output[:, 1]) + 0.001
                gate_weights = None
            
            mu_val = mu.cpu().numpy().item() if mu.dim() > 0 else mu.cpu().item()
            sigma_val = sigma.cpu().numpy().item() if sigma.dim() > 0 else sigma.cpu().item()
            
            expert_weights = None
            if gate_weights is not None:
                weights = gate_weights.cpu().numpy().flatten()
                expert_names = ["trend", "mean_reversion", "volatility", "chaos"]
                expert_weights = {
                    expert_names[i]: float(weights[i])
                    for i in range(min(len(weights), len(expert_names)))
                }
        
        signal = self.signal_generator.generate_signal(
            mu=mu_val,
            sigma=sigma_val,
            current_price=current_price,
            current_volatility=current_volatility,
            timestamp=timestamp,
            regime=regime,
            expert_weights=expert_weights
        )
        
        return signal
    
    def generate_from_ensemble(self,
                                models: list,
                                features: torch.Tensor,
                                current_price: float,
                                current_volatility: float,
                                timestamp: int) -> EdgeSignal:
        """
        Generate signal from an ensemble of models.
        
        Combines predictions using uncertainty-weighted averaging.
        """
        all_mus = []
        all_sigmas = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                features = features.to(self.device)
                output = model(features)
                
                if isinstance(output, tuple):
                    mu, sigma = output[0], output[1]
                else:
                    mu = output[:, 0]
                    sigma = torch.abs(output[:, 1]) + 0.001
                
                all_mus.append(mu.cpu().item())
                all_sigmas.append(sigma.cpu().item())
        
        sigmas = np.array(all_sigmas)
        precisions = 1 / (sigmas ** 2 + 1e-8)
        weights = precisions / precisions.sum()
        
        combined_mu = np.sum(np.array(all_mus) * weights)
        
        combined_sigma = np.sqrt(np.sum(sigmas ** 2 * weights))
        
        signal = self.signal_generator.generate_signal(
            mu=float(combined_mu),
            sigma=float(combined_sigma),
            current_price=current_price,
            current_volatility=current_volatility,
            timestamp=timestamp,
            regime="ENSEMBLE"
        )
        
        return signal


def create_signal_generator(
    min_edge_threshold: float = 0.5,
    max_position_pct: float = 0.1
) -> EdgeSignalGenerator:
    """
    Factory function to create signal generator.
    
    Args:
        min_edge_threshold: Minimum edge in σ units to trigger a trade.
                           Default 0.5 means expected profit must be at least
                           0.5 standard deviations above transaction costs.
        max_position_pct: Maximum position size as fraction of capital
    """
    costs = TradingCosts(
        maker_fee=0.0002,
        taker_fee=0.0004,
        base_slippage=0.0001,
        vol_slippage_mult=0.5,
        avg_funding_8h=0.0001
    )
    
    return EdgeSignalGenerator(
        min_edge_threshold=min_edge_threshold,
        max_position_pct=max_position_pct,
        costs=costs
    )
