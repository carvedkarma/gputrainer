"""
Professional Ensemble Predictor for Trading Signals

Design principles:
1. Direction models (Transformer, TFT, LSTM, CNN) vote on direction
2. VAE acts as regime gate (trend/range/chop detection)
3. GNN acts as risk filter (risk-on/off detection)
4. Weights based on walk-forward trading metrics, NOT accuracy
5. Calibrated probabilities via temperature scaling
6. Confidence margin = p_top1 - p_top2 (not raw max)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"

class RiskRegime(Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL = "NEUTRAL"
    CORRELATION_SHOCK = "CORRELATION_SHOCK"
    UNKNOWN = "UNKNOWN"

@dataclass
class ModelWeight:
    """Walk-forward trading metric weights for a model."""
    model_name: str
    expectancy: float  # Expected profit per trade
    precision_on_trade: float  # Precision when model decides to trade (not HOLD)
    profit_factor: float  # Gross profit / gross loss
    f1_directional: float  # F1 on LONG/SHORT only
    sharpe: float  # Walk-forward Sharpe ratio
    calibration_temp: float = 1.0  # Temperature for probability calibration
    
    @property
    def composite_weight(self) -> float:
        """Compute composite weight from trading metrics."""
        weights = {
            'expectancy': 0.3,
            'precision': 0.25,
            'profit_factor': 0.2,
            'f1': 0.15,
            'sharpe': 0.1
        }
        
        score = (
            weights['expectancy'] * max(0, self.expectancy) +
            weights['precision'] * self.precision_on_trade +
            weights['profit_factor'] * min(2.0, self.profit_factor) / 2.0 +
            weights['f1'] * self.f1_directional +
            weights['sharpe'] * max(0, min(3.0, self.sharpe)) / 3.0
        )
        return max(0.01, score)  # Minimum weight

@dataclass
class EnsembleSignal:
    """Output signal from ensemble predictor."""
    action: str  # LONG, SHORT, HOLD, NO_TRADE
    confidence: float
    confidence_margin: float  # p_top1 - p_top2
    edge: float
    
    # Regime information
    market_regime: str
    risk_regime: str
    regime_confidence: float
    
    # Model agreement
    agreement_pct: float
    weighted_agreement: float
    disagreement_score: float
    
    # Position sizing adjustments
    position_size_pct: float
    regime_adjusted_size: float
    
    # Thresholds used
    confidence_threshold_used: float
    regime_adjustment: str
    
    # Per-model breakdown
    model_votes: Dict[str, Dict[str, Any]]
    reasons: List[str]
    
    # Probabilities
    ensemble_probs: Dict[str, float]
    
    # === Multi-head output: Quantiles, regression, trading params ===
    # Aggregated from all direction models that support forward_multihead()
    quantiles: Optional[Dict[str, float]] = None  # q10, q25, q50, q75, q90
    mu: Optional[float] = None  # Expected return
    sigma: Optional[float] = None  # Uncertainty
    entry_offset: Optional[float] = None  # Entry price offset
    sl_distance: Optional[float] = None  # Stop loss distance (%)
    tp_distance: Optional[float] = None  # Take profit distance (%)
    
    # === Flow Forecast outputs (from VolStateHead and AccelerationHead) ===
    vol_state: Optional[str] = None  # "contraction", "neutral", "expansion"
    vol_state_probs: Optional[Dict[str, float]] = None  # P(contraction), P(neutral), P(expansion)
    acceleration: Optional[float] = None  # momentum change prediction
    forecast_mode: Optional[str] = None  # "QUANTILE_PATHS" or "NO_FORECAST"
    quantile_paths: Optional[Dict[str, List[float]]] = None  # {q10: [...], q50: [...], q90: [...]}
    
    # === Kelly Criterion Position Sizing (2024 Advanced Feature) ===
    kelly_fraction: Optional[float] = None  # Optimal position size from Kelly formula
    kelly_adjusted_size: Optional[float] = None  # Kelly fraction with safety cap (25% max)
    epistemic_uncertainty: Optional[float] = None  # MC Dropout uncertainty
    should_trade_uncertainty: Optional[bool] = None  # True if uncertainty is low enough

class EnsemblePredictor:
    """
    Professional ensemble predictor with:
    - Direction model voting
    - VAE regime gating
    - GNN risk filtering
    - Walk-forward metric weighting
    - Temperature-scaled calibration
    """
    
    DIRECTION_MODELS = ['transformer', 'tft', 'lstm', 'cnn', 'bidirectional', 'stacked', 'conv_lstm', 'resnet', 'inception', 'wavenet']
    REGIME_MODELS = ['vae', 'market_vae', 'conditional_vae', 'cvae']
    RISK_MODELS = ['gnn', 'cross_asset', 'temporal_gnn', 'crossasset']
    
    def __init__(self, model_instances: Dict[str, torch.nn.Module], device: str = "cuda", 
                 strict_weights: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EnsemblePredictor.
        
        Args:
            model_instances: Dict of model name -> model instance
            device: Device to run on (cuda/cpu)
            strict_weights: If True, fails hard if model_weights.json is missing.
                           Set to False only for development/testing.
            config: Optional configuration dict with:
                - inference_temperature: float (default 0.7) - T < 1 sharpens predictions
        """
        self.device = device
        self.model_instances = model_instances
        self.strict_weights = strict_weights
        
        # Configuration with defaults for PHASE 2 improvements
        self.config = config or {}
        if 'inference_temperature' not in self.config:
            # T=0.7 sharpens soft predictions - helps combat uniform 33/33/33 outputs
            self.config['inference_temperature'] = 0.7
        
        logger.info(f"[EnsemblePredictor] Config: inference_temperature={self.config['inference_temperature']}")
        
        # Classify models by role
        self.direction_models = {}
        self.regime_models = {}
        self.risk_models = {}
        
        for name, model in model_instances.items():
            name_lower = name.lower()
            if any(dm in name_lower for dm in self.REGIME_MODELS):
                self.regime_models[name] = model
            elif any(rm in name_lower for rm in self.RISK_MODELS):
                self.risk_models[name] = model
            else:
                self.direction_models[name] = model
        
        # Load model weights (will fail hard in strict mode if missing)
        self.model_weights = self._load_or_create_weights()
        
        # Base thresholds
        self.base_confidence_threshold = 0.15
        self.base_margin_threshold = 0.10
        self.majority_weight_threshold = 0.55
        
        logger.info(f"EnsemblePredictor initialized:")
        logger.info(f"  Direction models: {list(self.direction_models.keys())}")
        logger.info(f"  Regime models: {list(self.regime_models.keys())}")
        logger.info(f"  Risk models: {list(self.risk_models.keys())}")
    
    def _load_or_create_weights(self) -> Dict[str, ModelWeight]:
        """
        Load walk-forward metric weights. MANDATORY for production.
        
        PHASE 3: Ensemble weights are now MANDATORY.
        - Training must run walk-forward evaluation and save model_weights.json
        - Using default weights produces LOUD WARNINGS
        - Call save_walk_forward_weights() after training to populate
        """
        self._using_default_weights = False  # Track if we're using defaults
        
        # Try multiple path resolution strategies
        import os
        candidate_paths = [
            # Strategy 1: Relative to this file
            Path(__file__).parent.parent / "checkpoints" / "model_weights.json",
            # Strategy 2: Relative to current working directory
            Path(os.getcwd()) / "checkpoints" / "model_weights.json",
            # Strategy 3: Relative to gpu_trainer in cwd
            Path(os.getcwd()) / "gpu_trainer" / "checkpoints" / "model_weights.json",
            # Strategy 4: Absolute path from __file__ resolved
            Path(__file__).resolve().parent.parent / "checkpoints" / "model_weights.json",
        ]
        
        logger.info("=" * 70)
        logger.info("[MODEL WEIGHTS] Searching for model_weights.json...")
        logger.info(f"  __file__ = {__file__}")
        logger.info(f"  cwd = {os.getcwd()}")
        
        weights_path = None
        for i, path in enumerate(candidate_paths):
            exists = path.exists()
            logger.info(f"  Path {i+1}: {path}")
            logger.info(f"         exists={exists}")
            if exists and weights_path is None:
                weights_path = path
                logger.info(f"  >>> FOUND at path {i+1}")
        
        logger.info("=" * 70)
        
        if weights_path is not None and weights_path.exists():
            try:
                with open(weights_path) as f:
                    data = json.load(f)
                
                # Filter to only ModelWeight fields (ignore extra fields like total_trades)
                model_weight_fields = {'model_name', 'expectancy', 'precision_on_trade', 
                                      'profit_factor', 'f1_directional', 'sharpe', 'calibration_temp'}
                loaded_weights = {}
                for name, w in data.items():
                    filtered_w = {k: v for k, v in w.items() if k in model_weight_fields}
                    loaded_weights[name] = ModelWeight(**filtered_w)
                logger.info(f"Loaded {len(loaded_weights)} model weights from {weights_path}")
                for name, w in loaded_weights.items():
                    logger.info(f"  {name}: expectancy={w.expectancy:.4f}, sharpe={w.sharpe:.2f}")
                return loaded_weights
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load model weights: {e}")
                logger.error(f"  Path: {weights_path}")
        
        # =========================================================================
        # PHASE 3: FAIL HARD - Missing weights means untrained/broken ensemble
        # =========================================================================
        self._using_default_weights = True
        
        error_msg = """
================================================================================
CRITICAL ERROR: model_weights.json NOT FOUND
================================================================================

Ensemble predictions CANNOT proceed without real walk-forward metrics.
Using default weights produces HOLD-heavy, unresponsive predictions.

To fix this:
  1. Run training with walk-forward evaluation enabled
  2. Training will auto-save weights to: checkpoints/model_weights.json
  3. Restart the server after training completes

The walk-forward evaluation computes:
  - Expectancy (expected PnL per trade)
  - Sharpe ratio
  - Profit factor
  - Win rate

These metrics determine how much weight each model gets in ensemble voting.
Without real weights, all models vote equally which is NOT useful.

================================================================================
"""
        
        if self.strict_weights:
            logger.error(error_msg)
            raise RuntimeError("model_weights.json is REQUIRED. Run training first.")
        
        # Non-strict mode: warn and continue with defaults (development only)
        logger.warning("=" * 80)
        logger.warning("WARNING: model_weights.json NOT FOUND - USING DEFAULT WEIGHTS")
        logger.warning("=" * 80)
        logger.warning("  strict_weights=False allows this for development only.")
        logger.warning("  Production MUST have real walk-forward weights!")
        logger.warning("=" * 80)
        
        # Create default weights for all models (PLACEHOLDER - NOT RECOMMENDED)
        defaults = {}
        for name in self.model_instances:
            defaults[name] = ModelWeight(
                model_name=name,
                expectancy=0.001,  # Placeholder: 0.1% expected per trade
                precision_on_trade=0.55,  # Placeholder: 55% precision
                profit_factor=1.2,  # Placeholder: 1.2:1 profit factor
                f1_directional=0.45,  # Placeholder: 45% F1
                sharpe=0.5,  # Placeholder: 0.5 Sharpe
                calibration_temp=1.0  # Default: no calibration
            )
        return defaults
    
    @property
    def using_default_weights(self) -> bool:
        """Returns True if ensemble is using default weights (not walk-forward metrics)."""
        return getattr(self, '_using_default_weights', True)
    
    def save_weights(self, weights: Dict[str, ModelWeight]):
        """Save model weights from walk-forward evaluation."""
        weights_path = Path(__file__).parent.parent / "checkpoints" / "model_weights.json"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {name: {
            'model_name': w.model_name,
            'expectancy': w.expectancy,
            'precision_on_trade': w.precision_on_trade,
            'profit_factor': w.profit_factor,
            'f1_directional': w.f1_directional,
            'sharpe': w.sharpe,
            'calibration_temp': w.calibration_temp
        } for name, w in weights.items()}
        
        with open(weights_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.model_weights = weights
        logger.info(f"Saved model weights to {weights_path}")
    
    def _calibrate_probs(self, probs: np.ndarray, model_name: str) -> np.ndarray:
        """Apply per-model temperature scaling for probability calibration.
        
        NOTE: If global inference_temperature is applied at logit level (in _get_model_prediction),
        this per-model calibration is skipped to avoid double temperature scaling.
        The inference_temperature parameter is meant to sharpen ALL model predictions uniformly,
        while calibration_temp (per-model) was for fine-tuning individual model confidence.
        
        Current design: Use inference_temperature globally, skip per-model calibration.
        """
        # Skip per-model calibration if global inference temperature is applied
        inference_temp = self.config.get('inference_temperature', 0.7)
        if inference_temp != 1.0:
            # Global temperature already applied at logit level - skip per-model
            return probs
        
        # Only apply per-model calibration if no global temperature
        temp = self.model_weights.get(model_name, ModelWeight(
            model_name=model_name,
            expectancy=0, precision_on_trade=0.5,
            profit_factor=1.0, f1_directional=0.4, sharpe=0
        )).calibration_temp
        
        if temp == 1.0:
            return probs
        
        # Apply temperature scaling on probabilities (legacy path)
        logits = np.log(probs + 1e-8)
        scaled_logits = logits / temp
        calibrated = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
        return calibrated
    
    def _get_model_prediction(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Dict:
        """Get prediction from a single model.
        
        If model supports forward_multihead(), extract full multi-head output:
        - class_logits -> direction probs
        - mu, sigma -> regression outputs
        - quantiles -> q10, q25, q50, q75, q90
        - entry_offset, sl_distance, tp_distance -> trading parameters
        """
        try:
            model.eval()
            with torch.no_grad():
                # Check if model supports forward_multihead for full output
                has_multihead = hasattr(model, 'forward_multihead')
                
                if has_multihead:
                    # Use forward_multihead for full multi-head output
                    output = model.forward_multihead(features)
                    
                    # Apply inference temperature scaling (T < 1 sharpens predictions)
                    # Default T=0.7 sharpens soft/uniform predictions toward confident ones
                    inference_temp = self.config.get('inference_temperature', 0.7)
                    scaled_logits = output.class_logits / inference_temp
                    probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
                    
                    # Handle batch dimension
                    if len(probs.shape) == 2 and probs.shape[0] == 1:
                        probs = probs[0]
                    elif len(probs.shape) == 2:
                        probs = probs.mean(axis=0)  # Average across batch
                    
                    # Extract regression outputs (mu, sigma)
                    mu = float(output.mu.cpu().numpy().mean()) if output.mu is not None else None
                    sigma = float(output.sigma.cpu().numpy().mean()) if output.sigma is not None else None
                    
                    # Extract quantiles q10, q25, q50, q75, q90
                    quantiles = None
                    if output.quantiles is not None:
                        q = output.quantiles.cpu().numpy()
                        if len(q.shape) == 2:
                            q = q.mean(axis=0)  # Average across batch
                        quantiles = {
                            "q10": float(q[0]),
                            "q25": float(q[1]),
                            "q50": float(q[2]),
                            "q75": float(q[3]),
                            "q90": float(q[4])
                        }
                    
                    # Extract trading parameters
                    entry_offset = float(output.entry_offset.cpu().numpy().mean()) if output.entry_offset is not None else None
                    sl_distance = float(output.sl_distance.cpu().numpy().mean()) if output.sl_distance is not None else None
                    tp_distance = float(output.tp_distance.cpu().numpy().mean()) if output.tp_distance is not None else None
                    
                    # === Flow Forecast outputs (vol_state, acceleration) ===
                    vol_state_probs = None
                    vol_state = None
                    if output.vol_state_logits is not None:
                        vs_probs = F.softmax(output.vol_state_logits, dim=-1).cpu().numpy()
                        if len(vs_probs.shape) == 2:
                            vs_probs = vs_probs.mean(axis=0)  # Average across batch
                        vol_state_probs = {
                            "contraction": float(vs_probs[0]),
                            "neutral": float(vs_probs[1]),
                            "expansion": float(vs_probs[2])
                        }
                        vol_state_idx = int(vs_probs.argmax())
                        vol_state = ["contraction", "neutral", "expansion"][vol_state_idx]
                    
                    acceleration = None
                    if output.acceleration is not None:
                        acceleration = float(output.acceleration.cpu().numpy().mean())
                else:
                    # Fallback: standard forward() returns just class logits
                    output = model(features)
                    probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                    mu = None
                    sigma = None
                    quantiles = None
                    entry_offset = None
                    sl_distance = None
                    tp_distance = None
                    vol_state = None
                    vol_state_probs = None
                    acceleration = None
                
                # Calibrate probabilities
                probs = self._calibrate_probs(probs, model_name)
                
                # Direction and confidence margin
                action_idx = int(np.argmax(probs))
                sorted_probs = np.sort(probs)[::-1]
                confidence = float(probs[action_idx])
                confidence_margin = float(sorted_probs[0] - sorted_probs[1])
                
                action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
                
                result = {
                    "model": model_name,
                    "probs": probs.tolist(),
                    "action_idx": action_idx,
                    "action": action_map[action_idx],
                    "confidence": confidence,
                    "confidence_margin": confidence_margin,
                    "p_long": float(probs[2]),
                    "p_short": float(probs[0]),
                    "p_hold": float(probs[1]),
                    # Multi-head outputs (None if not available)
                    "mu": mu,
                    "sigma": sigma,
                    "quantiles": quantiles,
                    "entry_offset": entry_offset,
                    "sl_distance": sl_distance,
                    "tp_distance": tp_distance,
                    # Flow Forecast outputs
                    "vol_state": vol_state,
                    "vol_state_probs": vol_state_probs,
                    "acceleration": acceleration,
                    "has_multihead": has_multihead
                }
                
                return result
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_mc_dropout_uncertainty(
        self, 
        model: torch.nn.Module, 
        features: torch.Tensor, 
        model_name: str,
        n_samples: int = 10
    ) -> Dict[str, float]:
        """
        Monte Carlo Dropout for epistemic uncertainty estimation.
        
        Runs N forward passes with dropout ENABLED during inference.
        The variance of predictions indicates model uncertainty about this input.
        
        Paper: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
               (Gal & Ghahramani, 2016)
        
        High variance = model is uncertain (avoid trading)
        Low variance = model is confident (consider trading)
        
        2024 Research: MC Dropout is the most practical uncertainty quantification
        technique for deep learning, especially in financial applications.
        
        Args:
            model: Neural network model with dropout layers
            features: Input features tensor
            model_name: Name of the model
            n_samples: Number of forward passes (10 is typical)
            
        Returns:
            Dict with:
            - epistemic_uncertainty: Variance of predictions (higher = more uncertain)
            - predictive_entropy: Entropy of mean predictions
            - sample_probs: List of probability arrays from each sample
            - mean_probs: Mean probabilities across samples
            - action_agreement: Fraction of samples that agree on action
        """
        try:
            # Enable dropout during inference (key for MC Dropout)
            def enable_dropout(m):
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            
            model.apply(enable_dropout)
            
            all_probs = []
            all_actions = []
            inference_temp = self.config.get('inference_temperature', 0.7)
            
            for _ in range(n_samples):
                with torch.no_grad():
                    if hasattr(model, 'forward_multihead'):
                        output = model.forward_multihead(features)
                        scaled_logits = output.class_logits / inference_temp
                        probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
                    else:
                        output = model(features)
                        probs = F.softmax(output, dim=-1).cpu().numpy()
                    
                    # Handle batch dimension
                    if len(probs.shape) == 2 and probs.shape[0] == 1:
                        probs = probs[0]
                    elif len(probs.shape) == 2:
                        probs = probs.mean(axis=0)
                    
                    all_probs.append(probs)
                    all_actions.append(int(np.argmax(probs)))
            
            # Restore eval mode
            model.eval()
            
            # Stack all samples
            prob_array = np.stack(all_probs)  # [n_samples, 3]
            
            # Compute mean probabilities
            mean_probs = prob_array.mean(axis=0)
            
            # Epistemic uncertainty: variance of predictions
            epistemic_uncertainty = prob_array.var(axis=0).mean()
            
            # Per-class uncertainty
            per_class_var = prob_array.var(axis=0).tolist()
            
            # Predictive entropy of mean predictions
            predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
            
            # Action agreement: what fraction of samples agree on the action
            action_counts = np.bincount(all_actions, minlength=3)
            action_agreement = action_counts.max() / n_samples
            
            # Determine most likely action from mean probs
            mean_action_idx = int(np.argmax(mean_probs))
            action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
            
            return {
                "epistemic_uncertainty": float(epistemic_uncertainty),
                "per_class_uncertainty": per_class_var,
                "predictive_entropy": float(predictive_entropy),
                "mean_probs": mean_probs.tolist(),
                "mean_action": action_map[mean_action_idx],
                "action_agreement": float(action_agreement),
                "n_samples": n_samples,
                "should_trade": epistemic_uncertainty < 0.02 and action_agreement > 0.7
            }
            
        except Exception as e:
            logger.error(f"MC Dropout error for {model_name}: {e}")
            return {
                "epistemic_uncertainty": 1.0,
                "per_class_uncertainty": [0.33, 0.33, 0.33],
                "predictive_entropy": np.log(3),  # Max entropy for 3 classes
                "mean_probs": [0.33, 0.33, 0.33],
                "mean_action": "HOLD",
                "action_agreement": 0.33,
                "n_samples": 0,
                "should_trade": False
            }
    
    def calculate_kelly_position_size(
        self,
        win_probability: float,
        expected_win: float,
        expected_loss: float,
        max_position_pct: float = 0.25,
        kelly_fraction: float = 0.5,
        min_edge_threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        Kelly Criterion for optimal position sizing.
        
        The Kelly Criterion determines the optimal fraction of capital to risk
        to maximize long-term growth while avoiding ruin.
        
        Formula: f* = (p * b - q) / b
        Where:
        - f* = optimal fraction to bet
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = odds received on the wager (win/loss ratio)
        
        Paper: "A New Interpretation of Information Rate" (Kelly, 1956)
        
        2024 Research: Half-Kelly (kelly_fraction=0.5) is commonly used in practice
        to reduce variance while maintaining most of the growth benefit.
        
        Args:
            win_probability: Probability of winning trade (from calibrated model)
            expected_win: Expected profit if win (from quantiles, e.g., q75 - entry)
            expected_loss: Expected loss if lose (from quantiles, e.g., entry - q25)
            max_position_pct: Maximum position size cap (default 25%)
            kelly_fraction: Fraction of full Kelly to use (default 0.5 = Half-Kelly)
            min_edge_threshold: Minimum edge required to trade (default 1%)
            
        Returns:
            Dict with:
            - full_kelly: Uncapped Kelly fraction
            - adjusted_kelly: Half-Kelly (or custom fraction)
            - capped_position: Final position size after max cap
            - edge: Expected edge (expected_value / expected_loss)
            - should_bet: True if edge exceeds threshold
        """
        try:
            # Input validation
            win_probability = max(0.001, min(0.999, win_probability))  # Clamp to valid range
            expected_loss = max(0.001, expected_loss)  # Prevent division by zero
            expected_win = max(0.0, expected_win)
            
            # Calculate loss probability
            loss_probability = 1.0 - win_probability
            
            # Calculate odds (b = win amount / loss amount)
            odds = expected_win / expected_loss if expected_loss > 0 else 0
            
            # Kelly formula: f* = (p * b - q) / b
            # Rearranged: f* = p - q/b = p - (1-p)/b
            if odds > 0:
                full_kelly = (win_probability * odds - loss_probability) / odds
            else:
                full_kelly = 0.0
            
            # Apply Kelly fraction (Half-Kelly is common in practice)
            adjusted_kelly = full_kelly * kelly_fraction
            
            # Cap at maximum position size
            capped_position = max(0.0, min(max_position_pct, adjusted_kelly))
            
            # Calculate expected edge
            expected_value = (win_probability * expected_win) - (loss_probability * expected_loss)
            edge = expected_value / expected_loss if expected_loss > 0 else 0
            
            # Determine if we should bet
            should_bet = edge > min_edge_threshold and capped_position > 0.01
            
            return {
                "full_kelly": float(full_kelly),
                "adjusted_kelly": float(adjusted_kelly),
                "capped_position": float(capped_position),
                "edge": float(edge),
                "expected_value": float(expected_value),
                "win_probability": float(win_probability),
                "odds_ratio": float(odds),
                "should_bet": should_bet,
                "kelly_fraction_used": kelly_fraction,
                "max_cap_applied": capped_position < adjusted_kelly
            }
            
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return {
                "full_kelly": 0.0,
                "adjusted_kelly": 0.0,
                "capped_position": 0.0,
                "edge": 0.0,
                "expected_value": 0.0,
                "win_probability": 0.5,
                "odds_ratio": 1.0,
                "should_bet": False,
                "kelly_fraction_used": kelly_fraction,
                "max_cap_applied": False
            }
    
    def _detect_regime_from_vae(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Tuple[MarketRegime, float]:
        """Use VAE latent space for regime detection."""
        try:
            model.eval()
            with torch.no_grad():
                # Get latent representation
                # === FIX: Handle VAE encode correctly ===
                # MultiHeadVAE.encode() returns single z tensor, not (mu, log_var)
                # Use encode_to_latent() if available for (mu, log_var) tuple
                if hasattr(model, 'get_latent'):
                    z = model.get_latent(features)
                elif hasattr(model, 'encode_to_latent'):
                    # Use encode_to_latent() which returns (mu, log_var)
                    mu, log_var = model.encode_to_latent(features)
                    z = mu  # Use mu for regime stability
                elif hasattr(model, 'encode'):
                    # encode() may return single tensor (z) or tuple (mu, log_var)
                    result = model.encode(features)
                    if isinstance(result, tuple) and len(result) == 2:
                        mu, _ = result
                        z = mu
                    else:
                        z = result  # Single tensor returned
                else:
                    # Fallback: use forward pass
                    output = model(features)
                    probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                    
                    # High HOLD probability suggests ranging/choppy market
                    p_hold = probs[1]
                    p_directional = probs[0] + probs[2]
                    
                    if p_hold > 0.6:
                        return MarketRegime.CHOPPY, float(p_hold)
                    elif p_directional > 0.7:
                        return MarketRegime.TRENDING, float(p_directional)
                    else:
                        return MarketRegime.RANGING, 0.5
                
                # Analyze latent dimensions for regime
                z_np = z.cpu().numpy()[0]
                
                # === FIX: Check for NaN/inf in latent space ===
                # Corrupted latent values contaminate regime detection
                if np.isnan(z_np).any() or np.isinf(z_np).any():
                    logger.warning(f"[{model_name}] Latent space contains NaN/inf - returning UNKNOWN regime")
                    return MarketRegime.UNKNOWN, 0.0
                
                z_var = np.var(z_np)
                z_mean_abs = np.mean(np.abs(z_np))
                
                # High variance in latent = volatile/transitioning
                # Low variance + low mean = stable/ranging
                # Low variance + high mean = trending
                if z_var > 1.5:
                    return MarketRegime.HIGH_VOLATILITY, min(0.9, z_var / 3.0)
                elif z_mean_abs > 1.0:
                    return MarketRegime.TRENDING, min(0.9, z_mean_abs / 2.0)
                elif z_var < 0.5:
                    return MarketRegime.CHOPPY, 1.0 - z_var
                else:
                    return MarketRegime.RANGING, 0.5
                    
        except Exception as e:
            logger.error(f"Regime detection error for {model_name}: {e}")
            return MarketRegime.UNKNOWN, 0.0
    
    def _detect_risk_from_gnn(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Tuple[RiskRegime, float]:
        """Use GNN for cross-asset risk regime detection."""
        try:
            model.eval()
            with torch.no_grad():
                # Get prediction
                output = model(features)
                probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                
                # Get asset relations if available
                if hasattr(model, 'get_asset_relations'):
                    relations = model.get_asset_relations(features)
                    relations_np = relations.cpu().numpy()[0]
                    
                    # High correlation = potential shock/risk-off
                    off_diag = relations_np[~np.eye(relations_np.shape[0], dtype=bool)]
                    avg_correlation = float(np.mean(np.abs(off_diag)))
                    
                    if avg_correlation > 0.8:
                        return RiskRegime.CORRELATION_SHOCK, avg_correlation
                    elif avg_correlation > 0.6:
                        return RiskRegime.RISK_OFF, avg_correlation
                    elif avg_correlation < 0.3:
                        return RiskRegime.RISK_ON, 1.0 - avg_correlation
                    else:
                        return RiskRegime.NEUTRAL, 0.5
                
                # Fallback: use directional signal
                p_short = probs[0]
                p_long = probs[2]
                
                if p_short > 0.6:
                    return RiskRegime.RISK_OFF, float(p_short)
                elif p_long > 0.6:
                    return RiskRegime.RISK_ON, float(p_long)
                else:
                    return RiskRegime.NEUTRAL, 0.5
                    
        except Exception as e:
            logger.error(f"Risk detection error for {model_name}: {e}")
            return RiskRegime.UNKNOWN, 0.0
    
    def _compute_weighted_consensus(self, predictions: List[Dict]) -> Tuple[str, float, float, float]:
        """Compute weighted consensus from direction model predictions."""
        if not predictions:
            return "HOLD", 0.0, 0.0, 1.0
        
        # Get weights for each model
        weight_sum = 0
        weighted_votes = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0}
        weighted_probs = np.zeros(3)
        
        for pred in predictions:
            model_name = pred["model"]
            weight = self.model_weights.get(model_name, ModelWeight(
                model_name=model_name,
                expectancy=0, precision_on_trade=0.5,
                profit_factor=1.0, f1_directional=0.4, sharpe=0
            )).composite_weight
            
            weighted_votes[pred["action"]] += weight
            weighted_probs += weight * np.array(pred["probs"])
            weight_sum += weight
        
        # Normalize
        if weight_sum > 0:
            weighted_probs /= weight_sum
            for action in weighted_votes:
                weighted_votes[action] /= weight_sum
        
        # Determine consensus action
        consensus_action = max(weighted_votes, key=weighted_votes.get)
        weighted_agreement = weighted_votes[consensus_action]
        
        # Compute disagreement score (entropy of vote distribution)
        vote_probs = np.array(list(weighted_votes.values()))
        vote_probs = vote_probs / (vote_probs.sum() + 1e-8)
        disagreement = -np.sum(vote_probs * np.log(vote_probs + 1e-8)) / np.log(3)
        
        # Average confidence margin
        avg_margin = float(np.mean([p["confidence_margin"] for p in predictions]))
        
        return consensus_action, weighted_agreement, avg_margin, disagreement
    
    def predict(self, features: np.ndarray) -> EnsembleSignal:
        """
        Make ensemble prediction with regime gating and risk filtering.
        
        Steps:
        1. Get predictions from all direction models
        2. Compute weighted consensus
        3. Detect market regime from VAE (adjust thresholds)
        4. Detect risk regime from GNN (adjust position size)
        5. Apply gating logic
        6. Return final signal
        """
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Step 1: Get direction model predictions
        direction_predictions = []
        for name, model in self.direction_models.items():
            pred = self._get_model_prediction(model, x, name)
            if pred:
                direction_predictions.append(pred)
        
        # Step 2: Weighted consensus
        consensus_action, weighted_agreement, avg_margin, disagreement = \
            self._compute_weighted_consensus(direction_predictions)
        
        # Step 3: Regime detection from VAE
        market_regime = MarketRegime.UNKNOWN
        regime_confidence = 0.0
        for name, model in self.regime_models.items():
            regime, conf = self._detect_regime_from_vae(model, x, name)
            if conf > regime_confidence:
                market_regime = regime
                regime_confidence = conf
        
        # Step 4: Risk detection from GNN
        risk_regime = RiskRegime.UNKNOWN
        risk_confidence = 0.0
        for name, model in self.risk_models.items():
            risk, conf = self._detect_risk_from_gnn(model, x, name)
            if conf > risk_confidence:
                risk_regime = risk
                risk_confidence = conf
        
        # Step 5: Apply regime gating
        confidence_threshold = self.base_confidence_threshold
        margin_threshold = self.base_margin_threshold
        position_multiplier = 1.0
        regime_adjustment = "NONE"
        reasons = []
        
        # VAE regime gating
        if market_regime == MarketRegime.CHOPPY:
            confidence_threshold *= 1.5
            margin_threshold *= 1.5
            position_multiplier *= 0.5
            regime_adjustment = "RAISED_THRESHOLDS (choppy market)"
            reasons.append(f"VAE detects choppy market (conf={regime_confidence:.2f}) - raised thresholds")
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            position_multiplier *= 0.7
            regime_adjustment = "REDUCED_SIZE (high volatility)"
            reasons.append(f"VAE detects high volatility - reduced position size")
        elif market_regime == MarketRegime.TRENDING:
            position_multiplier *= 1.1
            reasons.append(f"VAE detects trending market - favorable conditions")
        
        # GNN risk gating
        if risk_regime == RiskRegime.CORRELATION_SHOCK:
            confidence_threshold *= 2.0
            position_multiplier *= 0.3
            reasons.append(f"GNN detects correlation shock - extreme caution")
        elif risk_regime == RiskRegime.RISK_OFF:
            if consensus_action == "LONG":
                confidence_threshold *= 1.3
            position_multiplier *= 0.7
            reasons.append(f"GNN detects risk-off regime - reduced exposure")
        elif risk_regime == RiskRegime.RISK_ON:
            if consensus_action == "SHORT":
                confidence_threshold *= 1.2
            reasons.append(f"GNN detects risk-on regime")
        
        # Step 6: Final decision
        # Compute ensemble probabilities
        if direction_predictions:
            ensemble_probs = np.mean([p["probs"] for p in direction_predictions], axis=0)
        else:
            ensemble_probs = np.array([0.2, 0.6, 0.2])
        
        # === Step 6b: Aggregate multi-head outputs (quantiles, mu, sigma, trading params) ===
        # Weight-average outputs from models that support forward_multihead()
        multihead_preds = [p for p in direction_predictions if p.get("has_multihead") and p.get("quantiles")]
        
        aggregated_quantiles = None
        aggregated_mu = None
        aggregated_sigma = None
        aggregated_entry_offset = None
        aggregated_sl_distance = None
        aggregated_tp_distance = None
        
        if multihead_preds:
            # Compute weights for multihead models
            total_weight = 0.0
            q_sum = {"q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0}
            mu_sum = 0.0
            sigma_sum = 0.0
            entry_sum = 0.0
            sl_sum = 0.0
            tp_sum = 0.0
            
            for pred in multihead_preds:
                weight_obj = self.model_weights.get(pred["model"], ModelWeight(
                    model_name=pred["model"],
                    expectancy=0, precision_on_trade=0.5,
                    profit_factor=1.0, f1_directional=0.4, sharpe=0
                ))
                w = weight_obj.composite_weight
                total_weight += w
                
                q = pred["quantiles"]
                for key in q_sum:
                    q_sum[key] += q[key] * w
                
                if pred.get("mu") is not None:
                    mu_sum += pred["mu"] * w
                if pred.get("sigma") is not None:
                    sigma_sum += pred["sigma"] * w
                if pred.get("entry_offset") is not None:
                    entry_sum += pred["entry_offset"] * w
                if pred.get("sl_distance") is not None:
                    sl_sum += pred["sl_distance"] * w
                if pred.get("tp_distance") is not None:
                    tp_sum += pred["tp_distance"] * w
            
            if total_weight > 0:
                aggregated_quantiles = {k: v / total_weight for k, v in q_sum.items()}
                aggregated_mu = mu_sum / total_weight
                aggregated_sigma = sigma_sum / total_weight
                aggregated_entry_offset = entry_sum / total_weight
                aggregated_sl_distance = sl_sum / total_weight
                aggregated_tp_distance = tp_sum / total_weight
                
                reasons.append(f"Multi-head: {len(multihead_preds)} models contributed quantiles (q50={aggregated_quantiles['q50']:.4f})")
        
        # === Step 6c: Aggregate Flow Forecast outputs (vol_state, acceleration) ===
        flow_preds = [p for p in direction_predictions if p.get("has_multihead") and p.get("vol_state")]
        
        aggregated_vol_state = None
        aggregated_vol_state_probs = None
        aggregated_acceleration = None
        aggregated_forecast_mode = None
        aggregated_quantile_paths = None
        
        if flow_preds:
            # Vote-based vol_state (majority wins)
            vs_counts = {"contraction": 0, "neutral": 0, "expansion": 0}
            vs_prob_sum = {"contraction": 0.0, "neutral": 0.0, "expansion": 0.0}
            accel_sum = 0.0
            flow_weight = 0.0
            
            for pred in flow_preds:
                w = self.model_weights.get(pred["model"], ModelWeight(
                    model_name=pred["model"],
                    expectancy=0, precision_on_trade=0.5,
                    profit_factor=1.0, f1_directional=0.4, sharpe=0
                )).composite_weight
                flow_weight += w
                
                vs = pred.get("vol_state")
                if vs:
                    vs_counts[vs] += 1
                    
                vs_probs = pred.get("vol_state_probs")
                if vs_probs:
                    for k in vs_prob_sum:
                        vs_prob_sum[k] += vs_probs.get(k, 0) * w
                
                if pred.get("acceleration") is not None:
                    accel_sum += pred["acceleration"] * w
            
            if flow_weight > 0:
                aggregated_vol_state_probs = {k: v / flow_weight for k, v in vs_prob_sum.items()}
                aggregated_vol_state = max(vs_counts.keys(), key=lambda k: aggregated_vol_state_probs[k])
                aggregated_acceleration = accel_sum / flow_weight
                
                # === Compute forecast_mode and quantile_paths ===
                if aggregated_quantiles:
                    q10 = aggregated_quantiles.get("q10", -0.01)
                    q25 = aggregated_quantiles.get("q25", -0.005)
                    q50 = aggregated_quantiles.get("q50", 0.0)
                    q75 = aggregated_quantiles.get("q75", 0.005)
                    q90 = aggregated_quantiles.get("q90", 0.01)
                    
                    # Volatility gate: NO_FORECAST when vol_state==contraction OR spread too narrow
                    spread = q75 - q25  # IQR as percentage return
                    cost = 0.001  # ~0.1% round-trip
                    min_spread = 3 * cost  # Must exceed 3x trading cost
                    
                    if aggregated_vol_state == "contraction" or spread < min_spread:
                        aggregated_forecast_mode = "NO_FORECAST"
                        aggregated_quantile_paths = None
                        reasons.append(f"Flow Forecast: NO_FORECAST (vol={aggregated_vol_state}, spread={spread*100:.3f}% < {min_spread*100:.3f}%)")
                    else:
                        aggregated_forecast_mode = "QUANTILE_PATHS"
                        
                        # Alpha-shaping based on vol_state: contraction=0.7, neutral=1.0, expansion=1.5
                        ALPHA_MAP = {"contraction": 0.7, "neutral": 1.0, "expansion": 1.5}
                        alpha = ALPHA_MAP.get(aggregated_vol_state, 1.0)
                        
                        # Generate paths: path[k] = 1 + ((k/h)^α * quantile)
                        # Note: paths are relative multipliers, not absolute prices (computed in API)
                        horizon = 16  # 16 bars = 4 hours at 15m
                        steps = list(range(1, horizon + 1))
                        
                        aggregated_quantile_paths = {
                            "q10": [float(np.exp((k / horizon) ** alpha * q10)) for k in steps],
                            "q50": [float(np.exp((k / horizon) ** alpha * q50)) for k in steps],
                            "q90": [float(np.exp((k / horizon) ** alpha * q90)) for k in steps],
                        }
                        
                        reasons.append(f"Flow Forecast: QUANTILE_PATHS (vol={aggregated_vol_state}, α={alpha:.1f}, accel={aggregated_acceleration:.4f})")
        
        confidence = float(ensemble_probs.max())
        sorted_probs = np.sort(ensemble_probs)[::-1]
        final_margin = float(sorted_probs[0] - sorted_probs[1])
        
        # Check thresholds
        passes_confidence = avg_margin >= confidence_threshold
        passes_agreement = weighted_agreement >= self.majority_weight_threshold
        passes_margin = final_margin >= margin_threshold
        
        final_action = consensus_action
        if consensus_action in ["LONG", "SHORT"]:
            if not (passes_confidence and passes_agreement and passes_margin):
                final_action = "HOLD"
                reasons.append(f"Gated: conf={passes_confidence}, agree={passes_agreement}, margin={passes_margin}")
        
        # Compute edge
        p_long = float(ensemble_probs[2])
        p_short = float(ensemble_probs[0])
        mu = (p_long - p_short) * 0.01
        cost = 0.001  # ~0.1% round-trip
        edge = abs(mu) - cost
        
        # Position sizing
        base_position = 0.05  # 5% base
        if final_action in ["LONG", "SHORT"]:
            position_size = base_position * confidence * position_multiplier
            position_size = max(0.01, min(0.10, position_size))  # 1-10% range
        else:
            position_size = 0.0
        
        regime_adjusted_size = position_size * position_multiplier
        
        # Model votes breakdown
        model_votes = {}
        for pred in direction_predictions:
            weight = self.model_weights.get(pred["model"], ModelWeight(
                model_name=pred["model"],
                expectancy=0, precision_on_trade=0.5,
                profit_factor=1.0, f1_directional=0.4, sharpe=0
            ))
            model_votes[pred["model"]] = {
                "action": pred["action"],
                "confidence": pred["confidence"],
                "confidence_margin": pred["confidence_margin"],
                "weight": weight.composite_weight,
                "probs": {
                    "SHORT": pred["probs"][0],
                    "HOLD": pred["probs"][1],
                    "LONG": pred["probs"][2]
                }
            }
        
        # Count agreement
        if direction_predictions:
            agreement_count = sum(1 for p in direction_predictions if p["action"] == consensus_action)
            agreement_pct = agreement_count / len(direction_predictions)
        else:
            agreement_pct = 0.0
        
        reasons.append(f"Consensus: {consensus_action} ({agreement_pct*100:.0f}% models, {weighted_agreement*100:.0f}% weight)")
        
        return EnsembleSignal(
            action=final_action,
            confidence=confidence,
            confidence_margin=final_margin,
            edge=edge,
            market_regime=market_regime.value,
            risk_regime=risk_regime.value,
            regime_confidence=regime_confidence,
            agreement_pct=agreement_pct,
            weighted_agreement=weighted_agreement,
            disagreement_score=disagreement,
            position_size_pct=position_size,
            regime_adjusted_size=regime_adjusted_size,
            confidence_threshold_used=confidence_threshold,
            regime_adjustment=regime_adjustment,
            model_votes=model_votes,
            reasons=reasons,
            # Multi-head aggregated outputs
            quantiles=aggregated_quantiles,
            mu=aggregated_mu,
            sigma=aggregated_sigma,
            entry_offset=aggregated_entry_offset,
            sl_distance=aggregated_sl_distance,
            tp_distance=aggregated_tp_distance,
            # Flow Forecast outputs
            vol_state=aggregated_vol_state,
            vol_state_probs=aggregated_vol_state_probs,
            acceleration=aggregated_acceleration,
            forecast_mode=aggregated_forecast_mode,
            quantile_paths=aggregated_quantile_paths,
            ensemble_probs={
                "SHORT": float(ensemble_probs[0]),
                "HOLD": float(ensemble_probs[1]),
                "LONG": float(ensemble_probs[2])
            }
        )
