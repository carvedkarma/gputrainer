"""
Multi-Head Loss Functions for Institutional Trading

Combined loss = L_class + λ₁·L_regression + λ₂·L_quantile

Where:
- L_class: CrossEntropyLoss for direction classification
- L_regression: MSE for expected return (μ)
- L_quantile: Pinball loss for quantile regression (q10, q25, q50, q75, q90)

PHASE 1c: Quantile-based SL/TP derivation
Instead of predicting SL/TP as separate targets, derive them from quantiles:
- For LONG: SL from q10 or q25 (downside risk), TP from q75 or q90 (upside)
- For SHORT: SL from q75 or q90 (upside risk), TP from q10 or q25 (downside)
This ensures internal consistency and uses the distribution we already train.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Module-level logger for loss functions
logger = logging.getLogger(__name__)


def derive_sl_tp_from_quantiles(
    quantiles: torch.Tensor,
    direction: torch.Tensor,
    current_price: float = 1.0,
    sl_quantile_idx: int = 0,  # q10 index
    tp_quantile_idx: int = 4,  # q90 index
    conservative_sl_idx: int = 1,  # q25 index (less aggressive SL)
    conservative_tp_idx: int = 3,  # q75 index (less aggressive TP)
    use_conservative: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PHASE 1c: Derive Stop Loss and Take Profit from predicted quantiles.
    
    This enforces internal consistency - SL/TP come from the same distribution
    that the model is already learning, rather than being separate targets.
    
    Args:
        quantiles: [batch, 5] predicted return quantiles (q10, q25, q50, q75, q90)
        direction: [batch] trade direction (0=SHORT, 1=HOLD, 2=LONG)
        current_price: Current price for converting returns to distances
        sl_quantile_idx: Index for aggressive SL (0=q10)
        tp_quantile_idx: Index for aggressive TP (4=q90)
        conservative_sl_idx: Index for conservative SL (1=q25)
        conservative_tp_idx: Index for conservative TP (3=q75)
        use_conservative: Whether to use q25/q75 instead of q10/q90
        
    Returns:
        sl_distance: [batch, 1] stop loss as percentage distance (always positive)
        tp_distance: [batch, 1] take profit as percentage distance (always positive)
    """
    batch_size = quantiles.shape[0]
    
    # Select quantile indices based on conservatism
    sl_idx = conservative_sl_idx if use_conservative else sl_quantile_idx
    tp_idx = conservative_tp_idx if use_conservative else tp_quantile_idx
    
    # Initialize outputs
    sl_distance = torch.zeros(batch_size, 1, device=quantiles.device)
    tp_distance = torch.zeros(batch_size, 1, device=quantiles.device)
    
    # LONG positions: SL from q10/q25 (downside), TP from q75/q90 (upside)
    long_mask = (direction == 2)
    if long_mask.any():
        # For longs: q10/q25 is the worst case downside (negative return)
        # SL distance = |q10| (how far down before we exit)
        sl_distance[long_mask] = torch.abs(quantiles[long_mask, sl_idx:sl_idx+1])
        # TP distance = q90/q75 (how far up before we take profit)
        tp_distance[long_mask] = quantiles[long_mask, tp_idx:tp_idx+1].clamp(min=0)
    
    # SHORT positions: SL from q75/q90 (upside), TP from q10/q25 (downside)
    short_mask = (direction == 0)
    if short_mask.any():
        # For shorts: q90/q75 is the worst case upside (positive return = loss for shorts)
        # SL distance = q90 (how far up before we exit)
        sl_distance[short_mask] = quantiles[short_mask, tp_idx:tp_idx+1].clamp(min=0)
        # TP distance = |q10/q25| (how far down = profit for shorts)
        tp_distance[short_mask] = torch.abs(quantiles[short_mask, sl_idx:sl_idx+1])
    
    # HOLD positions: No trade, keep zeros (or small defaults)
    hold_mask = (direction == 1)
    if hold_mask.any():
        # Default small values for HOLD
        sl_distance[hold_mask] = 0.005  # 0.5%
        tp_distance[hold_mask] = 0.005  # 0.5%
    
    # Enforce minimum SL/TP to avoid micro-trades
    min_distance = 0.001  # 0.1% minimum
    sl_distance = sl_distance.clamp(min=min_distance)
    tp_distance = tp_distance.clamp(min=min_distance)
    
    return sl_distance, tp_distance


@dataclass
class MultiHeadLossConfig:
    """Configuration for multi-head loss weights."""
    
    # === CRITICAL STABILITY FIX: REGRESSION HEAD REMOVED ENTIRELY (Feb 2026) ===
    # Diagnostics showed ENTIRE regression path causing gradient explosions:
    # - regression_head.shared layers: 8-59 (exploding)
    # - regression_head.sigma_head: 6-42 (exploding)
    # - classifier: 0.68-1.71 (stable)
    # 
    # FINAL ARCHITECTURE: Classification ONLY
    # Use external ATR/rolling volatility for position sizing instead of learned σ
    
    # ENABLED heads - ONLY classification
    lambda_class: float = 1.0       # Classification - ONLY ENABLED HEAD
    lambda_mu: float = 0.0          # μ REGRESSION - PERMANENTLY DISABLED
    lambda_sigma: float = 0.0       # σ REGRESSION - PERMANENTLY DISABLED (shared layers explode)
    
    # DISABLED heads - set to 0 to completely skip backward pass
    lambda_quantile: float = 0.0    # DISABLED - set to 0
    lambda_trading: float = 0.0     # DISABLED - set to 0
    lambda_candle: float = 0.0      # DISABLED - set to 0
    
    # Flow Forecast heads - DISABLED
    lambda_vol_state: float = 0.0   # DISABLED - set to 0
    lambda_acceleration: float = 0.0  # DISABLED - set to 0
    
    # Head enable flags - for skipping forward computation entirely
    head_enabled_quantile: bool = False      # Skip quantile forward
    head_enabled_trading: bool = False       # Skip trading forward
    head_enabled_candle: bool = False        # Skip candle forward
    head_enabled_vol_state: bool = False     # Skip vol_state forward
    head_enabled_acceleration: bool = False  # Skip acceleration forward
    head_enabled_mu: bool = False            # Skip mu forward (expected return)
    head_enabled_sigma: bool = False         # Skip sigma forward (uncertainty)
    
    # Classification options
    class_weights: Optional[torch.Tensor] = None  # For imbalanced classes
    label_smoothing: float = 0.0    # DISABLED - harms imbalanced classification
    
    # STABILITY FIX: Disable all aggressive classification tricks
    # Re-enable ONE AT A TIME after stable training is achieved
    use_focal_loss: bool = False    # DISABLED - use plain CrossEntropy first
    focal_gamma: float = 2.0        # Not used when focal loss disabled
    focal_alpha: Optional[torch.Tensor] = None
    
    # OHEM disabled for stability
    use_ohem: bool = False          # DISABLED - can destabilize early training
    ohem_keep_ratio: float = 0.3
    ohem_min_keep: int = 8
    
    # Confidence Penalty disabled for stability
    use_confidence_penalty: bool = False  # DISABLED - was interfering with learning
    confidence_penalty_beta: float = 0.0  # Zero weight
    
    # Inference calibration (NEW) - sharpen soft predictions
    inference_temperature: float = 0.7  # T < 1 sharpens predictions at inference
    
    # Regression options
    mu_huber_delta: float = 0.02    # Delta for Huber loss (robust to outliers)
    trading_huber_delta: float = 0.01  # Delta for trading distances
    candle_huber_delta: float = 0.02   # Delta for candle deltas
    acceleration_huber_delta: float = 0.02  # Delta for acceleration loss
    
    # Quantile options
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    
    # Future candle options
    n_future_candles: int = 5
    
    # Phase 1b: Log-sigma mode for proper Gaussian NLL calibration
    # ENABLED by default: model outputs log(sigma) for proper uncertainty calibration
    # This prevents σ from being "gamed" and couples uncertainty to prediction error
    use_log_sigma: bool = True  # Model outputs log(σ) for better calibration
    
    # Phase 1c: Derive SL/TP from quantiles instead of separate heads
    derive_sl_tp_from_quantiles: bool = True  # Use quantile-based SL/TP derivation


class PinballLoss(nn.Module):
    """
    Pinball (quantile) loss for quantile regression.
    
    For quantile q and error e = y - y_hat:
    L_q(e) = q * max(e, 0) + (1-q) * max(-e, 0)
           = max(q*e, (q-1)*e)
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer('q_tensor', torch.tensor(quantiles))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss.
        
        Args:
            predictions: [batch, n_quantiles] predicted quantile values
            targets: [batch, 1] or [batch] actual values
            
        Returns:
            Scalar loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        # Broadcast targets to match quantiles: [batch, n_quantiles]
        targets = targets.expand_as(predictions)
        
        # Compute errors: e = y - y_hat
        errors = targets - predictions  # [batch, n_quantiles]
        
        # Get quantiles on same device
        q = self.q_tensor.to(predictions.device)
        
        # Pinball loss: max(q*e, (q-1)*e)
        loss = torch.max(q * errors, (q - 1) * errors)
        
        return loss.mean()


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss Wrapper.
    
    Paper: "Training Region-based Object Detectors with Online Hard Example Mining" (CVPR 2016)
    
    OHEM focuses training on the hardest examples by:
    1. Computing loss for all samples in a batch
    2. Sorting by loss (descending)
    3. Keeping only top-k% hardest examples for backpropagation
    
    Benefits for trading signals:
    - Forces model to learn from difficult-to-classify trades
    - Improves LONG/SHORT recall by 15-25%
    - Reduces overfitting to easy HOLD examples
    
    2024 Research: OHEM + Focal Loss combination is state-of-the-art for imbalanced classification.
    """
    
    def __init__(self, base_loss: nn.Module, keep_ratio: float = 0.3, min_keep: int = 8):
        """
        Args:
            base_loss: Underlying loss function (e.g., FocalLoss, CrossEntropyLoss)
            keep_ratio: Fraction of hardest examples to keep (0.3 = top 30%)
            min_keep: Minimum number of examples to keep per batch
        """
        super().__init__()
        self.base_loss = base_loss
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
    
    def set_alpha(self, alpha: torch.Tensor):
        """
        Pass-through to set alpha weights on underlying FocalLoss.
        
        BUG FIX: Previously, when OHEM wrapped FocalLoss, the set_alpha call
        would fail because OHEMLoss didn't have this method. This caused
        class weights to never be applied, contributing to mode collapse.
        """
        if hasattr(self.base_loss, 'set_alpha'):
            self.base_loss.set_alpha(alpha)
            logger.info(f"[OHEM] Passed alpha weights to underlying FocalLoss: {alpha.tolist()}")
        else:
            logger.warning("[OHEM] base_loss does not support set_alpha - weights not applied")
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute OHEM-filtered loss.
        
        Args:
            inputs: [batch, num_classes] raw logits
            targets: [batch] class indices
            
        Returns:
            Scalar loss (mean of top-k hardest examples)
        """
        batch_size = inputs.size(0)
        
        # Get per-sample loss (without reduction)
        if hasattr(self.base_loss, 'reduction'):
            original_reduction = self.base_loss.reduction
            self.base_loss.reduction = 'none'
            per_sample_loss = self.base_loss(inputs, targets)
            self.base_loss.reduction = original_reduction
        else:
            # For losses that don't have reduction attribute
            per_sample_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate number of examples to keep
        num_keep = max(self.min_keep, int(batch_size * self.keep_ratio))
        num_keep = min(num_keep, batch_size)
        
        # Sort by loss (descending) and select top-k
        sorted_loss, _ = torch.sort(per_sample_loss, descending=True)
        hard_loss = sorted_loss[:num_keep]
        
        return hard_loss.mean()


class ConfidencePenaltyLoss(nn.Module):
    """
    Confidence Penalty (Entropy Maximization) Loss.
    
    Paper: "Regularizing Neural Networks by Penalizing Confident Output Distributions" (2017)
    
    Prevents overconfident predictions by adding an entropy bonus:
    L_total = L_classification - β * H(p)
    
    Where H(p) = -Σ p_i * log(p_i) is the prediction entropy.
    
    Benefits for trading:
    - Prevents model from being overconfident on uncertain market conditions
    - Improves selective classification (knows when NOT to trade)
    - Better calibrated confidence scores
    
    2024 Research: Outperforms label smoothing for selective classification.
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Weight for entropy penalty (0.1 is typical)
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence penalty to prevent OVERCONFIDENT predictions.
        
        BUG FIX: Previously returned negative entropy, which when ADDED to loss
        would REDUCE total loss for uniform predictions - actively encouraging
        mode collapse! Now returns POSITIVE penalty for LOW entropy (overconfidence).
        
        Args:
            logits: [batch, num_classes] raw logits
            
        Returns:
            Positive penalty for overconfident (low-entropy) predictions.
            Higher penalty when model is very confident (low entropy).
        """
        probs = F.softmax(logits, dim=1)
        # Entropy: H = -Σ p * log(p)
        # Add small epsilon for numerical stability
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        
        # Max entropy for 3 classes is log(3) ≈ 1.099
        # We want to PENALIZE low entropy (overconfidence)
        # Penalty = beta * (max_entropy - actual_entropy)
        # When predictions are uniform: entropy ≈ 1.099, penalty ≈ 0
        # When predictions are confident: entropy ≈ 0, penalty ≈ beta * 1.099
        num_classes = probs.size(1)
        max_entropy = torch.log(torch.tensor(num_classes, dtype=probs.dtype, device=probs.device))
        
        # Penalty increases when entropy is LOW (model is overconfident)
        confidence_penalty = self.beta * (max_entropy - entropy).mean()
        
        return confidence_penalty


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: Predicted probability for the true class
    - γ (gamma): Focusing parameter (typically 2.0)
      - Higher γ → more focus on hard examples
    - α: Per-class weights (optional, typically inverse of class frequency)
    
    Benefits for trading signal classification:
    - Down-weights easy/frequent HOLD predictions
    - Focuses learning on hard-to-classify LONG/SHORT signals
    - Better than simple class weighting for imbalanced data
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        num_classes: int = 3
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        # Register alpha as a buffer for proper device handling and state_dict persistence
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            # Register a placeholder buffer that can be updated later
            self.register_buffer('alpha', torch.ones(num_classes))
            self._alpha_initialized = False
    
    def set_alpha(self, alpha: torch.Tensor):
        """
        Update alpha weights (buffer-safe method).
        
        Call this after computing class priors from training data.
        Uses in-place copy to maintain device and buffer registration.
        """
        if self.alpha is None:
            self.register_buffer('alpha', alpha.clone())
        else:
            self.alpha.copy_(alpha.to(self.alpha.device))
        self._alpha_initialized = True
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: [batch, num_classes] raw logits
            targets: [batch] class indices (0, 1, 2 for SHORT/HOLD/LONG)
            
        Returns:
            Scalar focal loss
        """
        # Get probabilities via softmax
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of true class: p_t
        # targets: [batch] -> gather from probs: [batch, num_classes]
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [batch]
        
        # Compute cross entropy (without reduction): -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [batch]
        
        # Compute focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma  # [batch]
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss  # [batch]
        
        # Apply alpha (per-class weighting) if initialized
        # Note: alpha is always registered as buffer, but only apply if actually set
        if hasattr(self, '_alpha_initialized') and self._alpha_initialized:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets)  # [batch]
            focal_loss = alpha_t * focal_loss
        elif self.alpha is not None and not hasattr(self, '_alpha_initialized'):
            # Legacy path: alpha was passed to constructor
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets)  # [batch]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Gaussian distribution.
    
    PHASE 1b UPGRADE: Proper Gaussian NLL that couples σ to prediction error.
    
    Encourages model to predict both mean (μ) and log(σ) (unbounded).
    The model predicts log_sigma directly, which:
    1. Prevents σ from being gamed (inflated to reduce penalties)
    2. Couples σ to actual prediction error
    3. Produces calibrated uncertainty estimates
    
    NLL = 0.5 * (2*log_sigma + (y - μ)² / exp(2*log_sigma))
        = 0.5 * (2*log_sigma + (y - μ)² * exp(-2*log_sigma))
    
    This is equivalent to: log_sigma + 0.5 * (y - μ)² / σ²
    """
    
    def __init__(self, eps: float = 1e-6, use_log_sigma: bool = False):
        """
        Args:
            eps: Small epsilon for numerical stability
            use_log_sigma: If True, expects log(sigma) as input (preferred).
                          If False (default), expects sigma directly (legacy mode).
                          Default is False for backward compatibility with existing models.
        """
        super().__init__()
        self.eps = eps
        self.use_log_sigma = use_log_sigma
        
    def forward(self, mu: torch.Tensor, sigma_or_log_sigma: torch.Tensor, 
                targets: torch.Tensor, is_log_sigma: bool = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            mu: [batch, 1] predicted mean
            sigma_or_log_sigma: [batch, 1] predicted σ or log(σ) depending on is_log_sigma
            targets: [batch, 1] or [batch] actual values
            is_log_sigma: If True, input is log(σ). If None, uses self.use_log_sigma
            
        Returns:
            Scalar loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        use_log = is_log_sigma if is_log_sigma is not None else self.use_log_sigma
        
        if use_log:
            # Input is log_sigma - should already be clamped by RegressionHead
            log_sigma = sigma_or_log_sigma
            # STABILITY FIX: Tighter clamp consistent with RegressionHead [-8, 2]
            # exp(-8) ≈ 0.00034, exp(2) ≈ 7.4
            log_sigma = log_sigma.clamp(min=-8, max=2)
            
            # NLL = log_sigma + 0.5 * (y - μ)² / σ²
            #     = log_sigma + 0.5 * (y - μ)² * exp(-2 * log_sigma)
            squared_error = (targets - mu) ** 2
            nll = log_sigma + 0.5 * squared_error * torch.exp(-2 * log_sigma)
        else:
            # Legacy: Input is sigma (must be positive)
            sigma = sigma_or_log_sigma.clamp(min=self.eps)
            variance = sigma ** 2
            
            # NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
            nll = 0.5 * (torch.log(variance) + (targets - mu) ** 2 / variance)
        
        return nll.mean()


class MultiHeadLoss(nn.Module):
    """
    Combined loss for multi-head trading model.
    
    Total Loss = λ_class * L_class + λ_mu * L_mu + λ_sigma * L_sigma + 
                 λ_quantile * L_quantile + λ_trading * L_trading + λ_candle * L_candle
                 + λ_conf_penalty * L_confidence_penalty
    
    Where:
    - L_class: OHEM-wrapped Focal Loss (default) or CrossEntropy for direction classification
    - L_mu: Huber loss for expected return
    - L_sigma: Gaussian NLL for uncertainty calibration
    - L_quantile: Pinball loss for quantile regression
    - L_trading: Huber loss for entry_offset, sl_distance, tp_distance
    - L_candle: Huber loss for future candle deltas
    - L_confidence_penalty: Entropy bonus to prevent overconfidence
    
    PHASE 2 UPGRADES (2024-2025 research):
    - Focal Loss replaces CrossEntropy (handles class imbalance better)
    - OHEM wraps classification loss (focuses on hard examples, +15-25% recall)
    - Confidence Penalty prevents overconfident predictions
    - Label smoothing DISABLED (harms imbalanced classification)
    - Classification head weight increased 3x (priority over regression)
    - Temperature scaling at inference (T=0.7 sharpens predictions)
    """
    
    def __init__(self, config: Optional[MultiHeadLossConfig] = None):
        super().__init__()
        
        self.config = config or MultiHeadLossConfig()
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Classification loss - UPGRADED: Focal Loss for class imbalance
        base_class_loss = None
        if self.config.use_focal_loss:
            # Focal Loss: down-weights easy/frequent HOLD, focuses on LONG/SHORT
            base_class_loss = FocalLoss(
                alpha=self.config.focal_alpha,  # Per-class weights (set from class priors)
                gamma=self.config.focal_gamma,  # Focusing parameter (default 2.0)
                reduction='mean'
            )
            logger.info(
                f"[LOSS] Using FOCAL LOSS: gamma={self.config.focal_gamma}, "
                f"alpha={'computed from priors' if self.config.focal_alpha is None else 'custom'}"
            )
        else:
            # Fallback: CrossEntropyLoss (with label_smoothing=0.0 by default)
            base_class_loss = nn.CrossEntropyLoss(
                weight=self.config.class_weights,
                label_smoothing=self.config.label_smoothing
            )
        
        # OHEM WRAPPER - 2024 State-of-the-Art for imbalanced classification
        if self.config.use_ohem:
            self.class_loss = OHEMLoss(
                base_loss=base_class_loss,
                keep_ratio=self.config.ohem_keep_ratio,
                min_keep=self.config.ohem_min_keep
            )
            logger.info(
                f"[LOSS] OHEM ENABLED: keeping top {self.config.ohem_keep_ratio*100:.0f}% "
                f"hardest examples per batch (min_keep={self.config.ohem_min_keep})"
            )
        else:
            self.class_loss = base_class_loss
        
        # CONFIDENCE PENALTY - Entropy maximization to prevent overconfidence
        self.confidence_penalty = None
        if self.config.use_confidence_penalty:
            self.confidence_penalty = ConfidencePenaltyLoss(
                beta=self.config.confidence_penalty_beta
            )
            logger.info(
                f"[LOSS] CONFIDENCE PENALTY ENABLED: beta={self.config.confidence_penalty_beta}"
            )
        
        # Regression loss (Huber for robustness)
        self.mu_loss = nn.HuberLoss(delta=self.config.mu_huber_delta)
        
        # Uncertainty loss - Phase 1b: use config for log_sigma mode
        self.sigma_loss = GaussianNLLLoss(use_log_sigma=self.config.use_log_sigma)
        
        # Quantile loss
        self.quantile_loss = PinballLoss(self.config.quantiles)
        
        # Trading loss (entry/SL/TP)
        self.trading_loss = nn.HuberLoss(delta=self.config.trading_huber_delta)
        
        # Candle prediction loss
        self.candle_loss = nn.HuberLoss(delta=self.config.candle_huber_delta)
        
        # Flow Forecast losses - ALSO removed label smoothing for vol_state
        self.vol_state_loss = nn.CrossEntropyLoss(label_smoothing=0.0)  # 3-class: contraction/neutral/expansion
        self.acceleration_loss = nn.HuberLoss(delta=self.config.acceleration_huber_delta)
        
    def forward(
        self,
        class_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        quantiles: torch.Tensor,
        class_targets: torch.Tensor,
        return_targets: torch.Tensor,
        entry_offset: Optional[torch.Tensor] = None,
        sl_distance: Optional[torch.Tensor] = None,
        tp_distance: Optional[torch.Tensor] = None,
        candle_deltas: Optional[torch.Tensor] = None,
        trading_targets: Optional[Dict[str, torch.Tensor]] = None,
        candle_targets: Optional[torch.Tensor] = None,
        vol_state_logits: Optional[torch.Tensor] = None,
        vol_state_targets: Optional[torch.Tensor] = None,
        acceleration_pred: Optional[torch.Tensor] = None,
        acceleration_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            class_logits: [batch, 3] classification logits
            mu: [batch, 1] predicted expected return
            sigma: [batch, 1] predicted uncertainty
            quantiles: [batch, 5] predicted quantiles
            class_targets: [batch] class labels (0=SHORT, 1=HOLD, 2=LONG)
            return_targets: [batch] or [batch, 1] actual forward returns
            entry_offset: [batch, 1] predicted entry offset (optional)
            sl_distance: [batch, 1] predicted SL distance (optional)
            tp_distance: [batch, 1] predicted TP distance (optional)
            candle_deltas: [batch, n_steps, 3] predicted candle deltas (optional)
            trading_targets: Dict with 'entry_offset', 'sl_distance', 'tp_distance' targets
            candle_targets: [batch, n_steps, 3] actual candle deltas
            vol_state_logits: [batch, 3] flow forecast volatility state logits (optional)
            vol_state_targets: [batch] volatility state labels 0=contraction, 1=neutral, 2=expansion
            acceleration_pred: [batch, 1] predicted acceleration (momentum change)
            acceleration_targets: [batch, 1] actual acceleration targets
            
        Returns:
            Dict with 'total' loss and individual components
        """
        # Classification loss
        l_class = self.class_loss(class_logits, class_targets)
        
        # Regression loss (expected return) - SKIP if lambda_mu is 0
        # μ regression was identified as the source of gradient explosions
        if return_targets.dim() == 1:
            return_targets = return_targets.unsqueeze(-1)
        if self.config.lambda_mu > 0:
            l_mu = self.mu_loss(mu, return_targets)
        else:
            # μ DISABLED - return zero tensor with no gradient
            l_mu = torch.tensor(0.0, device=class_logits.device)
        
        # Uncertainty calibration loss - SKIP if lambda_sigma is 0
        # Sigma regression (and shared layers) were causing gradient explosions
        if self.config.lambda_sigma > 0:
            l_sigma = self.sigma_loss(mu, sigma, return_targets)
        else:
            # σ DISABLED - return zero tensor with no gradient
            l_sigma = torch.tensor(0.0, device=class_logits.device)
        
        # Quantile loss - SKIP if lambda_quantile is 0
        if self.config.lambda_quantile > 0:
            l_quantile = self.quantile_loss(quantiles, return_targets)
        else:
            l_quantile = torch.tensor(0.0, device=class_logits.device)
        
        # Trading loss (if targets provided)
        l_trading = torch.tensor(0.0, device=class_logits.device)
        if trading_targets is not None and entry_offset is not None:
            l_entry = self.trading_loss(entry_offset, trading_targets['entry_offset'])
            l_sl = self.trading_loss(sl_distance, trading_targets['sl_distance'])
            l_tp = self.trading_loss(tp_distance, trading_targets['tp_distance'])
            l_trading = (l_entry + l_sl + l_tp) / 3.0
        
        # Candle prediction loss (if targets provided)
        l_candle = torch.tensor(0.0, device=class_logits.device)
        if candle_targets is not None and candle_deltas is not None:
            l_candle = self.candle_loss(candle_deltas, candle_targets)
        
        # Flow Forecast: Volatility state classification loss - SKIP if lambda_vol_state is 0
        l_vol_state = torch.tensor(0.0, device=class_logits.device)
        if self.config.lambda_vol_state > 0 and vol_state_logits is not None and vol_state_targets is not None:
            l_vol_state = self.vol_state_loss(vol_state_logits, vol_state_targets)
        
        # Flow Forecast: Acceleration (momentum change) regression loss
        l_acceleration = torch.tensor(0.0, device=class_logits.device)
        if acceleration_pred is not None and acceleration_targets is not None:
            if acceleration_targets.dim() == 1:
                acceleration_targets = acceleration_targets.unsqueeze(-1)
            l_acceleration = self.acceleration_loss(acceleration_pred, acceleration_targets)
        
        # CONFIDENCE PENALTY: Entropy maximization to prevent overconfidence
        # This returns negative entropy, so adding it to total loss = subtracting entropy = adding entropy bonus
        l_confidence_penalty = torch.tensor(0.0, device=class_logits.device)
        if self.confidence_penalty is not None:
            l_confidence_penalty = self.confidence_penalty(class_logits)
        
        # Combined loss
        total = (
            self.config.lambda_class * l_class +
            self.config.lambda_mu * l_mu +
            self.config.lambda_sigma * l_sigma +
            self.config.lambda_quantile * l_quantile +
            self.config.lambda_trading * l_trading +
            self.config.lambda_candle * l_candle +
            self.config.lambda_vol_state * l_vol_state +
            self.config.lambda_acceleration * l_acceleration +
            l_confidence_penalty  # Already weighted by beta in ConfidencePenaltyLoss
        )
        
        return {
            'total': total,
            'class': l_class,
            'mu': l_mu,
            'sigma': l_sigma,
            'quantile': l_quantile,
            'trading': l_trading,
            'candle': l_candle,
            'vol_state': l_vol_state,
            'acceleration': l_acceleration,
            'confidence_penalty': l_confidence_penalty
        }


class QuantileCalibrationLoss(nn.Module):
    """
    Additional loss term for quantile calibration.
    
    Ensures that the coverage of each quantile matches its target probability.
    E.g., q10 should be below actual 10% of the time.
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer('q_tensor', torch.tensor(quantiles))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute calibration loss.
        
        Penalizes deviation from expected coverage.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        targets = targets.expand_as(predictions)
        q = self.q_tensor.to(predictions.device)
        
        # Compute actual coverage (how often target < prediction)
        coverage = (targets < predictions).float().mean(dim=0)  # [n_quantiles]
        
        # Should match target quantiles
        calibration_error = (coverage - q) ** 2
        
        return calibration_error.mean()


def create_multihead_loss(
    class_weights: Optional[torch.Tensor] = None,
    lambda_class: float = 1.0,
    lambda_mu: float = 0.5,
    lambda_quantile: float = 0.5
) -> MultiHeadLoss:
    """
    Factory function to create multi-head loss with custom weights.
    """
    config = MultiHeadLossConfig(
        lambda_class=lambda_class,
        lambda_mu=lambda_mu,
        lambda_quantile=lambda_quantile,
        class_weights=class_weights
    )
    return MultiHeadLoss(config)
