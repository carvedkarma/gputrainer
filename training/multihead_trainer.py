"""
Multi-Head Trainer for Institutional Trading Models

Trains models with:
1. Classification head (direction)
2. Regression head (expected return μ, uncertainty σ)
3. Quantile head (q10, q25, q50, q75, q90)

Uses combined loss with configurable weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, SequentialSampler, RandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
import math
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
import logging
from torch.utils.tensorboard import SummaryWriter

try:
    from .multihead_loss import MultiHeadLoss, MultiHeadLossConfig
    from ..models.multihead import MultiHeadOutput
    from ..utils.trade_gating import (
        TradeGateConfig, compute_trade_gate, apply_cooldown, 
        log_gate_statistics, GateFailure, DEFAULT_GATE_CONFIG
    )
    from .walk_forward import save_walk_forward_weights
except ImportError:
    # Fallback for direct script execution
    from training.multihead_loss import MultiHeadLoss, MultiHeadLossConfig
    from models.multihead import MultiHeadOutput
    from utils.trade_gating import (
        TradeGateConfig, compute_trade_gate, apply_cooldown,
        log_gate_statistics, GateFailure, DEFAULT_GATE_CONFIG
    )
    from training.walk_forward import save_walk_forward_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadDataset(Dataset):
    """
    Dataset for multi-head training.
    
    Each sample contains:
    - features: [seq_len, input_dim] input sequence
    - class_label: int (0=SHORT, 1=HOLD, 2=LONG)
    - forward_return: float (actual return for regression/quantile targets)
    - trading_targets: [3] tensor (entry_offset, sl_distance, tp_distance)
    - candle_targets: [n_future, 3] tensor (close, high, low deltas)
    - regime_id: int (0=BULL, 1=BEAR, 2=HIGH_VOL, 3=LOW_VOL_CHOP) - for training balance only
    - vol_state: int (0=contraction, 1=neutral, 2=expansion) - Flow Forecast target
    - acceleration: float (momentum change) - Flow Forecast target
    """
    
    def __init__(
        self,
        features: np.ndarray,
        class_labels: np.ndarray,
        forward_returns: np.ndarray,
        entry_offset: Optional[np.ndarray] = None,
        sl_distance: Optional[np.ndarray] = None,
        tp_distance: Optional[np.ndarray] = None,
        candle_targets: Optional[np.ndarray] = None,
        regime_ids: Optional[np.ndarray] = None,
        vol_state: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        n_future_candles: int = 5,
        sequence_length: int = 100
    ):
        self.features = features.astype(np.float32)
        self.class_labels = class_labels.astype(np.int64)
        self.forward_returns = forward_returns.astype(np.float32)
        self.sequence_length = sequence_length
        self.n_future_candles = n_future_candles
        
        # Trading targets (entry/SL/TP) - optional for backward compatibility
        # Handle both numpy arrays and scalars
        def to_float_array(val):
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(np.float32)
            # Scalar - convert to array of same length as features
            return np.full(len(features), float(val), dtype=np.float32)
        
        self.entry_offset = to_float_array(entry_offset)
        self.sl_distance = to_float_array(sl_distance)
        self.tp_distance = to_float_array(tp_distance)
        
        # Candle prediction targets - optional for backward compatibility
        self.candle_targets = candle_targets.astype(np.float32) if candle_targets is not None else None
        
        # Regime IDs for balanced training (0=BULL, 1=BEAR, 2=HIGH_VOL, 3=LOW_VOL_CHOP)
        self.regime_ids = regime_ids.astype(np.int64) if regime_ids is not None else None
        
        # Flow Forecast targets - optional for backward compatibility
        self.vol_state = vol_state.astype(np.int64) if vol_state is not None else None
        self.acceleration = to_float_array(acceleration)
        
        # Create sequences
        self.valid_indices = list(range(sequence_length, len(features)))
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        actual_idx = self.valid_indices[idx]
        
        # Get sequence ending at actual_idx
        start_idx = actual_idx - self.sequence_length
        seq = self.features[start_idx:actual_idx]
        
        # Trading targets: [entry_offset, sl_distance, tp_distance]
        # Check all three are available before accessing
        if self.entry_offset is not None and self.sl_distance is not None and self.tp_distance is not None:
            trading = np.array([
                self.entry_offset[actual_idx],
                self.sl_distance[actual_idx],
                self.tp_distance[actual_idx]
            ], dtype=np.float32)
        else:
            trading = np.zeros(3, dtype=np.float32)
        
        # Candle targets: reshape to [n_future, 3]
        if self.candle_targets is not None:
            c = self.candle_targets[actual_idx]
            # Columns are: close_1, high_1, low_1, close_2, high_2, low_2, ...
            # Reshape to [n_future, 3] where 3 = (close, high, low)
            c = c.reshape(self.n_future_candles, 3)
        else:
            c = np.zeros((self.n_future_candles, 3), dtype=np.float32)
        
        # Flow Forecast targets
        vol_state = self.vol_state[actual_idx] if self.vol_state is not None else 1  # default neutral
        accel = self.acceleration[actual_idx] if self.acceleration is not None else 0.0
        
        return (
            torch.from_numpy(seq),
            torch.tensor(self.class_labels[actual_idx]),
            torch.tensor(self.forward_returns[actual_idx]),
            torch.from_numpy(trading),
            torch.from_numpy(c),
            torch.tensor(vol_state, dtype=torch.long),
            torch.tensor(accel, dtype=torch.float32)
        )
    
    def get_regime_ids_for_valid_indices(self) -> np.ndarray:
        """Get regime IDs only for valid indices (for creating balanced sampler)."""
        if self.regime_ids is None:
            # Default to LOW_VOL_CHOP (3) if no regime labels
            return np.full(len(self.valid_indices), 3, dtype=np.int64)
        return self.regime_ids[self.valid_indices]
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get regime distribution for logging."""
        regime_ids = self.get_regime_ids_for_valid_indices()
        total = len(regime_ids)
        
        REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "HIGH_VOL", 3: "LOW_VOL_CHOP"}
        distribution = {}
        
        for regime_id, name in REGIME_NAMES.items():
            count = np.sum(regime_ids == regime_id)
            distribution[name] = count / total if total > 0 else 0.0
        
        return distribution


def create_regime_balanced_loader(
    dataset: 'MultiHeadDataset',
    batch_size: int = 64,
    num_workers: int = 4,
    target_balance: float = 0.25
) -> DataLoader:
    """
    Create a DataLoader with regime-balanced sampling.
    
    Uses WeightedRandomSampler to ensure ~25% of each regime in training batches.
    This prevents the model from overfitting to dominant market regimes.
    
    Args:
        dataset: MultiHeadDataset with regime_ids
        batch_size: Batch size
        num_workers: Number of data loader workers
        target_balance: Target proportion per regime (default 0.25 for 4 regimes)
        
    Returns:
        DataLoader with balanced sampling
    """
    regime_ids = dataset.get_regime_ids_for_valid_indices()
    total = len(regime_ids)
    
    # Calculate inverse frequency weights
    weights = np.ones(total, dtype=np.float64)
    
    for regime_id in range(4):  # 4 regimes
        count = np.sum(regime_ids == regime_id)
        if count > 0:
            actual_proportion = count / total
            regime_weight = target_balance / actual_proportion
            regime_weight = np.clip(regime_weight, 0.25, 4.0)
            mask = regime_ids == regime_id
            weights[mask] = regime_weight
    
    # Normalize weights
    weights = weights * (total / weights.sum())
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=total,
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


class TrainingHealthMonitor:
    """
    Real-time Training Health Monitor for detecting training issues.
    
    Monitors for:
    1. Loss divergence (loss increasing over rolling window)
    2. Accuracy collapse (accuracy dropping significantly)
    3. Class distribution skew (model predicting only one class)
    4. Gradient explosion (large gradient norms)
    5. NaN/Inf values in loss or gradients
    
    Sends alerts to GUI via callback when issues detected.
    
    2024 Research: Early detection of training issues saves compute and prevents bad models.
    """
    
    def __init__(
        self, 
        window_size: int = 10,
        loss_divergence_threshold: float = 0.5,
        accuracy_drop_threshold: float = 0.15,
        class_skew_threshold: float = 0.85,
        gradient_explosion_threshold: float = 10.0,
        alert_callback: Optional[Callable[[str, str, Dict], None]] = None
    ):
        """
        Args:
            window_size: Rolling window for trend detection
            loss_divergence_threshold: Alert if loss increases by this fraction
            accuracy_drop_threshold: Alert if accuracy drops by this absolute amount
            class_skew_threshold: Alert if one class > this fraction of predictions
            gradient_explosion_threshold: Alert if gradient norm exceeds this
            alert_callback: Function(alert_type, message, details) to call on alerts
        """
        self.window_size = window_size
        self.loss_divergence_threshold = loss_divergence_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.class_skew_threshold = class_skew_threshold
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.alert_callback = alert_callback
        
        # History tracking
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.class_distribution_history: List[Dict[int, float]] = []  # Percentages
        self.class_counts_history: List[Dict[int, int]] = []  # Actual counts
        self.gradient_norm_history: List[float] = []
        
        # Alert state (prevent spam)
        self.alerts_sent: Dict[str, int] = {}
        self.alert_cooldown = 5  # epochs between same alert type
        
        # Best values for comparison
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        class_predictions: Optional[np.ndarray] = None,
        gradient_norm: Optional[float] = None
    ) -> List[Dict]:
        """
        Update monitor with epoch metrics and check for issues.
        
        Returns list of alert dictionaries if issues detected.
        """
        alerts = []
        
        # Check for NaN/Inf
        if np.isnan(train_loss) or np.isinf(train_loss):
            alerts.append(self._create_alert(
                epoch, "CRITICAL", "nan_loss",
                "Training loss is NaN/Inf! Training has diverged.",
                {"loss": train_loss}
            ))
        
        # Update histories
        self.loss_history.append(train_loss)
        self.accuracy_history.append(train_accuracy)
        
        # Track best values
        if train_loss < self.best_loss:
            self.best_loss = train_loss
        if train_accuracy > self.best_accuracy:
            self.best_accuracy = train_accuracy
        
        # Check loss divergence (loss increasing trend)
        if len(self.loss_history) >= self.window_size:
            recent_losses = self.loss_history[-self.window_size:]
            early_avg = np.mean(recent_losses[:self.window_size//2])
            late_avg = np.mean(recent_losses[self.window_size//2:])
            
            if early_avg > 0 and (late_avg - early_avg) / early_avg > self.loss_divergence_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "loss_divergence",
                    f"Loss increasing: {early_avg:.4f} → {late_avg:.4f} (+{((late_avg-early_avg)/early_avg)*100:.1f}%)",
                    {"early_avg": early_avg, "late_avg": late_avg}
                ))
        
        # Check accuracy collapse
        if len(self.accuracy_history) >= self.window_size:
            peak_accuracy = max(self.accuracy_history[:-self.window_size//2]) if len(self.accuracy_history) > self.window_size else self.best_accuracy
            recent_accuracy = np.mean(self.accuracy_history[-self.window_size//2:])
            
            if peak_accuracy - recent_accuracy > self.accuracy_drop_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "accuracy_drop",
                    f"Accuracy dropped: {peak_accuracy*100:.1f}% → {recent_accuracy*100:.1f}%",
                    {"peak": peak_accuracy, "current": recent_accuracy}
                ))
        
        # Check class distribution skew
        if class_predictions is not None:
            unique, counts = np.unique(class_predictions, return_counts=True)
            total = len(class_predictions)
            distribution = {int(u): c/total for u, c in zip(unique, counts)}
            count_dict = {int(u): int(c) for u, c in zip(unique, counts)}
            count_dict['total'] = int(total)  # Add total for GUI display
            self.class_distribution_history.append(distribution)
            self.class_counts_history.append(count_dict)
            
            max_class_ratio = max(distribution.values()) if distribution else 0
            if max_class_ratio > self.class_skew_threshold:
                majority_class = max(distribution, key=distribution.get)
                class_names = {0: "SHORT", 1: "HOLD", 2: "LONG"}
                alerts.append(self._create_alert(
                    epoch, "WARNING", "class_skew",
                    f"Model predicting mostly {class_names.get(majority_class, majority_class)}: {max_class_ratio*100:.1f}%",
                    {"distribution": distribution}
                ))
        
        # Check gradient explosion
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
            if gradient_norm > self.gradient_explosion_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "gradient_explosion",
                    f"Large gradient norm: {gradient_norm:.2f} (threshold: {self.gradient_explosion_threshold})",
                    {"gradient_norm": gradient_norm}
                ))
        
        # Send alerts via callback
        for alert in alerts:
            if self.alert_callback and self._should_send_alert(epoch, alert['type']):
                self.alert_callback(alert['severity'], alert['message'], alert)
        
        return alerts
    
    def _create_alert(self, epoch: int, severity: str, alert_type: str, message: str, details: Dict) -> Dict:
        """Create alert dictionary."""
        return {
            "epoch": epoch,
            "severity": severity,  # CRITICAL, WARNING, INFO
            "type": alert_type,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _should_send_alert(self, epoch: int, alert_type: str) -> bool:
        """Check if we should send this alert (cooldown logic)."""
        last_sent = self.alerts_sent.get(alert_type, -999)
        if epoch - last_sent >= self.alert_cooldown:
            self.alerts_sent[alert_type] = epoch
            return True
        return False
    
    def get_health_summary(self) -> Dict:
        """Get overall health summary."""
        issues = []
        
        if len(self.loss_history) >= 3:
            recent_loss = np.mean(self.loss_history[-3:])
            if recent_loss > self.best_loss * 1.5:
                issues.append("loss_elevated")
        
        if len(self.accuracy_history) >= 3:
            recent_acc = np.mean(self.accuracy_history[-3:])
            if recent_acc < self.best_accuracy * 0.8:
                issues.append("accuracy_degraded")
        
        if len(self.class_distribution_history) >= 1:
            recent_dist = self.class_distribution_history[-1]
            if max(recent_dist.values()) > self.class_skew_threshold:
                issues.append("class_imbalanced")
        
        return {
            "status": "HEALTHY" if not issues else "ISSUES_DETECTED",
            "issues": issues,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "current_accuracy": self.accuracy_history[-1] if self.accuracy_history else None
        }


class MultiHeadTrainer:
    """
    Trainer for multi-head models.
    
    Handles:
    - Combined loss optimization
    - Class weighting for imbalanced data
    - Per-head metric tracking
    - Walk-forward validation
    - Training health monitoring with real-time alerts
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        loss_config: Optional[MultiHeadLossConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        gui_mode: bool = False,
        feature_scaler = None,
        feature_columns: Optional[List[str]] = None,
        training_mode: str = "stf",
        horizon_periods: int = 16,
        health_alert_callback: Optional[Callable[[str, str, Dict], None]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.gui_mode = gui_mode
        
        # Store training config for checkpoint saving
        self.feature_scaler = feature_scaler  # sklearn StandardScaler, not AMP GradScaler
        self.feature_columns = feature_columns
        self.training_mode = training_mode
        self.horizon_periods = horizon_periods
        
        # Setup loss
        if loss_config is None:
            loss_config = MultiHeadLossConfig(class_weights=class_weights)
        self.criterion = MultiHeadLoss(loss_config).to(device)
        
        # TRAINING HEALTH MONITOR - 2024 Best Practice
        # Raised gradient_explosion_threshold from 10.0 to 20.0 - with gradient clipping
        # at 1.0, pre-clip norms up to 15-20 are normal and clipping handles them
        self.health_monitor = TrainingHealthMonitor(
            window_size=10,
            loss_divergence_threshold=0.5,
            accuracy_drop_threshold=0.15,
            class_skew_threshold=0.85,
            gradient_explosion_threshold=20.0,  # Was 10.0, raised to reduce false alarms
            alert_callback=health_alert_callback
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler: Epoch-level warmup + cosine annealing
        base_lr = config.training.learning_rate
        if base_lr > 5e-5:
            base_lr = 5e-5
            logger.info(f"[SCHEDULER] Clamped base_lr to 5e-5 for stability")
        
        self.base_lr = base_lr
        
        warmup_epochs = getattr(config.training, 'warmup_epochs', 5)
        total_epochs = config.training.epochs
        config_min_lr = getattr(config.training, 'min_lr', 0.0)
        eta_min = config_min_lr if config_min_lr > 0 else base_lr * 0.05
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        # Set initial LR to warmup start value so epoch 0 trains at low LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = base_lr * 1e-3
        
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        
        # Gradient explosion tracking for auto LR reduction
        self.high_grad_norm_count = 0
        self.grad_norm_threshold = 20.0
        self.lr_reduction_factor = 0.5
        
        # === STABILITY PROOF LOGS ===
        logger.info("=" * 70)
        logger.info("[STABILITY PROOF] TRAINING CONFIGURATION")
        logger.info("=" * 70)
        
        logger.info("[STABILITY PROOF] Scheduler:")
        logger.info("  Type: Epoch-level LinearLR warmup + CosineAnnealingLR")
        logger.info("  base_lr: %.2e", self.base_lr)
        logger.info("  eta_min: %.2e (lr * 0.05)", self.eta_min)
        logger.info("  warmup_epochs: %d", self.warmup_epochs)
        logger.info("  total_epochs: %d", self.total_epochs)
        logger.info("  Strategy: Linear warmup %d epochs -> Cosine decay to eta_min", self.warmup_epochs)
        
        # Loss config - get from criterion (MultiHeadLoss has a config attribute)
        # Safe access with fallbacks in case config structure varies
        loss_cfg = getattr(self.criterion, 'config', None)
        if loss_cfg is None:
            # Fallback: try to get from config.loss if it exists
            loss_cfg = getattr(config, 'loss', None)
        
        if loss_cfg is not None:
            focal_enabled = getattr(loss_cfg, 'use_focal_loss', False)
            ohem_enabled = getattr(loss_cfg, 'use_ohem', False)
            conf_penalty_enabled = getattr(loss_cfg, 'use_confidence_penalty', False)
            lambda_class = getattr(loss_cfg, 'lambda_class', 1.0)
            lambda_mu = getattr(loss_cfg, 'lambda_mu', 0.2)
            lambda_sigma = getattr(loss_cfg, 'lambda_sigma', 0.1)
            lambda_quantile = getattr(loss_cfg, 'lambda_quantile', 0.2)
            lambda_trading = getattr(loss_cfg, 'lambda_trading', 0.1)
            lambda_candle = getattr(loss_cfg, 'lambda_candle', 0.1)
            lambda_vol_state = getattr(loss_cfg, 'lambda_vol_state', 0.2)
            lambda_acceleration = getattr(loss_cfg, 'lambda_acceleration', 0.1)
        else:
            # Default values if config not found
            focal_enabled = False
            ohem_enabled = False
            conf_penalty_enabled = False
            lambda_class = 1.0
            lambda_mu = 0.2
            lambda_sigma = 0.1
            lambda_quantile = 0.2
            lambda_trading = 0.1
            lambda_candle = 0.1
            lambda_vol_state = 0.2
            lambda_acceleration = 0.1
            logger.warning("[STABILITY PROOF] Could not find loss config, using defaults")
        
        prior_bias_enabled = False  # Explicitly disabled in train()
        
        logger.info("[STABILITY PROOF] Classification Tricks:")
        logger.info("  use_focal_loss: %s", "ENABLED" if focal_enabled else "DISABLED")
        logger.info("  use_ohem: %s", "ENABLED" if ohem_enabled else "DISABLED")
        logger.info("  use_confidence_penalty: %s", "ENABLED" if conf_penalty_enabled else "DISABLED")
        logger.info("  prior_bias_init: %s", "ENABLED" if prior_bias_enabled else "DISABLED")
        
        # Loss weights - show which heads are ENABLED vs DISABLED
        logger.info("[STABILITY PROOF] Loss Weights (0.0 = DISABLED):")
        logger.info("  lambda_class: %.2f %s", lambda_class, "✓ ENABLED" if lambda_class > 0 else "✗ DISABLED")
        logger.info("  lambda_mu: %.2f %s", lambda_mu, "✓ ENABLED" if lambda_mu > 0 else "✗ DISABLED")
        logger.info("  lambda_sigma: %.2f %s", lambda_sigma, "✓ ENABLED" if lambda_sigma > 0 else "✗ DISABLED")
        logger.info("  lambda_quantile: %.2f %s", lambda_quantile, "✓ ENABLED" if lambda_quantile > 0 else "✗ DISABLED")
        logger.info("  lambda_trading: %.2f %s", lambda_trading, "✓ ENABLED" if lambda_trading > 0 else "✗ DISABLED")
        logger.info("  lambda_candle: %.2f %s", lambda_candle, "✓ ENABLED" if lambda_candle > 0 else "✗ DISABLED")
        logger.info("  lambda_vol_state: %.2f %s", lambda_vol_state, "✓ ENABLED" if lambda_vol_state > 0 else "✗ DISABLED")
        logger.info("  lambda_acceleration: %.2f %s", lambda_acceleration, "✓ ENABLED" if lambda_acceleration > 0 else "✗ DISABLED")
        
        # Count enabled heads
        enabled_count = sum(1 for l in [lambda_class, lambda_mu, lambda_sigma, lambda_quantile, 
                                         lambda_trading, lambda_candle, lambda_vol_state, lambda_acceleration] if l > 0)
        logger.info("  TOTAL ENABLED HEADS: %d/8", enabled_count)
        logger.info("=" * 70)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.epoch_callback: Optional[Callable] = None
        
        # Logging
        log_dir = Path(config.training.log_dir) / "multihead"
        self.writer = SummaryWriter(log_dir)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with multi-head outputs (all 8 heads)."""
        self.model.train()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0, 
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0,
            'vol_state': 0.0, 'acceleration': 0.0, 'confidence_penalty': 0.0
        }
        correct = 0
        total = 0
        
        # Track gradient norms and predictions for health monitoring
        gradient_norms = []
        gradient_norms_pre_clip = []
        gradient_norms_post_clip = []
        all_predictions = []
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
            if len(batch_data) == 7:
                features, class_labels, returns, trading, candle_tgt, vol_state_tgt, accel_tgt = batch_data
            else:
                # Legacy 5-item format
                features, class_labels, returns, trading, candle_tgt = batch_data
                vol_state_tgt = None
                accel_tgt = None
            
            features = features.to(self.device)
            class_labels = class_labels.to(self.device)
            returns = returns.to(self.device)
            trading = trading.to(self.device)  # [batch, 3]
            candle_tgt = candle_tgt.to(self.device)  # [batch, n_future, 3]
            
            # Flow Forecast targets
            if vol_state_tgt is not None:
                vol_state_tgt = vol_state_tgt.to(self.device)
            if accel_tgt is not None:
                accel_tgt = accel_tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Multi-head forward pass
            output = self.model.forward_multihead(features)
            
            # Build trading_targets dict
            trading_targets = {
                "entry_offset": trading[:, 0:1],
                "sl_distance": trading[:, 1:2],
                "tp_distance": trading[:, 2:3],
            }
            
            # Compute combined loss (all 8 heads)
            losses = self.criterion(
                class_logits=output.class_logits,
                mu=output.mu,
                sigma=output.sigma,
                quantiles=output.quantiles,
                class_targets=class_labels,
                return_targets=returns,
                entry_offset=output.entry_offset,
                sl_distance=output.sl_distance,
                tp_distance=output.tp_distance,
                candle_deltas=output.candle_deltas,
                trading_targets=trading_targets,
                candle_targets=candle_tgt,
                vol_state_logits=output.vol_state_logits,
                vol_state_targets=vol_state_tgt,
                acceleration_pred=output.acceleration,
                acceleration_targets=accel_tgt
            )
            
            loss = losses['total']
            loss.backward()
            
            # Track gradient norm BEFORE clipping (for health monitoring)
            total_norm_pre = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_pre += p.grad.data.norm(2).item() ** 2
            grad_norm_pre = total_norm_pre ** 0.5
            gradient_norms.append(grad_norm_pre)
            gradient_norms_pre_clip.append(grad_norm_pre)
            
            # === DIAGNOSTICS: Per-layer gradient norms on explosion ===
            if grad_norm_pre > 20.0:
                # Compute per-layer grad norms to identify the exploding layer
                layer_grad_norms = {}
                param_grad_list = []
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.data.norm(2).item()
                        param_grad_list.append((name, norm))
                        
                        # Categorize by layer type
                        if 'class' in name.lower() or 'classifier' in name.lower():
                            layer_grad_norms['classifier'] = layer_grad_norms.get('classifier', 0.0) + norm**2
                        elif 'regression' in name.lower() or 'mu_head' in name.lower() or 'sigma_head' in name.lower():
                            layer_grad_norms['regression'] = layer_grad_norms.get('regression', 0.0) + norm**2
                        else:
                            layer_grad_norms['trunk'] = layer_grad_norms.get('trunk', 0.0) + norm**2
                
                # Take sqrt for L2 norm
                for k in layer_grad_norms:
                    layer_grad_norms[k] = layer_grad_norms[k] ** 0.5
                
                # Top-5 parameters by gradient norm
                param_grad_list.sort(key=lambda x: x[1], reverse=True)
                top5 = param_grad_list[:5]
                
                logger.warning(f"[GRAD EXPLOSION] Epoch {epoch}, Batch {batch_idx}: grad_norm={grad_norm_pre:.2f}")
                logger.warning(f"  Per-layer norms: trunk={layer_grad_norms.get('trunk', 0.0):.2f}, "
                             f"classifier={layer_grad_norms.get('classifier', 0.0):.2f}, "
                             f"regression={layer_grad_norms.get('regression', 0.0):.2f}")
                logger.warning(f"  Top-5 params by grad norm:")
                for pname, pnorm in top5:
                    logger.warning(f"    {pname}: {pnorm:.4f}")
            
            # Gradient clipping - STABILITY FIX: reduced from 1.0 to 0.7
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            
            # Track gradient norm AFTER clipping
            total_norm_post = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_post += p.grad.data.norm(2).item() ** 2
            grad_norm_post = total_norm_post ** 0.5
            gradient_norms_post_clip.append(grad_norm_post)
            
            # === STABILITY PROOF: Epoch 0 First Batch Diagnostics ===
            if epoch == 0 and batch_idx == 0:
                logger.info("=" * 70)
                logger.info("[STABILITY PROOF] EPOCH 0 FIRST BATCH DIAGNOSTICS")
                logger.info("=" * 70)
                
                # Label bincount for this batch
                label_bincount = torch.bincount(class_labels, minlength=3)
                logger.info("  Label bincount: SHORT=%d, HOLD=%d, LONG=%d", 
                           label_bincount[0].item(), label_bincount[1].item(), label_bincount[2].item())
                
                # Mean logits per class
                mean_logits = output.class_logits.mean(dim=0)
                logger.info("  Mean logits: SHORT=%.4f, HOLD=%.4f, LONG=%.4f",
                           mean_logits[0].item(), mean_logits[1].item(), mean_logits[2].item())
                
                # Argmax prediction bincount
                preds_batch = output.class_logits.argmax(dim=-1)
                pred_bincount = torch.bincount(preds_batch, minlength=3)
                logger.info("  Pred bincount: SHORT=%d, HOLD=%d, LONG=%d",
                           pred_bincount[0].item(), pred_bincount[1].item(), pred_bincount[2].item())
                
                # Initial gradient norm
                logger.info("  First batch grad_norm (pre-clip): %.4f", grad_norm_pre)
                logger.info("  First batch grad_norm (post-clip): %.4f", grad_norm_post)
                logger.info("=" * 70)
            
            self.optimizer.step()
            
            # Track losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # Track accuracy and predictions for health monitoring
            preds = output.class_logits.argmax(dim=-1)
            correct += (preds == class_labels).sum().item()
            total += len(class_labels)
            all_predictions.extend(preds.cpu().numpy().tolist())
            
            self.global_step += 1
            
        # Average losses
        n_batches = len(self.train_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total
        
        # Compute average gradient norms for this epoch (pre and post clip)
        avg_grad_norm_pre = np.mean(gradient_norms_pre_clip) if gradient_norms_pre_clip else 0.0
        avg_grad_norm_post = np.mean(gradient_norms_post_clip) if gradient_norms_post_clip else 0.0
        max_grad_norm_pre = np.max(gradient_norms_pre_clip) if gradient_norms_pre_clip else 0.0
        
        # Use pre-clip for backward compatibility with health monitor
        avg_losses['gradient_norm'] = avg_grad_norm_pre
        avg_losses['gradient_norm_pre_clip'] = avg_grad_norm_pre
        avg_losses['gradient_norm_post_clip'] = avg_grad_norm_post
        avg_losses['gradient_norm_max'] = max_grad_norm_pre
        
        # Gradient summary (compact - only log details when concerning)
        if max_grad_norm_pre > 5.0:
            logger.info("[GRAD] Epoch %d: avg=%.2f, max=%.2f (clipped to %.2f)",
                       epoch, avg_grad_norm_pre, max_grad_norm_pre, avg_grad_norm_post)
        
        # Per-loss means - only log non-zero auxiliary losses
        aux_active = any(avg_losses.get(k, 0) > 0.0001 for k in ['mu', 'sigma', 'quantile', 'vol_state', 'acceleration'])
        if aux_active:
            logger.info("[LOSS] Epoch %d: class=%.4f, mu=%.4f, sigma=%.4f, quantile=%.4f, vol_state=%.4f",
                       epoch, avg_losses['class'], avg_losses['mu'], avg_losses['sigma'],
                       avg_losses['quantile'], avg_losses['vol_state'])
        
        # HEALTH MONITORING: Check for training issues
        alerts = self.health_monitor.update(
            epoch=epoch,
            train_loss=avg_losses['total'],
            train_accuracy=avg_losses['accuracy'],
            class_predictions=np.array(all_predictions) if all_predictions else None,
            gradient_norm=avg_grad_norm_pre
        )
        
        # Log any alerts
        for alert in alerts:
            if alert['severity'] == 'CRITICAL':
                logger.error(f"[HEALTH CRITICAL] {alert['message']}")
            else:
                logger.warning(f"[HEALTH WARNING] {alert['message']}")
        
        return avg_losses
    
    def validate(self, epoch: int = 0) -> Dict[str, float]:
        """Validate with all heads (8 heads).
        
        Args:
            epoch: Current training epoch (used for monitoring sweep interval)
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0,
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0,
            'vol_state': 0.0, 'acceleration': 0.0
        }
        correct = 0
        total = 0
        
        # Per-class tracking
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        pred_counts = {0: 0, 1: 0, 2: 0}
        
        # Quantile calibration tracking
        quantile_below = torch.zeros(5)  # How often target < predicted quantile
        quantile_count = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
                if len(batch_data) == 7:
                    features, class_labels, returns, trading, candle_tgt, vol_state_tgt, accel_tgt = batch_data
                else:
                    features, class_labels, returns, trading, candle_tgt = batch_data
                    vol_state_tgt = None
                    accel_tgt = None
                
                features = features.to(self.device)
                class_labels = class_labels.to(self.device)
                returns = returns.to(self.device)
                trading = trading.to(self.device)
                candle_tgt = candle_tgt.to(self.device)
                
                # Flow Forecast targets
                if vol_state_tgt is not None:
                    vol_state_tgt = vol_state_tgt.to(self.device)
                if accel_tgt is not None:
                    accel_tgt = accel_tgt.to(self.device)
                
                output = self.model.forward_multihead(features)
                
                # Build trading_targets dict
                trading_targets = {
                    "entry_offset": trading[:, 0:1],
                    "sl_distance": trading[:, 1:2],
                    "tp_distance": trading[:, 2:3],
                }
                
                losses = self.criterion(
                    class_logits=output.class_logits,
                    mu=output.mu,
                    sigma=output.sigma,
                    quantiles=output.quantiles,
                    class_targets=class_labels,
                    return_targets=returns,
                    entry_offset=output.entry_offset,
                    sl_distance=output.sl_distance,
                    tp_distance=output.tp_distance,
                    candle_deltas=output.candle_deltas,
                    trading_targets=trading_targets,
                    candle_targets=candle_tgt,
                    vol_state_logits=output.vol_state_logits,
                    vol_state_targets=vol_state_tgt,
                    acceleration_pred=output.acceleration,
                    acceleration_targets=accel_tgt
                )
                
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                
                # Classification accuracy
                preds = output.class_logits.argmax(dim=-1)
                correct += (preds == class_labels).sum().item()
                total += len(class_labels)
                
                # Per-class accuracy + prediction counts
                for c in [0, 1, 2]:
                    mask = class_labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()
                    pred_counts[c] += (preds == c).sum().item()
                
                # Quantile calibration
                returns_expanded = returns.unsqueeze(-1).expand_as(output.quantiles)
                quantile_below += (returns_expanded < output.quantiles).float().sum(dim=0).cpu()
                quantile_count += len(returns)
        
        n_batches = len(self.val_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total
        
        # Per-class accuracy
        for c, name in [(0, 'short'), (1, 'hold'), (2, 'long')]:
            if class_total[c] > 0:
                avg_losses[f'acc_{name}'] = class_correct[c] / class_total[c]
            else:
                avg_losses[f'acc_{name}'] = 0.0
        
        # Prediction distribution (what % the model predicts as each class)
        pred_total_count = sum(pred_counts.values())
        if pred_total_count > 0:
            avg_losses['pred_short_pct'] = pred_counts[0] / pred_total_count
            avg_losses['pred_hold_pct'] = pred_counts[1] / pred_total_count
            avg_losses['pred_long_pct'] = pred_counts[2] / pred_total_count
        else:
            avg_losses['pred_short_pct'] = 0
            avg_losses['pred_hold_pct'] = 0
            avg_losses['pred_long_pct'] = 0
        
        # Quantile calibration (should be ~[0.1, 0.25, 0.5, 0.75, 0.9])
        if quantile_count > 0:
            calibration = quantile_below / quantile_count
            avg_losses['q10_cal'] = calibration[0].item()
            avg_losses['q25_cal'] = calibration[1].item()
            avg_losses['q50_cal'] = calibration[2].item()
            avg_losses['q75_cal'] = calibration[3].item()
            avg_losses['q90_cal'] = calibration[4].item()
        
        # Compute trading-aware metrics (monitoring only, runs every N epochs)
        trading_metrics = self._compute_trading_metrics(epoch=epoch)
        avg_losses.update(trading_metrics)
        
        # Compute per-regime metrics (if regime_ids available)
        regime_metrics = self._compute_regime_metrics()
        if regime_metrics:
            avg_losses['regime_metrics'] = regime_metrics
        
        return avg_losses
    
    def _compute_trading_metrics(self, epoch: int = 0) -> Dict[str, float]:
        """
        MONITORING SWEEP: Compute trading metrics for training observability.
        
        NOTE: This is for MONITORING ONLY during training.
        - Does NOT save or freeze any policy
        - Does NOT alter training behavior
        - Runs every MONITORING_EPOCH_INTERVAL epochs to save time
        
        For the actual frozen execution policy, use the dedicated
        post-training PolicySelector class after training completes.
        
        Returns:
            Dictionary of trading metrics (informational only)
        """
        # Only run monitoring sweep every N epochs to save time
        MONITORING_EPOCH_INTERVAL = 5
        if epoch > 0 and epoch % MONITORING_EPOCH_INTERVAL != 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0,
                'risk_adjusted_score': 0.0,
                'min_confidence': 0.0, 'spread_multiplier': 0.0, 'cooldown': 0,
                '_skipped': True
            }
        # Trading policy parameters
        FIXED_COST = 0.0009  # 0.09% round-trip cost
        SPREAD_MULTIPLIER = 3.0  # K: require spread >= K * cost
        COOLDOWN = 4  # Bars to wait after a trade
        
        # Confidence thresholds to sweep - softmax probability based
        # For 3-class: random = 0.33, so sweep from 0.35 (barely above random) to 0.70 (high conviction)
        CONFIDENCE_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        
        # Collect all model outputs
        all_predictions = []
        all_probs = []
        all_returns = []
        all_mus = []
        all_sigmas = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch[0].to(self.device)
                returns = batch[2]
                output = self.model.forward_multihead(features)
                
                probs = torch.softmax(output.class_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                all_mus.extend(output.mu.squeeze().cpu().numpy())
                all_sigmas.extend(output.sigma.squeeze().cpu().numpy())
        
        preds = np.array(all_predictions)
        probs_arr = np.array(all_probs)
        returns = np.array(all_returns)
        mus = np.array(all_mus)
        raw_sigmas = np.array(all_sigmas)
        
        # Prediction distribution
        n_total = len(preds)
        n_short = (preds == 0).sum()
        n_hold = (preds == 1).sum()
        n_long = (preds == 2).sum()
        logger.info("Predictions (epoch %d): S:%d(%.0f%%) H:%d(%.0f%%) L:%d(%.0f%%)",
                   epoch, n_short, 100*n_short/n_total if n_total > 0 else 0,
                   n_hold, 100*n_hold/n_total if n_total > 0 else 0,
                   n_long, 100*n_long/n_total if n_total > 0 else 0)
        if n_short + n_long == 0:
            logger.warning(">>> MODEL PREDICTS 100%% HOLD - NO TRADES POSSIBLE <<<")
        
        # Use SOFTMAX PROBABILITY as confidence (not mu/sigma which may be untrained)
        # For each sample, confidence = max(softmax prob) for the predicted class
        confidence = probs_arr.max(axis=1)
        
        # Compute data-derived ATR from actual return volatility (rolling std)
        # This replaces untrained sigma for trade exit calculations
        rolling_window = 20
        data_atr = np.full_like(returns, np.std(returns))  # Default: global std
        for i in range(rolling_window, len(returns)):
            data_atr[i] = np.std(returns[i-rolling_window:i])
        data_atr = np.maximum(data_atr, 1e-6)  # Prevent zero
        
        # Base trade signals (LONG=2, SHORT=0)
        long_signal = preds == 2
        short_signal = preds == 0
        directional_signal = long_signal | short_signal
        
        # === SWEEP CONFIDENCE THRESHOLDS TO FIND BEST POLICY ===
        MIN_TRADES = 30
        
        best_metrics = None
        best_score = float('-inf')
        best_threshold = 0.5
        
        sweep_results = []
        
        for min_conf in CONFIDENCE_THRESHOLDS:
            conf_gate = confidence >= min_conf
            trade_allowed = conf_gate & directional_signal
            final_trades = self._apply_cooldown(trade_allowed, COOLDOWN)
            
            # Compute PnL with data-derived ATR (not untrained model sigma)
            metrics = self._compute_pnl_with_atr_exits(
                final_trades, long_signal, short_signal, returns, 
                mus, data_atr, FIXED_COST
            )
            metrics['min_confidence'] = min_conf
            metrics['spread_multiplier'] = SPREAD_MULTIPLIER
            metrics['cooldown'] = COOLDOWN
            
            # Compute risk-adjusted score:
            # score = expectancy - 0.5*max_drawdown (or -0.25*abs(avg_loss) if no DD)
            max_dd = metrics.get('max_drawdown', 0.0)
            avg_loss = metrics.get('avg_loss', 0.0)
            if max_dd > 0:
                risk_penalty = 0.5 * max_dd
            else:
                risk_penalty = 0.25 * abs(avg_loss)
            
            risk_adjusted_score = metrics['expectancy'] - risk_penalty
            metrics['risk_adjusted_score'] = risk_adjusted_score
            
            sweep_results.append(metrics)
            
            # Track best - require MIN_TRADES and use risk-adjusted score
            if metrics['num_trades'] >= MIN_TRADES and risk_adjusted_score > best_score:
                best_score = risk_adjusted_score
                best_metrics = metrics
                best_threshold = min_conf
        
        # Compact sweep report
        logger.info("-" * 70)
        logger.info("TRADING SWEEP (epoch %d) | cooldown=%d bars | confidence=softmax prob", epoch, COOLDOWN)
        logger.info("%-8s %6s %8s %7s %7s %8s %7s", "Conf>=", "Trades", "Expect", "WinRate", "Sharpe", "PF", "")
        logger.info("-" * 70)
        for m in sweep_results:
            eligible = m['num_trades'] >= MIN_TRADES
            is_best = m['min_confidence'] == best_threshold and eligible and best_score > float('-inf')
            marker = " << BEST" if is_best else ""
            logger.info(
                "%-8.0f%% %5d  %+.4f  %5.1f%%  %+5.2f   %5.2f%s",
                m['min_confidence'] * 100, m['num_trades'],
                m['expectancy'], m['hit_rate'] * 100,
                m['sharpe'], m['profit_factor'], marker
            )
        logger.info("-" * 70)
        
        if best_metrics is None:
            best_metrics = sweep_results[-1] if sweep_results else {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0,
                'risk_adjusted_score': 0.0,
                'min_confidence': 0.5, 'spread_multiplier': 3.0, 'cooldown': 4
            }
        
        return best_metrics
    
    def _compute_hold_rate(self, val_metrics: Dict) -> Optional[float]:
        """
        Compute the HOLD prediction rate from health monitor class distribution.
        
        Used by stability guardrail to detect mode collapse.
        Uses existing training predictions from health_monitor, avoiding 
        redundant validation passes.
        
        Args:
            val_metrics: Validation metrics dictionary (not used, kept for API compat)
            
        Returns:
            Float in [0, 1] representing fraction of HOLD predictions,
            or None if not computable
        """
        try:
            # Use class distribution from health monitor (from training predictions)
            # This avoids a redundant validation pass
            if hasattr(self, 'health_monitor') and self.health_monitor.class_distribution_history:
                latest_dist = self.health_monitor.class_distribution_history[-1]
                return latest_dist.get(1, 0.0)  # Class 1 = HOLD
            
            return None
        except Exception as e:
            logger.warning(f"Could not compute HOLD rate: {e}")
            return None
    
    def _apply_cooldown(self, trade_signals: np.ndarray, cooldown: int) -> np.ndarray:
        """
        Apply cooldown to prevent signal spam.
        After taking a trade, no new trades for `cooldown` candles.
        
        Args:
            trade_signals: Boolean array of trade signals
            cooldown: Number of bars to wait after a trade
            
        Returns:
            Filtered trade signals with cooldown applied
        """
        result = np.zeros_like(trade_signals, dtype=bool)
        last_trade_idx = -cooldown - 1  # Start with no cooldown active
        
        for i in range(len(trade_signals)):
            if trade_signals[i] and (i - last_trade_idx) > cooldown:
                result[i] = True
                last_trade_idx = i
        
        return result
    
    def _compute_pnl_with_quantile_exits(
        self, 
        final_trades: np.ndarray,
        long_signal: np.ndarray,
        short_signal: np.ndarray,
        returns: np.ndarray,
        q10: np.ndarray,
        q25: np.ndarray,
        q75: np.ndarray,
        q90: np.ndarray,
        cost: float
    ) -> Dict[str, float]:
        """
        Compute PnL using asymmetric SL/TP derived from quantiles.
        
        For LONG trades:
            - SL distance from q10 (downside risk)
            - TP from q75 or q90 (upside potential)
        For SHORT trades:
            - SL distance from q90 (upside risk)
            - TP from q10 or q25 (downside potential)
        
        This ensures proper risk:reward asymmetry.
        """
        metrics = {}
        
        # Get trades
        long_trades = final_trades & long_signal
        short_trades = final_trades & short_signal
        trade_mask = long_trades | short_trades
        
        num_trades = trade_mask.sum()
        if num_trades == 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0
            }
        
        # Compute PnL with asymmetric exits
        # For simulation, we still use actual returns but cap based on quantiles
        trade_pnl = np.zeros(len(returns))
        
        for i in range(len(returns)):
            if long_trades[i]:
                # LONG trade: profit if price goes up
                actual_ret = returns[i]
                sl_level = q10[i]  # Stop at q10
                tp_level = q75[i]  # Take profit at q75
                
                # Simulate exit: hit SL if return goes below q10, hit TP if above q75
                if actual_ret <= sl_level:
                    pnl = sl_level - cost  # Stopped out
                elif actual_ret >= tp_level:
                    pnl = tp_level - cost  # Take profit hit
                else:
                    pnl = actual_ret - cost  # Normal exit
                trade_pnl[i] = pnl
                
            elif short_trades[i]:
                # SHORT trade: profit if price goes down
                actual_ret = returns[i]
                sl_level = q90[i]  # Stop at q90 (price going up = bad)
                tp_level = q25[i]  # Take profit at q25 (price going down = good)
                
                # For short: we profit when price goes down (negative return)
                # SL triggers if return > q90, TP if return < q25
                if actual_ret >= sl_level:
                    pnl = -sl_level - cost  # Stopped out
                elif actual_ret <= tp_level:
                    pnl = -tp_level - cost  # Take profit hit
                else:
                    pnl = -actual_ret - cost  # Normal exit
                trade_pnl[i] = pnl
        
        # Filter to actual trades
        trade_returns = trade_pnl[trade_mask]
        num_trades = len(trade_returns)
        
        # Expectancy
        metrics['expectancy'] = float(np.mean(trade_returns)) if num_trades > 0 else 0.0
        
        # Hit rate
        wins = (trade_returns > 0).sum()
        metrics['hit_rate'] = float(wins / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns < 0].sum())
        metrics['profit_factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        # Sharpe ratio
        if num_trades > 1 and np.std(trade_returns) > 0:
            trades_per_year = num_trades * (35040 / max(len(returns), 1))
            annual_factor = np.sqrt(max(trades_per_year, 1))
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * annual_factor
            metrics['sharpe'] = float(sharpe)
        else:
            metrics['sharpe'] = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Number of trades
        metrics['num_trades'] = int(num_trades)
        
        # Average win / loss
        if wins > 0:
            metrics['avg_win'] = float(np.mean(trade_returns[trade_returns > 0]))
        else:
            metrics['avg_win'] = 0.0
        
        losses_count = (trade_returns < 0).sum()
        if losses_count > 0:
            metrics['avg_loss'] = float(np.mean(trade_returns[trade_returns < 0]))
        else:
            metrics['avg_loss'] = 0.0
        
        # Win/loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
        
        return metrics
    
    def _compute_pnl_with_atr_exits(
        self, 
        final_trades: np.ndarray,
        long_signal: np.ndarray,
        short_signal: np.ndarray,
        returns: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        cost: float
    ) -> Dict[str, float]:
        """
        Compute PnL using ATR-based asymmetric SL/TP.
        
        Uses sigma (predicted volatility) as ATR proxy:
        - SL = 1.5 × sigma (stop loss distance)
        - TP = 2.2 × sigma (take profit distance)
        - Enforces minimum RR >= 1.5
        
        This provides proper risk:reward asymmetry based on volatility.
        """
        SL_ATR_MULT = 1.5   # Stop loss = 1.5 × ATR (sigma)
        TP_ATR_MULT = 2.5   # Take profit = 2.5 × ATR (sigma)
        MIN_RR = 1.5        # Minimum risk:reward ratio
        
        metrics = {}
        
        # Get trades
        long_trades = final_trades & long_signal
        short_trades = final_trades & short_signal
        trade_mask = long_trades | short_trades
        
        num_trades = trade_mask.sum()
        if num_trades == 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0
            }
        
        # Compute PnL with ATR-based asymmetric exits
        trade_pnl = np.zeros(len(returns))
        
        for i in range(len(returns)):
            if not (long_trades[i] or short_trades[i]):
                continue
                
            # Use sigma as ATR proxy (represents volatility/uncertainty)
            atr = max(sigmas[i], 1e-6)  # Prevent division by zero
            
            # Compute SL and TP distances
            sl_distance = SL_ATR_MULT * atr
            tp_distance = TP_ATR_MULT * atr
            
            # Enforce minimum RR ratio
            if tp_distance < MIN_RR * sl_distance:
                tp_distance = MIN_RR * sl_distance
            
            actual_ret = returns[i]
            
            if long_trades[i]:
                # LONG trade: profit if price goes up
                # SL triggers if return goes below -sl_distance
                # TP triggers if return goes above +tp_distance
                if actual_ret <= -sl_distance:
                    pnl = -sl_distance - cost  # Stopped out (loss)
                elif actual_ret >= tp_distance:
                    pnl = tp_distance - cost   # Take profit hit (win)
                else:
                    pnl = actual_ret - cost    # Normal exit
                trade_pnl[i] = pnl
                
            elif short_trades[i]:
                # SHORT trade: profit if price goes down
                # SL triggers if return goes above +sl_distance (price up = bad)
                # TP triggers if return goes below -tp_distance (price down = good)
                if actual_ret >= sl_distance:
                    pnl = -sl_distance - cost  # Stopped out (loss)
                elif actual_ret <= -tp_distance:
                    pnl = tp_distance - cost   # Take profit hit (win)
                else:
                    pnl = -actual_ret - cost   # Normal exit (profit when price down)
                trade_pnl[i] = pnl
        
        # Filter to actual trades
        trade_returns = trade_pnl[trade_mask]
        num_trades = len(trade_returns)
        
        # Expectancy
        metrics['expectancy'] = float(np.mean(trade_returns)) if num_trades > 0 else 0.0
        
        # Hit rate
        wins = (trade_returns > 0).sum()
        metrics['hit_rate'] = float(wins / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns < 0].sum())
        metrics['profit_factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        # Sharpe ratio - annualize based on trade frequency relative to 15m bars
        if num_trades > 1 and np.std(trade_returns) > 0:
            trades_per_year = num_trades * (35040 / max(len(returns), 1))
            annual_factor = np.sqrt(max(trades_per_year, 1))
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * annual_factor
            metrics['sharpe'] = float(sharpe)
        else:
            metrics['sharpe'] = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Number of trades
        metrics['num_trades'] = int(num_trades)
        
        # Average win / loss
        if wins > 0:
            metrics['avg_win'] = float(np.mean(trade_returns[trade_returns > 0]))
        else:
            metrics['avg_win'] = 0.0
        
        losses_count = (trade_returns < 0).sum()
        if losses_count > 0:
            metrics['avg_loss'] = float(np.mean(trade_returns[trade_returns < 0]))
        else:
            metrics['avg_loss'] = 0.0
        
        # Win/loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
        
        return metrics
    
    def _compute_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-regime trading metrics for validation.
        
        Tracks performance separately for:
        - BULL (0): Trending up markets
        - BEAR (1): Trending down markets  
        - HIGH_VOL (2): High volatility periods
        - LOW_VOL_CHOP (3): Low volatility ranging
        
        IMPORTANT: This method assumes val_loader iterates in sequential order
        (shuffle=False, no random sampler). This is enforced by the training
        pipeline which only uses balanced sampling for train_loader, not val_loader.
        
        Returns:
            Dictionary mapping regime name to metrics dict
        """
        REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "HIGH_VOL", 3: "LOW_VOL_CHOP"}
        
        # Check if val_loader dataset has regime_ids
        val_dataset = self.val_loader.dataset
        if not hasattr(val_dataset, 'regime_ids') or val_dataset.regime_ids is None:
            return {}  # No regime data available
        
        # CRITICAL: Verify val_loader uses sequential ordering (not shuffled/random sampler)
        # Regime ID alignment depends on deterministic sequential iteration
        sampler = self.val_loader.sampler
        
        # STRICT CHECK: Require SequentialSampler for regime ID alignment
        # Any non-sequential sampler will cause misalignment between predictions and regime IDs
        if not isinstance(sampler, SequentialSampler):
            sampler_name = type(sampler).__name__
            logger.info(f"val_loader uses {sampler_name} (not SequentialSampler) - skipping per-regime metrics for alignment safety")
            return {}
        
        # Pre-compute full regime IDs array for all valid indices (sequential order)
        all_regime_ids_precomputed = val_dataset.get_regime_ids_for_valid_indices()
        expected_samples = len(val_dataset)
        
        all_predictions = []
        all_returns = []
        total_processed = 0
        
        cost = 0.001  # 0.1% round-trip cost
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
                features = batch[0].to(self.device)
                returns = batch[2]  # returns is always at index 2
                output = self.model.forward_multihead(features)
                
                probs = torch.softmax(output.class_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                batch_size = len(returns)
                all_predictions.extend(preds.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                total_processed += batch_size
        
        preds = np.array(all_predictions)
        returns = np.array(all_returns)
        
        # STRICT ALIGNMENT: Processed samples must match available regime IDs
        if total_processed != len(all_regime_ids_precomputed):
            # Allow for drop_last=True which drops incomplete final batch
            batch_size = self.val_loader.batch_size or 1
            tolerance = batch_size  # At most one batch can be dropped
            difference = abs(total_processed - len(all_regime_ids_precomputed))
            
            if difference > tolerance:
                logger.error(f"Regime ID mismatch: processed {total_processed} samples but "
                            f"{len(all_regime_ids_precomputed)} regime IDs available (diff={difference})")
                return {}
            else:
                logger.info(f"Minor sample count difference ({difference}) likely from drop_last, proceeding")
        
        # Use precomputed regime IDs, sliced to match number of processed samples
        regime_ids = all_regime_ids_precomputed[:len(preds)]
        
        # Final strict length check
        if len(preds) != len(regime_ids):
            logger.error(f"Length mismatch after slicing: {len(preds)} predictions vs {len(regime_ids)} regime IDs")
            return {}
        
        regime_metrics = {}
        
        for regime_id, regime_name in REGIME_NAMES.items():
            regime_mask = regime_ids == regime_id
            regime_count = regime_mask.sum()
            
            if regime_count == 0:
                regime_metrics[regime_name] = {
                    'samples': 0, 'trades': 0, 'expectancy': 0.0, 'hit_rate': 0.0
                }
                continue
            
            regime_preds = preds[regime_mask]
            regime_returns = returns[regime_mask]
            
            # Filter for directional predictions
            long_mask = regime_preds == 2
            short_mask = regime_preds == 0
            trade_mask = long_mask | short_mask
            
            # Compute PnL for each trade
            trade_pnl = np.zeros(len(regime_returns))
            trade_pnl[long_mask] = regime_returns[long_mask] - cost
            trade_pnl[short_mask] = -regime_returns[short_mask] - cost
            
            trade_returns = trade_pnl[trade_mask]
            num_trades = len(trade_returns)
            
            if num_trades == 0:
                regime_metrics[regime_name] = {
                    'samples': int(regime_count),
                    'trades': 0,
                    'expectancy': 0.0,
                    'hit_rate': 0.0
                }
                continue
            
            expectancy = float(np.mean(trade_returns))
            wins = (trade_returns > 0).sum()
            hit_rate = float(wins / num_trades) if num_trades > 0 else 0.0
            
            regime_metrics[regime_name] = {
                'samples': int(regime_count),
                'trades': int(num_trades),
                'expectancy': expectancy,
                'hit_rate': hit_rate
            }
        
        # Log per-regime performance
        logger.info("Per-Regime Validation Metrics:")
        for regime_name, metrics in regime_metrics.items():
            if metrics['trades'] > 0:
                logger.info(f"  {regime_name}: {metrics['samples']} samples, {metrics['trades']} trades, "
                           f"Exp={metrics['expectancy']:.4f}, HitRate={metrics['hit_rate']:.2%}")
            else:
                logger.info(f"  {regime_name}: {metrics['samples']} samples, 0 trades")
        
        return regime_metrics
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 50,
        min_epochs: int = 40,
        save_best: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 0
    ) -> Dict[str, List[float]]:
        """
        Full training loop with min_epochs protection.
        
        Args:
            num_epochs: Total epochs to train (default from config, typically 300)
            early_stopping_patience: Epochs without improvement before stopping (default 50)
            min_epochs: Minimum epochs before early stopping can trigger (default 40)
            save_best: Whether to save best checkpoint
            checkpoint_path: Path to save checkpoint
            checkpoint_interval: Pause every N epochs for user review (0 = no pausing)
        
        Returns history of metrics per epoch.
        
        IMPORTANT: Early stopping uses val_loss ONLY (not monitoring sweep expectancy).
        PolicySelector handles policy selection post-training.
        Dual checkpoint saving: best_loss.pt (val loss) + best_trading.pt (trading score).
        """
        epochs = num_epochs or self.config.training.epochs
        
        logger.info(f"[TRAINING CONFIG] epochs={epochs}, min_epochs={min_epochs}, patience={early_stopping_patience}")
        logger.info(f"[TRAINING CONFIG] Early stopping uses val_loss only - PolicySelector handles policy post-training")
        
        # === DIAGNOSTIC: Log training label distribution at start ===
        # PHASE 2: Also compute focal alpha and set prior biases
        try:
            all_labels = []
            for batch in self.train_loader:
                labels = batch[1]  # labels are second element
                all_labels.extend(labels.cpu().numpy())
            all_labels = np.array(all_labels)
            n_total = len(all_labels)
            n_short = (all_labels == 0).sum()
            n_hold = (all_labels == 1).sum()
            n_long = (all_labels == 2).sum()
            logger.info("=" * 70)
            logger.info("TRAINING LABEL DISTRIBUTION:")
            logger.info("  SHORT (0): %5d / %d (%.1f%%)", n_short, n_total, 100*n_short/n_total if n_total > 0 else 0)
            logger.info("  HOLD  (1): %5d / %d (%.1f%%)", n_hold, n_total, 100*n_hold/n_total if n_total > 0 else 0)
            logger.info("  LONG  (2): %5d / %d (%.1f%%)", n_long, n_total, 100*n_long/n_total if n_total > 0 else 0)
            if n_hold / n_total > 0.90:
                logger.warning(">>> TRAINING DATA IS %.1f%% HOLD - MODEL WILL LEARN TO PREDICT HOLD <<<", 
                              100*n_hold/n_total)
                logger.warning(">>> CONSIDER: Use pure_directional or regime label mode to balance labels <<<")
            logger.info("=" * 70)
            
            # === PHASE 2: Compute class priors and focal alpha ===
            if n_total > 0:
                class_priors = torch.tensor([
                    n_short / n_total,
                    n_hold / n_total, 
                    n_long / n_total
                ], dtype=torch.float32)
                
                # Focal Loss alpha = inverse frequency (higher weight for rare classes)
                # Normalize so they sum to num_classes (3.0)
                inv_freq = 1.0 / (class_priors + 1e-6)
                focal_alpha = inv_freq / inv_freq.sum() * 3.0
                focal_alpha = torch.clamp(focal_alpha, max=10.0)  # Cap to prevent instability
                
                logger.info("PHASE 2 - FOCAL LOSS CONFIGURATION:")
                logger.info("  Class priors: SHORT=%.3f, HOLD=%.3f, LONG=%.3f", 
                           class_priors[0], class_priors[1], class_priors[2])
                logger.info("  Focal alpha:  SHORT=%.3f, HOLD=%.3f, LONG=%.3f",
                           focal_alpha[0], focal_alpha[1], focal_alpha[2])
                
                # Update the FocalLoss with computed alpha if using focal loss
                # Use buffer-safe set_alpha method to avoid device/state issues
                # BUG FIX: Now works through OHEM wrapper via pass-through set_alpha()
                if hasattr(self.criterion, 'class_loss') and hasattr(self.criterion.class_loss, 'set_alpha'):
                    self.criterion.class_loss.set_alpha(focal_alpha.to(self.device))
                    logger.info("  -> ✓ Updated FocalLoss alpha weights (buffer-safe)")
                    
                    # VALIDATION: Verify alpha was actually applied
                    class_loss = self.criterion.class_loss
                    if hasattr(class_loss, 'base_loss'):
                        # OHEM wrapper - check underlying FocalLoss
                        inner_loss = class_loss.base_loss
                        if hasattr(inner_loss, 'alpha') and hasattr(inner_loss, '_alpha_initialized'):
                            if inner_loss._alpha_initialized:
                                logger.info(f"  -> ✓ VERIFIED: FocalLoss alpha = {inner_loss.alpha.tolist()}")
                            else:
                                logger.warning("  -> ✗ WARNING: FocalLoss alpha NOT initialized!")
                    elif hasattr(class_loss, 'alpha') and hasattr(class_loss, '_alpha_initialized'):
                        # Direct FocalLoss
                        if class_loss._alpha_initialized:
                            logger.info(f"  -> ✓ VERIFIED: FocalLoss alpha = {class_loss.alpha.tolist()}")
                        else:
                            logger.warning("  -> ✗ WARNING: FocalLoss alpha NOT initialized!")
                else:
                    logger.warning("  -> ✗ WARNING: Could not set FocalLoss alpha - class_loss missing set_alpha method")
                
                # STABILITY FIX: Prior bias initialization DISABLED
                # Was causing model to collapse to one class early in training
                # if hasattr(self.model, 'class_head') and hasattr(self.model.class_head, 'set_class_priors'):
                #     self.model.class_head.set_class_priors(class_priors.to(self.device))
                #     logger.info("  -> Initialized classification head with prior biases")
                logger.info("  -> Prior bias initialization DISABLED for stability")
                
                logger.info("=" * 70)
                
        except Exception as e:
            logger.warning(f"Could not compute label distribution: {e}")
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'class_loss': [], 'mu_loss': [],
            'quantile_loss': []
        }
        
        # Track best trading metrics for model_weights.json save
        best_trading_metrics = None
        best_trading_score = float('-inf')
        best_val_epoch = 0
        
        # Dual checkpoint: best by trading score
        best_trading_checkpoint_score = float('-inf')
        best_trading_checkpoint_epoch = 0
        TRADING_MIN_TRADES = 150
        
        # === STABILITY GUARDRAILS ===
        # Track consecutive HOLD collapse and gradient explosion epochs
        consecutive_hold_collapse = 0
        consecutive_high_grad_norm = 0
        HOLD_COLLAPSE_THRESHOLD = 0.95  # >95% HOLD predictions
        HOLD_COLLAPSE_MAX_EPOCHS = 3  # Abort after 3 consecutive collapse epochs
        GRAD_NORM_THRESHOLD = 20.0
        GRAD_NORM_MAX_EPOCHS = 3  # Reduce LR after 3 consecutive high grad epochs
        last_good_state = None  # For potential rollback
        
        for epoch in range(epochs):
            # Capture LR used for this epoch (before stepping)
            epoch_lr = self.optimizer.param_groups[0]['lr']
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch=epoch)
            
            # Step epoch-level LR scheduler (after train+val)
            self.scheduler.step()
            
            # === STABILITY CHECK: HOLD Collapse Guardrail ===
            hold_rate = self._compute_hold_rate(val_metrics)
            if hold_rate is not None and hold_rate > HOLD_COLLAPSE_THRESHOLD:
                consecutive_hold_collapse += 1
                logger.warning(f"[GUARDRAIL] HOLD collapse detected: {hold_rate*100:.1f}% ({consecutive_hold_collapse}/{HOLD_COLLAPSE_MAX_EPOCHS} epochs)")
                
                if consecutive_hold_collapse >= HOLD_COLLAPSE_MAX_EPOCHS:
                    logger.error(f"[GUARDRAIL] TRAINING ABORTED: HOLD collapse for {HOLD_COLLAPSE_MAX_EPOCHS} consecutive epochs")
                    logger.error(f"[GUARDRAIL] Model is not learning to differentiate - check data/loss/LR")
                    # Clean up resources
                    self.writer.close()
                    # Return early with failure indication
                    history['aborted'] = True
                    history['abort_reason'] = 'hold_collapse'
                    history['abort_epoch'] = epoch + 1
                    return history
            else:
                consecutive_hold_collapse = 0  # Reset if predictions diversify
            
            # === STABILITY CHECK: Gradient Explosion Auto-LR Reduction ===
            grad_norm = train_metrics.get('gradient_norm', 0.0)
            if grad_norm > GRAD_NORM_THRESHOLD:
                consecutive_high_grad_norm += 1
                logger.warning(f"[GUARDRAIL] High gradient norm: {grad_norm:.2f} > {GRAD_NORM_THRESHOLD} ({consecutive_high_grad_norm}/{GRAD_NORM_MAX_EPOCHS} epochs)")
                
                if consecutive_high_grad_norm >= GRAD_NORM_MAX_EPOCHS:
                    new_base_lr = self.base_lr * 0.5
                    remaining_epochs = epochs - epoch - 1
                    
                    logger.info(f"[GUARDRAIL] 3/3 triggered - Reinitializing optimizer + scheduler")
                    logger.info(f"[GUARDRAIL] New base_lr={new_base_lr:.2e} (was {self.base_lr:.2e})")
                    logger.info(f"[GUARDRAIL] Remaining epochs={remaining_epochs}")
                    
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=new_base_lr,
                        weight_decay=self.config.training.weight_decay
                    )
                    logger.info(f"[GUARDRAIL] AdamW optimizer recreated - momentum states cleared")
                    
                    self.base_lr = new_base_lr
                    
                    if remaining_epochs > 0:
                        self.scheduler = CosineAnnealingLR(
                            self.optimizer,
                            T_max=remaining_epochs,
                            eta_min=new_base_lr * 0.05
                        )
                        logger.info(f"[GUARDRAIL] Scheduler reset: CosineAnnealingLR T_max={remaining_epochs} epochs, "
                                  f"current_lr={self.optimizer.param_groups[0]['lr']:.2e}")
                    else:
                        logger.info(f"[GUARDRAIL] Near end of training, using constant LR={new_base_lr:.2e}")
                        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda x: 1.0)
                    
                    consecutive_high_grad_norm = 0
            else:
                consecutive_high_grad_norm = 0  # Reset if gradients stabilize
            
            # Log metrics
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['class_loss'].append(val_metrics['class'])
            history['mu_loss'].append(val_metrics['mu'])
            history['quantile_loss'].append(val_metrics['quantile'])
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_metrics['total'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['total'], epoch)
            self.writer.add_scalar('Loss/class', val_metrics['class'], epoch)
            self.writer.add_scalar('Loss/mu', val_metrics['mu'], epoch)
            self.writer.add_scalar('Loss/quantile', val_metrics['quantile'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            # Quantile calibration
            for q in ['q10', 'q25', 'q50', 'q75', 'q90']:
                if f'{q}_cal' in val_metrics:
                    self.writer.add_scalar(f'Calibration/{q}', val_metrics[f'{q}_cal'], epoch)
            
            # Trading metrics
            if 'expectancy' in val_metrics:
                self.writer.add_scalar('Trading/expectancy', val_metrics['expectancy'], epoch)
                self.writer.add_scalar('Trading/hit_rate', val_metrics['hit_rate'], epoch)
                self.writer.add_scalar('Trading/sharpe', val_metrics['sharpe'], epoch)
                self.writer.add_scalar('Trading/profit_factor', val_metrics['profit_factor'], epoch)
                self.writer.add_scalar('Trading/max_drawdown', val_metrics['max_drawdown'], epoch)
                self.writer.add_scalar('Trading/num_trades', val_metrics['num_trades'], epoch)
                
                # Track best trading metrics for model_weights.json
                # Always track, but prefer runs with more trades
                current_score = val_metrics.get('risk_adjusted_score', val_metrics['expectancy'])
                num_trades = val_metrics.get('num_trades', 0)
                
                # Update best if: (a) more trades OR (b) same/more trades with better score
                should_update = False
                if best_trading_metrics is None:
                    should_update = True  # First observation
                elif num_trades > best_trading_metrics.get('num_trades', 0):
                    should_update = True  # More trades = better sample
                elif num_trades == best_trading_metrics.get('num_trades', 0) and current_score > best_trading_score:
                    should_update = True  # Same trades, better score
                
                if should_update:
                    best_trading_score = current_score
                    best_trading_metrics = val_metrics.copy()
                    best_trading_metrics['best_epoch'] = epoch + 1
                    logger.info(f"[BEST TRADING] Updated at epoch {epoch+1}: score={current_score:.4f}, trades={num_trades}")
            
            # Progress callback
            if self.epoch_callback:
                self.epoch_callback(epoch, epochs, train_metrics, val_metrics)
            
            # Logging
            if not self.gui_mode:
                short_acc = val_metrics.get('acc_short', 0)
                hold_acc = val_metrics.get('acc_hold', 0)
                long_acc = val_metrics.get('acc_long', 0)
                pred_s = val_metrics.get('pred_short_pct', 0)
                pred_h = val_metrics.get('pred_hold_pct', 0)
                pred_l = val_metrics.get('pred_long_pct', 0)
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss T:{train_metrics['total']:.4f} V:{val_metrics['total']:.4f} | "
                    f"Acc:{val_metrics['accuracy']:.1%} S:{short_acc:.0%} H:{hold_acc:.0%} L:{long_acc:.0%} | "
                    f"Pred S:{pred_s:.0%} H:{pred_h:.0%} L:{pred_l:.0%} | "
                    f"LR:{epoch_lr:.1e}"
                )
            
            # Early stopping check - uses val_loss ONLY (not monitoring sweep expectancy)
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                best_val_epoch = epoch + 1
                
                if save_best:
                    # Save best-by-loss checkpoint
                    loss_path = str(Path(checkpoint_path).parent / "best_loss.pt") if checkpoint_path else "checkpoints/best_loss.pt"
                    self._save_checkpoint(loss_path, val_metrics)
                    logger.info(f"[BEST LOSS] Saved at epoch {epoch+1} | val_loss={val_metrics['total']:.4f} | LR={epoch_lr:.2e}")
            else:
                self.patience_counter += 1
            
            # === DUAL CHECKPOINT: Save best-by-trading model ===
            if not val_metrics.get('_skipped', False) and val_metrics.get('num_trades', 0) >= TRADING_MIN_TRADES:
                expectancy = val_metrics.get('expectancy', 0.0)
                sharpe = val_metrics.get('sharpe', 0.0)
                pf = val_metrics.get('profit_factor', 0.0)
                trading_score = expectancy + 0.1 * sharpe + 0.02 * math.log(max(pf, 1e-6))
                
                if trading_score > best_trading_checkpoint_score:
                    best_trading_checkpoint_score = trading_score
                    best_trading_checkpoint_epoch = epoch + 1
                    
                    trade_path = str(Path(checkpoint_path).parent / "best_trading.pt") if checkpoint_path else "checkpoints/best_trading.pt"
                    self._save_checkpoint(trade_path, val_metrics)
                    logger.info(
                        f"[BEST TRADING] Saved at epoch {epoch+1} | "
                        f"score={trading_score:+.4f} | trades={val_metrics['num_trades']} | "
                        f"exp={expectancy:+.4f} | sharpe={sharpe:+.2f} | pf={pf:.2f}"
                    )
            
            # CRITICAL: Early stopping ONLY after min_epochs reached
            # This ensures multihead quantiles/vol_state/accel have enough epochs to converge
            if epoch + 1 >= min_epochs and self.patience_counter >= early_stopping_patience:
                logger.info(f"[EARLY STOPPING] Triggered at epoch {epoch+1} (min_epochs={min_epochs} reached, no improvement for {early_stopping_patience} epochs)")
                break
            elif epoch + 1 < min_epochs and self.patience_counter >= early_stopping_patience:
                # Log but DO NOT break - keep training until min_epochs
                logger.info(f"[MIN_EPOCHS PROTECTION] Epoch {epoch+1}/{epochs} - patience exhausted but min_epochs={min_epochs} not reached, continuing...")
            
            # === INTERACTIVE CHECKPOINT: Pause for user review ===
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) < epochs:
                short_acc = val_metrics.get('acc_short', 0) * 100
                hold_acc = val_metrics.get('acc_hold', 0) * 100
                long_acc = val_metrics.get('acc_long', 0) * 100
                ps = val_metrics.get('pred_short_pct', 0) * 100
                ph = val_metrics.get('pred_hold_pct', 0) * 100
                pl = val_metrics.get('pred_long_pct', 0) * 100
                
                print("\n" + "=" * 60)
                print(f"  CHECKPOINT @ Epoch {epoch+1}/{epochs}")
                print("=" * 60)
                print(f"  Val Accuracy:  {val_metrics.get('accuracy', 0)*100:.1f}%")
                print(f"  Val Loss:      {val_metrics.get('total', 0):.4f}")
                print(f"  Train Loss:    {train_metrics.get('total', 0):.4f}")
                print(f"  Best Val Loss: {self.best_val_loss:.4f} (epoch {best_val_epoch})")
                print(f"  Patience:      {self.patience_counter}/{early_stopping_patience}")
                print(f"  Current LR:    {self.optimizer.param_groups[0]['lr']:.2e}")
                print("-" * 60)
                print(f"  Per-class Accuracy (recall):")
                print(f"    SHORT: {short_acc:.1f}%  |  HOLD: {hold_acc:.1f}%  |  LONG: {long_acc:.1f}%")
                print(f"  Prediction Distribution:")
                print(f"    SHORT: {ps:.1f}%  |  HOLD: {ph:.1f}%  |  LONG: {pl:.1f}%")
                if val_metrics.get('num_trades', 0) > 0:
                    print(f"  Trading Metrics:")
                    print(f"    Expectancy:    {val_metrics.get('expectancy', 0):+.4f}")
                    print(f"    Win Rate:      {val_metrics.get('hit_rate', 0)*100:.1f}%")
                    print(f"    Sharpe:        {val_metrics.get('sharpe', 0):+.2f}")
                    print(f"    Profit Factor: {val_metrics.get('profit_factor', 0):.2f}")
                    print(f"    Trades:        {val_metrics.get('num_trades', 0)}")
                    print(f"    Avg Win/Loss:  {val_metrics.get('avg_win', 0):+.4f} / {val_metrics.get('avg_loss', 0):+.4f}")
                else:
                    print(f"  Trading: No trades yet (monitoring sweep runs every 5 epochs)")
                print("=" * 60)
                
                try:
                    user_input = input("  Continue training? [y/n] (default: y): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    user_input = 'n'
                
                if user_input == 'n':
                    print(f"\n  Stopping early at epoch {epoch+1} (user requested)")
                    print(f"  Best model saved from epoch {best_val_epoch}")
                    history['user_stopped'] = True
                    history['user_stopped_epoch'] = epoch + 1
                    break
                else:
                    remaining = epochs - (epoch + 1)
                    print(f"  Continuing... ({remaining} epochs remaining)")
                    print()
        
        self.writer.close()
        
        # === DUAL CHECKPOINT SUMMARY ===
        logger.info("=" * 70)
        logger.info("CHECKPOINT SUMMARY")
        logger.info("-" * 70)
        logger.info(f"  Best Loss:    epoch {best_val_epoch} | val_loss={self.best_val_loss:.4f}")
        if best_trading_checkpoint_epoch > 0:
            logger.info(f"  Best Trading: epoch {best_trading_checkpoint_epoch} | score={best_trading_checkpoint_score:+.4f}")
        else:
            logger.info(f"  Best Trading: No eligible checkpoint (needs >= {TRADING_MIN_TRADES} trades)")
        logger.info("=" * 70)
        
        # === SAVE WALK-FORWARD WEIGHTS TO model_weights.json ===
        if best_trading_metrics is not None:
            try:
                total_trades = best_trading_metrics.get('num_trades', 0)
                
                # Convert to format expected by save_walk_forward_weights
                wf_summary = {
                    'total_trades': total_trades,
                    'overall_win_rate': best_trading_metrics.get('hit_rate', 0.5),
                    'overall_expectancy': best_trading_metrics.get('expectancy', 0.0),
                    'overall_profit_factor': best_trading_metrics.get('profit_factor', 1.0),
                    'overall_sharpe': best_trading_metrics.get('sharpe', 0.0),
                    'worst_drawdown': best_trading_metrics.get('max_drawdown', 0.0),
                    'n_folds': 1,  # Single validation split (not true walk-forward)
                    'avg_trades_per_fold': total_trades,
                    'best_epoch': best_trading_metrics.get('best_epoch', epochs),
                    'source': 'validation_sweep'  # Mark as val-derived, not full walk-forward
                }
                
                # Determine model name from model attribute or default
                model_name = getattr(self.model, 'name', 'unknown').lower()
                
                # Save to checkpoints directory
                weights_dir = str(Path(__file__).parent.parent / "checkpoints")
                
                logger.info("=" * 70)
                logger.info("SAVING WALK-FORWARD WEIGHTS TO model_weights.json")
                logger.info(f"  Model: {model_name}")
                logger.info(f"  Trades: {total_trades}")
                logger.info(f"  Expectancy: {wf_summary['overall_expectancy']:.4f}")
                logger.info(f"  Win Rate: {wf_summary['overall_win_rate']:.2%}")
                logger.info(f"  Sharpe: {wf_summary['overall_sharpe']:.2f}")
                
                if total_trades < 30:
                    logger.warning(f"  ⚠️ LOW TRADE COUNT ({total_trades} < 30) - metrics may be unreliable")
                    
                logger.info("=" * 70)
                
                save_walk_forward_weights(model_name, wf_summary, weights_dir, force_save=True)
                logger.info(f"[SUCCESS] Saved weights to {weights_dir}/model_weights.json")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to save walk-forward weights: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("=" * 70)
            logger.warning("NO WALK-FORWARD WEIGHTS SAVED - No eligible trading metrics found")
            logger.warning("  Requires: num_trades >= 30 from monitoring sweeps")
            logger.warning("  Check: MIN_MOVE_FACTOR, confidence thresholds, spread filter")
            logger.warning("=" * 70)
        
        return history
    
    def _save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save model checkpoint with metrics, scaler, and feature config."""
        # Get FeatureEngineer version for tracking
        try:
            from data.pipeline import FeatureEngineer
            fe_version = FeatureEngineer.VERSION
        except:
            fe_version = "unknown"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model.name,
            'input_dim': self.model.input_dim,
            'model_type': 'multihead',
            'feature_engineer_version': fe_version,
            'training_mode': self.training_mode,  # Use actual training mode
            'horizon_periods': self.horizon_periods,  # Use actual horizon
        }
        
        # Include sklearn feature scaler if available (not AMP GradScaler)
        if self.feature_scaler is not None:
            try:
                # Save sklearn StandardScaler parameters
                checkpoint['scaler_mean'] = self.feature_scaler.mean_.tolist()
                checkpoint['scaler_scale'] = self.feature_scaler.scale_.tolist()
                checkpoint['scaler_var'] = self.feature_scaler.var_.tolist() if hasattr(self.feature_scaler, 'var_') else None
                checkpoint['scaler_n_features'] = self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else None
            except Exception as e:
                logger.warning(f"Could not save scaler state: {e}")
        
        # Include feature columns for validation at inference
        if self.feature_columns is not None:
            checkpoint['feature_columns'] = self.feature_columns
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (FE: {fe_version}, mode: {self.training_mode}, horizon: {self.horizon_periods})")


def create_multihead_dataloaders(
    features: np.ndarray,
    class_labels: np.ndarray,
    forward_returns: np.ndarray,
    sequence_length: int = 100,
    batch_size: int = 64,
    val_split: float = 0.2,
    purge_gap: int = 50
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders with proper time-series split.
    
    Uses chronological split with purge gap to prevent lookahead.
    
    Args:
        features: [n_samples, input_dim]
        class_labels: [n_samples]
        forward_returns: [n_samples]
        sequence_length: Sequence length for model
        batch_size: Batch size
        val_split: Fraction for validation
        purge_gap: Gap between train and val to prevent leakage
    """
    n_samples = len(features)
    
    # Chronological split
    split_idx = int(n_samples * (1 - val_split)) - purge_gap
    
    # Train: [0, split_idx)
    train_features = features[:split_idx]
    train_labels = class_labels[:split_idx]
    train_returns = forward_returns[:split_idx]
    
    # Val: [split_idx + purge_gap, end)
    val_start = split_idx + purge_gap
    val_features = features[val_start:]
    val_labels = class_labels[val_start:]
    val_returns = forward_returns[val_start:]
    
    # Create datasets
    train_dataset = MultiHeadDataset(
        train_features, train_labels, train_returns, sequence_length
    )
    val_dataset = MultiHeadDataset(
        val_features, val_labels, val_returns, sequence_length
    )
    
    # Create dataloaders (no shuffle for train to preserve temporal order)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader
