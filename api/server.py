import torch
import numpy as np
import time
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Callable
import asyncio
from datetime import datetime
import logging
import json
from pathlib import Path
import joblib
import glob as glob_module
from dataclasses import dataclass, field

# Walk-forward evaluation for ensemble weights
from training.walk_forward import save_walk_forward_weights, save_labeling_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Training label mapping (must match data/pipeline.py create_labels)
# 0 = SHORT, 1 = NEUTRAL, 2 = LONG
ACTION_MAP = {0: "SHORT", 1: "HOLD", 2: "LONG"}
ACTION_NAMES = ["SHORT", "HOLD", "LONG"]  # Index-aligned with training labels

# FIX #4: Lock sequence length at pipeline level
# This MUST match training config (config.data.sequence_length = 100)
# Never dynamically resize - always require exactly this many candles
SEQUENCE_LENGTH_LOCKED = 100


class PredictionDriftMonitor:
    """
    Monitor for detecting prediction drift and model degradation.
    
    Tracks:
    1. PSI (Population Stability Index) - Detects distribution shift
    2. ECE (Expected Calibration Error) - Detects calibration degradation
    3. Prediction entropy trends - Detects confidence collapse
    
    2024 Research: Early drift detection is critical for production ML systems.
    Models degrade over time as market regimes shift.
    """
    
    def __init__(self, window_size: int = 100, alert_threshold_psi: float = 0.25, alert_threshold_ece: float = 0.15):
        """
        Args:
            window_size: Number of predictions to track
            alert_threshold_psi: PSI threshold for drift alert (0.25 is industry standard)
            alert_threshold_ece: ECE threshold for calibration alert
        """
        self.window_size = window_size
        self.alert_threshold_psi = alert_threshold_psi
        self.alert_threshold_ece = alert_threshold_ece
        
        # Historical predictions
        self.prediction_probs: List[List[float]] = []  # [[p_short, p_hold, p_long], ...]
        self.prediction_outcomes: List[int] = []  # Actual outcomes (0, 1, 2)
        self.prediction_timestamps: List[str] = []
        
        # Baseline distribution (from training or first N predictions)
        self.baseline_distribution: Optional[np.ndarray] = None
        self.baseline_set = False
        
        # Alert history
        self.alerts: List[Dict] = []
        
    def add_prediction(self, probs: List[float], predicted_action: int, actual_outcome: Optional[int] = None):
        """Add a new prediction to the monitor."""
        self.prediction_probs.append(probs)
        self.prediction_timestamps.append(datetime.now().isoformat())
        
        if actual_outcome is not None:
            self.prediction_outcomes.append(actual_outcome)
        
        # Maintain window size
        if len(self.prediction_probs) > self.window_size * 2:
            self.prediction_probs = self.prediction_probs[-self.window_size:]
            self.prediction_timestamps = self.prediction_timestamps[-self.window_size:]
            if len(self.prediction_outcomes) > self.window_size:
                self.prediction_outcomes = self.prediction_outcomes[-self.window_size:]
        
        # Set baseline if not set and we have enough data
        if not self.baseline_set and len(self.prediction_probs) >= self.window_size // 2:
            self._set_baseline()
    
    def _set_baseline(self):
        """Set baseline distribution from initial predictions."""
        probs_array = np.array(self.prediction_probs)
        # Get mean probability per class
        self.baseline_distribution = probs_array.mean(axis=0)
        self.baseline_set = True
        logger.info(f"[DRIFT MONITOR] Baseline set: SHORT={self.baseline_distribution[0]:.3f}, "
                   f"HOLD={self.baseline_distribution[1]:.3f}, LONG={self.baseline_distribution[2]:.3f}")
    
    def compute_psi(self, n_bins: int = 10) -> Optional[float]:
        """
        Compute Population Stability Index (PSI) using binned probability distributions.
        
        PSI measures how much the prediction distribution has shifted from baseline.
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.25: Moderate shift, investigation needed
        PSI >= 0.25: Significant shift, action required
        
        Formula: PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        
        This implementation uses proper binning of confidence scores per class
        rather than simple mean comparison, per industry standards.
        """
        if not self.baseline_set or len(self.prediction_probs) < 20:
            return None
        
        # Compute PSI for each class using binned max-class probability distribution
        baseline_probs = np.array(self.prediction_probs[:self.window_size//2])
        recent_probs = np.array(self.prediction_probs[-self.window_size//2:])
        
        total_psi = 0.0
        eps = 1e-10
        
        # For each class, bin the probabilities and compute PSI
        for class_idx in range(3):  # SHORT, HOLD, LONG
            baseline_class_probs = baseline_probs[:, class_idx]
            recent_class_probs = recent_probs[:, class_idx]
            
            # Create histogram bins from 0 to 1
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            # Count samples in each bin
            baseline_counts, _ = np.histogram(baseline_class_probs, bins=bin_edges)
            recent_counts, _ = np.histogram(recent_class_probs, bins=bin_edges)
            
            # Convert to proportions
            baseline_pct = (baseline_counts + eps) / (baseline_counts.sum() + n_bins * eps)
            recent_pct = (recent_counts + eps) / (recent_counts.sum() + n_bins * eps)
            
            # PSI formula for this class
            class_psi = np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct))
            total_psi += class_psi
        
        # Average across classes
        avg_psi = total_psi / 3.0
        
        return float(avg_psi)
    
    def compute_ece(self, n_bins: int = 10) -> Optional[float]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures how well confidence scores match actual accuracy.
        A well-calibrated model should have 60% accuracy when it predicts with 60% confidence.
        
        Lower is better. ECE > 0.15 indicates poor calibration.
        """
        if len(self.prediction_probs) < 30 or len(self.prediction_outcomes) < 30:
            return None
        
        # Use only predictions where we have outcomes
        n_with_outcomes = min(len(self.prediction_probs), len(self.prediction_outcomes))
        probs = np.array(self.prediction_probs[-n_with_outcomes:])
        outcomes = np.array(self.prediction_outcomes[-n_with_outcomes:])
        
        # Get max confidence and predicted class
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        
        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)
        
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = (predictions[in_bin] == outcomes[in_bin]).mean()
                bin_confidence = confidences[in_bin].mean()
                bin_size = in_bin.sum()
                
                ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def get_drift_report(self) -> Dict:
        """Generate a comprehensive drift report."""
        psi = self.compute_psi()
        ece = self.compute_ece()
        
        # Compute entropy trend
        if len(self.prediction_probs) >= 20:
            recent_probs = np.array(self.prediction_probs[-20:])
            entropies = -np.sum(recent_probs * np.log(recent_probs + 1e-10), axis=1)
            avg_entropy = float(entropies.mean())
            max_entropy = float(np.log(3))  # Max for 3 classes
            entropy_ratio = avg_entropy / max_entropy
        else:
            avg_entropy = None
            entropy_ratio = None
        
        # Check for alerts
        alerts = []
        if psi is not None and psi >= self.alert_threshold_psi:
            alerts.append({
                "type": "PSI_DRIFT",
                "message": f"Significant prediction distribution shift detected (PSI={psi:.3f})",
                "severity": "WARNING" if psi < 0.5 else "CRITICAL"
            })
        
        if ece is not None and ece >= self.alert_threshold_ece:
            alerts.append({
                "type": "CALIBRATION_DEGRADED",
                "message": f"Model calibration has degraded (ECE={ece:.3f})",
                "severity": "WARNING"
            })
        
        if entropy_ratio is not None and entropy_ratio > 0.9:
            alerts.append({
                "type": "CONFIDENCE_COLLAPSE",
                "message": f"Model predicting near-uniform distribution (entropy ratio={entropy_ratio:.2f})",
                "severity": "CRITICAL"
            })
        
        return {
            "psi": psi,
            "psi_threshold": self.alert_threshold_psi,
            "psi_status": "OK" if psi is None or psi < self.alert_threshold_psi else "DRIFT_DETECTED",
            "ece": ece,
            "ece_threshold": self.alert_threshold_ece,
            "ece_status": "OK" if ece is None or ece < self.alert_threshold_ece else "CALIBRATION_ISSUE",
            "avg_entropy": avg_entropy,
            "entropy_ratio": entropy_ratio,
            "n_predictions_tracked": len(self.prediction_probs),
            "n_outcomes_tracked": len(self.prediction_outcomes),
            "baseline_set": self.baseline_set,
            "alerts": alerts,
            "overall_status": "HEALTHY" if not alerts else "ISSUES_DETECTED",
            "timestamp": datetime.now().isoformat()
        }


# Global drift monitor instance
drift_monitor = PredictionDriftMonitor(window_size=100)

app = FastAPI(title="BTC Trading GPU Trainer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    # Mapping from checkpoint filename patterns to standardized model types
    MODEL_TYPE_PATTERNS = {
        # Multi-head models (check first - have forward_multihead for quantile predictions)
        # Includes both new (best_*_multihead) and legacy (*_multihead_trained) naming patterns
        "multihead_transformer": ["transformer_multihead_trained", "multihead_transformer", "transformer_multihead", "best_transformer_multihead", "best_multihead_transformer"],
        "multihead_tft": ["tft_multihead_trained", "multihead_tft", "tft_multihead", "best_tft_multihead", "best_multihead_tft"],
        "multihead_lstm": ["lstm_multihead_trained", "multihead_lstm", "lstm_multihead", "best_lstm_multihead", "best_multihead_lstm"],
        "multihead_cnn": ["cnn_multihead_trained", "multihead_cnn", "cnn_multihead", "best_cnn_multihead", "best_multihead_cnn"],
        "multihead_gnn": ["gnn_multihead_trained", "multihead_gnn", "gnn_multihead", "best_gnn_multihead", "best_multihead_gnn"],
        "multihead_vae": ["vae_multihead_trained", "multihead_vae", "vae_multihead", "best_vae_multihead", "best_multihead_vae"],
        # Legacy classification-only models
        "transformer": ["transformer_trained", "transformer_price", "transformer", "best_transformer"],
        "tft": ["tft_trained", "temporal_fusion_transformer", "tft", "best_temporal_fusion", "best_tft"],
        "lstm": ["lstm_trained", "bidirectional_lstm", "lstm", "stacked_lstm", "conv_lstm", "best_lstm", "best_bidirectional"],
        "cnn": ["cnn_trained", "resnet_price", "resnet", "cnn", "inception", "wavenet", "best_resnet", "best_cnn"],
        "vae": ["vae_trained", "market_vae", "vae", "conditional_vae", "best_vae", "best_market_vae"],
        "gnn": ["gnn_trained", "cross_asset_gnn", "temporal_gnn", "gnn", "best_gnn", "best_cross_asset"],
    }
    
    # STF (Single-TimeFrame) feature names - 47 features from compute_technical_features
    STF_FEATURE_NAMES = [
        "returns", "log_returns",
        "sma_5", "ema_5", "std_5", "return_5",
        "sma_10", "ema_10", "std_10", "return_10",
        "sma_20", "ema_20", "std_20", "return_20",
        "sma_50", "ema_50", "std_50", "return_50",
        "sma_100", "ema_100", "std_100", "return_100",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
        "atr_14", "atr_7",
        "volume_sma_20", "volume_ratio",
        "adx_14",
        "stoch_k", "stoch_d",
        "obv", "obv_sma",
        "rsi_divergence",
        "vol_weighted_mom_5", "vol_weighted_mom_10",
        "vwap_deviation",
        "close_to_high_ratio",
        "volume_delta"
    ]
    STF_FEATURE_COUNT = 47
    MTF_FEATURE_COUNT = 66
    
    def __init__(self):
        self.models = {}
        self.model_instances = {}
        self.model_type_map = {}  # Maps checkpoint name -> standardized type (transformer, tft, etc.)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensemble = None
        self.scaler = None  # Dict of per-column scalers (NOT a single sklearn scaler)
        self.scaler_columns = None  # Column names for the scaler dict
        self.feature_config = None  # Feature configuration
        self.training_mode = "STF"  # Default to STF (15m only) - safer for 15m-trained models
        self.stf_serving_enabled = True  # Set to False if MTF config detected for STF deployment
        self.feature_engineer_version = None  # Version string from FeatureEngineer
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_model": None,
            "progress": 0.0,
            "metrics": {},
            # Enhanced training progress tracking
            "epoch_history": [],  # List of per-epoch metrics
            "start_time": None,  # Training start timestamp
            "eta_seconds": None,  # Estimated time remaining
            "health_warnings": [],  # Warnings from TrainingHealthMonitor
            "last_update": None,  # Last status update timestamp
            "per_head_losses": {},  # Per-head loss values
            "learning_rate": None,  # Current learning rate
            "best_val_loss": None,  # Best validation loss so far
            "early_stop_counter": 0,  # Epochs since last improvement
            # Live prediction distribution for GUI display
            "prediction_distribution": {"short": 0, "hold": 0, "long": 0, "total": 0},
            "gradient_norm": None
        }
        self.prediction_history = []
        # Primary checkpoint directory (new location)
        self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        # Secondary checkpoint directory (legacy location from GUI training)
        self.saved_models_dir = Path(__file__).parent.parent / "saved_models"
        self.scaler_path = self.checkpoint_dir / "scaler.joblib"
        # FIX #4: Use locked sequence length - NEVER dynamically resize
        self.sequence_length = SEQUENCE_LENGTH_LOCKED
        # Default to STF feature count (41) since most models are 15m-only trained
        self.input_dim = self.STF_FEATURE_COUNT
        self.instantiation_errors: Dict[str, str] = {}  # Track errors for /models/status
    
    def _map_filename_to_model_type(self, filename: str) -> str:
        """Map checkpoint filename to standardized model type.
        
        Examples:
            best_transformer_price -> transformer
            best_temporal_fusion_transformer -> tft
            best_bidirectional_lstm -> lstm
            best_resnet_price -> cnn
            best_market_vae -> vae
            best_cross_asset_gnn -> gnn
        """
        filename_lower = filename.lower()
        
        for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return model_type
        
        # If no pattern matched, return the filename as-is
        return filename
    
    def _config_to_dict(self, config) -> dict:
        """Convert a Config object (dataclass/object) to a dictionary.
        
        Handles both dict and object-style configs from checkpoints.
        Some checkpoints save config as a dataclass/object, others as dict.
        """
        if config is None:
            return {}
        
        # Already a dict
        if isinstance(config, dict):
            return config
        
        # Try to convert object to dict
        try:
            # Try vars() for regular objects
            return vars(config)
        except TypeError:
            pass
        
        try:
            # Try __dict__ directly
            if hasattr(config, '__dict__'):
                return config.__dict__
        except Exception:
            pass
        
        try:
            # Try dataclass asdict
            from dataclasses import asdict, is_dataclass
            if is_dataclass(config):
                return asdict(config)
        except Exception:
            pass
        
        try:
            # Try accessing common attributes manually
            result = {}
            common_attrs = ['input_dim', 'output_dim', 'hidden_dim', 'sequence_length', 
                           'dropout', 'd_model', 'nhead', 'num_layers', 'num_encoder_layers',
                           'model_type', 'latent_dim', 'hidden_dims', 'num_assets', 
                           'base_channels', 'num_blocks', 'kernel_size', 'use_attention']
            for attr in common_attrs:
                if hasattr(config, attr):
                    result[attr] = getattr(config, attr)
            return result
        except Exception:
            pass
        
        # Fallback: return empty dict
        logger.warning(f"Could not convert config of type {type(config)} to dict")
        return {}
    
    def transform_features(self, features_df) -> np.ndarray:
        """Transform features using the loaded scaler dict.
        
        The scaler is a dict of per-column sklearn scalers, NOT a single scaler.
        This matches how FeatureEngineer.save_scalers/load_scalers works.
        
        CRITICAL: Also enforces feature ordering via FeatureValidator.enforce_schema
        to prevent silent prediction errors from column reordering.
        """
        if self.scaler is None:
            logger.warning("No scaler loaded - returning raw features")
            return features_df.values.astype(np.float32)
        
        # Apply per-column scaling using the scaler dict
        transformed = features_df.copy()
        for col in features_df.columns:
            if col in self.scaler:
                valid_mask = ~features_df[col].isna()
                if valid_mask.any():
                    try:
                        transformed.loc[valid_mask, col] = self.scaler[col].transform(
                            features_df.loc[valid_mask, col].values.reshape(-1, 1)
                        ).flatten()
                    except Exception as e:
                        logger.warning(f"Failed to scale column {col}: {e}")
        
        raw_features = transformed.values.astype(np.float32)
        
        # === ENFORCE FEATURE ORDERING via FeatureValidator ===
        # This ensures features are in the correct order expected by the model
        if hasattr(self, 'feature_config') and self.feature_config is not None:
            try:
                from training.feature_registry import FeatureValidator
                validator = FeatureValidator(self.feature_config)
                
                # Get column names from transformed dataframe
                feature_names = list(transformed.columns)
                
                # Enforce schema - reorders, fills missing, drops extras
                enforced_features, stats = validator.enforce_schema(
                    feature_names=feature_names,
                    features=raw_features,
                    fill_value=0.0,
                    max_missing_pct=0.15
                )
                
                if stats.get('missing_count', 0) > 0:
                    logger.warning(f"[SCHEMA] Filled {stats['missing_count']} missing features")
                if stats.get('extra_count', 0) > 0:
                    logger.info(f"[SCHEMA] Dropped {stats['extra_count']} extra features")
                
                logger.info(f"[SCHEMA] Enforced: {stats.get('incoming_features', 0)} -> {stats.get('expected_features', 0)} features")
                return enforced_features
                
            except Exception as e:
                logger.warning(f"[SCHEMA] Could not enforce schema, using raw order: {e}")
                return raw_features
        
        return raw_features
        
    def _create_model_instance(self, model_type: str, config: dict):
        """Create model instance with correct constructor args for each model type.
        
        Covers all model classes from gpu_trainer/models/:
        - multihead.py: MultiHeadTransformer, MultiHeadLSTM, MultiHeadCNN, MultiHeadGNN, MultiHeadVAE
        - transformer.py: TransformerPriceModel, TemporalFusionTransformer
        - lstm.py: BidirectionalLSTM, StackedLSTM, ConvLSTM
        - cnn.py: ResNetPrice, InceptionNet, WaveNet
        - vae.py: MarketVAE, ConditionalVAE
        - gnn.py: CrossAssetGNN, TemporalGNN
        - ensemble.py: MetaLearner, AttentionEnsemble, MasterEnsemble
        """
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            model_type_lower = model_type.lower()
            input_dim = config.get("input_dim", 66)
            output_dim = config.get("output_dim", 3)
            hidden_dim = config.get("hidden_dim", 128)
            sequence_length = config.get("sequence_length", 100)
            dropout = config.get("dropout", 0.2)
            
            # === MULTI-HEAD MODELS (check first - has forward_multihead for quantile predictions) ===
            # Check for multi-head model indicators from state_dict
            is_multihead = config.get("is_multihead", False)
            
            # If multihead detected from state_dict but not in name, force multihead model selection
            if is_multihead and "multihead" not in model_type_lower:
                # Determine base model type and redirect to multi-head variant
                if "transformer" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadTransformer (detected from state_dict)")
                    model_type_lower = "multihead_transformer"
                elif "lstm" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadLSTM (detected from state_dict)")
                    model_type_lower = "multihead_lstm"
                elif "cnn" in model_type_lower or "resnet" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadCNN (detected from state_dict)")
                    model_type_lower = "multihead_cnn"
                elif "gnn" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadGNN (detected from state_dict)")
                    model_type_lower = "multihead_gnn"
                elif "vae" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadVAE (detected from state_dict)")
                    model_type_lower = "multihead_vae"
                elif "tft" in model_type_lower:
                    logger.info(f"Redirecting {model_type} to MultiHeadTFT (detected from state_dict)")
                    model_type_lower = "multihead_tft"
            
            # Patterns: multihead_transformer OR transformer_multihead (both naming conventions)
            if "multihead_transformer" in model_type_lower or "transformer_multihead" in model_type_lower:
                from models.multihead import MultiHeadTransformer
                return MultiHeadTransformer(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_layers=config.get("num_layers", 6),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_lstm" in model_type_lower or "lstm_multihead" in model_type_lower:
                from models.multihead import MultiHeadLSTM
                return MultiHeadLSTM(
                    input_dim=input_dim,
                    hidden_dim=config.get("hidden_dim", 256),
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_cnn" in model_type_lower or "cnn_multihead" in model_type_lower:
                from models.multihead import MultiHeadCNN
                return MultiHeadCNN(
                    input_dim=input_dim,
                    hidden_channels=config.get("hidden_channels", 256),
                    num_blocks=config.get("num_blocks", 4),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_gnn" in model_type_lower or "gnn_multihead" in model_type_lower:
                from models.multihead import MultiHeadGNN
                return MultiHeadGNN(
                    input_dim=input_dim,
                    hidden_dim=config.get("hidden_dim", 128),
                    num_layers=config.get("num_layers", 3),
                    num_heads=config.get("num_heads", 4),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_vae" in model_type_lower or "vae_multihead" in model_type_lower:
                from models.multihead import MultiHeadVAE
                return MultiHeadVAE(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_tft" in model_type_lower or "tft_multihead" in model_type_lower:
                from models.multihead import MultiHeadTFT
                return MultiHeadTFT(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_encoder_layers=config.get("num_encoder_layers", 4),
                    dropout=dropout,
                    num_classes=output_dim
                )
            
            # === TRANSFORMER MODELS ===
            if "temporal_fusion" in model_type_lower or "tft" in model_type_lower:
                from models.transformer import TemporalFusionTransformer
                return TemporalFusionTransformer(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_encoder_layers=config.get("num_encoder_layers", 4),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "transformer" in model_type_lower:
                from models.transformer import TransformerPriceModel
                return TransformerPriceModel(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_layers=config.get("num_layers", 6),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === LSTM MODELS ===
            elif "conv_lstm" in model_type_lower or "convlstm" in model_type_lower:
                from models.lstm import ConvLSTM
                return ConvLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 2),
                    kernel_size=config.get("kernel_size", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "stacked_lstm" in model_type_lower or "stackedlstm" in model_type_lower:
                from models.lstm import StackedLSTM
                return StackedLSTM(
                    input_dim=input_dim,
                    hidden_dims=config.get("hidden_dims", [256, 128, 64]),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "lstm" in model_type_lower or "bidirectional" in model_type_lower:
                from models.lstm import BidirectionalLSTM
                return BidirectionalLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    output_dim=output_dim,
                    use_attention=config.get("use_attention", True)
                )
                
            # === CNN MODELS ===
            elif "wavenet" in model_type_lower:
                from models.cnn import WaveNet
                return WaveNet(
                    input_dim=input_dim,
                    residual_channels=config.get("residual_channels", 64),
                    dilation_channels=config.get("dilation_channels", 64),
                    skip_channels=config.get("skip_channels", 128),
                    num_blocks=config.get("num_blocks", 4),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "inception" in model_type_lower:
                from models.cnn import InceptionNet
                return InceptionNet(
                    input_dim=input_dim,
                    base_channels=config.get("base_channels", 64),
                    num_inception_blocks=config.get("num_inception_blocks", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "resnet" in model_type_lower or "cnn" in model_type_lower:
                from models.cnn import ResNetPrice
                # ResNetPrice uses channels: List[int], not base_channels
                channels = config.get("channels", [64, 128, 256, 512])
                return ResNetPrice(
                    input_dim=input_dim,
                    channels=channels,
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === VAE MODELS ===
            elif "conditional_vae" in model_type_lower or "cvae" in model_type_lower:
                from models.vae import ConditionalVAE
                return ConditionalVAE(
                    input_dim=input_dim,
                    condition_dim=config.get("condition_dim", 16),
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    hidden_dims=config.get("hidden_dims", [128, 256]),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "vae" in model_type_lower or "market_vae" in model_type_lower:
                from models.vae import MarketVAE
                return MarketVAE(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    hidden_dims=config.get("hidden_dims", [128, 256, 512]),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === GNN MODELS ===
            elif "cross_asset" in model_type_lower or "crossasset" in model_type_lower:
                from models.gnn import CrossAssetGNN
                return CrossAssetGNN(
                    input_dim=input_dim,
                    num_assets=config.get("num_assets", 4),
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "gnn" in model_type_lower or "temporal_gnn" in model_type_lower:
                from models.gnn import TemporalGNN
                return TemporalGNN(
                    input_dim=input_dim,
                    num_nodes=config.get("num_nodes", 4),
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    num_heads=config.get("num_heads", 4),
                    temporal_window=config.get("temporal_window", 10),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === ENSEMBLE MODELS ===
            elif "master_ensemble" in model_type_lower or "masterensemble" in model_type_lower:
                from models.ensemble import MasterEnsemble
                return MasterEnsemble(
                    model_configs=config.get("model_configs", [{"name": "default"}]),
                    feature_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "attention_ensemble" in model_type_lower:
                from models.ensemble import AttentionEnsemble
                return AttentionEnsemble(
                    num_models=config.get("num_models", 3),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "meta_learner" in model_type_lower or "metalearner" in model_type_lower:
                from models.ensemble import MetaLearner
                return MetaLearner(
                    num_base_models=config.get("num_base_models", 3),
                    base_hidden_dim=hidden_dim,
                    meta_hidden_dim=config.get("meta_hidden_dim", 64),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "ensemble" in model_type_lower:
                # Generic ensemble fallback
                from models.ensemble import MasterEnsemble
                return MasterEnsemble(
                    model_configs=config.get("model_configs", [{"name": "default"}]),
                    feature_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
                self.instantiation_errors[model_type] = f"Unknown model type: {model_type}"
                return None
                
        except ImportError as e:
            error_msg = f"Failed to import model class for {model_type}: {e}"
            logger.error(error_msg)
            self.instantiation_errors[model_type] = error_msg
            return None
        except Exception as e:
            error_msg = f"Failed to create model instance for {model_type}: {e}"
            logger.error(error_msg)
            self.instantiation_errors[model_type] = error_msg
            return None
    
    def _infer_dims_from_state_dict(self, state_dict: dict, model_type: str) -> dict:
        """Infer model dimensions from state_dict weight shapes.
        
        This is critical for loading models when input_dim/hidden_dim weren't
        saved in the checkpoint config (which is the common case since they're
        computed at training time from data shape).
        
        Matches actual parameter names from gpu_trainer/models/*.py implementations.
        """
        inferred = {}
        model_type_lower = model_type.lower()
        
        def get_param_shape(key):
            """Get shape from state_dict, handling both Tensor and array-like objects."""
            if key in state_dict:
                param = state_dict[key]
                return tuple(param.shape) if hasattr(param, 'shape') else None
            return None
        
        try:
            # === LSTM MODELS ===
            # BidirectionalLSTM: input_bn.weight [input_dim], lstm.weight_ih_l0 [4*hidden_dim, input_dim]
            if "lstm" in model_type_lower or "bidirectional" in model_type_lower:
                shape = get_param_shape("input_bn.weight")
                if shape:
                    inferred["input_dim"] = shape[0]
                    
                shape = get_param_shape("lstm.weight_ih_l0")
                if shape:
                    inferred["hidden_dim"] = shape[0] // 4  # LSTM has 4 gates
                    if "input_dim" not in inferred:
                        inferred["input_dim"] = shape[1]
                        
            # === TRANSFORMER MODELS ===
            # TransformerPriceModel: input_projection.weight [d_model, input_dim]
            elif "transformer" in model_type_lower and "tft" not in model_type_lower and "temporal_fusion" not in model_type_lower:
                shape = get_param_shape("input_projection.weight")
                if shape:
                    inferred["d_model"] = shape[0]
                    inferred["input_dim"] = shape[1]
                    
            # TemporalFusionTransformer: 
            #   static_encoder.0.weight [d_model, input_dim] - use this for d_model
            #   temporal_encoder is LSTM which uses d_model/2 per direction
            elif "tft" in model_type_lower or "temporal_fusion" in model_type_lower:
                # Use static_encoder for d_model (more reliable than LSTM)
                static_shape = get_param_shape("static_encoder.0.weight")
                if static_shape:
                    inferred["d_model"] = static_shape[0]
                    inferred["input_dim"] = static_shape[1]
                else:
                    # Fallback to temporal_encoder if static_encoder not found
                    shape = get_param_shape("temporal_encoder.weight_ih_l0")
                    if shape:
                        inferred["input_dim"] = shape[1]
                        # LSTM hidden is d_model/2 per direction (bidirectional), so hidden*2 = d_model
                        # But weight_ih has shape [4*hidden, input_dim], so d_model = shape[0] // 4 * 2 = shape[0] // 2
                        inferred["d_model"] = shape[0] // 2
                    
            # === CNN MODELS ===
            # ResNetPrice: input_conv.0.weight [channels[0], input_dim, kernel_size]
            # MultiHeadCNN: input_conv.weight [hidden_channels, input_dim, kernel_size]
            elif "resnet" in model_type_lower or "cnn" in model_type_lower or "inception" in model_type_lower:
                # Try MultiHeadCNN format first (more common after migration)
                shape = get_param_shape("input_conv.weight")
                if shape:
                    inferred["input_dim"] = shape[1]  # Conv1d: [out_channels, in_channels, kernel]
                    inferred["hidden_channels"] = shape[0]
                    logger.info(f"CNN: inferred input_dim={shape[1]} from input_conv.weight")
                else:
                    # Fallback to ResNetPrice format
                    shape = get_param_shape("input_conv.0.weight")
                    if shape:
                        inferred["input_dim"] = shape[1]  # Conv1d: [out_channels, in_channels, kernel]
                        first_channels = shape[0]
                        # For ResNetPrice, infer the full channels list from the residual blocks
                        inferred["channels"] = [first_channels, first_channels*2, first_channels*4, first_channels*8]
                        logger.info(f"CNN: inferred input_dim={shape[1]} from input_conv.0.weight")
                    
            # WaveNet: input_conv.weight [residual_channels, input_dim, 1]
            elif "wavenet" in model_type_lower:
                shape = get_param_shape("input_conv.weight")
                if shape:
                    inferred["input_dim"] = shape[1]
                    inferred["residual_channels"] = shape[0]
                    
            # === VAE MODELS ===
            # MarketVAE: encoder has structure [Linear, BatchNorm, LeakyReLU, Dropout] x N
            # encoder.0.weight [hidden_dims[0], input_dim * sequence_length]
            # encoder.4.weight [hidden_dims[1], hidden_dims[0]]
            # encoder.8.weight [hidden_dims[2], hidden_dims[1]]
            # fc_mu.weight [latent_dim, hidden_dims[-1]]
            elif "vae" in model_type_lower:
                # Find input_dim and sequence_length from first encoder layer
                shape = get_param_shape("encoder.0.weight")
                if shape:
                    total_input = shape[1]
                    first_hidden = shape[0]
                    for seq_len in [100, 50, 60, 120, 80]:
                        if total_input % seq_len == 0:
                            inferred["input_dim"] = total_input // seq_len
                            inferred["sequence_length"] = seq_len
                            break
                    
                    # Scan all encoder layers to build hidden_dims list
                    # Each block is 4 layers (Linear, BatchNorm, LeakyReLU, Dropout)
                    hidden_dims = [first_hidden]
                    layer_idx = 4  # Start at second block
                    while True:
                        layer_shape = get_param_shape(f"encoder.{layer_idx}.weight")
                        if layer_shape and len(layer_shape) == 2:  # Linear layer
                            hidden_dims.append(layer_shape[0])
                            layer_idx += 4
                        else:
                            break
                    inferred["hidden_dims"] = hidden_dims
                    
                # Get latent_dim from fc_mu
                fc_mu_shape = get_param_shape("fc_mu.weight")
                if fc_mu_shape:
                    inferred["latent_dim"] = fc_mu_shape[0]
                    
            # === GNN MODELS ===
            # CrossAssetGNN: 
            #   temporal_encoder.0.weight [hidden_dim, input_dim] - uses full input
            #   node_encoder.0.weight [hidden_dim, input_dim // num_assets] - per-asset features
            elif "cross_asset" in model_type_lower or "crossasset" in model_type_lower:
                # Use temporal_encoder for full input_dim (not node_encoder which uses per-asset)
                temporal_shape = get_param_shape("temporal_encoder.0.weight")
                node_shape = get_param_shape("node_encoder.0.weight")
                
                if temporal_shape:
                    inferred["hidden_dim"] = temporal_shape[0]
                    inferred["input_dim"] = temporal_shape[1]
                    
                    # Calculate num_assets from the ratio
                    if node_shape:
                        features_per_asset = node_shape[1]
                        if features_per_asset > 0:
                            num_assets = inferred["input_dim"] // features_per_asset
                            if num_assets >= 1:
                                inferred["num_assets"] = num_assets
                elif node_shape:
                    # Fallback if temporal_encoder not found
                    inferred["hidden_dim"] = node_shape[0]
                    inferred["input_dim"] = node_shape[1]
                    
            # TemporalGNN/MultiHeadGNN: 
            #   temporal_encoder.0.weight [hidden_dim, input_dim] - MultiHeadGNN
            #   spatial_encoder.weight [hidden_dim, features_per_node] - TemporalGNN
            elif "gnn" in model_type_lower or "temporal_gnn" in model_type_lower:
                # Try MultiHeadGNN pattern first (temporal_encoder.0.weight)
                shape = get_param_shape("temporal_encoder.0.weight")
                if shape:
                    inferred["hidden_dim"] = shape[0]
                    inferred["input_dim"] = shape[1]
                    logger.info(f"GNN: Inferred from temporal_encoder.0.weight: {shape}")
                else:
                    # Fall back to TemporalGNN pattern (spatial_encoder.weight)
                    shape = get_param_shape("spatial_encoder.weight")
                    if shape:
                        inferred["hidden_dim"] = shape[0]
                        inferred["input_dim"] = shape[1]
                    
            if inferred:
                logger.info(f"Inferred dimensions for {model_type}: {inferred}")
            else:
                # Log available keys for debugging
                sample_keys = list(state_dict.keys())[:10]
                logger.warning(f"Could not infer dims for {model_type}. Sample keys: {sample_keys}")
                
        except Exception as e:
            logger.warning(f"Failed to infer dims from state_dict for {model_type}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
        return inferred
    
    def _instantiate_model(self, checkpoint: dict, model_name: str, checkpoint_path: str = None):
        """Instantiate a model from checkpoint config and state_dict.
        
        CRITICAL: Derives input_dim from checkpoint metadata to prevent schema drift.
        Priority: state_dict weights > .features.json > checkpoint config > default
        """
        raw_config = checkpoint.get("config", {})
        # Convert Config object to dict if needed
        config = self._config_to_dict(raw_config)
        
        # Check for model_config first (saved by updated trainer with input_dim)
        model_config = checkpoint.get("model_config", {})
        if model_config:
            logger.info(f"Found model_config in checkpoint: {model_config}")
            # Model config takes precedence for model-specific params
            for key, value in model_config.items():
                if value is not None:
                    config[key] = value
        
        state_dict = checkpoint.get("model_state_dict")
        
        if not state_dict:
            logger.warning(f"No state_dict in checkpoint for {model_name}")
            return None
        
        # === CRITICAL: Load per-checkpoint feature config (.features.json) ===
        feature_config = None
        if checkpoint_path:
            try:
                from training.feature_registry import load_feature_config_for_checkpoint
                feature_config = load_feature_config_for_checkpoint(checkpoint_path)
                if feature_config:
                    logger.info(f"[SCHEMA] Loaded feature config for {model_name}: input_dim={feature_config.input_dim}, mode={feature_config.mode}")
                    config["input_dim"] = feature_config.input_dim
                    config["sequence_length"] = feature_config.sequence_length
            except Exception as e:
                logger.warning(f"[SCHEMA] Failed to load feature config for {model_name}: {e}")
        
        # Detect multi-head model from state_dict keys (reliable detection method)
        state_keys = list(state_dict.keys())
        has_multihead_keys = any(
            k.startswith("class_head.") or 
            k.startswith("regression_head.") or 
            k.startswith("quantile_head.") or
            k.startswith("trading_head.") or
            k.startswith("candle_head.")
            for k in state_keys
        )
        if has_multihead_keys:
            config["is_multihead"] = True
            logger.info(f"Detected multi-head architecture from state_dict keys for {model_name}")
        
        # Try to determine model type from name, config, or infer from state_dict keys
        model_type = config.get("model_type", "") or config.get("name", "")
        if not model_type:
            # Try to infer from checkpoint name
            model_type = model_name
        
        # Infer dimensions from state_dict (since training doesn't save input_dim/hidden_dim in config)
        inferred_dims = self._infer_dims_from_state_dict(state_dict, model_type)
        
        # Merge inferred dims into config - inferred ALWAYS takes precedence
        # since they come from actual weights and reflect the true model architecture
        for key, value in inferred_dims.items():
            if value is not None:
                config[key] = value
        
        try:
            # Get config values with defaults (now possibly updated by inferred dims)
            input_dim = config.get("input_dim", 66)
            hidden_dim = config.get("hidden_dim", 128)
            d_model = config.get("d_model", 256)
            sequence_length = config.get("sequence_length", 100)
            
            # === CRITICAL ASSERTION: Verify input_dim matches STF expectation ===
            # If inferred from state_dict, the input_dim is authoritative
            # Refuse to serve if computed features would mismatch
            inferred_input_dim = inferred_dims.get("input_dim")
            if inferred_input_dim is not None:
                if inferred_input_dim != input_dim:
                    logger.error(f"[SCHEMA MISMATCH] {model_name}: inferred input_dim={inferred_input_dim} != config input_dim={input_dim}")
                    # Use the inferred dimension (from actual weights)
                    input_dim = inferred_input_dim
                    config["input_dim"] = input_dim
                
                # Check if this is STF (41-47 features) vs MTF (66+ features)
                if inferred_input_dim <= 50:  # STF: 41 (v1) or 47 (v2) features
                    logger.info(f"[SCHEMA] {model_name}: STF checkpoint detected (input_dim={inferred_input_dim})")
                elif inferred_input_dim > 50:  # MTF: 66+ features
                    logger.warning(f"[SCHEMA] {model_name}: MTF checkpoint detected (input_dim={inferred_input_dim}) - may not work with STF-only serving")
            
            logger.info(f"Creating model {model_type} with: input_dim={input_dim}, hidden_dim={hidden_dim}, d_model={d_model}")
            
            # Create model with proper constructor
            model = self._create_model_instance(model_type, config)
            
            if model is None:
                logger.warning(f"Could not create model instance for {model_name}")
                return None
            
            # Load state dict STRICT - refuse to serve on size mismatch
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"[SCHEMA] {model_name}: Strict state_dict load SUCCESS - no size mismatches")
            except RuntimeError as e:
                error_msg = str(e)
                if "size mismatch" in error_msg.lower():
                    # This is a critical schema drift error - DO NOT load with strict=False
                    logger.error(f"[SCHEMA FATAL] {model_name}: Size mismatch - refusing to load model!")
                    logger.error(f"[SCHEMA FATAL] Error: {error_msg}")
                    logger.error(f"[SCHEMA FATAL] This indicates training/inference feature schema drift.")
                    logger.error(f"[SCHEMA FATAL] Checkpoint input_dim: inferred={inferred_input_dim}, expected for STF={self.STF_FEATURE_COUNT}")
                    return None  # DO NOT serve this model
                else:
                    # Non-size-mismatch error - try non-strict as fallback
                    logger.warning(f"Strict load failed for {model_name} (non-size error), trying non-strict: {e}")
                    model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            # Update manager config from loaded model
            self.sequence_length = sequence_length
            self.input_dim = input_dim
            
            logger.info(f"Instantiated model: {model_name} (type={model_type}, input={input_dim}, seq={sequence_length})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to instantiate model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load_best_models(self):
        """Load best checkpoint models at startup.
        
        Searches both checkpoints/ and saved_models/ directories for models.
        """
        # Collect checkpoint files from both directories
        checkpoint_files = []
        
        # Primary: checkpoints/ directory
        if self.checkpoint_dir.exists():
            checkpoint_files.extend(list(self.checkpoint_dir.glob("*.pt")))
            logger.info(f"Found {len(checkpoint_files)} .pt files in checkpoints/")
        else:
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
        
        # Secondary: saved_models/ directory (legacy GUI training location)
        if self.saved_models_dir.exists():
            saved_model_files = list(self.saved_models_dir.glob("*.pt"))
            checkpoint_files.extend(saved_model_files)
            logger.info(f"Found {len(saved_model_files)} .pt files in saved_models/")
        else:
            logger.info(f"saved_models/ directory not found (not an error)")
            
        # Load scaler - search both directories for model-specific scalers
        scaler_loaded = False
        
        # Collect all scaler files from both directories
        all_scaler_files = []
        if self.checkpoint_dir.exists():
            all_scaler_files.extend(list(self.checkpoint_dir.glob("scaler*.joblib")))
            all_scaler_files.extend(list(self.checkpoint_dir.glob("*_scalers.joblib")))
        if self.saved_models_dir.exists():
            all_scaler_files.extend(list(self.saved_models_dir.glob("scaler*.joblib")))
            all_scaler_files.extend(list(self.saved_models_dir.glob("*_scalers.joblib")))
        
        if all_scaler_files:
            # Use the most recently modified scaler
            all_scaler_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            try:
                self.scaler = joblib.load(all_scaler_files[0])
                logger.info(f"Loaded scaler from {all_scaler_files[0]}")
                scaler_loaded = True
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")
        
        # Fallback to generic scaler.joblib
        if not scaler_loaded and self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded scaler from {self.scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")
        
        # Load expected feature list if exists (for alignment verification) - check both dirs
        feature_list_path = self.checkpoint_dir / "feature_columns.txt"
        if not feature_list_path.exists() and self.saved_models_dir.exists():
            feature_list_path = self.saved_models_dir / "feature_columns.txt"
        
        # Also check for feature_config.json
        feature_config_path = self.checkpoint_dir / "feature_config.json"
        if not feature_config_path.exists() and self.saved_models_dir.exists():
            feature_config_path = self.saved_models_dir / "feature_config.json"
        
        # Priority: feature_config.json > feature_columns.txt > STF default
        feature_config_loaded = False
        
        if feature_config_path.exists():
            try:
                from training.feature_registry import FeatureConfig
                self.feature_config = FeatureConfig.load(str(feature_config_path))
                logger.info(f"Loaded FeatureConfig from JSON (hash: {self.feature_config.version_hash}, dim: {self.feature_config.input_dim})")
                self.input_dim = self.feature_config.input_dim
                self.sequence_length = self.feature_config.sequence_length
                self.expected_features = self.feature_config.feature_columns
                feature_config_loaded = True
                
                # Detect training mode from feature count
                if self.input_dim <= self.STF_FEATURE_COUNT + 5:  # Allow some tolerance
                    self.training_mode = "STF"
                    logger.info(f"Detected STF training mode ({self.input_dim} features)")
                else:
                    self.training_mode = "MTF"
                    logger.info(f"Detected MTF training mode ({self.input_dim} features)")
            except Exception as e:
                logger.warning(f"Failed to load feature_config.json: {e}")
        
        if not feature_config_loaded and feature_list_path.exists():
            try:
                with open(feature_list_path, 'r') as f:
                    self.expected_features = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded expected feature list: {len(self.expected_features)} columns")
                self.input_dim = len(self.expected_features)
                
                # Detect training mode from feature count
                if self.input_dim <= self.STF_FEATURE_COUNT + 5:
                    self.training_mode = "STF"
                    logger.info(f"Detected STF training mode ({self.input_dim} features)")
                else:
                    self.training_mode = "MTF"
                    logger.info(f"Detected MTF training mode ({self.input_dim} features)")
                
                # Create FeatureConfig for schema enforcement
                from training.feature_registry import FeatureConfig
                from data.pipeline import FeatureEngineer
                self.feature_config = FeatureConfig(
                    feature_columns=self.expected_features,
                    sequence_length=self.sequence_length,
                    horizon_periods=24,
                    timeframes=["15m"] if self.training_mode == "STF" else ["5m", "15m", "1h", "4h"],
                    input_dim=self.input_dim,
                    feature_engineer_version=FeatureEngineer.VERSION,
                    mode=self.training_mode.lower()
                )
                logger.info(f"Created FeatureConfig for schema enforcement (hash: {self.feature_config.version_hash}, FE version: {FeatureEngineer.VERSION})")
                feature_config_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load feature list: {e}")
                self.expected_features = None
        
        # === CRITICAL: Default to STF if no feature config found ===
        # Detect whether loaded model expects v1 (41) or v2 (47) features
        if not feature_config_loaded:
            loaded_dim = None
            for model_name, model_info in self.models.items():
                if model_info and hasattr(model_info, 'get'):
                    loaded_dim = model_info.get('input_dim')
                    if loaded_dim:
                        break
            
            if loaded_dim and loaded_dim <= 42:
                STF_V1_NAMES = self.STF_FEATURE_NAMES[:41]
                logger.warning(f"No feature_config found - legacy v1 model detected ({loaded_dim} features)")
                self.training_mode = "STF"
                self.input_dim = loaded_dim
                self.expected_features = STF_V1_NAMES
            else:
                logger.warning(f"No feature_config found - defaulting to STF v2 mode ({self.STF_FEATURE_COUNT} features)")
                self.training_mode = "STF"
                self.input_dim = self.STF_FEATURE_COUNT
                self.expected_features = self.STF_FEATURE_NAMES.copy()
            
            # Create STF FeatureConfig
            try:
                from training.feature_registry import FeatureConfig
                from data.pipeline import FeatureEngineer
                self.feature_config = FeatureConfig(
                    feature_columns=self.expected_features,
                    sequence_length=self.sequence_length,
                    horizon_periods=24,
                    timeframes=["15m"],
                    input_dim=self.input_dim,
                    feature_engineer_version=FeatureEngineer.VERSION,
                    mode="stf"
                )
                logger.info(f"Created STF FeatureConfig as default (dim={self.input_dim}, hash: {self.feature_config.version_hash})")
            except Exception as e:
                logger.warning(f"Failed to create STF FeatureConfig: {e}")
        
        logger.info(f"=== Training Mode: {self.training_mode}, Expected Features: {self.input_dim} ===")
        
        # Check if we found any checkpoint files
        if not checkpoint_files:
            logger.warning("No checkpoint files found in checkpoints/ or saved_models/")
            return
        
        # Filter for loadable checkpoints - prioritize best_* and *_trained patterns
        loadable_checkpoints = [
            f for f in checkpoint_files 
            if f.stem.startswith("best_") or "_trained" in f.stem or "_multihead" in f.stem
        ]
        if not loadable_checkpoints:
            # Fallback: use all checkpoints
            loadable_checkpoints = checkpoint_files
            logger.info("No best_* or *_trained checkpoints found, loading all .pt files")
        
        logger.info(f"Loading {len(loadable_checkpoints)} checkpoint(s)...")
            
        for ckpt_path in loadable_checkpoints:
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model_name = ckpt_path.stem
                
                # Map filename to standardized model type
                model_type = self._map_filename_to_model_type(model_name)
                self.model_type_map[model_name] = model_type
                
                # Convert Config object to dict if needed
                raw_config = checkpoint.get("config", {})
                config_dict = self._config_to_dict(raw_config)
                
                # Store checkpoint metadata
                self.models[model_name] = {
                    "path": str(ckpt_path),
                    "accuracy": checkpoint.get("val_accuracy", 0),
                    "epoch": checkpoint.get("epoch", 0),
                    "config": config_dict,
                    "parameters": checkpoint.get("parameters", 0),
                    "model_type": model_type  # Standardized type for dashboard
                }
                
                # Try to instantiate model if state_dict present
                if "model_state_dict" in checkpoint:
                    model_instance = self._instantiate_model(checkpoint, model_name, checkpoint_path=str(ckpt_path))
                    if model_instance is not None:
                        self.model_instances[model_name] = model_instance
                        self.models[model_name]["loaded"] = True
                        logger.info(f"Loaded & instantiated: {model_name} -> {model_type} (acc={checkpoint.get('val_accuracy', 0):.2f}%)")
                    else:
                        self.models[model_name]["state_dict"] = checkpoint["model_state_dict"]
                        self.models[model_name]["loaded"] = False
                        logger.warning(f"Loaded checkpoint but failed to instantiate: {model_name} -> {model_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
                
        logger.info(f"Loaded {len(self.models)} checkpoints, {len(self.model_instances)} instantiated")
        logger.info(f"Model type mapping: {self.model_type_map}")
        
        # === FLOW FORECAST CAPABILITY CHECK ===
        # Assert that loaded models have vol_state_head and acceleration_head
        self.flow_forecast_capable = False
        self.model_capabilities = {}
        
        for model_name, model_instance in self.model_instances.items():
            has_vol_state = hasattr(model_instance, 'vol_state_head')
            has_accel = hasattr(model_instance, 'acceleration_head')
            has_quantile = hasattr(model_instance, 'quantile_head')
            
            self.model_capabilities[model_name] = {
                "vol_state_head": has_vol_state,
                "acceleration_head": has_accel,
                "quantile_head": has_quantile,
                "flow_forecast_ready": has_vol_state and has_accel and has_quantile
            }
            
            if has_vol_state and has_accel:
                self.flow_forecast_capable = True
                logger.info(f"✓ FLOW FORECAST: {model_name} has vol_state_head + acceleration_head")
            else:
                missing = []
                if not has_vol_state:
                    missing.append("vol_state_head")
                if not has_accel:
                    missing.append("acceleration_head")
                if not has_quantile:
                    missing.append("quantile_head")
                logger.warning(f"✗ FLOW FORECAST: {model_name} MISSING: {missing} (old checkpoint?)")
        
        if not self.flow_forecast_capable:
            logger.critical("=" * 60)
            logger.critical("FLOW FORECAST UNAVAILABLE: No models have vol_state + accel heads")
            logger.critical("This means checkpoints are from OLD training without flow forecast")
            logger.critical("FIX: Retrain models with multihead_trainer to add flow forecast heads")
            logger.critical("=" * 60)
    
    def get_model_status_by_type(self) -> Dict[str, Dict]:
        """Get model status organized by standardized model type for dashboard display."""
        status = {}
        
        # Initialize all 6 model types as pending
        for model_type in ["transformer", "tft", "lstm", "cnn", "vae", "gnn"]:
            status[model_type] = {
                "status": "pending",
                "accuracy": None,
                "loss": None,
                "epochs": 0,
                "best_epoch": 0,
                "checkpoint_name": None
            }
        
        # Update status from loaded models
        for model_name, model_info in self.models.items():
            model_type = model_info.get("model_type", self._map_filename_to_model_type(model_name))
            if model_type in status:
                is_instantiated = model_name in self.model_instances
                status[model_type] = {
                    "status": "complete" if is_instantiated else "loaded",
                    "accuracy": model_info.get("accuracy", 0),
                    "loss": model_info.get("config", {}).get("val_loss", None),
                    "epochs": model_info.get("epoch", 0),
                    "best_epoch": model_info.get("epoch", 0),
                    "checkpoint_name": model_name
                }
        
        return status
        
    def load_model(self, model_name: str, path: str, model_class=None):
        try:
            if model_class is not None:
                model = model_class
                model.load(path, self.device)
                model.to(self.device)
                model.eval()
                self.model_instances[model_name] = model
                logger.info(f"Loaded model instance: {model_name}")
            else:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.models[model_name] = checkpoint
                logger.info(f"Loaded model checkpoint: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
    def get_model(self, model_name: str):
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        return self.models.get(model_name)
    
    def get_multihead_model(self):
        """Get the first available multi-head model instance.
        
        Returns:
            Model instance with forward_multihead() method, or None if not available.
        """
        for name, model in self.model_instances.items():
            if hasattr(model, 'forward_multihead'):
                return model
        return None
    
    def predict(self, features: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Make prediction with correct label mapping.
        
        CRITICAL: Training labels are 0=SHORT, 1=NEUTRAL, 2=LONG
        
        If feature_names are provided and feature_config exists, schema enforcement
        will reindex, fill missing, and drop extras to match the model's expected schema.
        """
        if not self.model_instances:
            return self._default_prediction()
        
        # === SCHEMA ENFORCEMENT ===
        # If we have feature_config and feature_names, enforce schema instead of blocking
        if feature_names is not None and self.feature_config is not None:
            from training.feature_registry import FeatureValidator
            validator = FeatureValidator(self.feature_config)
            features, schema_stats = validator.enforce_schema(
                feature_names=feature_names,
                features=features,
                fill_value=0.0
            )
            logger.info(f"[predict] Schema enforced: {schema_stats['incoming_features']} -> {schema_stats['expected_features']} features")
        else:
            # Legacy: Feature count validation - strict mode fails on mismatch
            input_features = features.shape[-1] if len(features.shape) >= 2 else features.shape[0]
            if hasattr(self, 'expected_features') and self.expected_features:
                expected_count = len(self.expected_features)
                if input_features != expected_count:
                    error_msg = f"Feature count mismatch: received {input_features}, expected {expected_count}"
                    logger.error(error_msg)
                    # Return error prediction instead of potentially wrong prediction
                    return {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "probabilities": {"LONG": 0.33, "SHORT": 0.33, "HOLD": 0.34},
                        "error": error_msg,
                        "expected_features": expected_count,
                        "received_features": input_features
                    }
            
        predictions = []
        for name, model in self.model_instances.items():
            try:
                model.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    output = model(x)
                    probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
                    predictions.append({
                        "model": name,
                        "probs": probs.tolist()
                    })
            except Exception as e:
                logger.error(f"Prediction error for {name}: {e}")
                
        if not predictions:
            return self._default_prediction()
            
        avg_probs = np.mean([p["probs"] for p in predictions], axis=0)
        action_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[action_idx])
        
        probs_std = np.std([p["probs"] for p in predictions], axis=0)
        uncertainty = float(np.mean(probs_std))
        
        # CORRECT mapping: 0=SHORT, 1=HOLD, 2=LONG (matches training labels)
        return {
            "action": action_idx,
            "action_name": ACTION_NAMES[action_idx],  # SHORT, HOLD, or LONG
            "probabilities": avg_probs.tolist(),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model_weights": {p["model"]: 1.0 / len(predictions) for p in predictions},
            "reasoning": [f"{p['model']}: {ACTION_NAMES[np.argmax(p['probs'])]}" for p in predictions]
        }
        
    def _default_prediction(self) -> Dict:
        """Default prediction when no models loaded - returns HOLD (index 1)."""
        return {
            "action": 1,  # HOLD is index 1 in training labels
            "action_name": "HOLD",
            "probabilities": [0.2, 0.6, 0.2],  # [SHORT, HOLD, LONG]
            "confidence": 0.3,
            "uncertainty": 0.5,
            "model_weights": {},
            "reasoning": ["No models loaded - defaulting to HOLD"]
        }
    
    def predict_multihead(self, features: np.ndarray, feature_names: List[str] = None) -> Optional[Dict]:
        """Make prediction using multi-head model with learned quantiles.
        
        Returns None if no multi-head model is available.
        
        Multi-head models output:
        - Classification: direction probabilities
        - Regression: expected return (mu) and uncertainty (sigma)
        - Quantiles: q10, q25, q50, q75, q90 for SL/TP derivation
        
        Accepts inputs of shape:
        - [seq_len, features] -> single sample
        - [batch, seq_len, features] -> batch of samples
        
        Schema Enforcement (production-grade):
        - If feature_names provided and feature_config exists, reindex to expected schema
        - Missing features filled with 0.0, extra features dropped
        - Sequence length enforced via padding/trimming
        """
        # Check for multi-head model instances
        multihead_model = None
        for name, model in self.model_instances.items():
            if hasattr(model, 'forward_multihead'):
                multihead_model = model
                break
        
        if multihead_model is None:
            return None
        
        # Safe HOLD response for validation failures
        safe_hold_response = lambda reason: {
            "action": 1,
            "action_name": "HOLD",
            "direction_probs": {"LONG": 0.33, "SHORT": 0.33, "HOLD": 0.34},
            "confidence": 0.0,
            "mu": 0.0,
            "sigma": 0.01,
            "quantiles": {"q10": -0.01, "q25": -0.005, "q50": 0.0, "q75": 0.005, "q90": 0.01},
            "model_name": "HOLD (validation failed)",
            "is_learned": False,
            "error": reason,
            "feature_dim_validated": False
        }
        
        # === SCHEMA ENFORCEMENT ===
        # If we have feature_config and feature_names, enforce schema instead of blocking
        schema_stats = None
        if feature_names is not None and self.feature_config is not None:
            from training.feature_registry import FeatureValidator
            validator = FeatureValidator(self.feature_config)
            features, schema_stats = validator.enforce_schema(
                feature_names=feature_names,
                features=features,
                fill_value=0.0,
                max_missing_pct=0.15  # FIX #2: Abort to HOLD if >15% features missing
            )
            logger.info(f"[predict_multihead] Schema enforced: {schema_stats['incoming_features']} -> {schema_stats['expected_features']} features")
            
            # FIX #2: If too many features are missing, abort to HOLD
            if schema_stats.get("should_abort_to_hold", False):
                return safe_hold_response(
                    f"Too many missing features: {schema_stats['missing_filled']}/{schema_stats['expected_features']} "
                    f"({schema_stats['missing_pct']:.1%}). Missing: {schema_stats.get('missing_names', [])[:5]}"
                )
        else:
            # Legacy validation: Feature dimension validation (critical)
            input_features = features.shape[-1] if len(features.shape) >= 2 else features.shape[0]
            expected_dim = getattr(multihead_model, 'input_dim', self.input_dim)
            
            if input_features != expected_dim:
                logger.warning(f"[BLOCK] Feature mismatch: expected {expected_dim}, got {input_features}")
                return safe_hold_response(f"Feature dimension mismatch: expected {expected_dim}, got {input_features}")
            
            # Sequence length validation (critical - hard failure)
            if len(features.shape) == 2:
                actual_seq = features.shape[0]
            else:
                actual_seq = features.shape[1]
            
            expected_seq = getattr(self, 'sequence_length', 100)
            if actual_seq != expected_seq:
                logger.warning(f"[BLOCK] Sequence length mismatch: expected {expected_seq}, got {actual_seq}")
                return safe_hold_response(f"Sequence length mismatch: expected {expected_seq}, got {actual_seq}")
        
        try:
            multihead_model.eval()
            with torch.no_grad():
                # Handle both 2D [seq_len, features] and 3D [batch, seq_len, features] inputs
                if len(features.shape) == 2:
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                else:
                    x = torch.FloatTensor(features).to(self.device)
                    if len(x.shape) == 2:
                        x = x.unsqueeze(0)
                
                output = multihead_model.forward_multihead(x)
                
                # Extract probabilities (handle both single and batch)
                probs = torch.softmax(output.class_logits, dim=-1).cpu().numpy()
                if len(probs.shape) == 2 and probs.shape[0] == 1:
                    probs = probs[0]
                elif len(probs.shape) == 2:
                    probs = probs.mean(axis=0)  # Average across batch
                
                action_idx = int(np.argmax(probs))
                confidence = float(probs[action_idx])
                
                # Extract regression outputs
                mu_arr = output.mu.cpu().numpy()
                sigma_arr = output.sigma.cpu().numpy() if output.sigma is not None else np.array([[0.01]])
                
                mu = float(mu_arr.mean())
                sigma = float(sigma_arr.mean())
                
                # Extract learned quantiles
                quantiles_arr = output.quantiles.cpu().numpy()
                if len(quantiles_arr.shape) == 2 and quantiles_arr.shape[0] == 1:
                    quantiles = quantiles_arr[0]
                else:
                    quantiles = quantiles_arr.mean(axis=0)
                
                return {
                    "action": action_idx,
                    "action_name": ACTION_NAMES[action_idx],
                    "direction_probs": {
                        "LONG": float(probs[2]),
                        "SHORT": float(probs[0]),
                        "HOLD": float(probs[1])
                    },
                    "confidence": confidence,
                    "mu": mu,
                    "sigma": sigma,
                    "quantiles": {
                        "q10": float(quantiles[0]),
                        "q25": float(quantiles[1]),
                        "q50": float(quantiles[2]),
                        "q75": float(quantiles[3]),
                        "q90": float(quantiles[4])
                    },
                    "model_name": multihead_model.name,
                    "is_learned": True,
                    "feature_dim_validated": True
                }
        except Exception as e:
            logger.error(f"Multi-head prediction error: {e}")
            return None
    
    def update_training_status(self, **kwargs):
        self.training_status.update(kwargs)
        
    def add_prediction(self, prediction: Dict):
        prediction["timestamp"] = datetime.now().isoformat()
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)

model_manager = ModelManager()


class PredictionRequest(BaseModel):
    features: List[List[float]]
    sequence_length: int = 100

class CandleData(BaseModel):
    """Raw candle data for prediction."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class CandlePredictionRequest(BaseModel):
    """Request with raw candles - server will compute features."""
    candles: List[CandleData]
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"

class MTFCandleData(BaseModel):
    """Multi-timeframe candle data for ensemble prediction with MTF features."""
    candles_15m: List[CandleData]  # Base timeframe (required, 200+ candles)
    candles_5m: Optional[List[CandleData]] = None  # Optional context
    candles_1h: Optional[List[CandleData]] = None  # Optional context
    candles_4h: Optional[List[CandleData]] = None  # Optional context
    symbol: str = "BTCUSDT"
    
class PredictionResponse(BaseModel):
    action: str
    probabilities: Dict[str, float]
    confidence: float
    uncertainty: float
    model_weights: Optional[Dict[str, float]] = None
    reasoning: List[str]
    
class TrainingRequest(BaseModel):
    model_type: str
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    # Label generation mode: "cost_aware" | "pure_directional" | "regime"
    label_mode: str = "regime"  # Default to regime-based for best label distribution
    min_confidence: float = 0.40  # Stage 1: lowered from 0.7
    directional_threshold: float = 0.0020  # Stage 2: 0.20% for pure directional
    trend_threshold: float = 0.0015  # Stage 3: threshold for trending regime
    range_threshold: float = 0.0030  # Stage 3: threshold for ranging regime
    horizon: int = 16  # Forward prediction horizon in bars
    
class PredictionDistribution(BaseModel):
    """Live prediction distribution during training."""
    short: int = 0
    hold: int = 0
    long: int = 0
    total: int = 0

class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    current_model: Optional[str]
    progress: float
    metrics: Dict[str, Any]
    # Enhanced fields for real-time progress tracking
    epoch_history: List[Dict[str, Any]] = []
    start_time: Optional[str] = None
    eta_seconds: Optional[float] = None
    health_warnings: List[str] = []
    last_update: Optional[str] = None
    per_head_losses: Dict[str, float] = {}
    learning_rate: Optional[float] = None
    best_val_loss: Optional[float] = None
    early_stop_counter: int = 0
    # Live prediction distribution for GUI display
    prediction_distribution: Optional[PredictionDistribution] = None
    gradient_norm: Optional[float] = None
    
class ModelInfoResponse(BaseModel):
    name: str
    parameters: int
    accuracy: float
    last_trained: Optional[str]
    
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
    models_loaded: List[str]
    uptime_seconds: float
    ensemble_weights_loaded: bool = False
    ensemble_using_defaults: bool = True
    ensemble_weight_count: int = 0

class RegressionPredictionRequest(BaseModel):
    """Request for regression-based prediction (mu, sigma)."""
    features: List[List[float]]
    current_price: float = 0.0
    current_volatility: float = 0.01
    
class RegressionPredictionResponse(BaseModel):
    """Response with edge-based signal format."""
    action: str  # LONG, SHORT, NO_TRADE
    confidence: float  # edge / sigma
    expected_move: float  # mu
    uncertainty: float  # sigma
    edge: float  # mu - cost
    cost_estimate: float
    suggested_order_type: str  # MAKER or TAKER
    urgency: str  # LOW, MEDIUM, HIGH
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    regime: str
    expert_weights: Dict[str, float]
    reasons: List[str]
    
    # Legacy compatibility
    probabilities: Optional[Dict[str, float]] = None
    model_weights: Optional[Dict[str, float]] = None

class EnsemblePredictionRequest(BaseModel):
    """Request for professional ensemble prediction."""
    features: List[List[float]]

class QuantilePredictionRequest(BaseModel):
    """Request for quantile regression prediction."""
    features: List[List[float]]
    feature_names: Optional[List[str]] = None  # For schema enforcement

class QuantilePredictionResponse(BaseModel):
    """Response with quantile regression for Entry/SL/TP derivation."""
    direction_probs: Dict[str, float]
    quantiles: Dict[str, float]  # q10, q25, q50, q75, q90
    mfe_quantiles: Optional[Dict[str, float]] = None
    mae_quantiles: Optional[Dict[str, float]] = None
    model_name: str
    confidence: float

class EnsemblePredictionResponse(BaseModel):
    """Response with regime-gated ensemble signal."""
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
    
    # Position sizing
    position_size_pct: float
    regime_adjusted_size: float
    
    # Thresholds
    confidence_threshold_used: float
    regime_adjustment: str
    
    # Per-model breakdown
    model_votes: Dict[str, Any]
    
    # Ensemble probabilities
    ensemble_probs: Dict[str, float]
    
    # Reasons
    reasons: List[str]


class MultiHeadPredictionResponse(BaseModel):
    """
    Unified multihead prediction response.
    
    Returns ALL 8 heads from multi-head models:
    1. Direction (classification)
    2. Expected return (mu) and uncertainty (sigma)
    3. Quantiles (q10, q25, q50, q75, q90)
    4. Trading levels (entry_offset, sl_distance, tp_distance)
    5. Future candle predictions
    6. Derived trade plan
    7. Vol state (flow forecast) - volatility regime classification
    8. Acceleration (flow forecast) - momentum change prediction
    """
    # Direction
    action: str  # LONG, SHORT, HOLD
    direction_probs: Dict[str, float]  # P(SHORT), P(HOLD), P(LONG)
    confidence: float
    
    # Regression (μ, σ)
    expected_return: float  # mu
    uncertainty: float  # sigma
    edge: float  # mu - cost
    
    # Quantiles (learned, not heuristic)
    quantiles: Dict[str, float]  # q10, q25, q50, q75, q90 as percentage returns
    
    # Trading levels (learned from MFE/MAE)
    entry_offset_pct: float  # Optimal limit order offset
    stop_loss_pct: float  # Stop loss distance
    take_profit_pct: float  # Take profit distance
    
    # Derived price levels
    current_price: float
    entry_price: float  # current_price * (1 + entry_offset)
    stop_loss_price: float
    take_profit_price: float
    
    # Future candle predictions (optional)
    predicted_candles: Optional[List[Dict[str, float]]] = None  # [{close_delta, high_delta, low_delta}, ...]
    
    # Flow Forecast (volatility regime + acceleration)
    vol_state: str  # "contraction", "neutral", "expansion"
    vol_state_probs: Dict[str, float]  # P(contraction), P(neutral), P(expansion)
    acceleration: float  # momentum change prediction
    forecast_mode: str  # "QUANTILE_PATHS" or "NO_FORECAST" (gated)
    
    # Quantile path projections (for UI rendering)
    quantile_paths: Optional[Dict[str, List[float]]] = None  # {q10: [...], q50: [...], q90: [...]}
    
    # Trade plan
    suggested_order_type: str  # MAKER or TAKER
    urgency: str  # LOW, MEDIUM, HIGH
    position_size_pct: float
    risk_reward_ratio: float
    
    # Metadata
    model_name: str
    is_multihead: bool
    reasons: List[str]


start_time = datetime.now()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_used = None
    gpu_memory_total = None
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
    uptime = (datetime.now() - start_time).total_seconds()
    
    # Check ensemble weights status
    ensemble_weights_loaded = False
    ensemble_using_defaults = True
    ensemble_weight_count = 0
    
    weights_path = Path(__file__).parent.parent / "checkpoints" / "model_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                weights_data = json.load(f)
            ensemble_weights_loaded = True
            ensemble_weight_count = len(weights_data)
            ensemble_using_defaults = False
            logger.info(f"[HEALTH] Ensemble weights: {ensemble_weight_count} models with real walk-forward metrics")
        except Exception as e:
            logger.warning(f"[HEALTH] Failed to load ensemble weights: {e}")
    else:
        logger.warning(f"[HEALTH] ⚠️ model_weights.json NOT FOUND - ensemble using defaults")
    
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
        models_loaded=list(model_manager.models.keys()),
        uptime_seconds=uptime,
        ensemble_weights_loaded=ensemble_weights_loaded,
        ensemble_using_defaults=ensemble_using_defaults,
        ensemble_weight_count=ensemble_weight_count
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction with CORRECT label mapping.
    
    Training labels: 0=SHORT, 1=NEUTRAL/HOLD, 2=LONG
    Probabilities: [P(SHORT), P(HOLD), P(LONG)]
    """
    try:
        features = np.array(request.features)
        
        # Validate input shape
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        seq_len, n_features = features.shape
        if seq_len < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence length {seq_len} too short (minimum 10)"
            )
        
        # Validate feature count against model expectation
        if n_features != model_manager.input_dim:
            logger.error(f"Feature mismatch in /predict: got {n_features}, expected {model_manager.input_dim}")
            raise HTTPException(
                status_code=400,
                detail=f"Feature dimension mismatch: Expected {model_manager.input_dim}, got {n_features}. Use /predict/candles or /predict/ensemble/candles for proper feature alignment."
            )
        
        # === STF/MTF VALIDATION ===
        # Warn if client is likely using wrong feature schema
        if model_manager.training_mode == "STF" and n_features > 50:
            logger.error(f"[SCHEMA FATAL] /predict: Model trained with STF ({model_manager.input_dim}) but client sent {n_features} features (likely MTF)")
            raise HTTPException(
                status_code=422,
                detail=f"Schema mismatch: Model expects STF ({model_manager.input_dim} features) but received {n_features}. "
                       f"Use /predict/candles or /predict/ensemble/candles for proper feature computation."
            )
        
        logger.info(f"[/predict] Received {n_features} features, model expects {model_manager.input_dim} ({model_manager.training_mode} mode)")
        
        result = model_manager.predict(features)
        
        # CORRECT mapping: index 0=SHORT, 1=HOLD, 2=LONG
        action_idx = result["action"]
        action = result.get("action_name", ACTION_NAMES[action_idx])
        
        # Map probabilities correctly: [P(SHORT), P(HOLD), P(LONG)]
        # Convert numpy values to Python floats for JSON serialization
        probs = {
            "SHORT": float(result["probabilities"][0]),
            "HOLD": float(result["probabilities"][1]),
            "LONG": float(result["probabilities"][2])
        }
        confidence = float(result["confidence"])
        uncertainty = float(result["uncertainty"])
        model_weights = result.get("model_weights", {})
        reasoning = result.get("reasoning", [])
            
        prediction = {
            "action": action,
            "probabilities": probs,
            "confidence": confidence,
            "features_shape": list(features.shape)
        }
        model_manager.add_prediction(prediction)
        
        return PredictionResponse(
            action=action,
            probabilities=probs,
            confidence=confidence,
            uncertainty=uncertainty,
            model_weights=model_weights,
            reasoning=reasoning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/candles", response_model=PredictionResponse)
async def predict_from_candles(request: CandlePredictionRequest):
    """Make prediction from raw candle data.
    
    This endpoint handles:
    1. Converting candles to DataFrame
    2. Computing technical indicators/features
    3. Scaling features using the saved scaler
    4. Making prediction with correct label mapping
    
    CRITICAL: Training labels are 0=SHORT, 1=NEUTRAL/HOLD, 2=LONG
    """
    try:
        if len(request.candles) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 candles for feature computation, got {len(request.candles)}"
            )
        
        # Convert candles to DataFrame
        import pandas as pd
        candle_data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in request.candles
        ]
        df = pd.DataFrame(candle_data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Import feature computation from pipeline
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.pipeline import FeatureEngineer
            
            # Compute features using the correct method name
            fe = FeatureEngineer()
            features_df = fe.compute_technical_features(df)
            
            # === FIX: Selective NaN handling instead of full dropna() ===
            # Only require close price to exist, forward-fill feature NaNs
            if "close" in features_df.columns:
                features_df = features_df.dropna(subset=["close"])
            feature_cols = [c for c in features_df.columns if c not in ["datetime", "timestamp", "close", "open", "high", "low", "volume"]]
            if len(feature_cols) > 0:
                features_df[feature_cols] = features_df[feature_cols].ffill().fillna(0.0)
            
            if len(features_df) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No valid features after computation (all NaN)"
                )
            
            # === STF-ONLY ENFORCEMENT ===
            # This endpoint uses compute_technical_features() which produces STF (41 features)
            computed_feature_count = len([c for c in features_df.columns if c not in ["datetime", "timestamp", "close", "open", "high", "low", "volume"]])
            
            if model_manager.training_mode == "MTF":
                logger.error(f"[SCHEMA FATAL] /predict/candles uses STF pipeline but model expects MTF ({model_manager.input_dim} features)")
                raise HTTPException(
                    status_code=422,
                    detail=f"Model trained with MTF ({model_manager.input_dim} features) but /predict/candles only supports STF. "
                           f"Use /predict/ensemble/candles?mode=mtf instead."
                )
            
            logger.info(f"[STF-ONLY] /predict/candles: computed {computed_feature_count} features for STF model")
            
            # Scale features using the per-column scaler dict (via transform_features helper)
            features_np = model_manager.transform_features(features_df)
            
            # Use sequence length from loaded model config
            seq_len = model_manager.sequence_length
            if len(features_np) < seq_len:
                logger.warning(f"Not enough features ({len(features_np)}) for seq_len={seq_len}, using available")
                seq_len = len(features_np)
            
            features_seq = features_np[-seq_len:]
            
            # === REQUIRED INFERENCE RULE: Check schema mismatch ===
            # Enforce 15% threshold - HTTP 422 if too many features missing
            actual_feature_count = features_seq.shape[1]
            expected_feature_count = model_manager.input_dim
            
            if expected_feature_count > 0:
                missing_count = max(0, expected_feature_count - actual_feature_count)
                missing_pct = (missing_count / expected_feature_count) * 100
            else:
                missing_pct = 0 if actual_feature_count == 0 else 100
            
            if missing_pct > 15:
                logger.error(f"[HARD ERROR] Feature mismatch: computed {actual_feature_count}, model expects {expected_feature_count} ({missing_pct:.1f}% missing)")
                raise HTTPException(
                    status_code=422,
                    detail="Feature schema mismatch — wrong endpoint or retrain required"
                )
            
        except ImportError as e:
            logger.error(f"Failed to import feature pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature pipeline not available: {e}"
            )
        
        # Make prediction
        result = model_manager.predict(features_seq)
        
        action_idx = result["action"]
        action = result.get("action_name", ACTION_NAMES[action_idx])
        
        # Convert numpy values to Python floats for JSON serialization
        probs = {
            "SHORT": float(result["probabilities"][0]),
            "HOLD": float(result["probabilities"][1]),
            "LONG": float(result["probabilities"][2])
        }
        confidence = float(result["confidence"])
        uncertainty = float(result["uncertainty"])
        
        prediction = {
            "action": action,
            "probabilities": probs,
            "confidence": confidence,
            "features_shape": list(features_seq.shape),
            "candles_used": len(request.candles)
        }
        model_manager.add_prediction(prediction)
        
        return PredictionResponse(
            action=action,
            probabilities=probs,
            confidence=confidence,
            uncertainty=uncertainty,
            model_weights=result.get("model_weights", {}),
            reasoning=result.get("reasoning", []) + [
                f"Processed {len(request.candles)} candles → {features_seq.shape[0]} sequences"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Candle prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multihead/candles", response_model=MultiHeadPredictionResponse)
async def predict_multihead_from_candles(request: CandlePredictionRequest):
    """
    Unified multihead prediction endpoint.
    
    This is the CANONICAL endpoint for multi-head model inference.
    It uses forward_multihead() internally and returns ALL 6 heads:
    1. Direction probabilities (classification)
    2. Expected return (μ) and uncertainty (σ)
    3. Quantiles (q10, q25, q50, q75, q90)
    4. Trading levels (entry_offset, sl_distance, tp_distance) - learned from MFE/MAE
    5. Future candle predictions
    6. Derived trade plan with price levels
    
    This endpoint REPLACES /predict/candles for trading purposes.
    """
    try:
        if len(request.candles) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 candles for feature computation, got {len(request.candles)}"
            )
        
        # Get current price from last candle
        current_price = request.candles[-1].close
        
        # Convert candles to DataFrame
        import pandas as pd
        candle_data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in request.candles
        ]
        df = pd.DataFrame(candle_data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Import feature computation from pipeline
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.pipeline import FeatureEngineer
            
            # Compute features
            fe = FeatureEngineer()
            features_df = fe.compute_technical_features(df)
            
            # FIX #3: Drop OHLCV columns before feature extraction (they're not model features)
            # These are metadata columns that shouldn't be included in model input
            ohlcv_cols = ["datetime", "timestamp", "close", "open", "high", "low", "volume", "symbol"]
            feature_cols = [c for c in features_df.columns if c not in ohlcv_cols]
            
            # === STF-ONLY ENFORCEMENT ===
            # This endpoint uses compute_technical_features() which produces STF (41 features)
            computed_feature_count = len(feature_cols)
            
            if model_manager.training_mode == "MTF":
                logger.error(f"[SCHEMA FATAL] /predict/multihead/candles uses STF pipeline but model expects MTF ({model_manager.input_dim} features)")
                raise HTTPException(
                    status_code=422,
                    detail=f"Model trained with MTF ({model_manager.input_dim} features) but this endpoint only supports STF. "
                           f"Use /predict/ensemble/candles?mode=mtf instead."
                )
            
            logger.info(f"[STF-ONLY] /predict/multihead/candles: computed {computed_feature_count} features for STF model")
            
            # Handle NaNs with forward-fill then zero-fill (FIX #2)
            if len(feature_cols) > 0:
                features_df[feature_cols] = features_df[feature_cols].ffill().fillna(0.0)
            
            # Extract only feature columns (not OHLCV)
            features_only_df = features_df[feature_cols].copy()
            
            if len(features_only_df) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No valid features after computation (all NaN)"
                )
            
            # Scale features
            features_np = model_manager.transform_features(features_only_df)
            
            # FIX #4: Lock sequence_length=100 - NO dynamic resizing
            seq_len = model_manager.sequence_length
            if len(features_np) < seq_len:
                # FAIL if not enough data instead of silently reducing sequence length
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: need {seq_len} candles for sequence, got {len(features_np)}. "
                           f"This ensures training-inference alignment."
                )
            
            # Always use exactly seq_len (default 100)
            features_seq = features_np[-seq_len:]
            
            # === REQUIRED INFERENCE RULE: Check schema mismatch ===
            # Enforce 15% threshold - HTTP 422 if too many features missing
            actual_feature_count = features_seq.shape[1]
            expected_feature_count = model_manager.input_dim
            
            if expected_feature_count > 0:
                missing_count = max(0, expected_feature_count - actual_feature_count)
                missing_pct = (missing_count / expected_feature_count) * 100
            else:
                missing_pct = 0 if actual_feature_count == 0 else 100
            
            if missing_pct > 15:
                logger.error(f"[HARD ERROR] Feature mismatch: computed {actual_feature_count}, model expects {expected_feature_count} ({missing_pct:.1f}% missing)")
                raise HTTPException(
                    status_code=422,
                    detail="Feature schema mismatch — wrong endpoint or retrain required"
                )
            
        except ImportError as e:
            logger.error(f"Failed to import feature pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature pipeline not available: {e}"
            )
        
        # Check if multihead model is available
        multihead_model = model_manager.get_multihead_model()
        
        if multihead_model is None:
            # Fallback to basic prediction
            result = model_manager.predict(features_seq)
            
            return MultiHeadPredictionResponse(
                action=result.get("action_name", ACTION_NAMES[result["action"]]),
                direction_probs={
                    "SHORT": result["probabilities"][0],
                    "HOLD": result["probabilities"][1],
                    "LONG": result["probabilities"][2]
                },
                confidence=result["confidence"],
                expected_return=0.0,
                uncertainty=0.02,
                edge=0.0,
                quantiles={"q10": -0.02, "q25": -0.01, "q50": 0.0, "q75": 0.01, "q90": 0.02},
                entry_offset_pct=0.0,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                current_price=current_price,
                entry_price=current_price,
                stop_loss_price=current_price * 0.98,
                take_profit_price=current_price * 1.04,
                predicted_candles=None,
                suggested_order_type="TAKER",
                urgency="MEDIUM",
                position_size_pct=2.0,
                risk_reward_ratio=2.0,
                model_name="classification_fallback",
                is_multihead=False,
                reasons=["No multihead model available, using classification fallback"]
            )
        
        # Use multihead model with forward_multihead()
        import torch
        import torch.nn.functional as F
        
        device = next(multihead_model.parameters()).device
        x = torch.from_numpy(features_seq).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # This is the key - use forward_multihead NOT forward
            output = multihead_model.forward_multihead(x)
        
        # Extract all heads
        class_probs = F.softmax(output.class_logits, dim=-1).cpu().numpy()[0]
        mu = output.mu.cpu().item()
        sigma_or_log_sigma = output.sigma.cpu().item()
        quantiles_raw = output.quantiles.cpu().numpy()[0]  # [q10, q25, q50, q75, q90]
        entry_offset = output.entry_offset.cpu().item()
        learned_sl_distance = output.sl_distance.cpu().item()  # Keep for debugging
        learned_tp_distance = output.tp_distance.cpu().item()  # Keep for debugging
        candle_deltas = output.candle_deltas.cpu().numpy()[0]  # [n_steps, 3]
        
        # Flow Forecast: Extract vol_state and acceleration
        vol_state_probs_raw = F.softmax(output.vol_state_logits, dim=-1).cpu().numpy()[0] if output.vol_state_logits is not None else np.array([0.0, 1.0, 0.0])
        acceleration_pred = output.acceleration.cpu().item() if output.acceleration is not None else 0.0
        vol_state_idx = int(np.argmax(vol_state_probs_raw))
        VOL_STATE_NAMES = ["contraction", "neutral", "expansion"]
        vol_state_name = VOL_STATE_NAMES[vol_state_idx]
        
        # PHASE 1b: Handle log_sigma output
        # If model uses log_sigma, convert to sigma: σ = exp(log_sigma)
        use_log_sigma = getattr(multihead_model, 'use_log_sigma', True)  # Default to True for new models
        if use_log_sigma:
            sigma = float(np.exp(np.clip(sigma_or_log_sigma, -10, 5)))  # exp(log_sigma)
        else:
            sigma = float(sigma_or_log_sigma)  # Already sigma
        
        # Determine action
        action_idx = int(np.argmax(class_probs))
        action = ACTION_NAMES[action_idx]
        confidence = float(class_probs[action_idx])
        
        # PHASE 1a: Cost-aware edge calculation using TradingCosts
        # Calculate volatility from recent candles (ATR-based)
        recent_candles = df.tail(14)
        high_low = recent_candles['high'] - recent_candles['low']
        atr = float(high_low.mean())
        volatility = atr / current_price  # As percentage
        
        # Trading costs with volatility + 4h hold (16 bars @ 15m = 4 hours)
        hold_hours = 4.0  # 16 bars * 15min = 4 hours
        maker_fee = 0.0002  # 0.02%
        taker_fee = 0.0004  # 0.04%
        slippage = volatility * 0.1  # ~10% of ATR as slippage estimate
        funding_periods = hold_hours / 8.0  # Funding every 8 hours
        funding_rate = 0.0001  # ~0.01% typical
        
        # Total round-trip cost
        cost = taker_fee * 2 + slippage * 2 + funding_rate * funding_periods
        
        # Net edge = |μ| - cost (PHASE 1a)
        edge = abs(mu) - cost
        
        # PHASE 1c: Derive SL/TP from quantiles instead of learned heads
        # This ensures internal consistency - SL/TP come from the same distribution
        # Convert numpy.float32 to Python float immediately to avoid JSON serialization issues
        q10, q25, q50, q75, q90 = [float(q) for q in quantiles_raw]
        
        if action == "LONG":
            # LONG: SL from q10 (downside risk), TP from q90 (upside potential)
            sl_distance = abs(q10) if q10 < 0 else abs(q25)  # Use negative quantile
            tp_distance = q90 if q90 > 0 else q75  # Use positive quantile
        elif action == "SHORT":
            # SHORT: SL from q90 (upside risk), TP from q10 (downside potential)
            sl_distance = q90 if q90 > 0 else abs(q75)  # Use positive quantile (adverse move)
            tp_distance = abs(q10) if q10 < 0 else abs(q25)  # Use negative quantile (favorable move)
        else:  # HOLD
            # Conservative defaults for HOLD
            sl_distance = 0.005  # 0.5%
            tp_distance = 0.005  # 0.5%
        
        # Enforce minimum SL/TP to avoid micro-trades (at least 0.1%)
        sl_distance = float(max(abs(sl_distance), 0.001))
        tp_distance = float(max(abs(tp_distance), 0.001))
        
        # Derive price levels based on action - ensure all are Python floats
        entry_offset = float(entry_offset)
        entry_price = float(current_price * (1 + entry_offset))
        
        if action == "LONG":
            stop_loss_price = float(current_price * (1 - sl_distance))
            take_profit_price = float(current_price * (1 + tp_distance))
        elif action == "SHORT":
            stop_loss_price = float(current_price * (1 + sl_distance))
            take_profit_price = float(current_price * (1 - tp_distance))
        else:  # HOLD
            stop_loss_price = float(current_price * (1 - sl_distance))
            take_profit_price = float(current_price * (1 + tp_distance))
        
        # Risk-reward ratio
        risk = abs(current_price - stop_loss_price)
        reward = abs(take_profit_price - current_price)
        rr_ratio = float(reward / risk) if risk > 0 else 0.0
        
        # Suggested order type based on urgency
        if confidence > 0.7 and abs(mu) > 0.01:
            urgency = "HIGH"
            suggested_order_type = "TAKER"
        elif confidence > 0.5:
            urgency = "MEDIUM"
            suggested_order_type = "MAKER"
        else:
            urgency = "LOW"
            suggested_order_type = "MAKER"
        
        # Position sizing based on confidence and edge
        base_size = 2.0  # 2% base
        position_size = base_size * min(confidence * 2, 1.5) * (1 + edge)
        position_size = float(min(max(position_size, 0.5), 5.0))  # 0.5% to 5%
        
        # Format predicted candles
        predicted_candles = []
        for i in range(len(candle_deltas)):
            predicted_candles.append({
                "step": i + 1,
                "close_delta": float(candle_deltas[i, 0]),
                "high_delta": float(candle_deltas[i, 1]),
                "low_delta": float(candle_deltas[i, 2])
            })
        
        # Flow Forecast: Generate quantile paths with alpha shaping
        # Alpha controls path curvature: contraction=0.7, neutral=1.0, expansion=1.5
        ALPHA_MAP = {"contraction": 0.7, "neutral": 1.0, "expansion": 1.5}
        alpha = ALPHA_MAP.get(vol_state_name, 1.0)
        
        # Volatility gate: NO_FORECAST when vol_state==contraction OR spread too narrow
        spread = q75 - q25  # IQR as percentage return
        min_spread = 3 * cost  # Must exceed 3x trading cost
        
        if vol_state_name == "contraction" or spread < min_spread:
            forecast_mode = "NO_FORECAST"
            quantile_paths = None
        else:
            forecast_mode = "QUANTILE_PATHS"
            # Generate paths: path[k] = close * exp((k/h)^α * quantile)
            horizon = 16  # 16 bars = 4 hours at 15m
            steps = list(range(1, horizon + 1))
            
            quantile_paths = {
                "q10": [float(current_price * np.exp((k / horizon) ** alpha * q10)) for k in steps],
                "q50": [float(current_price * np.exp((k / horizon) ** alpha * q50)) for k in steps],
                "q90": [float(current_price * np.exp((k / horizon) ** alpha * q90)) for k in steps],
            }
        
        # Build reasons with institutional-grade details
        reasons = [
            f"Model prediction: {action} with {confidence:.1%} confidence",
            f"Expected return (μ): {mu:.4f} ({mu*100:.2f}%)",
            f"Uncertainty (σ): {sigma:.4f}" + (" [from log_sigma]" if use_log_sigma else ""),
            f"Trading cost: {cost*100:.3f}% (fees + slippage + funding)",
            f"Net edge: {edge:.4f} (μ - cost)",
            f"Entry offset: {entry_offset*100:.3f}%",
            f"SL: {sl_distance*100:.2f}% (from q{10 if action=='LONG' else 90}), TP: {tp_distance*100:.2f}% (from q{90 if action=='LONG' else 10})",
            f"[Debug] Learned SL/TP: {learned_sl_distance*100:.2f}%/{learned_tp_distance*100:.2f}%",
            f"Risk:Reward = 1:{rr_ratio:.2f}",
            f"Vol State: {vol_state_name} (α={alpha:.1f}), Acceleration: {acceleration_pred:.4f}",
            f"Forecast Mode: {forecast_mode}" + (f" (spread {spread*100:.3f}% < {min_spread*100:.3f}% min)" if forecast_mode == "NO_FORECAST" else "")
        ]
        
        return MultiHeadPredictionResponse(
            action=action,
            direction_probs={
                "SHORT": float(class_probs[0]),
                "HOLD": float(class_probs[1]),
                "LONG": float(class_probs[2])
            },
            confidence=float(confidence),
            expected_return=float(mu),
            uncertainty=float(sigma),
            edge=float(edge),
            quantiles={
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90
            },
            entry_offset_pct=entry_offset,
            stop_loss_pct=sl_distance,
            take_profit_pct=tp_distance,
            current_price=float(current_price),
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            predicted_candles=predicted_candles,
            vol_state=vol_state_name,
            vol_state_probs={
                "contraction": float(vol_state_probs_raw[0]),
                "neutral": float(vol_state_probs_raw[1]),
                "expansion": float(vol_state_probs_raw[2])
            },
            acceleration=float(acceleration_pred),
            forecast_mode=forecast_mode,
            quantile_paths=quantile_paths,
            suggested_order_type=suggested_order_type,
            urgency=urgency,
            position_size_pct=position_size,
            risk_reward_ratio=rr_ratio,
            model_name=getattr(multihead_model, '__class__.__name__', 'MultiHeadModel'),
            is_multihead=True,
            reasons=reasons
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multihead prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/regression", response_model=RegressionPredictionResponse)
async def predict_regression(request: RegressionPredictionRequest):
    """Make edge-based regression prediction.
    
    Returns mu (expected return), sigma (uncertainty), and edge-based signal.
    This is the institutional-grade signal format:
        edge = mu - cost
        confidence = edge / sigma
        action = LONG/SHORT if confidence > threshold, else NO_TRADE
    """
    try:
        features = np.array(request.features)
        
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        # Check for MoE or regression model
        result = model_manager.predict(features)
        
        # Extract mu and sigma from model output
        # For classification models, convert probabilities to pseudo-mu/sigma
        probs = result["probabilities"]
        
        # P(LONG) - P(SHORT) as directional signal
        p_long = probs[2]
        p_short = probs[0]
        p_hold = probs[1]
        
        # Convert to mu: expected direction * magnitude
        mu = (p_long - p_short) * 0.01  # Scale to ~1% expected move
        
        # Uncertainty from entropy of distribution
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        max_entropy = -3 * (1/3) * np.log(1/3)  # Max entropy for 3 classes
        sigma = 0.005 + 0.015 * (entropy / max_entropy)  # 0.5% to 2% uncertainty
        
        # Transaction costs
        maker_fee = 0.0002
        taker_fee = 0.0004
        slippage = 0.0001 + 0.5 * request.current_volatility
        cost = (taker_fee * 2) + (slippage * 2)
        
        # Calculate edge per institutional spec: edge = (μ - cost) / σ
        # This is the risk-adjusted expected profit
        sigma_safe = max(sigma, 0.001)
        edge = (abs(mu) - cost) / sigma_safe
        
        # Edge IS the confidence in this formulation
        confidence = edge
        
        # Determine action based on edge threshold
        # Edge > 0.5 means expected profit is 0.5 standard deviations above costs
        min_edge_threshold = 0.5
        
        should_trade = edge >= min_edge_threshold
        
        if should_trade:
            action = "LONG" if mu > 0 else "SHORT"
        else:
            action = "NO_TRADE"
        
        # Calculate position size using bounded Kelly
        # Since edge = (mu - cost) / sigma, we use edge * sigma for original profit
        if should_trade and sigma > 0:
            expected_profit = edge * sigma  # Recover (mu - cost)
            kelly = expected_profit / (sigma ** 2)  # Kelly = (mu - cost) / sigma^2
            half_kelly = kelly * 0.5
            position_size_pct = max(0, min(half_kelly, 0.1))  # Max 10%
        else:
            position_size_pct = 0
        
        # Suggested order type
        # Calculate edge with maker fees to compare
        cost_maker = (maker_fee * 2) + (slippage * 2)
        edge_maker = (abs(mu) - cost_maker) / sigma_safe
        suggested_order = "TAKER" if edge > edge_maker * 1.5 else "MAKER"
        
        # Urgency
        if confidence > 2.0 and abs(mu) > 0.01:
            urgency = "HIGH"
        elif confidence > 1.0:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Stops
        stop_loss_pct = sigma * 2
        take_profit_pct = abs(mu) * 1.2 if abs(mu) > stop_loss_pct * 1.5 else stop_loss_pct * 1.5
        
        # Reasons
        reasons = []
        if should_trade:
            reasons.append(f"Edge: {edge:.2f}σ (risk-adjusted)")
            reasons.append(f"Expected move: {mu*100:.3f}%")
            reasons.append(f"Uncertainty: {sigma*100:.3f}%")
        else:
            reasons.append(f"Edge too low: {edge:.2f}σ < {min_edge_threshold}σ required")
        
        return RegressionPredictionResponse(
            action=action,
            confidence=confidence,
            expected_move=float(mu),
            uncertainty=float(sigma),
            edge=float(edge),
            cost_estimate=float(cost),
            suggested_order_type=suggested_order,
            urgency=urgency,
            position_size_pct=float(position_size_pct),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            regime="UNKNOWN",  # Will be set by regime detector
            expert_weights=result.get("model_weights", {}),
            reasons=reasons,
            probabilities={
                "SHORT": float(probs[0]),
                "HOLD": float(probs[1]),
                "LONG": float(probs[2])
            },
            model_weights=result.get("model_weights", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regression prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/quantile", response_model=QuantilePredictionResponse)
async def predict_quantile(request: QuantilePredictionRequest):
    """Make quantile regression prediction for Entry/SL/TP derivation.
    
    Returns:
    - Direction probabilities (LONG, SHORT, HOLD)
    - Return quantiles (q10, q25, q50, q75, q90) for probability-based price projections
    - MFE/MAE quantiles when available
    
    Entry/SL/TP are derived from quantiles:
    - Entry = current price
    - For LONG: SL = price * (1 + q10), TP = price * (1 + q90)
    - For SHORT: SL = price * (1 + q90), TP = price * (1 + q10)
    
    NOTE: This endpoint uses learned quantiles if a multi-head model is loaded,
    otherwise falls back to heuristic synthesis from classification probabilities.
    """
    try:
        features = np.array(request.features)
        feature_names = request.feature_names
        
        if len(features.shape) == 2 and features.shape[0] == 1:
            features = features.reshape(1, features.shape[0], features.shape[1])
        elif len(features.shape) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Expected features shape [batch, seq_len, features], got {features.shape}"
            )
        
        # === SCHEMA ENFORCEMENT ===
        # FIX: Always require feature_names - no more guessing or permissive fallback!
        schema_stats = None
        expected_seq = model_manager.sequence_length  # FIX: Define expected_seq upfront
        
        if model_manager.feature_config is not None:
            from training.feature_registry import FeatureValidator
            validator = FeatureValidator(model_manager.feature_config)
            
            # === FIX: ALWAYS REJECT requests without feature_names ===
            # Even "dims match" is dangerous because order can still mismatch
            if feature_names is None:
                incoming_dim = features.shape[-1]
                expected_cols = model_manager.feature_config.feature_columns
                expected_dim = len(expected_cols)
                
                logger.error(
                    f"[/predict/quantile] REJECTED: feature_names is required. "
                    f"Incoming={incoming_dim}, expected={expected_dim}. "
                    f"Use /predict/ensemble/candles endpoint or provide feature_names."
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"feature_names is ALWAYS required for /predict/quantile to avoid data misalignment. "
                           f"Incoming features: {incoming_dim}, expected: {expected_dim}. "
                           f"Either provide feature_names matching your feature builder, "
                           f"or use the /predict/ensemble/candles endpoint which computes features server-side."
                )
            else:
                # Full schema enforcement with column names
                features_for_enforcement = features[0] if features.shape[0] == 1 else features
                features_for_enforcement, schema_stats = validator.enforce_schema(
                    feature_names=feature_names,
                    features=features_for_enforcement,
                    fill_value=0.0
                )
                logger.info(f"[/predict/quantile] Schema enforced: {schema_stats['incoming_features']} -> {schema_stats['expected_features']} features")
                
                # === REQUIRED INFERENCE RULE: Check schema mismatch ===
                # Enforce 15% threshold - HTTP 422 if too many features missing
                missing_pct = (schema_stats['missing_filled'] / schema_stats['expected_features']) * 100 if schema_stats['expected_features'] > 0 else 0
                
                if missing_pct > 15:
                    logger.error(f"[HARD ERROR] SCHEMA MISMATCH: {missing_pct:.1f}% features missing ({schema_stats['missing_filled']}/{schema_stats['expected_features']})")
                    raise HTTPException(
                        status_code=422,
                        detail="Feature schema mismatch — wrong endpoint or retrain required"
                    )
                
                # Restore batch dimension if needed
                if len(features_for_enforcement.shape) == 2:
                    features = features_for_enforcement.reshape(1, *features_for_enforcement.shape)
                else:
                    features = features_for_enforcement
        
        # Check if we have a multi-head model with learned quantiles
        # Pass feature_names for internal schema enforcement if not done above
        multihead_result = model_manager.predict_multihead(
            features[0] if features.shape[0] == 1 else features,
            feature_names=feature_names if schema_stats is None else None  # Already enforced if schema_stats exists
        )
        
        if multihead_result is not None:
            # Use learned quantiles from multi-head model
            direction_probs = multihead_result["direction_probs"]
            quantiles = multihead_result["quantiles"]
            confidence = multihead_result["confidence"]
            
            # MFE/MAE estimates from quantile spread
            q90 = quantiles["q90"]
            q10 = quantiles["q10"]
            mfe_quantiles = {
                "q10": float(max(quantiles["q75"], 0) * 0.8),
                "q50": float(max(q90, 0) * 0.9),
                "q90": float(max(q90, 0) * 1.2)
            }
            mae_quantiles = {
                "q10": float(min(q10, 0) * 0.8),
                "q50": float(min(q10, 0) * 1.0),
                "q90": float(min(q10, 0) * 1.3)
            }
            
            return QuantilePredictionResponse(
                direction_probs=direction_probs,
                quantiles=quantiles,
                mfe_quantiles=mfe_quantiles,
                mae_quantiles=mae_quantiles,
                model_name=multihead_result.get("model_name", "MultiHead"),
                confidence=confidence
            )
        
        # Fallback to heuristic synthesis from classification probabilities
        # Pass feature_names for schema enforcement if not done above
        result = model_manager.predict(
            features[0] if features.shape[0] == 1 else features,
            feature_names=feature_names if schema_stats is None else None
        )
        
        probs = result.get("probabilities", [0.33, 0.34, 0.33])
        direction_probs = {
            "LONG": float(probs[2]),
            "SHORT": float(probs[0]),
            "HOLD": float(probs[1])
        }
        
        confidence = float(result.get("confidence", max(probs)))
        
        p_long = probs[2]
        p_short = probs[0]
        directional_bias = p_long - p_short
        
        base_vol = 0.015
        uncertainty = float(result.get("uncertainty", 0.3))
        spread = base_vol * (1 + uncertainty)
        
        q50 = directional_bias * base_vol * 2
        q25 = q50 - spread * 0.67
        q75 = q50 + spread * 0.67
        q10 = q50 - spread * 1.28
        q90 = q50 + spread * 1.28
        
        quantiles = {
            "q10": float(q10),
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "q90": float(q90)
        }
        
        mfe_quantiles = {
            "q10": float(max(q75, 0) * 0.8),
            "q50": float(max(q90, 0) * 0.9),
            "q90": float(max(q90, 0) * 1.2)
        }
        
        mae_quantiles = {
            "q10": float(min(q10, 0) * 0.8),
            "q50": float(min(q10, 0) * 1.0),
            "q90": float(min(q10, 0) * 1.3)
        }
        
        return QuantilePredictionResponse(
            direction_probs=direction_probs,
            quantiles=quantiles,
            mfe_quantiles=mfe_quantiles,
            mae_quantiles=mae_quantiles,
            model_name=result.get("model_name", "Classification (Heuristic)"),
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantile prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global ensemble predictor instance
_ensemble_predictor = None

def get_ensemble_predictor():
    """Get or create ensemble predictor instance."""
    global _ensemble_predictor
    if _ensemble_predictor is None and model_manager.model_instances:
        try:
            from .ensemble_predictor import EnsemblePredictor
            
            # Check if model_weights.json exists to determine strict mode
            # During development/first run, weights don't exist yet
            weights_path = Path(__file__).parent.parent / "checkpoints" / "model_weights.json"
            weights_exist = weights_path.exists()
            
            if weights_exist:
                logger.info(f"[ENSEMBLE] Found model_weights.json - using strict_weights=True")
                strict_weights = True
            else:
                logger.warning(f"[ENSEMBLE] model_weights.json NOT FOUND")
                logger.warning(f"[ENSEMBLE] Using strict_weights=False for development")
                logger.warning(f"[ENSEMBLE] Run training to generate real walk-forward weights!")
                strict_weights = False
            
            _ensemble_predictor = EnsemblePredictor(
                model_instances=model_manager.model_instances,
                device=model_manager.device,
                strict_weights=strict_weights
            )
            
            if _ensemble_predictor.using_default_weights:
                logger.warning("[ENSEMBLE] ⚠️ USING DEFAULT WEIGHTS - predictions will be HOLD-heavy!")
            else:
                logger.info("[ENSEMBLE] ✓ Using real walk-forward weights for ensemble voting")
                
            logger.info("Initialized ensemble predictor")
        except Exception as e:
            logger.error(f"Failed to initialize ensemble predictor: {e}")
    return _ensemble_predictor

@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: EnsemblePredictionRequest):
    """
    Professional ensemble prediction with regime gating.
    
    This endpoint:
    1. Uses direction models (Transformer, TFT, LSTM, CNN) for voting
    2. Uses VAE for market regime detection (trend/range/chop)
    3. Uses GNN for risk regime detection (risk-on/off)
    4. Weights by walk-forward trading metrics (not accuracy)
    5. Applies confidence margin (p_top1 - p_top2) thresholds
    6. Adjusts position sizing based on regime
    """
    try:
        features = np.array(request.features)
        
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        # Validate feature count - warn if mismatch with model expectation
        actual_feature_count = features.shape[1]
        expected_feature_count = model_manager.input_dim
        
        if actual_feature_count != expected_feature_count:
            logger.error(f"FEATURE MISMATCH in /predict/ensemble: got {actual_feature_count}, expected {expected_feature_count}")
            logger.error("Consider using /predict/ensemble/candles which computes MTF features server-side")
            raise HTTPException(
                status_code=400,
                detail=f"Feature dimension mismatch: input.size(-1) must be equal to input_size. Expected {expected_feature_count}, got {actual_feature_count}. Use /predict/ensemble/candles for proper feature alignment."
            )
        
        predictor = get_ensemble_predictor()
        
        if predictor is None:
            # Fallback to basic prediction if ensemble not available
            result = model_manager.predict(features)
            return EnsemblePredictionResponse(
                action=result.get("action_name", "HOLD"),
                confidence=result["confidence"],
                confidence_margin=0.0,
                edge=0.0,
                market_regime="UNKNOWN",
                risk_regime="UNKNOWN",
                regime_confidence=0.0,
                agreement_pct=1.0,
                weighted_agreement=1.0,
                disagreement_score=0.0,
                position_size_pct=0.0,
                regime_adjusted_size=0.0,
                confidence_threshold_used=0.15,
                regime_adjustment="NONE",
                model_votes={},
                ensemble_probs={
                    "SHORT": result["probabilities"][0],
                    "HOLD": result["probabilities"][1],
                    "LONG": result["probabilities"][2]
                },
                reasons=["Ensemble predictor not initialized - using basic prediction"]
            )
        
        signal = predictor.predict(features)
        
        # === INTEGRATION: Add prediction to drift monitor ===
        # Extract probabilities as list [P(SHORT), P(HOLD), P(LONG)]
        probs_list = [
            signal.ensemble_probs.get("SHORT", 0.33),
            signal.ensemble_probs.get("HOLD", 0.34),
            signal.ensemble_probs.get("LONG", 0.33)
        ]
        action_to_idx = {"SHORT": 0, "HOLD": 1, "LONG": 2}
        predicted_action_idx = action_to_idx.get(signal.action, 1)
        drift_monitor.add_prediction(probs_list, predicted_action_idx)
        logger.debug(f"[DRIFT MONITOR] Added prediction: {signal.action} conf={signal.confidence:.3f}")
        
        return EnsemblePredictionResponse(
            action=signal.action,
            confidence=signal.confidence,
            confidence_margin=signal.confidence_margin,
            edge=signal.edge,
            market_regime=signal.market_regime,
            risk_regime=signal.risk_regime,
            regime_confidence=signal.regime_confidence,
            agreement_pct=signal.agreement_pct,
            weighted_agreement=signal.weighted_agreement,
            disagreement_score=signal.disagreement_score,
            position_size_pct=signal.position_size_pct,
            regime_adjusted_size=signal.regime_adjusted_size,
            confidence_threshold_used=signal.confidence_threshold_used,
            regime_adjustment=signal.regime_adjustment,
            model_votes=signal.model_votes,
            ensemble_probs=signal.ensemble_probs,
            reasons=signal.reasons
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ensemble/update-weights")
async def update_ensemble_weights(weights: Dict[str, Dict[str, float]]):
    """Update model weights from walk-forward evaluation results."""
    try:
        predictor = get_ensemble_predictor()
        if predictor is None:
            raise HTTPException(status_code=503, detail="Ensemble predictor not initialized")
        
        from .ensemble_predictor import ModelWeight
        new_weights = {}
        for name, metrics in weights.items():
            new_weights[name] = ModelWeight(
                model_name=name,
                expectancy=metrics.get("expectancy", 0.001),
                precision_on_trade=metrics.get("precision_on_trade", 0.55),
                profit_factor=metrics.get("profit_factor", 1.2),
                f1_directional=metrics.get("f1_directional", 0.45),
                sharpe=metrics.get("sharpe", 0.5),
                calibration_temp=metrics.get("calibration_temp", 1.0)
            )
        
        predictor.save_weights(new_weights)
        return {"message": "Weights updated", "models": list(new_weights.keys())}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ensemble/status")
async def get_ensemble_status():
    """Get ensemble predictor status and model classification."""
    predictor = get_ensemble_predictor()
    
    if predictor is None:
        return {
            "initialized": False,
            "reason": "No model instances loaded",
            "direction_models": [],
            "regime_models": [],
            "risk_models": []
        }
    
    return {
        "initialized": True,
        "direction_models": list(predictor.direction_models.keys()),
        "regime_models": list(predictor.regime_models.keys()),
        "risk_models": list(predictor.risk_models.keys()),
        "model_weights": {
            name: {
                "expectancy": w.expectancy,
                "precision_on_trade": w.precision_on_trade,
                "profit_factor": w.profit_factor,
                "f1_directional": w.f1_directional,
                "sharpe": w.sharpe,
                "composite_weight": w.composite_weight
            }
            for name, w in predictor.model_weights.items()
        },
        "thresholds": {
            "base_confidence": predictor.base_confidence_threshold,
            "base_margin": predictor.base_margin_threshold,
            "majority_weight": predictor.majority_weight_threshold
        }
    }

@app.post("/predict/ensemble/candles")
async def predict_ensemble_from_candles(request: MTFCandleData, mode: str = "stf"):
    """
    Make ensemble prediction from raw candle data.
    
    Query Parameters:
    - mode: "stf" (default) or "mtf"
      - stf: Single-TimeFrame (15m only) - uses compute_technical_features (41 features)
      - mtf: Multi-TimeFrame - uses MTF fusion (66 features) - only if model was trained on MTF
    
    This endpoint:
    - Uses 15m candles (required, 100+ candles) for STF mode
    - Produces training-compatible features (41 features for STF, 66 for MTF)
    - Errors on >15% missing features instead of silent HOLD fallback
    """
    # Validate mode parameter
    mode = mode.lower()
    if mode not in ["stf", "mtf"]:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use 'stf' or 'mtf'.")
    
    # ENFORCE STF SERVING: Refuse to serve STF requests if MTF config was loaded
    if mode == "stf" and not model_manager.stf_serving_enabled:
        logger.error("STF serving disabled due to MTF config mismatch. Cannot serve STF predictions.")
        raise HTTPException(
            status_code=503, 
            detail="STF serving disabled. Model was trained on MTF but STF mode requested. "
                   "Retrain model with STF features or use MTF mode."
        )
    
    # Check model training mode matches request mode
    if model_manager.training_mode == "STF" and mode == "mtf":
        logger.warning(f"Mode mismatch: model trained on STF but request mode is MTF. Using STF.")
        mode = "stf"
    elif model_manager.training_mode == "MTF" and mode == "stf":
        logger.warning(f"Mode mismatch: model trained on MTF but request mode is STF. Using MTF.")
        mode = "mtf"
    try:
        import pandas as pd
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Convert candles to DataFrames
        def candles_to_df(candles: List[CandleData]) -> pd.DataFrame:
            data = [{
                "datetime": pd.to_datetime(c.timestamp, unit="ms"),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            } for c in candles]
            df = pd.DataFrame(data)
            return df.sort_values("datetime").reset_index(drop=True)
        
        # Build timeframe data dict
        tf_data = {}
        
        if len(request.candles_15m) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 15m candles, got {len(request.candles_15m)}"
            )
        
        tf_data["15m"] = candles_to_df(request.candles_15m)
        
        if request.candles_5m and len(request.candles_5m) >= 50:
            tf_data["5m"] = candles_to_df(request.candles_5m)
        if request.candles_1h and len(request.candles_1h) >= 50:
            tf_data["1h"] = candles_to_df(request.candles_1h)
        if request.candles_4h and len(request.candles_4h) >= 20:
            tf_data["4h"] = candles_to_df(request.candles_4h)
        
        # === Feature mode validation and computation ===
        # Normalize mode to lowercase for comparison
        mode_lower = mode.lower()
        training_mode_lower = model_manager.training_mode.lower() if model_manager.training_mode else "stf"
        
        # CRITICAL: Validate mode matches model's training mode
        if mode_lower != training_mode_lower:
            logger.error(f"[MODE MISMATCH] Requested mode={mode_lower}, but model trained as {training_mode_lower}")
            raise HTTPException(
                status_code=400,
                detail=f"Mode mismatch: requested '{mode_lower}' but model expects '{training_mode_lower}'. "
                       f"Use ?mode={training_mode_lower} or load a compatible model."
            )
        
        logger.info(f"[/predict/ensemble/candles] Mode: {mode_lower.upper()}, Model training mode: {training_mode_lower.upper()}")
        
        if mode_lower == "stf":
            # STF mode: Use compute_technical_features (41 features)
            # Same pipeline used during training on 15m-only data
            from data.pipeline import FeatureEngineer
            fe = FeatureEngineer()
            
            # Log version for debugging train/infer parity
            logger.info(f"[STF] FeatureEngineer VERSION: {FeatureEngineer.VERSION}")
            
            # Validate version matches training if available
            if model_manager.feature_config:
                # Handle both dict (legacy) and FeatureConfig object
                if hasattr(model_manager.feature_config, 'feature_engineer_version'):
                    training_version = model_manager.feature_config.feature_engineer_version
                    is_legacy = model_manager.feature_config.is_legacy() if hasattr(model_manager.feature_config, 'is_legacy') else False
                else:
                    training_version = model_manager.feature_config.get("feature_engineer_version", "")
                    is_legacy = training_version in ('', 'legacy-unknown', 'unknown')
                
                if is_legacy:
                    # Legacy config - warn but allow inference
                    logger.warning(f"[LEGACY MODEL] Training version unknown, using current FeatureEngineer {FeatureEngineer.VERSION}")
                    logger.warning("Consider retraining to capture version tracking for production safety")
                elif training_version and training_version != FeatureEngineer.VERSION:
                    logger.error(f"[VERSION MISMATCH] Training used {training_version}, inference using {FeatureEngineer.VERSION}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"FeatureEngineer version mismatch: model trained with {training_version}, "
                               f"but current FeatureEngineer is {FeatureEngineer.VERSION}. "
                               f"This can cause silent signal degradation. Retrain with current version or rollback code."
                    )
                else:
                    logger.info(f"[VERSION OK] Training and inference both using {FeatureEngineer.VERSION}")
            
            features_df = fe.compute_technical_features(tf_data["15m"])
            computed_feature_count = len([c for c in features_df.columns if c not in ["datetime", "timestamp"]])
            logger.info(f"STF features computed using compute_technical_features: {computed_feature_count} features")
            
            # Validate STF feature count - hard error if significantly wrong
            if computed_feature_count != ModelManager.STF_FEATURE_COUNT:
                mismatch_pct = abs(computed_feature_count - ModelManager.STF_FEATURE_COUNT) / ModelManager.STF_FEATURE_COUNT
                if mismatch_pct > 0.15:  # >15% off indicates pipeline issue
                    logger.error(f"STF feature count mismatch: computed {computed_feature_count}, expected {ModelManager.STF_FEATURE_COUNT}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"STF feature pipeline error: computed {computed_feature_count} features, expected {ModelManager.STF_FEATURE_COUNT}. "
                               f"This indicates a bug in compute_technical_features()."
                    )
                else:
                    logger.warning(f"Minor STF feature count deviation: computed {computed_feature_count}, expected {ModelManager.STF_FEATURE_COUNT}")
        elif mode_lower == "mtf":
            # MTF mode: Use MTF fusion (66 features)
            # HARD GUARD: Only import MTF when explicitly requested
            # This ensures STF path never accidentally triggers MTF code
            logger.info("[MTF] Explicitly requested - importing MTFFeatureFusion")
            from data.mtf_fusion import MTFFeatureFusion
            mtf = MTFFeatureFusion()
            fused_df = mtf.align_timeframes(tf_data, request.symbol if hasattr(request, 'symbol') else "BTCUSDT")
            if fused_df is None or len(fused_df) == 0:
                raise HTTPException(status_code=400, detail="MTF fusion returned no data")
            features_df = fused_df
            computed_feature_count = len([c for c in features_df.columns if c not in ["datetime", "timestamp"]])
            logger.info(f"MTF features computed using MTFFeatureFusion: {computed_feature_count} features")
            
            # Validate MTF feature count - hard error if significantly wrong
            if computed_feature_count != ModelManager.MTF_FEATURE_COUNT:
                mismatch_pct = abs(computed_feature_count - ModelManager.MTF_FEATURE_COUNT) / ModelManager.MTF_FEATURE_COUNT
                if mismatch_pct > 0.15:  # >15% off indicates pipeline issue
                    logger.error(f"MTF feature count mismatch: computed {computed_feature_count}, expected {ModelManager.MTF_FEATURE_COUNT}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"MTF feature pipeline error: computed {computed_feature_count} features, expected {ModelManager.MTF_FEATURE_COUNT}. "
                               f"This indicates a bug in MTFFeatureFusion."
                    )
                else:
                    logger.warning(f"Minor MTF feature count deviation: computed {computed_feature_count}, expected {ModelManager.MTF_FEATURE_COUNT}")
        else:
            # Unknown mode - hard error
            logger.error(f"Unknown prediction mode: {mode_lower}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prediction mode: '{mode_lower}'. Must be 'stf' or 'mtf'."
            )
        
        # === Selective NaN handling ===
        if "close" in features_df.columns:
            features_df = features_df.dropna(subset=["close"])
        feature_cols = [c for c in features_df.columns if c not in ["datetime", "timestamp"]]
        features_df[feature_cols] = features_df[feature_cols].ffill().fillna(0.0)
        features_np = features_df[feature_cols].values
        
        logger.info(f"{mode.upper()} features computed: {features_np.shape[1]} features, {len(features_df)} rows")
        
        # Get feature column names
        # === FIX: Always use feature_cols - it's the source of truth for features_np ===
        # Previously: 'features_df.columns' in single-TF included datetime/timestamp which
        # caused feature_names length != feature dimension, breaking schema enforcement
        incoming_feature_names = list(feature_cols)
        
        # Scale features using saved scaler
        features_scaled = model_manager.transform_features(
            pd.DataFrame(features_np, columns=incoming_feature_names)
        )
        
        # === SCHEMA ENFORCEMENT (production-grade) ===
        # - Reindex to expected columns in correct order
        # - ERROR on >15% missing (not silent fill) - this indicates pipeline mismatch
        # - Enforce sequence_length = 100
        schema_stats = None
        
        logger.info(f"[STF Inference] Training mode: {model_manager.training_mode}, Expected features: {model_manager.input_dim}")
        
        if model_manager.feature_config is not None:
            from training.feature_registry import FeatureValidator
            validator = FeatureValidator(model_manager.feature_config)
            
            features_seq, schema_stats = validator.enforce_schema(
                feature_names=incoming_feature_names,
                features=features_scaled,
                fill_value=0.0
            )
            
            # Calculate missing percentage
            missing_pct = (schema_stats['missing_filled'] / schema_stats['expected_features']) * 100 if schema_stats['expected_features'] > 0 else 0
            
            logger.info(
                f"[Schema Enforcement] {schema_stats['incoming_features']} -> {schema_stats['expected_features']} features, "
                f"missing: {schema_stats['missing_filled']} ({missing_pct:.1f}%), dropped: {schema_stats['extra_dropped']}, "
                f"seq: {schema_stats['sequence_in']} -> {schema_stats['sequence_out']}"
            )
            
            # === CRITICAL: Error on >15% missing features ===
            # This indicates a pipeline mismatch - do NOT silently fill and produce garbage predictions
            # REQUIRED INFERENCE RULE: HTTP 422 with specific message
            if missing_pct > 15:
                logger.error(
                    f"[HARD ERROR] SCHEMA MISMATCH: {missing_pct:.1f}% features missing ({schema_stats['missing_filled']}/{schema_stats['expected_features']}). "
                    f"Training mode: {model_manager.training_mode}. "
                    f"Expected features: {model_manager.expected_features[:5] if model_manager.expected_features else 'unknown'}... "
                    f"Incoming features: {incoming_feature_names[:5]}..."
                )
                raise HTTPException(
                    status_code=422,
                    detail="Feature schema mismatch — wrong endpoint or retrain required"
                )
        else:
            # No feature config - use raw features with basic sequence handling
            logger.warning("No feature_config loaded - using raw features without schema enforcement")
            seq_len = min(model_manager.sequence_length, len(features_scaled))
            features_seq = features_scaled[-seq_len:]
            
            # Still validate feature count as a safety check
            actual_feature_count = features_seq.shape[1]
            expected_feature_count = model_manager.input_dim
            
            # Calculate mismatch percentage for the 15% rule
            if expected_feature_count > 0:
                missing_count = max(0, expected_feature_count - actual_feature_count)
                missing_pct = (missing_count / expected_feature_count) * 100
            else:
                missing_pct = 0 if actual_feature_count == 0 else 100
            
            # REQUIRED INFERENCE RULE: HTTP 422 if >15% missing
            if missing_pct > 15:
                logger.error(f"[HARD ERROR] Feature mismatch: computed {actual_feature_count}, model expects {expected_feature_count} ({missing_pct:.1f}% missing)")
                raise HTTPException(
                    status_code=422,
                    detail="Feature schema mismatch — wrong endpoint or retrain required"
                )
        
        # Make ensemble prediction
        predictor = get_ensemble_predictor()
        
        if predictor is None:
            # Fallback to basic prediction - ensure all numpy values are converted to Python floats
            result = model_manager.predict(features_seq)
            return {
                "action": result.get("action_name", "HOLD"),
                "confidence": float(result["confidence"]),
                "confidence_margin": 0.0,
                "edge": 0.0,
                "market_regime": "UNKNOWN",
                "risk_regime": "UNKNOWN",
                "regime_confidence": 0.0,
                "agreement_pct": 1.0,
                "weighted_agreement": 1.0,
                "disagreement_score": 0.0,
                "position_size_pct": 0.0,
                "regime_adjusted_size": 0.0,
                "confidence_threshold_used": 0.15,
                "regime_adjustment": "NONE",
                "model_votes": {},
                "ensemble_probs": {
                    "SHORT": float(result["probabilities"][0]),
                    "HOLD": float(result["probabilities"][1]),
                    "LONG": float(result["probabilities"][2])
                },
                "reasons": [f"{mode_lower.upper()} features: {features_seq.shape[1]}, used basic prediction"],
                "inference_mode": mode_lower,
                "training_mode": training_mode_lower,
                "mtf_mode": mode_lower == "mtf",
                "feature_count": int(features_seq.shape[1]),
                # Flow Forecast not available in fallback mode
                "vol_state": None,
                "vol_state_probs": None,
                "acceleration": None,
                "forecast_mode": None,
                "quantile_paths": None
            }
        
        signal = predictor.predict(features_seq)
        
        # Build schema enforcement info for response
        schema_info = []
        if schema_stats:
            if schema_stats.get('missing_filled', 0) > 0:
                schema_info.append(f"Schema: filled {schema_stats['missing_filled']} missing features")
                # Log which features were missing
                if schema_stats.get('missing_names'):
                    logger.info(f"[Schema] Missing features filled with 0: {schema_stats['missing_names']}")
            if schema_stats.get('extra_dropped', 0) > 0:
                schema_info.append(f"Schema: dropped {schema_stats['extra_dropped']} extra features")
                # Log which features were dropped
                if schema_stats.get('extra_names'):
                    logger.info(f"[Schema] Extra features dropped: {schema_stats['extra_names']}")
        
        # Convert all potential numpy values to Python native types for JSON serialization
        def to_float(val):
            """Convert numpy types to Python float."""
            if val is None:
                return None
            return float(val)
        
        def convert_probs(probs):
            """Convert probability dict values to Python floats."""
            if probs is None:
                return {}
            return {k: to_float(v) for k, v in probs.items()}
        
        def convert_quantiles(quants):
            """Convert quantile dict values to Python floats."""
            if quants is None:
                return {}
            return {k: to_float(v) for k, v in quants.items()}
        
        return {
            "action": signal.action,
            "confidence": to_float(signal.confidence),
            "confidence_margin": to_float(signal.confidence_margin),
            "edge": to_float(signal.edge),
            "market_regime": signal.market_regime,
            "risk_regime": signal.risk_regime,
            "regime_confidence": to_float(signal.regime_confidence),
            "agreement_pct": to_float(signal.agreement_pct),
            "weighted_agreement": to_float(signal.weighted_agreement),
            "disagreement_score": to_float(signal.disagreement_score),
            "position_size_pct": to_float(signal.position_size_pct),
            "regime_adjusted_size": to_float(signal.regime_adjusted_size),
            "confidence_threshold_used": to_float(signal.confidence_threshold_used),
            "regime_adjustment": signal.regime_adjustment,
            "model_votes": signal.model_votes,
            "ensemble_probs": convert_probs(signal.ensemble_probs),
            "reasons": signal.reasons + schema_info + [f"Mode: {mode_lower.upper()}, training_mode: {training_mode_lower.upper()}, features: {features_seq.shape[1]}"],
            "inference_mode": mode_lower,
            "training_mode": training_mode_lower,
            "mtf_mode": mode_lower == "mtf",
            "feature_count": int(features_seq.shape[1]),
            "schema_enforced": schema_stats is not None,
            "schema_stats": schema_stats,
            # === Multi-head outputs: quantiles, regression, trading params ===
            "quantiles": convert_quantiles(signal.quantiles),
            "mu": to_float(signal.mu),
            "sigma": to_float(signal.sigma),
            "entry_offset": to_float(signal.entry_offset),
            "sl_distance": to_float(signal.sl_distance),
            "tp_distance": to_float(signal.tp_distance),
            # === Flow Forecast outputs (vol_state, acceleration, quantile_paths) ===
            "vol_state": signal.vol_state,
            "vol_state_probs": signal.vol_state_probs,
            "acceleration": to_float(signal.acceleration),
            "forecast_mode": signal.forecast_mode,
            # Convert relative paths (multipliers) to absolute prices
            "quantile_paths": {
                k: [float(tf_data["15m"]["close"].iloc[-1] * v) for v in vals]
                for k, vals in signal.quantile_paths.items()
            } if signal.quantile_paths else None
        }
        
        # === FLOW FORECAST RESPONSE LOGGING ===
        flow_keys = ["forecast_mode", "vol_state", "vol_state_probs", "acceleration", "quantile_paths"]
        flow_present = {k: response.get(k) is not None for k in flow_keys}
        logger.info(f"[FLOW FORECAST RESPONSE] Keys present: {flow_present}")
        
        if response.get("forecast_mode"):
            logger.info(f"[FLOW FORECAST] mode={response['forecast_mode']}, vol_state={response.get('vol_state')}, "
                       f"accel={response.get('acceleration'):.4f if response.get('acceleration') else 'None'}, "
                       f"paths={'YES' if response.get('quantile_paths') else 'NO'}")
        else:
            logger.warning("[FLOW FORECAST] forecast_mode is None - model may not support flow forecast")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Ensemble prediction error (mode={mode}): {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/stf")
async def predict_stf_from_candles(request: MTFCandleData):
    """
    Dedicated Single-TimeFrame (STF) prediction endpoint.
    
    This is a convenience endpoint that:
    - ALWAYS uses STF mode (15m candles only)
    - Uses compute_technical_features (41 features)
    - Matches models trained on 15m data
    
    Use this endpoint if your model was trained on 15m-only data.
    For MTF models, use /predict/ensemble/candles?mode=mtf
    """
    # Delegate to ensemble endpoint with forced STF mode
    return await predict_ensemble_from_candles(request, mode="stf")


@app.get("/predict/mode")
async def get_prediction_mode():
    """
    Get the current model's training mode (STF or MTF).
    
    Returns the detected training mode and feature configuration.
    """
    return {
        "training_mode": model_manager.training_mode,
        "expected_features": model_manager.input_dim,
        "stf_feature_count": ModelManager.STF_FEATURE_COUNT,
        "mtf_feature_count": ModelManager.MTF_FEATURE_COUNT,
        "feature_columns": model_manager.expected_features[:10] if model_manager.expected_features else [],
        "feature_columns_count": len(model_manager.expected_features) if model_manager.expected_features else 0,
        "feature_config_loaded": model_manager.feature_config is not None,
        "recommendation": "Use /predict/stf for STF models, /predict/ensemble/candles?mode=mtf for MTF models"
    }


@app.get("/debug/feature-config")
async def get_debug_feature_config():
    """
    Debug endpoint for feature configuration inspection.
    
    Returns detailed feature config info to diagnose train/inference mismatch issues:
    - training_mode: STF or MTF
    - input_dim: Expected feature count from model
    - expected_features[0..5]: First 5 feature names for quick verification
    - feature_engineer_version: Version string from FeatureEngineer at training time
    - stf_serving_enabled: Whether STF endpoints will work
    """
    return {
        "training_mode": model_manager.training_mode,
        "input_dim": model_manager.input_dim,
        "expected_features_first_5": model_manager.expected_features[:5] if model_manager.expected_features else [],
        "expected_features_count": len(model_manager.expected_features) if model_manager.expected_features else 0,
        "feature_engineer_version": model_manager.feature_engineer_version,
        "stf_serving_enabled": model_manager.stf_serving_enabled,
        "stf_feature_count": ModelManager.STF_FEATURE_COUNT,
        "mtf_feature_count": ModelManager.MTF_FEATURE_COUNT,
        "feature_config_raw": {
            k: v for k, v in (model_manager.feature_config or {}).items() 
            if k not in ["feature_columns"]  # Exclude large arrays
        } if model_manager.feature_config else None,
        "scaler_loaded": model_manager.scaler is not None,
        "models_loaded": list(model_manager.models.keys())[:5],
        "device": model_manager.device
    }


@app.get("/debug/model-sensitivity")
async def debug_model_sensitivity():
    """
    Diagnostic endpoint to test if models respond to different inputs.
    
    Tests each model with 3 different random inputs and checks if outputs vary.
    Low variance suggests model has collapsed to constant output (training issue).
    """
    if not model_manager.model_instances:
        return {"error": "No models loaded", "diagnosis": "Cannot test model sensitivity"}
    
    device = model_manager.device
    seq_len = model_manager.sequence_length
    input_dim = model_manager.input_dim
    
    results = {}
    diagnosis = []
    
    for name, model in model_manager.model_instances.items():
        model.eval()
        model_probs = []
        
        for seed in [42, 123, 999]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            test_input = torch.randn(1, seq_len, input_dim).to(device)
            
            with torch.no_grad():
                try:
                    if hasattr(model, 'forward_multihead'):
                        output = model.forward_multihead(test_input)
                        if isinstance(output, dict):
                            logits = output.get('logits', output.get('direction', None))
                        else:
                            logits = output
                    else:
                        logits = model(test_input)
                    
                    if logits is not None and isinstance(logits, torch.Tensor):
                        probs = torch.softmax(logits, dim=-1)
                        model_probs.append(probs[0].cpu().numpy().tolist())
                except Exception as e:
                    model_probs.append(f"ERROR: {str(e)}")
        
        if len(model_probs) >= 2 and all(isinstance(p, list) for p in model_probs):
            variance = np.var(model_probs, axis=0).tolist()
            max_var = max(variance) if variance else 0
            is_collapsed = max_var < 0.001
            
            results[name] = {
                "predictions": model_probs,
                "variance_per_class": variance,
                "max_variance": max_var,
                "collapsed": is_collapsed
            }
            
            if is_collapsed:
                diagnosis.append(f"⚠️ {name}: LOW VARIANCE ({max_var:.6f}) - Model may have collapsed to constant output")
        else:
            results[name] = {"predictions": model_probs, "error": "Could not compute variance"}
    
    all_collapsed = all(r.get("collapsed", False) for r in results.values() if "collapsed" in r)
    
    return {
        "models_tested": list(results.keys()),
        "results": results,
        "diagnosis": diagnosis,
        "overall_status": "CRITICAL: All models collapsed" if all_collapsed else "SOME MODELS HEALTHY" if diagnosis else "ALL MODELS HEALTHY",
        "recommendation": "Models need retraining with balanced labels and proper loss configuration" if all_collapsed else None
    }


@app.get("/debug/label-distribution")
async def debug_label_distribution():
    """
    Check the label distribution that would be generated from typical price data.
    
    Returns distribution stats to diagnose if training labels are imbalanced.
    """
    import pandas as pd
    from data.pipeline import create_labels
    
    n = 10000
    np.random.seed(42)
    base_price = 80000
    
    returns = np.random.normal(0, 0.002, n)
    returns[1000:1500] = np.random.normal(0.004, 0.003, 500)
    returns[3000:3500] = np.random.normal(-0.004, 0.003, 500)
    returns[6000:6500] = np.random.normal(0.005, 0.004, 500)
    
    prices = base_price * np.cumprod(1 + returns)
    df = pd.DataFrame({'close': prices})
    
    labels = create_labels(df, horizon=16, threshold=0.001, trading_cost=0.0009)
    valid_labels = labels[~np.isnan(labels)]
    
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = counts.sum()
    
    distribution = {}
    for label, count in zip(unique, counts):
        label_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(int(label), str(int(label)))
        distribution[label_name] = {
            "count": int(count),
            "percentage": round(100 * count / total, 2)
        }
    
    hold_pct = distribution.get("HOLD", {}).get("percentage", 0)
    
    return {
        "total_labels": int(total),
        "distribution": distribution,
        "is_imbalanced": hold_pct > 70,
        "hold_percentage": hold_pct,
        "diagnosis": f"⚠️ HIGH HOLD RATIO ({hold_pct}%) - Training will bias toward HOLD predictions" if hold_pct > 70 else "Label distribution looks reasonable",
        "recommendation": "Use class weights or reduce threshold to balance training" if hold_pct > 70 else None
    }


@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status():
    return TrainingStatusResponse(**model_manager.training_status)

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    if model_manager.training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
        
    model_manager.update_training_status(
        is_training=True,
        current_epoch=0,
        total_epochs=request.epochs,
        current_model=request.model_type,
        progress=0.0,
        metrics={}
    )
    
    background_tasks.add_task(run_training, request)
    
    return {"message": "Training started", "model": request.model_type}

@app.post("/training/stop")
async def stop_training():
    if not model_manager.training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
        
    model_manager.update_training_status(is_training=False)
    return {"message": "Training stop requested"}

@app.get("/models")
async def list_models():
    models = []
    for name, model in model_manager.models.items():
        info = {
            "name": name,
            "loaded": True,
            "parameters": model.get("parameters", 0),
            "accuracy": model.get("accuracy", 0)
        }
        models.append(info)
    return {"models": models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    model = model_manager.get_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
    return {
        "name": model_name,
        "parameters": model.get("parameters", 0),
        "training_history": model.get("training_history", []),
        "best_accuracy": model.get("best_accuracy", 0)
    }

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    return {"predictions": model_manager.prediction_history[-limit:]}

@app.get("/metrics/performance")
async def get_performance_metrics():
    history = model_manager.prediction_history
    
    if not history:
        return {
            "total_predictions": 0,
            "accuracy": 0,
            "avg_confidence": 0,
            "action_distribution": {}
        }
        
    actions = [p["action"] for p in history]
    confidences = [p["confidence"] for p in history]
    
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
        
    return {
        "total_predictions": len(history),
        "avg_confidence": np.mean(confidences),
        "action_distribution": action_counts,
        "hold_rate": action_counts.get("HOLD", 0) / len(history) * 100
    }


@app.get("/api/drift-report")
async def get_drift_report():
    """
    Get prediction drift monitoring report.
    
    Returns PSI (Population Stability Index), ECE (Expected Calibration Error),
    and alerts for model degradation detection.
    
    2024 Best Practice: Monitor model drift in production to detect when
    retraining is needed before performance degrades significantly.
    """
    return drift_monitor.get_drift_report()


@app.post("/api/drift-report/add-prediction")
async def add_prediction_to_drift_monitor(probs: List[float], predicted_action: int, actual_outcome: Optional[int] = None):
    """
    Add a new prediction to the drift monitor.
    
    This is called automatically after each prediction in production.
    """
    drift_monitor.add_prediction(probs, predicted_action, actual_outcome)
    return {"status": "ok", "n_predictions": len(drift_monitor.prediction_probs)}


@app.post("/api/drift-report/add-outcome")
async def add_outcome_to_drift_monitor(outcome: int):
    """
    Add actual outcome for the most recent prediction (for ECE calculation).
    
    Call this when a trade is closed and we know the actual result.
    """
    drift_monitor.prediction_outcomes.append(outcome)
    return {"status": "ok", "n_outcomes": len(drift_monitor.prediction_outcomes)}


@app.post("/api/drift-report/reset")
async def reset_drift_monitor():
    """Reset the drift monitor (e.g., after retraining)."""
    global drift_monitor
    drift_monitor = PredictionDriftMonitor(window_size=100)
    return {"status": "reset", "message": "Drift monitor cleared"}


class WalkForwardRequest(BaseModel):
    n_folds: int = 5
    test_periods: int = 500
    train_periods: int = 2000
    purge_periods: int = 50
    min_confidence: float = 0.4


@app.post("/api/walk-forward/evaluate")
async def run_walk_forward_evaluation(request: WalkForwardRequest):
    """
    Run comprehensive walk-forward validation on loaded models.
    
    Returns detailed performance metrics across time folds including:
    - Per-fold Sharpe, expectancy, max drawdown
    - Per-regime performance breakdown
    - Overall summary statistics
    - Fold stability analysis
    
    2024 Best Practice: Walk-forward validation is essential for detecting
    overfitting and estimating real-world performance with proper time splits.
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        # Check if models are loaded
        if not model_manager.models:
            return {
                "status": "error",
                "error": "No models loaded. Train or load models first.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Import walk-forward evaluator
        try:
            from training.walk_forward import WalkForwardSplitter, WalkForwardEvaluator
            from data.pipeline import FeatureEngineer
        except ImportError as ie:
            return {
                "status": "error",
                "error": f"Failed to import walk-forward modules: {ie}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load candle data from parquet
        parquet_files = list((Path(__file__).parent.parent / "data").glob("*.parquet"))
        if not parquet_files:
            return {
                "status": "error",
                "error": "No parquet data files found for evaluation",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load most recent parquet file
        latest_parquet = max(parquet_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"[Walk-Forward] Loading data from {latest_parquet}")
        
        df = pd.read_parquet(latest_parquet)
        if len(df) < request.train_periods + request.test_periods + request.purge_periods:
            return {
                "status": "error",
                "error": f"Insufficient data: {len(df)} rows, need at least {request.train_periods + request.test_periods + request.purge_periods}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Engineer features
        try:
            feature_engineer = FeatureEngineer()
            features = feature_engineer.compute_features(df)
        except Exception as fe:
            logger.error(f"[Walk-Forward] Feature engineering failed: {fe}")
            return {
                "status": "error",
                "error": f"Feature engineering failed: {fe}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create splitter and evaluator
        splitter = WalkForwardSplitter(
            n_splits=request.n_folds,
            train_periods=request.train_periods,
            test_periods=request.test_periods,
            purge_periods=request.purge_periods,
            embargo_periods=10
        )
        
        evaluator = WalkForwardEvaluator(splitter=splitter)
        
        # Run evaluation for each loaded model
        model_results = {}
        
        for model_name, model_data in model_manager.models.items():
            try:
                model = model_data.get("model")
                if model is None:
                    logger.warning(f"[Walk-Forward] Skipping {model_name} - no model object")
                    continue
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"[Walk-Forward] Evaluating {model_name} on {device}...")
                
                results = evaluator.run_full_evaluation(
                    model=model,
                    candles=df,
                    features=features,
                    device=device
                )
                
                summary = evaluator.summarize_results(results)
                
                # Add per-fold details
                fold_details = []
                for r in results:
                    fold_details.append({
                        "fold_id": r.fold_id,
                        "n_trades": r.n_trades,
                        "win_rate": round(r.win_rate, 4),
                        "sharpe_ratio": round(r.sharpe_ratio, 4),
                        "expectancy": round(r.expectancy, 6),
                        "profit_factor": round(r.profit_factor, 4),
                        "max_drawdown": round(r.max_drawdown, 6),
                        "total_return": round(r.total_return, 6),
                        "regime_results": r.regime_results
                    })
                
                model_results[model_name] = {
                    "summary": {
                        "n_folds": summary["n_folds"],
                        "total_trades": summary["total_trades"],
                        "overall_win_rate": round(summary["overall_win_rate"], 4),
                        "overall_sharpe": round(summary["overall_sharpe"], 4),
                        "overall_expectancy": round(summary["overall_expectancy"], 6),
                        "overall_profit_factor": round(summary["overall_profit_factor"], 4),
                        "avg_trades_per_fold": round(summary["avg_trades_per_fold"], 2),
                        "worst_drawdown": round(summary["worst_drawdown"], 6),
                        "tail_risk": round(summary.get("tail_risk", 0), 6),
                        "sharpe_stability": round(np.std(summary["per_fold_sharpe"]), 4) if len(summary["per_fold_sharpe"]) > 1 else 0,
                        "expectancy_stability": round(np.std(summary["per_fold_expectancy"]), 6) if len(summary["per_fold_expectancy"]) > 1 else 0
                    },
                    "folds": fold_details,
                    "status": "success"
                }
                
                logger.info(f"[Walk-Forward] {model_name}: {summary['total_trades']} trades, "
                          f"sharpe={summary['overall_sharpe']:.3f}, "
                          f"expectancy={summary['overall_expectancy']:.5f}")
                
            except Exception as me:
                logger.error(f"[Walk-Forward] Error evaluating {model_name}: {me}")
                model_results[model_name] = {
                    "status": "error",
                    "error": str(me)
                }
        
        # Calculate overall summary across all models
        successful_models = [m for m, r in model_results.items() if r.get("status") == "success"]
        
        overall_summary = {
            "models_evaluated": len(successful_models),
            "best_sharpe_model": None,
            "best_expectancy_model": None,
            "recommendations": []
        }
        
        if successful_models:
            sharpe_rankings = sorted(
                [(m, model_results[m]["summary"]["overall_sharpe"]) for m in successful_models],
                key=lambda x: x[1], reverse=True
            )
            expectancy_rankings = sorted(
                [(m, model_results[m]["summary"]["overall_expectancy"]) for m in successful_models],
                key=lambda x: x[1], reverse=True
            )
            
            overall_summary["best_sharpe_model"] = sharpe_rankings[0][0] if sharpe_rankings else None
            overall_summary["best_expectancy_model"] = expectancy_rankings[0][0] if expectancy_rankings else None
            overall_summary["model_rankings_by_sharpe"] = sharpe_rankings
            overall_summary["model_rankings_by_expectancy"] = expectancy_rankings
            
            # Add recommendations
            for model, sharpe in sharpe_rankings:
                if sharpe < 0:
                    overall_summary["recommendations"].append(
                        f"{model}: Negative Sharpe ({sharpe:.3f}) - consider retraining or removing from ensemble"
                    )
                elif model_results[model]["summary"]["total_trades"] < 30:
                    overall_summary["recommendations"].append(
                        f"{model}: Low trade count ({model_results[model]['summary']['total_trades']}) - results may not be statistically significant"
                    )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_folds": request.n_folds,
                "train_periods": request.train_periods,
                "test_periods": request.test_periods,
                "purge_periods": request.purge_periods
            },
            "data_info": {
                "source": str(latest_parquet.name),
                "total_rows": len(df),
                "features": features.shape[1] if features is not None else 0
            },
            "model_results": model_results,
            "overall_summary": overall_summary
        }
        
    except Exception as e:
        logger.error(f"[Walk-Forward] Evaluation failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }


async def run_training(request: TrainingRequest):
    """
    Actual training implementation using MultiHeadTrainer.
    
    This replaces the placeholder with real model training.
    """
    import threading
    import pandas as pd
    from torch.utils.data import DataLoader
    
    try:
        logger.info(f"Starting REAL training for {request.model_type}")
        logger.info(f"  Epochs: {request.epochs}, Batch size: {request.batch_size}, LR: {request.learning_rate}")
        
        # Set training status fields at start
        # Normalize model name to match MODEL_TYPE_PATTERNS (e.g., multihead_transformer)
        model_type_raw = request.model_type.lower().replace("_multihead", "").replace("multihead_", "")
        model_type_normalized = f"multihead_{model_type_raw}"
        model_manager.update_training_status(
            is_training=True,
            current_epoch=0,
            progress=0,
            current_model=model_type_normalized,
            total_epochs=request.epochs,
            epoch_history=[],
            start_time=datetime.now().isoformat(),
            eta_seconds=None,
            health_warnings=[],
            last_update=datetime.now().isoformat(),
            per_head_losses={},
            learning_rate=None,
            best_val_loss=None,
            early_stop_counter=0
        )
        
        # Import training dependencies
        try:
            from training.multihead_trainer import MultiHeadTrainer, MultiHeadDataset
            from training.multihead_loss import MultiHeadLossConfig
            from data.pipeline import FeatureEngineer
            from data.regression_targets import RegressionTargetGenerator
            from config.training_config import TrainingConfig
        except ImportError as ie:
            logger.error(f"Failed to import training modules: {ie}")
            model_manager.update_training_status(is_training=False)
            return
        
        # Load candle data from parquet or database
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        parquet_files = list((Path(__file__).parent.parent / "data").glob("*.parquet"))
        
        if not parquet_files:
            logger.warning("No parquet files found - training requires data files")
            model_manager.update_training_status(is_training=False)
            return
        
        # Load data from parquet
        logger.info(f"Loading data from {len(parquet_files)} parquet files...")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        logger.info(f"Loaded {len(df):,} candles")
        
        # Create feature engineer and compute features
        engineer = FeatureEngineer(mode="STF")
        features_df = engineer.compute_technical_features(df)
        
        # ============== LABEL GENERATION WITH HOLD-FIX ==============
        # Parse label_mode from request (default: "regime" for best distribution)
        label_mode = getattr(request, 'label_mode', 'regime')
        use_pure_directional = (label_mode == "pure_directional")
        use_regime_labels = (label_mode == "regime")
        
        logger.info("=" * 70)
        logger.info(f"[LABEL CONFIG] Mode: {label_mode.upper()}")
        logger.info(f"  use_pure_directional: {use_pure_directional}")
        logger.info(f"  use_regime_labels: {use_regime_labels}")
        logger.info(f"  min_confidence: {request.min_confidence}")
        logger.info(f"  directional_threshold: {request.directional_threshold:.4%}")
        logger.info(f"  trend_threshold: {request.trend_threshold:.4%}")
        logger.info(f"  range_threshold: {request.range_threshold:.4%}")
        logger.info(f"  horizon: {request.horizon} bars")
        logger.info("=" * 70)
        
        # Use RegressionTargetGenerator for proper HOLD-fix labels
        target_gen = RegressionTargetGenerator(horizon_periods=request.horizon)
        targets_df = target_gen.generate_multihead_targets(
            df,
            n_future_candles=5,
            min_net_edge=0.0,
            min_confidence=request.min_confidence,
            use_volatility_cost=False,
            fixed_cost=0.0009,
            use_pure_directional=use_pure_directional,
            directional_threshold=request.directional_threshold,
            use_regime_labels=use_regime_labels,
            trend_threshold=request.trend_threshold,
            range_threshold=request.range_threshold
        )
        
        # Extract labels (already 0=SHORT, 1=HOLD, 2=LONG from RegressionTargetGenerator)
        labels = targets_df['class_label'].values
        forward_returns = targets_df['mu'].values  # mu = forward return
        
        # Get valid indices from features (after dropna)
        features_valid_mask = ~features_df.isna().any(axis=1)
        valid_indices = features_valid_mask[features_valid_mask].index.tolist()
        
        # Filter by valid labels AND valid forward_returns (not NaN or inf)
        final_valid_indices = []
        for i in valid_indices:
            if i >= len(labels) or i >= len(forward_returns):
                continue
            if np.isnan(labels[i]) or np.isnan(forward_returns[i]) or np.isinf(forward_returns[i]):
                continue
            final_valid_indices.append(i)
        
        # Extract aligned data using explicit indices
        features_np = features_df.loc[final_valid_indices].values.astype(np.float32)
        labels_np = np.array([labels[i] for i in final_valid_indices]).astype(np.int64)  # Already 0,1,2
        forward_returns_np = np.array([forward_returns[i] for i in final_valid_indices]).astype(np.float32)
        
        # ============== BUG FIX: FEATURE SCALING ==============
        # Previously features were used RAW without scaling, causing:
        # - RSI: 0-100, MACD: arbitrary, ATR: varies, Returns: -0.1 to 0.1
        # - Gradient instability due to vastly different feature scales
        # - Model learning dominated by high-magnitude features
        from sklearn.preprocessing import RobustScaler
        
        logger.info(f"[SCALING] Applying RobustScaler to {features_np.shape[1]} features...")
        feature_scaler = RobustScaler()
        features_np = feature_scaler.fit_transform(features_np).astype(np.float32)
        
        # Log feature statistics after scaling
        feature_min = np.min(features_np)
        feature_max = np.max(features_np)
        feature_mean = np.mean(features_np)
        feature_std = np.std(features_np)
        logger.info(f"[SCALING] Post-scaling stats: min={feature_min:.3f}, max={feature_max:.3f}, mean={feature_mean:.3f}, std={feature_std:.3f}")
        
        logger.info(f"Feature shape: {features_np.shape}, Labels: {len(labels_np)}")
        
        # Compute class weights
        class_counts = np.bincount(labels_np, minlength=3)
        total_samples = len(labels_np)
        MAX_CLASS_WEIGHT = 10.0
        class_weights = total_samples / (3 * class_counts + 1e-6)
        class_weights = np.clip(class_weights, 1.0, MAX_CLASS_WEIGHT)
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        # ============== LABEL DISTRIBUTION DIAGNOSTIC ==============
        hold_pct = class_counts[1] / total_samples * 100
        short_pct = class_counts[0] / total_samples * 100
        long_pct = class_counts[2] / total_samples * 100
        
        logger.info(f"[LABEL DIST] Class distribution: SHORT={class_counts[0]:,} ({short_pct:.1f}%), HOLD={class_counts[1]:,} ({hold_pct:.1f}%), LONG={class_counts[2]:,} ({long_pct:.1f}%)")
        logger.info(f"[LABEL DIST] Class weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]")
        
        # CRITICAL WARNING: If HOLD > 80%, model may learn to always predict HOLD
        if hold_pct > 80:
            logger.warning(f"[LABEL DIST] ⚠️ WARNING: HOLD class is {hold_pct:.1f}% of samples!")
            logger.warning(f"[LABEL DIST] Model may always predict HOLD. Try label_mode='regime' or 'pure_directional'")
        elif hold_pct > 60:
            logger.info(f"[LABEL DIST] Note: HOLD class is {hold_pct:.1f}% - class weights should help balance")
        else:
            logger.info(f"[LABEL DIST] ✓ Label distribution looks balanced")
        
        # Save labeling metadata for auditability
        label_distribution = {
            "short": short_pct,
            "hold": hold_pct,
            "long": long_pct,
            "short_count": int(class_counts[0]),
            "hold_count": int(class_counts[1]),
            "long_count": int(class_counts[2]),
            "total": total_samples
        }
        save_labeling_metadata(
            weights_dir="checkpoints",
            label_mode=request.label_mode,
            horizon=request.horizon,
            min_confidence=request.min_confidence,
            directional_threshold=request.directional_threshold,
            trend_threshold=request.trend_threshold,
            range_threshold=request.range_threshold,
            timeframe="15m",
            label_distribution=label_distribution
        )
        
        # Split data
        split_idx = int(len(features_np) * 0.8)
        train_features = features_np[:split_idx]
        train_labels = labels_np[:split_idx]
        train_returns = forward_returns_np[:split_idx]
        val_features = features_np[split_idx:]
        val_labels = labels_np[split_idx:]
        val_returns = forward_returns_np[split_idx:]
        
        # Create datasets
        sequence_length = 100
        train_dataset = MultiHeadDataset(
            train_features, train_labels, train_returns,
            sequence_length=sequence_length
        )
        val_dataset = MultiHeadDataset(
            val_features, val_labels, val_returns,
            sequence_length=sequence_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=request.batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
        
        # Create model
        input_dim = features_np.shape[1]
        model_type = request.model_type.lower().replace("_multihead", "").replace("multihead_", "")
        
        if model_type == "simple_mlp":
            # SimpleMLP: Stable baseline classifier (no gradient explosions)
            from models.simple_mlp import SimpleMLP, SimpleMLP_Config
            mlp_config = SimpleMLP_Config(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                n_candle_steps=5
            )
            model = SimpleMLP(mlp_config)
            model.name = "SimpleMLP"
            model.count_parameters = model.parameters_count  # Alias for compatibility
            logger.info(f"Created SimpleMLP (stable baseline) with {model.parameters_count():,} parameters")
        elif model_type == "transformer":
            from models.transformer import TransformerPriceModel
            model = TransformerPriceModel(input_dim=input_dim, d_model=128, nhead=4, num_layers=4)
            logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
        elif model_type == "tft":
            from models.transformer import TemporalFusionTransformer
            model = TemporalFusionTransformer(input_dim=input_dim, d_model=128, nhead=4)
            logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
        elif model_type == "lstm":
            from models.lstm import BidirectionalLSTM
            model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2)
            logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
        elif model_type == "cnn":
            from models.cnn import ResNetPrice
            model = ResNetPrice(input_dim=input_dim, channels=64)
            logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to simple_mlp for stability")
            from models.simple_mlp import SimpleMLP, SimpleMLP_Config
            mlp_config = SimpleMLP_Config(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                n_candle_steps=5
            )
            model = SimpleMLP(mlp_config)
            model.name = "SimpleMLP"
            model.count_parameters = model.parameters_count
            logger.info(f"Created SimpleMLP (default stable) with {model.parameters_count():,} parameters")
        
        # Create training config
        config = TrainingConfig()
        config.device = model_manager.device
        config.training.epochs = request.epochs
        config.training.learning_rate = request.learning_rate
        config.training.checkpoint_dir = str(checkpoint_dir)
        
        # Create trainer with class weights
        loss_config = MultiHeadLossConfig(class_weights=class_weights_tensor.to(config.device))
        trainer = MultiHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=config.device,
            loss_config=loss_config
        )
        
        logger.info("Starting training loop...")
        
        # ============== HARD GUARDRAILS FOR REAL TRAINING ==============
        def compute_weight_hash(model) -> str:
            """Compute hash of model weights to verify they change."""
            import hashlib
            weight_bytes = b''
            for param in model.parameters():
                weight_bytes += param.data.cpu().numpy().tobytes()
            return hashlib.md5(weight_bytes).hexdigest()
        
        def compute_weight_l2(model) -> float:
            """Compute L2 norm of weights to track changes."""
            total_norm = 0.0
            for param in model.parameters():
                total_norm += param.data.norm(2).item() ** 2
            return total_norm ** 0.5
        
        def check_gradients_nonzero(model) -> bool:
            """Verify gradients are non-zero (training is actually happening)."""
            has_nonzero_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-10:
                        has_nonzero_grad = True
                        break
            return has_nonzero_grad
        
        initial_weight_hash = compute_weight_hash(model)
        initial_weight_l2 = compute_weight_l2(model)
        logger.info(f"[GUARDRAIL] Initial weight hash: {initial_weight_hash[:16]}...")
        logger.info(f"[GUARDRAIL] Initial weight L2 norm: {initial_weight_l2:.6f}")
        
        # Run training in a separate thread to allow async status updates
        def train_thread():
            nonlocal initial_weight_hash, initial_weight_l2
            training_verified = False
            gradient_check_passed = False
            weight_change_verified = False
            
            try:
                epoch_weight_hashes = [initial_weight_hash]
                
                # Track epoch history for loss curves
                epoch_history = []
                training_start_time = time.time()
                
                def progress_callback(epoch, total_epochs, train_metrics, val_metrics):
                    nonlocal gradient_check_passed, weight_change_verified
                    
                    if isinstance(train_metrics, dict):
                        train_loss = train_metrics.get('total', 0.0)
                        train_acc = train_metrics.get('accuracy', 0.0)
                    else:
                        train_loss = float(train_metrics)
                        train_acc = 0.0
                    if isinstance(val_metrics, dict):
                        val_loss = val_metrics.get('total', 0.0)
                        val_acc = val_metrics.get('accuracy', 0.0)
                    else:
                        val_loss = float(val_metrics)
                        val_acc = 0.0
                    
                    # ============== GUARDRAIL: Check gradient flow ==============
                    health_warnings = []
                    if epoch == 0:
                        has_grad = check_gradients_nonzero(model)
                        if has_grad:
                            gradient_check_passed = True
                            logger.info(f"[GUARDRAIL] ✓ Gradients are non-zero - training is real")
                        else:
                            logger.error(f"[GUARDRAIL] ✗ CRITICAL: Gradients are ZERO - training may not be happening!")
                            health_warnings.append("CRITICAL: Gradients are ZERO")
                    
                    # ============== GUARDRAIL: Check weight changes ==============
                    current_hash = compute_weight_hash(model)
                    current_l2 = compute_weight_l2(model)
                    
                    if current_hash != epoch_weight_hashes[-1]:
                        weight_change_verified = True
                        l2_delta = abs(current_l2 - initial_weight_l2)
                        logger.info(f"[GUARDRAIL] ✓ Epoch {epoch+1}: Weights changed (L2 delta: {l2_delta:.6f})")
                    else:
                        logger.warning(f"[GUARDRAIL] ✗ Epoch {epoch+1}: Weights unchanged - possible training issue!")
                        health_warnings.append(f"Epoch {epoch+1}: Weights unchanged")
                    
                    epoch_weight_hashes.append(current_hash)
                    
                    # ============== ETA CALCULATION ==============
                    elapsed_seconds = time.time() - training_start_time
                    epochs_completed = epoch + 1
                    epochs_remaining = total_epochs - epochs_completed
                    if epochs_completed > 0:
                        seconds_per_epoch = elapsed_seconds / epochs_completed
                        eta_seconds = seconds_per_epoch * epochs_remaining
                    else:
                        eta_seconds = None
                    
                    # ============== PER-HEAD LOSSES ==============
                    per_head_losses = {}
                    if isinstance(val_metrics, dict):
                        for key in ['class', 'mu', 'sigma', 'quantile', 'trading', 'candle', 'vol_state', 'acceleration']:
                            if key in val_metrics:
                                per_head_losses[key] = float(val_metrics[key])
                    
                    # ============== EPOCH HISTORY (capped at last 50 epochs) ==============
                    epoch_entry = {
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "train_acc": float(train_acc),
                        "val_acc": float(val_acc),
                        "timestamp": datetime.now().isoformat()
                    }
                    epoch_history.append(epoch_entry)
                    # Keep only last 50 epochs to prevent memory bloat
                    if len(epoch_history) > 50:
                        epoch_history.pop(0)
                    
                    # ============== BEST VAL LOSS TRACKING ==============
                    current_best = model_manager.training_status.get("best_val_loss")
                    if current_best is None or val_loss < current_best:
                        best_val_loss = float(val_loss)
                        early_stop_counter = 0
                    else:
                        best_val_loss = current_best
                        early_stop_counter = model_manager.training_status.get("early_stop_counter", 0) + 1
                    
                    # ============== LEARNING RATE (from scheduler) ==============
                    try:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                    except:
                        current_lr = None
                    
                    # ============== LIVE PREDICTION DISTRIBUTION ==============
                    # Get actual model prediction counts from health_monitor (NOT label distribution)
                    pred_dist = {"short": 0, "hold": 0, "long": 0, "total": 0}
                    if hasattr(trainer, 'health_monitor') and trainer.health_monitor.class_counts_history:
                        # Use actual counts from latest epoch's predictions
                        latest_counts = trainer.health_monitor.class_counts_history[-1]
                        pred_dist = {
                            "short": latest_counts.get(0, 0),  # Class 0 = SHORT
                            "hold": latest_counts.get(1, 0),   # Class 1 = HOLD  
                            "long": latest_counts.get(2, 0),   # Class 2 = LONG
                            "total": latest_counts.get('total', 0)
                        }
                    
                    # ============== GRADIENT NORM ==============
                    grad_norm = train_metrics.get('gradient_norm', None) if isinstance(train_metrics, dict) else None
                    
                    model_manager.update_training_status(
                        current_epoch=epoch + 1,
                        progress=(epoch + 1) / total_epochs * 100,
                        metrics={
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "weight_l2": float(current_l2),
                            "gradient_check": gradient_check_passed,
                            "weight_change": weight_change_verified
                        },
                        epoch_history=list(epoch_history),
                        eta_seconds=eta_seconds,
                        health_warnings=health_warnings,
                        per_head_losses=per_head_losses,
                        learning_rate=current_lr,
                        best_val_loss=best_val_loss,
                        early_stop_counter=early_stop_counter,
                        prediction_distribution=pred_dist,
                        gradient_norm=float(grad_norm) if grad_norm is not None else None,
                        last_update=datetime.now().isoformat()
                    )
                
                trainer.epoch_callback = progress_callback
                
                # Configure checkpoint path for trainer
                checkpoint_path = str(checkpoint_dir / f"best_{model_type}_multihead.pt")
                
                # Record checkpoint timestamp before training
                import os
                checkpoint_exists_before = os.path.exists(checkpoint_path)
                checkpoint_mtime_before = os.path.getmtime(checkpoint_path) if checkpoint_exists_before else 0
                
                # CRITICAL: min_epochs=40 ensures multihead quantiles/vol_state/accel have enough epochs
                # patience=30 allows sufficient exploration after min_epochs reached
                # Early stopping uses val_loss ONLY - PolicySelector handles policy post-training
                trainer.train(
                    num_epochs=request.epochs,
                    early_stopping_patience=30,
                    min_epochs=40,
                    checkpoint_path=checkpoint_path,
                    save_best=True
                )
                
                # ============== GUARDRAIL: Verify checkpoint was saved ==============
                checkpoint_exists_after = os.path.exists(checkpoint_path)
                if checkpoint_exists_after:
                    checkpoint_mtime_after = os.path.getmtime(checkpoint_path)
                    if checkpoint_mtime_after > checkpoint_mtime_before:
                        logger.info(f"[GUARDRAIL] ✓ Checkpoint file updated: {checkpoint_path}")
                        
                        # Verify state_dict hash changed
                        final_hash = compute_weight_hash(model)
                        if final_hash != initial_weight_hash:
                            logger.info(f"[GUARDRAIL] ✓ Final weight hash: {final_hash[:16]}... (changed from initial)")
                            training_verified = True
                        else:
                            logger.error(f"[GUARDRAIL] ✗ CRITICAL: Weight hash unchanged after training!")
                    else:
                        logger.error(f"[GUARDRAIL] ✗ Checkpoint file NOT updated during training!")
                else:
                    logger.error(f"[GUARDRAIL] ✗ Checkpoint file does not exist after training!")
                
                # Final summary
                if training_verified and gradient_check_passed and weight_change_verified:
                    logger.info(f"[GUARDRAIL] ✓✓✓ ALL CHECKS PASSED - Training was REAL")
                else:
                    logger.error(f"[GUARDRAIL] TRAINING VERIFICATION FAILED:")
                    logger.error(f"  - Training verified: {training_verified}")
                    logger.error(f"  - Gradient check: {gradient_check_passed}")
                    logger.error(f"  - Weight change: {weight_change_verified}")
                
                logger.info(f"Training complete - model checkpoint saved via trainer to {checkpoint_path}")
                
                # Also save scaler for inference alignment
                try:
                    scaler_path = checkpoint_dir / "scaler.joblib"
                    engineer.save_scalers(str(scaler_path))
                    logger.info(f"Saved scalers to {scaler_path}")
                except Exception as se:
                    logger.warning(f"Failed to save scalers: {se}")
                
                # ============== WALK-FORWARD EVALUATION FOR ENSEMBLE WEIGHTS ==============
                # CRITICAL: Ensemble uses model_weights.json for voting. Without this, all models
                # are weighted equally which produces HOLD-heavy, unresponsive predictions.
                try:
                    logger.info(f"[WALK-FORWARD] Running OOS evaluation for {model_type}...")
                    
                    # Use last 20% of data for OOS evaluation (same split as validation)
                    oos_start = int(len(features_np) * 0.8)
                    oos_features = features_np[oos_start:]
                    oos_labels = labels_np[oos_start:]
                    oos_returns = forward_returns_np[oos_start:]
                    
                    # Run OOS predictions using SLIDING WINDOWS
                    # Each window of SEQUENCE_LENGTH candles produces ONE prediction for the last timestep
                    model.eval()
                    predictions = []
                    aligned_labels = []
                    aligned_returns = []
                    
                    SEQ_LEN = sequence_length  # 100 (from training config)
                    
                    with torch.no_grad():
                        for i in range(SEQ_LEN, len(oos_features)):
                            # Extract window [i-SEQ_LEN : i]
                            window = oos_features[i-SEQ_LEN:i]
                            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, seq_len, features]
                            
                            # Get prediction for the last timestep
                            if hasattr(model, 'forward_multihead'):
                                outputs = model.forward_multihead(window_tensor)
                                class_logits = outputs['class_logits']
                            else:
                                outputs = model(window_tensor)
                                if isinstance(outputs, dict):
                                    class_logits = outputs.get('class_logits', outputs.get('logits'))
                                else:
                                    class_logits = outputs
                            
                            # Get prediction for last timestep only
                            if class_logits.dim() == 3:
                                # [batch, seq, classes] -> take last timestep
                                last_logits = class_logits[0, -1, :]
                            else:
                                # [batch, classes]
                                last_logits = class_logits[0]
                            
                            pred = last_logits.argmax().item()
                            predictions.append(pred)
                            
                            # Align with corresponding label and forward return
                            # Window [i-SEQ_LEN:i] ends at position i-1, so:
                            # - The prediction is for what happens AFTER position i-1
                            # - Label[i-1] and forward_return[i-1] are the targets for position i-1
                            # - forward_return is computed as close.pct_change(16).shift(-16)
                            #   meaning forward_return[j] = return from j to j+16
                            # Therefore we compare prediction with label/return at i-1 (the last feature position)
                            # But labels are shifted by horizon, so we need label[i] (predicting the outcome)
                            # 
                            # CRITICAL: Since labels are created with shift(-horizon), label[j] corresponds
                            # to the direction from j to j+horizon. Window ending at i-1 predicts
                            # what happens starting at i-1, so we use label[i-1] and return[i-1]
                            aligned_labels.append(oos_labels[i-1])
                            aligned_returns.append(oos_returns[i-1])
                    
                    predictions = np.array(predictions)
                    oos_labels = np.array(aligned_labels)
                    oos_returns = np.array(aligned_returns)
                    
                    logger.info(f"[WALK-FORWARD] Generated {len(predictions)} sliding-window predictions")
                    
                    # Compute trading metrics for OOS data
                    # Direction: 0=SHORT, 1=HOLD, 2=LONG -> map to -1, 0, +1
                    direction_map = np.array([-1, 0, 1])
                    pred_directions = direction_map[predictions]
                    
                    # Filter to trades (non-HOLD predictions)
                    trade_mask = predictions != 1
                    n_trades_raw = trade_mask.sum()
                    
                    # ALWAYS save weights, even with 0 trades (user requirement)
                    # Warn if < 30 trades (HOLD-heavy model) but still save
                    MIN_TRADES_WARNING = 30
                    
                    if n_trades_raw > 0:
                        trade_returns = oos_returns[trade_mask]
                        trade_directions = pred_directions[trade_mask]
                        
                        # PnL per trade (direction * return - costs)
                        COST_PER_TRADE = 0.0009
                        trade_pnl = trade_directions * trade_returns - COST_PER_TRADE
                        
                        # Metrics
                        n_trades = len(trade_pnl)
                        wins = (trade_pnl > 0).sum()
                        win_rate = wins / n_trades if n_trades > 0 else 0.5
                        
                        mean_pnl = trade_pnl.mean() if n_trades > 0 else 0
                        std_pnl = trade_pnl.std() if n_trades > 1 else 1
                        sharpe = (mean_pnl / std_pnl * np.sqrt(252 * 4)) if std_pnl > 0 else 0  # Annualized (4 trades/day)
                        
                        avg_win = trade_pnl[trade_pnl > 0].mean() if wins > 0 else 0
                        losses = n_trades - wins
                        avg_loss = abs(trade_pnl[trade_pnl < 0].mean()) if losses > 0 else 0.0001
                        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
                        
                        # Max drawdown
                        cumulative = np.cumsum(trade_pnl)
                        running_max = np.maximum.accumulate(cumulative)
                        drawdown = running_max - cumulative
                        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
                        
                        expectancy = mean_pnl
                    else:
                        # Zero trades - use defaults
                        n_trades = 0
                        win_rate = 0.5
                        expectancy = 0.0
                        profit_factor = 1.0
                        sharpe = 0.0
                        max_drawdown = 0.0
                    
                    # Create walk-forward summary (ALWAYS)
                    wf_summary = {
                        "total_trades": int(n_trades),
                        "avg_trades_per_fold": int(n_trades),  # Single OOS fold
                        "overall_win_rate": float(win_rate),
                        "overall_expectancy": float(expectancy),
                        "overall_profit_factor": float(min(3.0, profit_factor)),  # Cap at 3
                        "overall_sharpe": float(min(3.0, sharpe)),  # Cap at 3
                        "worst_drawdown": float(max_drawdown),
                        "n_folds": 1
                    }
                    
                    # ALWAYS save to model_weights.json (regardless of trade count)
                    save_walk_forward_weights(model_type, wf_summary, str(checkpoint_dir))
                    
                    if n_trades < MIN_TRADES_WARNING:
                        logger.warning(f"[WALK-FORWARD] ⚠️ Only {n_trades} trades in OOS (< {MIN_TRADES_WARNING})")
                        logger.warning(f"  Model may be too conservative (HOLD-heavy)")
                        logger.warning(f"  Weights saved anyway - consider using --regime-labels or --pure-directional")
                    
                    logger.info(f"[WALK-FORWARD] ✓ Saved weights for {model_type}:")
                    logger.info(f"  Trades: {n_trades}, Win Rate: {win_rate:.1%}")
                    logger.info(f"  Expectancy: {expectancy:.4f}, Sharpe: {sharpe:.2f}")
                    logger.info(f"  Profit Factor: {profit_factor:.2f}, Max DD: {max_drawdown:.4f}")
                        
                except Exception as wf_err:
                    logger.error(f"[WALK-FORWARD] Failed to compute walk-forward metrics: {wf_err}")
                    import traceback
                    traceback.print_exc()
                
                # Reload models to include newly trained model
                model_manager.load_best_models()
                
            except Exception as e:
                logger.error(f"Training thread error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                model_manager.update_training_status(is_training=False)
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
        
        # Wait for training to complete (but allow async status updates)
        while thread.is_alive():
            await asyncio.sleep(1.0)
        
        logger.info(f"Training completed for {request.model_type}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        model_manager.update_training_status(is_training=False)

async def _register_with_replit():
    """Register GPU trainer's public URL (ngrok) with the Replit dashboard."""
    import os
    import httpx
    dashboard_url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    if not dashboard_url:
        return
    gpu_url = os.environ.get("GPU_SELF_URL")
    if not gpu_url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://127.0.0.1:4040/api/tunnels")
                if resp.status_code == 200:
                    tunnels = resp.json().get("tunnels", [])
                    for t in tunnels:
                        if t.get("proto") == "https":
                            gpu_url = t["public_url"].rstrip("/")
                            break
                    if not gpu_url and tunnels:
                        gpu_url = tunnels[0].get("public_url", "").rstrip("/")
        except Exception:
            pass
    if gpu_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{dashboard_url}/api/gpu/register", json={"url": gpu_url})
                if resp.status_code == 200:
                    logger.info(f"[STARTUP] Registered GPU URL with Replit: {gpu_url}")
                else:
                    logger.warning(f"[STARTUP] GPU registration failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[STARTUP] GPU registration error: {e}")
    else:
        logger.info("[STARTUP] No ngrok tunnel detected - Replit will use GPU_TRAINER_URL env var")


@app.on_event("startup")
async def startup_event():
    """Load models at server startup with STF validation."""
    logger.info("Loading models at startup...")
    model_manager.load_best_models()
    
    # STF DEPLOYMENT VALIDATION: Refuse to serve if MTF config loaded
    # This prevents "silent wrong config from old runs" issue
    stf_validation_ok = True
    validation_errors = []
    
    if model_manager.training_mode == "MTF":
        validation_errors.append(f"CRITICAL: Model trained in MTF mode but STF deployment expected")
        stf_validation_ok = False
    
    if model_manager.input_dim and model_manager.input_dim != ModelManager.STF_FEATURE_COUNT:
        if model_manager.input_dim == ModelManager.MTF_FEATURE_COUNT:
            validation_errors.append(f"CRITICAL: input_dim={model_manager.input_dim} (MTF) but STF deployment requires {ModelManager.STF_FEATURE_COUNT}")
            stf_validation_ok = False
        elif model_manager.input_dim > ModelManager.STF_FEATURE_COUNT:
            validation_errors.append(f"WARNING: input_dim={model_manager.input_dim} > STF expected {ModelManager.STF_FEATURE_COUNT}")
    
    if not stf_validation_ok:
        logger.critical("=" * 60)
        logger.critical("STF DEPLOYMENT VALIDATION FAILED")
        logger.critical("=" * 60)
        for err in validation_errors:
            logger.critical(err)
        logger.critical("")
        logger.critical("FIX: Retrain model with 15m STF features, or fix feature_config.json")
        logger.critical("STF endpoints will return errors until this is fixed.")
        logger.critical("=" * 60)
        # Mark as not serving STF
        model_manager.stf_serving_enabled = False
    else:
        model_manager.stf_serving_enabled = True
        logger.info(f"STF validation passed: mode={model_manager.training_mode}, input_dim={model_manager.input_dim}")
    
    logger.info(f"Startup complete. Device: {model_manager.device}, Models: {len(model_manager.models)}, STF Serving: {model_manager.stf_serving_enabled}")
    
    asyncio.create_task(_register_with_replit())


@app.get("/health")
async def health_check():
    """
    Health endpoint with detailed capability information.
    
    Returns supports[] array listing all available features.
    Server must poll this to understand exact disconnect reason.
    """
    supports = []
    disconnect_reasons = []
    
    # Check basic GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        supports.append("gpu")
    else:
        disconnect_reasons.append("GPU not available (CUDA not found)")
    
    # Check models loaded
    models_loaded = len(model_manager.model_instances) > 0
    if models_loaded:
        supports.append("models")
        supports.append(f"models:{len(model_manager.model_instances)}")
    else:
        disconnect_reasons.append("No model instances loaded")
    
    # Check STF serving
    if getattr(model_manager, 'stf_serving_enabled', False):
        supports.append("stf")
    else:
        disconnect_reasons.append("STF serving disabled (mode mismatch)")
    
    # Check ensemble predictor
    try:
        predictor = get_ensemble_predictor()
        if predictor is not None:
            supports.append("ensemble")
        else:
            disconnect_reasons.append("Ensemble predictor not initialized")
    except:
        disconnect_reasons.append("Ensemble predictor error")
    
    # Check flow forecast capability
    flow_capable = getattr(model_manager, 'flow_forecast_capable', False)
    if flow_capable:
        supports.append("flow_forecast")
        supports.append("vol_state")
        supports.append("acceleration")
        supports.append("quantile_paths")
    else:
        disconnect_reasons.append("Flow forecast not available (models missing vol_state_head/acceleration_head)")
    
    # Model-specific capabilities
    model_caps = getattr(model_manager, 'model_capabilities', {})
    for model_name, caps in model_caps.items():
        if caps.get("flow_forecast_ready"):
            supports.append(f"flow:{model_name}")
    
    # Scaler loaded
    if model_manager.scaler is not None:
        supports.append("scaler")
    else:
        disconnect_reasons.append("Scaler not loaded")
    
    healthy = len(disconnect_reasons) == 0 or (models_loaded and flow_capable)
    
    return {
        "status": "healthy" if healthy else "degraded",
        "gpu": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "models_loaded": len(model_manager.model_instances),
        "flow_forecast_capable": flow_capable,
        "stf_serving": getattr(model_manager, 'stf_serving_enabled', False),
        "supports": supports,
        "disconnect_reasons": disconnect_reasons,
        "model_capabilities": model_caps
    }


@app.post("/models/load")
async def load_model_endpoint(model_path: str):
    """Manually load and instantiate a specific model checkpoint."""
    try:
        path = Path(model_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(str(path), map_location=model_manager.device, weights_only=False)
        model_name = path.stem
        
        # Convert Config object to dict if needed
        raw_config = checkpoint.get("config", {})
        config_dict = model_manager._config_to_dict(raw_config)
        
        # Store metadata
        model_manager.models[model_name] = {
            "path": str(path),
            "accuracy": checkpoint.get("val_accuracy", 0),
            "epoch": checkpoint.get("epoch", 0),
            "config": config_dict,
            "parameters": checkpoint.get("parameters", 0)
        }
        
        # Instantiate model
        if "model_state_dict" in checkpoint:
            model_instance = model_manager._instantiate_model(checkpoint, model_name, checkpoint_path=str(path))
            if model_instance is not None:
                model_manager.model_instances[model_name] = model_instance
                model_manager.models[model_name]["loaded"] = True
                return {
                    "message": f"Model loaded and instantiated: {model_name}",
                    "path": str(path),
                    "accuracy": checkpoint.get("val_accuracy", 0),
                    "instantiated": True
                }
            else:
                model_manager.models[model_name]["loaded"] = False
                return {
                    "message": f"Model loaded but not instantiated: {model_name}",
                    "path": str(path),
                    "instantiated": False,
                    "warning": "Could not determine model architecture"
                }
        
        return {"message": f"Model checkpoint loaded: {model_name}", "path": str(path), "instantiated": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/expected")
async def get_expected_features():
    """Get the expected feature list for prediction alignment verification."""
    if hasattr(model_manager, 'expected_features') and model_manager.expected_features:
        return {
            "count": len(model_manager.expected_features),
            "features": model_manager.expected_features,
            "input_dim": model_manager.input_dim
        }
    return {
        "count": model_manager.input_dim,
        "features": None,
        "input_dim": model_manager.input_dim,
        "warning": "No feature list loaded - using default input_dim"
    }

@app.get("/models/status")
async def get_models_status():
    """Get detailed status of loaded models and scaler."""
    models_detail = {}
    for name, model_info in model_manager.models.items():
        models_detail[name] = {
            "checkpoint_loaded": model_info.get("loaded", False),
            "instantiated": name in model_manager.model_instances,
            "error": model_manager.instantiation_errors.get(name),
            "accuracy": model_info.get("accuracy", 0),
            "epoch": model_info.get("epoch", 0),
            "config": model_info.get("config", {}),
            "model_type": model_info.get("model_type", name)  # Standardized type
        }
    
    # Count successful vs failed instantiations
    total_checkpoints = len(model_manager.models)
    successful_instances = len(model_manager.model_instances)
    failed_instances = len(model_manager.instantiation_errors)
    
    # Determine training mode from input_dim
    # Quick training: 15m only with ~41 features
    # Full MTF: 5m/15m/1h/4h with ~66 features
    input_dim = model_manager.input_dim
    if input_dim <= 50:
        training_mode = "quick"
        training_mode_description = "Quick (15m only, ~41 features)"
    else:
        training_mode = "full"
        training_mode_description = "Full MTF (5m/15m/1h/4h, ~66 features)"
    
    return {
        "summary": {
            "checkpoints_found": total_checkpoints,
            "models_instantiated": successful_instances,
            "instantiation_failures": failed_instances,
            "ready_for_prediction": successful_instances > 0
        },
        "models_instantiated": list(model_manager.model_instances.keys()),
        "instantiation_errors": model_manager.instantiation_errors,
        "models_detail": models_detail,
        "model_type_map": model_manager.model_type_map,  # Checkpoint name -> type mapping
        "model_status_by_type": model_manager.get_model_status_by_type(),  # Dashboard-ready status
        "training_mode": training_mode,  # "quick" or "full" based on input_dim
        "training_mode_description": training_mode_description,
        "config": {
            "sequence_length": model_manager.sequence_length,
            "input_dim": model_manager.input_dim,
            "device": model_manager.device,
            "checkpoint_dir": str(model_manager.checkpoint_dir),
            "checkpoint_dir_exists": model_manager.checkpoint_dir.exists()
        },
        "scaler_loaded": model_manager.scaler is not None,
        "label_mapping": {
            "0": "SHORT",
            "1": "HOLD/NEUTRAL", 
            "2": "LONG"
        },
        "warning": "Models will return default HOLD (p=0.34 each) if no model_instances are loaded" if successful_instances == 0 else None
    }

@app.post("/bybit-proxy")
async def bybit_proxy(request: Dict[str, Any]):
    """Proxy Bybit API calls from Replit (which is geo-blocked from Bybit).
    
    Receives pre-signed requests from Replit's Bybit client and forwards them
    to api.bybit.com, returning the response. Auth signing happens on Replit;
    this endpoint just relays the request.
    """
    import httpx
    
    method = request.get("method", "GET")
    endpoint = request.get("endpoint", "")
    query_string = request.get("queryString", "")
    body = request.get("body", "")
    headers = request.get("headers", {})
    base_url = request.get("baseUrl", "https://api.bybit.com")
    
    url = f"{base_url}{endpoint}"
    if query_string:
        url += f"?{query_string}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            else:
                resp = await client.post(url, headers=headers, content=body)
        
        return resp.json()
    except Exception as e:
        logger.error(f"[BYBIT PROXY] Error forwarding to {url}: {e}")
        return {"error": str(e)}


@dataclass
class _DashboardSessionState:
    session_id: str
    next_prediction_id: int = 1
    next_cycle_id: int = 1
    next_trade_id: int = 1
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    cycle_logs: List[Dict[str, Any]] = field(default_factory=list)
    trades: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    trade_order: List[int] = field(default_factory=list)
    updated_at_ms: int = 0
    engine_hint: str = "unknown"
    account_equity_usd: float = 10000.0
    risk_per_trade_pct: float = 1.0
    base_leverage: float = 1.0
    max_leverage: float = 200.0
    auto_leverage: bool = True


_dashboard_sessions: Dict[str, _DashboardSessionState] = {}
_dashboard_max_items = 5000


def _resolve_session_id(payload: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> str:
    if session_id and str(session_id).strip():
        return str(session_id).strip()
    payload = payload or {}
    for key in ("session_id", "paper_session_id", "runner_session_id"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "default"


def _get_dashboard_session(session_id: str) -> _DashboardSessionState:
    sid = str(session_id or "default").strip() or "default"
    state = _dashboard_sessions.get(sid)
    if state is None:
        state = _DashboardSessionState(session_id=sid)
        _dashboard_sessions[sid] = state
    return state


def _trim_dashboard_state(state: _DashboardSessionState) -> None:
    if len(state.predictions) > _dashboard_max_items:
        state.predictions = state.predictions[-_dashboard_max_items:]
    if len(state.cycle_logs) > _dashboard_max_items:
        state.cycle_logs = state.cycle_logs[-_dashboard_max_items:]
    if len(state.trade_order) > _dashboard_max_items:
        keep = state.trade_order[-_dashboard_max_items:]
        keep_set = set(keep)
        state.trades = {tid: tr for tid, tr in state.trades.items() if tid in keep_set}
        state.trade_order = keep


def _latest_price_by_symbol(state: _DashboardSessionState) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in state.cycle_logs[-2000:]:
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        try:
            out[sym] = float(row.get("price", 0.0) or 0.0)
        except Exception:
            continue
    return out


_MARKET_PRICE_CACHE: Dict[str, Dict[str, Any]] = {}
_MARKET_PRICE_TTL_S = 0.9
_MARKET_CANDLE_CACHE: Dict[str, Dict[str, Any]] = {}
_MARKET_CANDLE_TTL_S = 3.0
_PAPER_MAX_LEVERAGE = 250.0
_COINBASE_PRODUCT_MAP: Dict[str, str] = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
    "ADAUSDT": "ADA-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
}
_ALLOWED_CANDLE_INTERVALS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
}


def _parse_symbol_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["BTCUSDT", "ETHUSDT"]
    out: List[str] = []
    for part in str(raw).split(","):
        sym = str(part or "").strip().upper()
        if not sym:
            continue
        if sym.endswith("-USD"):
            sym = sym.replace("-", "")
        out.append(sym)
    return sorted(set(out))[:20] or ["BTCUSDT", "ETHUSDT"]


async def _fetch_binance_batch(symbols: List[str]) -> Dict[str, float]:
    # Binance supports a JSON encoded `symbols` parameter for batch ticker price.
    if not symbols:
        return {}
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbols": json.dumps(symbols)}
    out: Dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return {}
            payload = resp.json()
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    for row in payload:
        try:
            sym = str(row.get("symbol", "")).upper()
            price = float(row.get("price", 0.0))
            if sym and np.isfinite(price) and price > 0:
                out[sym] = price
        except Exception:
            continue
    return out


async def _fetch_coinbase_price(symbol: str) -> Optional[float]:
    product = _COINBASE_PRODUCT_MAP.get(symbol)
    if not product:
        return None
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    try:
        async with httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            payload = resp.json()
    except Exception:
        return None
    try:
        px = float(payload.get("price", 0.0))
        if np.isfinite(px) and px > 0:
            return px
    except Exception:
        return None
    return None


def _normalize_candle_interval(raw: Optional[str]) -> str:
    text = str(raw or "1m").strip().lower()
    return text if text in _ALLOWED_CANDLE_INTERVALS else "1m"


def _normalize_candle_limit(raw: Any, default: int = 240) -> int:
    try:
        val = int(raw)
    except Exception:
        val = int(default)
    return int(max(30, min(1000, val)))


async def _fetch_binance_candles(symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": str(symbol).upper(), "interval": interval, "limit": int(limit)}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return []
            payload = resp.json()
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = int(row[0])
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5])
            if not all(np.isfinite(x) for x in (o, h, l, c, v)):
                continue
            if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                continue
            out.append(
                {
                    "ts": ts,
                    "open": round(o, 8),
                    "high": round(h, 8),
                    "low": round(l, 8),
                    "close": round(c, 8),
                    "volume": round(max(v, 0.0), 8),
                }
            )
        except Exception:
            continue
    return out


async def _resolve_market_prices(symbols: List[str], force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    out: Dict[str, Dict[str, Any]] = {}
    stale: List[str] = []
    ttl_ms = int(_MARKET_PRICE_TTL_S * 1000)

    for sym in symbols:
        row = _MARKET_PRICE_CACHE.get(sym)
        if row and not force_refresh and (now_ms - int(row.get("ts", 0))) <= ttl_ms:
            out[sym] = row
        else:
            stale.append(sym)

    if stale:
        binance = await _fetch_binance_batch(stale)
        for sym in stale:
            px = binance.get(sym)
            source = "binance"
            if px is None:
                px = await _fetch_coinbase_price(sym)
                source = "coinbase" if px is not None else "cache"
            if px is None:
                prior = _MARKET_PRICE_CACHE.get(sym)
                if prior:
                    out[sym] = prior
                continue
            row = {"price": round(float(px), 8), "ts": now_ms, "source": source}
            _MARKET_PRICE_CACHE[sym] = row
            out[sym] = row
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _normalize_risk_pct(value: Any, fallback: float = 1.0) -> float:
    raw = _safe_float(value, fallback)
    if raw <= 0.0:
        raw = fallback
    return _clip(raw, 0.01, 100.0)


def _session_paper_settings(state: _DashboardSessionState) -> Dict[str, Any]:
    return {
        "account_equity_usd": round(float(state.account_equity_usd), 2),
        "risk_per_trade_pct": round(float(state.risk_per_trade_pct), 4),
        "base_leverage": round(float(state.base_leverage), 4),
        "max_leverage": round(float(state.max_leverage), 4),
        "auto_leverage": bool(state.auto_leverage),
    }


def _resolve_trade_risk_and_leverage(state: _DashboardSessionState, trade: Dict[str, Any]) -> Dict[str, float]:
    risk_pct = _normalize_risk_pct(
        trade.get("risk_pct_used", trade.get("risk_pct")),
        fallback=state.risk_per_trade_pct,
    )
    base_lev = _clip(_safe_float(state.base_leverage, 1.0), 1.0, _PAPER_MAX_LEVERAGE)
    max_lev = _clip(_safe_float(state.max_leverage, 200.0), base_lev, _PAPER_MAX_LEVERAGE)

    explicit_lev = trade.get("leverage")
    if explicit_lev is None:
        for key in ("size_mult", "lane_size_mult", "position_leverage", "model_leverage"):
            if trade.get(key) is not None:
                explicit_lev = trade.get(key)
                break
    if explicit_lev is not None:
        hinted_lev = _clip(_safe_float(explicit_lev, base_lev), 1.0, _PAPER_MAX_LEVERAGE)
        if hinted_lev > max_lev:
            max_lev = hinted_lev
            state.max_leverage = max(float(state.max_leverage), float(hinted_lev))
        leverage = _clip(hinted_lev, 1.0, max_lev)
    else:
        confidence = _clip(_safe_float(trade.get("p_enter", trade.get("confidence")), 0.0), 0.0, 1.0)
        edge = abs(_safe_float(trade.get("edge", trade.get("expected_return")), 0.0))
        if bool(state.auto_leverage):
            leverage = base_lev
            leverage += max(confidence - 0.55, 0.0) * 2.0
            leverage += max(edge - 0.02, 0.0) * 3.0
            leverage = _clip(leverage, base_lev, max_lev)
        else:
            leverage = base_lev

    equity = max(_safe_float(trade.get("equity_usd"), state.account_equity_usd), 0.0)
    risk_usd = max(equity * (risk_pct / 100.0) * leverage, 0.0)
    return {
        "risk_pct_used": float(risk_pct),
        "leverage": float(leverage),
        "equity_usd_at_entry": float(equity),
        "risk_usd_used": float(risk_usd),
    }


def _normalize_engine_tag(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if "myth" in raw:
        return "mythos"
    if raw in {"v5", "v5_forecaster", "forecaster"}:
        return "v5"
    if "v5" in raw:
        return "v5"
    return "unknown"


def _infer_engine_from_payload(payload: Optional[Dict[str, Any]] = None, session_id: str = "") -> str:
    row = payload or {}
    candidates = [
        row.get("engine"),
        row.get("engine_type"),
        row.get("model_engine"),
        row.get("model_name"),
        row.get("lane"),
    ]
    for cand in candidates:
        eng = _normalize_engine_tag(cand)
        if eng != "unknown":
            return eng
    sid = str(session_id or "").lower()
    if "myth" in sid:
        return "mythos"
    if "v5" in sid:
        return "v5"
    return "unknown"


def _refresh_session_engine_hint(state: _DashboardSessionState, payload: Optional[Dict[str, Any]] = None) -> None:
    eng = _infer_engine_from_payload(payload, session_id=state.session_id)
    if eng != "unknown":
        state.engine_hint = eng


def _collect_session_models(state: _DashboardSessionState) -> List[str]:
    models = set()
    for row in state.predictions[-2000:]:
        name = str(row.get("model_name", "")).strip()
        if name:
            models.add(name)
    for row in state.cycle_logs[-2000:]:
        name = str(row.get("model_name", "")).strip()
        if name:
            models.add(name)
    for tid in state.trade_order[-4000:]:
        tr = state.trades.get(tid) or {}
        name = str(tr.get("model_name", "")).strip()
        if name:
            models.add(name)
    return sorted(models)[:40]


def _assets_snapshot(state: _DashboardSessionState) -> List[Dict[str, Any]]:
    prices = _latest_price_by_symbol(state)
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for tid in state.trade_order:
        tr = state.trades.get(tid)
        if not tr:
            continue
        sym = str(tr.get("symbol", "")).upper()
        if not sym:
            continue
        row = by_symbol.setdefault(
            sym,
            {
                "symbol": sym,
                "open": 0,
                "closed": 0,
                "wins": 0,
                "losses": 0,
                "long_taken": 0,
                "short_taken": 0,
                "net_r": 0.0,
                "fees_r": 0.0,
                "last_price": None,
            },
        )
        side = str(tr.get("side", "")).upper()
        if side == "LONG":
            row["long_taken"] += 1
        elif side == "SHORT":
            row["short_taken"] += 1
        status = str(tr.get("status", "open")).lower()
        if status == "open":
            row["open"] += 1
            continue
        row["closed"] += 1
        net_r = float(tr.get("net_r", tr.get("gross_r", 0.0)) or 0.0)
        cost_r = float(tr.get("cost_r", 0.0) or 0.0)
        row["net_r"] += net_r
        row["fees_r"] += cost_r
        if net_r > 0:
            row["wins"] += 1
        elif net_r < 0:
            row["losses"] += 1
    for sym, row in by_symbol.items():
        row["last_price"] = prices.get(sym)
        closed_n = int(row["closed"])
        row["win_rate"] = round(float(row["wins"] / max(closed_n, 1)), 4)
        row["expectancy_r"] = round(float(row["net_r"] / max(closed_n, 1)), 6)
        row["net_r"] = round(float(row["net_r"]), 6)
        row["fees_r"] = round(float(row["fees_r"]), 6)
    out = sorted(by_symbol.values(), key=lambda r: (r.get("open", 0), r.get("net_r", 0.0)), reverse=True)
    return out


def _session_summary(state: _DashboardSessionState) -> Dict[str, Any]:
    trades = [state.trades[tid] for tid in state.trade_order if tid in state.trades]
    open_trades = [t for t in trades if str(t.get("status", "open")).lower() == "open"]
    closed_trades = [t for t in trades if str(t.get("status", "")).lower() == "closed"]
    engine_counts: Dict[str, int] = {}
    for row in state.predictions[-2000:]:
        eng = _infer_engine_from_payload(row, session_id=state.session_id)
        engine_counts[eng] = engine_counts.get(eng, 0) + 1
    for row in state.cycle_logs[-2000:]:
        eng = _infer_engine_from_payload(row, session_id=state.session_id)
        engine_counts[eng] = engine_counts.get(eng, 0) + 1
    for tr in trades:
        eng = _infer_engine_from_payload(tr, session_id=state.session_id)
        engine_counts[eng] = engine_counts.get(eng, 0) + 1
    engine = state.engine_hint
    if engine_counts:
        engine = max(engine_counts.items(), key=lambda kv: kv[1])[0]
    if not engine or engine == "unknown":
        engine = _infer_engine_from_payload({}, session_id=state.session_id)

    wins = 0
    losses = 0
    total_net_r = 0.0
    total_gross_r = 0.0
    total_fee_r = 0.0
    total_fee_usd = 0.0
    total_net_usd = 0.0
    leverage_values: List[float] = []
    manual_closes = 0
    long_taken = sum(1 for t in trades if str(t.get("side", "")).upper() == "LONG")
    short_taken = sum(1 for t in trades if str(t.get("side", "")).upper() == "SHORT")
    long_success = 0
    short_success = 0
    for tr in closed_trades:
        side = str(tr.get("side", "")).upper()
        if bool(tr.get("manual_close")):
            manual_closes += 1
        gross_r = float(tr.get("gross_r", tr.get("net_r", 0.0)) or 0.0)
        cost_r = float(tr.get("cost_r", 0.0) or 0.0)
        net_r = float(tr.get("net_r", tr.get("gross_r", 0.0)) or 0.0)
        total_gross_r += gross_r
        total_fee_r += cost_r
        fee_usd = float(tr.get("pnl_usd_cost", 0.0) or 0.0)
        total_fee_usd += fee_usd
        total_net_r += net_r
        risk_usd = float(tr.get("risk_usd_used", 0.0) or 0.0)
        lev = float(tr.get("leverage", 0.0) or 0.0)
        if lev > 0.0:
            leverage_values.append(lev)
        net_usd = tr.get("pnl_usd")
        if net_usd is None:
            net_usd = net_r * risk_usd
        net_usd = float(net_usd or 0.0)
        # Recover missing USD valuation from R if needed.
        if abs(net_usd) < 1e-9 and abs(net_r) > 1e-9 and risk_usd > 0.0:
            net_usd = net_r * risk_usd
        total_net_usd += net_usd
        if net_r > 0:
            wins += 1
            if side == "LONG":
                long_success += 1
            elif side == "SHORT":
                short_success += 1
        elif net_r < 0:
            losses += 1
    closed_n = len(closed_trades)
    win_rate = float(wins / closed_n) if closed_n > 0 else 0.0
    expectancy = float(total_net_r / closed_n) if closed_n > 0 else 0.0

    gross_profit = sum(max(float(t.get("net_r", t.get("gross_r", 0.0)) or 0.0), 0.0) for t in closed_trades)
    gross_loss = sum(max(-float(t.get("net_r", t.get("gross_r", 0.0)) or 0.0), 0.0) for t in closed_trades)
    profit_factor = float(gross_profit / max(gross_loss, 1e-9)) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    if profit_factor == float("inf"):
        profit_factor = 9999.0

    prices = _latest_price_by_symbol(state)
    unrealized_r = 0.0
    unrealized_usd = 0.0
    for tr in open_trades:
        sym = str(tr.get("symbol", "")).upper()
        px = prices.get(sym)
        if px is None:
            continue
        try:
            entry = float(tr.get("entry_price") or tr.get("entryPrice") or 0.0)
            side = str(tr.get("side", "LONG")).upper()
            sl = float(tr.get("stop_loss") or tr.get("stopLoss") or entry)
            risk = abs(entry - sl)
            if risk <= 1e-9:
                continue
            rr = (px - entry) / risk if side == "LONG" else (entry - px) / risk
            unrealized_r += float(rr)
            open_risk_usd = float(tr.get("risk_usd_used", 0.0) or 0.0)
            if open_risk_usd <= 0.0:
                derived = _resolve_trade_risk_and_leverage(state, tr)
                open_risk_usd = float(derived.get("risk_usd_used", 0.0))
            unrealized_usd += float(rr) * open_risk_usd
            lev = float(tr.get("leverage", 0.0) or 0.0)
            if lev > 0.0:
                leverage_values.append(lev)
        except Exception:
            continue

    all_symbols = sorted(
        {
            str(t.get("symbol", "")).upper()
            for t in trades
            if str(t.get("symbol", "")).strip()
        }
        | {
            str(p.get("symbol", "")).upper()
            for p in state.predictions
            if str(p.get("symbol", "")).strip()
        }
        | {
            str(c.get("symbol", "")).upper()
            for c in state.cycle_logs
            if str(c.get("symbol", "")).strip()
        }
    )

    start_equity_usd = float(max(state.account_equity_usd, 0.0))
    equity_live_usd = start_equity_usd + total_net_usd + unrealized_usd
    avg_leverage = float(np.mean(leverage_values)) if leverage_values else 0.0
    max_leverage_used = float(max(leverage_values)) if leverage_values else 0.0

    return {
        "session_id": state.session_id,
        "engine": engine,
        "models": _collect_session_models(state),
        "symbols": all_symbols,
        "asset_count": len(all_symbols),
        "open_positions": len(open_trades),
        "closed_trades": closed_n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "expectancy_r": round(expectancy, 6),
        "profit_factor": round(float(profit_factor), 4),
        "gross_realized_r": round(float(total_gross_r), 6),
        "fees_r": round(float(total_fee_r), 6),
        "fees_usd": round(float(total_fee_usd), 4),
        "realized_net_r": round(total_net_r, 6),
        "unrealized_r": round(float(unrealized_r), 6),
        "paper_equity_usd": round(start_equity_usd, 2),
        "equity_live_usd": round(float(equity_live_usd), 2),
        "realized_net_usd": round(float(total_net_usd), 2),
        "unrealized_usd": round(float(unrealized_usd), 2),
        "avg_leverage": round(float(avg_leverage), 4),
        "max_leverage_used": round(float(max_leverage_used), 4),
        "paper_settings": _session_paper_settings(state),
        "long_taken": long_taken,
        "short_taken": short_taken,
        "long_success": long_success,
        "short_success": short_success,
        "long_success_rate": round(float(long_success / max(long_taken, 1)), 4),
        "short_success_rate": round(float(short_success / max(short_taken, 1)), 4),
        "manual_closes": manual_closes,
        "predictions": len(state.predictions),
        "cycle_logs": len(state.cycle_logs),
        "updated_at_ms": state.updated_at_ms,
    }


def _equity_curve(state: _DashboardSessionState) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    eq = 0.0
    peak = 0.0
    for i, tid in enumerate(state.trade_order, start=1):
        tr = state.trades.get(tid)
        if not tr or str(tr.get("status", "")).lower() != "closed":
            continue
        net_r = float(tr.get("net_r", tr.get("gross_r", 0.0)) or 0.0)
        eq += net_r
        peak = max(peak, eq)
        drawdown = eq - peak
        curve.append(
            {
                "seq": i,
                "trade_id": int(tid),
                "symbol": tr.get("symbol"),
                "net_r": round(net_r, 6),
                "equity_r": round(eq, 6),
                "drawdown_r": round(drawdown, 6),
                "exit_time": tr.get("exit_time"),
            }
        )
    return curve


@app.post("/api/gpu/push-prediction")
async def push_prediction_local(payload: Dict[str, Any], session_id: Optional[str] = Query(default=None)):
    sid = _resolve_session_id(payload, session_id=session_id)
    state = _get_dashboard_session(sid)
    pred = dict(payload)
    pred_id = state.next_prediction_id
    state.next_prediction_id += 1
    pred["id"] = pred_id
    pred["session_id"] = sid
    pred["engine"] = _infer_engine_from_payload(pred, session_id=sid)
    pred["ts"] = int(time.time() * 1000)
    state.predictions.append(pred)
    _refresh_session_engine_hint(state, pred)
    state.updated_at_ms = pred["ts"]
    _trim_dashboard_state(state)
    return {"ok": True, "id": pred_id, "session_id": sid}


@app.post("/api/live/cycle-log")
async def push_cycle_log_local(payload: Dict[str, Any], session_id: Optional[str] = Query(default=None)):
    sid = _resolve_session_id(payload, session_id=session_id)
    state = _get_dashboard_session(sid)
    row = dict(payload)
    row_id = state.next_cycle_id
    state.next_cycle_id += 1
    row["id"] = row_id
    row["session_id"] = sid
    row["engine"] = _infer_engine_from_payload(row, session_id=sid)
    row["server_ts"] = int(time.time() * 1000)
    state.cycle_logs.append(row)
    _refresh_session_engine_hint(state, row)
    state.updated_at_ms = row["server_ts"]
    _trim_dashboard_state(state)
    return {"ok": True, "id": row_id, "session_id": sid}


def _resolve_trade_context(
    trade_id: int,
    payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> tuple[_DashboardSessionState, Dict[str, Any]]:
    sid = _resolve_session_id(payload, session_id=session_id)
    trade = None
    state = _dashboard_sessions.get(sid)
    if state is not None:
        trade = state.trades.get(int(trade_id))
    if trade is None:
        for candidate in _dashboard_sessions.values():
            maybe = candidate.trades.get(int(trade_id))
            if maybe is not None and (not session_id or candidate.session_id == sid):
                state = candidate
                trade = maybe
                break
    if trade is None or state is None:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return state, trade


def _close_open_trade_record(
    *,
    state: _DashboardSessionState,
    trade: Dict[str, Any],
    exit_price: float,
    outcome: str,
    note: str,
    fee_bps: float = 8.0,
    explicit_cost_r: Optional[float] = None,
    manual_close: bool = False,
) -> Dict[str, Any]:
    symbol = str(trade.get("symbol", "")).upper()
    entry = float(trade.get("entry_price") or trade.get("entryPrice") or 0.0)
    side = str(trade.get("side", "LONG")).upper()
    initial_sl = float(trade.get("initial_sl") or trade.get("stop_loss") or trade.get("stopLoss") or entry)
    risk_abs = abs(entry - initial_sl)
    if risk_abs <= 1e-9:
        risk_abs = max(entry * 0.001, 1e-6)

    gross_r = (exit_price - entry) / risk_abs if side == "LONG" else (entry - exit_price) / risk_abs
    if explicit_cost_r is not None:
        cost_r = float(explicit_cost_r)
    else:
        cost_r = (float(fee_bps) / 10000.0) * 2.0 / (risk_abs / max(entry, 1e-9))
    net_r = gross_r - cost_r

    risk_usd = float(trade.get("risk_usd_used", 0.0) or 0.0)
    if risk_usd <= 0.0:
        risk_meta = _resolve_trade_risk_and_leverage(state, trade)
        trade["risk_pct_used"] = round(float(risk_meta["risk_pct_used"]), 4)
        trade["leverage"] = round(float(risk_meta["leverage"]), 4)
        trade["equity_usd_at_entry"] = round(float(risk_meta["equity_usd_at_entry"]), 2)
        trade["risk_usd_used"] = round(float(risk_meta["risk_usd_used"]), 2)
        risk_usd = float(trade["risk_usd_used"])

    gross_usd = round(gross_r * risk_usd, 2)
    cost_usd = round(cost_r * risk_usd, 2)
    net_usd = round(net_r * risk_usd, 2)
    now_ms = int(time.time() * 1000)
    trade.update(
        {
            "status": "closed",
            "manual_close": bool(manual_close),
            "exit_time": now_ms,
            "exit_price": round(float(exit_price), 6),
            "outcome": str(outcome),
            "exit_reason": str(note or outcome),
            "gross_r": round(float(gross_r), 6),
            "cost_r": round(float(cost_r), 6),
            "net_r": round(float(net_r), 6),
            "sized_r": round(float(net_r * float(trade.get("lane_size_mult", 1.0) or 1.0)), 6),
            "risk_usd_used": round(float(risk_usd), 2),
            "pnl_usd_gross": gross_usd,
            "pnl_usd_cost": cost_usd,
            "pnl_usd": net_usd,
            "leverage": round(float(trade.get("leverage", 1.0) or 1.0), 4),
            "engine": _infer_engine_from_payload(trade, session_id=state.session_id),
            "manager_last_action": str(outcome).lower(),
            "manager_last_action_ts": now_ms,
            "manager_symbol": symbol,
        }
    )
    state.updated_at_ms = now_ms
    _refresh_session_engine_hint(state, trade)
    return trade


async def _auto_close_open_trades_from_market(state: _DashboardSessionState) -> List[Dict[str, Any]]:
    """
    Server-side safety net:
    Auto-close open paper trades when live market price breaches SL/TP.
    """
    open_rows: List[tuple[int, Dict[str, Any]]] = []
    symbols: List[str] = []
    for tid in state.trade_order:
        tr = state.trades.get(int(tid))
        if not tr or str(tr.get("status", "open")).lower() != "open":
            continue
        sym = str(tr.get("symbol", "")).upper()
        if not sym:
            continue
        open_rows.append((int(tid), tr))
        symbols.append(sym)
    if not open_rows:
        return []

    price_rows = await _resolve_market_prices(sorted(set(symbols)), force_refresh=False)
    events: List[Dict[str, Any]] = []
    for tid, tr in open_rows:
        if str(tr.get("status", "open")).lower() != "open":
            continue
        sym = str(tr.get("symbol", "")).upper()
        row = price_rows.get(sym)
        if not row:
            continue
        try:
            px = float(row.get("price", 0.0))
            if not np.isfinite(px) or px <= 0.0:
                continue
            side = str(tr.get("side", "LONG")).upper()
            sl = float(tr.get("stop_loss") or tr.get("stopLoss") or tr.get("initial_sl") or 0.0)
            tp = float(tr.get("take_profit") or tr.get("tp2") or tr.get("take_profit_price") or 0.0)
            sl_hit = False
            tp_hit = False
            if side == "LONG":
                sl_hit = sl > 0.0 and px <= sl
                tp_hit = tp > 0.0 and px >= tp
            else:
                sl_hit = sl > 0.0 and px >= sl
                tp_hit = tp > 0.0 and px <= tp
            if not sl_hit and not tp_hit:
                continue
            outcome = "SL" if sl_hit else "TP"
            exit_px = float(sl if outcome == "SL" and sl > 0.0 else tp if tp > 0.0 else px)
            _close_open_trade_record(
                state=state,
                trade=tr,
                exit_price=exit_px,
                outcome=outcome,
                note="server_auto_barrier_close",
                fee_bps=8.0,
                explicit_cost_r=None,
                manual_close=False,
            )
            events.append(
                {
                    "trade_id": int(tid),
                    "symbol": sym,
                    "outcome": outcome,
                    "price": round(px, 6),
                    "exit_price": round(exit_px, 6),
                }
            )
        except Exception:
            continue
    return events


@app.post("/api/live/trade")
async def create_live_trade(payload: Dict[str, Any], session_id: Optional[str] = Query(default=None)):
    sid = _resolve_session_id(payload, session_id=session_id)
    state = _get_dashboard_session(sid)
    trade = dict(payload)
    trade_id = state.next_trade_id
    state.next_trade_id += 1
    trade["id"] = trade_id
    trade["session_id"] = sid
    trade["engine"] = _infer_engine_from_payload(trade, session_id=sid)
    trade.setdefault("status", "open")
    trade.setdefault("entry_time", int(time.time() * 1000))
    trade.setdefault("entry_price", trade.get("current_price"))
    risk_meta = _resolve_trade_risk_and_leverage(state, trade)
    trade["risk_pct_used"] = round(float(risk_meta["risk_pct_used"]), 4)
    trade["leverage"] = round(float(risk_meta["leverage"]), 4)
    trade["equity_usd_at_entry"] = round(float(risk_meta["equity_usd_at_entry"]), 2)
    trade["risk_usd_used"] = round(float(risk_meta["risk_usd_used"]), 2)
    trade.setdefault("pnl_usd", 0.0)
    trade.setdefault("pnl_usd_gross", 0.0)
    trade.setdefault("pnl_usd_cost", 0.0)
    state.trades[trade_id] = trade
    state.trade_order.append(trade_id)
    _refresh_session_engine_hint(state, trade)
    state.updated_at_ms = int(time.time() * 1000)
    _trim_dashboard_state(state)
    return {"id": trade_id, "session_id": sid}


@app.patch("/api/live/trade/{trade_id}")
async def update_live_trade(trade_id: int, payload: Dict[str, Any], session_id: Optional[str] = Query(default=None)):
    state, trade = _resolve_trade_context(int(trade_id), payload=payload, session_id=session_id)
    trade.update(payload or {})
    trade["id"] = int(trade_id)
    trade["session_id"] = state.session_id
    trade["engine"] = _infer_engine_from_payload(trade, session_id=state.session_id)
    if str(trade.get("status", "open")).lower() == "closed":
        risk_usd = float(trade.get("risk_usd_used", 0.0) or 0.0)
        if risk_usd <= 0.0:
            risk_meta = _resolve_trade_risk_and_leverage(state, trade)
            trade["risk_pct_used"] = round(float(risk_meta["risk_pct_used"]), 4)
            trade["leverage"] = round(float(risk_meta["leverage"]), 4)
            trade["equity_usd_at_entry"] = round(float(risk_meta["equity_usd_at_entry"]), 2)
            trade["risk_usd_used"] = round(float(risk_meta["risk_usd_used"]), 2)
            risk_usd = float(trade["risk_usd_used"])
        gross_r = float(trade.get("gross_r", trade.get("net_r", 0.0)) or 0.0)
        cost_r = float(trade.get("cost_r", 0.0) or 0.0)
        net_r = float(trade.get("net_r", gross_r - cost_r) or 0.0)
        gross_usd = float(trade.get("pnl_usd_gross", 0.0) or 0.0)
        cost_usd = float(trade.get("pnl_usd_cost", 0.0) or 0.0)
        net_usd = float(trade.get("pnl_usd", 0.0) or 0.0)
        if abs(gross_usd) < 1e-9 and abs(gross_r) > 1e-9 and risk_usd > 0.0:
            gross_usd = gross_r * risk_usd
        if abs(cost_usd) < 1e-9 and abs(cost_r) > 1e-9 and risk_usd > 0.0:
            cost_usd = cost_r * risk_usd
        if abs(net_usd) < 1e-9 and abs(net_r) > 1e-9 and risk_usd > 0.0:
            net_usd = net_r * risk_usd
        trade["pnl_usd_gross"] = round(float(gross_usd), 2)
        trade["pnl_usd_cost"] = round(float(cost_usd), 2)
        trade["pnl_usd"] = round(float(net_usd), 2)
    _refresh_session_engine_hint(state, trade)
    state.updated_at_ms = int(time.time() * 1000)
    return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade}


@app.post("/api/paper/trade/{trade_id}/manual-close")
async def manual_close_paper_trade(
    trade_id: int,
    payload: Dict[str, Any],
    session_id: Optional[str] = Query(default=None),
):
    state, trade = _resolve_trade_context(int(trade_id), payload=payload, session_id=session_id)

    if str(trade.get("status", "open")).lower() == "closed":
        return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade, "already_closed": True}

    symbol = str(trade.get("symbol", "")).upper()
    latest_px = _latest_price_by_symbol(state).get(symbol)
    entry = float(trade.get("entry_price") or trade.get("entryPrice") or 0.0)
    exit_price = float(payload.get("exit_price") or latest_px or entry or 0.0)
    _close_open_trade_record(
        state=state,
        trade=trade,
        exit_price=exit_price,
        outcome=str(payload.get("outcome") or "MANUAL_CLOSE"),
        note=str(payload.get("note") or "manual_close_dashboard"),
        fee_bps=float(payload.get("fee_bps", 8.0) or 0.0),
        explicit_cost_r=float(payload.get("cost_r")) if payload.get("cost_r") is not None else None,
        manual_close=True,
    )
    return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade}


@app.post("/api/paper/trade/{trade_id}/manager-action")
async def paper_trade_manager_action(
    trade_id: int,
    payload: Dict[str, Any],
    session_id: Optional[str] = Query(default=None),
):
    state, trade = _resolve_trade_context(int(trade_id), payload=payload, session_id=session_id)
    if str(trade.get("status", "open")).lower() == "closed":
        return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade, "already_closed": True}

    action = str(payload.get("action", "")).strip().lower()
    symbol = str(trade.get("symbol", "")).upper()
    latest_px = _latest_price_by_symbol(state).get(symbol)
    entry = float(trade.get("entry_price") or trade.get("entryPrice") or 0.0)
    side = str(trade.get("side", "LONG")).upper()
    current_sl = float(trade.get("stop_loss") or trade.get("stopLoss") or trade.get("initial_sl") or entry)
    now_ms = int(time.time() * 1000)

    if action in {"force_close", "close", "manual_close"}:
        exit_price = float(payload.get("exit_price") or latest_px or entry or 0.0)
        _close_open_trade_record(
            state=state,
            trade=trade,
            exit_price=exit_price,
            outcome=str(payload.get("outcome") or "FORCE_CLOSE"),
            note=str(payload.get("note") or "manager_force_close"),
            fee_bps=float(payload.get("fee_bps", 8.0) or 0.0),
            explicit_cost_r=float(payload.get("cost_r")) if payload.get("cost_r") is not None else None,
            manual_close=bool(payload.get("manual_close", True)),
        )
        return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade, "action": action}

    if action in {"breakeven", "move_sl"}:
        if action == "breakeven":
            target_sl = float(entry)
            note = "manager_breakeven"
        else:
            if payload.get("new_stop_loss") is None:
                raise HTTPException(status_code=400, detail="new_stop_loss is required for move_sl")
            target_sl = float(payload.get("new_stop_loss"))
            note = str(payload.get("note") or "manager_move_sl")
        if not np.isfinite(target_sl) or target_sl <= 0.0:
            raise HTTPException(status_code=400, detail="new stop loss must be a positive finite value")
        if side == "LONG":
            target_sl = max(target_sl, current_sl)
        else:
            target_sl = min(target_sl, current_sl)
        trade["stop_loss"] = round(float(target_sl), 6)
        trade["stopLoss"] = round(float(target_sl), 6)
        if action == "breakeven":
            trade["breakeven_moved"] = True
        trade["manager_last_action"] = action
        trade["manager_last_action_note"] = note
        trade["manager_last_action_ts"] = now_ms
        state.updated_at_ms = now_ms
        _refresh_session_engine_hint(state, trade)
        return {"ok": True, "id": int(trade_id), "session_id": state.session_id, "trade": trade, "action": action}

    if action == "partial_close":
        close_pct = float(payload.get("close_pct", payload.get("close_percent", 50.0)) or 50.0)
        close_pct = float(np.clip(close_pct, 1.0, 100.0))
        close_frac = close_pct / 100.0
        if close_frac >= 0.999:
            exit_price = float(payload.get("exit_price") or latest_px or entry or 0.0)
            _close_open_trade_record(
                state=state,
                trade=trade,
                exit_price=exit_price,
                outcome="PARTIAL_CLOSE_FULL",
                note=str(payload.get("note") or "manager_partial_close_full"),
                fee_bps=float(payload.get("fee_bps", 8.0) or 0.0),
                explicit_cost_r=float(payload.get("cost_r")) if payload.get("cost_r") is not None else None,
                manual_close=False,
            )
            return {
                "ok": True,
                "id": int(trade_id),
                "session_id": state.session_id,
                "trade": trade,
                "action": action,
                "close_pct": close_pct,
            }

        exit_price = float(payload.get("exit_price") or latest_px or entry or 0.0)
        initial_sl = float(trade.get("initial_sl") or trade.get("stop_loss") or trade.get("stopLoss") or entry)
        risk_abs = abs(entry - initial_sl)
        if risk_abs <= 1e-9:
            risk_abs = max(entry * 0.001, 1e-6)
        gross_r = (exit_price - entry) / risk_abs if side == "LONG" else (entry - exit_price) / risk_abs
        fee_bps = float(payload.get("fee_bps", 8.0) or 0.0)
        cost_r = (
            float(payload.get("cost_r"))
            if payload.get("cost_r") is not None
            else (fee_bps / 10000.0) * 2.0 / (risk_abs / max(entry, 1e-9))
        )
        net_r = gross_r - cost_r

        risk_usd_total = float(trade.get("risk_usd_used", 0.0) or 0.0)
        if risk_usd_total <= 0.0:
            risk_meta = _resolve_trade_risk_and_leverage(state, trade)
            trade["risk_pct_used"] = round(float(risk_meta["risk_pct_used"]), 4)
            trade["leverage"] = round(float(risk_meta["leverage"]), 4)
            trade["equity_usd_at_entry"] = round(float(risk_meta["equity_usd_at_entry"]), 2)
            trade["risk_usd_used"] = round(float(risk_meta["risk_usd_used"]), 2)
            risk_usd_total = float(trade["risk_usd_used"])
        realized_risk_usd = float(max(risk_usd_total * close_frac, 0.0))
        remaining_risk_usd = float(max(risk_usd_total - realized_risk_usd, 0.0))
        gross_usd = round(gross_r * realized_risk_usd, 2)
        cost_usd = round(cost_r * realized_risk_usd, 2)
        net_usd = round(net_r * realized_risk_usd, 2)

        child_id = state.next_trade_id
        state.next_trade_id += 1
        child = dict(trade)
        child.update(
            {
                "id": int(child_id),
                "session_id": state.session_id,
                "status": "closed",
                "partial_close": True,
                "partial_close_frac": round(float(close_frac), 6),
                "parent_trade_id": int(trade_id),
                "exit_time": now_ms,
                "exit_price": round(float(exit_price), 6),
                "outcome": "PARTIAL_CLOSE",
                "exit_reason": str(payload.get("note") or f"manager_partial_close_{close_pct:.1f}%"),
                "gross_r": round(float(gross_r), 6),
                "cost_r": round(float(cost_r), 6),
                "net_r": round(float(net_r), 6),
                "sized_r": round(float(net_r * float(trade.get("lane_size_mult", 1.0) or 1.0) * close_frac), 6),
                "risk_usd_used": round(float(realized_risk_usd), 2),
                "pnl_usd_gross": gross_usd,
                "pnl_usd_cost": cost_usd,
                "pnl_usd": net_usd,
                "manual_close": False,
                "engine": _infer_engine_from_payload(trade, session_id=state.session_id),
                "manager_last_action": "partial_close",
                "manager_last_action_ts": now_ms,
            }
        )
        state.trades[int(child_id)] = child
        state.trade_order.append(int(child_id))

        old_risk_pct = float(trade.get("risk_pct_used", 0.0) or 0.0)
        ratio = float(remaining_risk_usd / max(risk_usd_total, 1e-9))
        trade["risk_usd_used"] = round(float(remaining_risk_usd), 2)
        if old_risk_pct > 0.0:
            trade["risk_pct_used"] = round(float(old_risk_pct * ratio), 6)
        if trade.get("size_pct") is not None:
            try:
                trade["size_pct"] = round(float(float(trade.get("size_pct")) * ratio), 6)
            except Exception:
                pass
        trade["partial_close_count"] = int(trade.get("partial_close_count", 0) or 0) + 1
        trade["partial_realized_usd"] = round(float(trade.get("partial_realized_usd", 0.0) or 0.0) + float(net_usd), 2)
        trade["partial_realized_r"] = round(float(trade.get("partial_realized_r", 0.0) or 0.0) + float(net_r * close_frac), 6)
        trade["manager_last_action"] = action
        trade["manager_last_action_ts"] = now_ms
        trade["manager_last_action_note"] = str(payload.get("note") or f"partial_close_{close_pct:.1f}%")
        state.updated_at_ms = now_ms
        _refresh_session_engine_hint(state, trade)
        _trim_dashboard_state(state)
        return {
            "ok": True,
            "id": int(trade_id),
            "session_id": state.session_id,
            "trade": trade,
            "action": action,
            "close_pct": close_pct,
            "partial_trade_id": int(child_id),
            "partial_trade": child,
        }

    raise HTTPException(status_code=400, detail=f"Unsupported manager action: {action}")


@app.get("/api/paper/settings")
async def get_paper_settings(session_id: str = Query(default="default")):
    state = _get_dashboard_session(session_id)
    return {"session_id": state.session_id, "settings": _session_paper_settings(state)}


@app.post("/api/paper/settings")
async def update_paper_settings(payload: Dict[str, Any], session_id: Optional[str] = Query(default=None)):
    sid = _resolve_session_id(payload, session_id=session_id)
    state = _get_dashboard_session(sid)

    if "account_equity_usd" in payload:
        state.account_equity_usd = _clip(_safe_float(payload.get("account_equity_usd"), state.account_equity_usd), 0.0, 1e12)
    if "risk_per_trade_pct" in payload:
        state.risk_per_trade_pct = _normalize_risk_pct(payload.get("risk_per_trade_pct"), fallback=state.risk_per_trade_pct)
    if "base_leverage" in payload:
        state.base_leverage = _clip(
            _safe_float(payload.get("base_leverage"), state.base_leverage),
            1.0,
            _PAPER_MAX_LEVERAGE,
        )
    if "max_leverage" in payload:
        state.max_leverage = _clip(
            _safe_float(payload.get("max_leverage"), state.max_leverage),
            state.base_leverage,
            _PAPER_MAX_LEVERAGE,
        )
    if "auto_leverage" in payload:
        state.auto_leverage = _safe_bool(payload.get("auto_leverage"), state.auto_leverage)

    if _safe_bool(payload.get("revalue_open_positions"), True):
        for tid in state.trade_order:
            tr = state.trades.get(tid)
            if not tr or str(tr.get("status", "open")).lower() != "open":
                continue
            risk_meta = _resolve_trade_risk_and_leverage(state, tr)
            tr["risk_pct_used"] = round(float(risk_meta["risk_pct_used"]), 4)
            tr["leverage"] = round(float(risk_meta["leverage"]), 4)
            tr["equity_usd_at_entry"] = round(float(risk_meta["equity_usd_at_entry"]), 2)
            tr["risk_usd_used"] = round(float(risk_meta["risk_usd_used"]), 2)

    state.updated_at_ms = int(time.time() * 1000)
    return {
        "ok": True,
        "session_id": state.session_id,
        "settings": _session_paper_settings(state),
        "summary": _session_summary(state),
    }


@app.get("/api/paper/open-positions-summary")
async def open_positions_summary(session_id: str = Query(default="default")):
    state = _get_dashboard_session(session_id)
    await _auto_close_open_trades_from_market(state)
    positions: List[Dict[str, Any]] = []
    for tid in state.trade_order:
        tr = state.trades.get(tid)
        if not tr or str(tr.get("status", "open")).lower() != "open":
            continue
        positions.append(
            {
                "id": int(tid),
                "symbol": tr.get("symbol"),
                "side": tr.get("side", "LONG"),
                "entryPrice": float(tr.get("entry_price") or tr.get("entryPrice") or 0.0),
                "stopLoss": float(tr.get("stop_loss") or tr.get("stopLoss") or 0.0),
                "tp2": float(tr.get("take_profit") or tr.get("tp2") or tr.get("take_profit_price") or 0.0),
                "entryTs": int(tr.get("entry_time") or tr.get("entryTs") or 0),
                "signalConfidence": float(tr.get("p_enter") or tr.get("signalConfidence") or 0.0),
                "lane": tr.get("lane", "V5"),
                "lane_horizon": int(max(_safe_float(tr.get("lane_horizon", tr.get("horizon", 24)), 24.0), 1.0)),
                "leverage": float(tr.get("leverage") or 1.0),
                "riskUsd": float(tr.get("risk_usd_used") or 0.0),
            }
        )
    return {"session_id": session_id, "count": len(positions), "positions": positions}


@app.get("/api/market/ticks")
async def market_ticks(
    symbols: str = Query(default="BTCUSDT,ETHUSDT"),
    force_refresh: bool = Query(default=False),
):
    symbol_list = _parse_symbol_list(symbols)
    rows = await _resolve_market_prices(symbol_list, force_refresh=bool(force_refresh))
    prices = {sym: float(row["price"]) for sym, row in rows.items()}
    sources = {sym: str(row.get("source", "unknown")) for sym, row in rows.items()}
    latest_ts = max((int(row.get("ts", 0)) for row in rows.values()), default=int(time.time() * 1000))
    return {
        "symbols": symbol_list,
        "prices": prices,
        "sources": sources,
        "server_ts": latest_ts,
    }


@app.get("/api/market/candles")
async def market_candles(
    symbol: str = Query(default="BTCUSDT"),
    interval: str = Query(default="1m"),
    limit: int = Query(default=240, ge=30, le=1000),
    force_refresh: bool = Query(default=False),
):
    syms = _parse_symbol_list(symbol)
    sym = syms[0] if syms else "BTCUSDT"
    iv = _normalize_candle_interval(interval)
    lim = _normalize_candle_limit(limit, default=240)
    cache_key = f"{sym}|{iv}|{lim}"
    now_ms = int(time.time() * 1000)
    ttl_ms = int(_MARKET_CANDLE_TTL_S * 1000)

    cached = _MARKET_CANDLE_CACHE.get(cache_key)
    if cached and not force_refresh and (now_ms - int(cached.get("server_ts", 0))) <= ttl_ms:
        return {
            "symbol": sym,
            "interval": iv,
            "candles": list(cached.get("candles") or []),
            "source": str(cached.get("source") or "cache"),
            "server_ts": int(cached.get("server_ts") or now_ms),
        }

    candles = await _fetch_binance_candles(sym, iv, lim)
    source = "binance" if candles else "unavailable"
    if not candles and cached:
        candles = list(cached.get("candles") or [])
        source = "cache"

    payload = {
        "symbol": sym,
        "interval": iv,
        "candles": candles[-lim:] if candles else [],
        "source": source,
        "server_ts": now_ms,
    }
    if payload["candles"]:
        _MARKET_CANDLE_CACHE[cache_key] = payload
    return payload


@app.post("/api/dashboard/session/reset")
async def dashboard_reset_session(
    session_id: str = Query(default="default"),
    keep_settings: bool = Query(default=True),
):
    sid = str(session_id or "default").strip() or "default"
    prev = _get_dashboard_session(sid)
    cleared = {
        "predictions": len(prev.predictions),
        "cycle_logs": len(prev.cycle_logs),
        "trades": len(prev.trade_order),
    }

    restored_settings = _session_paper_settings(prev)
    fresh = _DashboardSessionState(session_id=sid)
    if bool(keep_settings):
        fresh.account_equity_usd = float(restored_settings.get("account_equity_usd", fresh.account_equity_usd))
        fresh.risk_per_trade_pct = float(restored_settings.get("risk_per_trade_pct", fresh.risk_per_trade_pct))
        fresh.base_leverage = float(restored_settings.get("base_leverage", fresh.base_leverage))
        fresh.max_leverage = float(restored_settings.get("max_leverage", fresh.max_leverage))
        fresh.auto_leverage = bool(restored_settings.get("auto_leverage", fresh.auto_leverage))
    _dashboard_sessions[sid] = fresh
    return {
        "ok": True,
        "session_id": sid,
        "cleared": cleared,
        "settings": _session_paper_settings(fresh),
    }


@app.get("/api/dashboard/sessions")
async def dashboard_sessions(engine: Optional[str] = Query(default=None)):
    engine_filter = _normalize_engine_tag(engine) if engine else None
    sessions = []
    for sid in sorted(_dashboard_sessions.keys()):
        summary = _session_summary(_dashboard_sessions[sid])
        if engine_filter and engine_filter != "unknown" and summary.get("engine") != engine_filter:
            continue
        sessions.append(summary)
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/api/dashboard/state")
async def dashboard_state(
    session_id: str = Query(default="default"),
    engine: Optional[str] = Query(default=None),
    predictions_limit: int = Query(default=250, ge=1, le=5000),
    cycles_limit: int = Query(default=600, ge=1, le=5000),
    trades_limit: int = Query(default=1000, ge=1, le=5000),
):
    state = _get_dashboard_session(session_id)
    await _auto_close_open_trades_from_market(state)
    summary = _session_summary(state)
    engine_filter = _normalize_engine_tag(engine) if engine else None
    if (
        engine_filter
        and engine_filter != "unknown"
        and summary.get("engine") not in {engine_filter, "unknown"}
    ):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' is not tagged as engine={engine_filter}")
    trades = [state.trades[tid] for tid in state.trade_order if tid in state.trades]
    curve = _equity_curve(state)
    max_dd = min((pt["drawdown_r"] for pt in curve), default=0.0)
    return {
        "summary": {**summary, "max_drawdown_r": round(float(max_dd), 6)},
        "predictions": state.predictions[-predictions_limit:],
        "cycle_logs": state.cycle_logs[-cycles_limit:],
        "trades": trades[-trades_limit:],
        "equity_curve": curve[-3000:],
        "assets": _assets_snapshot(state),
        "models": _collect_session_models(state),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_html():
    html_path = Path(__file__).parent / "dashboard_engine.html"
    if not html_path.exists():
        html_path = Path(__file__).parent / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<html><body><h1>Dashboard not found</h1><p>Create api/dashboard.html</p></body></html>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/dashboard/v5", response_class=HTMLResponse)
async def dashboard_html_v5():
    return await dashboard_html()


@app.get("/dashboard/mythos", response_class=HTMLResponse)
async def dashboard_html_mythos():
    return await dashboard_html()


@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    return HTMLResponse(content="<html><body><meta http-equiv='refresh' content='0; url=/dashboard/v5' /></body></html>")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
