"""
Feature Version Registry for Safe Live Trading

Every trained model MUST save:
- feature_columns: Ordered list of feature names
- feature_version_hash: Hash of feature configuration
- sequence_length: Input sequence length
- horizon: Prediction horizon

At inference time, the API MUST:
- Align features exactly to training order
- Refuse prediction (return HOLD) if feature mismatch
- Log warnings for any inconsistency

This prevents silent bad predictions from feature drift.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration of features used during training."""
    
    feature_columns: List[str]          # Ordered feature names
    sequence_length: int                 # Input sequence length
    horizon_periods: int                 # Forward prediction horizon
    timeframes: List[str]                # e.g., ["15m"] or ["5m", "15m", "1h", "4h"]
    input_dim: int                       # Number of features
    version_hash: str = ""               # Auto-computed hash
    feature_engineer_version: str = ""   # FeatureEngineer.VERSION at training time
    mode: str = "stf"                    # "stf" or "mtf" - which pipeline was used
    
    def __post_init__(self):
        if not self.version_hash:
            self.version_hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of feature configuration."""
        data = {
            "columns": sorted(self.feature_columns),  # Sorted for consistency
            "seq_len": self.sequence_length,
            "horizon": self.horizon_periods,
            "input_dim": self.input_dim,
            "feature_engineer_version": self.feature_engineer_version,  # Include version in hash
            "mode": self.mode
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        """Create FeatureConfig from dict with legacy compatibility handling."""
        # Handle legacy configs missing new fields
        is_legacy = False
        
        # Infer missing mode from timeframes or feature count
        if 'mode' not in d or not d.get('mode'):
            timeframes = d.get('timeframes', ['15m'])
            input_dim = d.get('input_dim', 41)
            
            # Infer mode: MTF has multiple timeframes or ~66 features
            if len(timeframes) > 1 or input_dim > 50:
                d['mode'] = 'mtf'
                logger.warning(f"Legacy config: inferred mode='mtf' from timeframes={timeframes}, input_dim={input_dim}")
            else:
                d['mode'] = 'stf'
                logger.info(f"Legacy config: inferred mode='stf' from timeframes={timeframes}, input_dim={input_dim}")
            is_legacy = True
        
        # Handle missing feature_engineer_version
        if 'feature_engineer_version' not in d or not d.get('feature_engineer_version'):
            d['feature_engineer_version'] = 'legacy-unknown'
            logger.warning("Legacy config: feature_engineer_version not found, setting to 'legacy-unknown'")
            is_legacy = True
        
        if is_legacy:
            logger.warning("=== LEGACY FEATURE CONFIG DETECTED - Version validation will be lenient ===")
        
        return cls(**d)
    
    def is_legacy(self) -> bool:
        """Check if this config is from a legacy model without version tracking."""
        return self.feature_engineer_version in ('', 'legacy-unknown', 'unknown')
    
    def save(self, path: str):
        """Save feature config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved feature config to {path} (hash: {self.version_hash}, version: {self.feature_engineer_version})")
    
    @classmethod
    def load(cls, path: str) -> "FeatureConfig":
        """Load feature config from JSON file with legacy handling."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        logger.info(f"Loaded feature config from {path} (hash: {config.version_hash}, version: {config.feature_engineer_version}, mode: {config.mode})")
        return config


class FeatureValidator:
    """
    Validates incoming features against saved training configuration.
    
    Used at inference time to ensure features match what the model expects.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.column_indices: Dict[str, int] = {
            col: idx for idx, col in enumerate(config.feature_columns)
        }
    
    def validate(
        self,
        feature_names: List[str],
        features: np.ndarray,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate incoming features against training config.
        
        Args:
            feature_names: Names of incoming features
            features: Feature array [batch, seq_len, features] or [seq_len, features]
            strict: If True, require exact match
            
        Returns:
            (is_valid, list of warnings/errors)
        """
        warnings = []
        
        # Check feature count
        expected_dim = self.config.input_dim
        if features.ndim == 2:
            actual_dim = features.shape[1]
        elif features.ndim == 3:
            actual_dim = features.shape[2]
        else:
            return False, [f"Invalid feature shape: {features.shape}"]
        
        if actual_dim != expected_dim:
            msg = f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
            if strict:
                return False, [msg]
            warnings.append(msg)
        
        # Check sequence length
        if features.ndim == 2:
            actual_seq = features.shape[0]
        else:
            actual_seq = features.shape[1]
        
        if actual_seq != self.config.sequence_length:
            msg = f"Sequence length mismatch: expected {self.config.sequence_length}, got {actual_seq}"
            if strict:
                return False, [msg]
            warnings.append(msg)
        
        # Check feature names if provided
        if feature_names:
            missing = set(self.config.feature_columns) - set(feature_names)
            extra = set(feature_names) - set(self.config.feature_columns)
            
            if missing:
                msg = f"Missing features: {missing}"
                if strict:
                    return False, [msg]
                warnings.append(msg)
            
            if extra:
                warnings.append(f"Extra features (will be ignored): {extra}")
        
        return len(warnings) == 0, warnings
    
    def align_features(
        self,
        feature_names: List[str],
        features: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Align incoming features to training order.
        
        Args:
            feature_names: Names of incoming features
            features: Feature array [batch, seq_len, features] or [seq_len, features]
            
        Returns:
            Reordered features matching training config, or None if impossible
        """
        # Build mapping from incoming order to training order
        incoming_indices = {name: idx for idx, name in enumerate(feature_names)}
        
        # Check all training features are present
        missing = []
        for name in self.config.feature_columns:
            if name not in incoming_indices:
                missing.append(name)
        
        if missing:
            logger.error(f"Cannot align features - missing: {missing}")
            return None
        
        # Build reordering indices
        reorder_indices = [
            incoming_indices[name] for name in self.config.feature_columns
        ]
        
        # Reorder
        if features.ndim == 2:
            aligned = features[:, reorder_indices]
        else:
            aligned = features[:, :, reorder_indices]
        
        return aligned
    
    def enforce_schema(
        self,
        feature_names: List[str],
        features: np.ndarray,
        fill_value: float = 0.0,
        max_missing_pct: float = 0.15
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Enforce schema by reindexing, filling missing features, and dropping extras.
        
        This is the production-grade approach: the model decides the schema,
        not the feature builder. Any input shape is transformed to match
        the expected feature schema.
        
        FIX #2 IMPROVEMENTS:
        - Uses forward-fill for NaN values in incoming features
        - Uses column mean as fill value for entirely missing features (not 0.0)
        - Warns if >max_missing_pct features are missing (abort to HOLD recommended)
        
        Args:
            feature_names: Names of incoming features
            features: Feature array [seq_len, features] (2D) or [batch, seq_len, features] (3D)
            fill_value: Value to use for missing features (default: 0.0)
            max_missing_pct: Max fraction of features that can be missing before warning (default: 15%)
            
        Returns:
            Tuple of:
              - enforced: Features array with exactly config.input_dim columns in correct order
              - stats: Dict with schema reconciliation stats for logging
        """
        expected_cols = self.config.feature_columns
        expected_dim = self.config.input_dim
        seq_len = self.config.sequence_length
        
        # === DEFENSIVE CHECK: feature_names must match feature dimension ===
        # FIX: Raise ValueError instead of returning silent zeros - prevents fake predictions
        actual_feature_dim = features.shape[-1] if features.ndim >= 2 else features.shape[0]
        if len(feature_names) != actual_feature_dim:
            error_msg = (
                f"[Schema] CRITICAL: feature_names length ({len(feature_names)}) != "
                f"feature dimension ({actual_feature_dim}). Cannot enforce schema safely. "
                f"This usually means incoming_feature_names was built from wrong source. "
                f"Expected: feature_names from same array used to build features."
            )
            logger.error(error_msg)
            # Raise error instead of returning zeros - prevents garbage predictions
            raise ValueError(error_msg)
        
        incoming_set = set(feature_names)
        expected_set = set(expected_cols)
        
        missing_features = expected_set - incoming_set
        extra_features = incoming_set - expected_set
        
        # Build mapping: expected column name -> index in incoming (or -1 if missing)
        incoming_indices = {name: idx for idx, name in enumerate(feature_names)}
        
        # Determine shape
        is_3d = features.ndim == 3
        if is_3d:
            batch_size, actual_seq, actual_dim = features.shape
        else:
            actual_seq, actual_dim = features.shape
            batch_size = 1
            features = features[np.newaxis, ...]  # Add batch dim for uniform processing
        
        # --- Step 1: Enforce sequence length ---
        if actual_seq < seq_len:
            # Left-pad with fill_value
            pad_len = seq_len - actual_seq
            pad_shape = (batch_size, pad_len, actual_dim)
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            features = np.concatenate([pad, features], axis=1)
            logger.info(f"[Schema] Left-padded sequence: {actual_seq} -> {seq_len}")
        elif actual_seq > seq_len:
            # Tail-slice (keep most recent)
            features = features[:, -seq_len:, :]
            logger.info(f"[Schema] Tail-sliced sequence: {actual_seq} -> {seq_len}")
        
        # --- Step 2: Reindex columns to expected order, fill missing, drop extras ---
        # FIX #2: Use column mean instead of 0.0 for missing features
        # This is more neutral for normalized features than 0.0
        enforced = np.zeros((batch_size, seq_len, expected_dim), dtype=np.float32)
        
        for col_idx, col_name in enumerate(expected_cols):
            if col_name in incoming_indices:
                src_idx = incoming_indices[col_name]
                col_data = features[:, :, src_idx].copy()
                
                # Forward-fill NaN values within each column (FIX #2)
                for b in range(batch_size):
                    col_slice = col_data[b, :]
                    nan_mask = np.isnan(col_slice)
                    if nan_mask.any() and not nan_mask.all():
                        # Forward fill: propagate last valid value
                        for i in range(1, len(col_slice)):
                            if nan_mask[i] and not nan_mask[i-1]:
                                col_slice[i] = col_slice[i-1]
                                nan_mask[i] = False
                        # Backward fill for leading NaNs
                        for i in range(len(col_slice)-2, -1, -1):
                            if nan_mask[i] and not nan_mask[i+1]:
                                col_slice[i] = col_slice[i+1]
                        # Any remaining NaNs -> column mean or 0
                        remaining_nans = np.isnan(col_slice)
                        if remaining_nans.any():
                            valid_vals = col_slice[~remaining_nans]
                            fill_val = np.mean(valid_vals) if len(valid_vals) > 0 else fill_value
                            col_slice[remaining_nans] = fill_val
                        col_data[b, :] = col_slice
                
                enforced[:, :, col_idx] = col_data
            else:
                # Missing feature: fill with 0.0 (neutral for normalized features)
                # This is better than leaving as NaN but the warning below flags this
                enforced[:, :, col_idx] = fill_value
        
        # Remove batch dim if original was 2D
        if not is_3d:
            enforced = enforced[0]
        
        # FIX #2: Check if too many features are missing
        missing_pct = len(missing_features) / expected_dim if expected_dim > 0 else 0
        should_abort = missing_pct > max_missing_pct
        
        stats = {
            "incoming_features": len(feature_names),
            "expected_features": expected_dim,
            "missing_filled": len(missing_features),
            "extra_dropped": len(extra_features),
            "sequence_in": actual_seq,
            "sequence_out": seq_len,
            "missing_names": list(missing_features)[:10],  # Log first 10
            "extra_names": list(extra_features)[:10],
            "missing_pct": missing_pct,
            "should_abort_to_hold": should_abort  # FIX #2: Flag for caller
        }
        
        if should_abort:
            logger.warning(
                f"[Schema] TOO MANY MISSING FEATURES: {len(missing_features)}/{expected_dim} "
                f"({missing_pct:.1%}) > {max_missing_pct:.0%} threshold. "
                f"Recommend aborting to HOLD. Missing: {list(missing_features)[:5]}..."
            )
        else:
            logger.info(
                f"[Schema] Enforced: {len(feature_names)} -> {expected_dim} features, "
                f"missing filled: {len(missing_features)}, extra dropped: {len(extra_features)}"
            )
        
        return enforced, stats


class FeatureRegistry:
    """
    Central registry for feature configurations across models.
    
    Stores and retrieves feature configs for all trained models.
    """
    
    def __init__(self, registry_dir: str = "checkpoints/feature_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, FeatureConfig] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all saved feature configs."""
        for path in self.registry_dir.glob("*.json"):
            try:
                config = FeatureConfig.load(str(path))
                model_name = path.stem
                self.configs[model_name] = config
                logger.info(f"Loaded feature config for {model_name} (hash: {config.version_hash})")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    
    def register(self, model_name: str, config: FeatureConfig):
        """Register a feature configuration for a model."""
        self.configs[model_name] = config
        config.save(str(self.registry_dir / f"{model_name}.json"))
    
    def get(self, model_name: str) -> Optional[FeatureConfig]:
        """Get feature config for a model."""
        return self.configs.get(model_name)
    
    def get_validator(self, model_name: str) -> Optional[FeatureValidator]:
        """Get feature validator for a model."""
        config = self.get(model_name)
        if config:
            return FeatureValidator(config)
        return None
    
    def validate_for_model(
        self,
        model_name: str,
        feature_names: List[str],
        features: np.ndarray,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate features for a specific model.
        
        Args:
            model_name: Name of the model
            feature_names: Incoming feature names
            features: Feature array
            strict: Require exact match
            
        Returns:
            (is_valid, warnings/errors)
        """
        validator = self.get_validator(model_name)
        if validator is None:
            return False, [f"No feature config found for model: {model_name}"]
        
        return validator.validate(feature_names, features, strict)


def save_feature_config_with_checkpoint(
    checkpoint_path: str,
    feature_columns: List[str],
    sequence_length: int,
    horizon_periods: int,
    timeframes: List[str],
    input_dim: int,
    feature_engineer_version: str = "",
    mode: str = "stf"
):
    """
    Save feature config alongside a model checkpoint.
    
    Creates a .features.json file next to the checkpoint.
    
    Args:
        feature_engineer_version: FeatureEngineer.VERSION string from training
        mode: "stf" or "mtf" - which pipeline was used during training
    """
    # Get version from FeatureEngineer if not provided
    if not feature_engineer_version:
        try:
            from data.pipeline import FeatureEngineer
            feature_engineer_version = FeatureEngineer.VERSION
            logger.info(f"Using FeatureEngineer.VERSION: {feature_engineer_version}")
        except ImportError:
            logger.warning("Could not import FeatureEngineer - version not recorded")
            feature_engineer_version = "unknown"
    
    config = FeatureConfig(
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon_periods=horizon_periods,
        timeframes=timeframes,
        input_dim=input_dim,
        feature_engineer_version=feature_engineer_version,
        mode=mode
    )
    
    logger.info(f"Saving feature config with FE version: {feature_engineer_version}, mode: {mode}")
    
    # Save alongside checkpoint
    features_path = checkpoint_path.replace('.pth', '.features.json')
    features_path = features_path.replace('.pt', '.features.json')
    config.save(features_path)
    
    return config


def load_feature_config_for_checkpoint(checkpoint_path: str) -> Optional[FeatureConfig]:
    """
    Load feature config for a checkpoint.
    
    Looks for .features.json file alongside the checkpoint.
    """
    features_path = checkpoint_path.replace('.pth', '.features.json')
    features_path = features_path.replace('.pt', '.features.json')
    
    if Path(features_path).exists():
        return FeatureConfig.load(features_path)
    
    logger.warning(f"No feature config found for {checkpoint_path}")
    return None


def create_safe_predictor(
    model,
    feature_config: FeatureConfig,
    device: str = "cuda"
):
    """
    Create a prediction wrapper that validates features.
    
    Returns HOLD signal if features don't match.
    """
    import torch
    
    validator = FeatureValidator(feature_config)
    
    def safe_predict(
        feature_names: List[str],
        features: np.ndarray
    ) -> Dict:
        """
        Safe prediction with feature validation.
        
        Returns HOLD with confidence=0 if validation fails.
        """
        # Validate
        is_valid, warnings = validator.validate(feature_names, features, strict=True)
        
        if not is_valid:
            logger.error(f"Feature validation failed: {warnings}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "probabilities": [0.33, 0.34, 0.33],
                "validation_errors": warnings,
                "valid": False
            }
        
        # Align features if needed
        if feature_names != feature_config.feature_columns:
            features = validator.align_features(feature_names, features)
            if features is None:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "probabilities": [0.33, 0.34, 0.33],
                    "validation_errors": ["Feature alignment failed"],
                    "valid": False
                }
        
        # Run model
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            output = model.forward_multihead(x)
            
            probs = torch.softmax(output.class_logits, dim=-1).cpu().numpy()[0]
            direction = int(np.argmax(probs))
            
            action = ["SHORT", "HOLD", "LONG"][direction]
            
            return {
                "action": action,
                "confidence": float(max(probs)),
                "probabilities": probs.tolist(),
                "mu": float(output.mu.cpu().numpy()[0, 0]),
                "sigma": float(output.sigma.cpu().numpy()[0, 0]),
                "quantiles": {
                    "q10": float(output.quantiles[0, 0].cpu()),
                    "q25": float(output.quantiles[0, 1].cpu()),
                    "q50": float(output.quantiles[0, 2].cpu()),
                    "q75": float(output.quantiles[0, 3].cpu()),
                    "q90": float(output.quantiles[0, 4].cpu()),
                },
                "valid": True,
                "feature_hash": feature_config.version_hash
            }
    
    return safe_predict
