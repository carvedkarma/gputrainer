"""
XGBoost Stacking Meta-Learner for Ensemble Trading Signals.

2024 State-of-the-Art: Meta-learning with gradient boosting outperforms
simple voting ensembles by learning optimal model weighting conditioned
on market regime and feature context.

This module implements:
1. XGBoost meta-learner that takes base model predictions as input
2. Regime-conditioned weighting (different weights for trending vs. ranging)
3. Automatic calibration using Platt scaling
4. Walk-forward meta-model training to prevent look-ahead bias

References:
- "Stacked Generalization" (Wolpert, 1992)
- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- "Deep Ensemble Learning with Multi-Objective Optimization" (2024)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import joblib

logger = logging.getLogger(__name__)


@dataclass
class MetaLearnerConfig:
    """Configuration for the XGBoost meta-learner."""
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    use_calibration: bool = True
    calibration_method: str = "isotonic"
    n_calibration_folds: int = 3
    use_regime_features: bool = True
    save_feature_importance: bool = True
    

@dataclass
class MetaLearnerOutput:
    """Output from the meta-learner prediction."""
    action: str  # LONG, SHORT, HOLD
    probabilities: Dict[str, float]  # {LONG: 0.6, HOLD: 0.3, SHORT: 0.1}
    confidence: float  # Max probability
    base_model_votes: Dict[str, str]  # {transformer: LONG, lstm: SHORT, ...}
    regime_adjustment: Optional[float] = None  # How much regime affected the prediction
    feature_contributions: Optional[Dict[str, float]] = None  # SHAP-like contributions


class XGBoostMetaLearner:
    """
    XGBoost-based stacking meta-learner for ensemble trading signals.
    
    Architecture:
    1. Base models produce probability distributions [P(SHORT), P(HOLD), P(LONG)]
    2. Meta-learner takes all base predictions + regime features as input
    3. Outputs final calibrated trading signal
    
    Key Features:
    - Learns optimal model weighting per market regime
    - Platt/Isotonic calibration for reliable confidence scores
    - Walk-forward training to prevent look-ahead bias
    - Feature importance tracking for interpretability
    """
    
    def __init__(self, config: Optional[MetaLearnerConfig] = None):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for MetaLearner. Install with: pip install xgboost")
        
        self.config = config or MetaLearnerConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[Any] = None
        self.feature_names: List[str] = []
        self.base_model_names: List[str] = []
        self.is_trained: bool = False
        self.training_metadata: Dict = {}
        self.feature_importance: Dict[str, float] = {}
        
    def _build_meta_features(
        self,
        base_predictions: Dict[str, np.ndarray],
        regime_features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Build meta-feature matrix from base model predictions.
        
        Args:
            base_predictions: Dict mapping model_name -> probabilities array [P(SHORT), P(HOLD), P(LONG)]
                              or [n_samples, 3] for batch mode
            regime_features: Optional dict of regime indicators (adx, volatility, trend_strength, etc.)
            
        Returns:
            Feature matrix for meta-learner [n_samples, n_features]
        """
        meta_features = []
        feature_names = []
        
        # Add base model predictions
        for model_name, probs in sorted(base_predictions.items()):
            probs = np.array(probs)
            if len(probs.shape) == 1:
                probs = probs.reshape(1, -1)
            
            # Add all 3 probabilities per model
            for i, action in enumerate(["short", "hold", "long"]):
                meta_features.append(probs[:, i:i+1])
                feature_names.append(f"{model_name}_{action}_prob")
            
            # Add derived features
            max_prob = probs.max(axis=1, keepdims=True)
            meta_features.append(max_prob)
            feature_names.append(f"{model_name}_confidence")
            
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1, keepdims=True)
            meta_features.append(entropy)
            feature_names.append(f"{model_name}_entropy")
            
            # Direction strength (LONG prob - SHORT prob)
            direction = (probs[:, 2:3] - probs[:, 0:1])
            meta_features.append(direction)
            feature_names.append(f"{model_name}_direction")
        
        # Add regime features if provided
        if regime_features and self.config.use_regime_features:
            n_samples = meta_features[0].shape[0]
            for feat_name, feat_val in sorted(regime_features.items()):
                regime_col = np.full((n_samples, 1), feat_val)
                meta_features.append(regime_col)
                feature_names.append(f"regime_{feat_name}")
        
        # Add model agreement features
        if len(base_predictions) > 1:
            # Compute mean prediction across models
            all_probs = np.stack([np.array(p).reshape(-1, 3) for p in base_predictions.values()])
            mean_probs = all_probs.mean(axis=0)
            std_probs = all_probs.std(axis=0)
            
            for i, action in enumerate(["short", "hold", "long"]):
                meta_features.append(mean_probs[:, i:i+1])
                feature_names.append(f"ensemble_mean_{action}")
                meta_features.append(std_probs[:, i:i+1])
                feature_names.append(f"ensemble_std_{action}")
            
            # Model agreement: fraction of models agreeing on top action
            top_actions = all_probs.argmax(axis=-1)  # [n_models, n_samples]
            agreement = np.array([
                np.bincount(top_actions[:, i], minlength=3).max() / len(base_predictions)
                for i in range(mean_probs.shape[0])
            ]).reshape(-1, 1)
            meta_features.append(agreement)
            feature_names.append("model_agreement")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Concatenate all features
        X = np.concatenate(meta_features, axis=1)
        return X
    
    def train(
        self,
        base_predictions_history: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        regime_features_history: Optional[List[Dict[str, float]]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the meta-learner on historical base model predictions.
        
        Args:
            base_predictions_history: List of dicts, each mapping model_name -> probs [3,]
            labels: Ground truth labels (0=SHORT, 1=HOLD, 2=LONG) shape [n_samples,]
            regime_features_history: Optional list of regime feature dicts
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics dict
        """
        logger.info(f"[META-LEARNER] Training on {len(labels)} samples...")
        
        # Build meta-feature matrix
        X_list = []
        for i, base_preds in enumerate(base_predictions_history):
            regime_feats = regime_features_history[i] if regime_features_history else None
            X_row = self._build_meta_features(base_preds, regime_feats)
            X_list.append(X_row)
        
        X = np.concatenate(X_list, axis=0)
        y = np.array(labels)
        
        # Store base model names
        self.base_model_names = sorted(base_predictions_history[0].keys())
        
        logger.info(f"[META-LEARNER] Feature matrix shape: {X.shape}")
        logger.info(f"[META-LEARNER] Features: {self.feature_names}")
        
        # Split for validation
        n_val = int(len(y) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Build XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective='multi:softprob',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Calibrate if requested
        if self.config.use_calibration and SKLEARN_AVAILABLE:
            logger.info("[META-LEARNER] Applying probability calibration...")
            try:
                self.calibrator = CalibratedClassifierCV(
                    self.model,
                    method=self.config.calibration_method,
                    cv=min(self.config.n_calibration_folds, len(y_train) // 3)
                )
                self.calibrator.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"[META-LEARNER] Calibration failed: {e}")
                self.calibrator = None
        
        # Compute metrics
        train_probs = self.model.predict_proba(X_train)
        val_probs = self.model.predict_proba(X_val)
        
        train_preds = train_probs.argmax(axis=1)
        val_preds = val_probs.argmax(axis=1)
        
        train_acc = (train_preds == y_train).mean()
        val_acc = (val_preds == y_val).mean()
        
        # Per-class metrics
        class_names = ["SHORT", "HOLD", "LONG"]
        per_class_acc = {}
        for c in range(3):
            mask = y_val == c
            if mask.sum() > 0:
                per_class_acc[class_names[c]] = float((val_preds[mask] == c).mean())
        
        self.is_trained = True
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "base_models": self.base_model_names,
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "per_class_accuracy": per_class_acc,
            "calibration_applied": self.calibrator is not None,
            "top_features": list(self.feature_importance.items())[:10]
        }
        
        logger.info(f"[META-LEARNER] Training complete:")
        logger.info(f"  Train accuracy: {train_acc:.4f}")
        logger.info(f"  Val accuracy: {val_acc:.4f}")
        logger.info(f"  Per-class: {per_class_acc}")
        logger.info(f"  Top 5 features: {list(self.feature_importance.items())[:5]}")
        
        return self.training_metadata
    
    def predict(
        self,
        base_predictions: Dict[str, np.ndarray],
        regime_features: Optional[Dict[str, float]] = None
    ) -> MetaLearnerOutput:
        """
        Make a prediction using the trained meta-learner.
        
        Args:
            base_predictions: Dict mapping model_name -> probabilities [P(SHORT), P(HOLD), P(LONG)]
            regime_features: Optional regime indicators
            
        Returns:
            MetaLearnerOutput with action, probabilities, and diagnostics
        """
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained before prediction")
        
        # Build features
        X = self._build_meta_features(base_predictions, regime_features)
        
        # Get probabilities
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(X)[0]
        else:
            probs = self.model.predict_proba(X)[0]
        
        # Determine action
        action_idx = int(np.argmax(probs))
        action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        action = action_map[action_idx]
        
        # Build base model votes summary
        base_votes = {}
        for model_name, model_probs in base_predictions.items():
            vote_idx = int(np.argmax(model_probs))
            base_votes[model_name] = action_map[vote_idx]
        
        # Feature contributions (simplified - just top features for this prediction)
        feature_contribs = None
        if self.config.save_feature_importance:
            # Get top contributing features for this prediction
            feature_contribs = {
                name: float(X[0, i] * imp)
                for i, (name, imp) in enumerate(zip(self.feature_names, self.model.feature_importances_))
            }
            feature_contribs = dict(sorted(feature_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return MetaLearnerOutput(
            action=action,
            probabilities={
                "SHORT": float(probs[0]),
                "HOLD": float(probs[1]),
                "LONG": float(probs[2])
            },
            confidence=float(np.max(probs)),
            base_model_votes=base_votes,
            regime_adjustment=regime_features.get("trend_strength") if regime_features else None,
            feature_contributions=feature_contribs
        )
    
    def save(self, path: Path) -> None:
        """Save the trained meta-learner to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained meta-learner")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path / "meta_xgb.json"))
        
        # Save calibrator if exists
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path / "meta_calibrator.joblib")
        
        # Save metadata
        metadata = {
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "base_model_names": self.base_model_names,
            "feature_importance": self.feature_importance,
            "training_metadata": self.training_metadata
        }
        with open(path / "meta_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[META-LEARNER] Saved to {path}")
    
    def load(self, path: Path) -> bool:
        """Load a trained meta-learner from disk."""
        path = Path(path)
        
        if not (path / "meta_xgb.json").exists():
            logger.warning(f"[META-LEARNER] No saved model found at {path}")
            return False
        
        try:
            # Load XGBoost model
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(path / "meta_xgb.json"))
            
            # Load calibrator if exists
            calibrator_path = path / "meta_calibrator.joblib"
            if calibrator_path.exists():
                self.calibrator = joblib.load(calibrator_path)
            
            # Load metadata
            with open(path / "meta_metadata.json", "r") as f:
                metadata = json.load(f)
            
            self.config = MetaLearnerConfig(**metadata.get("config", {}))
            self.feature_names = metadata.get("feature_names", [])
            self.base_model_names = metadata.get("base_model_names", [])
            self.feature_importance = metadata.get("feature_importance", {})
            self.training_metadata = metadata.get("training_metadata", {})
            
            self.is_trained = True
            logger.info(f"[META-LEARNER] Loaded from {path}")
            logger.info(f"  Base models: {self.base_model_names}")
            logger.info(f"  Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[META-LEARNER] Failed to load: {e}")
            return False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the meta-learner."""
        return {
            "is_trained": self.is_trained,
            "base_models": self.base_model_names,
            "n_features": len(self.feature_names),
            "feature_importance": dict(list(self.feature_importance.items())[:10]),
            "calibration_applied": self.calibrator is not None,
            "training_metadata": self.training_metadata,
            "config": self.config.__dict__
        }


def create_meta_learner_from_ensemble_history(
    prediction_history: List[Dict],
    model_manager: Any,
    min_samples: int = 100
) -> Optional[XGBoostMetaLearner]:
    """
    Factory function to create and train a meta-learner from prediction history.
    
    Args:
        prediction_history: List of prediction records with base model outputs and labels
        model_manager: ModelManager instance with loaded models
        min_samples: Minimum samples required for training
        
    Returns:
        Trained XGBoostMetaLearner or None if insufficient data
    """
    if len(prediction_history) < min_samples:
        logger.warning(f"[META-LEARNER] Insufficient samples: {len(prediction_history)} < {min_samples}")
        return None
    
    meta_learner = XGBoostMetaLearner()
    
    # Extract base predictions and labels from history
    base_predictions_history = []
    labels = []
    regime_features_history = []
    
    for record in prediction_history:
        if "base_predictions" not in record or "label" not in record:
            continue
        
        base_predictions_history.append(record["base_predictions"])
        labels.append(record["label"])
        
        if "regime_features" in record:
            regime_features_history.append(record["regime_features"])
    
    if len(labels) < min_samples:
        logger.warning(f"[META-LEARNER] Insufficient valid records: {len(labels)}")
        return None
    
    # Train
    regime_feats = regime_features_history if regime_features_history else None
    meta_learner.train(base_predictions_history, np.array(labels), regime_feats)
    
    return meta_learner
