"""
Prediction Drift Monitoring for Production Safety

Phase 4b: Monitors distribution shift and calibration drift to detect
when model predictions may be unreliable.

Key Metrics:
- PSI (Population Stability Index): Detects feature distribution shifts
- KL Divergence: Measures prediction distribution changes
- ECE (Expected Calibration Error): Monitors confidence calibration
- Brier Score: Overall probabilistic calibration

USAGE IN TRAINING PIPELINE:
    # After training, create and save the drift monitor
    monitor = create_drift_monitor_from_training(
        training_features=X_train,
        training_predictions=train_predictions,
        feature_names=feature_columns,
        training_direction_probs=train_direction_probs,  # Optional
        save_dir="checkpoints"
    )

USAGE IN INFERENCE/PRODUCTION:
    # Load the monitor
    monitor = load_drift_monitor("checkpoints")
    
    # After batch prediction, check for drift
    report = monitor.check_drift(
        current_features=batch_features,
        current_predictions=batch_mu,
        confidences=batch_confidences,  # Optional, for calibration
        correct=batch_correct,           # Optional, needs realized outcomes
        direction_probs=batch_probs       # Optional, for direction drift
    )
    
    if report.requires_action:
        logger.warning(f"Drift detected: {report.recommended_action}")
        # Reduce confidence, alert, or trigger retraining

INTEGRATION POINTS:
1. Training: Call create_drift_monitor_from_training() at end of training
2. Inference: Load monitor, call check_drift() every N predictions or daily
3. Calibration: Supply confidences + correct after outcomes are known

REQUIRED INPUTS:
- current_features: Always required for PSI
- current_predictions: Always required for mu distribution KL
- confidences + correct: Optional, needed for ECE/Brier (requires realized outcomes)
- direction_probs: Optional, needed for direction distribution KL
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    psi_warning: float = 0.1      # PSI > 0.1 = some shift
    psi_critical: float = 0.25    # PSI > 0.25 = significant shift
    kl_warning: float = 0.1       # KL divergence warning threshold
    kl_critical: float = 0.3      # KL divergence critical threshold
    ece_warning: float = 0.05     # 5% calibration error
    ece_critical: float = 0.10    # 10% calibration error
    brier_warning: float = 0.25   # Brier score warning
    brier_critical: float = 0.35  # Brier score critical


@dataclass
class DriftReport:
    """Report of drift detection results."""
    timestamp: str
    
    # Feature drift (PSI)
    feature_psi: Dict[str, float] = field(default_factory=dict)
    avg_psi: float = 0.0
    max_psi: float = 0.0
    psi_drifted_features: List[str] = field(default_factory=list)
    
    # Prediction drift (KL)
    prediction_kl: float = 0.0
    direction_kl: float = 0.0  # KL for classification distribution
    
    # Calibration metrics
    ece: float = 0.0           # Expected Calibration Error
    brier_score: float = 0.0   # Brier Score
    
    # Status
    feature_drift_status: str = "OK"      # OK, WARNING, CRITICAL
    prediction_drift_status: str = "OK"
    calibration_status: str = "OK"
    
    # Overall
    requires_action: bool = False
    recommended_action: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class PSICalculator:
    """
    Population Stability Index (PSI) Calculator.
    
    PSI measures how much a distribution has shifted from a reference.
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift (monitor)
    - PSI >= 0.25: Significant shift (action required)
    """
    
    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        self.n_bins = n_bins
        self.epsilon = epsilon
    
    def calculate_psi(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        bins: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate PSI between reference and current distributions.
        
        Args:
            reference: Reference (training) distribution
            current: Current (inference) distribution
            bins: Optional pre-computed bin edges
            
        Returns:
            psi: PSI value
            bins: Bin edges used
        """
        reference = np.asarray(reference).flatten()
        current = np.asarray(current).flatten()
        
        # Create bins from reference if not provided
        if bins is None:
            # Use quantile-based binning for robustness
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bins = np.percentile(reference, percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
        
        # Calculate bin proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)
        
        ref_pcts = ref_counts / len(reference) + self.epsilon
        cur_pcts = cur_counts / len(current) + self.epsilon
        
        # Normalize to ensure they sum to 1
        ref_pcts = ref_pcts / ref_pcts.sum()
        cur_pcts = cur_pcts / cur_pcts.sum()
        
        # PSI = sum((cur - ref) * ln(cur / ref))
        psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
        
        return psi, bins


class KLDivergenceCalculator:
    """
    Kullback-Leibler Divergence Calculator.
    
    KL(P || Q) measures how P differs from Q.
    Used for prediction distribution drift.
    """
    
    def __init__(self, n_bins: int = 20, epsilon: float = 1e-10):
        self.n_bins = n_bins
        self.epsilon = epsilon
    
    def calculate_kl(
        self, 
        p: np.ndarray, 
        q: np.ndarray,
        symmetric: bool = True
    ) -> float:
        """
        Calculate KL divergence between distributions.
        
        Args:
            p: Reference distribution (training)
            q: Current distribution (inference)
            symmetric: If True, compute symmetric KL (Jensen-Shannon style)
            
        Returns:
            KL divergence value
        """
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        
        # Create histogram bins from combined data
        combined = np.concatenate([p, q])
        bins = np.linspace(combined.min(), combined.max(), self.n_bins + 1)
        
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Add epsilon for numerical stability
        p_hist = p_hist + self.epsilon
        q_hist = q_hist + self.epsilon
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        if symmetric:
            # Jensen-Shannon divergence (symmetric)
            m = 0.5 * (p_hist + q_hist)
            kl_pm = np.sum(p_hist * np.log(p_hist / m))
            kl_qm = np.sum(q_hist * np.log(q_hist / m))
            return 0.5 * (kl_pm + kl_qm)
        else:
            # Standard KL(P || Q)
            return np.sum(p_hist * np.log(p_hist / q_hist))
    
    def calculate_categorical_kl(
        self,
        p_probs: np.ndarray,
        q_probs: np.ndarray
    ) -> float:
        """
        Calculate KL for categorical distributions (e.g., direction predictions).
        
        Args:
            p_probs: Reference probabilities [n_samples, n_classes]
            q_probs: Current probabilities [n_samples, n_classes]
            
        Returns:
            Average KL divergence
        """
        # Average class probabilities
        p_avg = np.mean(p_probs, axis=0) + self.epsilon
        q_avg = np.mean(q_probs, axis=0) + self.epsilon
        
        # Normalize
        p_avg = p_avg / p_avg.sum()
        q_avg = q_avg / q_avg.sum()
        
        return np.sum(p_avg * np.log(p_avg / q_avg))


class CalibrationMetrics:
    """
    Calibration metrics for probabilistic predictions.
    
    Well-calibrated predictions mean:
    - When model says 70% confident, it should be correct ~70% of the time
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def expected_calibration_error(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE = sum(|bin_accuracy - bin_confidence| * bin_weight)
        
        Args:
            confidences: Predicted confidence scores [0, 1]
            accuracies: Binary correct/incorrect (1/0)
            
        Returns:
            ece: Expected Calibration Error
            bin_info: Per-bin calibration info
        """
        confidences = np.asarray(confidences).flatten()
        accuracies = np.asarray(accuracies).flatten()
        
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_info = {
            'bin_edges': bin_edges.tolist(),
            'bin_confidences': [],
            'bin_accuracies': [],
            'bin_counts': []
        }
        
        ece = 0.0
        n_total = len(confidences)
        
        for i in range(self.n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = mask | (confidences == bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = accuracies[mask].mean()
                bin_count = mask.sum()
                
                ece += (bin_count / n_total) * abs(bin_acc - bin_conf)
                
                bin_info['bin_confidences'].append(float(bin_conf))
                bin_info['bin_accuracies'].append(float(bin_acc))
                bin_info['bin_counts'].append(int(bin_count))
            else:
                bin_info['bin_confidences'].append(0.0)
                bin_info['bin_accuracies'].append(0.0)
                bin_info['bin_counts'].append(0)
        
        return ece, bin_info
    
    def brier_score(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """
        Calculate Brier Score for probabilistic predictions.
        
        Brier = mean((p - y)^2)
        
        Lower is better. For reference:
        - Random (0.5): Brier = 0.25
        - Perfect: Brier = 0.0
        
        Args:
            probabilities: Predicted probabilities
            outcomes: Actual outcomes (0 or 1)
            
        Returns:
            Brier score
        """
        probabilities = np.asarray(probabilities).flatten()
        outcomes = np.asarray(outcomes).flatten()
        
        return np.mean((probabilities - outcomes) ** 2)


class DriftMonitor:
    """
    Main drift monitoring class.
    
    Monitors:
    1. Feature distribution shifts (PSI)
    2. Prediction distribution shifts (KL)
    3. Calibration drift (ECE, Brier)
    """
    
    def __init__(
        self,
        reference_features: Optional[np.ndarray] = None,
        reference_predictions: Optional[np.ndarray] = None,
        reference_direction_probs: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        thresholds: Optional[DriftThresholds] = None,
        history_dir: str = "checkpoints/drift_history"
    ):
        """
        Initialize drift monitor.
        
        Args:
            reference_features: Training feature distribution [n_samples, n_features]
            reference_predictions: Training predictions (mu values)
            reference_direction_probs: Training direction probabilities [n_samples, 3]
            feature_names: Names of features for reporting
            thresholds: Custom thresholds for drift detection
            history_dir: Directory to save drift history
        """
        self.reference_features = reference_features
        self.reference_predictions = reference_predictions
        self.reference_direction_probs = reference_direction_probs
        self.feature_names = feature_names or []
        self.thresholds = thresholds or DriftThresholds()
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculators
        self.psi_calc = PSICalculator()
        self.kl_calc = KLDivergenceCalculator()
        self.calib_calc = CalibrationMetrics()
        
        # Pre-compute reference bins for PSI
        self.feature_bins = {}
        if reference_features is not None:
            self._compute_reference_bins()
    
    def _compute_reference_bins(self):
        """Pre-compute bins from reference distribution."""
        if self.reference_features is None:
            return
        
        n_features = self.reference_features.shape[1]
        for i in range(n_features):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            _, bins = self.psi_calc.calculate_psi(
                self.reference_features[:, i],
                self.reference_features[:, i]  # Same data to get bins
            )
            self.feature_bins[feature_name] = bins
    
    def set_reference(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        direction_probs: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Set reference distributions from training data.
        
        Args:
            features: Training features [n_samples, n_features]
            predictions: Training predictions (mu values)
            direction_probs: Direction probabilities [n_samples, 3] (SHORT, HOLD, LONG)
            feature_names: Feature names
        """
        self.reference_features = features
        self.reference_predictions = predictions
        self.reference_direction_probs = direction_probs
        if feature_names:
            self.feature_names = feature_names
        self._compute_reference_bins()
        
        logger.info(f"Reference set: {features.shape[0]} samples, {features.shape[1]} features")
    
    def check_feature_drift(
        self,
        current_features: np.ndarray
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Check feature distribution drift using PSI.
        
        Args:
            current_features: Current feature values [n_samples, n_features]
            
        Returns:
            psi_values: PSI per feature
            drifted_features: List of features with significant drift
        """
        if self.reference_features is None:
            logger.warning("No reference features set, skipping feature drift check")
            return {}, []
        
        psi_values = {}
        drifted_features = []
        
        n_features = min(current_features.shape[1], self.reference_features.shape[1])
        
        for i in range(n_features):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            bins = self.feature_bins.get(feature_name)
            
            psi, _ = self.psi_calc.calculate_psi(
                self.reference_features[:, i],
                current_features[:, i],
                bins=bins
            )
            
            psi_values[feature_name] = float(psi)
            
            if psi >= self.thresholds.psi_warning:
                drifted_features.append(feature_name)
        
        return psi_values, drifted_features
    
    def check_prediction_drift(
        self,
        current_predictions: np.ndarray,
        current_direction_probs: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Check prediction distribution drift using KL divergence.
        
        Args:
            current_predictions: Current mu predictions
            current_direction_probs: Current direction probabilities [n_samples, 3]
            
        Returns:
            prediction_kl: KL divergence of mu distribution
            direction_kl: KL divergence of direction distribution
        """
        prediction_kl = 0.0
        direction_kl = 0.0
        
        if self.reference_predictions is not None:
            prediction_kl = self.kl_calc.calculate_kl(
                self.reference_predictions,
                current_predictions,
                symmetric=True
            )
        
        # Calculate direction distribution drift if provided
        if current_direction_probs is not None and self.reference_direction_probs is not None:
            direction_kl = self.kl_calc.calculate_categorical_kl(
                self.reference_direction_probs,
                current_direction_probs
            )
        
        return prediction_kl, direction_kl
    
    def check_calibration(
        self,
        confidences: np.ndarray,
        correct: np.ndarray
    ) -> Tuple[float, float, Dict]:
        """
        Check calibration metrics.
        
        Args:
            confidences: Predicted confidence scores
            correct: Binary correct/incorrect indicators
            
        Returns:
            ece: Expected Calibration Error
            brier: Brier Score
            bin_info: Per-bin calibration info
        """
        ece, bin_info = self.calib_calc.expected_calibration_error(confidences, correct)
        brier = self.calib_calc.brier_score(confidences, correct)
        
        return ece, brier, bin_info
    
    def check_drift(
        self,
        current_features: np.ndarray,
        current_predictions: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        correct: Optional[np.ndarray] = None,
        direction_probs: Optional[np.ndarray] = None
    ) -> DriftReport:
        """
        Comprehensive drift check.
        
        Args:
            current_features: Current feature values
            current_predictions: Current mu predictions
            confidences: Confidence scores (for calibration check)
            correct: Correct/incorrect indicators (for calibration check)
            direction_probs: Direction probabilities [n_samples, 3]
            
        Returns:
            DriftReport with all metrics and recommendations
        """
        report = DriftReport(timestamp=datetime.now().isoformat())
        
        # 1. Feature drift (PSI)
        psi_values, drifted_features = self.check_feature_drift(current_features)
        report.feature_psi = psi_values
        report.psi_drifted_features = drifted_features
        
        if psi_values:
            report.avg_psi = np.mean(list(psi_values.values()))
            report.max_psi = max(psi_values.values())
            
            if report.max_psi >= self.thresholds.psi_critical:
                report.feature_drift_status = "CRITICAL"
            elif report.max_psi >= self.thresholds.psi_warning:
                report.feature_drift_status = "WARNING"
        
        # 2. Prediction drift (KL)
        pred_kl, dir_kl = self.check_prediction_drift(current_predictions, direction_probs)
        report.prediction_kl = pred_kl
        report.direction_kl = dir_kl
        
        if pred_kl >= self.thresholds.kl_critical:
            report.prediction_drift_status = "CRITICAL"
        elif pred_kl >= self.thresholds.kl_warning:
            report.prediction_drift_status = "WARNING"
        
        # 3. Calibration (if outcomes provided)
        if confidences is not None and correct is not None:
            ece, brier, _ = self.check_calibration(confidences, correct)
            report.ece = ece
            report.brier_score = brier
            
            if ece >= self.thresholds.ece_critical or brier >= self.thresholds.brier_critical:
                report.calibration_status = "CRITICAL"
            elif ece >= self.thresholds.ece_warning or brier >= self.thresholds.brier_warning:
                report.calibration_status = "WARNING"
        
        # 4. Overall assessment
        statuses = [report.feature_drift_status, report.prediction_drift_status, report.calibration_status]
        
        if "CRITICAL" in statuses:
            report.requires_action = True
            report.recommended_action = "Reduce prediction confidence or retrain model"
        elif statuses.count("WARNING") >= 2:
            report.requires_action = True
            report.recommended_action = "Monitor closely, consider retraining"
        elif "WARNING" in statuses:
            report.recommended_action = "Continue monitoring"
        
        # Save to history
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: DriftReport):
        """Save drift report to history."""
        try:
            history_file = self.history_dir / "drift_history.json"
            
            history = []
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
            
            # Keep last 100 reports
            history.append(report.to_dict())
            history = history[-100:]
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save drift report: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of recent drift history."""
        history_file = self.history_dir / "drift_history.json"
        
        if not history_file.exists():
            return {"status": "No drift history available"}
        
        with open(history_file) as f:
            history = json.load(f)
        
        if not history:
            return {"status": "No drift reports yet"}
        
        recent = history[-10:]  # Last 10 reports
        
        return {
            "total_reports": len(history),
            "recent_critical_count": sum(1 for r in recent if r.get("requires_action")),
            "avg_psi": np.mean([r.get("avg_psi", 0) for r in recent]),
            "avg_ece": np.mean([r.get("ece", 0) for r in recent]),
            "last_report": recent[-1] if recent else None
        }


def create_drift_monitor_from_training(
    training_features: np.ndarray,
    training_predictions: np.ndarray,
    feature_names: List[str],
    training_direction_probs: Optional[np.ndarray] = None,
    save_dir: str = "checkpoints"
) -> DriftMonitor:
    """
    Convenience function to create a drift monitor from training data.
    
    Args:
        training_features: Features from training set [n_samples, n_features]
        training_predictions: Predictions (mu) from training set
        feature_names: Feature column names
        training_direction_probs: Direction probabilities [n_samples, 3] (optional)
        save_dir: Directory to save reference data
        
    Returns:
        Configured DriftMonitor
    """
    monitor = DriftMonitor(
        reference_features=training_features,
        reference_predictions=training_predictions,
        reference_direction_probs=training_direction_probs,
        feature_names=feature_names,
        history_dir=f"{save_dir}/drift_history"
    )
    
    # Save reference data for later loading
    ref_path = Path(save_dir) / "drift_reference.npz"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'features': training_features,
        'predictions': training_predictions,
        'feature_names': feature_names
    }
    
    if training_direction_probs is not None:
        save_data['direction_probs'] = training_direction_probs
    
    np.savez(ref_path, **save_data)
    
    logger.info(f"Drift monitor initialized with {len(training_features)} reference samples")
    
    return monitor


def load_drift_monitor(save_dir: str = "checkpoints") -> Optional[DriftMonitor]:
    """
    Load drift monitor from saved reference data.
    
    Args:
        save_dir: Directory with saved reference data
        
    Returns:
        DriftMonitor or None if not found
    """
    ref_path = Path(save_dir) / "drift_reference.npz"
    
    if not ref_path.exists():
        logger.warning(f"No drift reference found at {ref_path}")
        return None
    
    try:
        data = np.load(ref_path, allow_pickle=True)
        
        direction_probs = data.get('direction_probs') if 'direction_probs' in data.files else None
        
        monitor = DriftMonitor(
            reference_features=data['features'],
            reference_predictions=data['predictions'],
            reference_direction_probs=direction_probs,
            feature_names=list(data['feature_names']),
            history_dir=f"{save_dir}/drift_history"
        )
        
        logger.info("Drift monitor loaded from saved reference")
        return monitor
        
    except Exception as e:
        logger.error(f"Failed to load drift monitor: {e}")
        return None
