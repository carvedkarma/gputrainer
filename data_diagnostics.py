#!/usr/bin/env python3
"""
Data Diagnostics Script for BTC Futures Training

This script investigates the root cause of gradient explosions by auditing:
1. Raw feature statistics (min/max/mean/std)
2. Scaling verification (before vs after RobustScaler)
3. NaN/Inf/Outlier detection
4. Label distribution analysis
5. Per-batch feature monitoring during training
6. Tiny model test to isolate data vs architecture issues

Run: python -m gpu_trainer.data_diagnostics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw candle data and compute features."""
    from gpu_trainer.data.pipeline import FeatureEngineer
    from gpu_trainer.config import config
    
    data_path = config.data_dir / "BTCUSDT_15m.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} candles from {data_path}")
    
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    
    return df, features_df


def audit_raw_features(features_df: pd.DataFrame) -> Dict:
    """Audit raw features before scaling."""
    logger.info("\n" + "="*80)
    logger.info("AUDIT 1: RAW FEATURES (Before Scaling)")
    logger.info("="*80)
    
    results = {
        "total_rows": len(features_df),
        "total_columns": len(features_df.columns),
        "features": {}
    }
    
    # Check for NaN/Inf counts
    nan_counts = features_df.isna().sum()
    inf_counts = (np.isinf(features_df.select_dtypes(include=[np.number]))).sum()
    
    total_nan = nan_counts.sum()
    total_inf = inf_counts.sum() if len(inf_counts) > 0 else 0
    
    results["total_nan"] = int(total_nan)
    results["total_inf"] = int(total_inf)
    
    logger.info(f"Total rows: {len(features_df):,}")
    logger.info(f"Total columns: {len(features_df.columns)}")
    logger.info(f"Total NaN values: {total_nan:,}")
    logger.info(f"Total Inf values: {total_inf:,}")
    
    # Per-feature statistics
    logger.info("\nPer-Feature Statistics:")
    logger.info("-" * 100)
    logger.info(f"{'Feature':<30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'NaN':>8} {'Outliers':>10}")
    logger.info("-" * 100)
    
    problem_features = []
    
    for col in features_df.columns:
        col_data = features_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(col_data) == 0:
            logger.warning(f"{col:<30} ALL NaN/Inf!")
            problem_features.append((col, "ALL_INVALID"))
            continue
        
        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()
        nan_count = features_df[col].isna().sum() + np.isinf(features_df[col]).sum()
        
        # Count outliers (beyond 5 std from mean)
        if std_val > 0:
            outlier_count = ((col_data < mean_val - 5*std_val) | (col_data > mean_val + 5*std_val)).sum()
        else:
            outlier_count = 0
        
        results["features"][col] = {
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(mean_val),
            "std": float(std_val),
            "nan_count": int(nan_count),
            "outlier_count": int(outlier_count),
            "range": float(max_val - min_val)
        }
        
        # Flag problematic features
        is_problem = False
        problem_type = []
        
        if nan_count > len(features_df) * 0.1:  # >10% NaN
            is_problem = True
            problem_type.append("HIGH_NAN")
        
        if max_val - min_val > 1e6:  # Extreme range
            is_problem = True
            problem_type.append("EXTREME_RANGE")
        
        if outlier_count > len(col_data) * 0.01:  # >1% outliers
            is_problem = True
            problem_type.append("MANY_OUTLIERS")
        
        if std_val < 1e-10:  # Near-constant
            is_problem = True
            problem_type.append("CONSTANT")
        
        if is_problem:
            problem_features.append((col, problem_type))
        
        status = " ⚠️" if is_problem else ""
        logger.info(f"{col:<30} {min_val:>12.4f} {max_val:>12.4f} {mean_val:>12.4f} {std_val:>12.4f} {nan_count:>8} {outlier_count:>10}{status}")
    
    logger.info("-" * 100)
    
    if problem_features:
        logger.warning(f"\n⚠️ PROBLEM FEATURES ({len(problem_features)}):")
        for feat, prob in problem_features:
            logger.warning(f"  - {feat}: {prob}")
    
    results["problem_features"] = [(f, str(p)) for f, p in problem_features]
    
    return results


def audit_scaled_features(features_df: pd.DataFrame) -> Dict:
    """Audit features after RobustScaler scaling."""
    from gpu_trainer.data.pipeline import FeatureEngineer
    
    logger.info("\n" + "="*80)
    logger.info("AUDIT 2: SCALED FEATURES (After RobustScaler)")
    logger.info("="*80)
    
    # Fill NaN with 0 before scaling (as done in training)
    features_filled = features_df.fillna(0)
    
    # Fit and transform
    engineer = FeatureEngineer()
    engineer.fit_scalers(features_filled, method="robust")
    scaled_df = engineer.transform(features_filled)
    
    results = {
        "total_rows": len(scaled_df),
        "features": {}
    }
    
    logger.info("\nPost-Scaling Statistics (should be centered ~0, std ~1-2):")
    logger.info("-" * 100)
    logger.info(f"{'Feature':<30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'|Max|>10':>10}")
    logger.info("-" * 100)
    
    extreme_features = []
    
    for col in scaled_df.columns:
        col_data = scaled_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(col_data) == 0:
            continue
        
        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()
        extreme_count = ((col_data.abs() > 10)).sum()
        
        results["features"][col] = {
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(mean_val),
            "std": float(std_val),
            "extreme_count": int(extreme_count)
        }
        
        is_extreme = abs(max_val) > 10 or abs(min_val) > 10
        if is_extreme:
            extreme_features.append((col, max(abs(min_val), abs(max_val))))
        
        status = " ⚠️" if is_extreme else ""
        logger.info(f"{col:<30} {min_val:>12.4f} {max_val:>12.4f} {mean_val:>12.4f} {std_val:>12.4f} {extreme_count:>10}{status}")
    
    logger.info("-" * 100)
    
    if extreme_features:
        logger.warning(f"\n⚠️ FEATURES WITH EXTREME VALUES AFTER SCALING ({len(extreme_features)}):")
        extreme_features.sort(key=lambda x: x[1], reverse=True)
        for feat, max_abs in extreme_features[:10]:
            logger.warning(f"  - {feat}: |max| = {max_abs:.2f}")
    
    results["extreme_features"] = [(f, float(m)) for f, m in extreme_features]
    
    return results


def audit_labels(df: pd.DataFrame) -> Dict:
    """Audit label distribution."""
    from gpu_trainer.data.regression_targets import generate_multihead_targets
    
    logger.info("\n" + "="*80)
    logger.info("AUDIT 3: LABEL DISTRIBUTION")
    logger.info("="*80)
    
    # Generate labels with default settings
    targets_df = generate_multihead_targets(
        df, 
        horizon_periods=16, 
        n_future_candles=5,
        use_pure_directional=True,
        directional_threshold=0.002
    )
    
    labels = targets_df['class_label'].values
    forward_returns = targets_df['forward_return'].values
    
    results = {
        "total_samples": len(labels),
        "class_distribution": {},
        "forward_returns": {}
    }
    
    # Class distribution
    class_names = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    unique, counts = np.unique(labels, return_counts=True)
    
    logger.info("\nClass Distribution:")
    for cls, count in zip(unique, counts):
        pct = count / len(labels) * 100
        results["class_distribution"][class_names.get(cls, str(cls))] = {
            "count": int(count),
            "percentage": float(pct)
        }
        logger.info(f"  {class_names.get(cls, str(cls))}: {count:,} ({pct:.1f}%)")
    
    # Check for extreme imbalance
    max_pct = max(c["percentage"] for c in results["class_distribution"].values())
    if max_pct > 80:
        logger.warning(f"\n⚠️ SEVERE CLASS IMBALANCE: One class > 80%!")
    elif max_pct > 60:
        logger.warning(f"\n⚠️ MODERATE CLASS IMBALANCE: One class > 60%")
    
    # Forward returns distribution
    returns_clean = forward_returns[~np.isnan(forward_returns) & ~np.isinf(forward_returns)]
    
    results["forward_returns"] = {
        "min": float(np.min(returns_clean)),
        "max": float(np.max(returns_clean)),
        "mean": float(np.mean(returns_clean)),
        "std": float(np.std(returns_clean)),
        "nan_count": int(np.isnan(forward_returns).sum()),
        "inf_count": int(np.isinf(forward_returns).sum())
    }
    
    logger.info(f"\nForward Returns (horizon=16 bars, 4h):")
    logger.info(f"  Min: {results['forward_returns']['min']:.4%}")
    logger.info(f"  Max: {results['forward_returns']['max']:.4%}")
    logger.info(f"  Mean: {results['forward_returns']['mean']:.4%}")
    logger.info(f"  Std: {results['forward_returns']['std']:.4%}")
    logger.info(f"  NaN: {results['forward_returns']['nan_count']}")
    logger.info(f"  Inf: {results['forward_returns']['inf_count']}")
    
    # Check for extreme returns
    extreme_returns = (np.abs(returns_clean) > 0.1).sum()  # >10% moves
    if extreme_returns > 0:
        logger.warning(f"\n⚠️ Found {extreme_returns} extreme returns (>10%)")
    
    return results


def tiny_model_test(features_df: pd.DataFrame, df: pd.DataFrame, use_clipping: bool = True) -> Dict:
    """Test with minimal model to isolate data vs architecture issues.
    
    Args:
        features_df: Raw features DataFrame
        df: Original candle DataFrame
        use_clipping: If True, test with clipped features. If False, test unclipped.
    """
    from gpu_trainer.data.pipeline import FeatureEngineer
    from gpu_trainer.data.regression_targets import generate_multihead_targets
    
    clip_status = "WITH CLIPPING" if use_clipping else "WITHOUT CLIPPING"
    logger.info("\n" + "="*80)
    logger.info(f"AUDIT 4: TINY MODEL TEST ({clip_status})")
    logger.info("="*80)
    logger.info("Testing if DATA itself is trainable with minimal 1-layer model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Prepare data
    features_filled = features_df.fillna(0)
    engineer = FeatureEngineer()
    engineer.fit_scalers(features_filled, method="robust")
    
    # Use clipping or not based on parameter
    if use_clipping:
        scaled_df = engineer.transform_and_clip(features_filled, clip_range=5.0)
        logger.info("Using transform_and_clip (clipped to [-5, +5])")
    else:
        scaled_df = engineer.transform(features_filled)
        logger.info("Using transform (no clipping)")
    
    # Replace any remaining Inf with 0
    scaled_np = scaled_df.values.astype(np.float32)
    scaled_np = np.nan_to_num(scaled_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate labels
    targets_df = generate_multihead_targets(
        df, horizon_periods=16, n_future_candles=5,
        use_pure_directional=True, directional_threshold=0.002
    )
    labels = targets_df['class_label'].values.astype(np.int64)
    
    # Skip warmup period
    seq_len = 100
    features_np = scaled_np[seq_len:]
    labels_np = labels[seq_len:]
    
    # Simple 80/20 split
    split_idx = int(len(features_np) * 0.8)
    train_x = features_np[:split_idx]
    train_y = labels_np[:split_idx]
    
    logger.info(f"Train samples: {len(train_x):,}")
    logger.info(f"Feature dim: {train_x.shape[1]}")
    
    # Minimal model: 1 hidden layer, 32 units
    class TinyModel(nn.Module):
        def __init__(self, input_dim, hidden=32, n_classes=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, n_classes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = TinyModel(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 10 epochs
    batch_size = 256
    results = {
        "epochs": [],
        "final_status": "UNKNOWN"
    }
    
    logger.info("\nTraining tiny model for 10 epochs:")
    logger.info("-" * 60)
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        n_batches = 0
        
        # Shuffle
        perm = np.random.permutation(len(train_x))
        train_x = train_x[perm]
        train_y = train_y[perm]
        
        for i in range(0, len(train_x) - batch_size, batch_size):
            batch_x = torch.tensor(train_x[i:i+batch_size], device=device)
            batch_y = torch.tensor(train_y[i:i+batch_size], device=device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"❌ Epoch {epoch+1}: NaN/Inf loss detected!")
                results["final_status"] = "NAN_LOSS"
                return results
            
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_grad = epoch_grad_norm / n_batches
        
        results["epochs"].append({
            "epoch": epoch + 1,
            "loss": float(avg_loss),
            "grad_norm": float(avg_grad)
        })
        
        status = "✓" if avg_grad < 10 else "⚠️"
        logger.info(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, grad_norm={avg_grad:.2f} {status}")
        
        if avg_grad > 100:
            logger.error(f"❌ Gradient explosion at epoch {epoch+1}!")
            results["final_status"] = "GRADIENT_EXPLOSION"
            return results
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(features_np[split_idx:], device=device)
        test_y = torch.tensor(labels_np[split_idx:], device=device)
        logits = model(test_x)
        preds = logits.argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
    
    results["test_accuracy"] = float(acc)
    results["final_status"] = "SUCCESS"
    
    logger.info("-" * 60)
    logger.info(f"Test accuracy: {acc*100:.2f}%")
    
    # Check predictions distribution
    unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
    class_names = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    logger.info("Predictions distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"  {class_names.get(cls, str(cls))}: {count} ({count/len(preds)*100:.1f}%)")
    
    if max(counts) / len(preds) > 0.95:
        logger.warning("⚠️ Mode collapse: >95% predictions in one class")
        results["final_status"] = "MODE_COLLAPSE"
    
    return results


def per_batch_monitoring(features_df: pd.DataFrame, df: pd.DataFrame, n_batches: int = 20) -> Dict:
    """Run per-batch monitoring to detect when gradient explosion starts."""
    from gpu_trainer.data.pipeline import FeatureEngineer
    from gpu_trainer.data.regression_targets import generate_multihead_targets
    from torch.utils.data import DataLoader, TensorDataset
    
    logger.info("\n" + "="*80)
    logger.info("AUDIT 5: PER-BATCH MONITORING")
    logger.info("="*80)
    logger.info(f"Monitoring feature/gradient stats for first {n_batches} batches")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare data with clipping
    features_filled = features_df.fillna(0)
    engineer = FeatureEngineer()
    engineer.fit_scalers(features_filled, method="robust")
    scaled_df = engineer.transform_and_clip(features_filled, clip_range=5.0)
    scaled_np = scaled_df.values.astype(np.float32)
    
    # Generate labels
    targets_df = generate_multihead_targets(
        df, horizon_periods=16, n_future_candles=5,
        use_pure_directional=True, directional_threshold=0.002
    )
    labels = targets_df['class_label'].values.astype(np.int64)
    
    # Skip warmup
    seq_len = 100
    features_np = scaled_np[seq_len:]
    labels_np = labels[seq_len:]
    
    # Create dataloader
    dataset = TensorDataset(
        torch.tensor(features_np, dtype=torch.float32),
        torch.tensor(labels_np, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Minimal model
    class TinyModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 3)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = TinyModel(features_np.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    results = {"batches": []}
    
    logger.info("-" * 80)
    logger.info(f"{'Batch':>6} {'Feat Min':>10} {'Feat Max':>10} {'NaN':>6} {'Inf':>6} {'Grad Norm':>12} {'Status':>8}")
    logger.info("-" * 80)
    
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Feature stats
        feat_np = batch_x.cpu().numpy()
        feat_min = float(feat_np.min())
        feat_max = float(feat_np.max())
        feat_nan = int(np.isnan(feat_np).sum())
        feat_inf = int(np.isinf(feat_np).sum())
        
        # Forward/backward
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        
        optimizer.step()
        
        status = "✓" if grad_norm < 10 else ("⚠️" if grad_norm < 50 else "❌")
        logger.info(f"{batch_idx:>6} {feat_min:>10.4f} {feat_max:>10.4f} {feat_nan:>6} {feat_inf:>6} {grad_norm:>12.4f} {status:>8}")
        
        results["batches"].append({
            "batch": batch_idx,
            "feat_min": feat_min,
            "feat_max": feat_max,
            "nan_count": feat_nan,
            "inf_count": feat_inf,
            "grad_norm": float(grad_norm)
        })
    
    logger.info("-" * 80)
    
    # Summary
    max_grad = max(b["grad_norm"] for b in results["batches"])
    avg_grad = np.mean([b["grad_norm"] for b in results["batches"]])
    results["max_grad_norm"] = float(max_grad)
    results["avg_grad_norm"] = float(avg_grad)
    
    if max_grad > 50:
        results["status"] = "GRADIENT_EXPLOSION"
        logger.error(f"❌ Max gradient norm: {max_grad:.2f} - EXPLOSION DETECTED")
    elif max_grad > 10:
        results["status"] = "ELEVATED_GRADIENTS"
        logger.warning(f"⚠️ Max gradient norm: {max_grad:.2f} - ELEVATED")
    else:
        results["status"] = "STABLE"
        logger.info(f"✅ Max gradient norm: {max_grad:.2f} - STABLE")
    
    return results


def run_full_audit():
    """Run all diagnostic audits."""
    logger.info("="*80)
    logger.info("BTC FUTURES TRAINING DATA DIAGNOSTICS")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*80)
    
    try:
        df, features_df = load_training_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "raw_features": None,
        "scaled_features": None,
        "labels": None,
        "tiny_model_unclipped": None,
        "tiny_model_clipped": None,
        "per_batch": None
    }
    
    # Run audits
    results["raw_features"] = audit_raw_features(features_df)
    results["scaled_features"] = audit_scaled_features(features_df)
    results["labels"] = audit_labels(df)
    
    # Test BOTH clipped and unclipped paths to isolate the issue
    results["tiny_model_unclipped"] = tiny_model_test(features_df, df, use_clipping=False)
    results["tiny_model_clipped"] = tiny_model_test(features_df, df, use_clipping=True)
    
    # Per-batch monitoring
    results["per_batch"] = per_batch_monitoring(features_df, df)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*80)
    
    issues = []
    
    # Check raw features
    if results["raw_features"]["total_nan"] > 0:
        issues.append(f"Raw features have {results['raw_features']['total_nan']} NaN values")
    if results["raw_features"]["total_inf"] > 0:
        issues.append(f"Raw features have {results['raw_features']['total_inf']} Inf values")
    if len(results["raw_features"]["problem_features"]) > 0:
        issues.append(f"{len(results['raw_features']['problem_features'])} problematic features")
    
    # Check scaled features
    if len(results["scaled_features"]["extreme_features"]) > 0:
        issues.append(f"{len(results['scaled_features']['extreme_features'])} features with extreme values after scaling")
    
    # Check labels
    for cls, stats in results["labels"]["class_distribution"].items():
        if stats["percentage"] > 80:
            issues.append(f"SEVERE class imbalance: {cls} = {stats['percentage']:.1f}%")
    
    # Check tiny model results - compare clipped vs unclipped
    unclipped_status = results["tiny_model_unclipped"]["final_status"]
    clipped_status = results["tiny_model_clipped"]["final_status"]
    
    logger.info("\n📊 TINY MODEL COMPARISON:")
    logger.info(f"   Unclipped: {unclipped_status}")
    logger.info(f"   Clipped:   {clipped_status}")
    
    if unclipped_status == "GRADIENT_EXPLOSION" and clipped_status != "GRADIENT_EXPLOSION":
        issues.append("Clipping FIXES gradient explosion - extreme outliers are the root cause")
    elif unclipped_status == "GRADIENT_EXPLOSION" and clipped_status == "GRADIENT_EXPLOSION":
        issues.append("CRITICAL: Gradient explosion even with clipping - deeper data issue")
    elif unclipped_status == "NAN_LOSS":
        issues.append("CRITICAL: NaN loss with tiny model - DATA PROBLEM")
    elif clipped_status == "MODE_COLLAPSE":
        issues.append("Mode collapse with clipped data - possible label imbalance")
    
    # Check per-batch results
    if results["per_batch"]["status"] == "GRADIENT_EXPLOSION":
        issues.append(f"Per-batch max grad: {results['per_batch']['max_grad_norm']:.2f} - EXPLOSION")
    elif results["per_batch"]["status"] == "ELEVATED_GRADIENTS":
        issues.append(f"Per-batch max grad: {results['per_batch']['max_grad_norm']:.2f} - ELEVATED")
    
    recommendations = []
    
    if issues:
        logger.warning("\n⚠️ ISSUES FOUND:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        # Generate specific recommendations
        if any("FIXES gradient explosion" in i for i in issues):
            recommendations.append("✅ Feature clipping is EFFECTIVE - keep clip_range=5.0")
            recommendations.append("Ensure transform_and_clip() is used in training pipeline")
        
        if any("CRITICAL" in i for i in issues):
            recommendations.append("DATA IS THE ROOT CAUSE - Focus on data cleaning/preprocessing")
            recommendations.append("Check for outliers in features before scaling")
            recommendations.append("Verify label generation logic")
        
        if any("extreme values after scaling" in i for i in issues):
            recommendations.append("Clip scaled features to [-5, 5] range")
            recommendations.append("Use Winsorization for outliers before scaling")
        
        if any("class imbalance" in i for i in issues):
            recommendations.append("Adjust label thresholds to balance classes")
            recommendations.append("Use class weights in loss function")
        
        if any("ELEVATED" in i for i in issues):
            recommendations.append("Consider reducing learning rate")
            recommendations.append("Check if sequence length is appropriate")
    else:
        logger.info("\n✅ NO MAJOR ISSUES FOUND IN DATA")
        recommendations.append("Gradient explosions likely caused by model architecture")
        recommendations.append("Consider simplifying model or reducing LSTM layers")
    
    logger.info("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    results["recommendations"] = recommendations
    
    # Save results
    output_path = Path("gpu_trainer/diagnostic_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_full_audit()
