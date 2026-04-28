#!/usr/bin/env python3
"""
BTC Futures Trading - GPU Neural Network Trainer

This is the main entry point for training deep learning models
on your local GPU for cryptocurrency trading signals.

Usage:
    python main.py train --model transformer --epochs 100
    python main.py serve --port 8000
    python main.py backtest --start 2024-01-01 --end 2024-12-31
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("No GPU available. Training will use CPU (slower).")
            return False
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install -r requirements.txt")
        return False

async def fetch_data(args):
    """Fetch historical data from Binance (or via Replit proxy if configured)."""
    from data.pipeline import BinanceDataFetcher
    from config import config
    
    replit_url = getattr(args, 'replit_proxy', None) or config.replit_proxy_url
    
    if replit_url:
        logger.info(f"Using Replit proxy at: {replit_url}")
    else:
        logger.info("No Replit proxy configured. Trying direct Binance access...")
        logger.info("Tip: Set REPLIT_PROXY_URL or use --replit-proxy <url>")
    
    fetcher = BinanceDataFetcher(
        config.data.symbols, 
        config.data.timeframes,
        replit_proxy_url=replit_url
    )
    
    try:
        # Use bulk download from Replit if proxy is configured (much faster)
        if replit_url:
            logger.info("Attempting bulk download from Replit (faster)...")
            data = fetcher.fetch_bulk_from_replit()
            
            # Check if we got any data
            has_data = any(
                any(len(df) > 0 for df in tfs.values())
                for tfs in data.values()
            ) if data else False
            
            if not has_data:
                logger.warning("Bulk download empty, falling back to individual fetches...")
                data = await fetcher.fetch_all_historical(args.candles)
        else:
            data = await fetcher.fetch_all_historical(args.candles)
        
        total_candles = 0
        for symbol, timeframes in data.items():
            for tf, df in timeframes.items():
                if len(df) > 0:
                    path = config.data_dir / f"{symbol}_{tf}.parquet"
                    df.to_parquet(path)
                    total_candles += len(df)
                    logger.info(f"Saved {len(df)} candles for {symbol} {tf}")
                else:
                    logger.warning(f"No data received for {symbol} {tf}")
        
        if total_candles > 0:
            logger.info(f"Data fetch complete! Total: {total_candles} candles")
        else:
            logger.error("No data was fetched. Check your connection or Replit proxy URL.")
        
    finally:
        await fetcher.close()

def train(args):
    """Train a neural network model.
    
    IMPORTANT: This function implements proper train/val separation to prevent data leakage:
    1. Chronological split FIRST (before any scaling)
    2. Fit scalers ONLY on training data
    3. Purge gap at train/val boundary to prevent lookahead from label computation
    
    Multi-head mode (--multihead):
    - Uses MultiHeadTrainer with combined loss (CrossEntropy + Huber + GaussianNLL + Pinball)
    - Generates forward_returns and class_labels using generate_multihead_targets()
    - Uses MultiHeadDataset which returns (features, class_labels, forward_returns)
    - Uses multi-head model variants (MultiHeadTransformer, MultiHeadLSTM, MultiHeadCNN)
    """
    import torch
    import numpy as np
    import pandas as pd
    from config import config
    from data.pipeline import FeatureEngineer, TradingDataset, create_labels
    from torch.utils.data import DataLoader
    
    # Use MultiHeadTrainer for multi-head mode
    use_multihead = getattr(args, 'multihead', False)
    
    if use_multihead:
        from training.multihead_trainer import MultiHeadTrainer, MultiHeadDataset, MultiHeadLossConfig, create_regime_balanced_loader
        from data.regression_targets import generate_multihead_targets
        from data.regime_labeler import RegimeLabeler
        logger.info("MULTI-HEAD MODE: Using combined loss (Classification + Regression + Quantile)")
    else:
        from training.trainer import Trainer
        logger.info("LEGACY MODE: Using classification-only training")
    
    logger.info(f"Starting training for model: {args.model}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    check_gpu()
    
    logger.info("Loading training data...")
    data_path = config.data_dir / "BTCUSDT_15m.parquet"
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} candles from {data_path}")
    else:
        logger.error("No cached data found. Run 'python main.py fetch' first to download data.")
        logger.error("Training on synthetic data produces meaningless models - aborting.")
        return
    
    # === STEP 0.5: Compute regime labels for balanced training (candle-only, no leakage) ===
    regime_ids = None
    if use_multihead and config.data.regime_balanced:
        logger.info("Computing regime labels for balanced training...")
        regime_labeler = RegimeLabeler()
        regime_ids = regime_labeler.label_regimes(df)
        dist = regime_labeler.get_regime_distribution(regime_ids)
        logger.info(f"Regime distribution (raw): BULL={dist['BULL']*100:.1f}%, BEAR={dist['BEAR']*100:.1f}%, "
                   f"HIGH_VOL={dist['HIGH_VOL']*100:.1f}%, LOW_VOL_CHOP={dist['LOW_VOL_CHOP']*100:.1f}%")
    
    # === STEP 1: Compute features (before split, features don't leak future) ===
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    
    # === STEP 2: Create labels with lookahead (horizon candles into future) ===
    # Default: 16 bars = 4 hours for 15m timeframe
    horizon = getattr(args, 'horizon', 16)
    
    if use_multihead:
        # Multi-head mode: generate all targets (class, returns, trading, candles)
        n_future_candles = getattr(args, 'n_future_candles', 5)
        
        # Get cost/threshold config from args or config
        # Stage 1 fix: lowered min_confidence from 0.7 to 0.40 to reduce HOLD-heavy labels
        # Stage 2: pure_directional mode uses simple return threshold instead of cost-aware gating
        min_net_edge = getattr(args, 'min_net_edge', 0.0)  # Default: no edge filter for debugging
        min_confidence = getattr(args, 'min_confidence', 0.40)  # Default: 0.40 (lowered from 0.7)
        fixed_cost = getattr(args, 'cost', config.institution.cost_mode.get_cost())
        use_volatility_cost = getattr(args, 'volatility_cost', False)
        use_pure_directional = getattr(args, 'pure_directional', False)
        directional_threshold = getattr(args, 'directional_threshold', 0.0020)
        
        # Stage 3: regime-based labeling with ADX-adaptive thresholds
        use_regime_labels = getattr(args, 'regime_labels', False)
        trend_threshold = getattr(args, 'trend_threshold', 0.0015)
        range_threshold = getattr(args, 'range_threshold', 0.0030)
        
        mode = "REGIME_BASED" if use_regime_labels else ("PURE_DIRECTIONAL" if use_pure_directional else "COST_AWARE")
        logger.info(f"Label config: mode={mode}, horizon={horizon}, cost={fixed_cost:.4%}, "
                   f"min_net_edge={min_net_edge:.4%}, min_confidence={min_confidence:.2f}")
        if use_pure_directional:
            logger.info(f"  Pure directional threshold: {directional_threshold:.4%}")
        if use_regime_labels:
            logger.info(f"  Regime thresholds: trend={trend_threshold:.4%}, range={range_threshold:.4%}")
        
        targets_df = generate_multihead_targets(
            df, 
            horizon_periods=horizon, 
            n_future_candles=n_future_candles,
            min_net_edge=min_net_edge,
            min_confidence=min_confidence,
            use_volatility_cost=use_volatility_cost,
            fixed_cost=fixed_cost,
            use_pure_directional=use_pure_directional,
            directional_threshold=directional_threshold,
            use_regime_labels=use_regime_labels,
            trend_threshold=trend_threshold,
            range_threshold=range_threshold
        )
        
        # Classification and regression targets
        labels = targets_df['class_label'].values.astype(np.int64)
        forward_returns = targets_df['forward_return'].values.astype(np.float32)
        
        # Trading head targets (entry/SL/TP)
        entry_offset = targets_df['entry_offset'].values.astype(np.float32)
        sl_distance = targets_df['sl_distance'].values.astype(np.float32)
        tp_distance = targets_df['tp_distance'].values.astype(np.float32)
        
        # Candle prediction targets (collect all delta columns in DETERMINISTIC order)
        # Order must be: close_1, high_1, low_1, close_2, high_2, low_2, ... to match model output
        candle_cols = []
        for i in range(1, n_future_candles + 1):
            candle_cols.extend([
                f"candle_delta_close_{i}",
                f"candle_delta_high_{i}",
                f"candle_delta_low_{i}"
            ])
        candle_targets = targets_df[candle_cols].values.astype(np.float32)  # [N, n_future*3]
        logger.info(f"Candle target columns (ordered): {candle_cols[:6]}...{candle_cols[-3:]}")
        
        logger.info(f"Generated multi-head targets: labels shape={labels.shape}, returns shape={forward_returns.shape}")
        logger.info(f"  Trading targets: entry_offset, sl_distance, tp_distance")
        logger.info(f"  Candle targets: {len(candle_cols)} columns ({n_future_candles} steps x 3)")
    else:
        # Legacy mode: classification-only labels
        labels = create_labels(df, horizon=horizon, threshold=0.001)
        labels = (labels + 1).astype(int)  # Convert -1/0/1 to 0/1/2
        forward_returns = None
    
    # === STEP 3: CHRONOLOGICAL SPLIT FIRST (before scaling!) ===
    # This prevents scaler from learning distribution info from validation/test data
    sequence_length = config.data.sequence_length
    valid_start = sequence_length  # Skip warmup period for indicators
    
    features_np = features_df.values[valid_start:].astype(np.float32)
    labels_np = labels[valid_start:].astype(np.int64)
    
    # Also slice all targets for multi-head mode
    if use_multihead:
        forward_returns_np = forward_returns[valid_start:].astype(np.float32)
        entry_offset_np = entry_offset[valid_start:].astype(np.float32)
        sl_distance_np = sl_distance[valid_start:].astype(np.float32)
        tp_distance_np = tp_distance[valid_start:].astype(np.float32)
        candle_targets_np = candle_targets[valid_start:].astype(np.float32)
        # Also slice regime_ids if available
        regime_ids_np = regime_ids[valid_start:] if regime_ids is not None else None
    else:
        forward_returns_np = None
        entry_offset_np = None
        sl_distance_np = None
        tp_distance_np = None
        candle_targets_np = None
        regime_ids_np = None
    
    # === STEP 4: PURGE GAP and EXPLICIT SPLIT SIZING ===
    # Labels near train end look `horizon` candles ahead, which may be in val
    # Purge gap must be at least horizon + sequence_length to prevent lookahead
    purge_gap = horizon + sequence_length
    
    n_total = len(features_np)
    
    # EXPLICIT SIZING (not implicit remainder)
    # Validation must have at least horizon + sequence_length samples to be meaningful
    min_val_samples = horizon + sequence_length
    min_train_samples = sequence_length * 3  # At least 3x sequence for meaningful training
    
    # Reserve explicit validation window: ~10% of total but at least min_val_samples
    val_samples = max(int(n_total * 0.1), min_val_samples)
    
    # Train gets the rest after subtracting purge gap and validation
    train_samples = n_total - purge_gap - val_samples
    
    # Validate we have enough data
    if train_samples < min_train_samples:
        logger.error(f"Insufficient training data: {train_samples} < {min_train_samples}")
        logger.error(f"  Total: {n_total}, purge_gap: {purge_gap}, val_samples: {val_samples}")
        logger.error(f"  Need at least {min_train_samples + purge_gap + min_val_samples} total samples")
        return
    
    # Compute actual indices
    train_end = train_samples
    val_start = train_end + purge_gap  # Val starts AFTER purge gap
    val_end = val_start + val_samples
    
    # Final bounds check - fail rather than clamp to preserve validation integrity
    if val_end > n_total:
        logger.error(f"Val window exceeds data bounds: val_end={val_end} > n_total={n_total}")
        logger.error(f"  Reduce val_samples or provide more data")
        return
    
    # Log explicit split sizes
    logger.info(f"Data splits (total={n_total}):")
    logger.info(f"  Train: [0, {train_end}) = {train_samples} samples")
    logger.info(f"  Purge: [{train_end}, {val_start}) = {purge_gap} samples (discarded)")
    logger.info(f"  Val:   [{val_start}, {val_end}) = {val_samples} samples")
    tail_discarded = n_total - val_end
    if tail_discarded > 0:
        logger.info(f"  Tail:  [{val_end}, {n_total}) = {tail_discarded} samples (unused)")
    
    # Assert correct layout
    assert train_end + purge_gap == val_start, "Purge gap must be exactly between train and val"
    assert val_end <= n_total, "Val must not exceed data"
    assert val_samples >= min_val_samples, f"Val samples {val_samples} < minimum {min_val_samples}"
    
    # CRITICAL: Verify labels' lookahead never crosses into validation
    # Labels at index i look ahead `horizon` candles to compute target
    # Train labels at train_end-1 look at index train_end-1+horizon
    # This must be strictly less than val_start
    max_label_lookahead = train_end - 1 + horizon
    if max_label_lookahead >= val_start:
        logger.error(f"LEAKAGE DETECTED: Train labels look into validation!")
        logger.error(f"  Train ends at {train_end-1}, label lookahead={horizon}")
        logger.error(f"  Max lookahead index: {max_label_lookahead} >= val_start {val_start}")
        return
    
    logger.info(f"Leakage check PASSED: max_label_lookahead={max_label_lookahead} < val_start={val_start}")
    
    # Split the raw (unscaled) features
    train_features_raw = features_np[:train_end]
    train_labels = labels_np[:train_end]
    val_features_raw = features_np[val_start:val_end]
    val_labels = labels_np[val_start:val_end]
    
    # Split all targets for multi-head mode
    if use_multihead:
        train_returns = forward_returns_np[:train_end]
        val_returns = forward_returns_np[val_start:val_end]
        
        # Trading targets (entry/SL/TP)
        train_entry_offset = entry_offset_np[:train_end]
        train_sl_distance = sl_distance_np[:train_end]
        train_tp_distance = tp_distance_np[:train_end]
        val_entry_offset = entry_offset_np[val_start:val_end]
        val_sl_distance = sl_distance_np[val_start:val_end]
        val_tp_distance = tp_distance_np[val_start:val_end]
        
        # Candle targets
        train_candle_targets = candle_targets_np[:train_end]
        val_candle_targets = candle_targets_np[val_start:val_end]
        
        # Regime IDs for balanced training
        train_regime_ids = regime_ids_np[:train_end] if regime_ids_np is not None else None
        val_regime_ids = regime_ids_np[val_start:val_end] if regime_ids_np is not None else None
    else:
        train_returns = None
        val_returns = None
        train_entry_offset = train_sl_distance = train_tp_distance = None
        val_entry_offset = val_sl_distance = val_tp_distance = None
        train_candle_targets = val_candle_targets = None
        train_regime_ids = val_regime_ids = None
    
    # === STEP 5: FIT SCALER ON TRAINING DATA ONLY ===
    # This is critical - scaler must not see validation/test distribution
    train_features_df = pd.DataFrame(train_features_raw, columns=features_df.columns)
    engineer.fit_scalers(train_features_df)
    logger.info("Scaler fitted on TRAINING data only (no leakage)")
    
    # Transform both train and val with the train-fitted scaler
    # STABILITY FIX (Feb 2026): Use transform_and_clip to handle extreme outliers
    # that cause gradient explosions even after RobustScaler
    clip_range = getattr(args, 'feature_clip', 5.0)
    logger.info(f"Applying feature clipping to [-{clip_range}, +{clip_range}]")
    train_features_scaled = engineer.transform_and_clip(train_features_df, clip_range=clip_range).values.astype(np.float32)
    val_features_df = pd.DataFrame(val_features_raw, columns=features_df.columns)
    val_features_scaled = engineer.transform_and_clip(val_features_df, clip_range=clip_range).values.astype(np.float32)
    
    # === STEP 5.5: HARD DATA CLEANSING - Drop NaN/Inf rows ===
    # This is critical: NaN/Inf in features will cause NaN loss and corrupt training
    # IMPORTANT: Must clean ALL arrays together to maintain alignment!
    def clean_data_multihead(features, labels, returns, entry_offset, sl_distance, tp_distance, candle_targets, regime_ids, name):
        """Replace Inf->NaN, drop rows with any NaN, align ALL targets together (including regime_ids)."""
        # Replace Inf with NaN in features
        features = np.where(np.isinf(features), np.nan, features)
        
        # Find rows with any NaN in features
        nan_mask = np.isnan(features).any(axis=1)
        
        # Also check returns for NaN
        returns = np.where(np.isinf(returns), np.nan, returns)
        nan_mask = nan_mask | np.isnan(returns)
        
        # Check trading targets for NaN
        entry_offset = np.where(np.isinf(entry_offset), np.nan, entry_offset)
        sl_distance = np.where(np.isinf(sl_distance), np.nan, sl_distance)
        tp_distance = np.where(np.isinf(tp_distance), np.nan, tp_distance)
        nan_mask = nan_mask | np.isnan(entry_offset) | np.isnan(sl_distance) | np.isnan(tp_distance)
        
        # Check candle targets for NaN
        candle_targets = np.where(np.isinf(candle_targets), np.nan, candle_targets)
        nan_mask = nan_mask | np.isnan(candle_targets).any(axis=1)
        
        nan_count = nan_mask.sum()
        valid_mask = ~nan_mask
        
        if nan_count > 0:
            logger.warning(f"{name}: Dropping {nan_count} rows with NaN/Inf ({nan_count/len(features)*100:.1f}%)")
            features = features[valid_mask]
            labels = labels[valid_mask]
            returns = returns[valid_mask]
            entry_offset = entry_offset[valid_mask]
            sl_distance = sl_distance[valid_mask]
            tp_distance = tp_distance[valid_mask]
            candle_targets = candle_targets[valid_mask]
            # Also filter regime_ids if available
            if regime_ids is not None:
                regime_ids = regime_ids[valid_mask]
        
        # Final assertion - must be all finite
        assert np.isfinite(features).all(), f"{name}: Features still have non-finite values!"
        assert np.isfinite(returns).all(), f"{name}: Returns still have non-finite values!"
        assert np.isfinite(candle_targets).all(), f"{name}: Candle targets still have non-finite values!"
        logger.info(f"{name}: {len(features)} clean samples, all finite and aligned")
        
        return features, labels, returns, entry_offset, sl_distance, tp_distance, candle_targets, regime_ids
    
    def clean_data_legacy(features, labels, name):
        """Replace Inf->NaN, drop rows with any NaN (legacy classification mode)."""
        features = np.where(np.isinf(features), np.nan, features)
        nan_mask = np.isnan(features).any(axis=1)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"{name}: Dropping {nan_count} rows with NaN/Inf ({nan_count/len(features)*100:.1f}%)")
            valid_mask = ~nan_mask
            features = features[valid_mask]
            labels = labels[valid_mask]
        assert np.isfinite(features).all(), f"{name}: Still has non-finite values after cleaning!"
        logger.info(f"{name}: {len(features)} clean samples, all finite")
        return features, labels
    
    # Clean data - CRITICAL: Must clean ALL arrays together for alignment!
    if use_multihead:
        train_features_scaled, train_labels, train_returns, train_entry_offset, train_sl_distance, train_tp_distance, train_candle_targets, train_regime_ids = clean_data_multihead(
            train_features_scaled, train_labels, train_returns,
            train_entry_offset, train_sl_distance, train_tp_distance, train_candle_targets, train_regime_ids,
            name="Train")
        val_features_scaled, val_labels, val_returns, val_entry_offset, val_sl_distance, val_tp_distance, val_candle_targets, val_regime_ids = clean_data_multihead(
            val_features_scaled, val_labels, val_returns,
            val_entry_offset, val_sl_distance, val_tp_distance, val_candle_targets, val_regime_ids,
            name="Val")
    else:
        train_features_scaled, train_labels = clean_data_legacy(train_features_scaled, train_labels, name="Train")
        val_features_scaled, val_labels = clean_data_legacy(val_features_scaled, val_labels, name="Val")
    
    # === STEP 6: Create datasets ===
    if use_multihead:
        # Multi-head dataset returns (features, class_labels, forward_returns, trading_targets, candle_targets)
        train_dataset = MultiHeadDataset(
            train_features_scaled, train_labels, train_returns,
            entry_offset=train_entry_offset, sl_distance=train_sl_distance, tp_distance=train_tp_distance,
            candle_targets=train_candle_targets, regime_ids=train_regime_ids, n_future_candles=n_future_candles,
            sequence_length=sequence_length
        )
        val_dataset = MultiHeadDataset(
            val_features_scaled, val_labels, val_returns,
            entry_offset=val_entry_offset, sl_distance=val_sl_distance, tp_distance=val_tp_distance,
            candle_targets=val_candle_targets, regime_ids=val_regime_ids, n_future_candles=n_future_candles,
            sequence_length=sequence_length
        )
        logger.info(f"Created MultiHeadDataset: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Log regime distribution in training set
        if train_regime_ids is not None:
            dist = train_dataset.get_regime_distribution()
            logger.info(f"Training regime distribution: BULL={dist['BULL']*100:.1f}%, BEAR={dist['BEAR']*100:.1f}%, "
                       f"HIGH_VOL={dist['HIGH_VOL']*100:.1f}%, LOW_VOL_CHOP={dist['LOW_VOL_CHOP']*100:.1f}%")
    else:
        train_dataset = TradingDataset(train_features_scaled, train_labels, sequence_length, validate_data=True)
        val_dataset = TradingDataset(val_features_scaled, val_labels, sequence_length, validate_data=True)
    
    # Create dataloaders - use regime-balanced sampling for multihead training
    if use_multihead and config.data.regime_balanced and train_regime_ids is not None:
        logger.info("Using REGIME-BALANCED sampling for training (target ~25% per regime)")
        train_loader = create_regime_balanced_loader(
            train_dataset, batch_size=args.batch_size, num_workers=0, 
            target_balance=config.data.target_regime_balance
        )
    else:
        # Note: shuffle=True is OK for training since we've already done chronological split
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_dim = features_np.shape[1]
    logger.info(f"Input dimension: {input_dim}, Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # === STEP 7: Model selection ===
    if use_multihead:
        # Multi-head model variants with Classification + Regression + Quantile heads
        from models.multihead import MultiHeadTransformer, MultiHeadTFT, MultiHeadLSTM, MultiHeadCNN
        from models.simple_mlp import SimpleMLP, SimpleMLP_Config, MultiHeadSimpleMLP, MultiHeadSimpleMLP_Config
        
        if args.model == "simple_mlp":
            # SimpleMLP: Stable baseline classifier (no gradient explosions)
            # Use this when LSTM/Transformer training is unstable
            mlp_config = SimpleMLP_Config(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],  # Deeper for more capacity
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                n_candle_steps=n_future_candles
            )
            model = SimpleMLP(mlp_config)
            model.name = "SimpleMLP"
            logger.info(f"Using SimpleMLP (stable baseline): {model.parameters_count():,} parameters")
        elif args.model == "multihead_simple_mlp":
            # MultiHeadSimpleMLP: Progressive head re-enablement
            # Enable heads via CLI flags: --enable-quantile, --enable-vol-state, --enable-mu, --enable-sigma
            enable_quantile = getattr(args, 'enable_quantile', False)
            enable_vol_state = getattr(args, 'enable_vol_state', False)
            enable_mu = getattr(args, 'enable_mu', False)
            enable_sigma = getattr(args, 'enable_sigma', False)
            
            mlp_config = MultiHeadSimpleMLP_Config(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                n_candle_steps=n_future_candles,
                enable_quantile_head=enable_quantile,
                enable_vol_state_head=enable_vol_state,
                enable_mu_head=enable_mu,
                enable_sigma_head=enable_sigma
            )
            model = MultiHeadSimpleMLP(mlp_config)
            
            # Log which heads are enabled
            enabled_heads = ["classification"]
            if enable_quantile:
                enabled_heads.append("quantile")
            if enable_vol_state:
                enabled_heads.append("vol_state")
            if enable_mu:
                enabled_heads.append("mu")
            if enable_sigma:
                enabled_heads.append("sigma")
            logger.info(f"Using MultiHeadSimpleMLP: {model.parameters_count():,} parameters")
            logger.info(f"Enabled heads: {', '.join(enabled_heads)}")
            
            # Store head enablement for loss configuration
            model.head_config = {
                'enable_quantile': enable_quantile,
                'enable_vol_state': enable_vol_state,
                'enable_mu': enable_mu,
                'enable_sigma': enable_sigma
            }
        elif args.model == "transformer":
            model = MultiHeadTransformer(
                input_dim=input_dim,
                d_model=config.model.transformer_dim,
                nhead=config.model.transformer_heads,
                num_layers=config.model.transformer_layers,
                n_future_candles=n_future_candles  # Must match target generation!
            )
        elif args.model == "tft":
            model = MultiHeadTFT(
                input_dim=input_dim,
                d_model=config.model.transformer_dim,
                nhead=config.model.transformer_heads,
                num_encoder_layers=4,
                n_future_candles=n_future_candles  # Must match target generation!
            )
        elif args.model == "lstm":
            model = MultiHeadLSTM(
                input_dim=input_dim,
                hidden_dim=config.model.lstm_hidden,
                num_layers=config.model.lstm_layers,
                n_future_candles=n_future_candles  # Must match target generation!
            )
        elif args.model == "cnn":
            model = MultiHeadCNN(
                input_dim=input_dim,
                hidden_channels=config.model.cnn_channels,  # Note: param name is hidden_channels
                n_future_candles=n_future_candles  # Must match target generation!
            )
        else:
            logger.error(f"Multi-head mode not supported for model type: {args.model}")
            logger.error("Supported multi-head models: transformer, tft, lstm, cnn, simple_mlp, multihead_simple_mlp")
            return
        logger.info(f"Using MULTI-HEAD model: {model.name}")
    else:
        # Legacy classification-only models
        if args.model == "transformer":
            from models.transformer import TransformerPriceModel
            model = TransformerPriceModel(
                input_dim=input_dim,
                d_model=config.model.transformer_dim,
                nhead=config.model.transformer_heads,
                num_layers=config.model.transformer_layers
            )
        elif args.model == "tft":
            from models.transformer import TemporalFusionTransformer
            model = TemporalFusionTransformer(
                input_dim=input_dim,
                d_model=config.model.transformer_dim,
                nhead=config.model.transformer_heads
            )
        elif args.model == "lstm":
            from models.lstm import BidirectionalLSTM
            model = BidirectionalLSTM(
                input_dim=input_dim,
                hidden_dim=config.model.lstm_hidden,
                num_layers=config.model.lstm_layers
            )
        elif args.model == "cnn":
            from models.cnn import ResNetPrice
            model = ResNetPrice(
                input_dim=input_dim,
                channels=config.model.cnn_channels
            )
        elif args.model == "vae":
            from models.vae import MarketVAE
            model = MarketVAE(
                input_dim=input_dim,
                sequence_length=config.data.sequence_length,
                latent_dim=config.model.vae_latent_dim
            )
        elif args.model == "gnn":
            from models.gnn import CrossAssetGNN
            model = CrossAssetGNN(
                input_dim=input_dim,
                num_assets=len(config.data.symbols)
            )
        elif args.model == "simple_mlp":
            # SimpleMLP: Stable baseline classifier (works in legacy mode too)
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
            logger.info(f"Using SimpleMLP (stable baseline): {model.parameters_count():,} parameters")
        elif args.model == "multihead_simple_mlp":
            # MultiHeadSimpleMLP: Progressive head re-enablement (legacy mode)
            from models.simple_mlp import MultiHeadSimpleMLP, MultiHeadSimpleMLP_Config
            
            # Check which heads are enabled via CLI flags
            enable_quantile = getattr(args, 'enable_quantile', False)
            enable_vol_state = getattr(args, 'enable_vol_state', False)
            enable_mu = getattr(args, 'enable_mu', False)
            enable_sigma = getattr(args, 'enable_sigma', False)
            
            mlp_config = MultiHeadSimpleMLP_Config(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                n_quantiles=5,
                enable_quantile_head=enable_quantile,
                enable_vol_state_head=enable_vol_state,
                enable_mu_head=enable_mu,
                enable_sigma_head=enable_sigma
            )
            model = MultiHeadSimpleMLP(mlp_config)
            model.name = "MultiHeadSimpleMLP"
            model.count_parameters = model.parameters_count  # Alias for compatibility
            
            # Log enabled heads
            enabled = ["classification"]
            if enable_quantile: enabled.append("quantile")
            if enable_vol_state: enabled.append("vol_state")
            if enable_mu: enabled.append("mu")
            if enable_sigma: enabled.append("sigma")
            logger.info(f"Using MultiHeadSimpleMLP: {model.parameters_count():,} parameters")
            logger.info(f"Enabled heads: {', '.join(enabled)}")
        elif args.model == "enhanced_mlp":
            # EnhancedMultiHeadMLP: Deeper architecture with residual connections
            from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
            
            # Check which heads are enabled via CLI flags (reuse same flags)
            enable_quantile = getattr(args, 'enable_quantile', False)
            enable_vol_state = getattr(args, 'enable_vol_state', False)
            enable_mu = getattr(args, 'enable_mu', False)
            enable_sigma = getattr(args, 'enable_sigma', False)
            
            mlp_config = EnhancedMultiHeadMLP_Config(
                input_dim=input_dim,
                hidden_dims=[512, 256, 128, 64],  # Deeper architecture
                num_classes=3,
                dropout=0.3,
                use_layer_norm=True,
                use_residual=True,  # Enable residual connections
                enable_quantile_head=enable_quantile,
                enable_vol_state_head=enable_vol_state,
                enable_mu_head=enable_mu,
                enable_sigma_head=enable_sigma
            )
            model = EnhancedMultiHeadMLP(mlp_config)
            model.name = "EnhancedMultiHeadMLP"
            model.count_parameters = model.parameters_count
            
            # Log enabled heads
            enabled = ["classification"]
            if enable_quantile: enabled.append("quantile")
            if enable_vol_state: enabled.append("vol_state")
            if enable_mu: enabled.append("mu")
            if enable_sigma: enabled.append("sigma")
            logger.info(f"Using EnhancedMultiHeadMLP: {model.parameters_count():,} parameters")
            logger.info(f"Architecture: {mlp_config.hidden_dims} with residual connections")
            logger.info(f"Enabled heads: {', '.join(enabled)}")
        else:
            logger.error(f"Unknown model type: {args.model}")
            logger.error("Supported models: transformer, tft, lstm, cnn, vae, gnn, simple_mlp, multihead_simple_mlp, enhanced_mlp")
            return
        
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    
    # === STEP 7: Compute class weights for imbalanced dataset ===
    # With threshold=0.001 and costs=0.0009, HOLD class often dominates
    # Class weights help the model learn from minority classes (LONG/SHORT)
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    
    # Compute inverse frequency weights (higher weight for rare classes)
    # Formula: weight[i] = total_samples / (num_classes * count[i])
    # CRITICAL: Cap weights to prevent gradient explosion (max 10x)
    num_classes = 3  # SHORT, HOLD, LONG
    MAX_CLASS_WEIGHT = getattr(args, 'class_weight_cap', 10.0)  # CLI-configurable cap
    class_weights_list = []
    for class_idx in range(num_classes):
        if class_idx in unique_labels:
            idx = np.where(unique_labels == class_idx)[0][0]
            weight = total_samples / (num_classes * label_counts[idx])
            weight = min(weight, MAX_CLASS_WEIGHT)  # Cap to prevent explosion
        else:
            weight = 1.0  # Default weight if class not present
        class_weights_list.append(weight)
    
    class_weights = torch.FloatTensor(class_weights_list)
    logger.info(f"Class distribution: SHORT={label_counts[0] if 0 in unique_labels else 0}, "
                f"HOLD={label_counts[1] if 1 in unique_labels else 0}, "
                f"LONG={label_counts[2] if 2 in unique_labels else 0}")
    logger.info(f"Class weights (capped at {MAX_CLASS_WEIGHT}x): {class_weights.numpy()}")
    
    # === STEP 8: Create trainer ===
    if use_multihead:
        # Multi-head trainer with combined loss
        use_focal = getattr(args, 'focal_loss', False)
        focal_gamma = getattr(args, 'focal_gamma', 2.0)
        # Check if model has head_config (MultiHeadSimpleMLP progressive enablement)
        if hasattr(model, 'head_config'):
            head_cfg = model.head_config
            # Configure loss based on enabled heads
            loss_config = MultiHeadLossConfig(
                class_weights=class_weights,
                use_focal_loss=use_focal,
                focal_gamma=focal_gamma,
                # Enable lambda for enabled heads only
                lambda_quantile=0.3 if head_cfg.get('enable_quantile', False) else 0.0,
                lambda_mu=0.3 if head_cfg.get('enable_mu', False) else 0.0,
                lambda_sigma=0.2 if head_cfg.get('enable_sigma', False) else 0.0,
                lambda_vol_state=0.2 if head_cfg.get('enable_vol_state', False) else 0.0,
                # Set head enabled flags - all heads
                head_enabled_quantile=head_cfg.get('enable_quantile', False),
                head_enabled_vol_state=head_cfg.get('enable_vol_state', False),
                head_enabled_mu=head_cfg.get('enable_mu', False),
                head_enabled_sigma=head_cfg.get('enable_sigma', False),
            )
            logger.info(f"MultiHeadSimpleMLP loss config based on head_config:")
            logger.info(f"  - Classification: λ={loss_config.lambda_class} (always enabled)")
            logger.info(f"  - Quantile: λ={loss_config.lambda_quantile}")
            logger.info(f"  - Vol State: λ={loss_config.lambda_vol_state}")
            logger.info(f"  - Mu: λ={loss_config.lambda_mu}")
            logger.info(f"  - Sigma: λ={loss_config.lambda_sigma}")
        else:
            # Default loss config for other multihead models
            loss_config = MultiHeadLossConfig(class_weights=class_weights, use_focal_loss=use_focal, focal_gamma=focal_gamma)
        
        trainer = MultiHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=config.device,
            loss_config=loss_config
        )
        logger.info("Using MultiHeadTrainer with combined loss:")
        if use_focal:
            logger.info(f"  - FocalLoss for direction (λ={loss_config.lambda_class}, gamma={focal_gamma})")
        else:
            logger.info(f"  - CrossEntropyLoss for direction (λ={loss_config.lambda_class})")
        logger.info(f"  - HuberLoss for μ (λ={loss_config.lambda_mu})")
        logger.info(f"  - GaussianNLLLoss for σ (λ={loss_config.lambda_sigma})")
        logger.info(f"  - PinballLoss for quantiles (λ={loss_config.lambda_quantile})")
        logger.info(f"  - CrossEntropyLoss for vol_state (λ={loss_config.lambda_vol_state})")
        logger.info(f"  - HuberLoss for trading entry/SL/TP (λ={loss_config.lambda_trading})")
        logger.info(f"  - HuberLoss for candle deltas (λ={loss_config.lambda_candle})")
    else:
        # Legacy classification-only trainer
        use_focal = getattr(args, 'focal_loss', False)
        focal_gamma = getattr(args, 'focal_gamma', 2.0)
        trainer = Trainer(model, train_loader, val_loader, config, device=config.device, 
                          class_weights=class_weights, use_focal_loss=use_focal, focal_gamma=focal_gamma)
        if use_focal:
            logger.info(f"[FOCAL] Focal Loss enabled with gamma={focal_gamma}")
            logger.info(f"[FOCAL] This down-weights easy HOLD predictions, focusing on LONG/SHORT signals")
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    history = trainer.train(epochs=args.epochs)
    
    # Save model with appropriate suffix - PAIRED with matching scaler
    model_suffix = "_multihead" if use_multihead else ""
    model_prefix = f"{args.model}{model_suffix}"
    
    save_path = config.model_dir / f"{model_prefix}_trained.pt"
    model.save(str(save_path))
    logger.info(f"Model saved to {save_path}")
    
    # Save scaler with MATCHING prefix (critical for inference alignment)
    scaler_path = config.model_dir / f"{model_prefix}_scalers.joblib"
    engineer.save_scalers(str(scaler_path))
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature columns for validation at inference time
    feature_cols_path = config.model_dir / f"{model_prefix}_feature_columns.txt"
    with open(feature_cols_path, 'w') as f:
        f.write('\n'.join(features_df.columns.tolist()))
    logger.info(f"Feature columns saved to {feature_cols_path}")
    
    logger.info("Training complete!")

def train_rl(args):
    """Train reinforcement learning agent."""
    import torch
    from config import config
    from models.rl_agent import PPOAgent, TradingEnvironment
    import numpy as np
    
    logger.info("Training RL Agent with PPO...")
    
    check_gpu()
    
    dummy_data = np.random.randn(10000, 5)
    env = TradingEnvironment(
        data=dummy_data,
        initial_balance=config.rl.initial_capital,
        transaction_cost=config.rl.transaction_cost
    )
    
    agent = PPOAgent(
        state_dim=env._get_state().shape[0],
        action_dim=3,
        hidden_dim=256,
        gamma=config.rl.gamma,
        gae_lambda=config.rl.gae_lambda,
        clip_epsilon=config.rl.clip_epsilon,
        device=config.device
    )
    
    logger.info(f"RL Agent initialized. Training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            from models.rl_agent import Experience
            exp = Experience(state, action, reward, next_state, done, log_prob, value)
            agent.store_experience(exp)
            
            state = next_state
            total_reward += reward
            
        if len(agent.buffer) >= 256:
            metrics = agent.update()
            
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Trades = {info['num_trades']}, Sharpe = {info['sharpe']:.2f}")
            
    save_path = config.model_dir / "ppo_agent.pt"
    agent.save(str(save_path))
    logger.info(f"RL Agent saved to {save_path}")

def serve(args):
    """Start the FastAPI prediction server."""
    from api.server import start_server
    
    logger.info(f"Starting prediction server on port {args.port}...")
    check_gpu()
    
    start_server(host="0.0.0.0", port=args.port)

def backtest(args):
    """Run backtest on historical data using walk-forward evaluation.
    
    This implements proper hedge fund-style backtesting:
    - Purged time splits (gap between train/test)
    - Walk-forward: train on window A, test on B, roll forward
    - After-cost PnL with realistic fills
    - Per-regime performance reporting
    """
    import torch
    import numpy as np
    import pandas as pd
    from datetime import datetime as dt
    from config import config
    from data.pipeline import FeatureEngineer, create_labels
    from training.walk_forward import WalkForwardEvaluator, WalkForwardSplitter
    
    logger.info(f"Running walk-forward backtest from {args.start} to {args.end}")
    
    check_gpu()
    
    # Load data
    data_path = config.data_dir / "BTCUSDT_15m.parquet"
    if not data_path.exists():
        logger.error("No data found. Run 'python main.py fetch' first.")
        return
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} candles")
    
    # Filter by date range if timestamps are available
    if 'timestamp' in df.columns:
        start_ts = pd.Timestamp(args.start).timestamp() * 1000
        end_ts = pd.Timestamp(args.end).timestamp() * 1000
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        logger.info(f"Filtered to {len(df)} candles in date range")
    
    if len(df) < 1000:
        logger.error(f"Not enough data for backtest: {len(df)} candles")
        return
    
    # Compute features
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    features_np = features_df.values.astype(np.float32)
    
    # Load model
    model_name = args.model or "transformer"
    model_path = config.model_dir / f"{model_name}_trained.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error(f"Run 'python main.py train --model {model_name}' first")
        return
    
    # Load model based on type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = features_np.shape[1]
    
    if model_name == "transformer":
        from models.transformer import TransformerPriceModel
        model = TransformerPriceModel(input_dim=input_dim)
    elif model_name == "tft":
        from models.transformer import TemporalFusionTransformer
        model = TemporalFusionTransformer(input_dim=input_dim)
    elif model_name == "lstm":
        from models.lstm import BidirectionalLSTM
        model = BidirectionalLSTM(input_dim=input_dim)
    elif model_name == "cnn":
        from models.cnn import ResNetPrice
        model = ResNetPrice(input_dim=input_dim)
    else:
        logger.error(f"Unknown model type: {model_name}")
        return
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model: {model_name}")
    
    # Configure walk-forward evaluation
    n_folds = args.folds
    purge_gap = args.purge
    n_samples = len(features_np)
    
    # Convert days to 15m samples: 1 day = 24 hours * 4 samples/hour = 96 samples
    samples_per_day = 96
    train_days = getattr(args, 'train_days', 30)
    test_days = getattr(args, 'test_days', 7)
    
    train_periods = train_days * samples_per_day
    test_periods = test_days * samples_per_day
    embargo_periods = 48  # 12 hours at 15m
    
    logger.info(f"Window config: train={train_days}d ({train_periods} samples), test={test_days}d ({test_periods} samples)")
    
    # Validate we have enough data for at least one fold
    total_fold_size = train_periods + purge_gap + test_periods + embargo_periods
    if n_samples < total_fold_size:
        logger.error(f"Not enough data for walk-forward: need {total_fold_size}, have {n_samples}")
        logger.error(f"  Required: train={train_periods} (~30 days), purge={purge_gap}, test={test_periods} (~7 days), embargo={embargo_periods}")
        logger.error(f"  Try a longer date range or fetch more data first")
        return
    
    # Calculate step size between folds
    step_size = (n_samples - total_fold_size) // max(n_folds - 1, 1)
    if step_size <= 0:
        logger.warning(f"Data only supports 1 fold (step_size={step_size}), reducing n_folds to 1")
        n_folds = 1
    
    splitter = WalkForwardSplitter(
        n_splits=n_folds,
        train_periods=train_periods,
        test_periods=test_periods,
        purge_periods=purge_gap,
        embargo_periods=embargo_periods
    )
    
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        holding_periods=48  # 12 hours at 15m intervals
    )
    
    logger.info(f"Walk-forward config: {n_folds} folds, train={train_periods} (~30d), test={test_periods} (~7d), purge={purge_gap}")
    
    # Generate and validate ALL splits before running any evaluation
    splits = list(splitter.split(n_samples))
    
    if len(splits) == 0:
        logger.error("No valid walk-forward splits could be generated")
        return
    
    # Validate ALL folds have valid boundaries before execution - FAIL FAST on any invalid fold
    valid_splits = []
    invalid_folds = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        errors = []
        
        # Check non-empty
        if len(train_idx) == 0 or len(test_idx) == 0:
            errors.append(f"empty indices (train={len(train_idx)}, test={len(test_idx)})")
        
        # Check non-overlapping (with purge gap)
        if len(train_idx) > 0 and len(test_idx) > 0 and test_idx[0] <= train_idx[-1]:
            errors.append(f"overlapping train/test")
        
        # Check purge gap is maintained
        if len(train_idx) > 0 and len(test_idx) > 0:
            actual_gap = test_idx[0] - train_idx[-1] - 1
            if actual_gap < purge_gap:
                errors.append(f"purge gap {actual_gap} < required {purge_gap}")
        
        # Check test end doesn't exceed data
        if len(test_idx) > 0 and test_idx[-1] >= n_samples:
            errors.append(f"test exceeds data bounds")
        
        # Check expected train/test lengths (institutional requirement)
        if len(train_idx) != train_periods:
            errors.append(f"train length {len(train_idx)} != expected {train_periods}")
        if len(test_idx) != test_periods:
            errors.append(f"test length {len(test_idx)} != expected {test_periods}")
        
        if errors:
            invalid_folds.append((i, errors))
        else:
            valid_splits.append((i, train_idx, test_idx))
    
    # FAIL FAST: If any fold is invalid, abort entirely
    if invalid_folds:
        logger.error(f"{len(invalid_folds)}/{len(splits)} folds failed validation:")
        for fold_id, errors in invalid_folds:
            logger.error(f"  Fold {fold_id}: {', '.join(errors)}")
        logger.error("Aborting backtest - reduce --folds or provide more data")
        return
    
    logger.info(f"All {len(valid_splits)} folds validated successfully")
    results = []
    
    for orig_fold_id, train_idx, test_idx in valid_splits:
        logger.info(f"Fold {orig_fold_id + 1}: train[{train_idx[0]}:{train_idx[-1]}] test[{test_idx[0]}:{test_idx[-1]}]")
        
        result = evaluator.evaluate_fold(
            model=model,
            candles=df.reset_index(drop=True),
            features=features_np,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_id=orig_fold_id,
            device=device
        )
        
        results.append(result)
        
        logger.info(f"  Trades: {result.n_trades}, Win Rate: {result.win_rate:.1%}, "
                   f"Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown:.1%}")
    
    # Aggregate results
    total_trades = sum(r.n_trades for r in results)
    avg_win_rate = np.mean([r.win_rate for r in results if r.n_trades > 0])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results if r.n_trades > 0])
    avg_expectancy = np.mean([r.expectancy for r in results if r.n_trades > 0])
    max_drawdown = max(r.max_drawdown for r in results) if results else 0
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Total Trades:    {total_trades}")
    logger.info(f"Avg Win Rate:    {avg_win_rate:.1%}")
    logger.info(f"Avg Sharpe:      {avg_sharpe:.2f}")
    logger.info(f"Avg Expectancy:  {avg_expectancy:.4f}")
    logger.info(f"Max Drawdown:    {max_drawdown:.1%}")
    logger.info("="*60)
    
    # Decision: is model worth deploying?
    if avg_sharpe > 0.5 and avg_expectancy > 0:
        logger.info("✓ Model shows positive edge after costs - consider deploying")
    elif avg_sharpe > 0:
        logger.info("⚠ Model shows marginal edge - needs improvement")
    else:
        logger.info("✗ Model does NOT beat costs - do not deploy")

def train_all_mtf(args):
    """Train ALL models with MTF fusion (81-feature pipeline).
    
    This retrains all model architectures using the same MTF fusion feature
    pipeline that the dashboard uses, ensuring consistent input dimensions.
    """
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from config import config
    from data.pipeline import FeatureEngineer, TradingDataset, create_labels
    from data.mtf_fusion import MTFFeatureFusion, add_cross_asset_features
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    
    logger.info("="*60)
    logger.info("  RETRAINING ALL MODELS WITH MTF FUSION (81 FEATURES)")
    logger.info("="*60)
    
    check_gpu()
    
    # Model types to train
    model_types = args.models.split(",") if args.models else ["transformer", "tft", "lstm", "cnn", "vae", "gnn"]
    
    # Use config values for consistency
    assets = config.data.symbols  # ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    mtf_all_tfs = ["5m", "15m", "1h", "4h"]
    mtf_base_tf = "15m"
    prediction_horizon_bars = args.horizon  # Default 10 bars (~2.5 hours), configurable via CLI
    sequence_length = config.data.sequence_length
    
    logger.info(f"Config: assets={len(assets)}, horizon={prediction_horizon_bars} bars, seq_len={sequence_length}")
    
    # Define checkpoint directory early for feature list saving
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading MTF data for assets: {assets}")
    data_dir = config.data_dir
    asset_data = {}
    total_candles = 0
    
    for asset in assets:
        tf_data = {}
        for tf in mtf_all_tfs:
            asset_path = data_dir / f"{asset}_{tf}.parquet"
            if asset_path.exists():
                df = pd.read_parquet(asset_path)
                tf_data[tf] = df
                total_candles += len(df)
                logger.info(f"  {asset} {tf}: {len(df):,} candles")
        if tf_data:
            asset_data[asset] = tf_data
    
    if not asset_data:
        logger.error("No data files found! Run 'python main.py fetch' first.")
        return
    
    logger.info(f"Total: {total_candles:,} raw candles")
    
    # Fuse timeframes using MTF fusion
    logger.info("Fusing timeframes to 15m base (leakage-proof alignment)...")
    fusioner = MTFFeatureFusion(prediction_horizon_bars)
    all_fused = []
    
    for symbol, tf_data in asset_data.items():
        if mtf_base_tf not in tf_data:
            logger.warning(f"  {symbol}: Skipping - no 15m data")
            continue
        fused = fusioner.align_timeframes(tf_data, symbol)
        fused["symbol"] = symbol
        all_fused.append(fused)
        logger.info(f"  {symbol}: {len(fused):,} fused samples, {len(fused.columns)} features")
    
    if not all_fused:
        logger.error("No fused data!")
        return
    
    # Combine all assets
    combined = pd.concat(all_fused, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} samples")
    
    # NOTE: Cross-asset features REMOVED (FIX #1 - training-inference alignment)
    # These features (btc_ret, relative_strength_vs_btc, btc_correlation_proxy, 
    # sector_momentum, outperform_sector) cannot be computed at inference time
    # because we only have BTC candles. Removing them ensures training and
    # inference use identical feature sets.
    # If you need cross-asset features, you must also compute them during inference
    # by fetching ETH/SOL/BNB candles.
    logger.info("Skipping cross-asset features (BTC-only inference alignment)")
    
    # Time-based train/val split per asset with proper purge gap
    logger.info("Splitting by time per asset with leakage-safe purge gap...")
    train_ratio = 0.70
    val_ratio = 0.15
    
    # Purge gap must be at least prediction_horizon + sequence_length to prevent lookahead
    purge_gap = prediction_horizon_bars + sequence_length
    
    train_dfs = []
    val_dfs = []
    
    for symbol in combined["symbol"].unique():
        asset_df = combined[combined["symbol"] == symbol].copy()
        asset_df = asset_df.sort_values("datetime").reset_index(drop=True)
        n = len(asset_df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Split with explicit purge gap between train and val
        train_end = n_train - purge_gap  # Train ends before purge gap
        val_start = n_train  # Val starts after purge gap
        val_end = min(val_start + n_val, n)  # Ensure val_end doesn't exceed data
        
        # Bounds validation
        min_train_samples = sequence_length * 2
        min_val_samples = sequence_length
        
        if train_end <= 0:
            logger.warning(f"  {symbol}: Skipping - insufficient data for training (train_end={train_end})")
            continue
        if train_end < min_train_samples:
            logger.warning(f"  {symbol}: Skipping - train samples {train_end} < min {min_train_samples}")
            continue
        if val_end - val_start < min_val_samples:
            logger.warning(f"  {symbol}: Skipping - val samples {val_end - val_start} < min {min_val_samples}")
            continue
        
        # Leakage validation: train labels at train_end-1 look ahead horizon bars
        # This lookahead must not cross into validation (val_start)
        max_label_lookahead = train_end - 1 + prediction_horizon_bars
        if max_label_lookahead >= val_start:
            logger.error(f"  {symbol}: LEAKAGE - train labels ({max_label_lookahead}) would cross into val ({val_start})")
            logger.error(f"    Increase purge_gap or provide more data")
            continue
        
        train_df = asset_df.iloc[:train_end].copy()
        val_df = asset_df.iloc[val_start:val_end].copy()
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        logger.info(f"  {symbol}: train={len(train_df):,}, purge={purge_gap}, val={len(val_df):,} (leakage check: PASSED)")
    
    train_combined = pd.concat(train_dfs, ignore_index=True)
    val_combined = pd.concat(val_dfs, ignore_index=True)
    
    # Create labels
    logger.info(f"Creating labels (horizon={prediction_horizon_bars} bars, ~2.5h)...")
    train_labels = fusioner.create_labels(train_combined)
    val_labels = fusioner.create_labels(val_combined)
    
    # Drop rows with NaN labels
    train_valid = train_labels.notna()
    val_valid = val_labels.notna()
    train_combined = train_combined[train_valid].reset_index(drop=True)
    train_labels = train_labels[train_valid].reset_index(drop=True)
    val_combined = val_combined[val_valid].reset_index(drop=True)
    val_labels = val_labels[val_valid].reset_index(drop=True)
    
    # Select numeric feature columns only - SORTED for deterministic ordering
    exclude_cols = {"datetime", "symbol", "timestamp", "open", "high", "low", "close", "volume"}
    feature_cols = sorted([c for c in train_combined.columns 
                          if c not in exclude_cols and train_combined[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]])
    
    # Explicit feature count validation
    n_features = len(feature_cols)
    logger.info(f"MTF features: {n_features} columns (sorted, deterministic order)")
    
    # Assert expected feature count range (should be ~80-85 for MTF fusion with cross-asset)
    # This catches schema drift or unexpected column additions
    MIN_EXPECTED_FEATURES = 70
    MAX_EXPECTED_FEATURES = 100
    if n_features < MIN_EXPECTED_FEATURES or n_features > MAX_EXPECTED_FEATURES:
        logger.error(f"Feature count {n_features} outside expected range [{MIN_EXPECTED_FEATURES}, {MAX_EXPECTED_FEATURES}]")
        logger.error(f"This indicates schema drift or pipeline mismatch - aborting")
        return
    
    # Save feature list for prediction pipeline consistency
    feature_list_path = checkpoint_dir / "feature_columns.txt"
    with open(feature_list_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    logger.info(f"Saved feature list ({n_features} columns) to {feature_list_path}")
    
    # Log first/last few features for verification
    if n_features > 6:
        logger.info(f"  First 3: {feature_cols[:3]}")
        logger.info(f"  Last 3: {feature_cols[-3:]}")
    
    train_features_raw = train_combined[feature_cols].copy()
    val_features_raw = val_combined[feature_cols].copy()
    train_labels_np = train_labels.values.astype(np.int64)
    val_labels_np = val_labels.values.astype(np.int64)
    logger.info(f"Train: {len(train_features_raw):,}, Val: {len(val_features_raw):,}")
    
    # Fit scalers on training data only
    engineer = FeatureEngineer()
    logger.info("Fitting scalers on training data only (no leakage)")
    engineer.fit_scalers(train_features_raw)
    
    # Transform both sets
    train_features_scaled = engineer.transform(train_features_raw)
    val_features_scaled = engineer.transform(val_features_raw)
    
    train_features_np = train_features_scaled.values.astype(np.float32)
    val_features_np = val_features_scaled.values.astype(np.float32)
    
    # Skip initial sequence_length samples (sequence_length already defined at top)
    valid_start = sequence_length
    train_features_np = train_features_np[valid_start:]
    train_labels_np = train_labels_np[valid_start:]
    val_features_np = val_features_np[valid_start:]
    val_labels_np = val_labels_np[valid_start:]
    
    # Clean data - drop NaN/Inf rows
    def clean_data(features, labels, name):
        features = np.where(np.isinf(features), np.nan, features)
        nan_mask = np.isnan(features).any(axis=1)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"{name}: Dropping {nan_count} rows with NaN/Inf ({nan_count/len(features)*100:.1f}%)")
            valid_mask = ~nan_mask
            features = features[valid_mask]
            labels = labels[valid_mask]
        assert np.isfinite(features).all(), f"{name}: Non-finite values remain!"
        logger.info(f"{name}: {len(features)} clean samples")
        return features, labels
    
    train_features_np, train_labels_np = clean_data(train_features_np, train_labels_np, "Train")
    val_features_np, val_labels_np = clean_data(val_features_np, val_labels_np, "Val")
    
    # Class weights
    MAX_CLASS_WEIGHT = 10.0
    class_counts = np.bincount(train_labels_np, minlength=3)
    total_samples = len(train_labels_np)
    class_weights = total_samples / (3 * class_counts + 1e-6)
    class_weights = np.clip(class_weights, 1.0, MAX_CLASS_WEIGHT)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"Class distribution: SHORT={class_counts[0]:,}, NEUTRAL={class_counts[1]:,}, LONG={class_counts[2]:,}")
    logger.info(f"Class weights (capped at {MAX_CLASS_WEIGHT}x): [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]")
    
    # Create datasets
    train_dataset = TradingDataset(train_features_np, train_labels_np, sequence_length, validate_data=True)
    val_dataset = TradingDataset(val_features_np, val_labels_np, sequence_length, validate_data=True)
    
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    input_dim = train_features_np.shape[1]
    logger.info(f"")
    logger.info(f"Input dimension: {input_dim} features")
    logger.info(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
    logger.info(f"")
    
    # Save the scaler for prediction use - must match server's expected path
    # Server loads from: checkpoints/scaler.joblib (checkpoint_dir already defined at top)
    scaler_path = checkpoint_dir / "scaler.joblib"
    engineer.save_scalers(str(scaler_path))
    logger.info(f"Saved scalers to {scaler_path}")
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"")
        logger.info("="*60)
        logger.info(f"  TRAINING: {model_type.upper()}")
        logger.info("="*60)
        
        try:
            # Use config values for hyperparameters to ensure consistency
            if model_type == "transformer":
                from models.transformer import TransformerPriceModel
                model = TransformerPriceModel(
                    input_dim=input_dim,
                    d_model=config.model.transformer_dim,
                    nhead=config.model.transformer_heads,
                    num_layers=config.model.transformer_layers
                )
            elif model_type == "tft":
                from models.transformer import TemporalFusionTransformer
                model = TemporalFusionTransformer(
                    input_dim=input_dim,
                    d_model=config.model.transformer_dim,
                    nhead=config.model.transformer_heads
                )
            elif model_type == "lstm":
                from models.lstm import BidirectionalLSTM
                model = BidirectionalLSTM(
                    input_dim=input_dim,
                    hidden_dim=config.model.lstm_hidden,
                    num_layers=config.model.lstm_layers
                )
            elif model_type == "cnn":
                from models.cnn import ResNetPrice
                model = ResNetPrice(
                    input_dim=input_dim,
                    channels=config.model.cnn_channels
                )
            elif model_type == "vae":
                from models.vae import MarketVAE
                model = MarketVAE(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    latent_dim=config.model.vae_latent_dim
                )
            elif model_type == "gnn":
                from models.gnn import CrossAssetGNN
                model = CrossAssetGNN(
                    input_dim=input_dim,
                    num_assets=len(assets)
                )
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            logger.info(f"Parameters: {model.count_parameters():,}")
            
            config.training.epochs = args.epochs
            config.training.learning_rate = args.lr
            
            trainer = Trainer(model, train_loader, val_loader, config, device=config.device,
                              class_weights=class_weights_tensor)
            
            history = trainer.train(epochs=args.epochs)
            
            # Save model
            save_path = config.model_dir / f"{model_type}_mtf_trained.pt"
            model.save(str(save_path))
            logger.info(f"Saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("")
    logger.info("="*60)
    logger.info("  RETRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Models trained with {input_dim} features (MTF fusion pipeline)")
    logger.info(f"Checkpoints saved to: {config.training.checkpoint_dir}")
    logger.info(f"Now restart GPU trainer API: python main.py serve")

def main():
    parser = argparse.ArgumentParser(
        description="BTC Futures Trading - GPU Neural Network Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    fetch_parser = subparsers.add_parser("fetch", help="Fetch historical data")
    fetch_parser.add_argument("--candles", type=int, default=175000, help="Number of candles to fetch (default: 5 years of 15m data)")
    fetch_parser.add_argument("--replit-proxy", type=str, dest="replit_proxy",
                              help="Replit proxy URL for Binance data (e.g., https://your-app.replit.app)")
    
    train_parser = subparsers.add_parser("train", help="Train a neural network model")
    train_parser.add_argument("--model", type=str, required=True,
                             choices=["transformer", "tft", "lstm", "cnn", "vae", "gnn", "simple_mlp", "multihead_simple_mlp", "enhanced_mlp"],
                             help="Model type to train (use 'simple_mlp' for stable baseline, 'multihead_simple_mlp' for progressive head testing, 'enhanced_mlp' for deeper architecture)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--horizon", type=int, default=16, 
                             help="Label lookahead horizon in bars (default: 16 = 4h at 15m timeframe)")
    train_parser.add_argument("--cost", type=float, default=0.0009, 
                             help="Fixed round-trip trading cost (default: 0.09%% = 0.0009)")
    train_parser.add_argument("--min-net-edge", type=float, default=0.0, dest="min_net_edge",
                             help="Minimum net edge after costs for trade signals (default: 0.0 = no filter)")
    train_parser.add_argument("--min-confidence", type=float, default=0.40, dest="min_confidence",
                             help="Minimum mu/sigma ratio for trade signals (default: 0.40, lowered from 0.7)")
    train_parser.add_argument("--volatility-cost", action="store_true", dest="volatility_cost",
                             help="Use volatility-based cost instead of fixed cost")
    train_parser.add_argument("--pure-directional", action="store_true", dest="pure_directional",
                             help="Stage 2: Use simple return threshold instead of cost-aware gating")
    train_parser.add_argument("--directional-threshold", type=float, default=0.0020, dest="directional_threshold",
                             help="Return threshold for pure directional mode (default: 0.20%% = 0.0020)")
    train_parser.add_argument("--regime-labels", action="store_true", dest="regime_labels",
                             help="Stage 3: Use ADX-based adaptive thresholds for different regimes")
    train_parser.add_argument("--trend-threshold", type=float, default=0.0015, dest="trend_threshold",
                             help="Return threshold for trending regime (default: 0.15%% = 0.0015)")
    train_parser.add_argument("--range-threshold", type=float, default=0.0030, dest="range_threshold",
                             help="Return threshold for ranging regime (default: 0.30%% = 0.0030)")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--multihead", action="store_true", 
                             help="Use multi-head training with combined loss (Classification + Regression + Quantile)")
    # Progressive head enablement for MultiHeadSimpleMLP
    train_parser.add_argument("--enable-quantile", action="store_true", dest="enable_quantile",
                             help="Enable quantile head (MultiHeadSimpleMLP only)")
    train_parser.add_argument("--enable-vol-state", action="store_true", dest="enable_vol_state",
                             help="Enable volatility state head (MultiHeadSimpleMLP only)")
    train_parser.add_argument("--enable-mu", action="store_true", dest="enable_mu",
                             help="Enable mu/expected return head (MultiHeadSimpleMLP only)")
    train_parser.add_argument("--enable-sigma", action="store_true", dest="enable_sigma",
                             help="Enable sigma/uncertainty head - most unstable (MultiHeadSimpleMLP only)")
    train_parser.add_argument("--focal-loss", action="store_true", dest="focal_loss",
                             help="Use Focal Loss instead of CrossEntropy (down-weights easy HOLD predictions)")
    train_parser.add_argument("--focal-gamma", type=float, default=2.0, dest="focal_gamma",
                             help="Focal loss gamma parameter (default: 2.0, higher = more focus on hard examples)")
    train_parser.add_argument("--class-weight-cap", type=float, default=10.0, dest="class_weight_cap",
                             help="Max class weight multiplier (default: 10.0, increase to boost LONG/SHORT)")
    
    train_all_parser = subparsers.add_parser("train-all", help="Retrain ALL models with MTF fusion (81 features)")
    train_all_parser.add_argument("--models", type=str, default="transformer,tft,lstm,cnn,vae,gnn",
                                  help="Comma-separated list of models to train (default: all)")
    train_all_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per model")
    train_all_parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Batch size")
    train_all_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_all_parser.add_argument("--horizon", type=int, default=16, 
                                  help="Prediction horizon in 15m bars (default: 16 = 4h)")
    train_all_parser.add_argument("--cost", type=float, default=0.0009, 
                                  help="Fixed round-trip trading cost (default: 0.09%%)")
    train_all_parser.add_argument("--min-net-edge", type=float, default=0.0, dest="min_net_edge",
                                  help="Minimum net edge after costs (default: 0.0)")
    train_all_parser.add_argument("--min-confidence", type=float, default=0.40, dest="min_confidence",
                                  help="Minimum mu/sigma ratio (default: 0.40, lowered from 0.7)")
    train_all_parser.add_argument("--pure-directional", action="store_true", dest="pure_directional",
                                  help="Stage 2: Use simple return threshold instead of cost-aware gating")
    train_all_parser.add_argument("--directional-threshold", type=float, default=0.0020, dest="directional_threshold",
                                  help="Return threshold for pure directional mode (default: 0.20%%)")
    train_all_parser.add_argument("--regime-labels", action="store_true", dest="regime_labels",
                                  help="Stage 3: Use ADX-based adaptive thresholds for different regimes")
    train_all_parser.add_argument("--trend-threshold", type=float, default=0.0015, dest="trend_threshold",
                                  help="Return threshold for trending regime (default: 0.15%%)")
    train_all_parser.add_argument("--range-threshold", type=float, default=0.0030, dest="range_threshold",
                                  help="Return threshold for pure directional mode (default: 0.20%%)")
    
    rl_parser = subparsers.add_parser("train-rl", help="Train reinforcement learning agent")
    rl_parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    
    serve_parser = subparsers.add_parser("serve", help="Start prediction API server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--model", type=str, default="transformer", help="Model to use")
    backtest_parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    backtest_parser.add_argument("--purge", type=int, default=100, help="Purge gap (samples) between train/test")
    backtest_parser.add_argument("--train-days", type=int, default=30, dest="train_days", help="Training window in days (default: 30)")
    backtest_parser.add_argument("--test-days", type=int, default=7, dest="test_days", help="Test window in days (default: 7)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    if args.command == "fetch":
        asyncio.run(fetch_data(args))
    elif args.command == "train":
        train(args)
    elif args.command == "train-all":
        train_all_mtf(args)
    elif args.command == "train-rl":
        train_rl(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "backtest":
        backtest(args)

if __name__ == "__main__":
    main()
