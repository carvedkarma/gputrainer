"""
Multi-Timeframe Feature Fusion Module

Implements leakage-proof multi-timeframe feature alignment where:
- Base timeframe: 15m (all predictions aligned to 15m candle closes)
- Context timeframes: 5m (micro), 1h (trend), 4h (regime)
- Alignment: merge_asof(direction="backward") to prevent lookahead

Each 15m candle gets enriched with context from other timeframes,
only using data that was available at the 15m candle close time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MTFFeatureFusion:
    """
    Multi-Timeframe Feature Fusion with leakage-proof alignment.
    
    All timeframes are aligned to the base 15m timeline using
    merge_asof with direction='backward' to ensure only completed
    candles are used.
    """
    
    BASE_TF = "15m"
    CONTEXT_TFS = ["5m", "1h", "4h"]
    ALL_TFS = ["5m", "15m", "1h", "4h"]
    
    TF_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240
    }
    
    def __init__(self, prediction_horizon_bars: int = 10):
        """
        Args:
            prediction_horizon_bars: Number of 15m bars for prediction (10 = 2.5 hours)
        """
        self.prediction_horizon = prediction_horizon_bars
        
    def align_timeframes(
        self,
        data_by_tf: Dict[str, pd.DataFrame],
        symbol: str
    ) -> pd.DataFrame:
        """
        Align all timeframes to the 15m base timeline.
        
        For each 15m timestamp t:
        - 5m features use only candles with close <= t
        - 1h/4h features use the most recent completed candle with close <= t
        
        Args:
            data_by_tf: Dict mapping timeframe -> DataFrame with OHLCV data
            symbol: Asset symbol for logging
            
        Returns:
            DataFrame with 15m index and aligned features from all timeframes
        """
        if self.BASE_TF not in data_by_tf:
            raise ValueError(f"Base timeframe {self.BASE_TF} not found in data")
        
        base_df = data_by_tf[self.BASE_TF].copy()
        
        # Handle multiple possible timestamp column names from different sources
        timestamp_cols = ["timestamp", "open_time", "openTime", "time", "datetime"]
        found_col = None
        for col in timestamp_cols:
            if col in base_df.columns:
                found_col = col
                break
        
        if found_col:
            # Check if already datetime or needs conversion from ms
            if pd.api.types.is_datetime64_any_dtype(base_df[found_col]):
                base_df["datetime"] = base_df[found_col]
            else:
                base_df["datetime"] = pd.to_datetime(base_df[found_col], unit="ms")
        elif isinstance(base_df.index, pd.DatetimeIndex):
            base_df["datetime"] = base_df.index
        else:
            raise ValueError(f"Base DataFrame must have timestamp column (one of {timestamp_cols}) or DatetimeIndex. Found columns: {list(base_df.columns)}")
            
        base_df = base_df.sort_values("datetime").reset_index(drop=True)
        
        features_15m = self._compute_tf_features(base_df, "15m")
        fused = base_df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        for col in features_15m.columns:
            fused[col] = features_15m[col].values
        
        for tf in self.CONTEXT_TFS:
            if tf not in data_by_tf:
                logger.warning(f"Context timeframe {tf} not found for {symbol}, skipping")
                continue
                
            tf_df = data_by_tf[tf].copy()
            
            # Handle multiple possible timestamp column names
            found_col = None
            for col in timestamp_cols:
                if col in tf_df.columns:
                    found_col = col
                    break
            
            if found_col:
                if pd.api.types.is_datetime64_any_dtype(tf_df[found_col]):
                    tf_df["datetime"] = tf_df[found_col]
                else:
                    tf_df["datetime"] = pd.to_datetime(tf_df[found_col], unit="ms")
            elif isinstance(tf_df.index, pd.DatetimeIndex):
                tf_df["datetime"] = tf_df.index
            else:
                logger.warning(f"No timestamp column found for {tf}, skipping")
                continue
                
            tf_df = tf_df.sort_values("datetime").reset_index(drop=True)
            
            tf_features = self._compute_tf_features(tf_df, tf)
            tf_features["datetime"] = tf_df["datetime"]
            
            fused = pd.merge_asof(
                fused.sort_values("datetime"),
                tf_features.sort_values("datetime"),
                on="datetime",
                direction="backward",
                suffixes=("", f"_{tf}_dup")
            )
            
        confluence = self._compute_confluence_features(fused)
        for col in confluence.columns:
            fused[col] = confluence[col].values
            
        logger.info(f"[{symbol}] MTF fusion complete: {len(fused)} rows, {len(fused.columns)} features")
        
        return fused
    
    def fuse(self, symbol: str, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compatibility wrapper used by api/server.py.
        Aligns all timeframes to the base timeframe without lookahead.
        
        Args:
            symbol: Asset symbol for logging
            data_by_tf: Dict mapping timeframe -> DataFrame with OHLCV data
            
        Returns:
            DataFrame with 15m index and aligned features from all timeframes
        """
        return self.align_timeframes(data_by_tf, symbol)
    
    def _compute_tf_features(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        Compute features for a single timeframe with proper suffix.
        
        Args:
            df: OHLCV DataFrame
            tf: Timeframe string for suffix (e.g., "5m", "15m", "1h", "4h")
            
        Returns:
            DataFrame with features suffixed by timeframe
        """
        suffix = f"_{tf}"
        features = pd.DataFrame(index=df.index)
        
        features[f"ret_1{suffix}"] = df["close"].pct_change(1)
        features[f"ret_3{suffix}"] = df["close"].pct_change(3)
        features[f"ret_6{suffix}"] = df["close"].pct_change(6)
        features[f"ret_12{suffix}"] = df["close"].pct_change(12)
        
        if tf == "5m":
            features[f"ret_20{suffix}"] = df["close"].pct_change(20)
            features[f"rolling_vol_20{suffix}"] = df["close"].pct_change().rolling(20).std()
            features[f"range_pct{suffix}"] = (df["high"] - df["low"]) / df["close"]
            features[f"wick_upper{suffix}"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-8)
            features[f"wick_lower{suffix}"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-8)
            features[f"body_ratio{suffix}"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
            
        elif tf == "15m":
            features[f"rsi_14{suffix}"] = self._compute_rsi(df["close"], 14)
            features[f"rsi_7{suffix}"] = self._compute_rsi(df["close"], 7)
            features[f"atr_14{suffix}"] = self._compute_atr(df, 14)
            features[f"atr_7{suffix}"] = self._compute_atr(df, 7)
            features[f"rolling_vol_20{suffix}"] = df["close"].pct_change().rolling(20).std()
            features[f"range_pct{suffix}"] = (df["high"] - df["low"]) / df["close"]
            features[f"wick_upper{suffix}"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-8)
            features[f"wick_lower{suffix}"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-8)
            features[f"body_ratio{suffix}"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
            features[f"volume_ratio{suffix}"] = df["volume"] / df["volume"].rolling(20).mean()
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            features[f"macd{suffix}"] = ema_12 - ema_26
            features[f"macd_signal{suffix}"] = features[f"macd{suffix}"].ewm(span=9).mean()
            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            features[f"bb_position{suffix}"] = (df["close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-8)
            
        elif tf == "1h":
            features[f"rsi_14{suffix}"] = self._compute_rsi(df["close"], 14)
            features[f"ema_slope_12{suffix}"] = df["close"].ewm(span=12).mean().diff(3) / df["close"]
            features[f"ema_slope_26{suffix}"] = df["close"].ewm(span=26).mean().diff(3) / df["close"]
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            features[f"ema_diff{suffix}"] = (ema_12 - ema_26) / df["close"]
            features[f"rolling_vol_50{suffix}"] = df["close"].pct_change().rolling(50).std()
            features[f"atr_14{suffix}"] = self._compute_atr(df, 14)
            features[f"adx_14{suffix}"] = self._compute_adx(df, 14)
            features[f"trend_strength{suffix}"] = abs(df["close"].pct_change(20)) / (df["close"].pct_change().rolling(20).std() + 1e-8)
            
        elif tf == "4h":
            features[f"trend_strength{suffix}"] = abs(df["close"].pct_change(20)) / (df["close"].pct_change().rolling(20).std() + 1e-8)
            features[f"ema_slope_12{suffix}"] = df["close"].ewm(span=12).mean().diff(3) / df["close"]
            features[f"rolling_vol_20{suffix}"] = df["close"].pct_change().rolling(20).std()
            ema_50 = df["close"].ewm(span=50).mean()
            features[f"dist_ema_50{suffix}"] = (df["close"] - ema_50) / ema_50
            ema_200 = df["close"].ewm(span=200).mean()
            features[f"dist_ema_200{suffix}"] = (df["close"] - ema_200) / ema_200
            features[f"vol_regime{suffix}"] = df["close"].pct_change().rolling(20).std() / df["close"].pct_change().rolling(100).std()
            features[f"rsi_14{suffix}"] = self._compute_rsi(df["close"], 14)
            
        return features
    
    def _compute_confluence_features(self, fused: pd.DataFrame) -> pd.DataFrame:
        """
        Compute confluence features that combine multiple timeframes.
        
        These capture alignment/divergence between timeframes which is
        where a lot of edge comes from.
        """
        confluence = pd.DataFrame(index=fused.index)
        
        if "ret_1_15m" in fused.columns and "ret_1_1h" in fused.columns:
            sign_15m = np.sign(fused["ret_1_15m"])
            sign_1h = np.sign(fused["ret_1_1h"])
            confluence["trend_agree_15m_1h"] = (sign_15m == sign_1h).astype(float)
            
        if "ret_1_15m" in fused.columns and "ret_1_4h" in fused.columns:
            sign_15m = np.sign(fused["ret_1_15m"])
            sign_4h = np.sign(fused["ret_1_4h"])
            confluence["trend_agree_15m_4h"] = (sign_15m == sign_4h).astype(float)
            
        if "ret_1_1h" in fused.columns and "ret_1_4h" in fused.columns:
            sign_1h = np.sign(fused["ret_1_1h"])
            sign_4h = np.sign(fused["ret_1_4h"])
            confluence["trend_agree_1h_4h"] = (sign_1h == sign_4h).astype(float)
            
        if "ret_1_5m" in fused.columns and "ret_1_15m" in fused.columns and "ret_1_1h" in fused.columns:
            all_bullish = (fused["ret_1_5m"] > 0) & (fused["ret_1_15m"] > 0) & (fused["ret_1_1h"] > 0)
            all_bearish = (fused["ret_1_5m"] < 0) & (fused["ret_1_15m"] < 0) & (fused["ret_1_1h"] < 0)
            confluence["momentum_aligned"] = (all_bullish | all_bearish).astype(float)
            
        rsi_cols = [c for c in fused.columns if c.startswith("rsi_14_")]
        if len(rsi_cols) >= 2:
            rsi_above_50 = sum((fused[c] > 50).astype(float) for c in rsi_cols)
            confluence["rsi_bullish_count"] = rsi_above_50
            confluence["rsi_all_bullish"] = (rsi_above_50 == len(rsi_cols)).astype(float)
            confluence["rsi_all_bearish"] = (rsi_above_50 == 0).astype(float)
            
        if "rolling_vol_20_15m" in fused.columns and "rolling_vol_50_1h" in fused.columns:
            confluence["vol_ratio_15m_1h"] = fused["rolling_vol_20_15m"] / (fused["rolling_vol_50_1h"] + 1e-8)
            
        if "rolling_vol_20_15m" in fused.columns and "rolling_vol_20_4h" in fused.columns:
            confluence["vol_ratio_15m_4h"] = fused["rolling_vol_20_15m"] / (fused["rolling_vol_20_4h"] + 1e-8)
            
        if "atr_14_15m" in fused.columns and "atr_14_1h" in fused.columns:
            confluence["atr_ratio_15m_1h"] = fused["atr_14_15m"] / (fused["atr_14_1h"] + 1e-8)
            
        ema_slope_cols = [c for c in fused.columns if "ema_slope" in c]
        if ema_slope_cols:
            slopes = fused[ema_slope_cols]
            all_positive = (slopes > 0).all(axis=1)
            all_negative = (slopes < 0).all(axis=1)
            confluence["ema_slopes_aligned"] = (all_positive | all_negative).astype(float)
            
        return confluence
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average Directional Index."""
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._compute_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(period).mean()
        return adx
    
    def create_labels(self, fused_df: pd.DataFrame) -> pd.Series:
        """
        Create classification labels based on forward returns.
        
        Uses the prediction horizon (number of 15m bars forward).
        
        Returns:
            Series with labels: 0=SHORT, 1=HOLD, 2=LONG
        """
        forward_ret = fused_df["close"].shift(-self.prediction_horizon) / fused_df["close"] - 1
        
        threshold = forward_ret.rolling(100, min_periods=20).std() * 0.5
        threshold = threshold.fillna(0.003)
        
        labels = pd.Series(1, index=fused_df.index)
        labels[forward_ret > threshold] = 2
        labels[forward_ret < -threshold] = 0
        
        labels.iloc[-self.prediction_horizon:] = np.nan
        
        return labels


def fuse_multi_timeframe_data(
    asset_data: Dict[str, Dict[str, pd.DataFrame]],
    prediction_horizon_bars: int = 10
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fuse multi-timeframe data for all assets.
    
    Args:
        asset_data: Dict mapping symbol -> (timeframe -> DataFrame)
                   e.g., {"BTCUSDT": {"5m": df1, "15m": df2, "1h": df3, "4h": df4}}
        prediction_horizon_bars: Number of 15m bars for prediction target
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    fusioner = MTFFeatureFusion(prediction_horizon_bars)
    
    all_features = []
    all_labels = []
    
    for symbol, tf_data in asset_data.items():
        logger.info(f"Fusing MTF data for {symbol}...")
        
        fused = fusioner.align_timeframes(tf_data, symbol)
        
        fused["symbol"] = symbol
        
        labels = fusioner.create_labels(fused)
        
        valid_mask = labels.notna()
        all_features.append(fused[valid_mask])
        all_labels.append(labels[valid_mask])
        
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    logger.info(f"MTF fusion complete: {len(combined_features)} total samples from {len(asset_data)} assets")
    
    return combined_features, combined_labels


def add_cross_asset_features(
    combined_df: pd.DataFrame,
    reference_symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    """
    Add cross-asset features (BTC correlation, relative strength, etc.)
    
    These features capture relationships between assets which can provide
    additional predictive signal.
    
    Args:
        combined_df: Combined DataFrame with 'symbol' and 'datetime' columns
        reference_symbol: Symbol to use as reference (usually BTC)
        
    Returns:
        DataFrame with cross-asset features added
    """
    if "symbol" not in combined_df.columns or "datetime" not in combined_df.columns:
        logger.warning("Cannot compute cross-asset features: missing 'symbol' or 'datetime' columns")
        return combined_df
    
    ref_data = combined_df[combined_df["symbol"] == reference_symbol][["datetime", "ret_1_15m"]].copy()
    ref_data = ref_data.rename(columns={"ret_1_15m": "btc_ret"})
    ref_data = ref_data.drop_duplicates("datetime").sort_values("datetime")
    
    result = pd.merge_asof(
        combined_df.sort_values("datetime"),
        ref_data,
        on="datetime",
        direction="backward"
    )
    
    result["btc_correlation_proxy"] = result["ret_1_15m"] * result["btc_ret"]
    
    result["relative_strength_vs_btc"] = result["ret_1_15m"] - result["btc_ret"]
    
    sector_mean = combined_df.groupby("datetime")["ret_1_15m"].transform("mean")
    result["sector_momentum"] = sector_mean
    
    result["outperform_sector"] = (result["ret_1_15m"] > sector_mean).astype(float)
    
    return result
