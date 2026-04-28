"""
Regression Training Targets for Institutional-Grade Trading

Instead of classification (LONG/SHORT/HOLD), we predict:
- μ (mu): Expected future return over horizon (4h default)
- σ (sigma): Uncertainty/volatility of the prediction
- P(move > cost): Probability that net move beats transaction costs
- Quantiles: p10, p50, p90 for asymmetric risk assessment

The trading signal is then computed as:
    edge = μ - cost
    confidence = edge / σ
    enter_trade = confidence > threshold
    
This approach naturally produces "strong signals only" because weak 
signals have low edge and high uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Transaction costs for edge calculation."""
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee
    slippage_base: float = 0.0001  # Base slippage
    slippage_vol_mult: float = 0.5  # Slippage increases with volatility
    funding_per_8h: float = 0.0001  # Average funding rate
    
    def total_round_trip_cost(self, volatility: float = 0.01, 
                               is_taker: bool = True,
                               hold_hours: float = 4) -> float:
        """
        Calculate total round-trip transaction cost.
        
        Args:
            volatility: Current volatility (for slippage estimate)
            is_taker: Whether using taker orders
            hold_hours: Expected hold time in hours
            
        Returns:
            Total cost as a fraction (e.g., 0.001 = 0.1%)
        """
        fee = self.taker_fee if is_taker else self.maker_fee
        slippage = self.slippage_base + self.slippage_vol_mult * volatility
        
        funding_periods = hold_hours / 8
        funding_cost = abs(self.funding_per_8h) * funding_periods
        
        total = (fee * 2) + (slippage * 2) + funding_cost
        
        return total


class RegressionTargetGenerator:
    """
    Generates regression targets from price data.
    
    Targets:
    1. mu (μ): Expected return = (close_future - close_current) / close_current
    2. sigma (σ): Realized volatility over the horizon
    3. p_profitable: Probability that |move| > cost (from historical distribution)
    4. quantiles: p10, p50, p90 of return distribution
    
    The model learns to predict these directly, then we compute:
        edge = μ - cost
        confidence = edge / σ
    """
    
    def __init__(self, 
                 horizon_periods: int = 16,  # 4 hours in 15-minute candles (v4.5.1 default)
                 lookback_periods: int = 48,  # 12 hours for volatility
                 costs: Optional[TradingCosts] = None):
        self.horizon_periods = horizon_periods
        self.lookback_periods = lookback_periods
        self.costs = costs or TradingCosts()
        
        logger.info(f"RegressionTargetGenerator: horizon={horizon_periods}, lookback={lookback_periods}")
        
    def compute_forward_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward returns over horizon.
        
        Returns:
            Series of forward returns (future_close / current_close - 1)
        """
        future_prices = prices.shift(-self.horizon_periods)
        returns = (future_prices - prices) / prices
        return returns
    
    def compute_realized_volatility(self, prices: pd.Series, bars_per_day: int = 96) -> pd.Series:
        """
        Compute realized volatility (standard deviation of returns).
        
        Uses rolling window of lookback periods.
        
        Args:
            bars_per_day: Number of bars per day for annualization (96 for 15m, 288 for 5m)
        """
        log_returns = np.log(prices / prices.shift(1))
        volatility = log_returns.rolling(window=self.lookback_periods).std()
        
        annualization = np.sqrt(bars_per_day)  # 96 for 15m, 288 for 5m
        volatility_annualized = volatility * annualization
        
        return volatility_annualized
    
    def compute_forward_volatility(self, prices: pd.Series, bars_per_day: int = 96) -> pd.Series:
        """
        Compute forward realized volatility (uncertainty of the prediction).
        
        This is the actual volatility that will occur during the holding period.
        
        Args:
            bars_per_day: Number of bars per day for annualization (96 for 15m, 288 for 5m)
        """
        log_returns = np.log(prices / prices.shift(1))
        
        forward_vol = log_returns.shift(-self.horizon_periods).rolling(
            window=self.horizon_periods
        ).std()
        
        forward_vol = forward_vol.shift(self.horizon_periods)
        
        # FIXED: Use correct annualization for 15m bars (was hardcoded to 288 for 5m)
        annualization = np.sqrt(bars_per_day)  # 96 for 15m, 288 for 5m
        
        return forward_vol * annualization
    
    def compute_max_favorable_excursion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Maximum Favorable Excursion (MFE) - the best price during horizon.
        
        For long: max high during horizon relative to entry
        For short: min low during horizon relative to entry
        
        Returns the maximum potential profit during the trade.
        """
        mfe = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df) - self.horizon_periods):
            entry_close = df["close"].iloc[i]
            horizon_highs = df["high"].iloc[i+1:i+1+self.horizon_periods]
            horizon_lows = df["low"].iloc[i+1:i+1+self.horizon_periods]
            
            max_up = (horizon_highs.max() - entry_close) / entry_close
            max_down = (entry_close - horizon_lows.min()) / entry_close
            
            mfe.iloc[i] = max(max_up, max_down)
        
        return mfe
    
    def compute_max_adverse_excursion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Maximum Adverse Excursion (MAE) - the worst drawdown during horizon.
        
        Returns the maximum potential loss (as positive number) during the trade.
        """
        mae = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df) - self.horizon_periods):
            entry_close = df["close"].iloc[i]
            horizon_lows = df["low"].iloc[i+1:i+1+self.horizon_periods]
            horizon_highs = df["high"].iloc[i+1:i+1+self.horizon_periods]
            
            max_drawdown_long = (entry_close - horizon_lows.min()) / entry_close
            max_drawdown_short = (horizon_highs.max() - entry_close) / entry_close
            
            mae.iloc[i] = min(max_drawdown_long, max_drawdown_short)
        
        return mae
    
    def compute_return_quantiles(self, prices: pd.Series,
                                  quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
                                  ) -> pd.DataFrame:
        """
        Compute rolling quantiles of forward returns.
        
        Uses expanding window to compute quantiles at each point.
        Now includes q10, q25, q50, q75, q90 for full distribution.
        """
        forward_returns = self.compute_forward_returns(prices)
        
        result = pd.DataFrame(index=prices.index)
        
        for q in quantiles:
            col_name = f"return_p{int(q*100)}"
            result[col_name] = forward_returns.expanding(min_periods=100).quantile(q)
        
        return result
    
    def _triple_barrier_labels(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5,
        max_holding: int = None
    ) -> pd.Series:
        """
        Triple Barrier Method for label generation.
        
        For each bar, simulate a trade and see which barrier gets hit first:
        - Upper barrier (TP): price rises by tp_atr_mult * ATR -> LONG (2)
        - Lower barrier (SL): price drops by sl_atr_mult * ATR -> SHORT (0)  
        - Time barrier: neither hit within max_holding bars -> HOLD (1)
        
        This produces clean, outcome-based labels because they reflect
        what actually happened in the price path, not just the endpoint.
        
        Args:
            df: OHLCV DataFrame
            atr: Pre-computed ATR series
            tp_atr_mult: ATR multiplier for take profit barrier
            sl_atr_mult: ATR multiplier for stop loss barrier
            max_holding: Maximum bars to hold (defaults to horizon_periods)
        
        Returns:
            Series with labels: 0=SHORT, 1=HOLD, 2=LONG
        """
        if max_holding is None:
            max_holding = self.horizon_periods
            
        prices = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atr_vals = atr.values
        n = len(prices)
        
        labels = np.ones(n, dtype=np.int64)  # Default HOLD
        
        for i in range(n - max_holding):
            entry = prices[i]
            a = atr_vals[i]
            
            if np.isnan(a) or a <= 0:
                a = entry * 0.005
            
            upper = entry + tp_atr_mult * a
            lower = entry - sl_atr_mult * a
            
            for j in range(1, max_holding + 1):
                idx = i + j
                if idx >= n:
                    break
                    
                hit_upper = highs[idx] >= upper
                hit_lower = lows[idx] <= lower
                
                if hit_upper and hit_lower:
                    if (highs[idx] - entry) / a > (entry - lows[idx]) / a:
                        labels[i] = 2  # LONG won
                    else:
                        labels[i] = 0  # SHORT won
                    break
                elif hit_upper:
                    labels[i] = 2  # LONG
                    break
                elif hit_lower:
                    labels[i] = 0  # SHORT
                    break
        
        return pd.Series(labels, index=df.index)

    def label_enter_quality(
        self,
        df: pd.DataFrame,
        htf_features: pd.DataFrame,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5,
        horizon_bars: int = 16,
        slope_eps: float = 0.05,
        r_min_expiry: float = 1.0,
    ) -> pd.DataFrame:
        """
        HTF-gated Triple Barrier labeling for ENTER quality model.
        
        Uses training.triple_barrier as single source of truth for barrier simulation.
        
        Process:
        1. Check HTF trend alignment (1H and 4H agree on direction)
        2. Gate candidates by slope strength and range position
        3. For candidates: run Triple Barrier with direction from HTF
        4. TP hit first => ENTER=1
        5. Expiry with realized R >= r_min_expiry => ENTER=1 (good but messy)
        6. SL hit or weak expiry => ENTER=0
        
        Args:
            df: OHLCV DataFrame with close/high/low columns
            htf_features: DataFrame with HTF feature columns (h1_trend_sign, h4_trend_sign, etc.)
            tp_atr_mult: ATR multiplier for take profit barrier
            sl_atr_mult: ATR multiplier for stop loss barrier  
            horizon_bars: Maximum bars to hold before time expiry
            slope_eps: Minimum abs(h1_sma20_slope) to consider trend strong enough
            r_min_expiry: Minimum realized R-multiple at expiry to count as ENTER=1 (default 0.5)
            
        Returns:
            DataFrame with columns:
            - enter_label: 0 or 1
            - side_hint: +1 (LONG), -1 (SHORT), or 0 (no candidate)
            - outcome: "TP", "SL", "EXP_WIN", "EXP_LOSS", "NO_CANDIDATE"
            - realized_r: realized R-multiple at exit (NaN for non-candidates)
            - tp_price: take profit price level
            - sl_price: stop loss price level
        """
        from training.triple_barrier import (
            compute_atr_14, triple_barrier_outcome_for_index,
            compute_mfe_mae_for_index, compute_trade_cost_r, compute_soft_quality,
        )
        
        n = len(df)
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        atr_vals = compute_atr_14(df)
        
        h1_trend = htf_features['h1_trend_sign'].values if 'h1_trend_sign' in htf_features.columns else np.zeros(n)
        h4_trend = htf_features['h4_trend_sign'].values if 'h4_trend_sign' in htf_features.columns else np.zeros(n)
        h1_slope = htf_features['h1_sma20_slope'].values if 'h1_sma20_slope' in htf_features.columns else np.zeros(n)
        h1_range_pos = htf_features['h1_range_pos'].values if 'h1_range_pos' in htf_features.columns else np.full(n, 0.5)
        
        enter_labels = np.zeros(n, dtype=np.int64)
        side_hints = np.zeros(n, dtype=np.int64)
        outcomes = np.full(n, "NO_CANDIDATE", dtype=object)
        realized_r = np.full(n, np.nan)
        tp_prices = np.full(n, np.nan)
        sl_prices = np.full(n, np.nan)
        mfe_r_arr = np.full(n, np.nan)
        mae_r_arr = np.full(n, np.nan)
        y_soft_arr = np.full(n, np.nan)
        
        soft_label_temp = getattr(self, '_soft_label_temp', 2.0)
        
        n_candidates = 0
        n_tp = 0
        n_sl = 0
        n_exp_win = 0
        n_exp_loss = 0
        
        for i in range(n - horizon_bars):
            if h1_trend[i] == 0 or h4_trend[i] == 0:
                continue
            if h1_trend[i] != h4_trend[i]:
                continue
            
            if abs(h1_slope[i]) < slope_eps:
                continue
            
            if h1_trend[i] > 0 and h1_range_pos[i] < 0.2:
                continue
            if h1_trend[i] < 0 and h1_range_pos[i] > 0.8:
                continue
            
            n_candidates += 1
            side = int(h1_trend[i])
            side_hints[i] = side
            
            a = float(atr_vals[i])
            if np.isnan(a) or a <= 0:
                a = prices[i] * 0.005
            
            if side > 0:
                tp_prices[i] = prices[i] + tp_atr_mult * a
                sl_prices[i] = prices[i] - sl_atr_mult * a
            else:
                tp_prices[i] = prices[i] - tp_atr_mult * a
                sl_prices[i] = prices[i] + sl_atr_mult * a
            
            outcome, r = triple_barrier_outcome_for_index(
                highs, lows, prices, i, side, a,
                tp_atr_mult, sl_atr_mult, horizon_bars, r_min_expiry,
            )
            
            outcomes[i] = outcome
            realized_r[i] = r
            
            mfe, mae = compute_mfe_mae_for_index(
                highs, lows, prices, i, side, a,
                sl_atr_mult, horizon_bars,
            )
            mfe_r_arr[i] = mfe
            mae_r_arr[i] = mae
            
            cost_r_val = compute_trade_cost_r(prices[i], a, sl_atr_mult)
            y_soft_arr[i] = compute_soft_quality(mfe, mae, cost_r_val, soft_label_temp)
            
            if outcome == "TP":
                enter_labels[i] = 1
                n_tp += 1
            elif outcome == "SL":
                enter_labels[i] = 0
                n_sl += 1
            elif outcome == "EXP_WIN":
                enter_labels[i] = 1
                n_exp_win += 1
            elif outcome == "EXP_LOSS":
                enter_labels[i] = 0
                n_exp_loss += 1
        
        logger.info("=" * 70)
        logger.info("ENTER QUALITY LABELING: HTF-Gated Triple Barrier + R_min Expiry")
        logger.info("=" * 70)
        logger.info(f"Config: horizon={horizon_bars}, TP={tp_atr_mult}x ATR, SL={sl_atr_mult}x ATR, slope_eps={slope_eps}, r_min_expiry={r_min_expiry}")
        logger.info(f"Total bars: {n:,}")
        logger.info(f"Candidates (trend-aligned): {n_candidates:,} ({100*n_candidates/max(n,1):.1f}%)")
        logger.info(f"  TP hits    (ENTER=1): {n_tp:,} ({100*n_tp/max(n_candidates,1):.1f}% of candidates)")
        logger.info(f"  Expiry win (ENTER=1): {n_exp_win:,} ({100*n_exp_win/max(n_candidates,1):.1f}% of candidates, R>={r_min_expiry})")
        logger.info(f"  SL hits    (ENTER=0): {n_sl:,} ({100*n_sl/max(n_candidates,1):.1f}% of candidates)")
        logger.info(f"  Expiry loss(ENTER=0): {n_exp_loss:,} ({100*n_exp_loss/max(n_candidates,1):.1f}% of candidates)")
        logger.info(f"Non-candidates (ENTER=0): {n - n_candidates:,}")
        logger.info(f"ENTER=1 total: {enter_labels.sum():,} ({100*enter_labels.sum()/max(n,1):.1f}% of all bars)")
        
        valid_r = realized_r[~np.isnan(realized_r)]
        if len(valid_r) > 0:
            logger.info(f"Realized R distribution: mean={np.mean(valid_r):.2f}, median={np.median(valid_r):.2f}, "
                       f"win_r={np.mean(valid_r[valid_r>0]):.2f}, loss_r={np.mean(valid_r[valid_r<0]):.2f}")
        logger.info("=" * 70)
        
        self._debug_htf_timestamps(df, htf_features, n_samples=10)
        
        valid_soft = y_soft_arr[~np.isnan(y_soft_arr)]
        if len(valid_soft) > 0:
            logger.info(f"Soft labels: mean={np.mean(valid_soft):.3f}, median={np.median(valid_soft):.3f}, "
                       f"std={np.std(valid_soft):.3f}, min={np.min(valid_soft):.3f}, max={np.max(valid_soft):.3f}")
            valid_mfe = mfe_r_arr[~np.isnan(mfe_r_arr)]
            valid_mae = mae_r_arr[~np.isnan(mae_r_arr)]
            logger.info(f"MFE_R: mean={np.mean(valid_mfe):.2f}, MAE_R: mean={np.mean(valid_mae):.2f}")
        
        result = pd.DataFrame({
            'enter_label': enter_labels,
            'side_hint': side_hints,
            'outcome': outcomes,
            'realized_r': realized_r,
            'tp_price': tp_prices,
            'sl_price': sl_prices,
            'mfe_r': mfe_r_arr,
            'mae_r': mae_r_arr,
            'y_soft': y_soft_arr,
        }, index=df.index)
        
        return result
    
    def _debug_htf_timestamps(self, df: pd.DataFrame, htf_features: pd.DataFrame, n_samples: int = 10):
        """Debug print: verify HTF features at candidate bars use only past data.
        
        For n_samples random candidate rows, print the 15m timestamp and the HTF
        feature values to confirm they come from completed bars before time t.
        """
        import random
        
        candidates = []
        h1_trend = htf_features.get('h1_trend_sign')
        h4_trend = htf_features.get('h4_trend_sign')
        if h1_trend is None or h4_trend is None:
            return
        
        for i in range(len(df)):
            if h1_trend.iloc[i] != 0 and h4_trend.iloc[i] != 0 and h1_trend.iloc[i] == h4_trend.iloc[i]:
                candidates.append(i)
        
        if len(candidates) < n_samples:
            return
        
        sample_indices = sorted(random.sample(candidates, min(n_samples, len(candidates))))
        
        logger.info("-" * 70)
        logger.info("HTF TIMESTAMP AUDIT (%d random candidates)", len(sample_indices))
        logger.info("%-22s | %6s %6s | %8s %8s | %s", "15m_time", "h1_trn", "h4_trn", "h1_slope", "h1_rpos", "status")
        logger.info("-" * 70)
        
        has_ts = hasattr(df.index, 'tz') or df.index.dtype.kind == 'M'
        
        for idx in sample_indices:
            t = df.index[idx] if has_ts else f"row_{idx}"
            h1t = float(h1_trend.iloc[idx])
            h4t = float(h4_trend.iloc[idx])
            h1s = float(htf_features['h1_sma20_slope'].iloc[idx]) if 'h1_sma20_slope' in htf_features.columns else 0
            h1r = float(htf_features['h1_range_pos'].iloc[idx]) if 'h1_range_pos' in htf_features.columns else 0.5
            
            status = "OK (aligned)" if h1t == h4t else "MISMATCH"
            logger.info("%-22s | %+5.0f  %+5.0f  | %+7.3f  %7.3f  | %s",
                       str(t)[:22], h1t, h4t, h1s, h1r, status)
        
        logger.info("-" * 70)

    def generate_multihead_targets(
        self, 
        df: pd.DataFrame, 
        n_future_candles: int = 5,
        min_net_edge: float = 0.0,
        min_confidence: float = 0.40,
        use_volatility_cost: bool = False,
        fixed_cost: float = 0.0009,
        use_pure_directional: bool = False,
        directional_threshold: float = 0.0020,
        use_regime_labels: bool = False,
        trend_threshold: float = 0.0015,
        range_threshold: float = 0.0030,
        use_triple_barrier: bool = False,
        tb_tp_mult: float = 2.0,
        tb_sl_mult: float = 1.5
    ) -> pd.DataFrame:
        """
        Generate targets specifically for multi-head model training.
        
        Label Modes (in priority order):
        - Stage 4: use_triple_barrier=True - Triple Barrier Method (ATR-scaled, outcome-based)
        - Stage 3: use_regime_labels=True - ADX-based adaptive thresholds
        - Stage 2: use_pure_directional=True - simple return threshold
        - Stage 1: Cost-aware mode (default) - net_edge & confidence gates
        
        Args:
            df: DataFrame with OHLCV data
            n_future_candles: Number of future candles to predict
            min_net_edge: Minimum net edge after costs for trade signals
            min_confidence: Minimum mu/sigma ratio for trade signals
            use_volatility_cost: If True, use volatility-based cost calculation
            fixed_cost: Fixed round-trip trading cost
            use_pure_directional: If True, use simple return threshold
            directional_threshold: Return threshold for pure directional mode
            use_regime_labels: If True, use ADX-based adaptive thresholds
            trend_threshold: Return threshold for trending regime
            range_threshold: Return threshold for ranging regime
            use_triple_barrier: If True, use Triple Barrier Method (recommended)
            tb_tp_mult: ATR multiplier for TP barrier (default 2.0)
            tb_sl_mult: ATR multiplier for SL barrier (default 1.5)
        
        Returns:
            DataFrame with multihead targets
        """
        prices = df["close"]
        highs = df["high"]
        lows = df["low"]
        
        mu = self.compute_forward_returns(prices)
        
        current_vol = self.compute_realized_volatility(prices)
        sigma = self.compute_forward_volatility(prices)
        sigma = sigma.fillna(current_vol)
        
        hold_hours = self.horizon_periods * 0.25
        trading_costs = pd.Series(index=df.index, dtype=float)
        
        if use_volatility_cost:
            for i in range(len(df)):
                vol = current_vol.iloc[i] if not pd.isna(current_vol.iloc[i]) else 0.01
                trading_costs.iloc[i] = self.costs.total_round_trip_cost(
                    volatility=vol, 
                    is_taker=True, 
                    hold_hours=hold_hours
                )
        else:
            trading_costs[:] = fixed_cost
        
        net_edge = mu.abs() - trading_costs
        
        sigma_safe = sigma.clip(lower=0.001)
        confidence_ratio = mu.abs() / sigma_safe
        
        valid_mu = mu.dropna()
        total_samples = len(valid_mu)
        
        atr = self._compute_atr(df, period=14)
        
        logger.info("=" * 70)
        if use_triple_barrier:
            logger.info("LABEL GENERATION: STAGE 4 - TRIPLE BARRIER METHOD")
            logger.info("=" * 70)
            logger.info(f"Config: horizon={self.horizon_periods} bars, TP={tb_tp_mult}x ATR, SL={tb_sl_mult}x ATR")
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"ATR stats: mean={atr.mean():.2f}, median={atr.median():.2f}")
            
            class_label = self._triple_barrier_labels(
                df, atr, tp_atr_mult=tb_tp_mult, sl_atr_mult=tb_sl_mult,
                max_holding=self.horizon_periods
            )
            
            long_mask = class_label == 2
            short_mask = class_label == 0
            
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            n_hold = (class_label == 1).sum()
            logger.info(f"Barrier hits: TP(LONG)={n_long:,}, SL(SHORT)={n_short:,}, TIME(HOLD)={n_hold:,}")
            logger.info("-" * 70)
            logger.info("NOTE: Labels reflect actual price path outcomes, not just endpoint returns")
            logger.info("NOTE: ATR-scaled barriers adapt to current volatility regime")
            logger.info("-" * 70)
            
        elif use_regime_labels:
            logger.info("LABEL GENERATION: STAGE 3 - REGIME-BASED MODE")
            logger.info("=" * 70)
            logger.info(f"Config: horizon={self.horizon_periods} bars")
            logger.info(f"Thresholds: trend={trend_threshold:.4%}, range={range_threshold:.4%}")
            logger.info(f"Total samples: {total_samples:,}")
            
            adx = self._compute_adx(df, period=14)
            
            is_trending = adx > 25
            is_ranging = adx < 20
            is_transition = ~is_trending & ~is_ranging
            
            pct_trend = is_trending.sum() / max(1, len(adx)) * 100
            pct_range = is_ranging.sum() / max(1, len(adx)) * 100
            pct_trans = is_transition.sum() / max(1, len(adx)) * 100
            logger.info(f"Regime distribution: TREND={pct_trend:.1f}%, RANGE={pct_range:.1f}%, TRANSITION={pct_trans:.1f}%")
            
            class_label = pd.Series(1, index=df.index)
            
            trend_long = is_trending & (mu > trend_threshold)
            trend_short = is_trending & (mu < -trend_threshold)
            
            range_long = is_ranging & (mu > range_threshold)
            range_short = is_ranging & (mu < -range_threshold)
            
            trans_threshold = (trend_threshold + range_threshold) / 2
            trans_long = is_transition & (mu > trans_threshold)
            trans_short = is_transition & (mu < -trans_threshold)
            
            long_mask = trend_long | range_long | trans_long
            short_mask = trend_short | range_short | trans_short
            
            class_label[long_mask] = 2
            class_label[short_mask] = 0
            
            n_trend_trades = (trend_long.sum() + trend_short.sum())
            n_range_trades = (range_long.sum() + range_short.sum())
            n_trans_trades = (trans_long.sum() + trans_short.sum())
            logger.info(f"Trades by regime: TREND={n_trend_trades:,}, RANGE={n_range_trades:,}, TRANSITION={n_trans_trades:,}")
            logger.info("-" * 70)
            
        elif use_pure_directional:
            logger.info("LABEL GENERATION: STAGE 2 - PURE DIRECTIONAL MODE")
            logger.info("=" * 70)
            logger.info(f"Config: horizon={self.horizon_periods} bars, directional_threshold={directional_threshold:.4%}")
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"Stats: mean(|mu|)={valid_mu.abs().mean():.4%}, mean(sigma)={sigma_safe.dropna().mean():.4%}")
            
            class_label = pd.Series(1, index=df.index)
            
            long_mask = mu > directional_threshold
            short_mask = mu < -directional_threshold
            
            class_label[long_mask] = 2
            class_label[short_mask] = 0
            
            pct_above_threshold = (mu.abs() > directional_threshold).sum() / max(1, total_samples) * 100
            logger.info(f"Trade density: {pct_above_threshold:.1f}% samples exceed threshold")
            logger.info("-" * 70)
        else:
            logger.info("LABEL GENERATION: STAGE 1 - COST-AWARE MODE")
            logger.info("=" * 70)
            
            valid_net_edge = net_edge.dropna()
            valid_conf = confidence_ratio.dropna()
            
            pct_positive_edge = (valid_net_edge > min_net_edge).sum() / max(1, total_samples) * 100
            pct_high_conf = (valid_conf > min_confidence).sum() / max(1, total_samples) * 100
            pct_both_gates = ((valid_net_edge > min_net_edge) & (valid_conf > min_confidence)).sum() / max(1, total_samples) * 100
            
            logger.info(f"Config: horizon={self.horizon_periods} bars, cost_mode={'volatility' if use_volatility_cost else 'fixed'}, "
                       f"avg_cost={trading_costs.mean():.4%}")
            logger.info(f"Thresholds: min_net_edge={min_net_edge:.4%}, min_confidence={min_confidence:.2f}")
            logger.info(f"Total samples: {total_samples:,}")
            
            if pct_both_gates < 5.0:
                logger.warning(f"!!! LOW TRADE DENSITY: Only {pct_both_gates:.2f}% samples pass both gates !!!")
                logger.warning(f"!!! Try: use_triple_barrier=True !!!")
            
            class_label = pd.Series(1, index=df.index)
            
            long_mask = (mu > 0) & (net_edge > min_net_edge) & (confidence_ratio > min_confidence)
            class_label[long_mask] = 2
            
            short_mask = (mu < 0) & (net_edge > min_net_edge) & (confidence_ratio > min_confidence)
            class_label[short_mask] = 0
        
        # Log class distribution with target range validation
        n_long = (class_label == 2).sum()
        n_short = (class_label == 0).sum()
        n_hold = (class_label == 1).sum()
        total = n_long + n_short + n_hold
        
        pct_long = n_long/max(1,total)*100
        pct_short = n_short/max(1,total)*100
        pct_hold = n_hold/max(1,total)*100
        
        logger.info(f"CLASS DISTRIBUTION RESULT:")
        logger.info(f"  SHORT: {n_short:,} ({pct_short:.1f}%) [target: 15-25%]")
        logger.info(f"  HOLD:  {n_hold:,} ({pct_hold:.1f}%) [target: 50-70%]")
        logger.info(f"  LONG:  {n_long:,} ({pct_long:.1f}%) [target: 15-25%]")
        
        # Validate against targets
        hold_ok = 50 <= pct_hold <= 70
        long_ok = 15 <= pct_long <= 25
        short_ok = 15 <= pct_short <= 25
        
        if hold_ok and long_ok and short_ok:
            logger.info("  STATUS: All targets MET!")
        else:
            issues = []
            if not hold_ok:
                issues.append(f"HOLD ({pct_hold:.1f}%)")
            if not long_ok:
                issues.append(f"LONG ({pct_long:.1f}%)")
            if not short_ok:
                issues.append(f"SHORT ({pct_short:.1f}%)")
            logger.warning(f"  STATUS: Targets NOT MET: {', '.join(issues)}")
            if pct_hold > 80 and not use_pure_directional:
                logger.warning("  RECOMMENDATION: Try use_pure_directional=True")
        logger.info("=" * 70)
        
        # Legacy edge for backward compatibility
        edge = self.compute_directional_edge(mu, sigma_safe)
        
        # === TRADING HEAD TARGETS ===
        # Entry offset: MFE-based optimal limit entry (learned from price path)
        atr = self._compute_atr(df, period=14)
        entry_offset, sl_distance, tp_distance = self._compute_mfe_trading_targets(
            df, prices, class_label, atr, n_future_candles
        )
        
        # ============================================================
        # PHASE 2: CONSTRAINED CANDLE PARAMETERIZATION
        # ============================================================
        # Instead of predicting raw close/high/low which can be inconsistent,
        # we predict: delta_close, log_range (always positive), skew in [-1, 1]
        # Then reconstruct: range = exp(log_range)
        #                   high = close + range * (0.5 + 0.5 * skew)
        #                   low = close - range * (0.5 - 0.5 * skew)
        # This guarantees: high >= low and valid candles
        
        candle_targets = {}
        for i in range(1, n_future_candles + 1):
            # Delta close: (future_close - current_close) / current_close
            future_close = prices.shift(-i)
            future_high = highs.shift(-i)
            future_low = lows.shift(-i)
            
            delta_close = (future_close - prices) / prices
            candle_targets[f"candle_delta_close_{i}"] = delta_close
            
            # Constrained parameterization:
            # Range = (high - low) / close (always positive)
            candle_range = (future_high - future_low) / prices
            candle_range = candle_range.clip(lower=0.0001)  # Prevent zero/negative
            
            # Log range for numerical stability (network predicts unbounded, we exp it)
            log_range = np.log(candle_range)
            candle_targets[f"candle_log_range_{i}"] = log_range
            
            # Skew in [-1, 1]: where is close relative to high/low
            # skew = (close - midpoint) / (range/2)
            # = 2 * (close - (high + low)/2) / (high - low)
            # = (close - low) / (high - low) * 2 - 1  (when close is at high, skew = 1)
            # Use future close position within high-low range
            range_safe = (future_high - future_low).clip(lower=prices * 0.0001)
            skew = 2 * (future_close - future_low) / range_safe - 1
            skew = skew.clip(lower=-1, upper=1)
            candle_targets[f"candle_skew_{i}"] = skew
            
            # Keep legacy targets for backward compatibility (but use constrained for new training)
            candle_targets[f"candle_delta_high_{i}"] = (future_high - prices) / prices
            candle_targets[f"candle_delta_low_{i}"] = (future_low - prices) / prices
        
        # ============================================================
        # PHASE 1b: LOG_SIGMA FOR GAUSSIAN NLL
        # ============================================================
        # For proper Gaussian NLL loss: loss = (y-mu)^2/(2*sigma^2) + log(sigma)
        # We predict log_sigma (unbounded) and exp it to get sigma (always positive)
        log_sigma = np.log(sigma_safe)
        
        # ============================================================
        # FLOW FORECAST: VOL_STATE AND ACCELERATION TARGETS
        # ============================================================
        # Vol_state: Volatility regime prediction (expansion/neutral/contraction)
        # forward_vol / current_vol ratio determines regime:
        #   < 0.9 = contraction (0)
        #   0.9 - 1.1 = neutral (1)
        #   > 1.1 = expansion (2)
        
        # Forward volatility over horizon
        forward_vol = self.compute_forward_volatility(prices)
        forward_vol_safe = forward_vol.fillna(current_vol)
        current_vol_safe = current_vol.fillna(0.01)
        
        vol_ratio = forward_vol_safe / current_vol_safe.clip(lower=0.001)
        vol_state = pd.Series(1, index=df.index, dtype=int)  # Default neutral
        vol_state[vol_ratio < 0.9] = 0   # Contraction
        vol_state[vol_ratio > 1.1] = 2   # Expansion
        
        # Log vol_state distribution
        n_contraction = (vol_state == 0).sum()
        n_neutral = (vol_state == 1).sum()
        n_expansion = (vol_state == 2).sum()
        total = max(1, n_contraction + n_neutral + n_expansion)
        logger.info(f"Vol_state distribution: CONTRACTION={n_contraction} ({n_contraction/total*100:.1f}%), "
                   f"NEUTRAL={n_neutral} ({n_neutral/total*100:.1f}%), "
                   f"EXPANSION={n_expansion} ({n_expansion/total*100:.1f}%)")
        
        # Acceleration: momentum change = momentum_forward - momentum_now
        # Momentum = 4-bar return (for 15m, this is 1 hour momentum)
        momentum_window = min(4, self.horizon_periods // 4) if self.horizon_periods >= 4 else 1
        momentum_now = (prices / prices.shift(momentum_window) - 1).fillna(0)
        
        # Forward momentum: return from horizon-4 to horizon
        forward_prices = prices.shift(-self.horizon_periods)
        forward_prices_back = prices.shift(-(self.horizon_periods - momentum_window))
        momentum_forward = (forward_prices / forward_prices_back - 1).fillna(0)
        
        # Acceleration = momentum change over horizon
        acceleration = momentum_forward - momentum_now
        
        logger.info(f"Acceleration stats: mean={acceleration.mean():.6f}, std={acceleration.std():.6f}, "
                   f"min={acceleration.min():.6f}, max={acceleration.max():.6f}")
        
        targets = pd.DataFrame({
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,  # For Gaussian NLL loss
            "class_label": class_label,
            "forward_return": mu,  # Same as mu, explicit for quantile loss
            "edge": edge,
            "net_edge": net_edge,  # Cost-aware edge
            "trading_cost": trading_costs,  # For debugging
            "confidence_ratio": confidence_ratio,  # mu/sigma
            "current_volatility": current_vol,
            "entry_offset": entry_offset,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "vol_state": vol_state,  # Flow forecast: 0=contraction, 1=neutral, 2=expansion
            "acceleration": acceleration,  # Flow forecast: momentum change over horizon
            "vol_ratio": vol_ratio,  # For debugging
            **candle_targets
        })
        
        # Log cost-aware labeling stats
        avg_cost = trading_costs.mean()
        logger.info(f"Generated multihead targets (cost-aware): "
                   f"LONG={long_mask.sum()}, SHORT={short_mask.sum()}, "
                   f"HOLD={(class_label == 1).sum()}, "
                   f"avg_trading_cost={avg_cost:.4%}, "
                   f"candle_steps={n_future_candles}")
        
        return targets
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range (ATR)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range)
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average Directional Index (ADX) for regime detection.
        
        ADX values interpretation:
        - ADX > 25: Strong trend (trending regime)
        - ADX < 20: Weak trend (ranging regime)
        - 20 <= ADX <= 25: Transition zone
        
        Returns:
            pd.Series: ADX values (0-100 scale)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # +DM and -DM
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Smoothed values (Wilder's smoothing)
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10))
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(20)  # Default to transition zone
    
    def _compute_mfe_trading_targets(
        self, 
        df: pd.DataFrame, 
        prices: pd.Series,
        class_label: pd.Series,
        atr: pd.Series,
        n_future_candles: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MFE-based (Maximum Favorable Excursion) trading targets.
        
        MFE Logic:
        - For LONG: entry_offset = how far below current price did price dip 
          before moving up (optimal limit buy placement)
        - For SHORT: entry_offset = how far above current price did price spike
          before moving down (optimal limit sell placement)
        - For HOLD: entry_offset = 0 (no trade)
        
        SL/TP are derived from MAE (Maximum Adverse Excursion) and MFE:
        - SL = MAE (maximum adverse move during the trade)
        - TP = MFE (maximum favorable move during the trade)
        
        All values are normalized by ATR for cross-volatility learning.
        
        Returns:
            entry_offset: Optimal entry as percentage of price (negative = buy lower)
            sl_distance: Stop loss distance as percentage
            tp_distance: Take profit distance as percentage
        """
        highs = df["high"]
        lows = df["low"]
        
        # Initialize with zeros
        entry_offset = pd.Series(0.0, index=df.index)
        sl_distance = pd.Series(0.0, index=df.index)
        tp_distance = pd.Series(0.0, index=df.index)
        
        # Horizon for MFE/MAE calculation (same as forward return horizon)
        horizon = self.horizon_periods
        
        for i in range(len(df) - horizon):
            current_price = prices.iloc[i]
            label = class_label.iloc[i]
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else current_price * 0.01
            
            # Get future price path
            future_highs = highs.iloc[i+1:i+horizon+1]
            future_lows = lows.iloc[i+1:i+horizon+1]
            future_closes = prices.iloc[i+1:i+horizon+1]
            
            if len(future_highs) == 0:
                continue
            
            # MFE/MAE from the current entry point
            max_high = future_highs.max()
            min_low = future_lows.min()
            
            # LONG trade analysis
            if label == 2:  # LONG
                # Best entry = lowest price in first few candles (optimal limit buy)
                # We look at the first 1/4 of horizon for entry opportunity
                entry_window = max(1, horizon // 4)
                best_entry_price = future_lows.iloc[:entry_window].min()
                
                # Entry offset: how much lower than current could we have bought?
                # Negative means better (lower) entry
                entry_off = (best_entry_price - current_price) / current_price
                entry_offset.iloc[i] = np.clip(entry_off, -0.05, 0.0)  # Max 5% better entry
                
                # MAE (Maximum Adverse Excursion) = worst drawdown during trade
                mae = (min_low - current_price) / current_price
                sl_distance.iloc[i] = abs(mae) + (current_atr / current_price) * 0.5
                
                # MFE (Maximum Favorable Excursion) = best profit potential
                mfe = (max_high - current_price) / current_price
                tp_distance.iloc[i] = max(mfe, current_atr / current_price * 2)
                
            # SHORT trade analysis
            elif label == 0:  # SHORT
                # Best entry = highest price in first few candles (optimal limit sell)
                entry_window = max(1, horizon // 4)
                best_entry_price = future_highs.iloc[:entry_window].max()
                
                # Entry offset: how much higher than current could we have sold?
                # Positive means better (higher) entry for short
                entry_off = (best_entry_price - current_price) / current_price
                entry_offset.iloc[i] = np.clip(entry_off, 0.0, 0.05)  # Max 5% better entry
                
                # MAE for short = worst spike up during trade
                mae = (max_high - current_price) / current_price
                sl_distance.iloc[i] = abs(mae) + (current_atr / current_price) * 0.5
                
                # MFE for short = best drop potential
                mfe = (current_price - min_low) / current_price
                tp_distance.iloc[i] = max(mfe, current_atr / current_price * 2)
                
            else:  # HOLD
                # For HOLD, use ATR-based defaults (fallback)
                entry_offset.iloc[i] = 0.0
                sl_distance.iloc[i] = (current_atr / current_price) * 1.5
                tp_distance.iloc[i] = (current_atr / current_price) * 3.0
        
        # Clip to reasonable ranges
        sl_distance = sl_distance.clip(lower=0.003, upper=0.05)  # 0.3% to 5%
        tp_distance = tp_distance.clip(lower=0.005, upper=0.10)  # 0.5% to 10%
        
        # Log statistics
        long_mask = class_label == 2
        short_mask = class_label == 0
        logger.info(f"MFE Trading Targets - "
                   f"LONG entry_offset mean: {entry_offset[long_mask].mean():.4f}, "
                   f"SHORT entry_offset mean: {entry_offset[short_mask].mean():.4f}, "
                   f"SL mean: {sl_distance.mean():.4f}, TP mean: {tp_distance.mean():.4f}")
        
        return entry_offset, sl_distance, tp_distance
    
    def compute_probability_profitable(self, 
                                        prices: pd.Series,
                                        volatility: pd.Series) -> pd.Series:
        """
        Compute probability that absolute move exceeds transaction costs.
        
        P(|return| > cost) estimated from historical distribution.
        """
        forward_returns = self.compute_forward_returns(prices)
        
        costs = pd.Series(index=prices.index, dtype=float)
        for i in range(len(volatility)):
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.01
            costs.iloc[i] = self.costs.total_round_trip_cost(vol)
        
        profitable = (np.abs(forward_returns) > costs).astype(float)
        
        p_profitable = profitable.rolling(window=500, min_periods=50).mean()
        
        return p_profitable
    
    def compute_directional_edge(self, 
                                  mu: pd.Series, 
                                  volatility: pd.Series) -> pd.Series:
        """
        Compute directional edge = (|μ| - cost) / σ.
        
        Positive edge means expected profit after costs.
        """
        costs = pd.Series(index=mu.index, dtype=float)
        for i in range(len(volatility)):
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.01
            costs.iloc[i] = self.costs.total_round_trip_cost(vol)
        
        edge = (np.abs(mu) - costs) / volatility.clip(lower=0.001)
        
        return edge
    
    def generate_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all regression targets from candle DataFrame.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame with target columns:
            - mu: Expected forward return
            - sigma: Forward volatility (uncertainty)
            - p_profitable: Probability of profitable trade
            - return_p10, return_p50, return_p90: Return quantiles
            - mfe: Maximum favorable excursion
            - mae: Maximum adverse excursion
            - edge: (|mu| - cost) / sigma
            - optimal_direction: 1 for long, -1 for short, 0 for hold
        """
        prices = df["close"]
        
        logger.info(f"Generating targets: horizon={self.horizon_periods}, "
                   f"lookback={self.lookback_periods}")
        
        mu = self.compute_forward_returns(prices)
        logger.info(f"  μ (forward returns): {mu.notna().sum()} valid")
        
        current_vol = self.compute_realized_volatility(prices)
        sigma = self.compute_forward_volatility(prices)
        sigma = sigma.fillna(current_vol)  # Use current vol as fallback
        logger.info(f"  σ (forward volatility): {sigma.notna().sum()} valid")
        
        p_profitable = self.compute_probability_profitable(prices, current_vol)
        logger.info(f"  P(profitable): {p_profitable.notna().sum()} valid")
        
        quantiles = self.compute_return_quantiles(prices)
        logger.info(f"  Quantiles computed")
        
        mfe = self.compute_max_favorable_excursion(df)
        mae = self.compute_max_adverse_excursion(df)
        logger.info(f"  MFE/MAE computed")
        
        edge = self.compute_directional_edge(mu, sigma)
        logger.info(f"  Edge computed")
        
        optimal_direction = pd.Series(0, index=df.index)
        edge_threshold = 0.5  # Require edge > 0.5 sigma
        optimal_direction[mu > 0] = 1   # Long
        optimal_direction[mu < 0] = -1  # Short
        optimal_direction[np.abs(edge) < edge_threshold] = 0  # No trade
        
        targets = pd.DataFrame({
            "mu": mu,
            "sigma": sigma,
            "p_profitable": p_profitable,
            "return_p10": quantiles["return_p10"],
            "return_p50": quantiles["return_p50"],
            "return_p90": quantiles["return_p90"],
            "mfe": mfe,
            "mae": mae,
            "edge": edge,
            "optimal_direction": optimal_direction,
            "current_volatility": current_vol
        })
        
        valid_count = targets.dropna().shape[0]
        total_count = len(targets)
        logger.info(f"Generated {valid_count}/{total_count} valid target rows")
        
        return targets


class MultiHorizonTargetGenerator:
    """
    Generate targets for multiple time horizons.
    
    This allows the model to predict returns at different horizons:
    - 1h (12 periods of 5m)
    - 4h (48 periods of 5m)
    - 12h (144 periods of 5m)
    - 24h (288 periods of 5m)
    
    Useful for regime-adaptive trading and timeframe agreement analysis.
    """
    
    def __init__(self, 
                 horizons: Dict[str, int] = None,
                 costs: Optional[TradingCosts] = None):
        self.horizons = horizons or {
            "1h": 12,
            "4h": 48,
            "12h": 144,
            "24h": 288
        }
        self.costs = costs or TradingCosts()
        
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate targets for all horizons.
        
        Returns DataFrame with columns like:
        - mu_1h, sigma_1h, edge_1h
        - mu_4h, sigma_4h, edge_4h
        - ...
        """
        all_targets = pd.DataFrame(index=df.index)
        
        for horizon_name, horizon_periods in self.horizons.items():
            logger.info(f"Generating {horizon_name} targets...")
            
            generator = RegressionTargetGenerator(
                horizon_periods=horizon_periods,
                lookback_periods=horizon_periods * 2,
                costs=self.costs
            )
            
            targets = generator.generate_all_targets(df)
            
            for col in ["mu", "sigma", "edge", "p_profitable", "optimal_direction"]:
                all_targets[f"{col}_{horizon_name}"] = targets[col]
        
        horizon_agreement = pd.Series(0, index=df.index)
        for _, row in all_targets.iterrows():
            directions = [row.get(f"optimal_direction_{h}", 0) for h in self.horizons.keys()]
            if all(d > 0 for d in directions if d != 0):
                horizon_agreement[row.name] = 1
            elif all(d < 0 for d in directions if d != 0):
                horizon_agreement[row.name] = -1
        
        all_targets["horizon_agreement"] = horizon_agreement
        
        return all_targets


def create_regression_dataset(
    candle_df: pd.DataFrame,
    features_df: pd.DataFrame,
    horizon_periods: int = 16,  # Default 16 bars = 4h at 15m timeframe (was 48 for 5m)
    min_edge_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset for regression model.
    
    Args:
        candle_df: OHLCV candle data
        features_df: Computed feature matrix
        horizon_periods: Prediction horizon in candle periods
        min_edge_threshold: Minimum edge for trade labels
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_regression: Regression targets [mu, sigma] (n_samples, 2)
        y_direction: Direction labels [-1, 0, 1] (n_samples,)
    """
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    targets = generator.generate_all_targets(candle_df)
    
    common_idx = features_df.index.intersection(targets.index)
    features_df = features_df.loc[common_idx]
    targets = targets.loc[common_idx]
    
    mask = targets[["mu", "sigma"]].notna().all(axis=1)
    features_df = features_df[mask]
    targets = targets[mask]
    
    X = features_df.values.astype(np.float32)
    
    y_regression = targets[["mu", "sigma"]].values.astype(np.float32)
    
    y_direction = targets["optimal_direction"].values.astype(np.float32)
    
    logger.info(f"Created regression dataset: X={X.shape}, y_reg={y_regression.shape}, y_dir={y_direction.shape}")
    
    return X, y_regression, y_direction


def generate_multihead_targets(
    df: pd.DataFrame, 
    horizon_periods: int = 24,  # Default 24 bars = 6h at 15m timeframe
    n_future_candles: int = 5,
    min_net_edge: float = 0.0,
    min_confidence: float = 0.40,
    use_volatility_cost: bool = False,
    fixed_cost: float = 0.0009,
    use_pure_directional: bool = False,
    directional_threshold: float = 0.0020,
    use_regime_labels: bool = False,
    trend_threshold: float = 0.0015,
    range_threshold: float = 0.0030,
    use_triple_barrier: bool = False,
    tb_tp_mult: float = 2.0,
    tb_sl_mult: float = 1.5
) -> pd.DataFrame:
    """
    Standalone function to generate multi-head training targets.
    
    Label Modes (recommended order):
    - Stage 4: use_triple_barrier=True - Triple Barrier (ATR-scaled, outcome-based) [RECOMMENDED]
    - Stage 3: use_regime_labels=True - ADX-based adaptive thresholds
    - Stage 2: use_pure_directional=True - simple return threshold
    - Stage 1: Cost-aware mode (default) - net_edge & confidence gates
    
    Args:
        df: DataFrame with OHLCV data
        horizon_periods: Prediction horizon in candle periods (default 24 = 6h in 15m candles)
        n_future_candles: Number of future candles to predict
        use_triple_barrier: If True, use Triple Barrier Method (recommended)
        tb_tp_mult: ATR multiplier for TP barrier (default 2.0)
        tb_sl_mult: ATR multiplier for SL barrier (default 1.5)
        
    Returns:
        DataFrame with multihead targets
    """
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    return generator.generate_multihead_targets(
        df, 
        n_future_candles=n_future_candles,
        min_net_edge=min_net_edge,
        min_confidence=min_confidence,
        use_volatility_cost=use_volatility_cost,
        fixed_cost=fixed_cost,
        use_pure_directional=use_pure_directional,
        directional_threshold=directional_threshold,
        use_regime_labels=use_regime_labels,
        trend_threshold=trend_threshold,
        range_threshold=range_threshold,
        use_triple_barrier=use_triple_barrier,
        tb_tp_mult=tb_tp_mult,
        tb_sl_mult=tb_sl_mult
    )


def generate_enter_quality_targets(
    df: pd.DataFrame,
    htf_features: pd.DataFrame,
    horizon_periods: int = 16,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    slope_eps: float = 0.05,
    r_min_expiry: float = 1.0,
) -> pd.DataFrame:
    """Convenience function for ENTER quality labeling."""
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    return generator.label_enter_quality(
        df, htf_features,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        horizon_bars=horizon_periods,
        slope_eps=slope_eps,
        r_min_expiry=r_min_expiry,
    )


def generate_v46_quality_targets(
    df: pd.DataFrame,
    htf_features: pd.DataFrame,
    horizon_periods: int = 16,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    r_min_expiry: float = 1.0,
    soft_label_temp: float = 1.2,
) -> pd.DataFrame:
    """v4.6 Directional Separation labeling — bidirectional, no HTF gating.

    For EVERY bar, computes both LONG and SHORT triple-barrier outcomes
    and produces:
      - y_quality: 1 if max(long_r, short_r) indicates a good trade
      - y_dir: 1=LONG better, 0=SHORT better
      - y_dir_conf: sigmoid-mapped confidence from margin
      - y_htf_score: 0-3 HTF alignment class from past-only features
      - y_soft_quality: soft label from best-direction MFE/MAE
      - long_r, short_r, best_r: raw R outcomes
      - mfe_r, mae_r: from best direction
      - enter_label, side_hint, outcome, realized_r: backward-compat columns
    """
    from training.triple_barrier import (
        compute_atr_14, bidirectional_outcome_for_index,
        compute_htf_score_target, compute_mfe_mae_for_index,
        compute_trade_cost_r, compute_soft_quality,
    )

    n = len(df)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    atr_vals = compute_atr_14(df)

    h1_trend = htf_features['h1_trend_sign'].values if 'h1_trend_sign' in htf_features.columns else np.zeros(n)
    h4_trend = htf_features['h4_trend_sign'].values if 'h4_trend_sign' in htf_features.columns else np.zeros(n)

    y_quality_arr = np.zeros(n, dtype=np.int64)
    y_dir_arr = np.zeros(n, dtype=np.int64)
    y_dir_conf_arr = np.full(n, 0.5, dtype=np.float64)
    y_htf_score_arr = np.zeros(n, dtype=np.int64)
    long_r_arr = np.full(n, np.nan, dtype=np.float64)
    short_r_arr = np.full(n, np.nan, dtype=np.float64)
    best_r_arr = np.full(n, np.nan, dtype=np.float64)
    mfe_r_arr = np.full(n, np.nan, dtype=np.float64)
    mae_r_arr = np.full(n, np.nan, dtype=np.float64)
    y_soft_arr = np.full(n, 0.5, dtype=np.float64)

    enter_labels = np.zeros(n, dtype=np.int64)
    side_hints = np.zeros(n, dtype=np.int64)
    outcomes = np.full(n, "NO_CANDIDATE", dtype=object)
    realized_r = np.full(n, np.nan, dtype=np.float64)

    n_quality_1 = 0
    n_long_better = 0
    n_short_better = 0
    htf_class_counts = [0, 0, 0, 0]

    for i in range(n - horizon_periods):
        a = float(atr_vals[i])
        if np.isnan(a) or a <= 0:
            a = closes[i] * 0.005

        result = bidirectional_outcome_for_index(
            highs, lows, closes, i, a,
            tp_atr_mult, sl_atr_mult, horizon_periods, r_min_expiry,
        )

        y_quality_arr[i] = result['y_quality']
        y_dir_arr[i] = result['y_dir']
        y_dir_conf_arr[i] = result['y_dir_conf']
        long_r_arr[i] = result['long_r']
        short_r_arr[i] = result['short_r']
        best_r_arr[i] = result['best_outcome_r']

        htf_score = compute_htf_score_target(float(h1_trend[i]), float(h4_trend[i]))
        y_htf_score_arr[i] = htf_score
        htf_class_counts[htf_score] += 1

        best_side = +1 if result['y_dir'] == 1 else -1
        mfe, mae = compute_mfe_mae_for_index(
            highs, lows, closes, i, best_side, a,
            sl_atr_mult, horizon_periods,
        )
        mfe_r_arr[i] = mfe
        mae_r_arr[i] = mae

        cost_r_val = compute_trade_cost_r(closes[i], a, sl_atr_mult)
        y_soft_arr[i] = compute_soft_quality(mfe, mae, cost_r_val, soft_label_temp)

        enter_labels[i] = result['y_quality']
        side_hints[i] = best_side
        if result['y_quality'] == 1:
            outcomes[i] = result['long_outcome'] if best_side > 0 else result['short_outcome']
            realized_r[i] = result['best_outcome_r']
            n_quality_1 += 1
        else:
            outcomes[i] = "BOTH_LOSE"
            realized_r[i] = result['best_outcome_r']

        if result['y_dir'] == 1:
            n_long_better += 1
        else:
            n_short_better += 1

    labeled_bars = n - horizon_periods
    logger.info("=" * 70)
    logger.info("v4.6 BIDIRECTIONAL LABELING (no HTF gate)")
    logger.info("=" * 70)
    logger.info(f"Total bars: {n:,}, Labeled: {labeled_bars:,}")
    logger.info(f"y_quality=1: {n_quality_1:,} ({100*n_quality_1/max(labeled_bars,1):.1f}%)")
    logger.info(f"y_dir: LONG_better={n_long_better:,} SHORT_better={n_short_better:,}")
    logger.info(f"HTF score distribution: {dict(enumerate(htf_class_counts))}")

    valid_best = best_r_arr[~np.isnan(best_r_arr)]
    if len(valid_best) > 0:
        logger.info(f"best_R: mean={np.mean(valid_best):.3f} median={np.median(valid_best):.3f} "
                     f"p25={np.percentile(valid_best, 25):.3f} p75={np.percentile(valid_best, 75):.3f}")
    logger.info("=" * 70)

    return pd.DataFrame({
        'enter_label': enter_labels,
        'side_hint': side_hints,
        'outcome': outcomes,
        'realized_r': realized_r,
        'y_quality': y_quality_arr,
        'y_dir': y_dir_arr,
        'y_dir_conf': y_dir_conf_arr,
        'y_htf_score': y_htf_score_arr,
        'long_r': long_r_arr,
        'short_r': short_r_arr,
        'best_r': best_r_arr,
        'mfe_r': mfe_r_arr,
        'mae_r': mae_r_arr,
        'y_soft': y_soft_arr,
    }, index=df.index)


def _auto_calibrate_q_min_tp(
    tp_quality_all: np.ndarray,
    tp_first_all: np.ndarray,
    exp_win_all: np.ndarray,
    n_total: int,
    target_rate: float = 0.18,
    target_min: float = 0.12,
    target_max: float = 0.25,
    search_steps: int = 30,
    search_lo: float = 0.0,
    search_hi: float = 0.9,
) -> float:
    """v4.7.1: Search q_min_tp to get ENTER positive rate closest to target.

    ENTER=1 if (TP-first AND q >= q_min_tp) OR (EXP_WIN strong).
    q_min_tp only gates the TP-first path (the dominant population).
    """
    n_exp_win = int(exp_win_all.sum())

    tp_q_valid = tp_quality_all[tp_first_all & ~np.isnan(tp_quality_all)]
    n_tp = len(tp_q_valid)

    if n_total == 0:
        return 0.3

    logger.info(f"[LABEL_BALANCE] v4.7.1 q_min_tp search: n_total={n_total} "
                f"n_tp_first={n_tp} n_exp_win={n_exp_win} "
                f"search_range=[{search_lo:.2f}, {search_hi:.2f}]")

    if n_tp > 0:
        pcts = np.percentile(tp_q_valid, [10, 25, 50, 75, 90, 95, 99])
        logger.info(f"[TP_QUAL] q: min={tp_q_valid.min():.3f} p10={pcts[0]:.3f} p25={pcts[1]:.3f} "
                    f"p50={pcts[2]:.3f} p75={pcts[3]:.3f} p90={pcts[4]:.3f} "
                    f"p95={pcts[5]:.3f} p99={pcts[6]:.3f} max={tp_q_valid.max():.3f}")

    def rate_for_q(q_thresh):
        tp_kept = int((tp_q_valid >= q_thresh).sum()) if n_tp > 0 else 0
        return (tp_kept + n_exp_win) / n_total

    best_q = search_lo
    best_dist = abs(rate_for_q(best_q) - target_rate)

    for step in range(search_steps):
        q = search_lo + (search_hi - search_lo) * step / max(search_steps - 1, 1)
        r = rate_for_q(q)
        d = abs(r - target_rate)
        if d < best_dist:
            best_dist = d
            best_q = q

    final_rate = rate_for_q(best_q)
    if final_rate < target_min and best_q > search_lo:
        for q in np.linspace(search_lo, best_q, 20):
            r = rate_for_q(q)
            if target_min <= r <= target_max:
                best_q = q
                break
    elif final_rate > target_max and best_q < search_hi:
        for q in np.linspace(best_q, search_hi, 20):
            r = rate_for_q(q)
            if target_min <= r <= target_max:
                best_q = q
                break

    best_q = max(search_lo, min(best_q, search_hi))
    chosen_rate = rate_for_q(best_q)
    tp_kept = int((tp_q_valid >= best_q).sum()) if n_tp > 0 else 0

    logger.info(f"[LABEL_BALANCE] chosen_q_min_tp={best_q:.4f} "
                f"tp_first_rate_raw={n_tp/n_total:.3f} "
                f"tp_kept_after_q={tp_kept} enter_rate_train={chosen_rate:.3f} "
                f"target={target_rate:.2f}")

    return round(float(best_q), 4)


def generate_v47_quality_targets(
    df: pd.DataFrame,
    htf_features: pd.DataFrame,
    horizon_periods: int = 16,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    q_min_tp: float = 0.3,
    r_min_expiry_strict: float = 1.0,
    soft_label_temp: float = 1.2,
    auto_balance: bool = True,
    target_enter_rate: float = 0.18,
    target_enter_rate_min: float = 0.12,
    target_enter_rate_max: float = 0.25,
    balance_search_steps: int = 30,
    train_mask: np.ndarray = None,
) -> pd.DataFrame:
    """v4.7.1 TP Quality Score Balancing — continuous discriminator for TP-first bars.

    ENTER=1 only if:
      (A) TP was hit BEFORE SL AND tp_quality >= q_min_tp, OR
      (B) Expiry with R >= r_min_expiry_strict

    If auto_balance=True, searches q_min_tp in [0.0, 0.9] on training bars
    to achieve target_enter_rate.

    Returns same columns as v4.7 plus tp_quality column and updated diagnostics.
    """
    from training.triple_barrier import (
        compute_atr_14, bidirectional_outcome_v47_for_index,
        compute_htf_score_target, compute_mfe_mae_for_index,
        compute_trade_cost_r, compute_soft_quality,
    )

    n = len(df)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    atr_vals = compute_atr_14(df)

    h1_trend = htf_features['h1_trend_sign'].values if 'h1_trend_sign' in htf_features.columns else np.zeros(n)
    h4_trend = htf_features['h4_trend_sign'].values if 'h4_trend_sign' in htf_features.columns else np.zeros(n)

    best_r_all = np.full(n, np.nan, dtype=np.float64)
    tp_first_all = np.zeros(n, dtype=bool)
    tp_quality_all = np.full(n, np.nan, dtype=np.float64)
    exp_win_all = np.zeros(n, dtype=bool)
    long_r_arr = np.full(n, np.nan, dtype=np.float64)
    short_r_arr = np.full(n, np.nan, dtype=np.float64)
    y_dir_arr = np.zeros(n, dtype=np.int64)
    y_dir_conf_arr = np.full(n, 0.5, dtype=np.float64)

    outcome_types = np.full(n, 'SKIP', dtype=object)
    long_outcomes = np.full(n, 'SKIP', dtype=object)
    short_outcomes = np.full(n, 'SKIP', dtype=object)

    labeled_count = n - horizon_periods
    for i in range(labeled_count):
        a = float(atr_vals[i])
        if np.isnan(a) or a <= 0:
            a = closes[i] * 0.005

        result = bidirectional_outcome_v47_for_index(
            highs, lows, closes, i, a,
            tp_atr_mult, sl_atr_mult, horizon_periods,
            r_min_expiry_strict,
        )

        best_r_all[i] = result['best_outcome_r']
        long_r_arr[i] = result['long_r']
        short_r_arr[i] = result['short_r']
        y_dir_arr[i] = result['y_dir']
        y_dir_conf_arr[i] = result['y_dir_conf']
        outcome_types[i] = result['best_outcome_type']
        long_outcomes[i] = result['long_outcome']
        short_outcomes[i] = result['short_outcome']

        best_side = +1 if result['y_dir'] == 1 else -1
        tp_first_all[i] = (result['long_tp_first'] if best_side > 0 else result['short_tp_first'])
        tp_quality_all[i] = result['tp_quality']
        exp_win_all[i] = (
            (result['long_outcome'] == 'EXP_WIN' and result['long_r'] >= r_min_expiry_strict)
            if best_side > 0 else
            (result['short_outcome'] == 'EXP_WIN' and result['short_r'] >= r_min_expiry_strict)
        )

    valid_best = best_r_all[~np.isnan(best_r_all)]
    if len(valid_best) > 0:
        pcts = np.percentile(valid_best, [0, 10, 25, 50, 75, 90, 95, 99, 100])
        n_unique = len(np.unique(valid_best))
        logger.info(f"[BEST_R_HIST] ALL: min={pcts[0]:.4f} p10={pcts[1]:.4f} p25={pcts[2]:.4f} "
                    f"p50={pcts[3]:.4f} p75={pcts[4]:.4f} p90={pcts[5]:.4f} "
                    f"p95={pcts[6]:.4f} p99={pcts[7]:.4f} max={pcts[8]:.4f} "
                    f"n_unique={n_unique}")

    if auto_balance:
        if train_mask is not None:
            cal_tq = tp_quality_all[train_mask]
            cal_tp = tp_first_all[train_mask]
            cal_ew = exp_win_all[train_mask]
            cal_n = int(train_mask.sum())
        else:
            cal_tq = tp_quality_all[:labeled_count]
            cal_tp = tp_first_all[:labeled_count]
            cal_ew = exp_win_all[:labeled_count]
            cal_n = labeled_count

        chosen_q = _auto_calibrate_q_min_tp(
            cal_tq, cal_tp, cal_ew, cal_n,
            target_rate=target_enter_rate,
            target_min=target_enter_rate_min,
            target_max=target_enter_rate_max,
            search_steps=balance_search_steps,
        )
        logger.info(f"[LABEL_BALANCE] target={target_enter_rate:.2f} "
                     f"range=[{target_enter_rate_min:.2f}, {target_enter_rate_max:.2f}]")
        q_min_tp = chosen_q
    else:
        logger.info(f"[LABEL_BALANCE] using fixed q_min_tp={q_min_tp:.4f} (auto_balance=False)")

    BACKOFF_STEP = 0.05
    BACKOFF_FLOOR = 0.0
    MAX_BACKOFF_ATTEMPTS = 20

    for backoff_attempt in range(MAX_BACKOFF_ATTEMPTS + 1):

        y_quality_arr = np.zeros(n, dtype=np.int64)
        y_htf_score_arr = np.zeros(n, dtype=np.int64)
        mfe_r_arr = np.full(n, np.nan, dtype=np.float64)
        mae_r_arr = np.full(n, np.nan, dtype=np.float64)
        y_soft_arr = np.full(n, 0.5, dtype=np.float64)
        enter_labels = np.zeros(n, dtype=np.int64)
        side_hints = np.zeros(n, dtype=np.int64)
        outcomes_col = np.full(n, "NO_CANDIDATE", dtype=object)
        realized_r = np.full(n, np.nan, dtype=np.float64)

        n_tp_first_total = 0
        n_tp_kept = 0
        n_expiry_strong = 0
        n_sl_hit = 0
        n_tp_hit_total = 0
        n_expiry_total = 0
        n_quality_1 = 0
        n_long_better = 0
        n_short_better = 0
        htf_class_counts = [0, 0, 0, 0]

        for i in range(labeled_count):
            if np.isnan(best_r_all[i]):
                continue

            a = float(atr_vals[i])
            if np.isnan(a) or a <= 0:
                a = closes[i] * 0.005

            is_enter = 0
            if tp_first_all[i]:
                n_tp_first_total += 1
                if not np.isnan(tp_quality_all[i]) and tp_quality_all[i] >= q_min_tp:
                    is_enter = 1
                    n_tp_kept += 1
            elif exp_win_all[i]:
                is_enter = 1
                n_expiry_strong += 1

            y_quality_arr[i] = is_enter
            enter_labels[i] = is_enter
            if is_enter:
                n_quality_1 += 1

            lo = str(long_outcomes[i])
            so = str(short_outcomes[i])
            if lo == 'TP' or so == 'TP':
                n_tp_hit_total += 1
            if lo == 'SL' or so == 'SL':
                n_sl_hit += 1
            if lo in ('EXP_WIN', 'EXP_LOSS') or so in ('EXP_WIN', 'EXP_LOSS'):
                n_expiry_total += 1

            best_side = +1 if y_dir_arr[i] == 1 else -1
            side_hints[i] = best_side
            realized_r[i] = best_r_all[i]

            if is_enter:
                outcomes_col[i] = lo if best_side > 0 else so
            else:
                outcomes_col[i] = "REJECTED"

            if y_dir_arr[i] == 1:
                n_long_better += 1
            else:
                n_short_better += 1

            htf_score = compute_htf_score_target(float(h1_trend[i]), float(h4_trend[i]))
            y_htf_score_arr[i] = htf_score
            htf_class_counts[htf_score] += 1

            mfe, mae = compute_mfe_mae_for_index(
                highs, lows, closes, i, best_side, a,
                sl_atr_mult, horizon_periods,
            )
            mfe_r_arr[i] = mfe
            mae_r_arr[i] = mae

            cost_r_val = compute_trade_cost_r(closes[i], a, sl_atr_mult)
            y_soft_arr[i] = compute_soft_quality(mfe, mae, cost_r_val, soft_label_temp)

        if n_quality_1 > 0:
            if backoff_attempt > 0:
                logger.info(f"[LABEL_BALANCE] safety backoff succeeded after {backoff_attempt} step(s), "
                             f"q_min_tp={q_min_tp:.4f}, ENTER=1={n_quality_1}")
            break

        if q_min_tp <= BACKOFF_FLOOR:
            logger.error(f"[LABEL_BALANCE] ENTER=1 count is 0 even at floor q_min_tp={BACKOFF_FLOOR:.2f}")
            break

        new_q = max(q_min_tp - BACKOFF_STEP, BACKOFF_FLOOR)
        logger.warning(f"[LABEL_BALANCE] ENTER=1 count is 0 with q_min_tp={q_min_tp:.4f}, "
                        f"backing off to {new_q:.4f} (attempt {backoff_attempt + 1})")
        q_min_tp = round(new_q, 4)

    enter_rate = n_quality_1 / max(labeled_count, 1)
    max_best_r = float(np.max(valid_best)) if len(valid_best) > 0 else 0.0

    logger.info("=" * 70)
    logger.info("v4.7.1 TP QUALITY SCORE BALANCING")
    logger.info("=" * 70)
    logger.info(f"[LABEL_V471] Total bars: {n:,}, Labeled: {labeled_count:,}")
    logger.info(f"[LABEL_V471] TP_hits={n_tp_hit_total:,}, SL_hits={n_sl_hit:,}, Expiry={n_expiry_total:,}")
    if len(valid_best) > 0:
        logger.info(f"[LABEL_V471] best_R: mean={np.mean(valid_best):.3f}, median={np.median(valid_best):.3f}, "
                     f"p75={np.percentile(valid_best, 75):.3f}, p90={np.percentile(valid_best, 90):.3f}, "
                     f"p95={np.percentile(valid_best, 95):.3f}, max={max_best_r:.4f}")

    tp_q_valid = tp_quality_all[tp_first_all & ~np.isnan(tp_quality_all)]
    if len(tp_q_valid) > 0:
        tpcts = np.percentile(tp_q_valid, [50, 75, 90, 95, 100])
        logger.info(f"[TP_QUAL] q: p50={tpcts[0]:.3f} p75={tpcts[1]:.3f} p90={tpcts[2]:.3f} "
                    f"p95={tpcts[3]:.3f} max={tpcts[4]:.3f}")

    logger.info(f"[LABEL_V471] TP_first_total={n_tp_first_total:,}, TP_kept_after_q={n_tp_kept:,}, "
                f"Expiry_strong={n_expiry_strong:,}, ENTER_rate={100*enter_rate:.1f}%")
    logger.info(f"[LABEL_V471] q_min_tp={q_min_tp:.4f}, r_min_expiry_strict={r_min_expiry_strict:.4f}")
    expiry_pos_share = n_expiry_strong / max(n_quality_1, 1)
    logger.info(f"[LABEL_V471] Expiry positive share: {100*expiry_pos_share:.1f}%")
    logger.info(f"[LABEL_V471] y_dir: LONG_better={n_long_better:,} SHORT_better={n_short_better:,}")
    logger.info(f"[LABEL_V471] HTF score distribution: {dict(enumerate(htf_class_counts))}")
    logger.info("=" * 70)

    if n_quality_1 == 0:
        raise ValueError(
            f"[LABEL_ERROR] ENTER positives are zero after all backoff attempts. "
            f"q_min_tp={q_min_tp:.4f} max_best_R={max_best_r:.4f} "
            f"labeled_count={labeled_count} tp_first_any={tp_first_all.sum()} "
            f"exp_win_any={exp_win_all.sum()}"
        )

    result_df = pd.DataFrame({
        'enter_label': enter_labels,
        'side_hint': side_hints,
        'outcome': outcomes_col,
        'realized_r': realized_r,
        'y_quality': y_quality_arr,
        'y_dir': y_dir_arr,
        'y_dir_conf': y_dir_conf_arr,
        'y_htf_score': y_htf_score_arr,
        'long_r': long_r_arr,
        'short_r': short_r_arr,
        'best_r': best_r_all,
        'tp_quality': tp_quality_all,
        'mfe_r': mfe_r_arr,
        'mae_r': mae_r_arr,
        'y_soft': y_soft_arr,
    }, index=df.index)

    result_df.attrs['v47_diagnostics'] = {
        'q_min_tp': q_min_tp,
        'r_min_expiry_strict': r_min_expiry_strict,
        'enter_rate': enter_rate,
        'n_enter_1': n_quality_1,
        'n_tp_first_total': n_tp_first_total,
        'n_tp_kept': n_tp_kept,
        'n_expiry_strong': n_expiry_strong,
        'n_tp_hit_total': n_tp_hit_total,
        'n_sl_hit': n_sl_hit,
        'n_expiry_total': n_expiry_total,
        'expiry_pos_share': expiry_pos_share,
        'auto_balanced': auto_balance,
    }

    return result_df


def generate_multi_preset_targets(
    df: pd.DataFrame,
    htf_features: pd.DataFrame,
    presets: list,
    horizon_periods: int = 16,
    q_min_tp: float = 0.3,
    r_min_expiry_strict: float = 1.0,
    soft_label_temp: float = 1.0,
    auto_balance: bool = True,
    target_enter_rate: float = 0.18,
    target_enter_rate_min: float = 0.12,
    target_enter_rate_max: float = 0.25,
    balance_search_steps: int = 30,
) -> pd.DataFrame:
    """Multi-preset triple-barrier label generation.

    Computes triple-barrier outcomes for MULTIPLE barrier presets (TP/SL
    combinations) per bar, reusing ``bidirectional_outcome_v47_for_index``
    once per preset.

    Args:
        df: OHLCV DataFrame with close/high/low columns.
        htf_features: DataFrame with HTF feature columns (h1_trend_sign,
            h4_trend_sign, etc.).
        presets: List of dicts, each containing:
            - ``tp_mult`` (float): ATR multiplier for take-profit barrier.
            - ``sl_mult`` (float): ATR multiplier for stop-loss barrier.
            - ``label``  (str):  Short name used as column suffix
              (e.g. ``'tight'``, ``'standard'``, ``'wide'``, ``'asymmetric'``).
        horizon_periods: Maximum bars to hold before time expiry.
        q_min_tp: Minimum TP quality score for ENTER=1 (may be auto-calibrated).
        r_min_expiry_strict: Minimum realized R at expiry for ENTER=1.
        soft_label_temp: Temperature for soft quality label.
        auto_balance: If True, search ``q_min_tp`` per preset to achieve
            ``target_enter_rate``.
        target_enter_rate: Target ENTER=1 positive rate.
        target_enter_rate_min: Lower bound for acceptable enter rate.
        target_enter_rate_max: Upper bound for acceptable enter rate.
        balance_search_steps: Number of search steps for auto-balance.

    Returns:
        DataFrame with per-preset columns:
            - ``realized_r_{label}``, ``outcome_{label}``, ``side_hint_{label}``
        Plus shared columns: ``y_dir``, ``y_dir_conf``, ``y_htf_score``.
    """
    from training.triple_barrier import (
        compute_atr_14, bidirectional_outcome_v47_for_index,
        compute_htf_score_target, compute_mfe_mae_for_index,
        compute_trade_cost_r, compute_soft_quality,
    )

    n = len(df)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    atr_vals = compute_atr_14(df)

    h1_trend = htf_features['h1_trend_sign'].values if 'h1_trend_sign' in htf_features.columns else np.zeros(n)
    h4_trend = htf_features['h4_trend_sign'].values if 'h4_trend_sign' in htf_features.columns else np.zeros(n)

    labeled_count = n - horizon_periods

    y_dir_arr = np.zeros(n, dtype=np.int64)
    y_dir_conf_arr = np.full(n, 0.5, dtype=np.float64)
    y_htf_score_arr = np.zeros(n, dtype=np.int64)

    first_preset = True

    result_cols: dict = {}

    for preset in presets:
        tp_mult = float(preset['tp_mult'])
        sl_mult = float(preset['sl_mult'])
        plabel = str(preset['label'])

        best_r_all = np.full(n, np.nan, dtype=np.float64)
        tp_first_all = np.zeros(n, dtype=bool)
        tp_quality_all = np.full(n, np.nan, dtype=np.float64)
        exp_win_all = np.zeros(n, dtype=bool)
        long_r_arr = np.full(n, np.nan, dtype=np.float64)
        short_r_arr = np.full(n, np.nan, dtype=np.float64)
        preset_dir_arr = np.zeros(n, dtype=np.int64)
        preset_dir_conf_arr = np.full(n, 0.5, dtype=np.float64)
        long_outcomes = np.full(n, 'SKIP', dtype=object)
        short_outcomes = np.full(n, 'SKIP', dtype=object)

        for i in range(labeled_count):
            a = float(atr_vals[i])
            if np.isnan(a) or a <= 0:
                a = closes[i] * 0.005

            result = bidirectional_outcome_v47_for_index(
                highs, lows, closes, i, a,
                tp_mult, sl_mult, horizon_periods,
                r_min_expiry_strict,
            )

            best_r_all[i] = result['best_outcome_r']
            long_r_arr[i] = result['long_r']
            short_r_arr[i] = result['short_r']
            preset_dir_arr[i] = result['y_dir']
            preset_dir_conf_arr[i] = result['y_dir_conf']
            long_outcomes[i] = result['long_outcome']
            short_outcomes[i] = result['short_outcome']

            best_side = +1 if result['y_dir'] == 1 else -1
            tp_first_all[i] = (result['long_tp_first'] if best_side > 0 else result['short_tp_first'])
            tp_quality_all[i] = result['tp_quality']
            exp_win_all[i] = (
                (result['long_outcome'] == 'EXP_WIN' and result['long_r'] >= r_min_expiry_strict)
                if best_side > 0 else
                (result['short_outcome'] == 'EXP_WIN' and result['short_r'] >= r_min_expiry_strict)
            )

            if first_preset:
                y_dir_arr[i] = result['y_dir']
                y_dir_conf_arr[i] = result['y_dir_conf']
                htf_score = compute_htf_score_target(float(h1_trend[i]), float(h4_trend[i]))
                y_htf_score_arr[i] = htf_score

        preset_q_min = q_min_tp
        if auto_balance:
            cal_tq = tp_quality_all[:labeled_count]
            cal_tp = tp_first_all[:labeled_count]
            cal_ew = exp_win_all[:labeled_count]
            cal_n = labeled_count

            preset_q_min = _auto_calibrate_q_min_tp(
                cal_tq, cal_tp, cal_ew, cal_n,
                target_rate=target_enter_rate,
                target_min=target_enter_rate_min,
                target_max=target_enter_rate_max,
                search_steps=balance_search_steps,
            )

        realized_r = np.full(n, np.nan, dtype=np.float64)
        side_hints = np.zeros(n, dtype=np.int64)
        outcomes_col = np.full(n, "NO_CANDIDATE", dtype=object)

        n_quality_1 = 0
        n_tp_kept = 0
        n_expiry_strong = 0

        for i in range(labeled_count):
            if np.isnan(best_r_all[i]):
                continue

            best_side = +1 if preset_dir_arr[i] == 1 else -1
            side_hints[i] = best_side
            realized_r[i] = best_r_all[i]

            is_enter = 0
            if tp_first_all[i]:
                if not np.isnan(tp_quality_all[i]) and tp_quality_all[i] >= preset_q_min:
                    is_enter = 1
                    n_tp_kept += 1
            elif exp_win_all[i]:
                is_enter = 1
                n_expiry_strong += 1

            if is_enter:
                n_quality_1 += 1
                outcomes_col[i] = str(long_outcomes[i]) if best_side > 0 else str(short_outcomes[i])
            else:
                outcomes_col[i] = "REJECTED"

        enter_rate = n_quality_1 / max(labeled_count, 1)
        valid_r = realized_r[~np.isnan(realized_r)]
        mean_r = float(np.mean(valid_r)) if len(valid_r) > 0 else 0.0
        median_r = float(np.median(valid_r)) if len(valid_r) > 0 else 0.0

        logger.info("=" * 70)
        logger.info(f"MULTI-PRESET [{plabel}] TP={tp_mult:.2f}x SL={sl_mult:.2f}x")
        logger.info(f"  Labeled bars: {labeled_count:,}")
        logger.info(f"  ENTER=1: {n_quality_1:,} ({100*enter_rate:.1f}%)")
        logger.info(f"  TP kept: {n_tp_kept:,}, Expiry strong: {n_expiry_strong:,}")
        logger.info(f"  q_min_tp: {preset_q_min:.4f}")
        logger.info(f"  Realized R: mean={mean_r:.4f}, median={median_r:.4f}")
        if len(valid_r) > 0:
            pcts = np.percentile(valid_r, [25, 50, 75, 90])
            logger.info(f"  R percentiles: p25={pcts[0]:.4f} p50={pcts[1]:.4f} "
                        f"p75={pcts[2]:.4f} p90={pcts[3]:.4f}")
        logger.info("=" * 70)

        result_cols[f'realized_r_{plabel}'] = realized_r
        result_cols[f'outcome_{plabel}'] = outcomes_col
        result_cols[f'side_hint_{plabel}'] = side_hints

        first_preset = False

    result_cols['y_dir'] = y_dir_arr
    result_cols['y_dir_conf'] = y_dir_conf_arr
    result_cols['y_htf_score'] = y_htf_score_arr

    return pd.DataFrame(result_cols, index=df.index)
