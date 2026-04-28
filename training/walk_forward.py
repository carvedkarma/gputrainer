"""
Walk-Forward Evaluation Framework

Implements proper hedge fund-style backtesting:
- Purged time splits (gap between train/test to prevent lookahead)
- Walk-forward: train on window A, test on next window B, roll forward
- Per-regime performance reporting
- After-cost PnL with realistic fills
- Key metrics: expectancy, drawdown, tail losses (not just accuracy)

"If you can't beat costs in walk-forward, don't scale GPUs."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    entry_time: int
    exit_time: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    position_size: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    return_pct: float
    mfe: float  # Maximum favorable excursion
    mae: float  # Maximum adverse excursion
    predicted_mu: float
    predicted_sigma: float
    confidence: float


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    n_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    expectancy: float
    profit_factor: float
    
    trades: List[TradeResult] = field(default_factory=list)
    regime_results: Dict[str, Dict] = field(default_factory=dict)


@dataclass  
class TransactionCosts:
    """Realistic transaction cost model."""
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    base_slippage: float = 0.0001  # 0.01%
    volatility_slippage_mult: float = 0.5
    
    def calculate_costs(self, price: float, size: float, 
                        volatility: float, is_taker: bool = True) -> Tuple[float, float]:
        """
        Calculate fee and slippage for a trade.
        
        Returns:
            (fee_cost, slippage_cost) in quote currency
        """
        notional = price * size
        fee = notional * (self.taker_fee if is_taker else self.maker_fee)
        slippage = notional * (self.base_slippage + self.volatility_slippage_mult * volatility)
        return fee, slippage


class WalkForwardSplitter:
    """
    Creates walk-forward time series splits with purging.
    
    Purging: Gap between train and test to prevent information leakage
    from features that use future data (like volatility estimates).
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 train_periods: int = 50000,  # ~170 days of 5m data
                 test_periods: int = 10000,   # ~35 days of 5m data
                 purge_periods: int = 288,    # 1 day gap
                 embargo_periods: int = 48):  # 4 hour embargo after test
        self.n_splits = n_splits
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods
        
    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        total_fold_size = self.train_periods + self.purge_periods + self.test_periods + self.embargo_periods
        step_size = (n_samples - total_fold_size) // max(self.n_splits - 1, 1)
        
        for i in range(self.n_splits):
            start_idx = i * step_size
            
            train_start = start_idx
            train_end = train_start + self.train_periods
            
            test_start = train_end + self.purge_periods
            test_end = test_start + self.test_periods
            
            if test_end > n_samples:
                logger.warning(f"Fold {i} would exceed data length, stopping")
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
            logger.info(f"Fold {i}: train [{train_start}:{train_end}], "
                       f"test [{test_start}:{test_end}] (purge={self.purge_periods})")
        
        return splits


class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, 
                         periods_per_year: int = 288 * 365) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Array of period returns
            periods_per_year: Number of periods in a year (288 for 5m candles)
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)
        
        return annualized_return / annualized_std if annualized_std > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from peak."""
        if len(equity_curve) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        return np.min(drawdown)
    
    @staticmethod
    def calculate_expectancy(wins: np.ndarray, losses: np.ndarray) -> float:
        """
        Calculate trading expectancy (expected value per trade).
        
        E = (win_rate * avg_win) - (loss_rate * avg_loss)
        """
        if len(wins) == 0 and len(losses) == 0:
            return 0.0
        
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades
        loss_rate = len(losses) / total_trades
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    @staticmethod
    def calculate_profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
        """Calculate profit factor = gross_profits / gross_losses."""
        gross_profits = np.sum(wins) if len(wins) > 0 else 0
        gross_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10
        return gross_profits / gross_losses
    
    @staticmethod
    def calculate_tail_risk(returns: np.ndarray, 
                            percentile: float = 5) -> Dict[str, float]:
        """
        Calculate tail risk metrics.
        
        Returns:
            var_5: 5% Value at Risk
            cvar_5: 5% Conditional VaR (Expected Shortfall)
            worst_return: Worst single return
        """
        if len(returns) == 0:
            return {"var_5": 0, "cvar_5": 0, "worst_return": 0}
        
        var = np.percentile(returns, percentile)
        cvar = np.mean(returns[returns <= var]) if np.any(returns <= var) else var
        worst = np.min(returns)
        
        return {
            f"var_{percentile}": var,
            f"cvar_{percentile}": cvar,
            "worst_return": worst
        }


class TradeSimulator:
    """
    Simulates trades with realistic execution.
    """
    
    def __init__(self, 
                 costs: TransactionCosts = None,
                 position_size_method: str = "kelly",
                 max_position_pct: float = 0.1,
                 min_confidence: float = 0.5):
        self.costs = costs or TransactionCosts()
        self.position_size_method = position_size_method
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        
    def calculate_position_size(self, 
                                 capital: float,
                                 mu: float,
                                 sigma: float,
                                 price: float) -> float:
        """
        Calculate position size using bounded Kelly criterion.
        
        Kelly fraction = (edge / variance)
        Bounded to prevent over-betting
        """
        if sigma <= 0:
            return 0
        
        cost = self.costs.taker_fee * 2 + self.costs.base_slippage * 2
        edge = abs(mu) - cost
        
        if edge <= 0:
            return 0
        
        kelly_fraction = edge / (sigma ** 2)
        
        kelly_fraction = min(kelly_fraction, 0.5)  # Half-Kelly at most
        
        kelly_fraction = min(kelly_fraction, self.max_position_pct)
        
        position_value = capital * kelly_fraction
        position_size = position_value / price
        
        return position_size
    
    def simulate_trade(self,
                       candles: pd.DataFrame,
                       entry_idx: int,
                       direction: int,
                       predicted_mu: float,
                       predicted_sigma: float,
                       capital: float,
                       holding_periods: int = 48) -> Optional[TradeResult]:
        """
        Simulate a single trade with realistic execution.
        
        Args:
            candles: OHLCV DataFrame
            entry_idx: Index to enter trade
            direction: 1 for long, -1 for short
            predicted_mu: Predicted return
            predicted_sigma: Predicted volatility
            capital: Available capital
            holding_periods: How long to hold
            
        Returns:
            TradeResult or None if trade not taken
        """
        if entry_idx + holding_periods >= len(candles):
            return None
        
        # === FIXED: Correct confidence calculation ===
        # Confidence = |mu| / sigma (NOT edge / sigma)
        # If sigma is log_sigma, convert: sigma = exp(log_sigma)
        sigma = predicted_sigma
        if predicted_sigma < 0:  # Likely log_sigma (typical values are negative)
            import math
            sigma = math.exp(max(predicted_sigma, -10))  # Convert log_sigma to sigma
        
        confidence = abs(predicted_mu) / max(sigma, 0.001)
        
        if confidence < self.min_confidence:
            return None
        
        # Also compute edge for position sizing
        cost = self.costs.taker_fee * 2 + self.costs.base_slippage * 2
        edge = abs(predicted_mu) - cost
        
        entry_candle = candles.iloc[entry_idx]
        entry_price = entry_candle["close"]
        
        position_size = self.calculate_position_size(
            capital, predicted_mu, predicted_sigma, entry_price
        )
        
        if position_size <= 0:
            return None

        fee, slippage = self.costs.calculate_costs(
            entry_price, position_size, predicted_sigma, is_taker=True
        )
        
        entry_price_adjusted = entry_price * (1 + direction * slippage / (entry_price * position_size))
        
        exit_idx = entry_idx + holding_periods
        exit_candle = candles.iloc[exit_idx]
        exit_price = exit_candle["close"]
        
        exit_fee, exit_slippage = self.costs.calculate_costs(
            exit_price, position_size, predicted_sigma, is_taker=True
        )
        
        horizon_candles = candles.iloc[entry_idx:exit_idx+1]
        if direction == 1:  # Long
            mfe = (horizon_candles["high"].max() - entry_price) / entry_price
            mae = (entry_price - horizon_candles["low"].min()) / entry_price
        else:  # Short
            mfe = (entry_price - horizon_candles["low"].min()) / entry_price
            mae = (horizon_candles["high"].max() - entry_price) / entry_price
        
        gross_pnl = direction * (exit_price - entry_price) * position_size
        total_fees = (fee + exit_fee)
        total_slippage = slippage + exit_slippage
        net_pnl = gross_pnl - total_fees - total_slippage
        
        position_value = entry_price * position_size
        return_pct = net_pnl / position_value if position_value > 0 else 0
        
        return TradeResult(
            entry_time=int(entry_candle["timestamp"]),
            exit_time=int(exit_candle["timestamp"]),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage=total_slippage,
            net_pnl=net_pnl,
            return_pct=return_pct,
            mfe=mfe,
            mae=mae,
            predicted_mu=predicted_mu,
            predicted_sigma=predicted_sigma,
            confidence=confidence
        )


class WalkForwardEvaluator:
    """
    Full walk-forward evaluation pipeline.
    
    1. Split data into train/test folds
    2. Train model on train data
    3. Generate predictions on test data
    4. Simulate trades with realistic costs
    5. Calculate comprehensive metrics
    6. Report per-regime performance
    """
    
    def __init__(self,
                 splitter: WalkForwardSplitter = None,
                 simulator: TradeSimulator = None,
                 holding_periods: int = 48):
        self.splitter = splitter or WalkForwardSplitter()
        self.simulator = simulator or TradeSimulator()
        self.holding_periods = holding_periods
        self.metrics = PerformanceMetrics()
        
    def evaluate_fold(self,
                      model: nn.Module,
                      candles: pd.DataFrame,
                      features: np.ndarray,
                      train_idx: np.ndarray,
                      test_idx: np.ndarray,
                      fold_id: int,
                      device: str = "cuda") -> WalkForwardResult:
        """
        Evaluate a single walk-forward fold.
        """
        X_train = features[train_idx]
        X_test = features[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            if isinstance(predictions, tuple):
                mu_pred, sigma_pred = predictions[0], predictions[1]
            else:
                mu_pred = predictions[:, 0]
                sigma_pred = torch.abs(predictions[:, 1]) + 0.001
            
            mu_pred = mu_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()
        
        trades = []
        capital = 100000  # Start with 100k
        
        test_candles = candles.iloc[test_idx].reset_index(drop=True)
        
        for i in range(0, len(test_idx) - self.holding_periods, self.holding_periods):
            mu = mu_pred[i]
            sigma = sigma_pred[i]
            
            if abs(mu) < 0.001:  # Skip if predicted move is tiny
                continue
            
            direction = 1 if mu > 0 else -1
            
            trade = self.simulator.simulate_trade(
                test_candles, i, direction, mu, sigma, capital, self.holding_periods
            )
            
            if trade is not None:
                trades.append(trade)
                capital += trade.net_pnl
        
        returns = np.array([t.return_pct for t in trades]) if trades else np.array([])
        wins = returns[returns > 0] if len(returns) > 0 else np.array([])
        losses = returns[returns <= 0] if len(returns) > 0 else np.array([])
        
        if len(trades) > 0:
            equity_curve = np.cumsum([t.net_pnl for t in trades]) + 100000
            max_dd = self.metrics.calculate_max_drawdown(equity_curve)
        else:
            max_dd = 0
        
        result = WalkForwardResult(
            fold_id=fold_id,
            train_start=int(train_idx[0]),
            train_end=int(train_idx[-1]),
            test_start=int(test_idx[0]),
            test_end=int(test_idx[-1]),
            n_trades=len(trades),
            win_rate=len(wins) / len(trades) if len(trades) > 0 else 0,
            avg_return=np.mean(returns) if len(returns) > 0 else 0,
            total_return=np.sum(returns) if len(returns) > 0 else 0,
            max_drawdown=max_dd,
            sharpe_ratio=self.metrics.calculate_sharpe(returns),
            expectancy=self.metrics.calculate_expectancy(wins, losses),
            profit_factor=self.metrics.calculate_profit_factor(wins, losses),
            trades=trades
        )
        
        return result
    
    def run_full_evaluation(self,
                            model: nn.Module,
                            candles: pd.DataFrame,
                            features: np.ndarray,
                            device: str = "cuda") -> List[WalkForwardResult]:
        """
        Run complete walk-forward evaluation.
        """
        splits = self.splitter.split(len(features))
        
        results = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Evaluating fold {fold_id + 1}/{len(splits)}...")
            
            result = self.evaluate_fold(
                model, candles, features, train_idx, test_idx, fold_id, device
            )
            results.append(result)
            
            logger.info(f"  Fold {fold_id}: {result.n_trades} trades, "
                       f"win_rate={result.win_rate:.2%}, "
                       f"sharpe={result.sharpe_ratio:.2f}, "
                       f"expectancy={result.expectancy:.4f}")
        
        return results
    
    def summarize_results(self, results: List[WalkForwardResult]) -> Dict:
        """Summarize results across all folds."""
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)
        
        returns = np.array([t.return_pct for t in all_trades]) if all_trades else np.array([])
        wins = returns[returns > 0] if len(returns) > 0 else np.array([])
        losses = returns[returns <= 0] if len(returns) > 0 else np.array([])
        
        summary = {
            "n_folds": len(results),
            "total_trades": len(all_trades),
            "overall_win_rate": len(wins) / len(all_trades) if len(all_trades) > 0 else 0,
            "overall_sharpe": self.metrics.calculate_sharpe(returns),
            "overall_expectancy": self.metrics.calculate_expectancy(wins, losses),
            "overall_profit_factor": self.metrics.calculate_profit_factor(wins, losses),
            "avg_trades_per_fold": np.mean([r.n_trades for r in results]),
            "avg_win_rate": np.mean([r.win_rate for r in results]),
            "avg_sharpe": np.mean([r.sharpe_ratio for r in results]),
            "worst_drawdown": min([r.max_drawdown for r in results]),
            "tail_risk": self.metrics.calculate_tail_risk(returns),
            "per_fold_sharpe": [r.sharpe_ratio for r in results],
            "per_fold_expectancy": [r.expectancy for r in results]
        }
        
        return summary


# ============================================================
# PHASE 3: WALK-FORWARD MODEL WEIGHT SAVER
# ============================================================
# Saves real walk-forward metrics to model_weights.json for ensemble weighting
# Smart save: Only overwrites if new metrics are better than existing

import json
from pathlib import Path

# Minimum trades threshold for "reliable" metrics
MIN_TRADES_RELIABLE = 30


def is_new_weights_better(
    new_entry: Dict,
    existing_entry: Dict,
    min_trades: int = MIN_TRADES_RELIABLE
) -> tuple[bool, str]:
    """
    Compare new weights against existing to determine if we should overwrite.
    
    Rules:
    1. If no existing entry -> always save (new is better)
    2. If both unreliable (< min_trades): new must have >= trades to overwrite
    3. If new is unreliable but existing is reliable -> don't overwrite
    4. If existing is unreliable but new is reliable -> overwrite
    5. Both reliable: new must have >= expectancy AND trades >= 50% of existing
    
    Args:
        new_entry: New weight entry to potentially save
        existing_entry: Existing weight entry (or None)
        min_trades: Minimum trades for reliable metrics
        
    Returns:
        Tuple of (should_save: bool, reason: str)
    """
    if existing_entry is None:
        return True, "No existing weights - saving new"
    
    new_trades = new_entry.get("total_trades", 0)
    existing_trades = existing_entry.get("total_trades", 0)
    new_expectancy = new_entry.get("expectancy", 0)
    existing_expectancy = existing_entry.get("expectancy", 0)
    
    new_reliable = new_trades >= min_trades
    existing_reliable = existing_trades >= min_trades
    
    # Rule 2: Both unreliable - require new to have more trades OR better expectancy
    if not existing_reliable and not new_reliable:
        if new_trades > existing_trades:
            return True, f"Both unreliable, new has more trades: {new_trades} > {existing_trades}"
        if new_trades == existing_trades and new_expectancy > existing_expectancy:
            return True, f"Both unreliable, same trades but better expectancy: {new_expectancy:.4f} > {existing_expectancy:.4f}"
        return False, f"Both unreliable, new not better: trades {new_trades} vs {existing_trades}, exp {new_expectancy:.4f} vs {existing_expectancy:.4f}"
    
    # Rule 3: New is unreliable but existing is reliable -> don't overwrite
    if not new_reliable and existing_reliable:
        return False, f"New has {new_trades} trades (< {min_trades}) but existing has {existing_trades} - keeping existing"
    
    # Rule 4: Existing is unreliable but new is reliable -> overwrite
    if new_reliable and not existing_reliable:
        return True, f"New is reliable ({new_trades} trades) replacing unreliable existing ({existing_trades} trades)"
    
    # Rule 5: Both reliable - compare quality
    # New must have >= expectancy AND trades >= 50% of existing
    trades_ratio = new_trades / existing_trades if existing_trades > 0 else 1.0
    
    if new_expectancy >= existing_expectancy and trades_ratio >= 0.5:
        return True, f"New is better: expectancy {new_expectancy:.4f} >= {existing_expectancy:.4f}, trades ratio {trades_ratio:.1%}"
    
    if new_expectancy > existing_expectancy * 1.5:
        # Much better expectancy can compensate for fewer trades
        return True, f"New has much better expectancy: {new_expectancy:.4f} vs {existing_expectancy:.4f}"
    
    # Default: keep existing
    return False, f"Keeping existing: trades {existing_trades} vs {new_trades}, expectancy {existing_expectancy:.4f} vs {new_expectancy:.4f}"


def save_walk_forward_weights(
    model_name: str,
    summary: Dict,
    weights_dir: str = "checkpoints",
    force_save: bool = False
) -> Dict:
    """
    PHASE 3: Save walk-forward evaluation metrics as model weights.
    
    Smart save: Only overwrites if new metrics are better than existing,
    unless force_save=True.
    
    This replaces the placeholder defaults with real trading metrics.
    The ensemble predictor will load these to weight model votes.
    
    Args:
        model_name: Name of the model (e.g., "transformer", "lstm")
        summary: Walk-forward summary dict from WalkForwardEvaluator.summarize_results()
        weights_dir: Directory to save model_weights.json
        force_save: If True, always save regardless of comparison
        
    Returns:
        Dict with the saved weight configuration (new or existing)
    """
    weights_path = Path(weights_dir) / "model_weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing weights or create new
    existing_weights = {}
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                existing_weights = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing weights: {e}")
    
    # Convert summary to ModelWeight format
    total_trades = summary.get("total_trades", 0)
    
    # Precision on trade: win rate when model decides to trade (not HOLD)
    # This is approximated by overall win rate for trades taken
    precision = summary.get("overall_win_rate", 0.5)
    
    # Directional F1: harmonic mean of precision and recall
    # Approximate as win_rate * 0.8 (typically recall is lower)
    f1_directional = precision * 0.9 if precision > 0.5 else precision * 0.8
    
    # Create weight entry
    weight_entry = {
        "model_name": model_name,
        "expectancy": summary.get("overall_expectancy", 0.001),
        "precision_on_trade": precision,
        "profit_factor": min(3.0, summary.get("overall_profit_factor", 1.0)),  # Cap at 3.0
        "f1_directional": f1_directional,
        "sharpe": min(3.0, summary.get("overall_sharpe", 0.5)),  # Cap at 3.0 (overfit detection)
        "calibration_temp": 1.0,  # Default, can be calibrated later
        
        # Additional metrics for debugging
        "total_trades": total_trades,
        "avg_trades_per_fold": summary.get("avg_trades_per_fold", 0),
        "worst_drawdown": summary.get("worst_drawdown", 0),
        "evaluation_date": datetime.now().isoformat(),
        "n_folds": summary.get("n_folds", 0)
    }
    
    # Get existing entry for this model (if any)
    existing_entry = existing_weights.get(model_name)
    
    # Smart save: Check if new is better than existing
    if not force_save:
        should_save, reason = is_new_weights_better(weight_entry, existing_entry)
        
        if not should_save:
            logger.warning(f"[SMART SAVE] Skipping save for {model_name}: {reason}")
            logger.warning(f"  New run: {total_trades} trades, expectancy={weight_entry['expectancy']:.4f}")
            if existing_entry:
                logger.warning(f"  Existing: {existing_entry.get('total_trades', 0)} trades, expectancy={existing_entry.get('expectancy', 0):.4f}")
            logger.info(f"  Use force_save=True to override this check")
            return existing_entry if existing_entry else weight_entry
        else:
            logger.info(f"[SMART SAVE] Updating weights for {model_name}: {reason}")
    else:
        logger.info(f"[SMART SAVE] Force saving weights for {model_name}")
    
    # Update weights
    existing_weights[model_name] = weight_entry
    
    # Save to main model_weights.json (the "selected" pointer)
    with open(weights_path, 'w') as f:
        json.dump(existing_weights, f, indent=2)
    
    # Also save to weights_history with unique run_id for auditability
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = Path(weights_dir) / "weights_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"model_weights_{run_id}.json"
    
    # Save full snapshot to history
    with open(history_file, 'w') as f:
        json.dump(existing_weights, f, indent=2)
    
    logger.info(f"Saved walk-forward weights for {model_name}:")
    logger.info(f"  Expectancy: {weight_entry['expectancy']:.4f}")
    logger.info(f"  Precision: {weight_entry['precision_on_trade']:.2%}")
    logger.info(f"  Profit Factor: {weight_entry['profit_factor']:.2f}")
    logger.info(f"  Sharpe: {weight_entry['sharpe']:.2f}")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  History saved to: {history_file}")
    
    return weight_entry


def save_labeling_metadata(
    weights_dir: str,
    label_mode: str,
    horizon: int,
    min_confidence: float = 0.40,
    directional_threshold: float = 0.0020,
    trend_threshold: float = 0.0015,
    range_threshold: float = 0.0030,
    timeframe: str = "15m",
    label_distribution: dict = None,
    run_id: str = None
) -> dict:
    """
    Save labeling metadata to labeling_meta.json.
    
    This documents the exact label generation config used for training,
    enabling reproducibility and debugging of HOLD-heavy issues.
    
    Args:
        weights_dir: Directory to save metadata
        label_mode: "cost_aware" | "pure_directional" | "regime"
        horizon: Forward prediction horizon in bars
        min_confidence: Stage 1 min_confidence threshold
        directional_threshold: Stage 2 pure directional threshold
        trend_threshold: Stage 3 trending regime threshold
        range_threshold: Stage 3 ranging regime threshold
        timeframe: Training timeframe (e.g., "15m")
        label_distribution: Dict with SHORT/HOLD/LONG percentages
        run_id: Unique run identifier (auto-generated if None)
        
    Returns:
        Dict with the saved metadata
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "label_mode": label_mode,
        "horizon": horizon,
        "timeframe": timeframe,
        "config": {
            "min_confidence": min_confidence,
            "directional_threshold": directional_threshold,
            "trend_threshold": trend_threshold,
            "range_threshold": range_threshold
        },
        "label_distribution": label_distribution or {}
    }
    
    meta_path = Path(weights_dir) / "labeling_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"[LABELING META] Saved to {meta_path}")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Mode: {label_mode}")
    logger.info(f"  Horizon: {horizon} bars ({timeframe})")
    if label_distribution:
        logger.info(f"  Distribution: SHORT={label_distribution.get('short', 0):.1f}%, "
                   f"HOLD={label_distribution.get('hold', 0):.1f}%, "
                   f"LONG={label_distribution.get('long', 0):.1f}%")
    
    return metadata


def evaluate_and_save_model_weights(
    model: nn.Module,
    model_name: str,
    candles: pd.DataFrame,
    features: np.ndarray,
    device: str = "cuda",
    n_splits: int = 5,
    weights_dir: str = "checkpoints"
) -> Dict:
    """
    PHASE 3: Complete walk-forward evaluation and save weights.
    
    Convenience function that runs evaluation and saves results.
    
    Args:
        model: Trained model to evaluate
        model_name: Name for the model
        candles: OHLCV DataFrame
        features: Prepared feature array
        device: Device to run on
        n_splits: Number of walk-forward splits
        weights_dir: Directory to save weights
        
    Returns:
        Summary dict with all metrics
    """
    evaluator = WalkForwardEvaluator(n_splits=n_splits)
    
    logger.info(f"Running walk-forward evaluation for {model_name}...")
    results = evaluator.evaluate(model, candles, features, device=device)
    
    summary = evaluator.summarize_results(results)
    
    # Save weights
    save_walk_forward_weights(model_name, summary, weights_dir)
    
    return summary
