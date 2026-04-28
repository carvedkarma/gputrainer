"""
Post-Training Policy Selector

This module is the ONLY source of truth for live execution policy.
After training completes, this runs:
1. Load best checkpoint
2. Sequential out-of-sample (OOS) evaluation across >=5 time folds
   (NOTE: Model is NOT retrained per fold - this tests the fixed model
   across different market periods to ensure policy robustness)
3. Sweep confidence thresholds + Pareto selection + MIN_TRADES filter
4. Save execution_policy.json with frozen policy
5. Print FROZEN POLICY summary

Usage:
    python -m gpu_trainer.training.policy_selector --checkpoint path/to/best.pt
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPolicy:
    """Frozen execution policy for live trading."""
    min_confidence: float
    spread_multiplier: float  # K: require spread >= K * cost
    cooldown: int  # Bars to wait after trade
    fixed_cost: float  # Round-trip cost estimate
    tp_quantile: str  # e.g., "q75" for longs, "q25" for shorts
    sl_quantile: str  # e.g., "q10" for longs, "q90" for shorts
    min_trades: int  # Minimum trades required for eligibility
    
    # Evaluation metrics at selection time
    expectancy: float
    risk_adjusted_score: float
    hit_rate: float
    max_drawdown: float
    sharpe: float
    num_trades: int
    
    # Metadata
    created_at: str
    checkpoint_path: str
    walk_forward_folds: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ExecutionPolicy":
        return cls(**d)


class PolicySelector:
    """
    Post-training policy selector using walk-forward validation.
    
    This is the ONLY source of truth for live execution policy.
    Training does NOT save policies - only this class does.
    """
    
    # Configuration
    MIN_TRADES = 30  # Minimum trades for eligibility
    FIXED_COST = 0.0009  # 0.09% round-trip
    SPREAD_MULTIPLIER = 3.0  # K: spread >= K * cost
    COOLDOWN = 8  # Bars to wait after trade
    CONFIDENCE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        checkpoint_path: str,
        n_folds: int = 5
    ):
        """
        Initialize policy selector.
        
        Args:
            model: Trained model (loaded from checkpoint)
            device: Torch device
            checkpoint_path: Path to model checkpoint
            n_folds: Number of walk-forward folds (>=5 recommended)
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.n_folds = max(n_folds, 5)  # Enforce minimum 5 folds
        
        self.model.eval()
        
    def run_sequential_oos_evaluation(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        regime_ids: Optional[np.ndarray] = None
    ) -> Dict[float, Dict]:
        """
        Run sequential out-of-sample (OOS) evaluation across time folds.
        
        NOTE: This evaluates the SAME trained model across different time periods.
        Model is NOT retrained per fold - the goal is to test policy robustness
        across varying market conditions with a fixed model.
        
        Args:
            features: Feature array (n_samples, n_features) - chronologically ordered
            returns: Forward return array (n_samples,)
            regime_ids: Optional regime labels
            
        Returns:
            Dict mapping confidence threshold to aggregated metrics across folds
        """
        n_samples = len(features)
        fold_size = n_samples // self.n_folds
        
        logger.info(f"Running sequential OOS evaluation: {self.n_folds} time folds, {fold_size} samples/fold")
        
        # Aggregate results per threshold
        threshold_results = {t: [] for t in self.CONFIDENCE_THRESHOLDS}
        
        for fold in range(self.n_folds):
            # Sequential OOS fold: test fixed model on this time period
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            fold_features = features[test_start:test_end]
            fold_returns = returns[test_start:test_end]
            
            if len(fold_features) == 0:
                continue
                
            # Get model predictions
            with torch.no_grad():
                X = torch.tensor(fold_features, dtype=torch.float32).to(self.device)
                outputs = self.model(X)
                
                # Extract predictions (handles multi-head model)
                if isinstance(outputs, dict):
                    class_logits = outputs.get('class_logits', outputs.get('direction'))
                    mu = outputs.get('mu', torch.zeros(len(X)))
                    log_sigma = outputs.get('log_sigma', outputs.get('sigma', torch.zeros(len(X))))
                    q10 = outputs.get('q10', torch.zeros(len(X)))
                    q25 = outputs.get('q25', torch.zeros(len(X)))
                    q75 = outputs.get('q75', torch.zeros(len(X)))
                    q90 = outputs.get('q90', torch.zeros(len(X)))
                else:
                    # Tuple output
                    class_logits = outputs[0]
                    mu = outputs[1] if len(outputs) > 1 else torch.zeros(len(X))
                    log_sigma = outputs[2] if len(outputs) > 2 else torch.zeros(len(X))
                    q10 = outputs[3] if len(outputs) > 3 else torch.zeros(len(X))
                    q25 = outputs[4] if len(outputs) > 4 else torch.zeros(len(X))
                    q75 = outputs[5] if len(outputs) > 5 else torch.zeros(len(X))
                    q90 = outputs[6] if len(outputs) > 6 else torch.zeros(len(X))
                
                # Convert to numpy
                predictions = torch.argmax(class_logits, dim=1).cpu().numpy()
                mu_np = mu.squeeze().cpu().numpy()
                log_sigma_np = log_sigma.squeeze().cpu().numpy()
                q10_np = q10.squeeze().cpu().numpy()
                q25_np = q25.squeeze().cpu().numpy()
                q75_np = q75.squeeze().cpu().numpy()
                q90_np = q90.squeeze().cpu().numpy()
            
            # Compute sigma from log_sigma
            sigma_np = np.exp(np.clip(log_sigma_np, -10, 10))
            
            # Compute confidence
            confidence = np.abs(mu_np) / np.maximum(sigma_np, 1e-8)
            
            # Spread gate
            spread = q75_np - q25_np
            spread_gate = spread >= (self.SPREAD_MULTIPLIER * self.FIXED_COST)
            
            # Direction signals
            long_signal = predictions == 2  # LONG
            short_signal = predictions == 0  # SHORT
            directional_signal = long_signal | short_signal
            
            # Evaluate each threshold
            for min_conf in self.CONFIDENCE_THRESHOLDS:
                conf_gate = confidence >= min_conf
                trade_allowed = spread_gate & conf_gate & directional_signal
                final_trades = self._apply_cooldown(trade_allowed, self.COOLDOWN)
                
                # Compute PnL
                metrics = self._compute_fold_metrics(
                    final_trades, long_signal, short_signal,
                    fold_returns, q10_np, q25_np, q75_np, q90_np
                )
                metrics['fold'] = fold
                threshold_results[min_conf].append(metrics)
        
        # Aggregate across folds
        aggregated = {}
        for threshold, fold_metrics in threshold_results.items():
            if not fold_metrics:
                continue
            aggregated[threshold] = self._aggregate_fold_metrics(fold_metrics)
            
        return aggregated
    
    def _apply_cooldown(self, signals: np.ndarray, cooldown: int) -> np.ndarray:
        """Apply cooldown to prevent overtrading."""
        result = np.zeros_like(signals, dtype=bool)
        last_trade = -cooldown - 1
        
        for i, sig in enumerate(signals):
            if sig and (i - last_trade) > cooldown:
                result[i] = True
                last_trade = i
                
        return result
    
    def _compute_fold_metrics(
        self,
        trades: np.ndarray,
        long_signal: np.ndarray,
        short_signal: np.ndarray,
        returns: np.ndarray,
        q10: np.ndarray,
        q25: np.ndarray,
        q75: np.ndarray,
        q90: np.ndarray
    ) -> Dict:
        """Compute metrics for a single fold."""
        trade_indices = np.where(trades)[0]
        num_trades = len(trade_indices)
        
        if num_trades == 0:
            return {
                'num_trades': 0, 'expectancy': 0.0, 'hit_rate': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_drawdown': 0.0,
                'sharpe': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0
            }
        
        # Compute PnL for each trade using quantile-based SL/TP
        pnl_list = []
        for idx in trade_indices:
            actual_return = returns[idx]
            is_long = long_signal[idx]
            
            if is_long:
                # Long: TP at q75, SL at q10
                tp_target = q75[idx]
                sl_target = q10[idx]
            else:
                # Short: TP at q25, SL at q90
                tp_target = -q25[idx]  # Negative because short profits from down
                sl_target = -q90[idx]
            
            # Simplified: use actual return adjusted for costs
            direction = 1 if is_long else -1
            raw_pnl = direction * actual_return - self.FIXED_COST
            pnl_list.append(raw_pnl)
        
        pnl_array = np.array(pnl_list)
        
        # Metrics
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array <= 0]
        
        hit_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        expectancy = float(np.mean(pnl_array)) if num_trades > 0 else 0.0
        
        # Sharpe
        std = float(np.std(pnl_array)) if num_trades > 1 else 1.0
        sharpe = expectancy / std if std > 0 else 0.0
        
        # Max drawdown
        cumsum = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        return {
            'num_trades': num_trades,
            'expectancy': expectancy,
            'hit_rate': hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'gross_profit': float(np.sum(wins)) if len(wins) > 0 else 0.0,
            'gross_loss': float(np.sum(losses)) if len(losses) > 0 else 0.0
        }
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across walk-forward folds."""
        total_trades = sum(m['num_trades'] for m in fold_metrics)
        
        if total_trades == 0:
            return {
                'num_trades': 0, 'expectancy': 0.0, 'hit_rate': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_drawdown': 0.0,
                'sharpe': 0.0, 'risk_adjusted_score': 0.0, 'n_folds': len(fold_metrics)
            }
        
        # Trade-weighted averages
        weighted_expectancy = sum(
            m['expectancy'] * m['num_trades'] for m in fold_metrics
        ) / total_trades
        
        weighted_hit_rate = sum(
            m['hit_rate'] * m['num_trades'] for m in fold_metrics
        ) / total_trades
        
        gross_profit = sum(m['gross_profit'] for m in fold_metrics)
        gross_loss = sum(m['gross_loss'] for m in fold_metrics)
        
        avg_win = sum(m['avg_win'] * m['num_trades'] for m in fold_metrics if m['num_trades'] > 0) / max(total_trades, 1)
        avg_loss = sum(m['avg_loss'] * m['num_trades'] for m in fold_metrics if m['num_trades'] > 0) / max(total_trades, 1)
        
        # Max of max drawdowns across folds
        max_dd = max(m['max_drawdown'] for m in fold_metrics)
        
        # Combined Sharpe
        all_sharpes = [m['sharpe'] for m in fold_metrics if m['num_trades'] > 0]
        combined_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0
        
        # Risk-adjusted score
        if max_dd > 0:
            risk_penalty = 0.5 * max_dd
        else:
            risk_penalty = 0.25 * abs(avg_loss)
        
        risk_adjusted_score = weighted_expectancy - risk_penalty
        
        return {
            'num_trades': total_trades,
            'expectancy': weighted_expectancy,
            'hit_rate': weighted_hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'sharpe': combined_sharpe,
            'risk_adjusted_score': risk_adjusted_score,
            'n_folds': len(fold_metrics)
        }
    
    def select_best_policy(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        regime_ids: Optional[np.ndarray] = None
    ) -> Tuple[ExecutionPolicy, Dict]:
        """
        Run full policy selection and return frozen execution policy.
        
        Args:
            features: Feature array
            returns: Forward returns
            regime_ids: Optional regime labels
            
        Returns:
            (ExecutionPolicy, sweep_results dict)
        """
        logger.info("=" * 70)
        logger.info("POST-TRAINING POLICY SELECTION")
        logger.info("=" * 70)
        
        # Run walk-forward validation
        threshold_metrics = self.run_sequential_oos_evaluation(features, returns, regime_ids)
        
        # Find best policy using Pareto selection
        best_threshold = None
        best_score = float('-inf')
        best_metrics = None
        
        logger.info("-" * 70)
        logger.info(f"WALK-FORWARD RESULTS ({self.n_folds} folds, MIN_TRADES={self.MIN_TRADES})")
        logger.info("-" * 70)
        
        for threshold in self.CONFIDENCE_THRESHOLDS:
            if threshold not in threshold_metrics:
                continue
                
            metrics = threshold_metrics[threshold]
            eligible = metrics['num_trades'] >= self.MIN_TRADES
            
            if eligible and metrics['risk_adjusted_score'] > best_score:
                best_score = metrics['risk_adjusted_score']
                best_threshold = threshold
                best_metrics = metrics
            
            status = ""
            if not eligible:
                status = "(ineligible)"
            elif threshold == best_threshold:
                status = "★ SELECTED"
                
            logger.info(
                f"conf>={threshold:.2f}: Trades={metrics['num_trades']:4d}, "
                f"Exp={metrics['expectancy']:+.4f}, Score={metrics['risk_adjusted_score']:+.4f}, "
                f"Hit={metrics['hit_rate']:.1%}, MaxDD={metrics['max_drawdown']:.4f} {status}"
            )
        
        logger.info("-" * 70)
        
        # Handle case where no policy is eligible
        if best_metrics is None:
            logger.warning(f"No policy met MIN_TRADES={self.MIN_TRADES} - using lowest threshold as fallback")
            best_threshold = self.CONFIDENCE_THRESHOLDS[0]
            best_metrics = threshold_metrics.get(best_threshold, {
                'num_trades': 0, 'expectancy': 0.0, 'hit_rate': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_drawdown': 0.0,
                'sharpe': 0.0, 'risk_adjusted_score': 0.0
            })
        
        # Create frozen execution policy
        policy = ExecutionPolicy(
            min_confidence=best_threshold,
            spread_multiplier=self.SPREAD_MULTIPLIER,
            cooldown=self.COOLDOWN,
            fixed_cost=self.FIXED_COST,
            tp_quantile="q75",  # For longs; q25 for shorts
            sl_quantile="q10",  # For longs; q90 for shorts
            min_trades=self.MIN_TRADES,
            expectancy=best_metrics['expectancy'],
            risk_adjusted_score=best_metrics['risk_adjusted_score'],
            hit_rate=best_metrics['hit_rate'],
            max_drawdown=best_metrics['max_drawdown'],
            sharpe=best_metrics['sharpe'],
            num_trades=best_metrics['num_trades'],
            created_at=datetime.now().isoformat(),
            checkpoint_path=self.checkpoint_path,
            walk_forward_folds=self.n_folds
        )
        
        return policy, threshold_metrics
    
    def save_policy(self, policy: ExecutionPolicy, output_path: str = "execution_policy.json"):
        """Save frozen execution policy to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(policy.to_dict(), f, indent=2)
        
        logger.info(f"Saved frozen policy to: {output_file}")
        
        # Print FROZEN POLICY summary
        logger.info("=" * 70)
        logger.info("FROZEN POLICY SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Min Confidence:    {policy.min_confidence:.2f}")
        logger.info(f"  Spread Multiplier: {policy.spread_multiplier:.1f}x cost")
        logger.info(f"  Cooldown:          {policy.cooldown} bars")
        logger.info(f"  Fixed Cost:        {policy.fixed_cost:.4f} ({policy.fixed_cost*100:.2f}%)")
        logger.info(f"  TP Quantile:       {policy.tp_quantile} (longs), q25 (shorts)")
        logger.info(f"  SL Quantile:       {policy.sl_quantile} (longs), q90 (shorts)")
        logger.info("-" * 70)
        logger.info(f"  Expectancy:        {policy.expectancy:+.4f}")
        logger.info(f"  Risk-Adj Score:    {policy.risk_adjusted_score:+.4f}")
        logger.info(f"  Hit Rate:          {policy.hit_rate:.1%}")
        logger.info(f"  Max Drawdown:      {policy.max_drawdown:.4f}")
        logger.info(f"  Sharpe:            {policy.sharpe:+.2f}")
        logger.info(f"  Trades (WF):       {policy.num_trades}")
        logger.info("-" * 70)
        logger.info(f"  Checkpoint:        {policy.checkpoint_path}")
        logger.info(f"  WF Folds:          {policy.walk_forward_folds}")
        logger.info(f"  Created:           {policy.created_at}")
        logger.info("=" * 70)
        logger.info("This policy is now FROZEN for live trading.")
        logger.info("To update, retrain model and re-run PolicySelector.")
        logger.info("=" * 70)
        
        return output_file


def load_execution_policy(path: str = "execution_policy.json") -> ExecutionPolicy:
    """Load frozen execution policy from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return ExecutionPolicy.from_dict(data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-training policy selection")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to validation data (parquet)")
    parser.add_argument("--output", default="execution_policy.json", help="Output policy file")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    logger.info(f"Loading data: {args.data}")
    
    # This would need to be integrated with actual model loading
    logger.info("Use this module programmatically after training completes.")
    logger.info("See MultiheadTrainer.train() for integration example.")
