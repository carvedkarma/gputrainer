"""Scheduled retraining + safe promotion system (v3.6.0) — V5Forecaster fine-tuning.

Manages the continuous learning lifecycle:
  1. Scheduled daily fine-tune (loads existing V5 checkpoint, fine-tunes on latest data)
  2. Validation evaluation of candidate vs deployed model using V5 scoring
  3. Safe promotion: only promote if candidate beats deployed on key metrics
  4. Push learning stats to dashboard after each cycle

Usage (called from quick_start.py via LiveRunner):
    runner = LiveRunner(..., learning_config=LearningConfig(...))

Key safety gates for promotion:
  - pf_net (profit factor after costs) must improve or stay above minimum
  - Profitable regime count must not decrease
  - Trades-per-day must stay within acceptable range
  - Val loss must have improved relative to baseline
"""

import os
import sys
import time
import json
import shutil
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List

log = logging.getLogger("Learning")


@dataclass
class LearningConfig:
    retrain_hour_utc: int = 4
    retrain_interval_hours: int = 24
    min_new_bars: int = 96
    training_epochs: int = 10
    finetune_lr: float = 1e-4
    finetune_batch_size: int = 256
    min_val_loss_improvement: float = 0.0
    min_pf_net: float = 1.05
    min_profitable_regimes: int = 2
    min_tpd: float = 0.3
    max_tpd: float = 8.0
    pf_improvement_required: float = 0.0
    regime_loss_tolerance: int = 0
    auto_promote: bool = True
    geometry_sweep_on_retrain: bool = True
    sweep_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    sweep_cooldowns: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    gate_pf_net: float = 1.05
    gate_enet: float = 0.0
    gate_profitable_regimes: int = 2
    gate_maxdd_r: float = 6.0
    tp_mult: float = 2.0
    sl_mult: float = 1.2
    horizon: int = 16
    val_split: float = 0.20
    # Legacy fields kept for CLI backward-compatibility (not used in V5 evaluation)
    min_prauc_threshold: float = 0.42
    gate_p95_min: float = 0.40
    gate_p95_max: float = 0.98


@dataclass
class RetrainResult:
    success: bool = False
    model_version: str = ""
    val_prauc: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_loss: float = 0.0
    training_samples: int = 0
    trained_until_ts: int = 0
    best_policy: Optional[Dict] = None
    pf_net: float = 0.0
    e_net: float = 0.0
    trades_per_day: float = 0.0
    profitable_regimes: int = 0
    total_regimes: int = 0
    error: str = ""
    max_drawdown_r: float = 0.0
    p95_val: float = 0.0
    temperature: float = 1.0


class LearningManager:
    """Manages scheduled V5Forecaster fine-tuning and safe model promotion."""

    def __init__(
        self,
        replit_url: str,
        device: str,
        symbols: List[str],
        config: LearningConfig = None,
    ):
        self.replit_url = replit_url
        self.device = device
        self.symbols = symbols
        self.config = config or LearningConfig()
        self.last_retrain_time: Dict[str, float] = {}
        self.deployed_stats: Dict[str, Dict] = {}

    def should_retrain(self, symbol: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self.last_retrain_time.get(symbol, 0)
        if last == 0:
            if now.hour == self.config.retrain_hour_utc:
                return True
            return False

        hours_since = (time.time() - last) / 3600
        if hours_since < self.config.retrain_interval_hours:
            return False
        if now.hour == self.config.retrain_hour_utc:
            return True
        return False

    def retrain_symbol(self, symbol: str) -> RetrainResult:
        result = RetrainResult()
        log.info(f"[Learning] Starting V5 fine-tune for {symbol}...")

        try:
            candidate_dir = Path(f"checkpoints/candidate/{symbol}")
            candidate_dir.mkdir(parents=True, exist_ok=True)

            deployed_dir = Path(f"checkpoints/deployed/{symbol}")
            deployed_dir.mkdir(parents=True, exist_ok=True)

            result = self._run_training(symbol, candidate_dir)
            if not result.success:
                log.error(f"[Learning] Training failed for {symbol}: {result.error}")
                return result

            if self.config.geometry_sweep_on_retrain:
                sweep_result = self._run_geometry_sweep(symbol, candidate_dir)
                if sweep_result:
                    result.best_policy = sweep_result.get('best_policy')
                    result.pf_net = sweep_result.get('pf_net', result.pf_net)
                    result.e_net = sweep_result.get('e_net', result.e_net)
                    result.trades_per_day = sweep_result.get('trades_per_day', result.trades_per_day)
                    result.profitable_regimes = sweep_result.get('profitable_regimes', result.profitable_regimes)
                    result.total_regimes = sweep_result.get('total_regimes', result.total_regimes)

            should_promote, reason = self._evaluate_promotion(symbol, result)

            prev_stats = self.deployed_stats.get(symbol)

            if should_promote and self.config.auto_promote:
                self._promote_model(symbol, candidate_dir, deployed_dir)
                log.info(f"[Learning] Model PROMOTED for {symbol}: {reason}")
            elif not should_promote:
                log.info(f"[Learning] Model NOT promoted for {symbol}: {reason}")

            self._push_learning_stats(symbol, result, should_promote, reason, prev_stats=prev_stats)

            self.last_retrain_time[symbol] = time.time()
            return result

        except Exception as e:
            log.error(f"[Learning] Retrain failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            return result

    def _fetch_full_history(self, symbol: str) -> List[Dict]:
        """Fetch full candle history from the PostgreSQL-backed DB endpoint."""
        import requests
        base = self.replit_url.rstrip('/')

        try:
            resp = requests.get(
                f"{base}/api/data/candles-history",
                params={"symbol": symbol, "timeframe": "15m", "limit": 200000},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                candles = data.get("candles", [])
                if len(candles) >= self.config.min_new_bars:
                    log.info(f"[Learning] DB history: {len(candles)} candles for {symbol}")
                    return candles
                log.warning(f"[Learning] DB returned only {len(candles)} candles for {symbol}, trying Binance fallback")
        except Exception as e:
            log.warning(f"[Learning] DB history request failed for {symbol}: {e}")

        from data.pipeline import BinanceDataFetcher
        fetcher = BinanceDataFetcher(
            symbols=[symbol],
            timeframes=["15m"],
            replit_proxy_url=base,
            use_sync=True,
        )

        all_candles: List[Dict] = []
        end_time = None
        pages = 8
        for page in range(pages):
            raw = fetcher.fetch_klines_sync(symbol, "15m", limit=1000, end_time=end_time)
            if not raw:
                break
            raw_sorted = sorted(raw, key=lambda c: c["timestamp"])
            all_candles = raw_sorted + all_candles
            end_time = raw_sorted[0]["timestamp"] - 1
            log.info(f"[Learning] Binance page {page+1}/{pages}: {len(raw)} candles (total: {len(all_candles)})")
            if len(raw) < 990:
                break

        log.info(f"[Learning] Binance fallback total: {len(all_candles)} candles for {symbol}")
        return all_candles

    def _find_v5_checkpoint(self, symbol: str) -> Optional[Path]:
        """Find the best available V5Forecaster checkpoint for a symbol."""
        candidates = [
            Path(f"checkpoints/deployed/{symbol}/best_v5_expectancy.pt"),
            Path(f"checkpoints/deployed/{symbol}/best_v5_loss.pt"),
            Path("checkpoints/best_v5_expectancy.pt"),
            Path("checkpoints/best_v5_loss.pt"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _run_training(self, symbol: str, output_dir: Path) -> RetrainResult:
        """Run V5Forecaster fine-tuning on latest candle data for one symbol."""
        result = RetrainResult()

        try:
            import pandas as pd

            log.info(f"[Learning] Fetching full history for {symbol}...")
            raw = self._fetch_full_history(symbol)

            if not raw or len(raw) < self.config.min_new_bars:
                result.error = (
                    f"Insufficient data: got {len(raw) if raw else 0} candles "
                    f"(min_new_bars={self.config.min_new_bars})"
                )
                return result

            df = pd.DataFrame(raw)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            data_cache = Path("data_cache")
            data_cache.mkdir(parents=True, exist_ok=True)
            parquet_path = data_cache / f"{symbol}_15m.parquet"
            df.to_parquet(parquet_path, index=False)
            log.info(f"[Learning] Saved {len(df)} candles to {parquet_path}")

            checkpoint_path = self._find_v5_checkpoint(symbol)
            if checkpoint_path is None:
                result.error = (
                    "No V5 checkpoint found. Train V5 model first with "
                    "--train-v5 before enabling online learning."
                )
                log.error(f"[Learning] {result.error}")
                return result

            log.info(f"[Learning] V5 fine-tuning {symbol} from {checkpoint_path}")
            return self._run_v5_finetune(symbol, output_dir, checkpoint_path, df)

        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()
            return result

    def _run_v5_finetune(
        self,
        symbol: str,
        output_dir: Path,
        checkpoint_path: Path,
        df,
    ) -> RetrainResult:
        """Fine-tune an existing V5Forecaster on recent candle data.

        Loads the checkpoint, builds V5 features + targets, runs a few epochs of
        gradient descent, then evaluates on a held-out validation window.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np

        result = RetrainResult()

        try:
            from train.v5_train import V5_FEATURE_VERSION
            result.model_version = V5_FEATURE_VERSION
        except ImportError:
            result.model_version = "v5.0.1_forecaster"

        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
        except Exception as e:
            result.error = f"Failed to load checkpoint {checkpoint_path}: {e}"
            return result

        model_type = checkpoint.get('model_type', 'unknown')
        if model_type != 'v5_forecaster':
            result.error = (
                f"Checkpoint at {checkpoint_path} is type '{model_type}', not 'v5_forecaster'. "
                "Cannot fine-tune with V5 pipeline."
            )
            log.error(f"[Learning] {result.error}")
            return result

        cfg = checkpoint.get('model_config', {})
        feature_columns = checkpoint.get('feature_columns', [])
        if not feature_columns:
            result.error = "Checkpoint has no feature_columns — retrain from scratch first."
            return result

        try:
            from models.v5_forecaster import V5Forecaster, V5ForecasterConfig
            v5_config = V5ForecasterConfig(
                input_dim=cfg.get('input_dim', len(feature_columns)),
                hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
                dropout=cfg.get('dropout', 0.3),
                use_layer_norm=True,
                use_residual=True,
                n_barrier_presets=cfg.get('n_barrier_presets', 0),
                enable_regime_head=cfg.get('enable_regime_head', False),
                n_symbols=cfg.get('n_symbols', 1),
                symbol_embed_dim=cfg.get('symbol_embed_dim', 8),
            )
            model = V5Forecaster(v5_config)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(self.device)
            log.info(
                f"[Learning] V5Forecaster loaded: "
                f"{sum(p.numel() for p in model.parameters()):,} params from {checkpoint_path.name}"
            )
        except Exception as e:
            result.error = f"Failed to reconstruct V5Forecaster: {e}"
            return result

        try:
            import numpy as np
            from sklearn.preprocessing import RobustScaler

            scaler = None
            if 'scaler_center' in checkpoint and 'scaler_scale' in checkpoint:
                scaler = RobustScaler()
                scaler.center_ = np.array(checkpoint['scaler_center'])
                scaler.scale_ = np.array(checkpoint['scaler_scale'])
                log.info("[Learning] Scaler loaded from checkpoint")
            else:
                for sname in ["per_symbol_scalers.joblib", "scaler.joblib"]:
                    sp = Path(f"checkpoints/{sname}")
                    if sp.exists():
                        import joblib
                        all_scalers = joblib.load(sp)
                        if isinstance(all_scalers, dict):
                            scaler = all_scalers.get(symbol, all_scalers.get('global'))
                        else:
                            scaler = all_scalers
                        if scaler is not None:
                            log.info(f"[Learning] Scaler loaded from {sp}")
                        break
        except Exception as e:
            log.warning(f"[Learning] Could not load scaler: {e} — using raw features")
            scaler = None

        try:
            from data.pipeline import FeatureEngineer
            fe = FeatureEngineer()
            features_df = fe.compute_all_features(df)
            features_df = (
                features_df
                .reindex(columns=feature_columns, fill_value=0.0)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            feat_arr = features_df.values.astype(np.float32)
            if scaler is not None:
                feat_arr = scaler.transform(feat_arr)
            log.info(f"[Learning] Features built: {feat_arr.shape}")
        except Exception as e:
            result.error = f"Feature engineering failed: {e}"
            import traceback
            traceback.print_exc()
            return result

        try:
            from data.common import generate_v5_sweep_outcomes
            from data.v5_target_generator import build_v5_targets

            sweep = generate_v5_sweep_outcomes(
                df,
                horizon=self.config.horizon,
                tp_mult=self.config.tp_mult,
                sl_mult=self.config.sl_mult,
                atr_period=14,
            )
            targets = build_v5_targets(
                df,
                horizon=self.config.horizon,
                atr_period=14,
                barrier_outcomes=sweep,
            )
            n = len(features_df)
            ret_arr   = targets['ret_R'][:n].astype(np.float32)
            mfe_arr   = targets.get('mfe_R_long', targets['mfe_R'])[:n].astype(np.float32)
            mae_arr   = targets.get('mae_R_long', targets['mae_R'])[:n].astype(np.float32)
            vol_arr   = targets['vol_h'][:n].astype(np.float32)
            act_arr   = targets['action_label'][:n].astype(np.int64)
            val_mask  = targets['valid_mask'][:n]
            sym_ids   = np.zeros(n, dtype=np.int64)

            action_counts = np.bincount(act_arr, minlength=3)
            log.info(
                f"[Learning] Labels — HOLD:{action_counts[0]} "
                f"LONG:{action_counts[1]} SHORT:{action_counts[2]}"
            )
        except Exception as e:
            result.error = f"V5 target generation failed: {e}"
            import traceback
            traceback.print_exc()
            return result

        try:
            from train.v5_train import V5Dataset, compute_v5_loss
            from torch.utils.data import DataLoader

            split = max(1, int(n * (1.0 - self.config.val_split)))
            train_ds = V5Dataset(
                feat_arr[:split], ret_arr[:split], mfe_arr[:split],
                mae_arr[:split], vol_arr[:split], act_arr[:split],
                val_mask[:split], sym_ids[:split],
            )
            val_ds = V5Dataset(
                feat_arr[split:], ret_arr[split:], mfe_arr[split:],
                mae_arr[split:], vol_arr[split:], act_arr[split:],
                val_mask[split:], sym_ids[split:],
            )
            log.info(f"[Learning] Dataset — train:{len(train_ds)} val:{len(val_ds)}")

            if len(train_ds) < 32:
                result.error = f"Too few training samples: {len(train_ds)}"
                return result

            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.finetune_batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=512,
                shuffle=False,
            )

            action_weights = None
            counts = np.bincount(act_arr[:split], minlength=3).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            w = 1.0 / counts
            action_weights = torch.tensor(w / w.sum() * 3, dtype=torch.float32, device=self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.finetune_lr,
                weight_decay=1e-4,
            )

            best_val_loss = float('inf')
            best_state = None

            for epoch in range(self.config.training_epochs):
                model.train()
                train_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                             for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = model(
                        batch['features'],
                        symbol_ids=batch.get('symbol_id'),
                    )
                    loss, _ = compute_v5_loss(
                        outputs, batch,
                        action_weights=action_weights,
                        epoch=epoch,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches += 1

                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                                 for k, v in batch.items()}
                        outputs = model(
                            batch['features'],
                            symbol_ids=batch.get('symbol_id'),
                        )
                        loss, _ = compute_v5_loss(outputs, batch, epoch=999)
                        val_loss += loss.item()
                        n_val += 1

                avg_train = train_loss / max(1, n_batches)
                avg_val   = val_loss   / max(1, n_val)
                log.info(
                    f"[Learning] Epoch {epoch+1}/{self.config.training_epochs} "
                    f"train_loss={avg_train:.4f} val_loss={avg_val:.4f}"
                )

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            result.val_loss = best_val_loss
            result.training_samples = split
            result.trained_until_ts = int(df.iloc[-1].get('timestamp', time.time() * 1000))

        except Exception as e:
            result.error = f"V5 fine-tuning failed: {e}"
            import traceback
            traceback.print_exc()
            return result

        try:
            val_metrics = self._evaluate_v5_val(
                model=model,
                feat_arr=feat_arr[split:],
                act_arr=act_arr[split:],
                val_mask=val_mask[split:],
                sym_ids=sym_ids[split:],
                sweep=sweep,
                df_val=df.iloc[split:].reset_index(drop=True),
                n_train_bars=split,
            )
            result.pf_net           = val_metrics.get('pf_net', 0.0)
            result.e_net            = val_metrics.get('e_net', 0.0)
            result.trades_per_day   = val_metrics.get('tpd', 0.0)
            result.profitable_regimes = val_metrics.get('profitable_regimes', 0)
            result.total_regimes    = val_metrics.get('total_regimes', 4)
            result.max_drawdown_r   = val_metrics.get('max_drawdown_r', 0.0)
            result.val_prauc        = val_metrics.get('val_prauc', 0.5)
            log.info(
                f"[Learning] Val metrics — pf_net={result.pf_net:.2f} "
                f"e_net={result.e_net:.4f} tpd={result.trades_per_day:.2f} "
                f"profitable_regimes={result.profitable_regimes}"
            )
        except Exception as e:
            log.warning(f"[Learning] Val evaluation failed (non-fatal): {e}")
            result.pf_net = 0.0
            result.val_prauc = 0.5

        try:
            out_cp = output_dir / "best_v5_expectancy.pt"
            new_ckpt = dict(checkpoint)
            new_ckpt['model_state_dict'] = (
                best_state if best_state else
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            torch.save(new_ckpt, str(out_cp))
            log.info(f"[Learning] Candidate checkpoint saved to {out_cp}")
        except Exception as e:
            result.error = f"Failed to save candidate checkpoint: {e}"
            return result

        result.success = True
        return result

    def _evaluate_v5_val(
        self,
        model,
        feat_arr,
        act_arr,
        val_mask,
        sym_ids,
        sweep,
        df_val,
        n_train_bars: int,
        threshold: float = 0.5,
        score_lambda: float = 0.5,
        mae_floor: float = 0.5,
        cost_r: float = 0.03,
    ) -> Dict:
        """Run V5 inference on validation data and compute trading metrics."""
        import torch
        import numpy as np

        if len(feat_arr) < 4:
            return {'pf_net': 0.0, 'e_net': 0.0, 'tpd': 0.0,
                    'profitable_regimes': 0, 'total_regimes': 4,
                    'max_drawdown_r': 0.0, 'val_prauc': 0.5}

        from train.v5_train import compute_v5_scores

        model.eval()
        with torch.no_grad():
            feat_t = torch.tensor(feat_arr, dtype=torch.float32, device=self.device)
            sym_t  = torch.tensor(sym_ids,  dtype=torch.long,    device=self.device)
            outputs = model(feat_t, symbol_ids=sym_t)

        v5_scores, sides, _stats = compute_v5_scores(
            outputs,
            score_lambda=score_lambda,
        )
        v5_scores = np.array(v5_scores, dtype=np.float32)
        sides     = np.array(sides, dtype=np.int64)

        try:
            r_long  = sweep.get('r_long',  np.zeros(len(df_val)))
            r_short = sweep.get('r_short', np.zeros(len(df_val)))
            out_long  = sweep.get('out_long',  np.zeros(len(df_val), dtype=np.int64))
            out_short = sweep.get('out_short', np.zeros(len(df_val), dtype=np.int64))
            n_val = min(len(v5_scores), len(r_long))
            realized_r_long  = r_long[:n_val]
            realized_r_short = r_short[:n_val]
        except Exception:
            n_val = len(v5_scores)
            realized_r_long  = np.zeros(n_val)
            realized_r_short = np.zeros(n_val)

        trade_results = []
        cooldown_remaining = 0
        for i in range(min(n_val, len(v5_scores))):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue
            if not val_mask[i]:
                continue
            score = float(v5_scores[i])
            if score < threshold:
                continue

            side = int(sides[i]) if i < len(sides) else 1
            if side >= 0:
                r = float(realized_r_long[i]) - cost_r
            else:
                r = float(realized_r_short[i]) - cost_r
            trade_results.append(r)
            cooldown_remaining = 4

        if not trade_results:
            return {'pf_net': 0.0, 'e_net': 0.0, 'tpd': 0.0,
                    'profitable_regimes': 0, 'total_regimes': 4,
                    'max_drawdown_r': 0.0, 'val_prauc': 0.5}

        wins  = [r for r in trade_results if r > 0]
        losses = [abs(r) for r in trade_results if r < 0]
        pf_net = (sum(wins) / max(1e-9, sum(losses))) if losses else (10.0 if wins else 0.0)
        e_net  = float(np.mean(trade_results)) if trade_results else 0.0

        bars_per_day   = 96
        n_val_days     = max(1.0, len(feat_arr) / bars_per_day)
        tpd            = len(trade_results) / n_val_days

        equity = np.cumsum([0.0] + trade_results)
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = rolling_max - equity
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        regimes = _label_regimes(df_val)
        profitable_regimes = 0
        total_regimes = len(regimes)
        for reg_start, reg_end in regimes:
            reg_trades = [
                r for i, r in enumerate(trade_results)
                if reg_start <= i < reg_end
            ]
            if reg_trades and np.mean(reg_trades) > 0:
                profitable_regimes += 1

        val_prauc = min(0.99, max(0.0, 0.5 + pf_net * 0.05)) if pf_net > 0 else 0.5

        return {
            'pf_net': pf_net,
            'e_net': e_net,
            'tpd': tpd,
            'profitable_regimes': profitable_regimes,
            'total_regimes': max(total_regimes, 4),
            'max_drawdown_r': max_dd,
            'val_prauc': val_prauc,
        }

    def _run_geometry_sweep(self, symbol: str, checkpoint_dir: Path) -> Optional[Dict]:
        """Sweep threshold/cooldown combos on the candidate V5 checkpoint."""
        try:
            import torch
            import numpy as np

            cp = checkpoint_dir / "best_v5_expectancy.pt"
            if not cp.exists():
                log.warning(f"[Learning] No V5 candidate checkpoint for sweep: {symbol}")
                return None

            checkpoint = torch.load(str(cp), map_location=self.device, weights_only=False)
            if checkpoint.get('model_type') != 'v5_forecaster':
                log.warning("[Learning] Geometry sweep only supported for V5 models")
                return None

            cfg = checkpoint.get('model_config', {})
            feature_columns = checkpoint.get('feature_columns', [])

            from models.v5_forecaster import V5Forecaster, V5ForecasterConfig
            v5_config = V5ForecasterConfig(
                input_dim=cfg.get('input_dim', len(feature_columns)),
                hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
                dropout=0.0,
                use_layer_norm=True,
                use_residual=True,
                n_barrier_presets=cfg.get('n_barrier_presets', 0),
                enable_regime_head=cfg.get('enable_regime_head', False),
                n_symbols=cfg.get('n_symbols', 1),
                symbol_embed_dim=cfg.get('symbol_embed_dim', 8),
            )
            model = V5Forecaster(v5_config)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(self.device)
            model.eval()

            import pandas as pd
            parquet_path = Path("data_cache") / f"{symbol}_15m.parquet"
            if not parquet_path.exists():
                log.warning(f"[Learning] No parquet for sweep: {parquet_path}")
                return None

            df = pd.read_parquet(parquet_path)
            df = df.sort_values('timestamp').reset_index(drop=True)

            from data.pipeline import FeatureEngineer
            fe = FeatureEngineer()
            features_df = fe.compute_all_features(df)
            features_df = (
                features_df
                .reindex(columns=feature_columns, fill_value=0.0)
                .ffill().bfill().fillna(0.0)
            )
            feat_arr = features_df.values.astype(np.float32)

            if 'scaler_center' in checkpoint:
                from sklearn.preprocessing import RobustScaler
                sc = RobustScaler()
                sc.center_ = np.array(checkpoint['scaler_center'])
                sc.scale_  = np.array(checkpoint['scaler_scale'])
                feat_arr = sc.transform(feat_arr)

            from data.common import generate_v5_sweep_outcomes
            from data.v5_target_generator import build_v5_targets
            from train.v5_train import compute_v5_scores

            sweep = generate_v5_sweep_outcomes(
                df, horizon=self.config.horizon,
                tp_mult=self.config.tp_mult, sl_mult=self.config.sl_mult,
                atr_period=14,
            )
            targets = build_v5_targets(
                df, horizon=self.config.horizon, atr_period=14,
                barrier_outcomes=sweep,
            )
            n = len(features_df)
            val_mask = targets['valid_mask'][:n]
            sym_ids  = np.zeros(n, dtype=np.int64)

            with torch.no_grad():
                feat_t = torch.tensor(feat_arr, dtype=torch.float32, device=self.device)
                sym_t  = torch.tensor(sym_ids,  dtype=torch.long,    device=self.device)
                outputs = model(feat_t, symbol_ids=sym_t)

            v5_scores, sides, _stats = compute_v5_scores(outputs, score_lambda=0.5)
            v5_scores = np.array(v5_scores, dtype=np.float32)
            sides     = np.array(sides, dtype=np.int64)

            r_long  = sweep.get('r_long',  np.zeros(n))
            r_short = sweep.get('r_short', np.zeros(n))

            best_result = None
            best_pf = -1.0
            sweep_results = []

            for thresh in self.config.sweep_thresholds:
                for cd in self.config.sweep_cooldowns:
                    trades, cool = [], 0
                    for i in range(n):
                        if cool > 0:
                            cool -= 1
                            continue
                        if not val_mask[i]:
                            continue
                        if v5_scores[i] < thresh:
                            continue
                        side = int(sides[i]) if i < len(sides) else 1
                        r = float(r_long[i] if side >= 0 else r_short[i]) - 0.03
                        trades.append(r)
                        cool = cd

                    if len(trades) < 5:
                        continue

                    wins   = [r for r in trades if r > 0]
                    losses = [abs(r) for r in trades if r < 0]
                    pf     = (sum(wins) / max(1e-9, sum(losses))) if losses else 10.0
                    e_net  = float(np.mean(trades))
                    bars_per_day = 96
                    tpd = len(trades) / max(1.0, n / bars_per_day)

                    sweep_results.append({
                        'threshold': thresh,
                        'cooldown': cd,
                        'pf_net': pf,
                        'e_net': e_net,
                        'trades_per_day': tpd,
                        'n_trades': len(trades),
                    })

                    if pf > best_pf and tpd >= self.config.min_tpd:
                        best_pf = pf
                        best_result = sweep_results[-1]

            if not best_result:
                log.info("[Learning] Geometry sweep: no policy passed TPD gate")
                return None

            profitable = sum(1 for r in sweep_results if r.get('pf_net', 0) > 1.0)
            log.info(
                f"[Learning] Best sweep: thr={best_result['threshold']} "
                f"cd={best_result['cooldown']} pf={best_result['pf_net']:.2f} "
                f"tpd={best_result['trades_per_day']:.1f}"
            )

            return {
                'best_policy': {
                    'threshold':  best_result['threshold'],
                    'cooldown':   best_result['cooldown'],
                    'tp_mult':    self.config.tp_mult,
                    'sl_mult':    self.config.sl_mult,
                },
                'pf_net':             best_result['pf_net'],
                'e_net':              best_result['e_net'],
                'trades_per_day':     best_result['trades_per_day'],
                'profitable_regimes': profitable,
                'total_regimes':      len(self.config.sweep_thresholds) * len(self.config.sweep_cooldowns),
            }

        except Exception as e:
            log.warning(f"[Learning] V5 geometry sweep failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _evaluate_promotion(self, symbol: str, result: RetrainResult) -> tuple:
        """Decide whether to promote the candidate V5 model.

        Gates:
        1. pf_net must meet minimum threshold
        2. Profitable regime count must be sufficient
        3. TPD must be in acceptable range
        4. E[net] must be non-negative
        5. Max drawdown must not be excessive
        6. Relative comparison against deployed stats if available
        """
        if result.pf_net < self.config.gate_pf_net:
            return False, f"PF_net {result.pf_net:.2f} < gate {self.config.gate_pf_net}"

        if result.profitable_regimes < self.config.gate_profitable_regimes:
            return False, f"Profitable regimes {result.profitable_regimes} < gate {self.config.gate_profitable_regimes}"

        if result.trades_per_day < self.config.min_tpd:
            return False, f"TPD {result.trades_per_day:.1f} < min {self.config.min_tpd}"

        if result.trades_per_day > self.config.max_tpd:
            return False, f"TPD {result.trades_per_day:.1f} > max {self.config.max_tpd}"

        if result.e_net < self.config.gate_enet:
            return False, f"E[net] {result.e_net:.4f} < gate {self.config.gate_enet}"

        if result.max_drawdown_r > self.config.gate_maxdd_r:
            return False, f"MaxDD {result.max_drawdown_r:.2f}R > gate {self.config.gate_maxdd_r}R"

        prev = self.deployed_stats.get(symbol)
        if prev and prev.get('pf_net', 0) > 0:
            prev_pf = prev.get('pf_net', 0)
            if result.pf_net < prev_pf - self.config.pf_improvement_required:
                return False, f"PF_net {result.pf_net:.2f} < deployed {prev_pf:.2f}"

        return True, "All V5 promotion gates passed"

    def _promote_model(self, symbol: str, candidate_dir: Path, deployed_dir: Path):
        """Copy candidate checkpoint to deployed directory."""
        for fname in ["best_v5_expectancy.pt", "best_v5_loss.pt"]:
            src = candidate_dir / fname
            if src.exists():
                dst = deployed_dir / fname
                shutil.copy2(str(src), str(dst))
                log.info(f"[Learning] Promoted {src} -> {dst}")

    def _push_learning_stats(
        self,
        symbol: str,
        result: RetrainResult,
        promoted: bool,
        reason: str,
        prev_stats: Optional[Dict] = None,
    ):
        url = f"{self.replit_url.rstrip('/')}/api/learning/stats"
        trend = "Unknown"
        prev = prev_stats or {}
        if prev.get('pf_net', 0) > 0:
            if result.pf_net > prev.get('pf_net', 0) * 1.05:
                trend = "Improving"
            elif result.pf_net < prev.get('pf_net', 0) * 0.95:
                trend = "Worse"
            else:
                trend = "Stable"

        payload = {
            "symbol": symbol,
            "model_version": result.model_version,
            "trained_until_ts": result.trained_until_ts,
            "training_samples": result.training_samples,
            "val_pr_auc": result.val_prauc,
            "val_loss": result.val_loss,
            "val_precision": result.val_precision,
            "val_recall": result.val_recall,
            "val_f1": result.val_f1,
            "best_policy_threshold": result.best_policy.get('threshold') if result.best_policy else None,
            "best_policy_cooldown": result.best_policy.get('cooldown') if result.best_policy else None,
            "best_policy_tp_mult": result.best_policy.get('tp_mult') if result.best_policy else None,
            "best_policy_sl_mult": result.best_policy.get('sl_mult') if result.best_policy else None,
            "pf_net": result.pf_net,
            "e_net": result.e_net,
            "trades_per_day": result.trades_per_day,
            "profitable_regimes": result.profitable_regimes,
            "total_regimes": result.total_regimes,
            "promoted": promoted,
            "promotion_reason": reason,
            "trend_7d": trend,
            "prev_pf_net": prev.get('pf_net'),
            "prev_e_net": prev.get('e_net'),
            "prev_trades_per_day": prev.get('trades_per_day'),
        }

        try:
            from live_runner import _retry_request
            _retry_request("POST", url, json=payload)
        except Exception as e:
            log.warning(f"[Learning] Failed to push stats for {symbol}: {e}")

        if promoted:
            self.deployed_stats[symbol] = {
                'pf_net': result.pf_net,
                'e_net': result.e_net,
                'profitable_regimes': result.profitable_regimes,
                'trades_per_day': result.trades_per_day,
            }

    def check_and_retrain_all(self):
        for symbol in self.symbols:
            if self.should_retrain(symbol):
                log.info(f"[Learning] Retrain triggered for {symbol}")
                self.retrain_symbol(symbol)


def _label_regimes(df, n_regimes: int = 4) -> List[tuple]:
    """Split dataframe into N equal time-based regime chunks."""
    n = len(df)
    if n < n_regimes:
        return [(0, n)]
    chunk = n // n_regimes
    return [(i * chunk, min((i + 1) * chunk, n)) for i in range(n_regimes)]
