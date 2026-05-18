"""Multi-asset live inference loop (v5.0 — V5 Composite Scoring Engine).

Monitors multiple symbols in parallel on 15m intervals, runs the V5Forecaster
multi-head inference (action probs, ret_mu, MFE, MAE), computes composite
score using the training formula, and enters trades when score >= threshold.

v5.0 scoring:
  - V5 composite score: edge(p_side × |mu|/risk) - λ × penalty
  - Side determined by model (edge_long vs edge_short)
  - Score threshold (default 0.02) + min |ret_mu| (default 0.03)
  - No lane routing — pure model-driven decisions

Usage:
    python quick_start.py --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,XRPUSDT,ADAUSDT
"""

import sys
import time
import json
import copy
import logging
import traceback
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib

log = logging.getLogger("LiveRunner")


SYSTEM_VERSION = "v5.0_composite_scoring"

REQUIRED_CANDLES = 800
MAX_CACHE_BARS = 2000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0
OPEN_POSITION_MONITOR_INTERVAL_S = 4.0

MIN_H1_BARS = 100
MIN_H4_BARS = 50

V5_SCORE_LAMBDA = 0.5
V5_SCORE_THRESHOLD = 0.5   # default live threshold — calibrated for MAE floor of 0.5
V5_MIN_MU_R = 0.03
V5_MAE_FLOOR = 0.5         # minimum MAE to prevent score explosion in low-vol markets

COST_BPS = 8.0
LIVE_AWARENESS_WINDOW = 40
LIVE_AWARENESS_MIN_TRADES = 8
LIVE_AWARENESS_MIN_SIDE_TRADES = 4


def _normalize_dashboard_engine(engine: Optional[str]) -> str:
    raw = str(engine or "").strip().lower()
    if not raw:
        return "v5"
    if "myth" in raw:
        return "mythos"
    if raw in {"v5", "v5_forecaster", "forecaster"}:
        return "v5"
    return raw


def _is_exchange_api_url(url: Optional[str]) -> bool:
    txt = str(url or "").strip()
    if not txt:
        return False
    parsed = urlparse(txt if "://" in txt else f"https://{txt}")
    host = str(parsed.netloc or parsed.path or "").strip().lower()
    if not host:
        return False
    exchange_hosts = (
        "api.binance.com",
        "api-gcp.binance.com",
        "api1.binance.com",
        "api2.binance.com",
        "api3.binance.com",
        "binance.vision",
    )
    return any(token in host for token in exchange_hosts)


def _retry_request(method: str, url: str, **kwargs) -> Optional[requests.Response]:
    timeout = kwargs.pop('timeout', 15)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code < 500:
                return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning(f"Request {method} {url} attempt {attempt+1}/{RETRY_ATTEMPTS} failed: {e}")
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    log.error(f"All {RETRY_ATTEMPTS} attempts failed for {method} {url}")
    return None


def _get_exchange_time_offset() -> float:
    try:
        resp = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        if resp.status_code == 200:
            server_ms = resp.json()["serverTime"]
            local_ms = time.time() * 1000
            offset_ms = server_ms - local_ms
            if abs(offset_ms) > 1000:
                log.warning(f"Clock offset vs Binance: {offset_ms:.0f}ms")
            return offset_ms / 1000.0
        return 0.0
    except Exception:
        return 0.0


def _load_model(device: str, symbol: Optional[str] = None, model_backend: str = "v5"):
    """Load the trained model, scaler, feature columns, and temperature.

    Supports:
      - v5/v6/legacy checkpoints (.pt)
      - mythos runtime artifact (.json)

    If symbol is provided, first checks checkpoints/deployed/{symbol}/ for a
    per-symbol model. Falls back to the global checkpoints/ directory.

    Returns: (model, engineer, feature_columns, temperature, symbol_map)
    """
    backend = str(model_backend or "v5").strip().lower()
    if "myth" in backend:
        from mythos.runtime import MythosRuntimeModel

        candidates = []
        if symbol:
            sym = str(symbol).upper()
            candidates.extend(
                [
                    Path(f"checkpoints/deployed/{sym}/mythos_best_{sym}.json"),
                    Path(f"checkpoints/mythos_models/mythos_best_{sym}.json"),
                    Path(f"checkpoints/mythos_best_{sym}.json"),
                ]
            )
        candidates.extend(
            [
                Path("checkpoints/mythos_models/mythos_best_BTCUSDT.json"),
                Path("checkpoints/mythos_best_BTCUSDT.json"),
            ]
        )
        artifact_path = next((p for p in candidates if p.exists()), None)
        if artifact_path is None:
            log.error(
                "No Mythos artifact found%s. Expected one of: %s",
                f" for {symbol}" if symbol else "",
                ", ".join(str(p) for p in candidates),
            )
            sys.exit(1)
        model = MythosRuntimeModel.from_artifact(artifact_path)
        log.info(f"Loaded Mythos runtime model from {artifact_path}")
        return model, None, list(getattr(model, "feature_columns", [])), 1.0, None

    import torch
    from data.pipeline import FeatureEngineer

    search_dirs = []
    if symbol:
        search_dirs.append(Path(f"checkpoints/deployed/{symbol}"))
    search_dirs.append(Path("checkpoints"))

    checkpoint_names = [
        "best_enter_prauc.pt",
        "best_v5_expectancy.pt",
        "best_enter_loss.pt",
        "best_v5_loss.pt",
    ]

    checkpoint_path = None
    for d in search_dirs:
        for name in checkpoint_names:
            p = d / name
            if p.exists():
                checkpoint_path = p
                break
        if checkpoint_path:
            break

    if not checkpoint_path:
        log.error(f"No trained model found{' for '+symbol if symbol else ''}! Run training first.")
        sys.exit(1)

    log.info(f"Loading model from {checkpoint_path}{' ('+symbol+')' if symbol else ''}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_version = checkpoint.get('feature_version', 'unknown')
    from quick_start import FEATURE_VERSION
    ACCEPTED_VERSIONS = {FEATURE_VERSION, "v5.0.1_forecaster"}
    if saved_version not in ACCEPTED_VERSIONS:
        log.error(f"Feature version mismatch! Model: '{saved_version}', accepted: {ACCEPTED_VERSIONS}")
        sys.exit(1)

    feature_columns = checkpoint.get('feature_columns', [])
    if not feature_columns:
        log.error("No feature_columns in checkpoint — retrain.")
        sys.exit(1)

    cfg = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'legacy')

    if model_type == 'v6_forecaster':
        from models.v6_forecaster import V6Forecaster, V6ForecasterConfig
        v6_config = V6ForecasterConfig(
            input_dim=cfg.get('input_dim', 85),
            seq_len=cfg.get('seq_len', 16),
            conv_channels=cfg.get('conv_channels', 128),
            n_conv_layers=cfg.get('n_conv_layers', 3),
            conv_kernel_size=cfg.get('conv_kernel_size', 3),
            n_attn_layers=cfg.get('n_attn_layers', 2),
            n_attn_heads=cfg.get('n_attn_heads', 4),
            attn_ff_dim=cfg.get('attn_ff_dim', 256),
            n_experts=cfg.get('n_experts', 4),
            expert_top_k=cfg.get('expert_top_k', 2),
            expert_hidden_dims=cfg.get('expert_hidden_dims', [192, 128, 96]),
            trunk_output_dim=cfg.get('trunk_output_dim', 96),
            dropout=cfg.get('dropout', 0.15),
            n_symbols=cfg.get('n_symbols', 1),
            symbol_embed_dim=cfg.get('symbol_embed_dim', 16),
            feature_mask_ratio=0.0,
            enable_aux_head=False,
            enable_confidence_head=cfg.get('enable_confidence_head', True),
            n_barrier_presets=cfg.get('n_barrier_presets', 0),
            enable_regime_head=cfg.get('enable_regime_head', False),
        )
        model = V6Forecaster(v6_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model._is_v5 = True
        model._is_v6 = True
        model._v6_seq_len = v6_config.seq_len
        log.info(f"Loaded V6Forecaster ({model.parameters_count():,} params, seq_len={v6_config.seq_len})")
    elif model_type == 'v5_forecaster':
        from models.v5_forecaster import V5Forecaster, V5ForecasterConfig
        v5_config = V5ForecasterConfig(
            input_dim=cfg.get('input_dim', 85),
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
        model._is_v5 = True
        model._is_v6 = False
        log.info(f"Loaded V5Forecaster ({model.parameters_count():,} params)")
    else:
        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        n_symbols = cfg.get('n_symbols', 1)
        symbol_embed_dim = cfg.get('symbol_embed_dim', 0)
        enable_value_head = cfg.get('enable_value_head', False)
        enable_edge_head = cfg.get('enable_edge_head', False)
        mlp_config = EnhancedMultiHeadMLP_Config(
            input_dim=cfg.get('input_dim', 63),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            num_classes=3,
            dropout=0.3,
            use_layer_norm=True,
            use_residual=True,
            enable_enter_head=True,
            enable_quantile_head=False,
            enable_vol_state_head=False,
            enable_mu_head=False,
            enable_sigma_head=False,
            enable_value_head=enable_value_head,
            enable_edge_head=enable_edge_head,
            n_symbols=n_symbols,
            symbol_embed_dim=symbol_embed_dim,
        )
        model = EnhancedMultiHeadMLP(mlp_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model._is_v5 = False
        log.info(f"Loaded EnhancedMultiHeadMLP")

    model.to(device)
    model.eval()

    engineer = FeatureEngineer()
    scaler_loaded = False

    if model_type in ('v5_forecaster', 'v6_forecaster') and 'scaler_center' in checkpoint and 'scaler_scale' in checkpoint:
        from sklearn.preprocessing import RobustScaler
        global_scaler = RobustScaler()
        global_scaler.center_ = np.array(checkpoint['scaler_center'])
        global_scaler.scale_ = np.array(checkpoint['scaler_scale'])
        is_identity = (np.all(global_scaler.center_ == 0) and np.all(global_scaler.scale_ == 1))

        if 'per_symbol_scalers' in checkpoint and checkpoint['per_symbol_scalers']:
            per_sym = {}
            for sym, sdata in checkpoint['per_symbol_scalers'].items():
                s = RobustScaler()
                s.center_ = np.array(sdata.get('center_', sdata.get('center', [])))
                s.scale_ = np.array(sdata.get('scale_', sdata.get('scale', [])))
                per_sym[sym] = s
            engineer._per_symbol_scalers = per_sym
            scaler_loaded = True
            log.info(f"Per-symbol scalers loaded from checkpoint ({len(per_sym)} symbols)")
        elif not is_identity:
            engineer._v5_global_scaler = global_scaler
            scaler_loaded = True
            log.info("Global scaler loaded from checkpoint (embedded)")
        else:
            scaler_loaded = False
            log.info("Checkpoint has identity global scaler — looking for per-symbol scalers on disk")

    if not scaler_loaded:
        scaler_path = None
        for d in search_dirs:
            for sname in ["per_symbol_scalers.joblib", "scaler.joblib"]:
                sp = d / sname
                if sp.exists():
                    scaler_path = sp
                    break
            if scaler_path:
                break
        if scaler_path:
            if 'per_symbol' in str(scaler_path):
                from sklearn.preprocessing import RobustScaler as _RS
                raw_scalers = joblib.load(str(scaler_path))
                if isinstance(raw_scalers, dict) and raw_scalers:
                    first_val = next(iter(raw_scalers.values()))
                    if hasattr(first_val, 'center_'):
                        engineer._per_symbol_scalers = raw_scalers
                        scaler_loaded = True
                        log.info(f"Per-symbol scalers loaded from {scaler_path} ({len(raw_scalers)} symbols)")
                    else:
                        engineer.load_scalers(str(scaler_path))
                        scaler_loaded = True
                        log.info(f"Column scalers loaded from {scaler_path}")
                else:
                    engineer.load_scalers(str(scaler_path))
                    scaler_loaded = True
                    log.info(f"Scaler loaded from {scaler_path}")
            else:
                engineer.load_scalers(str(scaler_path))
                scaler_loaded = True
                log.info(f"Scaler loaded from {scaler_path}")
        if not scaler_loaded:
            log.warning("No saved scaler — prediction quality may be reduced")

    temperature = 1.0
    for d in search_dirs:
        tp = d / "temp_scale_v5.0.json"
        if tp.exists():
            try:
                with open(tp) as f:
                    temp_data = json.load(f)
                temperature = float(temp_data.get("temperature", 1.0))
                log.info(f"[INFER] using_temperature={temperature:.4f} (from {tp})")
            except Exception as e:
                log.warning(f"Failed to load temperature: {e}")
            break

    symbol_map = checkpoint.get('symbol_map', None)

    n_symbols = cfg.get('n_symbols', 1)
    log.info(f"Model loaded: {len(feature_columns)} features, version {saved_version}, "
             f"model_type={model_type}, n_symbols={n_symbols}, temperature={temperature:.4f}")
    return model, engineer, feature_columns, temperature, symbol_map


def _fetch_candles_for_symbol(fetcher, symbol: str, timeframe: str = "15m",
                               limit: int = REQUIRED_CANDLES) -> Optional[pd.DataFrame]:
    """Fetch recent candles for a symbol via the BinanceDataFetcher."""
    try:
        log.info(f"Fetching {symbol} {timeframe} (limit={limit}...)")
        raw = fetcher.fetch_klines_sync(symbol, timeframe, limit=limit)
        if not raw or len(raw) < 100:
            log.warning(f"Insufficient candles for {symbol}: got {len(raw) if raw else 0}")
            return None

        df = pd.DataFrame(raw)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(int)

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        log.error(f"Failed to fetch candles for {symbol}: {e}")
        return None


def _check_htf_warmup(df: pd.DataFrame, symbol: str) -> Optional[str]:
    """Check if there are enough candles to form valid HTF bars.

    Returns None if OK, or a WARMUP reason string if insufficient bars.
    Logs once per cycle when in warmup state.
    """
    if 'timestamp' not in df.columns:
        return f"{symbol} WARMUP: no timestamp column"

    ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    ohlcv = pd.DataFrame({
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'volume': df['volume'].values,
    }, index=ts)

    h1_bars = ohlcv.resample('1h', label='left', closed='left').agg({
        'open': 'first'
    }).dropna()
    h4_bars = ohlcv.resample('4h', label='left', closed='left').agg({
        'open': 'first'
    }).dropna()

    n_h1 = len(h1_bars)
    n_h4 = len(h4_bars)

    if n_h1 < MIN_H1_BARS or n_h4 < MIN_H4_BARS:
        msg = (f"{symbol} WARMUP: h1_bars={n_h1} h4_bars={n_h4} "
               f"(min {MIN_H1_BARS}/{MIN_H4_BARS}) -> skip gates/trading")
        return msg

    return None


def _fetch_htf_candles_direct(fetcher, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch 1H and 4H candles directly from the exchange instead of resampling.

    Returns dict with '1h' and '4h' DataFrames, or None on failure.
    """
    result = {}
    for tf, limit in [("1h", 300), ("4h", 200)]:
        try:
            log.info(f"Fetching {symbol} {tf} (limit={limit}, direct HTF...)")
            raw = fetcher.fetch_klines_sync(symbol, tf, limit=limit)
            if not raw or len(raw) < 10:
                log.warning(f"Insufficient direct {tf} candles for {symbol}: got {len(raw) if raw else 0}")
                return None
            df = pd.DataFrame(raw)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp').reset_index(drop=True)
            result[tf] = df
            log.info(f"  {symbol} {tf}: got {len(df)} bars")
        except Exception as e:
            log.error(f"Failed to fetch direct {tf} candles for {symbol}: {e}")
            return None
    return result


_FUNDING_CACHE_TTL = 1800
_OI_CACHE_TTL = 900
_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
_DATA_DIR.mkdir(exist_ok=True)


def _fetch_funding_cached(df: pd.DataFrame, cache: Dict, symbol: str) -> pd.DataFrame:
    """Fetch funding rate features with caching (TTL=30min). Uses BTCUSDT funding as market indicator."""
    from quick_start import (
        fetch_funding_rates, compute_funding_features,
        FUNDING_FEATURE_COUNT, FUNDING_FEATURE_NAMES,
    )
    n = len(df)
    now = time.time()
    cache_key = "BTCUSDT"

    if cache_key in cache and (now - cache[cache_key]["fetched_at"]) < _FUNDING_CACHE_TTL:
        funding_df = cache[cache_key]["data"]
    else:
        try:
            funding_df = fetch_funding_rates(df, _DATA_DIR)
            cache[cache_key] = {"data": funding_df, "fetched_at": now}
        except Exception as e:
            log.warning(f"[{symbol}] Funding fetch failed, using zeros: {e}")
            return pd.DataFrame(
                np.zeros((n, FUNDING_FEATURE_COUNT)),
                columns=FUNDING_FEATURE_NAMES,
                index=df.index,
            )

    try:
        return compute_funding_features(df, funding_df)
    except Exception as e:
        log.warning(f"[{symbol}] Funding feature computation failed, using zeros: {e}")
        return pd.DataFrame(
            np.zeros((n, FUNDING_FEATURE_COUNT)),
            columns=FUNDING_FEATURE_NAMES,
            index=df.index,
        )


def _fetch_oi_cached(df: pd.DataFrame, cache: Dict, symbol: str) -> pd.DataFrame:
    """Fetch open interest features with caching (TTL=15min). Per-symbol OI data."""
    from quick_start import (
        fetch_open_interest_hist, compute_oi_features,
        OI_FEATURE_COUNT, OI_FEATURE_NAMES,
    )
    n = len(df)
    now = time.time()

    if symbol in cache and (now - cache[symbol]["fetched_at"]) < _OI_CACHE_TTL:
        oi_df = cache[symbol]["data"]
    else:
        try:
            oi_df = fetch_open_interest_hist(df, _DATA_DIR, symbol=symbol)
            cache[symbol] = {"data": oi_df, "fetched_at": now}
        except Exception as e:
            log.warning(f"[{symbol}] OI fetch failed, using zeros: {e}")
            return pd.DataFrame(
                np.zeros((n, OI_FEATURE_COUNT)),
                columns=OI_FEATURE_NAMES,
                index=df.index,
            )

    try:
        return compute_oi_features(df, oi_df)
    except Exception as e:
        log.warning(f"[{symbol}] OI feature computation failed, using zeros: {e}")
        return pd.DataFrame(
            np.zeros((n, OI_FEATURE_COUNT)),
            columns=OI_FEATURE_NAMES,
            index=df.index,
        )


def _fetch_ls_ratio_cached(df: pd.DataFrame, cache: Dict, symbol: str) -> pd.DataFrame:
    """Fetch L/S ratio features with caching (TTL=15min). Per-symbol L/S data."""
    from quick_start import (
        fetch_ls_ratio_hist, compute_ls_ratio_features,
        LS_RATIO_FEATURE_COUNT, LS_RATIO_FEATURE_NAMES,
    )
    n = len(df)
    now = time.time()

    if symbol in cache and (now - cache[symbol]["fetched_at"]) < _OI_CACHE_TTL:
        ls_df = cache[symbol]["data"]
    else:
        try:
            ls_df = fetch_ls_ratio_hist(df, _DATA_DIR, symbol=symbol)
            cache[symbol] = {"data": ls_df, "fetched_at": now}
        except Exception as e:
            log.warning(f"[{symbol}] L/S ratio fetch failed, using zeros: {e}")
            return pd.DataFrame(
                np.zeros((n, LS_RATIO_FEATURE_COUNT)),
                columns=LS_RATIO_FEATURE_NAMES,
                index=df.index,
            )

    try:
        return compute_ls_ratio_features(df, ls_df)
    except Exception as e:
        log.warning(f"[{symbol}] L/S ratio feature computation failed, using zeros: {e}")
        return pd.DataFrame(
            np.zeros((n, LS_RATIO_FEATURE_COUNT)),
            columns=LS_RATIO_FEATURE_NAMES,
            index=df.index,
        )


def _log_feature_check(symbol: str, funding_features: pd.DataFrame,
                       oi_features: pd.DataFrame, logged: Dict):
    """One-time diagnostic log per symbol showing funding/OI feature values."""
    if logged.get(symbol):
        return
    logged[symbol] = True

    last_f = funding_features.iloc[-1]
    last_o = oi_features.iloc[-1]

    fr = last_f.get('funding_rate', 0)
    fd = last_f.get('funding_rate_delta_8h', 0)
    fz = last_f.get('funding_rate_zscore_30d', 0)
    oi = last_o.get('open_interest', 0)
    od = last_o.get('oi_delta_1h', 0)
    oz = last_o.get('oi_zscore_30d', 0)

    log.info(f"[Feature Check] {symbol}: funding_rate={fr:.6f} delta_8h={fd:.6f} "
             f"zscore_30d={fz:.4f} | OI={oi:.4f} oi_delta_1h={od:.4f} oi_zscore_30d={oz:.4f}")

    all_zero = abs(fr) < 1e-8 and abs(fd) < 1e-8 and abs(fz) < 1e-8
    oi_zero = abs(oi) < 1e-8 and abs(od) < 1e-8 and abs(oz) < 1e-8
    if all_zero:
        log.warning(f"[Feature Check] {symbol}: ALL funding features are zero — fetch may have failed")
    if oi_zero:
        log.warning(f"[Feature Check] {symbol}: ALL OI features are zero — fetch may have failed")


def _compute_features_for_symbol(df: pd.DataFrame, engineer, feature_columns: list,
                                  symbol: str,
                                  funding_cache: Optional[Dict] = None,
                                  oi_cache: Optional[Dict] = None,
                                  ls_ratio_cache: Optional[Dict] = None,
                                  feature_check_logged: Optional[Dict] = None,
                                  seq_len: int = 1) -> Optional[np.ndarray]:
    """Compute features for the latest bar(s) of a symbol's candle data.

    When seq_len=1 (default/V5): returns (1, n_features) scaled array for latest bar.
    When seq_len>1 (V6): returns (seq_len, n_features) scaled array for last seq_len bars,
    zero-padded at the start if fewer bars are available.

    Uses the same pipeline as make_enter_prediction: FeatureEngineer.compute_all_features,
    then real funding + OI features from Binance FAPI, then scale + clip.
    """
    from quick_start import (
        FUNDING_FEATURE_COUNT, OI_FEATURE_COUNT,
        FUNDING_FEATURE_NAMES, OI_FEATURE_NAMES,
    )

    if funding_cache is None:
        funding_cache = {}
    if oi_cache is None:
        oi_cache = {}
    if ls_ratio_cache is None:
        ls_ratio_cache = {}
    if feature_check_logged is None:
        feature_check_logged = {}

    try:
        feat_engineer_local = engineer.__class__()
        features_df = feat_engineer_local.compute_all_features(df)
        features_df = features_df.fillna(0)

        n = len(df)
        funding_features = _fetch_funding_cached(df, funding_cache, symbol)
        features_df = pd.concat([features_df, funding_features], axis=1)

        oi_features = _fetch_oi_cached(df, oi_cache, symbol)
        features_df = pd.concat([features_df, oi_features], axis=1)
        features_df = features_df.fillna(0)

        ls_features = _fetch_ls_ratio_cached(df, ls_ratio_cache, symbol)
        features_df = pd.concat([features_df, ls_features], axis=1)
        features_df = features_df.fillna(0)

        _log_feature_check(symbol, funding_features, oi_features, feature_check_logged)

        features_df = features_df.reindex(columns=feature_columns, fill_value=0)

        n_bars = min(seq_len, len(features_df))
        tail_features = features_df.iloc[-n_bars:].copy()

        if hasattr(engineer, '_per_symbol_scalers') and symbol in engineer._per_symbol_scalers:
            sym_scaler = engineer._per_symbol_scalers[symbol]
            raw = tail_features.values.astype(np.float32)
            tail_scaled = sym_scaler.transform(raw).astype(np.float32)
            tail_scaled = np.clip(tail_scaled, -5.0, 5.0)
        elif hasattr(engineer, '_per_symbol_scalers') and symbol not in engineer._per_symbol_scalers:
            log.warning(f"Per-symbol scalers loaded but {symbol} not found — falling back to global/column scaler")
            if hasattr(engineer, '_v5_global_scaler'):
                raw = tail_features.values.astype(np.float32)
                tail_scaled = engineer._v5_global_scaler.transform(raw).astype(np.float32)
                tail_scaled = np.clip(tail_scaled, -5.0, 5.0)
            else:
                tail_scaled = engineer.transform_and_clip(
                    pd.DataFrame(tail_features.values, columns=feature_columns),
                    clip_range=5.0
                ).values.astype(np.float32)
        elif hasattr(engineer, '_v5_global_scaler'):
            raw = tail_features.values.astype(np.float32)
            tail_scaled = engineer._v5_global_scaler.transform(raw).astype(np.float32)
            tail_scaled = np.clip(tail_scaled, -5.0, 5.0)
        else:
            tail_scaled = engineer.transform_and_clip(
                pd.DataFrame(tail_features.values, columns=feature_columns),
                clip_range=5.0
            ).values.astype(np.float32)
        tail_scaled = np.where(np.isinf(tail_scaled), 0, tail_scaled)
        tail_scaled = np.where(np.isnan(tail_scaled), 0, tail_scaled)

        if seq_len > 1 and n_bars < seq_len:
            pad = np.zeros((seq_len - n_bars, tail_scaled.shape[1]), dtype=np.float32)
            tail_scaled = np.concatenate([pad, tail_scaled], axis=0)

        if seq_len == 1:
            return tail_scaled, features_df
        else:
            return tail_scaled, features_df

    except Exception as e:
        log.error(f"Feature computation failed for {symbol}: {e}")
        traceback.print_exc()
        return None, None


def _run_inference(model, scaled_features: np.ndarray, device: str,
                   temperature: float = 1.0, symbol_id: Optional[int] = None) -> dict:
    """Run single-row or sequence model inference with temperature calibration.
    
    Supports V6Forecaster (3D seq input), V5Forecaster (2D single bar), and
    legacy EnhancedMultiHeadMLP.
    
    For V6: scaled_features is (seq_len, n_features) — passed as (1, seq_len, features).
    For V5: scaled_features is (1, n_features) — passed as (1, features).
    
    Returns dict with:
      p_enter, e_net_pred, enter_logit, v5_action_probs, v5_ret_mu, v5_mfe, v5_mae
      v6_confidence (V6 only): model's self-assessed prediction accuracy (0-1)
    """
    import torch
    is_v5 = getattr(model, '_is_v5', False)
    is_v6 = getattr(model, '_is_v6', False)

    with torch.no_grad():
        x = torch.FloatTensor(scaled_features).to(device)

        if is_v6:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            sym_ids = None
            if symbol_id is not None and model.symbol_embedding is not None:
                sym_ids = torch.tensor([symbol_id], dtype=torch.long, device=device)
            output = model(x, symbol_ids=sym_ids)

            action_logits = output['action_logits']
            calibrated_logits = action_logits / max(temperature, 0.01)
            action_probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy().flatten()

            p_hold = float(action_probs[0])
            p_enter = 1.0 - p_hold

            ret_mu = float(output['ret_mu'].cpu().item())
            mfe = float(output['mfe'].cpu().item())
            mae = float(output['mae'].cpu().item())
            confidence = float(output.get('confidence', torch.tensor(0.5)).cpu().item())

            enter_logit = float(action_logits[0, 1].cpu().item() - action_logits[0, 0].cpu().item())

            return {
                'p_enter': p_enter,
                'e_net_pred': ret_mu,
                'enter_logit': enter_logit,
                'temperature_used': temperature,
                'v5_action_probs': action_probs.tolist(),
                'v5_ret_mu': ret_mu,
                'v5_mfe': mfe,
                'v5_mae': mae,
                'v6_confidence': confidence,
            }
        elif is_v5:
            sym_ids = None
            if symbol_id is not None and model.symbol_embedding is not None:
                sym_ids = torch.tensor([symbol_id], dtype=torch.long, device=device)
            output = model(x, symbol_ids=sym_ids)

            action_logits = output['action_logits']
            calibrated_logits = action_logits / max(temperature, 0.01)
            action_probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy().flatten()

            p_hold = float(action_probs[0])
            p_enter = 1.0 - p_hold

            ret_mu = float(output['ret_mu'].cpu().item())
            mfe = float(output['mfe'].cpu().item())
            mae = float(output['mae'].cpu().item())

            enter_logit = float(action_logits[0, 1].cpu().item() - action_logits[0, 0].cpu().item())

            return {
                'p_enter': p_enter,
                'e_net_pred': ret_mu,
                'enter_logit': enter_logit,
                'temperature_used': temperature,
                'v5_action_probs': action_probs.tolist(),
                'v5_ret_mu': ret_mu,
                'v5_mfe': mfe,
                'v5_mae': mae,
            }
        else:
            sym_ids = None
            if symbol_id is not None and model.symbol_embedding is not None:
                sym_ids = torch.tensor([symbol_id], dtype=torch.long, device=device)
            output = model.forward_multihead(x, symbol_ids=sym_ids)

            enter_logit = float(output.enter_logits.cpu().item())
            calibrated_logit = enter_logit / max(temperature, 0.01)
            p_enter = float(torch.sigmoid(torch.tensor(calibrated_logit)).item())

            e_net_pred = 0.0
            if output.value_logits is not None:
                e_net_pred = float(output.value_logits.cpu().item())

            return {
                'p_enter': p_enter,
                'e_net_pred': e_net_pred,
                'enter_logit': enter_logit,
                'temperature_used': temperature,
            }


def _apply_htf_gates(features_df: pd.DataFrame) -> dict:
    """Apply HTF gates on the last row — identical to make_enter_prediction."""
    last_row = features_df.iloc[-1]
    h1_trend = last_row.get('h1_trend_sign', 0)
    h4_trend = last_row.get('h4_trend_sign', 0)
    h1_slope = last_row.get('h1_sma20_slope', 0)
    h1_range_pos = last_row.get('h1_range_pos', 0.5)

    trend_aligned = (h1_trend == h4_trend) and (h1_trend != 0)
    slope_ok = abs(h1_slope) > 0.05
    range_ok = True
    if h1_trend > 0 and h1_range_pos < 0.2:
        range_ok = False
    if h1_trend < 0 and h1_range_pos > 0.8:
        range_ok = False

    if h1_trend > 0:
        side = "LONG"
    elif h1_trend < 0:
        side = "SHORT"
    else:
        side = "NEUTRAL"

    return {
        'trend_aligned': trend_aligned,
        'slope_ok': slope_ok,
        'range_ok': range_ok,
        'side': side,
        'h1_trend': h1_trend,
        'h4_trend': h4_trend,
        'h1_slope': h1_slope,
        'h1_range_pos': h1_range_pos,
    }


def _compute_htf_score(htf: dict, direction: str) -> int:
    """Compute HTF score (0–3) for a given trade direction.

    +1 if h1_trend matches direction
    +1 if h4_trend matches direction
    +1 if slope_ok == True
    """
    score = 0
    dir_sign = 1 if direction == "LONG" else (-1 if direction == "SHORT" else 0)
    if dir_sign == 0:
        return 0

    h1 = htf.get('h1_trend', 0)
    h4 = htf.get('h4_trend', 0)
    slope_ok = htf.get('slope_ok', False)
    range_ok = htf.get('range_ok', False)

    if int(h1) == dir_sign:
        score += 1
    if int(h4) == dir_sign:
        score += 1
    if slope_ok:
        score += 1

    log.info(f"HTF_SCORE: score={score} h1={h1} h4={h4} slope_ok={slope_ok} range_ok={range_ok} dir={direction}")
    return score


SCALP_ATR_RATIO_MIN = 1.20
SCALP_TR_Z_MIN = 1.0
SCALP_BB_Z_MIN = 1.0
SCALP_SLOPE_MIN = 0.0005
SCALP_MACD_MIN = 0.0001
SCALP_VOL_RATIO_MIN = 1.2


def _compute_scalp_gates(df_candles: pd.DataFrame, features_df: pd.DataFrame,
                         direction: str, atr: float) -> dict:
    """Compute all SCALP gate metrics and pass/fail flags.

    Returns dict with:
      vol_expansion_ok, momentum_ok, atr_ratio, true_range_z, bb_width_z,
      ema20_slope, macd_hist_val, volume_ratio, range_ok_adjusted_mult
    """
    n = len(df_candles)
    last_row = features_df.iloc[-1] if len(features_df) > 0 else {}

    atr14 = atr
    atr50 = _compute_atr(df_candles, window=50) if n > 51 else atr
    atr_ratio = atr14 / atr50 if atr50 > 0 else 0.0

    highs = df_candles['high'].values.astype(float)
    lows = df_candles['low'].values.astype(float)
    closes = df_candles['close'].values.astype(float)
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if len(trs) >= 20:
        tr_arr = np.array(trs)
        tr_mean = tr_arr[-20:].mean()
        tr_std = tr_arr[-20:].std()
        true_range_z = (tr_arr[-1] - tr_mean) / tr_std if tr_std > 0 else 0.0
    else:
        true_range_z = 0.0

    bb_width = float(last_row.get('bb_width', 0))
    if n > 20 and 'bb_width' in features_df.columns:
        bw_vals = features_df['bb_width'].iloc[-20:].values.astype(float)
        bw_mean = np.nanmean(bw_vals)
        bw_std = np.nanstd(bw_vals)
        bb_width_z = (bb_width - bw_mean) / bw_std if bw_std > 0 else 0.0
    else:
        bb_width_z = 0.0

    vol_expansion_ok = (atr_ratio >= SCALP_ATR_RATIO_MIN and
                        (true_range_z >= SCALP_TR_Z_MIN or bb_width_z >= SCALP_BB_Z_MIN))

    if n >= 20:
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean()
        ema20_slope = (ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2] if ema20.iloc[-2] != 0 else 0.0
    else:
        ema20_slope = 0.0

    macd_hist_val = float(last_row.get('macd_hist', 0))

    slope_pass = abs(ema20_slope) >= SCALP_SLOPE_MIN
    macd_pass = abs(macd_hist_val) >= SCALP_MACD_MIN
    signal_component = slope_pass or macd_pass

    recent_vol = df_candles['volume'].iloc[-4:].mean() if n >= 4 else 0
    avg_vol = df_candles['volume'].iloc[-20:].mean() if n >= 20 else (df_candles['volume'].mean() if n > 0 else 1)
    volume_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else 0.0

    momentum_ok = signal_component and volume_ratio >= SCALP_VOL_RATIO_MIN

    return {
        'vol_expansion_ok': vol_expansion_ok,
        'momentum_ok': momentum_ok,
        'atr_ratio': round(atr_ratio, 4),
        'true_range_z': round(true_range_z, 4),
        'bb_width_z': round(bb_width_z, 4),
        'ema20_slope': round(ema20_slope, 6),
        'macd_hist_val': round(macd_hist_val, 6),
        'volume_ratio': round(volume_ratio, 4),
    }


def _compute_momentum_ok(features_df: pd.DataFrame, direction: str) -> bool:
    """Check momentum conditions for FLOW: ADX >= 18 OR MACD matches direction."""
    last_row = features_df.iloc[-1]
    adx = last_row.get('adx_14', 0)
    if adx >= 18:
        return True
    macd_val = last_row.get('macd', 0)
    if direction == "LONG" and macd_val > 0:
        return True
    if direction == "SHORT" and macd_val < 0:
        return True
    return False


def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Compute ATR from candle data."""
    n = min(window + 1, len(df))
    if n < 3:
        return float(df.iloc[-1]['close']) * 0.005

    highs = df.iloc[-n:]['high'].values.astype(float)
    lows = df.iloc[-n:]['low'].values.astype(float)
    closes = df.iloc[-n:]['close'].values.astype(float)

    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        true_ranges.append(tr)
    return float(np.mean(true_ranges))


def _resolve_mythos_live_tp_sl(
    *,
    cfg_obj,
    side: str,
    edge: float,
    confidence: float,
    uncertainty: float,
    feature_row: Dict[str, float],
    fallback_tp_mult: float,
    fallback_sl_mult: float,
) -> Tuple[float, float]:
    """
    Mirror walk-forward adaptive TP/SL logic for Mythos live paper execution.
    Falls back to static multipliers if adaptive config is unavailable/disabled.
    """
    base_tp = float(max(getattr(cfg_obj, "tp_mult", fallback_tp_mult), 0.10)) if cfg_obj is not None else float(max(fallback_tp_mult, 0.10))
    base_sl = float(max(getattr(cfg_obj, "sl_mult", fallback_sl_mult), 0.10)) if cfg_obj is not None else float(max(fallback_sl_mult, 0.10))
    if cfg_obj is None or not bool(getattr(cfg_obj, "adaptive_tp_sl_enable", True)):
        return base_tp, base_sl
    try:
        edge_unit = float(max(getattr(cfg_obj, "min_expected_r", 0.01), 1e-6))
        edge_score = float(np.tanh(float(edge) / max(2.0 * edge_unit, 1e-6)))
        conf_score = float(np.clip((float(confidence) - 0.5) * 2.0, -1.0, 1.0))
        unc_norm = float(np.clip(float(uncertainty), 0.0, 2.0) / 2.0)
        trend_ema = float(feature_row.get("trend_ema", 0.0))
        vol_16 = float(max(feature_row.get("vol_16", 0.0), 1e-6))
        vol_64 = float(max(feature_row.get("vol_64", vol_16), 1e-6))
        vol_ratio = float(np.clip(vol_16 / max(vol_64, 1e-6), 0.25, 4.0))
        vol_stress = float(max(vol_ratio - 1.0, 0.0))
        trend_score = float(np.clip(np.tanh(abs(trend_ema) * 4.0), 0.0, 1.0))
        quality = float(np.clip(0.45 * conf_score + 0.45 * edge_score - 0.30 * unc_norm, -1.0, 1.0))

        tp_gain = (
            float(np.clip(getattr(cfg_obj, "adaptive_tp_quality_gain", 0.35), 0.0, 2.0)) * quality
            + float(np.clip(getattr(cfg_obj, "adaptive_tp_trend_gain", 0.20), 0.0, 2.0)) * trend_score
            - float(np.clip(getattr(cfg_obj, "adaptive_tp_vol_penalty", 0.18), 0.0, 2.0)) * vol_stress
        )
        sl_gain = (
            -float(np.clip(getattr(cfg_obj, "adaptive_sl_quality_tighten", 0.25), 0.0, 2.0)) * max(quality, 0.0)
            + float(np.clip(getattr(cfg_obj, "adaptive_sl_uncertainty_widen", 0.30), 0.0, 2.0)) * unc_norm
            + float(np.clip(getattr(cfg_obj, "adaptive_sl_vol_widen", 0.20), 0.0, 2.0)) * vol_stress
        )
        if str(side).upper() == "SHORT":
            tp_gain += float(np.clip(getattr(cfg_obj, "adaptive_short_tp_bias", 0.05), -1.0, 1.0))
            sl_gain += float(np.clip(getattr(cfg_obj, "adaptive_short_sl_bias", 0.04), -1.0, 1.0))

        tp_mult = float(base_tp * (1.0 + tp_gain))
        sl_mult = float(base_sl * (1.0 + sl_gain))
        tp_mult = float(
            np.clip(
                tp_mult,
                float(np.clip(getattr(cfg_obj, "adaptive_tp_min_mult", 1.2), 0.10, 20.0)),
                float(max(getattr(cfg_obj, "adaptive_tp_max_mult", 3.6), getattr(cfg_obj, "adaptive_tp_min_mult", 1.2))),
            )
        )
        sl_mult = float(
            np.clip(
                sl_mult,
                float(np.clip(getattr(cfg_obj, "adaptive_sl_min_mult", 0.8), 0.10, 20.0)),
                float(max(getattr(cfg_obj, "adaptive_sl_max_mult", 2.4), getattr(cfg_obj, "adaptive_sl_min_mult", 0.8))),
            )
        )
        return tp_mult, sl_mult
    except Exception:
        return base_tp, base_sl


def _build_prediction_payload(
    symbol: str, side: str, p_enter: float, current_price: float, atr: float,
    entry_price: float, tp_mult: float, sl_mult: float, htf: dict,
    tp_mult_used: Optional[float] = None, sl_mult_used: Optional[float] = None,
    exec_result=None
) -> dict:
    """Build a prediction dict compatible with the dashboard push endpoint."""
    tp_mult_eff = float(tp_mult_used if tp_mult_used is not None else tp_mult)
    sl_mult_eff = float(sl_mult_used if sl_mult_used is not None else sl_mult)
    if side == "LONG":
        sl_price = entry_price - sl_mult_eff * atr
        tp_price = entry_price + tp_mult_eff * atr
    else:
        sl_price = entry_price + sl_mult_eff * atr
        tp_price = entry_price - tp_mult_eff * atr

    sl_pct = abs(entry_price - sl_price) / entry_price
    tp_pct = abs(tp_price - entry_price) / entry_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    ACCOUNT_RISK = 0.02
    position_size = ACCOUNT_RISK / sl_pct * 100 if sl_pct > 0 else 1.0
    if p_enter > 0.75:
        position_size *= 1.25
    elif p_enter < 0.55:
        position_size *= 0.5
    position_size = min(max(position_size, 0.5), 5.0)

    risk_pct = min(position_size * sl_pct * 100, 5.0)

    reasons = []
    reasons.append(f"ENTER signal: p_enter={p_enter:.1%}")
    reasons.append(f"HTF trend: {side} (1H={htf['h1_trend']:+.0f}, 4H={htf['h4_trend']:+.0f})")
    if exec_result and exec_result.executed and exec_result.method == "pullback_confirm":
        reasons.append(f"Improved entry via {exec_result.method} ({exec_result.cost_improvement_bps:+.1f} bps)")

    return {
        "symbol": symbol,
        "action": side,
        "confidence": round(p_enter, 4),
        "direction_probs": {
            "SHORT": round(1.0 if side == "SHORT" else 0.0, 4),
            "HOLD": 0.0,
            "LONG": round(1.0 if side == "LONG" else 0.0, 4),
        },
        "quantiles": {},
        "vol_state": "neutral",
        "vol_state_probs": {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33},
        "expected_return": round(p_enter - 0.5, 6),
        "uncertainty": round(1.0 - p_enter, 6),
        "edge": round(p_enter - 0.5, 4),
        "entry_price": round(entry_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "v5_forecaster_live",
        "is_multihead": True,
        "urgency": "high" if p_enter > 0.7 else "medium",
        "suggested_order_type": "limit",
        "reasons": reasons,
        "risk_pct": round(risk_pct, 2),
        "p_enter": round(p_enter, 4),
        "atr": round(atr, 2),
        "tp_mult_used": round(tp_mult_eff, 4),
        "sl_mult_used": round(sl_mult_eff, 4),
    }


class LiveRunner:
    def __init__(
        self,
        replit_url: str,
        symbols: List[str],
        device: str,
        interval: str = "15m",
        enter_threshold: float = 0.85,
        tp_mult: float = 3.5,
        sl_mult: float = 1.5,
        cooldown_bars: int = 8,
        paper: bool = False,
        execution_mode: str = "signal_only",
        record_trades: bool = False,
        portfolio_manager=None,
        execution_module=None,
        dry_run: bool = False,
        dry_run_candles: int = 200,
        per_symbol_models: bool = False,
        limit_15m: int = REQUIRED_CANDLES,
        direct_htf: bool = False,
        budget_core: float = None,
        budget_flow: float = None,
        budget_scalp: float = None,
        side_aware_scoring: bool = False,
        direction_balance_cap: bool = False,
        direction_balance_threshold: float = 0.75,
        v5_live_threshold: float = None,
        v5_mae_floor: float = None,
        predictive_sltp: bool = False,
        paper_session_id: Optional[str] = None,
        dashboard_engine: str = "v5",
        live_model: str = "v5",
        equity_floor_usd: float = 0.0,
        equity_hard_stop_usd: float = 0.0,
        mythos_tm_enabled: bool = True,
        mythos_tm_policy: str = "max",
        mythos_tm_scale_out: bool = True,
        mythos_tm_time_adaptive: bool = True,
        mythos_tm_vol_trailing: bool = True,
        exec_max_spread_bps: float = 14.0,
        exec_spread_adaptive_enable: bool = True,
        exec_spread_window: int = 80,
        exec_spread_target_slippage_bps: float = 3.0,
        exec_spread_min_bps: float = 4.0,
    ):
        self.replit_url = replit_url
        self.symbols = symbols
        self.device = device
        self.interval = interval
        self.enter_threshold = enter_threshold  # kept for backward compat but V5 scoring is primary
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.cooldown_bars = cooldown_bars
        self.paper = paper
        self.execution_mode = execution_mode
        self.record_trades = record_trades
        self._exchange_api_url = _is_exchange_api_url(replit_url)
        self._web_api_enabled = bool(replit_url) and (not self._exchange_api_url)
        self._web_api_disable_reason = ""
        self._web_api_404_count = 0
        if self.execution_mode == "paper" and self.record_trades and not self._web_api_enabled:
            self._web_api_disable_reason = "exchange-api-url"
            log.warning(
                "[PAPER_LOCAL] URL looks like exchange API (%s). "
                "Running local paper engine without web sync (/api/paper/*, /api/live/*).",
                replit_url,
            )
        self.portfolio = portfolio_manager
        self.execution = execution_module
        self.dry_run = dry_run
        self.dry_run_candles = dry_run_candles
        self.per_symbol_models = per_symbol_models
        self.limit_15m = limit_15m
        self.direct_htf = direct_htf
        self.gpu_self_url = self._detect_gpu_self_url()
        self.side_aware_scoring = side_aware_scoring
        self.direction_balance_cap = direction_balance_cap
        self.direction_balance_threshold = direction_balance_threshold
        self._recent_signal_sides: list = []
        self.predictive_sltp = predictive_sltp
        self.paper_session_id = str(paper_session_id or "default").strip() or "default"
        self.dashboard_engine = _normalize_dashboard_engine(dashboard_engine)
        raw_live_model = str(live_model or "auto").strip().lower()
        if raw_live_model in {"", "auto"}:
            resolved_live_model = "mythos" if self.dashboard_engine == "mythos" else "v5"
        else:
            resolved_live_model = raw_live_model
        if resolved_live_model != "mythos" and self.dashboard_engine == "mythos":
            log.warning(
                "[MODE_SYNC] dashboard_engine=mythos but live_model=%s — forcing live_model=mythos",
                resolved_live_model,
            )
            resolved_live_model = "mythos"
        self.live_model = resolved_live_model
        self.equity_floor_usd = float(max(equity_floor_usd, 0.0))
        self.equity_hard_stop_usd = float(max(equity_hard_stop_usd, 0.0))
        if self.equity_hard_stop_usd > 0.0 and self.equity_floor_usd > 0.0 and self.equity_hard_stop_usd > self.equity_floor_usd:
            self.equity_hard_stop_usd = self.equity_floor_usd
        self.mythos_tm_enabled = bool(mythos_tm_enabled)
        self.mythos_tm_policy = str(mythos_tm_policy or "max").strip().lower()
        if self.mythos_tm_policy not in {"defensive", "balanced", "aggressive", "max"}:
            self.mythos_tm_policy = "max"
        self.mythos_tm_scale_out = bool(mythos_tm_scale_out)
        self.mythos_tm_time_adaptive = bool(mythos_tm_time_adaptive)
        self.mythos_tm_vol_trailing = bool(mythos_tm_vol_trailing)
        self.exec_max_spread_bps = float(max(exec_max_spread_bps, 0.0))
        self.exec_spread_adaptive_enable = bool(exec_spread_adaptive_enable)
        self.exec_spread_window = int(max(exec_spread_window, 5))
        self.exec_spread_target_slippage_bps = float(max(exec_spread_target_slippage_bps, 0.1))
        self.exec_spread_min_bps = float(max(exec_spread_min_bps, 0.1))
        self._recent_entry_spread_bps: List[float] = []
        self._recent_entry_slippage_bps: List[float] = []
        self._equity_hard_stop_latched: bool = False
        self._equity_snapshot_cache_ts: float = 0.0
        self._equity_snapshot_live_usd: Optional[float] = None

        # ── Regime-aware signal router ─────────────────────────────────────────
        # Tracks H4 SMA20 regime per symbol with 3-bar confirmation.
        # Blocks signals that contradict the broader market trend.
        self._regime_history: Dict[str, list] = {}   # symbol -> last 3 regime readings
        self._regime_confirmed: Dict[str, str] = {}  # symbol -> 'BULL' | 'BEAR'

        try:
            from config.shared_v5_trade_config import load_shared_defaults
            _shared_defaults = load_shared_defaults()
        except Exception:
            _shared_defaults = None
        _shared_scoring = None if self.live_model == "mythos" else _shared_defaults

        self.v5_score_lambda = V5_SCORE_LAMBDA if _shared_scoring is None else _shared_scoring.score_lambda
        self.v5_score_threshold = (
            v5_live_threshold if v5_live_threshold is not None
            else (V5_SCORE_THRESHOLD if _shared_scoring is None else _shared_scoring.score_threshold)
        )
        self.v5_min_mu_r = V5_MIN_MU_R if _shared_scoring is None else _shared_scoring.min_mu_r_score
        self.v5_mae_floor = v5_mae_floor if v5_mae_floor is not None else V5_MAE_FLOOR
        self.v5_min_p_side: float = 0.0 if _shared_scoring is None else _shared_scoring.min_p_side
        self.v5_min_p_short: float = 0.0 if _shared_scoring is None else _shared_scoring.min_p_short
        self.v5_slippage_bps: float = 0.0 if _shared_scoring is None else _shared_scoring.slippage_base_bps
        self.cooldown_bars = (
            _shared_scoring.cooldown_bars
            if (_shared_scoring is not None and cooldown_bars == 8)
            else cooldown_bars
        )

        self.halt_on_data_staleness: bool = _shared_defaults.halt_on_data_staleness if _shared_defaults else True
        self.max_data_staleness_seconds: float = _shared_defaults.max_data_staleness_seconds if _shared_defaults else 300.0
        self.halt_on_api_errors: bool = _shared_defaults.halt_on_api_errors if _shared_defaults else True
        self.max_consecutive_api_errors: int = _shared_defaults.max_consecutive_api_errors if _shared_defaults else 5
        self.max_daily_loss_r: Optional[float] = _shared_defaults.max_daily_loss_r if _shared_defaults else None

        log.info(f"[INIT] LiveRunner {SYSTEM_VERSION} execution_mode={execution_mode} "
                 f"record_trades={record_trades} symbols={symbols} session={self.paper_session_id} "
                 f"engine={self.dashboard_engine} model={self.live_model}")
        if self.live_model == "mythos":
            log.info(
                "[CONFIG] Mythos runtime selected for live/paper inference "
                "(V5 scorer settings ignored)"
            )
        else:
            log.info(
                "[CONFIG] V5 scoring (shared defaults applied): "
                "lambda=%.3f threshold=%.3f min_mu_r=%.3f mae_floor=%.3f "
                "min_p_side=%.3f min_p_short=%.3f slippage_bps=%.1f cooldown=%d",
                self.v5_score_lambda, self.v5_score_threshold, self.v5_min_mu_r,
                self.v5_mae_floor, self.v5_min_p_side, self.v5_min_p_short,
                self.v5_slippage_bps, self.cooldown_bars,
            )
            log.info(
                "[CONFIG] V5 trade model path: checkpoint model_type=v5_forecaster -> V5Forecaster; "
                "fallback model_type=legacy -> EnhancedMultiHeadMLP"
            )
        log.info(
            "[CONFIG] Halt switches: data_staleness=%s/%gs api_errors=%s/%d daily_loss_r=%s",
            self.halt_on_data_staleness, self.max_data_staleness_seconds,
            self.halt_on_api_errors, self.max_consecutive_api_errors, self.max_daily_loss_r,
        )
        log.info(
            "[CONFIG] Equity guards: floor_usd=%s hard_stop_usd=%s",
            self.equity_floor_usd,
            self.equity_hard_stop_usd,
        )
        log.info(
            "[CONFIG] Execution quality: max_spread_bps=%.2f adaptive=%s window=%d target_slippage_bps=%.2f min_spread_bps=%.2f",
            self.exec_max_spread_bps,
            self.exec_spread_adaptive_enable,
            self.exec_spread_window,
            self.exec_spread_target_slippage_bps,
            self.exec_spread_min_bps,
        )
        if self.live_model == "mythos":
            log.info(
                "[CONFIG] Mythos trade manager: enabled=%s policy=%s scale_out=%s time_adaptive=%s vol_trailing=%s",
                self.mythos_tm_enabled,
                self.mythos_tm_policy,
                self.mythos_tm_scale_out,
                self.mythos_tm_time_adaptive,
                self.mythos_tm_vol_trailing,
            )

        self.model = None
        self.engineer = None
        self.feature_columns = None
        self.symbol_models: Dict[str, Tuple] = {}
        self.fetcher = None
        self.cycle_count = 0
        self.candle_cache: Dict[str, pd.DataFrame] = {}
        self.exchange_time_offset = 0.0
        self.cooldown_tracker: Dict[str, int] = {}
        self.warmup_logged: Dict[str, bool] = {}
        self._funding_cache: Dict[str, Dict] = {}
        self._oi_cache: Dict[str, Dict] = {}
        self._ls_ratio_cache: Dict[str, Dict] = {}
        self._feature_check_logged: Dict[str, bool] = {}

        from trade_manager import TradeManager
        trade_policy = self.mythos_tm_policy if self.live_model == "mythos" else "balanced"
        self.trade_manager = TradeManager(
            policy_mode=trade_policy,
            enable_scale_out=(self.mythos_tm_scale_out if self.live_model == "mythos" else True),
            enable_regime_time_stop=(self.mythos_tm_time_adaptive if self.live_model == "mythos" else True),
            enable_volatility_trailing=(self.mythos_tm_vol_trailing if self.live_model == "mythos" else True),
        )

        self._consecutive_api_errors: int = 0
        self._daily_closed_r: float = 0.0
        self._daily_r_date: str = ""
        self._last_candle_time: float = 0.0
        self._closed_trade_stats: List[Dict[str, object]] = []

        if self.execution_mode == "live" and self.execution is None:
            log.warning(
                "[LIVE_RUNNER] WARNING: execution_mode='live' but NO real exchange adapter "
                "is wired (self.execution is None).  No real orders will be placed.  "
                "All 'LIVE_OPEN' log lines reflect signal-only mode — they are NOT real trades.  "
                "Wire an execution adapter via execution_module= to place real orders."
            )

    def _disable_web_api(self, reason: str) -> None:
        if not self._web_api_enabled:
            return
        self._web_api_enabled = False
        self._web_api_disable_reason = str(reason or "unknown")
        log.warning(
            "[PAPER_LOCAL] Disabling web sync/telemetry endpoints: %s. "
            "Paper engine will continue locally.",
            self._web_api_disable_reason,
        )

    def _note_web_api_status(self, source: str, status_code: int) -> None:
        if status_code == 404:
            self._web_api_404_count += 1
            if self.execution_mode == "paper" and self._web_api_404_count >= 2:
                self._disable_web_api(f"{source} returned HTTP 404 repeatedly")
        else:
            self._web_api_404_count = 0

    @staticmethod
    def _detect_gpu_self_url() -> Optional[str]:
        """Detect the GPU trainer's own public URL (e.g. ngrok tunnel)."""
        import os
        explicit = os.environ.get("GPU_SELF_URL")
        if explicit:
            log.info(f"[GPU URL] Using explicit GPU_SELF_URL: {explicit}")
            return explicit.rstrip("/")
        try:
            resp = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            if resp.status_code == 200:
                tunnels = resp.json().get("tunnels", [])
                for t in tunnels:
                    if t.get("proto") == "https":
                        url = t["public_url"].rstrip("/")
                        log.info(f"[GPU URL] Auto-detected ngrok URL: {url}")
                        return url
                if tunnels:
                    url = tunnels[0].get("public_url", "").rstrip("/")
                    log.info(f"[GPU URL] Auto-detected ngrok URL: {url}")
                    return url
        except Exception:
            pass
        log.warning("[GPU URL] No ngrok tunnel detected and GPU_SELF_URL not set")
        return None

    def _register_gpu_url(self):
        """Register GPU trainer URL with the Replit dashboard."""
        if not self._web_api_enabled:
            return
        if not self.gpu_self_url:
            return
        try:
            url = f"{self.replit_url.rstrip('/')}/api/gpu/register"
            resp = requests.post(url, json={"url": self.gpu_self_url}, timeout=5)
            if resp.status_code == 200:
                log.info(f"[GPU REG] Registered URL with Replit: {self.gpu_self_url}")
            else:
                log.warning(f"[GPU REG] Registration failed: {resp.status_code}")
        except Exception as e:
            log.warning(f"[GPU REG] Failed to register: {e}")

    def _start_execution_service(self):
        """Start the Bybit execution service push loop if API keys are available."""
        if not self._web_api_enabled:
            log.info("[Execution Service] Web backend disabled — skipping execution service push loop")
            return
        import os
        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        if not api_key or not api_secret:
            log.info("[Execution Service] No BYBIT_API_KEY/SECRET — skipping execution service")
            return
        try:
            from execution_service import start_execution_service
            start_execution_service(
                replit_url=self.replit_url,
                gpu_self_url=self.gpu_self_url,
                api_key=api_key,
                api_secret=api_secret,
                interval=5,
                daemon=True,
            )
            log.info("[Execution Service] Bybit state push loop started")
        except Exception as e:
            log.warning(f"[Execution Service] Failed to start: {e}")

    def _init_fetcher(self):
        from data.pipeline import BinanceDataFetcher
        proxy_url = self.replit_url.rstrip('/') if self._web_api_enabled else None
        if proxy_url is None:
            log.info("[DATA_FETCH] Using direct Binance fetch path only (proxy disabled)")
        self.fetcher = BinanceDataFetcher(
            symbols=self.symbols,
            timeframes=[self.interval],
            replit_proxy_url=proxy_url,
            use_sync=True,
        )
        if self.execution and self.execution.fetcher is None:
            self.execution.fetcher = self.fetcher

    def _on_position_close(self, pos, exit_price: float, outcome: str, gross_r: float):
        """Callback when portfolio closes a position — push trade update to dashboard."""
        if hasattr(self, 'separation_verifier') and self.separation_verifier:
            self.separation_verifier.record_exit_resolve()
        from portfolio import Position as _Pos

        initial_sl = pos.original_sl if pos.original_sl is not None else pos.sl_price
        original_risk_abs = abs(pos.entry_price - initial_sl)

        if pos.is_long:
            gross_r_calc = (exit_price - pos.entry_price) / original_risk_abs if original_risk_abs > 0 else 0
        else:
            gross_r_calc = (pos.entry_price - exit_price) / original_risk_abs if original_risk_abs > 0 else 0

        if abs(gross_r_calc - gross_r) > 0.01:
            log.warning(f"[R_CHECK] MISMATCH sym={pos.symbol} side={pos.side} "
                        f"entry={pos.entry_price} exit={exit_price} initial_sl={initial_sl} "
                        f"orig_risk_abs={original_risk_abs:.4f} "
                        f"gross_r_calc={gross_r_calc:.6f} gross_r_stored={gross_r:.6f}")
            gross_r = gross_r_calc

        cost_bps = COST_BPS
        cost_r = (cost_bps / 10000) * 2 / (original_risk_abs / pos.entry_price) if original_risk_abs > 0 else 0.0
        net_r = gross_r - cost_r
        sized_r = net_r * pos.size_mult

        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if today_str != self._daily_r_date:
            self._daily_closed_r = 0.0
            self._daily_r_date = today_str
        self._daily_closed_r += net_r
        if (self.max_daily_loss_r is not None
                and self._daily_closed_r <= -abs(self.max_daily_loss_r)):
            log.warning(
                "[HALT] Daily loss limit reached: daily_r=%.2f <= -%.2f — "
                "new entries will be BLOCKED until next UTC day",
                self._daily_closed_r, abs(self.max_daily_loss_r),
            )

        log.info(f"[R_CHECK] sym={pos.symbol} side={pos.side} entry={pos.entry_price:.2f} "
                 f"exit={exit_price:.2f} initial_sl={initial_sl:.2f} orig_risk_abs={original_risk_abs:.4f} "
                 f"gross_r_calc={gross_r_calc:.6f} gross_r_stored={gross_r:.6f} "
                 f"cost_r={cost_r:.6f} net_r={net_r:.6f}")

        check = abs(net_r - (gross_r - cost_r))
        if check > 1e-6:
            log.error(f"[NET_CHECK] INVARIANT VIOLATED: net_r={net_r:.6f} != gross_r={gross_r:.6f} - cost_r={cost_r:.6f} (diff={check:.8f})")

        if hasattr(self, 'verifier') and self.verifier:
            vf = self.verifier
            fs = vf.verify_net_r(vf.stats.total_cycles, gross_r, cost_r, net_r)
            vf.add_failures(fs)

        risk_usd = 0.0
        try:
            import requests as _req
            money_resp = _req.get(f"{self.replit_url.rstrip('/')}/api/money-config", timeout=5)
            if money_resp.status_code == 200:
                mc = money_resp.json()
                equity = mc.get('account_equity_usd', 1500)
                risk_pct = mc.get('risk_per_trade_pct', 1.0)
                risk_usd = equity * (risk_pct / 100) * pos.size_mult
        except Exception:
            pass
        gross_usd = round(gross_r * risk_usd, 2)
        cost_usd = round(cost_r * risk_usd, 2)
        net_usd = round(net_r * risk_usd, 2)
        self._closed_trade_stats.append({
            "symbol": pos.symbol,
            "side": pos.side,
            "outcome": outcome,
            "gross_r": float(gross_r),
            "cost_r": float(cost_r),
            "net_r": float(net_r),
            "sized_r": float(sized_r),
        })
        aware = self._decision_awareness_snapshot()
        overall = aware.get("overall", {})
        log.info(
            "[LIVE_AWARENESS_UPDATE] model=%s n=%s exp=%+.3f wr=%.1f%% long_exp=%+.3f short_exp=%+.3f",
            self.live_model,
            int(overall.get("n", 0)),
            float(overall.get("expectancy_r", 0.0)),
            100.0 * float(overall.get("win_rate", 0.0)),
            float((aware.get("by_side", {}).get("LONG", {}) or {}).get("expectancy_r", 0.0)),
            float((aware.get("by_side", {}).get("SHORT", {}) or {}).get("expectancy_r", 0.0)),
        )

        exit_reason = outcome
        if outcome == "TIME_EXIT":
            exit_reason = f"SCALP_TIME_STOP ({pos.horizon} bars)"

        tm_state = self.trade_manager.get_state(pos.symbol)
        mfe = tm_state['max_favorable_r'] if tm_state else None
        mae = tm_state['max_adverse_r'] if tm_state else None
        be_moved = tm_state['breakeven_moved'] if tm_state else pos.breakeven_moved
        bars_held = self.cycle_count - pos.bar_index if pos.bar_index > 0 else None
        if bars_held is None and pos.entry_time > 0:
            bars_held = max(1, int((time.time() - pos.entry_time) / (15 * 60)))
        if bars_held is None:
            bars_held = 0
        is_time_exit = outcome == "TIME_EXIT"

        self.trade_manager.clear_position(pos.symbol)

        try:
            self._update_trade_record(
                trade_id=pos.dashboard_trade_id,
                exit_price=exit_price,
                outcome=outcome,
                gross_r=gross_r,
                net_r=net_r,
                sized_r=sized_r,
                cost_r=cost_r,
                exit_reason=exit_reason,
                max_favorable_r=mfe,
                max_adverse_r=mae,
                time_exit=is_time_exit,
                breakeven_moved=be_moved,
                bars_held=bars_held,
                initial_sl=initial_sl,
                risk_usd=risk_usd,
                gross_usd=gross_usd,
                cost_usd=cost_usd,
                net_usd=net_usd,
            )
        except Exception as e:
            log.warning(f"Failed to update trade record {pos.dashboard_trade_id}: {e}")

    def _push_prediction(self, prediction: dict):
        if not self._web_api_enabled:
            return
        from quick_start import push_prediction
        if not isinstance(prediction, dict):
            prediction = {}
        prediction.setdefault("session_id", self.paper_session_id)
        prediction.setdefault("engine", self.dashboard_engine)
        push_prediction(self.replit_url, prediction)

    def _push_cycle_log(self, symbol: str, price: float, p_enter: float,
                        htf: dict, direction: str, decision: str, reasons: list,
                        lane_info: Optional[dict] = None,
                        e_net_pred: float = None, enter_logit: float = None,
                        temperature_used: float = None,
                        decision_stage: str = "candidate",
                        execution_status: Optional[str] = None):
        if not self._web_api_enabled:
            return
        url = f"{self.replit_url.rstrip('/')}/api/live/cycle-log"
        li = lane_info or {}
        stage = str(decision_stage or "candidate").strip().lower()
        if stage not in {"candidate", "execution"}:
            stage = "candidate"
        status = str(execution_status or ("candidate" if stage == "candidate" else "unknown")).strip().lower()
        payload = {
            "symbol": symbol,
            "cycle_ts": int(time.time() * 1000),
            "session_id": self.paper_session_id,
            "engine": self.dashboard_engine,
            "price": float(price),
            "p_enter": float(p_enter),
            "htf_h1_trend": str(htf.get('h1_trend', '')),
            "htf_h4_trend": str(htf.get('h4_trend', '')),
            "slope_ok": bool(htf.get('slope_ok', False)),
            "range_ok": bool(htf.get('range_ok', False)),
            "direction": str(direction),
            "threshold_used": float(li.get('threshold_used', self.v5_score_threshold)),
            "decision": str(decision),
            "reasons": [str(r) for r in reasons] if reasons else [],
            "decision_stage": stage,
            "execution_status": status,
            "htf_score": li.get('htf_score'),
            "hold_reason": li.get('hold_reason'),
            "e_net_pred": round(float(e_net_pred), 4) if e_net_pred is not None else li.get('ret_mu'),
            "enter_logit": round(float(enter_logit), 4) if enter_logit is not None else None,
            "temperature_used": round(float(temperature_used), 4) if temperature_used is not None else None,
            "v5_score": li.get('v5_score'),
            "v5_threshold": li.get('v5_threshold'),
            "v5_side": li.get('v5_side'),
            "v5_mfe": li.get('v5_mfe'),
            "v5_mae": li.get('v5_mae'),
            "v5_ret_mu": li.get('ret_mu'),
            "v5_p_long": li.get('p_long'),
            "v5_p_short": li.get('p_short'),
            "sl_price": li.get('sl_price'),
            "tp_price": li.get('tp_price'),
            "model_name": li.get('model_name', f"{self.dashboard_engine}_runtime"),
        }
        if self.gpu_self_url:
            payload["gpu_callback_url"] = self.gpu_self_url
        payload_keys = [k for k, v in payload.items() if v is not None]
        log.debug(f"[CYCLE_PAYLOAD] sym={symbol} fields_present={payload_keys}")
        _retry_request("POST", url, json=payload)

    def _push_trade_record(self, symbol: str, side: str, entry_price: float,
                           sl_price: float, tp_price: float, p_enter: float,
                           size_pct: float, lane_info: Optional[dict] = None) -> Optional[int]:
        if not self._web_api_enabled:
            return None
        url = f"{self.replit_url.rstrip('/')}/api/live/trade"
        li = lane_info or {}
        payload = {
            "symbol": symbol,
            "side": side,
            "session_id": self.paper_session_id,
            "engine": self.dashboard_engine,
            "entry_time": int(time.time() * 1000),
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "initial_sl": sl_price,
            "p_enter": p_enter,
            "size_pct": size_pct,
            "status": "open",
            "lane": li.get('lane', 'V5'),
            "v5_score": li.get('v5_score'),
            "htf_score": li.get('htf_score'),
            "lane_threshold_used": li.get('threshold_used'),
            "lane_size_mult": li.get('lane_size_mult', 1.0),
            "lane_horizon": li.get('lane_horizon', 24),
            "model_name": li.get('model_name', f"{self.dashboard_engine}_runtime"),
        }
        resp = _retry_request("POST", url, json=payload)
        if resp and resp.status_code == 200:
            data = resp.json()
            return data.get("id")
        return None

    def _update_trade_record(self, trade_id: int, exit_price: float,
                             outcome: str, gross_r: float, net_r: float, sized_r: float,
                             cost_r: float = 0.0,
                             exit_reason: Optional[str] = None,
                             max_favorable_r: Optional[float] = None,
                             max_adverse_r: Optional[float] = None,
                             time_exit: Optional[bool] = None,
                             breakeven_moved: Optional[bool] = None,
                             bars_held: Optional[int] = None,
                             initial_sl: Optional[float] = None,
                             risk_usd: float = 0.0,
                             gross_usd: float = 0.0,
                             cost_usd: float = 0.0,
                             net_usd: float = 0.0):
        if not self._web_api_enabled:
            return
        url = f"{self.replit_url.rstrip('/')}/api/live/trade/{trade_id}"
        payload = {
            "session_id": self.paper_session_id,
            "engine": self.dashboard_engine,
            "exit_time": int(time.time() * 1000),
            "exit_price": exit_price,
            "outcome": outcome,
            "gross_r": round(gross_r, 6),
            "net_r": round(net_r, 6),
            "sized_r": round(sized_r, 6),
            "cost_r": round(cost_r, 6),
            "status": "closed",
            "exit_reason": exit_reason or outcome,
            "bars_held": bars_held if bars_held is not None else 0,
            "risk_usd_used": round(risk_usd, 2),
            "pnl_usd_gross": round(gross_usd, 2),
            "pnl_usd_cost": round(cost_usd, 2),
            "pnl_usd": round(net_usd, 2),
        }
        if initial_sl is not None:
            payload["initial_sl"] = round(initial_sl, 6)
        if max_favorable_r is not None:
            payload["max_favorable_r"] = round(max_favorable_r, 4)
        if max_adverse_r is not None:
            payload["max_adverse_r"] = round(max_adverse_r, 4)
        if time_exit is not None:
            payload["time_exit"] = time_exit
        if breakeven_moved is not None:
            payload["breakeven_moved"] = breakeven_moved
        _retry_request("PATCH", url, json=payload)

    def _update_trade_sl(self, trade_id: int, new_sl: float):
        """Update stop loss on an open trade record in the dashboard."""
        if not self._web_api_enabled:
            return
        url = f"{self.replit_url.rstrip('/')}/api/live/trade/{trade_id}"
        payload = {"stop_loss": new_sl, "session_id": self.paper_session_id}
        _retry_request("PATCH", url, json=payload)

    def _post_trade_manager_action(
        self,
        trade_id: int,
        action: str,
        extra_payload: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        if not self._web_api_enabled:
            return None
        if not self.replit_url or trade_id <= 0:
            return None
        payload: Dict[str, object] = {"action": str(action)}
        if extra_payload:
            payload.update(extra_payload)
        resp = _retry_request(
            "POST",
            f"{self.replit_url.rstrip('/')}/api/paper/trade/{int(trade_id)}/manager-action",
            params={"session_id": self.paper_session_id},
            json=payload,
        )
        if resp is None or resp.status_code != 200:
            return None
        try:
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _apply_partial_close_from_tm(self, pos, close_pct: float, reason: str) -> bool:
        close_pct = float(np.clip(close_pct, 1.0, 100.0))
        tm_note = f"tm_auto_{reason.lower()}"
        if pos.dashboard_trade_id:
            payload = {
                "close_pct": close_pct,
                "note": tm_note,
                "manual_close": False,
            }
            result = self._post_trade_manager_action(int(pos.dashboard_trade_id), "partial_close", payload)
            if result:
                self._sync_portfolio_from_web(source=f"tm_partial_{pos.symbol}")
                return True
            return False

        # Fallback path for non-dashboard contexts: keep the position open but
        # reduce risk and effective size so manager behavior still matches.
        keep_ratio = max(0.0, 1.0 - (close_pct / 100.0))
        pos.risk_pct = float(max(pos.risk_pct * keep_ratio, 0.0))
        pos.size_mult = float(max(pos.size_mult * keep_ratio, 0.0))
        log.info(
            "[TM_PARTIAL_FALLBACK] sym=%s pct=%.1f keep_ratio=%.3f reason=%s",
            pos.symbol,
            close_pct,
            keep_ratio,
            reason,
        )
        return True

    def _get_model_for_symbol(self, symbol: str):
        if self.per_symbol_models:
            if symbol not in self.symbol_models:
                try:
                    m, e, fc, temp, sm = _load_model(self.device, symbol=symbol, model_backend=self.live_model)
                    self._apply_live_model_runtime_overrides(m, symbol=symbol)
                    self.symbol_models[symbol] = (m, e, fc, temp, sm)
                except SystemExit:
                    log.warning(f"No per-symbol model for {symbol}, using global model")
                    self.symbol_models[symbol] = (self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map)
            return self.symbol_models[symbol]
        return self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map

    def _apply_live_model_runtime_overrides(self, model, symbol: Optional[str] = None):
        """Apply live runtime controls (e.g. cooldown) without retraining."""
        if not getattr(model, "_is_mythos", False):
            return
        cfg = getattr(model, "cfg", None)
        if cfg is None:
            return
        try:
            prev = int(getattr(cfg, "cooldown_bars", self.cooldown_bars))
            now = int(max(self.cooldown_bars, 0))
            cfg.cooldown_bars = now
            if prev != now:
                sym = str(symbol or getattr(model, "symbol", "GLOBAL")).upper()
                log.info(
                    "[MYTHOS_RUNTIME_OVERRIDE] symbol=%s cooldown_bars %s -> %s",
                    sym,
                    prev,
                    now,
                )
        except Exception:
            return

    def _effective_staleness_limit_seconds(self) -> float:
        """Interval-aware staleness budget to avoid false halts on 15m+ bars."""
        interval_s = float(max(self._interval_seconds(), 60))
        # Allow at least one full bar + grace and always respect configured floor.
        adaptive = max(interval_s * 1.25, interval_s + 120.0)
        return float(max(self.max_data_staleness_seconds, adaptive))

    def _decision_awareness_snapshot(self, window: int = LIVE_AWARENESS_WINDOW) -> Dict[str, object]:
        rows = list(self._closed_trade_stats[-max(int(window), 1):])
        side_map: Dict[str, List[float]] = {"LONG": [], "SHORT": []}
        all_net: List[float] = []
        for row in rows:
            net_r = float(row.get("net_r", 0.0) or 0.0)
            side = str(row.get("side", "")).upper()
            all_net.append(net_r)
            if side in side_map:
                side_map[side].append(net_r)

        def _mk(vals: List[float]) -> Dict[str, float]:
            n = len(vals)
            wins = sum(1 for x in vals if x > 0.0)
            return {
                "n": float(n),
                "expectancy_r": float(sum(vals) / n) if n > 0 else 0.0,
                "win_rate": float(wins / n) if n > 0 else 0.0,
            }

        return {
            "window": int(window),
            "overall": _mk(all_net),
            "by_side": {k: _mk(v) for k, v in side_map.items()},
        }

    def _apply_live_awareness(self, side: str, metric: float, confidence: float, mode: str) -> Tuple[float, float, Dict[str, object]]:
        """Adaptive bias from recent realized quality, shared by V5 and Mythos."""
        snap = self._decision_awareness_snapshot()
        overall = snap["overall"]
        overall_n = int(overall.get("n", 0))
        info: Dict[str, object] = {
            "ready": False,
            "profile": "neutral",
            "n": overall_n,
            "expectancy_r": 0.0,
            "metric_before": float(metric),
            "metric_after": float(metric),
            "confidence_before": float(confidence),
            "confidence_after": float(confidence),
        }
        if overall_n < LIVE_AWARENESS_MIN_TRADES:
            return float(metric), float(confidence), info

        side_key = "LONG" if str(side).upper() == "LONG" else "SHORT"
        side_stats = (snap.get("by_side") or {}).get(side_key, {})
        side_n = int(side_stats.get("n", 0))
        side_exp = float(side_stats.get("expectancy_r", 0.0))
        if side_n < LIVE_AWARENESS_MIN_SIDE_TRADES:
            side_exp = float(overall.get("expectancy_r", 0.0))
            side_n = overall_n

        metric_adj = float(metric)
        conf_adj = float(confidence)
        profile = "neutral"
        if mode == "v5":
            if side_exp <= -0.10:
                metric_adj *= 0.88
                conf_adj -= 0.03
                profile = "defensive"
            elif side_exp >= 0.10:
                metric_adj *= 1.06
                conf_adj += 0.02
                profile = "aggressive"
        else:  # mythos
            if side_exp <= -0.10:
                metric_adj -= 0.0025
                conf_adj -= 0.03
                profile = "defensive"
            elif side_exp >= 0.10:
                metric_adj += 0.0025
                conf_adj += 0.02
                profile = "aggressive"

        conf_adj = float(np.clip(conf_adj, 0.0, 1.0))
        info.update(
            {
                "ready": True,
                "profile": profile,
                "n": side_n,
                "expectancy_r": side_exp,
                "metric_after": metric_adj,
                "confidence_after": conf_adj,
            }
        )
        return metric_adj, conf_adj, info

    def _update_candle_cache(self, symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
        if symbol not in self.candle_cache:
            self.candle_cache[symbol] = new_df.copy()
        else:
            existing = self.candle_cache[symbol]
            combined = pd.concat([existing, new_df], ignore_index=True)
            if 'timestamp' in combined.columns:
                combined = combined.drop_duplicates(subset='timestamp', keep='last')
                combined = combined.sort_values('timestamp').reset_index(drop=True)
            if len(combined) > MAX_CACHE_BARS:
                combined = combined.iloc[-MAX_CACHE_BARS:].reset_index(drop=True)
            self.candle_cache[symbol] = combined
        return self.candle_cache[symbol]

    def _interval_seconds(self) -> int:
        if self.interval == "15m":
            return 15 * 60
        elif self.interval == "5m":
            return 5 * 60
        elif self.interval == "1m":
            return 60
        return 15 * 60

    def _sync_portfolio_from_web(self, source: str = "cycle"):
        """Reconcile in-memory portfolio against the web app's open paper positions.

        Fetches /api/paper/open-positions-summary and:
        - Adds any positions the web app shows as open that are not in memory
        - Removes any positions that the web app considers closed (no longer in
          the OPEN list) but are still in the in-memory portfolio
        This prevents phantom "portfolio full" blocks caused by stale in-memory
        state after web-app SL/TP closes or trainer restarts.
        """
        if not self._web_api_enabled:
            return
        if not self.replit_url:
            return
        try:
            import requests as _req
            resp = _req.get(
                f"{self.replit_url.rstrip('/')}/api/paper/open-positions-summary",
                params={"session_id": self.paper_session_id},
                timeout=8,
            )
            if resp.status_code != 200:
                log.warning(f"[PortfolioSync/{source}] HTTP {resp.status_code}")
                self._note_web_api_status(f"PortfolioSync/{source}", int(resp.status_code))
                return
            self._web_api_404_count = 0

            data = resp.json()
            web_positions = {p["symbol"]: p for p in data.get("positions", [])}
            web_symbols = set(web_positions.keys())
            mem_symbols = set(self.portfolio.open_positions.keys())

            # Remove positions the web app no longer tracks as OPEN
            stale = mem_symbols - web_symbols
            for sym in stale:
                pos = self.portfolio.open_positions.pop(sym, None)
                if pos:
                    self.trade_manager.clear_position(sym)
                    log.info(f"[PortfolioSync/{source}] Removed stale in-memory position for {sym} (closed in web app)")

            # Update existing in-memory positions with latest web stop/TP/risk state.
            shared = mem_symbols & web_symbols
            for sym in shared:
                try:
                    wp = web_positions[sym]
                    pos = self.portfolio.open_positions.get(sym)
                    if pos is None:
                        continue
                    new_sl = float(wp.get("stopLoss") or pos.sl_price)
                    new_tp = float(wp.get("tp2") or pos.tp_price)
                    changed = False
                    if np.isfinite(new_sl) and new_sl > 0 and abs(new_sl - float(pos.sl_price)) > 1e-9:
                        pos.sl_price = new_sl
                        changed = True
                    if np.isfinite(new_tp) and new_tp > 0 and abs(new_tp - float(pos.tp_price)) > 1e-9:
                        pos.tp_price = new_tp
                        changed = True
                    try:
                        new_horizon = int(wp.get("lane_horizon") or wp.get("horizon") or pos.horizon)
                        if new_horizon > 0 and new_horizon != int(pos.horizon):
                            pos.horizon = new_horizon
                            changed = True
                    except Exception:
                        pass
                    try:
                        lev = float(wp.get("leverage") or pos.size_mult)
                        if np.isfinite(lev) and lev > 0:
                            pos.size_mult = lev
                    except Exception:
                        pass
                    if pos.entry_price > 0:
                        pos.risk_pct = max(abs(pos.entry_price - pos.sl_price) / max(pos.entry_price, 1e-9) * 100.0, 0.0)
                    if changed:
                        log.info(f"[PortfolioSync/{source}] Updated {sym} SL/TP from web app state")
                except Exception:
                    continue

            # Add positions the web app shows as OPEN but not in memory
            from portfolio import Position as _Pos
            new_syms = web_symbols - mem_symbols
            for sym in new_syms:
                wp = web_positions[sym]
                entry_price = float(wp.get("entryPrice") or 0)
                sl_price = float(wp.get("stopLoss") or entry_price)
                tp_price = float(wp.get("tp2") or entry_price)
                side = str(wp.get("side", "LONG")).upper()
                atr = abs(entry_price - sl_price) if sl_price else entry_price * 0.005
                risk_pct = atr / entry_price * 100 if entry_price > 0 else 1.0
                lane = str(wp.get("lane") or ("MYTHOS" if self.live_model == "mythos" else "V5"))

                try:
                    restored_horizon = int(float(wp.get("lane_horizon") or wp.get("horizon") or 96))
                except Exception:
                    restored_horizon = 96
                restored_horizon = int(max(restored_horizon, 1))
                self.trade_manager.clear_position(sym)

                pos = _Pos(
                    symbol=sym, side=side,
                    entry_price=entry_price,
                    entry_time=float(wp.get("entryTs") or 0) / 1000.0,
                    atr=atr,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    p_enter=float(wp.get("signalConfidence") or 0.5),
                    size_mult=1.0,
                    risk_pct=risk_pct,
                    bar_index=self.cycle_count,
                    lane=lane,
                    horizon=restored_horizon,
                )
                try:
                    pos.dashboard_trade_id = int(wp.get("id") or 0) or None
                except Exception:
                    pos.dashboard_trade_id = None
                self.portfolio.open_positions[sym] = pos
                log.info(f"[PortfolioSync/{source}] Restored {side} {sym} @ {entry_price:.2f} from web app")

            if stale or new_syms:
                log.info(f"[PortfolioSync/{source}] Sync complete: +{len(new_syms)} restored, -{len(stale)} removed. "
                         f"In-memory: {len(self.portfolio.open_positions)} | Web: {len(web_symbols)}")
            else:
                log.debug(f"[PortfolioSync/{source}] In sync — {len(mem_symbols)} open positions match web app")

        except Exception as e:
            log.warning(f"[PortfolioSync/{source}] Sync failed: {e}")

    def _session_equity_live_usd(self, force_refresh: bool = False) -> Optional[float]:
        """Read live paper equity from dashboard state for kill-switch checks."""
        if not self._web_api_enabled:
            return self._equity_snapshot_live_usd
        if not self.replit_url:
            return None
        now = time.time()
        if (not force_refresh) and (now - self._equity_snapshot_cache_ts) <= 3.0:
            return self._equity_snapshot_live_usd
        try:
            import requests as _req
            resp = _req.get(
                f"{self.replit_url.rstrip('/')}/api/dashboard/state",
                params={
                    "session_id": self.paper_session_id,
                    "engine": self.dashboard_engine,
                    "predictions_limit": 1,
                    "cycles_limit": 1,
                    "trades_limit": 1,
                },
                timeout=6,
            )
            if resp.status_code != 200:
                self._note_web_api_status("dashboard/state", int(resp.status_code))
                return self._equity_snapshot_live_usd
            self._web_api_404_count = 0
            data = resp.json() or {}
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
            eq = float(summary.get("equity_live_usd", summary.get("paper_equity_usd", 0.0)))
            if not np.isfinite(eq):
                return self._equity_snapshot_live_usd
            self._equity_snapshot_live_usd = eq
            self._equity_snapshot_cache_ts = now
            return eq
        except Exception:
            return self._equity_snapshot_live_usd

    def _enforce_equity_hard_stop(self, prices: Optional[Dict[str, float]] = None):
        """
        Emergency capital protection:
        if live equity breaches hard stop, flatten all open risk and latch halt.
        """
        if self._equity_hard_stop_latched:
            return
        if self.equity_hard_stop_usd <= 0.0:
            return
        eq = self._session_equity_live_usd(force_refresh=True)
        if eq is None or eq > self.equity_hard_stop_usd:
            return
        open_positions = dict(self.portfolio.open_positions)
        if not open_positions:
            self._equity_hard_stop_latched = True
            log.error(
                "[EQUITY_HARD_STOP] Latched with equity=%.2f <= hard_stop=%.2f (no open positions)",
                eq,
                self.equity_hard_stop_usd,
            )
            return
        price_map = dict(prices or {})
        missing = [sym for sym in open_positions.keys() if sym not in price_map]
        if missing:
            fetched = self._fetch_live_prices_batch(missing)
            price_map.update(fetched)
        for sym in list(open_positions.keys()):
            if sym not in self.portfolio.open_positions:
                continue
            px = price_map.get(sym)
            if px is None or not np.isfinite(float(px)) or float(px) <= 0.0:
                continue
            self.portfolio.close_position(sym, float(px), "EQUITY_HARD_STOP")
        if self.execution_mode == "paper" and self.record_trades and self._web_api_enabled:
            self._sync_portfolio_from_web(source="equity_hard_stop")
        remaining = list(self.portfolio.open_positions.keys())
        if remaining:
            log.error(
                "[EQUITY_HARD_STOP] Triggered but not fully flattened (missing prices/sync): "
                "equity=%.2f <= hard_stop=%.2f remaining=%s",
                eq,
                self.equity_hard_stop_usd,
                ",".join(sorted(remaining)),
            )
            return
        self._equity_hard_stop_latched = True
        log.error(
            "[EQUITY_HARD_STOP] Triggered: equity=%.2f <= hard_stop=%.2f. "
            "All open positions flattened. New entries halted.",
            eq,
            self.equity_hard_stop_usd,
        )

    def run(self):
        """Main loop — runs continuously until interrupted."""
        mode_label = {"signal_only": "SIGNAL_ONLY", "paper": "PAPER", "live": "LIVE"}.get(self.execution_mode, "UNKNOWN")
        log.info("=" * 80)
        log.info(f"  LIVE RUNNER v5.0 ({mode_label})")
        log.info(f"  [MODE] execution_mode={self.execution_mode} record_trades={self.record_trades} "
                 f"paper={self.paper} live={self.execution_mode == 'live'}")
        log.info(f"  Symbols: {', '.join(self.symbols)}")
        if self.live_model == "mythos":
            log.info("  Runtime model: MYTHOS")
            log.info(
                "  Mythos TM: enabled=%s policy=%s scale_out=%s time_adaptive=%s vol_trailing=%s",
                self.mythos_tm_enabled,
                self.mythos_tm_policy,
                self.mythos_tm_scale_out,
                self.mythos_tm_time_adaptive,
                self.mythos_tm_vol_trailing,
            )
        else:
            log.info(f"  V5 Scoring: lambda={self.v5_score_lambda} threshold={self.v5_score_threshold} min_mu_r={self.v5_min_mu_r}")
        log.info(f"  TP={self.tp_mult}x SL={self.sl_mult}x | Cooldown: {self.cooldown_bars} bars")
        log.info(f"  Per-symbol models: {self.per_symbol_models}")

        self.portfolio.on_close_callback = self._on_position_close
        log.info(f"  15m fetch limit: {self.limit_15m} | Direct HTF fetch: {self.direct_htf}")
        if self.live_model == "mythos":
            log.info("  Pure Mythos mode: HTF warmup/trend gates disabled")
        else:
            log.info(f"  HTF warmup gates: min h1={MIN_H1_BARS} h4={MIN_H4_BARS} bars")
        if self.dry_run:
            log.info(f"  DRY RUN MODE — replaying cached candles")
        log.info("=" * 80)

        self.exchange_time_offset = _get_exchange_time_offset()
        log.info(f"Exchange time offset: {self.exchange_time_offset*1000:.0f}ms")

        self.gpu_self_url = self._detect_gpu_self_url()
        self._register_gpu_url()
        self._start_execution_service()

        self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map = _load_model(
            self.device, model_backend=self.live_model
        )
        self._apply_live_model_runtime_overrides(self.model)
        if self.live_model == "mythos" and not getattr(self.model, "_is_mythos", False):
            log.error("[MYTHOS] live_model=mythos but loaded model is not Mythos. Aborting.")
            sys.exit(1)
        if getattr(self.model, "_is_mythos", False):
            log.info(
                "[MYTHOS] Runtime model active (symbol=%s artifact=%s)",
                getattr(self.model, "symbol", "UNKNOWN"),
                getattr(self.model, "artifact_path", "UNKNOWN"),
            )
        if getattr(self.model, '_is_v6', False):
            log.info(f"[V6] V6Forecaster active — seq_len={self.model._v6_seq_len}, confidence gating enabled (min=0.4)")
        self._init_fetcher()

        for sym in self.symbols:
            self.cooldown_tracker[sym] = 0

        # Restore portfolio state from the web app so the in-memory portfolio
        # reflects any positions that were opened in previous runs or by the
        # web app's paper engine while the trainer was offline.
        if self.execution_mode == "paper" and self.record_trades and self._web_api_enabled:
            self._sync_portfolio_from_web(source="startup")

        if self.dry_run:
            self._run_dry()
            return

        self._should_stop = False
        try:
            while not self._should_stop:
                self._run_cycle()

                if self._should_stop:
                    log.info("Verification cycle limit reached — exiting run loop.")
                    break

                if hasattr(self, 'learning_manager') and self.learning_manager:
                    try:
                        self.learning_manager.check_and_retrain_all()
                    except Exception as e:
                        log.error(f"Learning check failed: {e}")

                interval_s = self._interval_seconds()
                now = time.time() + self.exchange_time_offset
                next_bar = (int(now) // interval_s + 1) * interval_s
                wait = max(next_bar - now + 5, 10)
                log.info(f"Next cycle in {wait:.0f}s...")
                self._monitor_open_positions_during_wait(wait)
        except KeyboardInterrupt:
            log.info("Live runner stopped by user.")
            self._print_summary()

    def _run_dry(self):
        """Dry-run mode: replay cached data instead of fetching live."""
        log.info("Loading cached data for dry run...")

        data_dir = Path("data_cache")

        for symbol in self.symbols:
            parquet_path = data_dir / f"{symbol}_15m.parquet"
            if not parquet_path.exists():
                parquet_path = data_dir / "BTCUSDT_15m.parquet"
                if not parquet_path.exists():
                    log.error(f"No cached data for dry run. Run --predict-only first to download data.")
                    return

            df_full = pd.read_parquet(parquet_path)
            n_bars = min(self.dry_run_candles, len(df_full) - REQUIRED_CANDLES)
            if n_bars <= 0:
                log.error(f"Insufficient cached data for {symbol}")
                continue

            log.info(f"Dry run: {symbol} — replaying {n_bars} bars from cache")

            for i in range(n_bars):
                end_idx = REQUIRED_CANDLES + i
                df_slice = df_full.iloc[end_idx - REQUIRED_CANDLES:end_idx].copy().reset_index(drop=True)

                self.cycle_count += 1
                self.portfolio.set_bar(self.cycle_count)
                prices = {symbol: float(df_slice.iloc[-1]['close'])}
                if self.execution_mode in ("paper", "live") and self.record_trades:
                    self.portfolio.check_exits(prices)

                candidate = self._process_symbol(symbol, df_candles=df_slice)
                if not candidate:
                    continue
                accepted = self.portfolio.filter_and_rank([candidate])
                if not accepted:
                    continue
                for accepted_candidate in accepted:
                    self._execute_candidate(accepted_candidate)

        self._print_summary()

    def _fetch_live_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch latest prices for open-position symbols in one request."""
        syms = [str(s or "").upper() for s in symbols if str(s or "").strip()]
        if not syms:
            return {}
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbols": json.dumps(sorted(set(syms)))}
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return {}
            payload = resp.json()
        except Exception:
            return {}
        rows = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
        out: Dict[str, float] = {}
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper()
                px = float(row.get("price", 0.0))
                if sym and np.isfinite(px) and px > 0.0:
                    out[sym] = px
            except Exception:
                continue
        return out

    def _fetch_symbol_microstructure(self, symbol: str) -> Dict[str, float]:
        """Fetch bid/ask spread context for execution-quality gating."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return {}
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/bookTicker",
                params={"symbol": sym},
                timeout=5,
            )
            if resp.status_code != 200:
                return {}
            payload = resp.json()
            bid = float(payload.get("bidPrice", 0.0))
            ask = float(payload.get("askPrice", 0.0))
            if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0.0 and ask > 0.0):
                return {}
            mid = (bid + ask) * 0.5
            if not np.isfinite(mid) or mid <= 0.0:
                return {}
            spread_bps = max(ask - bid, 0.0) / mid * 10000.0
            return {
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(mid),
                "spread_bps": float(max(spread_bps, 0.0)),
            }
        except Exception:
            return {}

    def _adaptive_spread_limit_bps(self) -> float:
        base = float(max(self.exec_max_spread_bps, self.exec_spread_min_bps))
        if (not self.exec_spread_adaptive_enable) or (len(self._recent_entry_slippage_bps) < 8):
            return base
        window = int(max(self.exec_spread_window, 5))
        slips = self._recent_entry_slippage_bps[-window:]
        avg_slip = float(np.mean(slips)) if slips else 0.0
        target = float(max(self.exec_spread_target_slippage_bps, 0.1))
        ratio = avg_slip / target
        factor = 1.0
        if ratio > 1.0:
            factor = 1.0 / (1.0 + 0.60 * (ratio - 1.0))
        elif ratio < 0.8:
            factor = 1.0 + 0.20 * (0.8 - ratio)
        factor = float(np.clip(factor, 0.50, 1.20))
        return float(np.clip(base * factor, self.exec_spread_min_bps, max(base * 1.5, self.exec_spread_min_bps)))

    def _record_execution_quality(self, spread_bps: Optional[float], slippage_bps: Optional[float]):
        if spread_bps is not None and np.isfinite(float(spread_bps)):
            self._recent_entry_spread_bps.append(float(max(spread_bps, 0.0)))
        if slippage_bps is not None and np.isfinite(float(slippage_bps)):
            self._recent_entry_slippage_bps.append(float(max(slippage_bps, 0.0)))
        window = int(max(self.exec_spread_window, 5))
        if len(self._recent_entry_spread_bps) > window:
            self._recent_entry_spread_bps = self._recent_entry_spread_bps[-window:]
        if len(self._recent_entry_slippage_bps) > window:
            self._recent_entry_slippage_bps = self._recent_entry_slippage_bps[-window:]

    def _fetch_intrabar_high_low_batch(self, symbols: List[str], interval: str = "1m", limit: int = 2) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Fetch short-horizon candle highs/lows for barrier-touch detection."""
        highs: Dict[str, float] = {}
        lows: Dict[str, float] = {}
        syms = [str(s or "").upper() for s in symbols if str(s or "").strip()]
        for sym in sorted(set(syms)):
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": sym, "interval": interval, "limit": max(int(limit), 1)},
                    timeout=5,
                )
                if resp.status_code != 200:
                    continue
                rows = resp.json()
                if not isinstance(rows, list) or not rows:
                    continue
                hi_vals = []
                lo_vals = []
                for row in rows:
                    if not isinstance(row, list) or len(row) < 5:
                        continue
                    try:
                        hi = float(row[2])
                        lo = float(row[3])
                    except Exception:
                        continue
                    if np.isfinite(hi) and hi > 0.0:
                        hi_vals.append(hi)
                    if np.isfinite(lo) and lo > 0.0:
                        lo_vals.append(lo)
                if hi_vals:
                    highs[sym] = max(hi_vals)
                if lo_vals:
                    lows[sym] = min(lo_vals)
            except Exception:
                continue
        return highs, lows

    def _monitor_open_positions_during_wait(self, wait_seconds: float):
        """
        During inter-cycle sleep, keep checking live price against SL/TP.
        This allows auto-close soon after breach without waiting a full cycle.
        """
        if wait_seconds <= 0:
            return
        if self.execution_mode not in ("paper", "live") or not self.record_trades:
            time.sleep(wait_seconds)
            return
        deadline = time.time() + float(wait_seconds)
        while not self._should_stop:
            now = time.time()
            if now >= deadline:
                break
            # Keep in-memory positions in sync with dashboard/manual actions
            # between main cycles so manual closes are reflected quickly.
            if self.execution_mode == "paper" and self.record_trades and self._web_api_enabled:
                self._sync_portfolio_from_web(source="wait_loop")
            open_symbols = list(self.portfolio.open_positions.keys())
            if open_symbols:
                prices = self._fetch_live_prices_batch(open_symbols)
                if prices:
                    highs, lows = self._fetch_intrabar_high_low_batch(open_symbols, interval="1m", limit=2)
                    self.portfolio.check_exits(prices, highs=highs, lows=lows)
                    self._enforce_equity_hard_stop(prices)
            remaining = max(deadline - time.time(), 0.0)
            sleep_s = min(OPEN_POSITION_MONITOR_INTERVAL_S, remaining)
            if sleep_s <= 0:
                break
            time.sleep(sleep_s)

    def _run_trade_manager(self, prices: Dict[str, float],
                           highs: Dict[str, float], lows: Dict[str, float]):
        """Run Smart Trade Manager over all open positions.

        For each open position, compute current HTF score and p_enter,
        then ask TradeManager for an action.
        """
        open_positions = dict(self.portfolio.open_positions)
        if not open_positions:
            return

        for symbol, pos in open_positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            tm_state_before = copy.deepcopy(self.trade_manager.get_state(symbol))

            candle_high = highs.get(symbol)
            candle_low = lows.get(symbol)

            current_htf_score = getattr(pos, 'htf_score', None)
            current_p_enter = float(getattr(pos, "p_enter", 0.0) or 0.0)

            action = self.trade_manager.update_position(
                symbol=symbol,
                pos=pos,
                current_price=price,
                current_bar=self.cycle_count,
                current_htf_score=current_htf_score,
                current_p_enter=current_p_enter,
                candle_high=candle_high,
                candle_low=candle_low,
            )

            if action.action == "HOLD":
                continue

            if action.action in ("MOVE_SL", "TRAIL_SL") and action.new_sl is not None:
                pos.sl_price = action.new_sl
                if action.reason == "BREAKEVEN":
                    pos.breakeven_moved = True
                if pos.dashboard_trade_id:
                    try:
                        self._update_trade_sl(pos.dashboard_trade_id, action.new_sl)
                    except Exception as e:
                        log.warning(f"Failed to update SL on dashboard for {symbol}: {e}")

            elif action.action == "CLOSE_PARTIAL":
                close_pct = float(action.close_pct if action.close_pct is not None else 25.0)
                try:
                    applied = self._apply_partial_close_from_tm(pos, close_pct=close_pct, reason=action.reason)
                    if applied:
                        log.info(
                            "[TM_PARTIAL] sym=%s pct=%.1f reason=%s",
                            symbol,
                            close_pct,
                            action.reason,
                        )
                    else:
                        if tm_state_before is None:
                            self.trade_manager.clear_position(symbol)
                        else:
                            self.trade_manager.position_state[symbol] = tm_state_before
                        log.warning(
                            "[TM_PARTIAL] sym=%s pct=%.1f reason=%s not applied; reverted TM state",
                            symbol,
                            close_pct,
                            action.reason,
                        )
                except Exception as e:
                    if tm_state_before is None:
                        self.trade_manager.clear_position(symbol)
                    else:
                        self.trade_manager.position_state[symbol] = tm_state_before
                    log.warning(f"Failed partial-close TM action for {symbol}: {e}")

            elif action.action == "CLOSE_FULL":
                if symbol not in self.portfolio.open_positions:
                    continue
                close_price = action.close_price or price
                self.portfolio.close_position(symbol, close_price, action.reason)

    def _run_cycle(self):
        """One 15m cycle: fetch, predict, rank, execute for all symbols."""
        self.cycle_count += 1
        self.portfolio.set_bar(self.cycle_count)

        # Sync portfolio state from web app every cycle in paper mode so that
        # positions closed by the web app engine (SL/TP) are removed from
        # the in-memory portfolio and don't phantom-block new entries.
        if self.execution_mode == "paper" and self.record_trades and self._web_api_enabled:
            self._sync_portfolio_from_web(source=f"cycle_{self.cycle_count}")

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        log.info("")
        log.info(f"{'='*60}")
        log.info(f"CYCLE {self.cycle_count} — {timestamp}")
        log.info(f"{'='*60}")

        prices = {}
        highs = {}
        lows = {}
        candle_dfs = {}
        _cycle_fetch_ok = False
        for symbol in self.symbols:
            try:
                df = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval, limit=self.limit_15m)
            except Exception as _fetch_err:
                log.warning("[HALT_TRACK] fetch error for %s: %s", symbol, _fetch_err)
                df = None
            if df is not None and len(df) > 0:
                prices[symbol] = float(df.iloc[-1]['close'])
                if 'high' in df.columns:
                    highs[symbol] = float(df.iloc[-1]['high'])
                if 'low' in df.columns:
                    lows[symbol] = float(df.iloc[-1]['low'])
                candle_dfs[symbol] = df
                _cycle_fetch_ok = True
                if 'timestamp' in df.columns:
                    try:
                        last_ts = float(df.iloc[-1]['timestamp'])
                        self._last_candle_time = (last_ts / 1000.0
                                                   if last_ts > 1e10 else last_ts)
                    except Exception:
                        self._last_candle_time = time.time()
                else:
                    self._last_candle_time = time.time()
            else:
                log.warning("[HALT_TRACK] No candle data returned for %s", symbol)
        if _cycle_fetch_ok:
            self._consecutive_api_errors = 0
        else:
            self._consecutive_api_errors += 1
            log.warning("[HALT_TRACK] consecutive_api_errors=%d (all symbols failed this cycle)",
                        self._consecutive_api_errors)
        if self.execution_mode in ("paper", "live") and self.record_trades:
            self.portfolio.check_exits(prices, highs=highs, lows=lows)
            self._enforce_equity_hard_stop(prices)
            if self.live_model != "mythos" or self.mythos_tm_enabled:
                self._run_trade_manager(prices, highs, lows)

        awareness = self._decision_awareness_snapshot()
        overall = awareness.get("overall", {})
        log.info(
            "[LIVE_AWARENESS] model=%s n=%s exp=%+.3f wr=%.1f%% | long_exp=%+.3f short_exp=%+.3f",
            self.live_model,
            int(overall.get("n", 0)),
            float(overall.get("expectancy_r", 0.0)),
            100.0 * float(overall.get("win_rate", 0.0)),
            float((awareness.get("by_side", {}).get("LONG", {}) or {}).get("expectancy_r", 0.0)),
            float((awareness.get("by_side", {}).get("SHORT", {}) or {}).get("expectancy_r", 0.0)),
        )

        candidates = []
        for symbol in self.symbols:
            result = self._process_symbol(symbol)
            if result:
                candidates.append(result)

        if not candidates:
            log.info("No candidates this cycle.")
            return

        accepted = self.portfolio.filter_and_rank(candidates)
        accepted_ids = {id(c) for c in accepted}
        for c in candidates:
            if id(c) in accepted_ids:
                continue
            reject_reason = str(c.get("reject_reason") or "portfolio_reject")
            log.info("  [REJECT] %s %s reason=%s", c.get("symbol", "?"), c.get("side", "?"), reject_reason)
            li = dict(c.get("v5_info") or {})
            li["hold_reason"] = reject_reason
            try:
                self._push_cycle_log(
                    symbol=str(c.get("symbol", "")),
                    price=float(c.get("current_price", 0.0) or 0.0),
                    p_enter=float(c.get("p_enter", 0.0) or 0.0),
                    htf=c.get("htf") or {},
                    direction=str(c.get("side", "NEUTRAL")),
                    decision="ENTER_REJECTED",
                    reasons=[reject_reason],
                    lane_info=li,
                    decision_stage="execution",
                    execution_status="rejected",
                )
            except Exception:
                continue
        if not accepted:
            log.info("All candidates rejected by portfolio rules.")
            return

        for c in accepted:
            self._execute_candidate(c)

    def _compute_htf_from_direct(self, htf_direct: Dict[str, pd.DataFrame],
                                  df_candles: pd.DataFrame) -> dict:
        """Compute HTF gate values from directly-fetched 1H/4H candle data.

        Mirrors _apply_htf_gates but uses real HTF bars instead of resampled features.
        Uses the last COMPLETED bar (second-to-last) for each timeframe to avoid lookahead.
        """
        result = {
            'trend_aligned': False, 'slope_ok': False, 'range_ok': False,
            'side': 'NEUTRAL', 'h1_trend': 0, 'h4_trend': 0,
            'h1_slope': 0.0, 'h1_range_pos': 0.5,
        }

        for tf_key, tf_label in [('1h', 'h1'), ('4h', 'h4')]:
            df_tf = htf_direct.get(tf_key)
            if df_tf is None or len(df_tf) < 22:
                continue

            closes = df_tf['close'].values.astype(float)
            highs = df_tf['high'].values.astype(float)
            lows = df_tf['low'].values.astype(float)

            sma20 = pd.Series(closes).rolling(20, min_periods=1).mean().values
            atr_vals = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
                atr_vals.append(tr)
            atr_series = pd.Series([atr_vals[0]] + atr_vals).rolling(14, min_periods=1).mean().values

            idx = -2
            slope = (sma20[idx] - sma20[max(idx-3, 0)]) / (atr_series[idx] + 1e-9)
            trend_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)

            result[f'{tf_label}_trend'] = trend_sign

            if tf_label == 'h1':
                result['h1_slope'] = float(slope)
                htf_high = highs[idx]
                htf_low = lows[idx]
                current_close = float(df_candles.iloc[-1]['close'])
                result['h1_range_pos'] = float(np.clip(
                    (current_close - htf_low) / (htf_high - htf_low + 1e-9), 0.0, 1.0
                ))

        h1_trend = result['h1_trend']
        h4_trend = result['h4_trend']
        result['trend_aligned'] = (h1_trend == h4_trend) and (h1_trend != 0)
        result['slope_ok'] = abs(result['h1_slope']) > 0.05

        h1_range_pos = result['h1_range_pos']
        result['range_ok'] = True
        if h1_trend > 0 and h1_range_pos < 0.2:
            result['range_ok'] = False
        if h1_trend < 0 and h1_range_pos > 0.8:
            result['range_ok'] = False

        if h1_trend > 0:
            result['side'] = 'LONG'
        elif h1_trend < 0:
            result['side'] = 'SHORT'
        else:
            result['side'] = 'NEUTRAL'

        log.info(f"  HTF gates (direct): aligned={result['trend_aligned']} "
                 f"h1={h1_trend:+d} h4={h4_trend:+d} slope={result['h1_slope']:.3f} "
                 f"range_pos={h1_range_pos:.2f}")

        return result

    def _process_symbol_mythos(
        self,
        symbol: str,
        df_candles: pd.DataFrame,
        model,
    ) -> Optional[dict]:
        """Process one symbol using pure Mythos runtime inference."""
        try:
            from mythos.features import build_feature_frame

            features_df = build_feature_frame(df_candles)
        except Exception as e:
            log.error(f"{symbol}: failed to build Mythos features: {e}")
            return None
        if features_df is None or features_df.empty:
            return None

        row = features_df.iloc[-1].to_dict()
        pred = model.predict_from_feature_row(row)
        side = str(pred.get("side", "NEUTRAL")).upper()
        edge = float(pred.get("edge", 0.0))
        confidence = float(pred.get("confidence", 0.0))
        uncertainty = float(pred.get("uncertainty", 1.0))
        abstain = bool(pred.get("abstain", False))
        reason = str(pred.get("reason", ""))
        edge, confidence, aware = self._apply_live_awareness(
            side=side, metric=edge, confidence=confidence, mode="mythos"
        )

        # Pure Mythos mode: no HTF trend scoring/gating overlay.
        htf = {"h1_trend": 0, "h4_trend": 0, "slope_ok": True, "range_ok": True}

        current_price = float(df_candles.iloc[-1]["close"])
        atr = _compute_atr(df_candles)
        edge_floor = float(getattr(model, "live_edge_threshold", 0.0))
        conf_floor = float(getattr(model, "live_min_confidence", 0.0))
        htf_score = 0
        cfg_obj = getattr(model, "cfg", None)
        min_expected_r = float(max(getattr(cfg_obj, "min_expected_r", 0.01), 1e-6))
        min_lev = float(max(getattr(cfg_obj, "min_size_mult", 1.0), 1.0))
        cfg_max_lev = float(
            max(
                getattr(cfg_obj, "max_size_mult", min_lev),
                getattr(cfg_obj, "max_leverage", min_lev),
                min_lev,
            )
        )
        max_lev = float(np.clip(cfg_max_lev, min_lev, 250.0))
        edge_n = float(np.clip(max(edge, 0.0) / max(2.5 * min_expected_r, 1e-6), 0.0, 1.0))
        conf_n = float(np.clip(confidence, 0.0, 1.0))
        unc_n = float(np.clip(1.0 / (1.0 + max(uncertainty, 0.0)), 0.0, 1.0))
        size_quality = float(np.clip(0.45 * conf_n + 0.35 * edge_n + 0.20 * unc_n, 0.0, 1.0))
        lane_size_mult = float(min_lev + (max_lev - min_lev) * size_quality)
        regime_raw = str(pred.get("regime", "") or "").strip().upper()
        regime_horizon_bias = 0
        if "TREND" in regime_raw or "BREAKOUT" in regime_raw:
            regime_horizon_bias = 10
        elif "MOMENTUM" in regime_raw:
            regime_horizon_bias = 5
        elif "MEAN" in regime_raw or "CHOP" in regime_raw:
            regime_horizon_bias = -5
        quality_horizon = float(np.clip(0.55 * conf_n + 0.35 * edge_n + 0.10 * unc_n, 0.0, 1.0))
        uncertainty_penalty = float(max(uncertainty - 0.85, 0.0) * 8.0)
        lane_horizon = int(
            np.clip(
                round(12 + (quality_horizon * 40.0) + regime_horizon_bias - uncertainty_penalty),
                8,
                96,
            )
        )
        tp_mult_used, sl_mult_used = _resolve_mythos_live_tp_sl(
            cfg_obj=cfg_obj,
            side=side,
            edge=edge,
            confidence=confidence,
            uncertainty=uncertainty,
            feature_row=row,
            fallback_tp_mult=self.tp_mult,
            fallback_sl_mult=self.sl_mult,
        )

        mythos_info = {
            "lane": "MYTHOS",
            "model_name": "mythos_runtime_live",
            "v5_score": round(edge, 4),
            "v5_threshold": edge_floor,
            "v5_side": side,
            "threshold_used": edge_floor,
            "htf_score": htf_score,
            "mythos_uncertainty": round(uncertainty, 4),
            "mythos_regime": pred.get("regime"),
            "mythos_reason": reason,
            "mythos_expert": pred.get("expert_name"),
            "awareness_profile": aware.get("profile"),
            "awareness_expectancy_r": round(float(aware.get("expectancy_r", 0.0)), 4),
            "awareness_n": int(aware.get("n", 0)),
            "lane_size_mult": round(lane_size_mult, 4),
            "mythos_size_quality": round(size_quality, 4),
            "tp_mult_used": round(float(tp_mult_used), 4),
            "sl_mult_used": round(float(sl_mult_used), 4),
            "lane_horizon": lane_horizon,
            "mythos_time_horizon": lane_horizon,
            "mythos_horizon_profile": "regime_adaptive",
        }
        if atr and atr > 0 and not (atr != atr):
            sl_dist = float(sl_mult_used * atr)
            tp_dist = float(tp_mult_used * atr)
            if side == "LONG":
                mythos_info["sl_price"] = round(current_price - sl_dist, 6)
                mythos_info["tp_price"] = round(current_price + tp_dist, 6)
            else:
                mythos_info["sl_price"] = round(current_price + sl_dist, 6)
                mythos_info["tp_price"] = round(current_price - tp_dist, 6)
            mythos_info["sl_dist_atr"] = round(sl_dist / atr, 3)
            mythos_info["tp_dist_atr"] = round(tp_dist / atr, 3)

        if side not in {"LONG", "SHORT"} or abstain or edge < edge_floor or confidence < conf_floor:
            blocks = []
            if side not in {"LONG", "SHORT"}:
                blocks.append(f"neutral_side={side}")
            if abstain:
                blocks.append(f"abstain:{reason or 'router_abstain'}")
            if edge < edge_floor:
                blocks.append(f"edge={edge:.4f}<floor={edge_floor:.4f}")
            if confidence < conf_floor:
                blocks.append(f"conf={confidence:.3f}<min={conf_floor:.3f}")
            hold_reason = "; ".join(blocks) if blocks else "mythos_hold"
            mythos_info["hold_reason"] = hold_reason
            log.info(
                "[MYTHOS_DECISION] sym=%s side=%s edge=%.4f conf=%.3f aware=%s exp=%+.3f n=%s abstain=%s -> HOLD reason=%s",
                symbol, side, edge, confidence, aware.get("profile"), aware.get("expectancy_r", 0.0), aware.get("n", 0), abstain, hold_reason,
            )
            try:
                self._push_cycle_log(
                    symbol=symbol,
                    price=current_price,
                    p_enter=confidence,
                    htf=htf,
                    direction=side,
                    decision="HOLD",
                    reasons=[hold_reason],
                    lane_info=mythos_info,
                    e_net_pred=edge,
                )
            except Exception as e:
                log.warning(f"Failed to push Mythos HOLD cycle log for {symbol}: {e}")
            return None

        try:
            log.info(
                "[MYTHOS_DECISION] sym=%s side=%s edge=%.4f conf=%.3f aware=%s exp=%+.3f n=%s expert=%s regime=%s -> ENTER",
                symbol, side, edge, confidence, aware.get("profile"), aware.get("expectancy_r", 0.0), aware.get("n", 0),
                pred.get("expert_name"), pred.get("regime"),
            )
            self._push_cycle_log(
                symbol=symbol,
                price=current_price,
                p_enter=confidence,
                htf=htf,
                direction=side,
                decision="ENTER",
                reasons=[f"mythos_edge={edge:.4f} conf={confidence:.3f} expert={pred.get('expert_name', '?')}"],
                lane_info=mythos_info,
                e_net_pred=edge,
            )
        except Exception as e:
            log.warning(f"Failed to push Mythos ENTER cycle log for {symbol}: {e}")

        sl_pct = float(max(sl_mult_used, 0.1)) * atr / max(current_price, 1e-9)
        risk_pct = min(2.0 * sl_pct * 100, 5.0)
        return {
            "symbol": symbol,
            "side": side,
            "p_enter": confidence,
            "current_price": current_price,
            "atr": atr,
            "htf": htf,
            "risk_pct": risk_pct,
            "df_candles": df_candles,
            "expected_net_r": edge,
            "v5_info": mythos_info,
            "features_df": features_df,
        }

    def _process_symbol(self, symbol: str, df_candles: Optional[pd.DataFrame] = None) -> Optional[dict]:
        """Process one symbol: fetch data, compute features, run inference, apply gates."""
        if df_candles is None:
            df_candles = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval, limit=self.limit_15m)
        if df_candles is None:
            return None

        df_candles = self._update_candle_cache(symbol, df_candles)
        model, engineer, feature_columns, temperature, symbol_map = self._get_model_for_symbol(symbol)
        if getattr(model, "_is_mythos", False):
            return self._process_symbol_mythos(
                symbol=symbol,
                df_candles=df_candles,
                model=model,
            )

        htf_direct = None
        use_direct_htf = False

        if self.direct_htf:
            htf_direct = _fetch_htf_candles_direct(self.fetcher, symbol)
            if htf_direct:
                n_h1 = len(htf_direct.get('1h', []))
                n_h4 = len(htf_direct.get('4h', []))
                log.info(f"  {symbol} direct HTF: h1={n_h1} bars, h4={n_h4} bars")
                if n_h1 < MIN_H1_BARS or n_h4 < MIN_H4_BARS:
                    warmup_msg = (f"{symbol} WARMUP (direct HTF): h1_bars={n_h1} h4_bars={n_h4} "
                                  f"(min {MIN_H1_BARS}/{MIN_H4_BARS}) -> skip gates/trading")
                    if not self.warmup_logged.get(symbol):
                        log.warning(warmup_msg)
                        self.warmup_logged[symbol] = True
                    try:
                        current_price = float(df_candles.iloc[-1]['close'])
                        self._push_cycle_log(
                            symbol=symbol, price=current_price, p_enter=0.0,
                            htf={'h1_trend': 0, 'h4_trend': 0, 'slope_ok': False, 'range_ok': False},
                            direction="NEUTRAL", decision="WARMUP",
                            reasons=[warmup_msg],
                        )
                    except Exception:
                        pass
                    return None
                else:
                    self.warmup_logged[symbol] = False
                    use_direct_htf = True
            else:
                log.warning(f"{symbol}: direct HTF fetch failed, falling back to resampled warmup check")

        if not use_direct_htf:
            warmup_reason = _check_htf_warmup(df_candles, symbol)
            if warmup_reason:
                if not self.warmup_logged.get(symbol):
                    log.warning(warmup_reason)
                    self.warmup_logged[symbol] = True
                try:
                    current_price = float(df_candles.iloc[-1]['close'])
                    self._push_cycle_log(
                        symbol=symbol, price=current_price, p_enter=0.0,
                        htf={'h1_trend': 0, 'h4_trend': 0, 'slope_ok': False, 'range_ok': False},
                        direction="NEUTRAL", decision="WARMUP",
                        reasons=[warmup_reason],
                    )
                except Exception:
                    pass
                return None
            else:
                self.warmup_logged[symbol] = False

        v6_seq_len = getattr(model, '_v6_seq_len', 1) if getattr(model, '_is_v6', False) else 1

        scaled, features_df = _compute_features_for_symbol(
            df_candles, engineer, feature_columns, symbol,
            funding_cache=self._funding_cache,
            oi_cache=self._oi_cache,
            ls_ratio_cache=self._ls_ratio_cache,
            feature_check_logged=self._feature_check_logged,
            seq_len=v6_seq_len,
        )
        if scaled is None:
            return None

        sym_id = None
        if symbol_map and symbol in symbol_map:
            sym_id = symbol_map[symbol]

        infer_result = _run_inference(model, scaled, self.device,
                                      temperature=temperature, symbol_id=sym_id)
        p_enter = infer_result['p_enter']
        e_net_pred = infer_result['e_net_pred']
        enter_logit = infer_result['enter_logit']
        temperature_used = infer_result['temperature_used']

        if use_direct_htf and htf_direct:
            htf = self._compute_htf_from_direct(htf_direct, df_candles)
        else:
            htf = _apply_htf_gates(features_df)

        current_price = float(df_candles.iloc[-1]['close'])
        atr = _compute_atr(df_candles)

        v5_probs = infer_result.get('v5_action_probs', [0.5, 0.25, 0.25])
        p_hold = v5_probs[0]
        p_long = v5_probs[1]
        p_short = v5_probs[2]
        v5_mfe = infer_result.get('v5_mfe', 0.0)
        v5_mae = infer_result.get('v5_mae', 0.0)
        ret_mu = infer_result.get('v5_ret_mu', e_net_pred)

        risk = max(v5_mae, self.v5_mae_floor)  # floor prevents score explosion in low-vol
        abs_mu = abs(ret_mu)
        mu_over_risk = abs_mu / risk if risk > 0 else 0.0
        edge_long = p_long * mu_over_risk
        edge_short = p_short * mu_over_risk
        best_edge = max(edge_long, edge_short)
        side = "LONG" if edge_long >= edge_short else "SHORT"
        p_side = p_long if side == "LONG" else p_short
        penalty = self.v5_score_lambda * (1.0 - p_side) * mu_over_risk
        v5_score = best_edge - penalty

        if abs_mu < self.v5_min_mu_r:
            v5_score = -999.0

        # Side-aware scoring: both heads must agree — SHORT needs mu_R<0, LONG needs mu_R>0
        if self.side_aware_scoring and v5_score > -999.0:
            if side == "SHORT" and ret_mu >= 0:
                log.info(f"  {symbol}: side_aware BLOCK SHORT — ret_mu={ret_mu:.4f} is non-negative")
                v5_score = -999.0
            elif side == "LONG" and ret_mu <= 0:
                log.info(f"  {symbol}: side_aware BLOCK LONG — ret_mu={ret_mu:.4f} is non-positive")
                v5_score = -999.0

        v5_score, p_enter, aware = self._apply_live_awareness(
            side=side, metric=v5_score, confidence=p_enter, mode="v5"
        )

        # Direction balance cap: track recent signal sides, cut size when one direction dominates
        _direction_size_mult = 1.0
        if self.direction_balance_cap and v5_score > -999.0:
            self._recent_signal_sides.append(side)
            if len(self._recent_signal_sides) > 20:
                self._recent_signal_sides = self._recent_signal_sides[-20:]
            if len(self._recent_signal_sides) >= 5:
                short_frac = self._recent_signal_sides.count("SHORT") / len(self._recent_signal_sides)
                long_frac = 1.0 - short_frac
                dominant_frac = max(short_frac, long_frac)
                if dominant_frac >= 0.85:
                    _direction_size_mult = 0.25
                    log.info(f"  {symbol}: direction_balance 0.25x — dominant={dominant_frac:.0%}")
                elif dominant_frac >= self.direction_balance_threshold:
                    _direction_size_mult = 0.5
                    log.info(f"  {symbol}: direction_balance 0.5x — dominant={dominant_frac:.0%}")

        htf = _apply_htf_gates(features_df) if not (use_direct_htf and htf_direct) else self._compute_htf_from_direct(htf_direct, df_candles)
        htf_score = _compute_htf_score(htf, side)

        v6_confidence = infer_result.get('v6_confidence', None)
        v6_conf_str = f" conf={v6_confidence:.3f}" if v6_confidence is not None else ""
        side_aware_str = " [SIDE-AWARE]" if self.side_aware_scoring else ""
        log.info(f"  {symbol}: price={current_price:.2f} v5_score={v5_score:.4f} thr={self.v5_score_threshold} "
                 f"p_enter={p_enter:.4f} ret_mu={ret_mu:.4f} mfe={v5_mfe:.4f} mae={v5_mae:.4f} "
                 f"p_long={p_long:.3f} p_short={p_short:.3f} side={side}{v6_conf_str}{side_aware_str}")

        V6_CONFIDENCE_MIN = 0.4

        reasons = []
        decision = "HOLD"

        if self.cooldown_tracker.get(symbol, 0) > 0:
            bars_left = self.cooldown_tracker[symbol]
            self.cooldown_tracker[symbol] -= 1
            reasons.append(f"Cooldown active ({bars_left} bars left)")
            decision = "COOLDOWN"
            v5_info = {
                'v5_score': round(v5_score, 4), 'v5_threshold': self.v5_score_threshold,
                'v5_side': side, 'v5_mfe': round(v5_mfe, 4), 'v5_mae': round(v5_mae, 4),
                'ret_mu': round(ret_mu, 4), 'p_long': round(p_long, 4), 'p_short': round(p_short, 4),
                'htf_score': htf_score, 'threshold_used': self.v5_score_threshold,
                'hold_reason': f'COOLDOWN ({bars_left} bars)',
            }
            try:
                self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=decision, reasons=reasons,
                    lane_info=v5_info,
                    e_net_pred=e_net_pred, enter_logit=enter_logit,
                    temperature_used=temperature_used)
            except Exception as e:
                log.warning(f"Failed to push cycle log for {symbol}: {e}")
            return None

        # ── Unified threshold for all directions ────────────────────────────────
        effective_threshold = self.v5_score_threshold

        score_pass = v5_score >= effective_threshold
        if v6_confidence is not None and v6_confidence < V6_CONFIDENCE_MIN:
            score_pass = False

        v5_info = {
            'v5_score': round(v5_score, 4), 'v5_threshold': effective_threshold,
            'v5_side': side, 'v5_mfe': round(v5_mfe, 4), 'v5_mae': round(v5_mae, 4),
            'ret_mu': round(ret_mu, 4), 'p_long': round(p_long, 4), 'p_short': round(p_short, 4),
            'htf_score': htf_score, 'threshold_used': effective_threshold,
            'awareness_profile': aware.get('profile'),
            'awareness_expectancy_r': round(float(aware.get('expectancy_r', 0.0)), 4),
            'awareness_n': int(aware.get('n', 0)),
        }
        if v6_confidence is not None:
            v5_info['v6_confidence'] = round(v6_confidence, 4)

        if not score_pass:
            hold_reasons = []
            if v6_confidence is not None and v6_confidence < V6_CONFIDENCE_MIN:
                hold_reasons.append(f"v6_conf={v6_confidence:.3f}<{V6_CONFIDENCE_MIN}")
            if abs_mu < self.v5_min_mu_r:
                hold_reasons.append(f"abs_mu={abs_mu:.4f}<min={self.v5_min_mu_r}")
            elif v5_score < effective_threshold:
                hold_reasons.append(f"v5_score={v5_score:.4f}<thr={effective_threshold:.4f}({side})")
            hold_reason = '; '.join(hold_reasons)
            reasons.append(hold_reason)
            v5_info['hold_reason'] = hold_reason

            log.info(f"[V5_DECISION] sym={symbol} score={v5_score:.4f} thr={effective_threshold:.4f}({side}) "
                     f"side={side} -> HOLD reason={hold_reason}")

            try:
                self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=decision, reasons=reasons,
                    lane_info=v5_info,
                    e_net_pred=e_net_pred, enter_logit=enter_logit,
                    temperature_used=temperature_used)
            except Exception as e:
                log.warning(f"Failed to push cycle log for {symbol}: {e}")
            return None

        decision = "ENTER"
        reasons.append(f"v5_score={v5_score:.4f}>=thr={effective_threshold:.4f}({side}) p_enter={p_enter:.1%}")
        v5_info['hold_reason'] = None

        # ── REGIME-AWARE GATE ────────────────────────────────────────────────────
        # Gate 1 — EMA REGIME BLOCK: H4 SMA20 as the broad trend boundary.
        #   BULL regime (price > H4_SMA20): block SHORTs (model is contrarian).
        #   BEAR regime (price < H4_SMA20): block LONGs (buying into downtrend).
        #   3-bar confirmation prevents whipsaw at boundaries.
        # Gate 2 — H4 DIRECTION GATE: H4 trend sign must agree with trade side.
        _regime_blocked = False
        _regime_reason = ''
        try:
            n_candles = len(df_candles)
            H4_PERIOD = 16          # 16 × 15m bars = 4 hours
            H4_SMA_BARS = 20        # SMA over last 20 H4 bars = 80 hours
            h4_closes: list = []
            for _i in range((n_candles - (H4_PERIOD - 1)) // H4_PERIOD):
                _idx = _i * H4_PERIOD + (H4_PERIOD - 1)
                if _idx < n_candles:
                    h4_closes.append(float(df_candles.iloc[_idx]['close']))

            if len(h4_closes) >= H4_SMA_BARS:
                h4_sma20 = sum(h4_closes[-H4_SMA_BARS:]) / H4_SMA_BARS
                regime_now = 'BULL' if current_price > h4_sma20 else 'BEAR'

                # 3-bar rolling confirmation — prevents reacting to single wicks
                _hist = self._regime_history.setdefault(symbol, [])
                _hist.append(regime_now)
                if len(_hist) > 3:
                    self._regime_history[symbol] = _hist[-3:]
                    _hist = self._regime_history[symbol]
                if len(_hist) >= 3 and all(r == _hist[-1] for r in _hist[-3:]):
                    self._regime_confirmed[symbol] = _hist[-1]
                confirmed_regime = self._regime_confirmed.get(symbol, regime_now)

                # Gate 1: EMA regime block
                if confirmed_regime == 'BULL' and side == 'SHORT':
                    _regime_blocked = True
                    _regime_reason = (
                        f"REGIME_BLOCK_BULL: price=${current_price:.0f} > "
                        f"H4_SMA20=${h4_sma20:.0f} — shorting a bull market")
                elif confirmed_regime == 'BEAR' and side == 'LONG':
                    _regime_blocked = True
                    _regime_reason = (
                        f"REGIME_BLOCK_BEAR: price=${current_price:.0f} < "
                        f"H4_SMA20=${h4_sma20:.0f} — longing a bear market")

                # Gate 2: H4 direction hard gate (only if not already blocked)
                if not _regime_blocked:
                    _h4_trend_val = int(htf.get('h4_trend', 0))
                    _h4_dir_sign = 1 if side == 'LONG' else -1
                    if _h4_trend_val != 0 and _h4_trend_val != _h4_dir_sign:
                        _regime_blocked = True
                        _regime_reason = (
                            f"H4_DIR_GATE: H4_trend={_h4_trend_val:+d} opposes "
                            f"side={side} — H4 must agree with trade direction")

                # Annotate the cycle log with regime metadata
                v5_info['h4_sma20'] = round(h4_sma20, 2)
                v5_info['regime_confirmed'] = confirmed_regime
                if not _regime_blocked:
                    v5_info['regime_gate'] = f'PASS_{confirmed_regime}'
                    log.info(
                        f"[REGIME_GATE] {symbol}: PASS — regime={confirmed_regime} "
                        f"side={side} H4_SMA20=${h4_sma20:.0f} price=${current_price:.0f}")
            else:
                log.info(
                    f"[REGIME_GATE] {symbol}: SKIP "
                    f"(only {len(h4_closes)} H4 bars available, need {H4_SMA_BARS})")
        except Exception as _rge:
            log.warning(f"[REGIME_GATE] {symbol}: error computing regime gate — {_rge}")

        if _regime_blocked:
            log.info(f"[REGIME_GATE] {symbol}: BLOCKED — {_regime_reason}")
            decision = "HOLD"
            reasons.clear()
            reasons.append(_regime_reason)
            v5_info['hold_reason'] = _regime_reason
            v5_info['regime_gate'] = _regime_reason
            try:
                self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=decision, reasons=reasons,
                    lane_info=v5_info,
                    e_net_pred=e_net_pred, enter_logit=enter_logit,
                    temperature_used=temperature_used)
            except Exception as e:
                log.warning(f"Failed to push cycle log for {symbol}: {e}")
            return None
        # ── END REGIME-AWARE GATE ─────────────────────────────────────────────────

        if atr and atr > 0 and not (atr != atr):
            base_sl_dist = self.sl_mult * atr
            base_tp_dist = self.tp_mult * atr
            if self.predictive_sltp and v5_mfe > 0 and v5_mae > 0:
                # MFE/MAE heads are in R-units (1R = sl_mult × ATR).
                # Widen SL only (never tighten) when model expects larger adverse move.
                mae_dist = v5_mae * self.sl_mult * atr
                sl_dist = max(base_sl_dist, mae_dist)
                sl_dist = min(sl_dist, base_sl_dist * 2.5)
                # Use MFE prediction for TP, floored at base TP so RR never degrades.
                mfe_dist = v5_mfe * self.sl_mult * atr
                tp_dist = max(base_tp_dist, mfe_dist)
                tp_dist = min(tp_dist, base_tp_dist * 3.0)
                log.info(f"  [PRED_SLTP] {symbol}: mae={v5_mae:.3f}R→sl={sl_dist/atr:.2f}×ATR "
                         f"mfe={v5_mfe:.3f}R→tp={tp_dist/atr:.2f}×ATR "
                         f"(base sl={base_sl_dist/atr:.1f} tp={base_tp_dist/atr:.1f}×ATR)")
            else:
                sl_dist = base_sl_dist
                tp_dist = base_tp_dist
            if side == "LONG":
                v5_info['sl_price'] = round(current_price - sl_dist, 6)
                v5_info['tp_price'] = round(current_price + tp_dist, 6)
            else:
                v5_info['sl_price'] = round(current_price + sl_dist, 6)
                v5_info['tp_price'] = round(current_price - tp_dist, 6)
            v5_info['sl_dist_atr'] = round(sl_dist / atr, 3)
            v5_info['tp_dist_atr'] = round(tp_dist / atr, 3)
            log.info(f"[V5_DECISION] sym={symbol} score={v5_score:.4f} thr={self.v5_score_threshold} "
                     f"side={side} -> ENTER | SL: ${v5_info['sl_price']:.2f} | TP: ${v5_info['tp_price']:.2f}")
        else:
            log.warning(f"[V5_DECISION] sym={symbol} score={v5_score:.4f} -> ENTER (ATR invalid={atr}, SL/TP omitted)")


        try:
            self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                htf=htf, direction=side, decision=decision, reasons=reasons,
                lane_info=v5_info,
                e_net_pred=e_net_pred, enter_logit=enter_logit,
                temperature_used=temperature_used)
        except Exception as e:
            log.warning(f"Failed to push cycle log for {symbol}: {e}")

        sl_pct = self.sl_mult * atr / current_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0)

        return {
            'symbol': symbol,
            'side': side,
            'p_enter': p_enter,
            'current_price': current_price,
            'atr': atr,
            'htf': htf,
            'risk_pct': risk_pct,
            'df_candles': df_candles,
            'expected_net_r': v5_score,
            'v5_info': v5_info,
            'features_df': features_df,
        }

    def _check_halt_conditions(self, symbol: str) -> Optional[str]:
        """Return a halt reason string if any pre-trade halt condition is active, else None.

        Halt conditions:
          1. Data staleness — last candle timestamp is older than max_data_staleness_seconds.
          2. Consecutive API errors — too many consecutive failures fetching data.
          3. Daily loss R limit — cumulative closed R for today exceeded max_daily_loss_r.

        When halted, new entries are blocked but existing position management continues.
        """
        now_ts = time.time()
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if today_str != self._daily_r_date:
            self._daily_closed_r = 0.0
            self._daily_r_date = today_str

        if self.halt_on_data_staleness and self._last_candle_time > 0:
            age = now_ts - self._last_candle_time
            stale_limit = self._effective_staleness_limit_seconds()
            if age > stale_limit:
                return (
                    f"DATA_STALE last_candle_age={age:.0f}s > "
                    f"max={stale_limit:.0f}s (cfg={self.max_data_staleness_seconds:.0f}s interval={self.interval})"
                )

        if self.halt_on_api_errors and self._consecutive_api_errors >= self.max_consecutive_api_errors:
            return (f"API_ERRORS consecutive={self._consecutive_api_errors} >= "
                    f"max={self.max_consecutive_api_errors}")

        if (self.max_daily_loss_r is not None
                and self._daily_closed_r <= -abs(self.max_daily_loss_r)):
            return (f"DAILY_LOSS_LIMIT cumulative_r={self._daily_closed_r:.2f} <= "
                    f"-{abs(self.max_daily_loss_r):.2f}")

        if self._equity_hard_stop_latched:
            return "EQUITY_HARD_STOP_LATCHED"

        if self.equity_floor_usd > 0.0:
            eq_live = self._session_equity_live_usd(force_refresh=False)
            if eq_live is not None and eq_live <= self.equity_floor_usd:
                return (
                    f"EQUITY_FLOOR equity_live_usd={eq_live:.2f} <= floor={self.equity_floor_usd:.2f}"
                )

        return None

    def _execute_candidate(self, candidate: dict):
        """Execute a trade candidate using V5 composite scoring.

        Gated by execution_mode:
        - signal_only: log the signal, push cycle log, do NOT create Position or POST trade
        - paper: create Position, POST trade, simulate exits
        - live: create Position, POST trade, place exchange orders
        """
        from portfolio import Position

        halt_reason = self._check_halt_conditions(candidate.get('symbol', ''))
        if halt_reason:
            log.warning(
                "[HALT] New entry BLOCKED for %s — halt_reason=%s "
                "(existing positions continue to be managed)",
                candidate.get('symbol', '?'), halt_reason,
            )
            try:
                v5_info = dict(candidate.get("v5_info", {}) or {})
                v5_info["hold_reason"] = str(halt_reason)
                self._push_cycle_log(
                    symbol=str(candidate.get("symbol", "")),
                    price=float(candidate.get("current_price", 0.0) or 0.0),
                    p_enter=float(candidate.get("p_enter", 0.0) or 0.0),
                    htf=candidate.get("htf") or {},
                    direction=str(candidate.get("side", "NEUTRAL")),
                    decision="ENTER_BLOCKED",
                    reasons=[str(halt_reason)],
                    lane_info=v5_info,
                    decision_stage="execution",
                    execution_status="rejected",
                )
            except Exception:
                pass
            return

        symbol = candidate['symbol']
        side = candidate['side']
        current_price = candidate['current_price']
        atr = candidate['atr']
        p_enter = candidate['p_enter']
        htf = candidate['htf']
        v5_info = candidate.get('v5_info', {})
        lane = str(v5_info.get("lane", "V5"))
        lane_threshold = float(v5_info.get("threshold_used", self.v5_score_threshold))
        lane_horizon = int(v5_info.get("lane_horizon", 24))
        lane_size_mult = float(v5_info.get("lane_size_mult", 1.0))
        model_name = str(v5_info.get("model_name", "v5_forecaster_live"))
        htf_score = v5_info.get('htf_score', 0)
        lane_sl_mult = float(v5_info.get("sl_mult_used", self.sl_mult))
        lane_tp_mult = float(v5_info.get("tp_mult_used", self.tp_mult))
        micro = self._fetch_symbol_microstructure(symbol)
        spread_bps = float(micro.get("spread_bps", np.nan))
        spread_limit_bps = self._adaptive_spread_limit_bps()
        v5_info["micro_spread_limit_bps"] = round(float(spread_limit_bps), 4)
        if np.isfinite(spread_bps):
            v5_info["micro_spread_bps"] = round(spread_bps, 4)
            v5_info["micro_bid"] = round(float(micro.get("bid", 0.0)), 6)
            v5_info["micro_ask"] = round(float(micro.get("ask", 0.0)), 6)
        if np.isfinite(spread_bps) and spread_bps > spread_limit_bps:
            reason = (
                f"SPREAD_GATE spread={spread_bps:.2f}bps>"
                f"limit={spread_limit_bps:.2f}bps"
            )
            log.info("  %s: %s", symbol, reason)
            try:
                v5_info["hold_reason"] = reason
                self._push_cycle_log(
                    symbol=symbol,
                    price=current_price,
                    p_enter=p_enter,
                    htf=htf,
                    direction=side,
                    decision="EXECUTION_SKIPPED",
                    reasons=[reason],
                    lane_info=v5_info,
                    decision_stage="execution",
                    execution_status="rejected",
                )
            except Exception:
                pass
            return

        if self.execution_mode == "signal_only" or not self.record_trades:
            sig_sl_dist = lane_sl_mult * atr
            sig_tp_dist = lane_tp_mult * atr
            if side == "LONG":
                sig_sl = current_price - sig_sl_dist
                sig_tp = current_price + sig_tp_dist
            else:
                sig_sl = current_price + sig_sl_dist
                sig_tp = current_price - sig_tp_dist
            no_exec_reason = "SIGNAL_ONLY" if self.execution_mode == "signal_only" else "RECORD_TRADES_OFF"
            log.info(f"  [NO_EXEC] {no_exec_reason} would_open_trade symbol={symbol} side={side} "
                     f"v5_score={v5_info.get('v5_score','?')} p={p_enter:.4f} "
                     f"entry={current_price:.2f} sl={sig_sl:.2f} tp={sig_tp:.2f}")
            try:
                self._push_cycle_log(
                    symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=no_exec_reason,
                    reasons=[
                        f"would_enter=true",
                        f"side={side} entry={current_price:.2f} sl={sig_sl:.2f} tp={sig_tp:.2f}",
                        f"v5_score={v5_info.get('v5_score','?')} p={p_enter:.4f}",
                    ],
                    lane_info=v5_info,
                    decision_stage="execution",
                    execution_status="not_executed",
                )
            except Exception as e:
                log.warning(f"Failed to push {no_exec_reason} cycle log for {symbol}: {e}")
            return

        entry_price = current_price
        exec_result = None

        if self.execution:
            exec_candles = None
            if self.dry_run:
                exec_candles = candidate['df_candles'].tail(15).reset_index(drop=True)

            exec_result = self.execution.attempt_entry(
                symbol=symbol, side=side,
                signal_price=current_price, atr=atr, p_enter=p_enter,
                dry_run_candles=exec_candles,
            )

            if not exec_result.executed:
                skip_reason = str(exec_result.reason or "execution_module_skipped")
                if self.execution_mode == "paper":
                    # In paper mode we preserve signal-level behavior even when
                    # pullback execution misses, so diagnostics still produce trades.
                    log.info(
                        "  [PAPER_EXEC_FALLBACK] %s execution skip (%s) -> opening at signal price",
                        symbol,
                        skip_reason,
                    )
                    v5_info["execution_skip_reason"] = skip_reason
                    v5_info["execution_fallback"] = "signal_price"
                    exec_result = None
                else:
                    log.info(f"  {symbol}: execution module skipped trade — {skip_reason}")
                    v5_info["hold_reason"] = skip_reason
                    try:
                        self._push_cycle_log(
                            symbol=symbol,
                            price=current_price,
                            p_enter=p_enter,
                            htf=htf,
                            direction=side,
                            decision="EXECUTION_SKIPPED",
                            reasons=[skip_reason],
                            lane_info=v5_info,
                            decision_stage="execution",
                            execution_status="rejected",
                        )
                    except Exception:
                        pass
                    return

            if exec_result is not None:
                entry_price = exec_result.entry_price
        entry_slippage_bps = abs(float(entry_price) - float(current_price)) / max(float(current_price), 1e-9) * 10000.0
        self._record_execution_quality(
            spread_bps=spread_bps if np.isfinite(spread_bps) else None,
            slippage_bps=entry_slippage_bps,
        )
        v5_info["entry_slippage_bps"] = round(float(entry_slippage_bps), 4)

        v5_mae = v5_info.get('v5_mae', 0.0)
        v5_mfe = v5_info.get('v5_mfe', 0.0)
        base_sl_dist = lane_sl_mult * atr
        base_tp_dist = lane_tp_mult * atr
        if self.predictive_sltp and v5_mfe > 0 and v5_mae > 0:
            mae_dist = v5_mae * lane_sl_mult * atr
            sl_dist = max(base_sl_dist, mae_dist)
            sl_dist = min(sl_dist, base_sl_dist * 2.5)
            mfe_dist = v5_mfe * lane_sl_mult * atr
            tp_dist = max(base_tp_dist, mfe_dist)
            tp_dist = min(tp_dist, base_tp_dist * 3.0)
        else:
            sl_dist = base_sl_dist
            tp_dist = base_tp_dist

        if side == "LONG":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        prediction = _build_prediction_payload(
            symbol=symbol, side=side, p_enter=p_enter,
            current_price=current_price, atr=atr,
            entry_price=entry_price, tp_mult=self.tp_mult, sl_mult=self.sl_mult,
            tp_mult_used=float(tp_dist / max(atr, 1e-9)),
            sl_mult_used=float(sl_dist / max(atr, 1e-9)),
            htf=htf, exec_result=exec_result,
        )
        prediction["model_name"] = model_name

        sl_pct = abs(entry_price - sl_price) / entry_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0)
        size_pct = risk_pct * max(lane_size_mult, 0.0)

        pos = Position(
            symbol=symbol, side=side,
            entry_price=entry_price, entry_time=time.time(),
            atr=atr, tp_price=tp_price, sl_price=sl_price,
            p_enter=p_enter, size_mult=lane_size_mult, risk_pct=risk_pct,
            bar_index=self.cycle_count,
            lane=lane, horizon=lane_horizon, htf_score=htf_score,
            threshold_used=lane_threshold,
        )
        self.portfolio.open_position(pos)

        self.cooldown_tracker[symbol] = self.cooldown_bars

        try:
            trade_id = self._push_trade_record(
                symbol=symbol, side=side, entry_price=entry_price,
                sl_price=sl_price, tp_price=tp_price,
                p_enter=p_enter, size_pct=size_pct, lane_info={
                    'lane': lane,
                    'htf_score': htf_score,
                    'v5_score': v5_info.get('v5_score'),
                    'threshold_used': lane_threshold,
                    'lane_size_mult': lane_size_mult,
                    'lane_horizon': lane_horizon,
                    'model_name': model_name,
                },
            )
            if trade_id:
                pos.dashboard_trade_id = trade_id
        except Exception as e:
            log.warning(f"Failed to push trade record for {symbol}: {e}")

        if self.execution_mode == "paper":
            log.info(f"  [PAPER_OPEN] {lane} {symbol} {side} @ {entry_price:.2f} "
                     f"| v5_score={v5_info.get('v5_score','?')}")
        elif self.execution_mode == "live":
            if self.execution is not None:
                log.info(f"  [LIVE_OPEN] real_order_sent {lane} {symbol} {side} @ {entry_price:.2f} "
                         f"| v5_score={v5_info.get('v5_score','?')}")
            else:
                log.info(f"  [LIVE_SIGNAL_ONLY] no_adapter_wired {lane} {symbol} {side} @ {entry_price:.2f} "
                         f"| v5_score={v5_info.get('v5_score','?')} "
                         f"[WARNING: execution_mode=live but no real exchange adapter — no order placed]")
        self._push_prediction(prediction)
        try:
            self._push_cycle_log(
                symbol=symbol,
                price=entry_price,
                p_enter=p_enter,
                htf=htf,
                direction=side,
                decision="ENTER_EXECUTED",
                reasons=[f"trade_opened side={side} entry={entry_price:.2f}"],
                lane_info=v5_info,
                decision_stage="execution",
                execution_status="executed",
            )
        except Exception as e:
            log.warning(f"Failed to push execution cycle log for {symbol}: {e}")

    def _print_summary(self):
        summary = self.portfolio.summary()

        def _fmt_pf(pf: float) -> str:
            if np.isinf(pf):
                return "inf"
            return f"{pf:.2f}"

        closed_rows = list(self._closed_trade_stats)
        # Fallback for resumed sessions where in-memory callback history is empty.
        if not closed_rows and self.portfolio.trade_history:
            closed_rows = [
                {
                    "side": t.side,
                    "gross_r": float(t.gross_r),
                    "cost_r": 0.0,
                    "net_r": float(t.gross_r),
                    "outcome": t.outcome,
                }
                for t in self.portfolio.trade_history
            ]

        net_values = [float(r.get("net_r", 0.0)) for r in closed_rows]
        total_closed = len(net_values)
        net_wins = sum(1 for r in net_values if r > 0)
        net_losses = total_closed - net_wins
        net_win_rate = (net_wins / total_closed) if total_closed > 0 else 0.0
        net_expectancy = (sum(net_values) / total_closed) if total_closed > 0 else 0.0
        net_profit = sum(r for r in net_values if r > 0)
        net_loss_abs = abs(sum(r for r in net_values if r < 0))
        net_pf = (net_profit / net_loss_abs) if net_loss_abs > 1e-12 else (float("inf") if net_profit > 0 else 0.0)

        side_stats: Dict[str, Dict[str, object]] = {}
        for side in ("LONG", "SHORT"):
            side_rows = [r for r in closed_rows if str(r.get("side", "")).upper() == side]
            side_vals = [float(r.get("net_r", 0.0)) for r in side_rows]
            side_taken = len(side_vals)
            side_success = sum(1 for r in side_vals if r > 0)
            side_success_rate = (side_success / side_taken) if side_taken > 0 else 0.0
            side_profit = sum(r for r in side_vals if r > 0)
            side_loss_abs = abs(sum(r for r in side_vals if r < 0))
            side_pf = (side_profit / side_loss_abs) if side_loss_abs > 1e-12 else (float("inf") if side_profit > 0 else 0.0)
            side_stats[side] = {
                "taken": side_taken,
                "success": side_success,
                "success_rate": side_success_rate,
                "net_r_sum": sum(side_vals),
                "net_expectancy": (sum(side_vals) / side_taken) if side_taken > 0 else 0.0,
                "net_pf": side_pf,
            }

        log.info("")
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info(f"  Cycles: {self.cycle_count}")
        log.info(f"  Trades: {total_closed} (net W:{net_wins} L:{net_losses})")
        log.info(
            "  Net: win_rate=%s expectancy=%+0.3fR pf=%s",
            f"{net_win_rate:.1%}", net_expectancy, _fmt_pf(net_pf)
        )
        long_stats = side_stats.get("LONG", {})
        short_stats = side_stats.get("SHORT", {})
        log.info(
            "  LONG: taken=%s successful=%s (%s) netR=%+0.2f exp=%+0.3fR pf=%s",
            int(long_stats.get("taken", 0)),
            int(long_stats.get("success", 0)),
            f"{float(long_stats.get('success_rate', 0.0)):.1%}",
            float(long_stats.get("net_r_sum", 0.0)),
            float(long_stats.get("net_expectancy", 0.0)),
            _fmt_pf(float(long_stats.get("net_pf", 0.0))),
        )
        log.info(
            "  SHORT: taken=%s successful=%s (%s) netR=%+0.2f exp=%+0.3fR pf=%s",
            int(short_stats.get("taken", 0)),
            int(short_stats.get("success", 0)),
            f"{float(short_stats.get('success_rate', 0.0)):.1%}",
            float(short_stats.get("net_r_sum", 0.0)),
            float(short_stats.get("net_expectancy", 0.0)),
            _fmt_pf(float(short_stats.get("net_pf", 0.0))),
        )
        log.info(f"  Gross (legacy): win_rate={summary['win_rate']:.1%} avg_r={summary['avg_r']:+.2f}")
        log.info(f"  Open: {summary['open_positions']} | Risk: {summary['total_risk_pct']:.1f}%")
        log.info("=" * 60)
