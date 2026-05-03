#!/usr/bin/env python3
"""
BTC Futures GPU Trainer - Quick Start (v3.3.0 ENTER QUALITY + Funding + OI)
=============================================================
One-script setup: Downloads data from your Replit dashboard,
trains the ENTER QUALITY model on your GPU, and pushes predictions back.

The model predicts WHETHER to enter a trend-following trade (binary ENTER=0/1),
not WHICH direction. Direction comes from HTF (1H/4H) trend alignment.

Usage:
    python quick_start.py --url https://YOUR-APP.replit.app

That's it. Everything else is automatic.

===========================================================================
RECOMMENDED V5 TRAINING COMMAND (as of v5.3.0)
===========================================================================
Use this command for walk-forward training. Key rules:
  - DO NOT add --v5-multi-regime (it disables the EMA200 hard gate)
  - DO NOT add --v5-sigma-discount (reduces trade frequency without benefit)
  - Use --v5-ema200-soft-mult 0.50 instead of --v5-ema200-regime-gate (soft gate
    reduces size by 50% against-trend instead of hard blocking — recovers +21R/fold)
  - Use --v5-regime-side-map WITHOUT --v5-multi-regime: the side map routes
    short signals in downtrend regimes
  - Use --v5-min-threshold 0.04 (prevents threshold collapsing to 0.015 floor)
  - Use --v5-trail-activation 1.5 --v5-trail-distance 1.0 (gives trades room to run)
  - Use --v5-short-oversample --v5-short-min-fraction 0.35 to fix LONG bias in labels
  - Use --v5-per-side-threshold with --v5-per-symbol-threshold for separate
    LONG/SHORT thresholds per symbol

python quick_start.py --train-v5 --v5-walk-forward --v5-ema200-soft-mult 0.50 \\
    --v5-adx-gate --v5-adx-min 18 --v5-min-threshold 0.04 \\
    --v5-trailing-sl --v5-trail-activation 1.5 --v5-trail-distance 1.0 \\
    --v5-corr-thresh 0.90 --v5-side-aware-scoring --v5-recency-weight \\
    --v5-short-oversample --v5-short-min-fraction 0.35 \\
    --v5-per-symbol-threshold --v5-per-side-threshold \\
    --v5-regime-side-map "trending_up=LONG,trending_down=SHORT,choppy=BOTH"

Model size: [512, 256, 128, 64] hidden dims (~250K params for ~32K samples)
Target label mix: HOLD ~25-30%, LONG ~30-35%, SHORT ~30-35% (with oversample)
Target trades/day in forward test: 3-6
Bear-market folds: expect balanced LONG/SHORT split (not 100% LONG)
===========================================================================
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("QuickStart")

FEATURE_VERSION = "v4.6.0_directional_separation"
DIST_FEATURE_VERSION = "v4.9.1_enhanced_distributional"
SYSTEM_VERSION = "v5.7.0_brilliant_v5"

FUNDING_FEATURE_NAMES = ["funding_rate", "funding_rate_delta_8h", "funding_rate_zscore_30d"]
FUNDING_FEATURE_COUNT = len(FUNDING_FEATURE_NAMES)

OI_FEATURE_NAMES = ["open_interest", "oi_delta_1h", "oi_zscore_30d"]
OI_FEATURE_COUNT = len(OI_FEATURE_NAMES)

LS_RATIO_FEATURE_NAMES = ["ls_ratio", "ls_deviation", "ls_extreme", "crowd_sentiment"]
LS_RATIO_FEATURE_COUNT = len(LS_RATIO_FEATURE_NAMES)


def _apply_mythos_profile_overrides(args) -> list:
    """
    Apply deterministic MYTHOS profile overrides before constructing MythosConfig.
    Returns a list of (name, old, new) changes.
    """
    profile = str(getattr(args, "mythos_profile", "modern")).strip().lower()
    if profile in {"modern", "default"}:
        return []
    if profile != "legacy-stable":
        return []

    # Legacy-stable profile approximates pre-collapse behavior by disabling
    # the post-out-of-box stacked gates and drawdown throttles.
    overrides = {
        "mythos_bayes_quality_enable": False,
        "mythos_nonconformity_enable": False,
        "mythos_side_rebalance_enable": False,
        "mythos_intelligence_enable": False,
        "mythos_intelligence_side_switch_enable": False,
        "mythos_counterfactual_target_reject_rate": 0.70,
        "mythos_counterfactual_reject_tolerance": 0.10,
        "mythos_counterfactual_adaptive_relax": 0.0,
        "mythos_counterfactual_adaptive_min_adv_floor": 1.0,
        "mythos_nonconformity_target_reject_rate": 0.48,
        "mythos_nonconformity_reject_tolerance": 0.12,
        "mythos_nonconformity_adaptive_relax": 0.0,
        "mythos_nonconformity_adaptive_max_relax": 0.0,
        "mythos_nonconformity_soft_override_margin": 0.0,
        "mythos_leverage_side_policy_enable": False,
        "mythos_execution_fee_bps": 0.0,
        "mythos_execution_slippage_bps": 0.0,
        "mythos_execution_cost_cap_r": 0.0,
        "mythos_emergency_stop_r": -200.0,
        "mythos_emergency_max_drawdown_r": 200.0,
        "mythos_drawdown_size_start_r": 200.0,
        "mythos_drawdown_size_full_r": 400.0,
        "mythos_drawdown_size_min_scale": 1.0,
        "mythos_disable_conviction_boost_dd_r": 200.0,
        "mythos_disable_leverage_dd_r": 200.0,
        "mythos_dd_risk_recovery_r": 190.0,
        "mythos_meta_warmup_samples": 192,
        "mythos_meta_ready_prob_floor": 0.47,
        "mythos_meta_ready_prob_ceiling": 0.53,
    }

    changes = []
    for name, value in overrides.items():
        if not hasattr(args, name):
            continue
        old = getattr(args, name)
        if old != value:
            setattr(args, name, value)
            changes.append((name, old, value))
    return changes


def check_gpu():
    try:
        import torch
        try:
            if torch.cuda.device_count() > 0:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                log.info(f"GPU: {name} ({mem:.1f} GB)")
                return "cuda"
            else:
                log.warning("No GPU found - training will be slow on CPU")
                return "cpu"
        except Exception as e:
            log.warning(f"CUDA device check failed ({e}) - falling back to CPU")
            return "cpu"
    except ImportError:
        log.error("PyTorch not installed! Run: pip install -r requirements.txt")
        sys.exit(1)


def download_data(replit_url: str, data_dir: Path, force_fresh: bool = False, symbol: str = "BTCUSDT"):
    """Download 15m candle data for a single symbol from the dashboard API."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol}_15m.csv"
    parquet_path = data_dir / f"{symbol}_15m.parquet"

    if parquet_path.exists() and not force_fresh:
        import pandas as pd
        existing = pd.read_parquet(parquet_path)
        log.info(f"[DOWNLOAD] {symbol}: found existing data ({len(existing)} candles), skipping download")
        return parquet_path

    url = f"{replit_url.rstrip('/')}/api/data/export-csv?symbol={symbol}&timeframe=15m"
    log.info(f"[DOWNLOAD] {symbol}: fetching 15m data from dashboard...")
    log.info(f"  URL: {url}")

    try:
        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot connect to {replit_url}")
        log.error("Make sure your Replit dashboard is running!")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"[DOWNLOAD] {symbol}: server returned error: {e}")
        return None

    with open(csv_path, 'wb') as f:
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    log.info(f"  Downloaded {total / 1024:.0f} KB")

    import pandas as pd
    df = pd.read_csv(csv_path)
    log.info(f"  {symbol}: loaded {len(df)} candles")

    if len(df) < 100:
        log.error(f"[DOWNLOAD] {symbol}: only {len(df)} candles - insufficient data")
        csv_path.unlink(missing_ok=True)
        return None

    date_min = datetime.fromtimestamp(df['timestamp'].min() / 1000).strftime('%Y-%m-%d')
    date_max = datetime.fromtimestamp(df['timestamp'].max() / 1000).strftime('%Y-%m-%d')
    log.info(f"  {symbol}: date range {date_min} to {date_max}")

    df.to_parquet(parquet_path, index=False)
    log.info(f"  Saved to {parquet_path}")

    csv_path.unlink(missing_ok=True)
    return parquet_path


MIN_BARS_FOR_TRAINING = 20000


def download_missing_data(replit_url: str, data_dir: Path, symbols: list, force_fresh: bool = False):
    """Download 15m candle data for all requested symbols."""
    data_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for sym in symbols:
        parquet_path = data_dir / f"{sym}_15m.parquet"
        if parquet_path.exists() and not force_fresh:
            import pandas as pd
            existing = pd.read_parquet(parquet_path)
            results[sym] = len(existing)
            log.info(f"[DOWNLOAD] {sym}: already cached ({len(existing)} candles)")
        else:
            path = download_data(replit_url, data_dir, force_fresh=force_fresh, symbol=sym)
            if path and path.exists():
                import pandas as pd
                df = pd.read_parquet(path)
                results[sym] = len(df)
            else:
                results[sym] = 0
                log.error(f"[DOWNLOAD] {sym}: FAILED to download data")
    return results


def preflight_data_check(data_dir: Path, symbols: list, allow_partial: bool = False, min_bars: int = MIN_BARS_FOR_TRAINING):
    """Preflight check: verify all requested symbol parquets exist and have sufficient rows.
    
    Returns list of valid symbols. Aborts if any missing and allow_partial is False.
    """
    import pandas as pd
    valid_symbols = []
    any_missing = False

    log.info("=" * 60)
    log.info("  PREFLIGHT DATA CHECK")
    log.info("=" * 60)

    for sym in symbols:
        sym_path = data_dir / f"{sym}_15m.parquet"
        exists = sym_path.exists()
        rows = 0
        if exists:
            try:
                df = pd.read_parquet(sym_path)
                rows = len(df)
            except Exception as e:
                log.error(f"[DATA_CHECK] sym={sym} file={sym_path} exists=True rows=ERROR ({e})")
                any_missing = True
                continue

        status = "OK" if (exists and rows >= min_bars) else "MISSING" if not exists else f"LOW ({rows} < {min_bars})"
        log.info(f"[DATA_CHECK] sym={sym} file={sym_path} exists={exists} rows={rows} status={status}")

        if exists and rows >= min_bars:
            valid_symbols.append(sym)
        elif exists and rows > 0:
            log.warning(f"[DATA_CHECK] {sym}: only {rows} bars (min={min_bars}), including with warning")
            valid_symbols.append(sym)
        else:
            any_missing = True
            log.error(f"[DATA_CHECK] {sym}: NO USABLE DATA at {sym_path}")

    if any_missing and not allow_partial:
        log.error("[DATA_CHECK] ABORTING: missing data for one or more symbols.")
        log.error("[DATA_CHECK] Use --download-missing-data to auto-fetch, or --allow-partial-data to train on available symbols only.")
        sys.exit(1)

    if not valid_symbols:
        log.error("[DATA_CHECK] ABORTING: no valid symbol data found at all.")
        sys.exit(1)

    log.info(f"[DATA_CHECK] Proceeding with {len(valid_symbols)} symbols: {valid_symbols}")
    log.info("-" * 60)
    return valid_symbols


def fetch_funding_rates(candle_df, data_dir: Path):
    """Fetch historical funding rates from Binance Futures API, paginating to cover full candle range."""
    import requests
    import pandas as pd

    cache_path = data_dir / "funding_rates.parquet"

    candle_start_ms = int(candle_df['timestamp'].min())
    candle_end_ms = int(candle_df['timestamp'].max())

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if len(existing) > 0:
            cached_start = existing['timestamp'].min()
            cached_end = existing['timestamp'].max()
            if cached_start <= candle_start_ms and cached_end >= candle_end_ms - 8 * 3600 * 1000:
                log.info(f"Using cached funding rates: {len(existing)} records")
                return existing

    log.info("Fetching historical funding rates from Binance Futures...")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_records = []
    current_start = candle_start_ms
    page = 0

    while current_start < candle_end_ms:
        params = {
            "symbol": "BTCUSDT",
            "startTime": current_start,
            "endTime": candle_end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                import time as _time
                _time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Funding rate fetch error (page {page}): {e}")
            break

        if not data:
            break

        for item in data:
            all_records.append({
                "timestamp": int(item["fundingTime"]),
                "funding_rate": float(item["fundingRate"]),
            })

        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        page += 1

        if page % 5 == 0:
            log.info(f"  Fetched {len(all_records)} funding records so far...")
        import time as _time
        _time.sleep(0.1)

    if not all_records:
        log.warning("No funding data fetched - funding features will be zero")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    funding_df = pd.DataFrame(all_records)
    funding_df = funding_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    funding_start = funding_df['timestamp'].min()
    funding_end = funding_df['timestamp'].max()
    expected_records = (candle_end_ms - candle_start_ms) / (8 * 3600 * 1000)
    coverage_pct = len(funding_df) / max(expected_records, 1) * 100
    log.info(f"Funding coverage: {coverage_pct:.0f}% ({len(funding_df)} records for ~{expected_records:.0f} expected 8h intervals)")
    if coverage_pct < 50:
        log.warning(f"Low funding coverage ({coverage_pct:.0f}%) - some candles will have zero funding features")

    funding_df.to_parquet(cache_path, index=False)
    log.info(f"Fetched {len(funding_df)} funding rate records (cached to {cache_path})")

    return funding_df


def compute_funding_features(candle_df, funding_df):
    """Compute funding features aligned to 15m candle timestamps via merge_asof backward.

    Returns DataFrame with 3 columns: funding_rate, funding_rate_delta_8h, funding_rate_zscore_30d
    All values are z-scored/normalized and clipped to ±5.
    """
    import pandas as pd
    import numpy as np

    n = len(candle_df)

    if funding_df.empty:
        log.warning("Empty funding data - returning zero features")
        return pd.DataFrame(
            np.zeros((n, FUNDING_FEATURE_COUNT)),
            columns=FUNDING_FEATURE_NAMES,
            index=candle_df.index,
        )

    candle_ts = candle_df[['timestamp']].copy()
    candle_ts = candle_ts.reset_index(drop=True)
    candle_ts['_candle_idx'] = candle_ts.index

    funding_sorted = funding_df[['timestamp', 'funding_rate']].copy()
    funding_sorted = funding_sorted.sort_values('timestamp').reset_index(drop=True)

    funding_sorted['funding_rate_prev'] = funding_sorted['funding_rate'].shift(1)
    funding_sorted['funding_rate_delta_8h'] = funding_sorted['funding_rate'] - funding_sorted['funding_rate_prev']

    rolling_window = 90
    rolling_mean = funding_sorted['funding_rate'].rolling(rolling_window, min_periods=1).mean()
    rolling_std = funding_sorted['funding_rate'].rolling(rolling_window, min_periods=1).std().clip(lower=1e-8)
    funding_sorted['funding_rate_zscore_30d'] = (funding_sorted['funding_rate'] - rolling_mean) / rolling_std

    funding_sorted = funding_sorted.fillna(0)

    merged = pd.merge_asof(
        candle_ts.sort_values('timestamp'),
        funding_sorted[['timestamp', 'funding_rate', 'funding_rate_delta_8h', 'funding_rate_zscore_30d']],
        on='timestamp',
        direction='backward',
    )

    merged = merged.sort_values('_candle_idx').reset_index(drop=True)

    result = pd.DataFrame(index=candle_df.index)
    result['funding_rate'] = merged['funding_rate'].values * 100
    result['funding_rate_delta_8h'] = merged['funding_rate_delta_8h'].values * 100
    result['funding_rate_zscore_30d'] = merged['funding_rate_zscore_30d'].values

    result = result.fillna(0)
    result = result.clip(lower=-5, upper=5)

    n_nonzero = (result.abs() > 1e-8).any(axis=1).sum()
    log.info(f"Funding features: {n_nonzero}/{n} rows with non-zero funding data")

    import random
    sample_indices = sorted(random.sample(range(min(100, n), n), min(10, max(1, n - 100))))
    log.info("FUNDING ALIGNMENT CHECK (10 random rows):")
    log.info(f"{'Row':>8} | {'Candle TS':>15} | {'FR':>10} | {'Delta8h':>10} | {'Z30d':>10}")
    log.info("-" * 65)
    for idx in sample_indices:
        ts = candle_df.iloc[idx].get('timestamp', 0)
        fr = result.iloc[idx]['funding_rate']
        delta = result.iloc[idx]['funding_rate_delta_8h']
        zscore = result.iloc[idx]['funding_rate_zscore_30d']
        log.info(f"{idx:>8} | {int(ts):>15} | {fr:>+10.4f} | {delta:>+10.4f} | {zscore:>+10.4f}")
    log.info("-" * 65)

    return result


def fetch_open_interest_hist(candle_df, data_dir: Path, period: str = "15m", symbol: str = "BTCUSDT"):
    """Fetch historical Open Interest from Binance Futures API.

    Binance only provides ~30 days of OI history via /futures/data/openInterestHist.
    We use 15m period (matching candle TF) for clean 1:1 alignment, and paginate
    through the full 30-day window to get ~2880 records instead of the 500 limit per call.

    For candle data older than 30 days, OI features will be zero (Binance limitation).
    Coverage is measured against the 30-day OI window, not the full candle range.
    """
    import requests
    import pandas as pd
    import time as _time

    cache_path = data_dir / f"open_interest_hist_{symbol}.parquet"
    legacy_cache = data_dir / "open_interest_hist.parquet"

    now_ms = int(datetime.now().timestamp() * 1000)
    max_oi_lookback_ms = 30 * 24 * 60 * 60 * 1000
    oi_window_start_ms = now_ms - max_oi_lookback_ms
    oi_window_end_ms = now_ms

    candle_end_ms = int(candle_df['timestamp'].max())
    fetch_start_ms = max(oi_window_start_ms, int(candle_df['timestamp'].min()))
    fetch_end_ms = min(oi_window_end_ms, candle_end_ms)

    if fetch_start_ms >= fetch_end_ms:
        log.warning(f"OI: No overlap between candle range and 30-day OI window for {symbol}")
        return pd.DataFrame(columns=["oi_time_ms", "sumOpenInterest", "symbol", "period"])

    oi_window_days = (fetch_end_ms - fetch_start_ms) / (24 * 60 * 60 * 1000)
    log.info(f"[OI_FETCH] symbol={symbol} period={period} window=30d target_records=~2880")

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if len(existing) > 0:
            cached_start = existing['oi_time_ms'].min()
            cached_end = existing['oi_time_ms'].max()
            cached_n_nonzero = int((existing['sumOpenInterest'] > 0).sum())
            cached_period = existing.iloc[0].get('period', 'unknown') if 'period' in existing.columns else 'unknown'
            period_ms = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 3600*1000}.get(cached_period, 15*60*1000)
            cache_age_hours = (now_ms - cached_end) / (3600 * 1000)
            stale = cache_age_hours >= 24
            log.info(f"[OI_CACHE] symbol={symbol} path={cache_path} stale={stale} "
                     f"nonzero={cached_n_nonzero} age={cache_age_hours:.1f}h")
            if (cached_n_nonzero > 100
                    and cached_start <= fetch_start_ms + period_ms
                    and cached_end >= fetch_end_ms - 2 * period_ms
                    and not stale):
                log.info(f"[OI_CACHE] symbol={symbol} REUSING cached OI data: {len(existing)} records")
                return existing
            else:
                reasons = []
                if cached_n_nonzero <= 100:
                    reasons.append(f"low_nonzero={cached_n_nonzero}")
                if stale:
                    reasons.append(f"stale={cache_age_hours:.1f}h")
                log.info(f"[OI_CACHE] symbol={symbol} REFETCHING — {', '.join(reasons)}")

    if legacy_cache.exists():
        try:
            legacy = pd.read_parquet(legacy_cache)
            legacy_nonzero = (legacy['sumOpenInterest'] > 0).sum() if 'sumOpenInterest' in legacy.columns else 0
            if legacy_nonzero < 100:
                log.info(f"Removing stale legacy OI cache ({legacy_nonzero} non-zero records)")
                legacy_cache.unlink()
        except Exception:
            legacy_cache.unlink(missing_ok=True)

    periods_to_try = ["15m", "5m", "1h"]
    if period not in periods_to_try:
        periods_to_try = [period] + periods_to_try

    url = "https://fapi.binance.com/futures/data/openInterestHist"

    # Track whether startTime was rejected so we don't repeat the same error
    # across all three period fallbacks.
    starttime_rejected = False

    for try_period in periods_to_try:
        log.info(f"Fetching OI from Binance Futures for {symbol} (period={try_period})...")
        all_records = []
        current_start = fetch_start_ms
        page = 0
        period_failed = False
        consecutive_errors = 0

        while current_start < fetch_end_ms:
            params = {
                "symbol": symbol,
                "period": try_period,
                "limit": 500,
                "endTime": int(fetch_end_ms),
            }
            # Only include startTime when it hasn't been globally rejected and
            # this isn't the first page of a startTime-less fallback session.
            if not starttime_rejected:
                params["startTime"] = int(current_start)

            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = min(2 ** consecutive_errors, 10)
                    log.warning(f"OI rate limited - sleeping {wait}s")
                    _time.sleep(wait)
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        period_failed = True
                        break
                    continue
                if resp.status_code == 400:
                    resp_text = resp.text[:200] if resp.text else "no body"
                    if "startTime" in resp_text and not starttime_rejected:
                        # Binance rejected startTime — retry this period without it.
                        log.warning(
                            f"OI period={try_period} startTime rejected by Binance "
                            f"— retrying without startTime for all subsequent requests"
                        )
                        starttime_rejected = True
                        params.pop("startTime", None)
                        resp = requests.get(url, params=params, timeout=30)
                        if resp.status_code != 200:
                            log.warning(f"OI period={try_period} also failed without startTime — skipping")
                            period_failed = True
                            break
                    else:
                        log.warning(f"OI period={try_period} HTTP 400: {resp_text}")
                        period_failed = True
                        break
                if resp.status_code in (403, 418, 451):
                    resp_text = resp.text[:200] if resp.text else "no body"
                    log.warning(f"OI period={try_period} blocked (HTTP {resp.status_code}): {resp_text}")
                    period_failed = True
                    break
                if resp.status_code != 200:
                    resp.raise_for_status()
                data = resp.json()
                consecutive_errors = 0
            except requests.exceptions.HTTPError as e:
                log.warning(f"OI period={try_period} error: {e}")
                consecutive_errors += 1
                if consecutive_errors > 3:
                    period_failed = True
                    break
                _time.sleep(1)
                continue
            except Exception as e:
                log.warning(f"OI fetch error (page {page}): {e}")
                consecutive_errors += 1
                if consecutive_errors > 3:
                    period_failed = True
                    break
                _time.sleep(1)
                continue

            if not data:
                break

            for item in data:
                all_records.append({
                    "oi_time_ms": int(item["timestamp"]),
                    "sumOpenInterest": float(item["sumOpenInterest"]),
                    "symbol": item.get("symbol", symbol),
                    "period": try_period,
                })

            last_ts = int(data[-1]["timestamp"])
            if last_ts <= current_start:
                break

            period_ms_step = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 3600*1000}.get(try_period, 15*60*1000)
            current_start = last_ts + period_ms_step
            page += 1

            if page % 5 == 0:
                log.info(f"  OI page {page}: {len(all_records)} records so far...")
            _time.sleep(0.3)

        if not period_failed and all_records:
            period = try_period
            break
        if period_failed:
            # If startTime was already rejected globally, remaining period
            # fallbacks will hit the same issue — bail out early.
            if starttime_rejected:
                log.warning(f"OI: startTime rejected by Binance for {symbol} — OI features will be zero")
                break
            log.warning(f"OI period={try_period} unavailable, trying next fallback...")
            continue
        if not all_records:
            log.warning(f"OI period={try_period} returned no data, trying next fallback...")
            continue

    if not all_records:
        log.warning(f"OI: all periods failed for {symbol} — OI features will be zero")
        return pd.DataFrame(columns=["oi_time_ms", "sumOpenInterest", "symbol", "period"])

    oi_df = pd.DataFrame(all_records)
    oi_df = oi_df.drop_duplicates(subset=["oi_time_ms"]).sort_values("oi_time_ms").reset_index(drop=True)

    n_nonzero = (oi_df['sumOpenInterest'] > 0).sum()

    period_minutes = {"5m": 5, "15m": 15, "1h": 60}.get(period, 15)
    oi_span_minutes = (fetch_end_ms - fetch_start_ms) / (60 * 1000)
    expected_records = oi_span_minutes / period_minutes
    coverage_pct = len(oi_df) / max(expected_records, 1) * 100

    log.info(f"[OI_FETCH] symbol={symbol} period={period} window=30d fetched_records={len(oi_df)} nonzero={n_nonzero}")
    log.info(f"[OI_FETCH] coverage={coverage_pct:.0f}% of {oi_window_days:.1f}-day window "
             f"({len(oi_df)}/{expected_records:.0f} expected {period} intervals)")

    if coverage_pct < 50:
        log.warning(f"Low OI coverage ({coverage_pct:.0f}%) — check Binance API availability")
    elif coverage_pct < 90:
        log.info(f"OI coverage {coverage_pct:.0f}% — acceptable, some gaps expected near window edges")

    if n_nonzero < 100:
        log.warning(f"[OI_FETCH] symbol={symbol} WARNING: only {n_nonzero} non-zero records (<100 minimum) — OI features may be unreliable")

    oi_df.to_parquet(cache_path, index=False)
    log.info(f"[OI_CACHE] symbol={symbol} saved {len(oi_df)} records to {cache_path}")

    return oi_df


def compute_oi_features(candle_df, oi_df):
    """Compute OI features aligned to 15m candle timestamps via merge_asof backward.

    Returns DataFrame with 3 columns: open_interest, oi_delta_1h, oi_zscore_30d
    All features computed on OI event series BEFORE alignment (leak-free).

    Because Binance OI history covers only ~30 days, candles outside that window
    will have zero OI features. Coverage is reported for the overlapping window.
    """
    import pandas as pd
    import numpy as np

    n = len(candle_df)
    zero_result = pd.DataFrame(
        np.zeros((n, OI_FEATURE_COUNT)),
        columns=OI_FEATURE_NAMES,
        index=candle_df.index,
    )

    if oi_df.empty or len(oi_df) < 2:
        log.warning("Empty/insufficient OI data - returning zero features")
        return zero_result

    oi_nonzero = oi_df[oi_df['sumOpenInterest'] > 0].copy()
    if len(oi_nonzero) < 2:
        log.warning(f"OI data has {len(oi_nonzero)} non-zero records — returning zero features")
        return zero_result

    oi_sorted = oi_nonzero[['oi_time_ms', 'sumOpenInterest']].copy()
    oi_sorted = oi_sorted.sort_values('oi_time_ms').drop_duplicates('oi_time_ms').reset_index(drop=True)

    period = oi_df['period'].iloc[0] if 'period' in oi_df.columns else '15m'
    if period == '5m':
        delta_lookback = 12
        zscore_window = 8640
    elif period == '15m':
        delta_lookback = 4
        zscore_window = 2880
    else:
        delta_lookback = 1
        zscore_window = 720

    oi_sorted['oi_delta_1h'] = oi_sorted['sumOpenInterest'] - oi_sorted['sumOpenInterest'].shift(delta_lookback)

    min_periods_zscore = max(delta_lookback + 1, 20)
    rolling_mean = oi_sorted['oi_delta_1h'].rolling(zscore_window, min_periods=min_periods_zscore).mean()
    rolling_std = oi_sorted['oi_delta_1h'].rolling(zscore_window, min_periods=min_periods_zscore).std().clip(lower=1e-8)
    oi_sorted['oi_zscore_30d'] = (oi_sorted['oi_delta_1h'] - rolling_mean) / rolling_std

    oi_sorted = oi_sorted.fillna(0)

    oi_median = oi_sorted['sumOpenInterest'].median()
    if oi_median > 0:
        oi_sorted['open_interest_scaled'] = oi_sorted['sumOpenInterest'] / oi_median
    else:
        oi_sorted['open_interest_scaled'] = oi_sorted['sumOpenInterest']

    delta_std = oi_sorted['oi_delta_1h'].std()
    if delta_std > 0:
        oi_sorted['oi_delta_1h_scaled'] = oi_sorted['oi_delta_1h'] / delta_std
    else:
        oi_sorted['oi_delta_1h_scaled'] = oi_sorted['oi_delta_1h']

    candle_ts = candle_df[['timestamp']].copy().reset_index(drop=True)
    candle_ts['_candle_idx'] = candle_ts.index

    oi_for_merge = oi_sorted[['oi_time_ms', 'open_interest_scaled', 'oi_delta_1h_scaled', 'oi_zscore_30d']].copy()
    oi_for_merge = oi_for_merge.rename(columns={'oi_time_ms': 'timestamp'})

    oi_min_ts = oi_for_merge['timestamp'].min()
    oi_max_ts = oi_for_merge['timestamp'].max()
    period_ms = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 3600*1000}.get(period, 15*60*1000)
    tolerance_ms = period_ms * 2

    merged = pd.merge_asof(
        candle_ts.sort_values('timestamp'),
        oi_for_merge.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=tolerance_ms,
    )

    merged = merged.sort_values('_candle_idx').reset_index(drop=True)

    result = pd.DataFrame(index=candle_df.index)
    result['open_interest'] = merged['open_interest_scaled'].values
    result['oi_delta_1h'] = merged['oi_delta_1h_scaled'].values
    result['oi_zscore_30d'] = merged['oi_zscore_30d'].values

    result = result.fillna(0)
    result = result.clip(lower=-5, upper=5)

    symbol_tag = oi_df['symbol'].iloc[0] if 'symbol' in oi_df.columns and len(oi_df) > 0 else "UNKNOWN"
    n_nonzero = (result.abs() > 1e-8).any(axis=1).sum()
    log.info(f"[OI_ALIGN] symbol={symbol_tag} features_nonzero={n_nonzero}/{n} tolerance={tolerance_ms}ms")

    candle_in_oi_window = candle_df[(candle_df['timestamp'] >= oi_min_ts) & (candle_df['timestamp'] <= oi_max_ts)]
    n_in_window = len(candle_in_oi_window)
    if n_in_window > 0:
        window_indices = candle_in_oi_window.index
        n_window_nonzero = (result.loc[window_indices].abs() > 1e-8).any(axis=1).sum()
        window_coverage = n_window_nonzero / n_in_window * 100
        log.info(f"[OI_ALIGN] symbol={symbol_tag} coverage_in_window={window_coverage:.0f}% "
                 f"({n_window_nonzero}/{n_in_window} candles in OI window) tolerance={tolerance_ms}ms")
        if window_coverage < 50:
            log.warning(f"[OI_ALIGN] symbol={symbol_tag} POOR alignment ({window_coverage:.0f}%) — check timestamps")
    else:
        log.warning(f"[OI_ALIGN] symbol={symbol_tag} NO candles in OI data window")

    import random
    oi_window_candle_indices = list(candle_df[candle_df['timestamp'] >= oi_min_ts].index)
    if len(oi_window_candle_indices) > 10:
        sample_indices = sorted(random.sample(oi_window_candle_indices, 10))
    elif len(oi_window_candle_indices) > 0:
        sample_indices = oi_window_candle_indices
    else:
        sample_indices = sorted(random.sample(list(range(n)), min(10, n)))

    log.info(f"[OI_ALIGN] symbol={symbol_tag} ALIGNMENT SAMPLE (from OI window):")
    log.info(f"{'Row':>8} | {'Candle TS':>15} | {'OI':>10} | {'Delta1h':>10} | {'Z30d':>10}")
    log.info("-" * 65)
    for idx in sample_indices:
        ts = candle_df.iloc[idx].get('timestamp', 0)
        oi_val = result.iloc[idx]['open_interest']
        delta = result.iloc[idx]['oi_delta_1h']
        zscore = result.iloc[idx]['oi_zscore_30d']
        log.info(f"{idx:>8} | {int(ts):>15} | {oi_val:>+10.4f} | {delta:>+10.4f} | {zscore:>+10.4f}")
    log.info("-" * 65)

    return result


def _oi_sanity_check(candle_df, oi_df, symbol: str = "BTCUSDT", coverage_threshold: float = 80.0):
    """Pre-training OI coverage check with auto-disable.

    Returns oi_enabled (bool): True only if OI coverage within 30-day window >= threshold.
    Default behavior: OI is disabled unless coverage passes threshold.
    Logs decision once per symbol.
    """
    import numpy as np
    now_ms = int(datetime.now().timestamp() * 1000)
    oi_window_start = now_ms - 30 * 24 * 60 * 60 * 1000

    if oi_df.empty or len(oi_df) < 2:
        log.warning(f"[OI_CHECK] DISABLED for {symbol} – no OI data (coverage=0%, nonzero=0). "
                    f"OI features zeroed out. Funding remains enabled.")
        return False

    n_nonzero = int((oi_df['sumOpenInterest'] > 0).sum())
    candle_in_window = candle_df[candle_df['timestamp'] >= oi_window_start]
    n_candles_in_window = len(candle_in_window)

    if n_candles_in_window == 0:
        log.info(f"[OI_CHECK] DISABLED for {symbol} – no candles within 30-day OI window. "
                 f"OI features zeroed out.")
        return False

    oi_in_window = oi_df[oi_df['oi_time_ms'] >= oi_window_start]
    period = oi_df['period'].iloc[0] if 'period' in oi_df.columns else '15m'
    period_minutes = {"5m": 5, "15m": 15, "1h": 60}.get(period, 15)
    window_minutes = (now_ms - oi_window_start) / (60 * 1000)
    expected = window_minutes / period_minutes
    coverage_pct = len(oi_in_window) / max(expected, 1) * 100

    if coverage_pct < coverage_threshold:
        log.warning(f"[OI_CHECK] DISABLED for {symbol} – coverage {coverage_pct:.0f}% < {coverage_threshold:.0f}% threshold "
                    f"(nonzero={n_nonzero}). OI features zeroed out. Funding remains enabled.")
        return False
    else:
        log.info(f"[OI_CHECK] ENABLED for {symbol} – coverage {coverage_pct:.0f}% >= {coverage_threshold:.0f}% "
                 f"(nonzero={n_nonzero}). OI features active.")
        return True


def fetch_ls_ratio_hist(candle_df, data_dir: Path, symbol: str = "BTCUSDT"):
    """Fetch global long/short account ratio history from Binance Futures.

    Uses a probe-from-end strategy: checks the most recent chunk first to
    quickly determine if data is available at all, then binary-searches
    backwards by 30-day jumps to find the earliest available date. This
    avoids scanning thousands of unavailable historical chunks one-by-one.
    Results are always cached (even empty) so future folds skip the probe.
    """
    import requests
    import pandas as pd

    cache_path = data_dir / f"ls_ratio_{symbol}.parquet"
    empty_df = pd.DataFrame(columns=["timestamp", "long_short_ratio", "long_account", "short_account"])

    # Always use cache if it exists and is readable — even an empty cache is
    # valid (means we already determined data is unavailable for this symbol).
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            log.info(f"[LS_RATIO] Using cached L/S ratio data for {symbol}: {len(cached)} rows")
            return cached
        except Exception:
            pass

    candle_timestamps = candle_df['timestamp'].values
    start_ms = int(candle_timestamps.min())
    end_ms = int(candle_timestamps.max())

    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    chunk_ms = 500 * 5 * 60 * 1000  # 500 × 5-min bars ≈ 41.7 hours

    def _probe(ts_start, ts_end):
        """Fetch one chunk. Returns list of records on success, None on HTTP error."""
        params = {
            "symbol": symbol,
            "period": "5m",
            "limit": 500,
            "startTime": int(ts_start),
            "endTime": int(ts_end),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "timestamp": int(d["timestamp"]),
                        "long_short_ratio": float(d["longShortRatio"]),
                        "long_account": float(d["longAccount"]),
                        "short_account": float(d["shortAccount"]),
                    }
                    for d in data
                ]
            return None
        except Exception:
            return None

    # Step 1 — probe the most recent chunk to see if ANY data is available.
    probe_start = max(start_ms, end_ms - chunk_ms)
    if _probe(probe_start, end_ms) is None:
        log.warning(f"[LS_RATIO] No data available for {symbol} (recent probe failed) — using zeros")
        try:
            empty_df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return empty_df

    # Step 2 — binary-search backwards by 30-day jumps to find earliest available date.
    STEP_BACK = 30 * 24 * 60 * 60 * 1000  # 30 days in ms
    boundary_start = probe_start
    test_start = boundary_start - STEP_BACK
    while test_start >= start_ms:
        if _probe(test_start, test_start + chunk_ms) is not None:
            boundary_start = test_start
            test_start -= STEP_BACK
        else:
            break  # data not available this far back — boundary found

    log.info(
        f"[LS_RATIO] Data available from "
        f"{pd.Timestamp(boundary_start, unit='ms').date()} for {symbol} — fetching forward"
    )

    # Step 3 — fetch all chunks from the boundary to end_ms.
    all_records = []
    current_start = boundary_start
    gap_logged = False
    while current_start < end_ms:
        chunk_end = min(current_start + chunk_ms, end_ms)
        result = _probe(current_start, chunk_end)
        if result is not None:
            all_records.extend(result)
            gap_logged = False
        else:
            if not gap_logged:
                log.warning(
                    f"[LS_RATIO] Gap in data at "
                    f"{pd.Timestamp(current_start, unit='ms').date()} for {symbol}"
                )
                gap_logged = True
        current_start = chunk_end

    if not all_records:
        log.warning(f"[LS_RATIO] No records collected for {symbol} — using zeros")
        try:
            empty_df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return empty_df

    df = (
        pd.DataFrame(all_records)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass
    log.info(f"[LS_RATIO] Fetched and cached {len(df)} L/S ratio records for {symbol}")
    return df


def compute_ls_ratio_features(candle_df, ls_df):
    """Compute L/S ratio features aligned to candle timestamps.

    Returns DataFrame with 4 columns: ls_ratio, ls_deviation, ls_extreme, crowd_sentiment
    """
    import numpy as np
    import pandas as pd

    n = len(candle_df)
    result = pd.DataFrame(index=candle_df.index)

    if ls_df.empty or len(ls_df) < 2:
        for col in LS_RATIO_FEATURE_NAMES:
            result[col] = 0.0
        log.info(f"[LS_RATIO_FEAT] No L/S data — returning zeros ({n} rows)")
        return result

    ls_sorted = ls_df.sort_values("timestamp").copy()
    ls_sorted["ls_time_ms"] = ls_sorted["timestamp"]

    candle_ts = candle_df["timestamp"].values
    ls_ts = ls_sorted["ls_time_ms"].values
    ls_ratio_vals = ls_sorted["long_short_ratio"].values

    ls_ratio_arr = np.ones(n)
    ls_deviation_arr = np.zeros(n)
    ls_extreme_arr = np.zeros(n)
    crowd_sentiment_arr = np.zeros(n)

    rolling_window = 288 * 7

    for i in range(n):
        mask = ls_ts <= candle_ts[i]
        recent_idx = np.where(mask)[0]
        if len(recent_idx) == 0:
            continue
        recent_idx = recent_idx[-min(rolling_window, len(recent_idx)):]
        recent_vals = ls_ratio_vals[recent_idx]

        current = recent_vals[-1]
        ls_ratio_arr[i] = current

        if len(recent_vals) > 1:
            avg = recent_vals.mean()
            std = recent_vals.std()
            z_score = (current - avg) / max(std, 0.01)
            ls_deviation_arr[i] = np.clip(z_score, -5, 5)
            ls_extreme_arr[i] = 1.0 if abs(z_score) > 2 else 0.0

        if current > 2:
            crowd_sentiment_arr[i] = -1.0
        elif current < 0.5:
            crowd_sentiment_arr[i] = 1.0

    result["ls_ratio"] = ls_ratio_arr
    result["ls_deviation"] = ls_deviation_arr
    result["ls_extreme"] = ls_extreme_arr
    result["crowd_sentiment"] = crowd_sentiment_arr

    nz = int(np.sum(ls_ratio_arr != 1.0))
    log.info(f"[LS_RATIO_FEAT] Computed L/S features: {n} rows, {nz} non-default, "
             f"ratio range [{ls_ratio_arr.min():.3f}, {ls_ratio_arr.max():.3f}]")
    return result


def train_enter_model(data_path: Path, device: str, epochs: int, batch_size: int, lr: float,
                      checkpoint_interval: int = 25, warmup_epochs: int = 5, min_lr: float = None,
                      tp_mult: float = 2.0, sl_mult: float = 1.5, horizon: int = 16, slope_eps: float = 0.05,
                      r_min_expiry: float = 1.0, target_tpd: float = 5.5, target_tpd_tol: float = 1.5,
                      symbols: list = None, value_loss_weight: float = 0.5, value_clip: float = 3.0,
                      smoke_calib: bool = False, smoke_infer: bool = False,
                      use_focal_loss: bool = True, focal_gamma: float = 1.0, focal_alpha: float = 0.45,
                      use_ohem: bool = False, ohem_neg_pct: float = 0.08,
                      use_edge_head: bool = True, edge_loss_weight: float = 0.15,
                      use_soft_labels: bool = True, soft_label_temp: float = 1.2,
                      loss_warmup_epochs: int = 10, warmup_pos_weight: float = 2.0,
                      transition_epochs: int = 15, focal_gamma_final: float = 0.5,
                      lr_drop_on_transition: float = 0.65,
                      disable_ohem_during_transition: bool = True,
                      collapse_guard: bool = True,
                      collapse_guard_pred1: float = 0.98,
                      collapse_guard_sep: float = 0.02,
                      collapse_guard_freeze_epochs: int = 5,
                      verify_enter_metrics: bool = False,
                      w_quality: float = 1.0, w_dir: float = 0.5, w_htf: float = 0.5,
                      verify_v46_separation: bool = False,
                      use_v47_labels: bool = True,
                      q_min_tp: float = 0.3, r_min_expiry_strict: float = 1.0,
                      auto_balance_enter_labels: bool = True,
                      target_enter_rate: float = 0.18,
                      target_enter_rate_min: float = 0.12,
                      target_enter_rate_max: float = 0.25,
                      balance_search_steps: int = 30,
                      pos_weight_min: float = 0.5, pos_weight_max: float = 6.0,
                      verify_v47_labels: bool = False,
                      symbol_embed_dim: int = 8):
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from config import config

    log.info("=" * 60)
    log.info("  ENTER QUALITY MODEL - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {FEATURE_VERSION}")
    log.info(f"[PR_TUNE] focal(gamma={focal_gamma}, alpha={focal_alpha}) ohem(neg_pct={ohem_neg_pct}) edge_head(weight={edge_loss_weight}) soft_labels(temp={soft_label_temp})")
    log.info(f"[PR_AUC_PACK] focal_loss={use_focal_loss} (gamma={focal_gamma}, alpha={focal_alpha})")
    log.info(f"[PR_AUC_PACK] ohem={use_ohem} (neg_pct={ohem_neg_pct})")
    log.info(f"[PR_AUC_PACK] edge_head={use_edge_head} (weight={edge_loss_weight})")
    log.info(f"[PR_AUC_PACK] soft_labels={use_soft_labels} (temp={soft_label_temp})")
    log.info(f"[V46_COMPOSITE] w_quality={w_quality} w_dir={w_dir} w_htf={w_htf}")
    log.info(f"[LABEL_QUALITY] r_min_expiry={r_min_expiry}")
    log.info(f"[V47_CONFIG] use_v47_labels={use_v47_labels} q_min_tp={q_min_tp} r_min_expiry_strict={r_min_expiry_strict}")
    log.info(f"[V47_CONFIG] auto_balance={auto_balance_enter_labels} target_rate={target_enter_rate} range=[{target_enter_rate_min}, {target_enter_rate_max}]")
    log.info(f"[V47_CONFIG] pos_weight_guardrails=[{pos_weight_min}, {pos_weight_max}]")
    log.info(f"[TRANSITION] transition_epochs={transition_epochs} focal_gamma_final={focal_gamma_final} lr_drop={lr_drop_on_transition}")
    log.info(f"[TRANSITION] disable_ohem_during_transition={disable_ohem_during_transition}")
    log.info(f"[COLLAPSE_GUARD] enabled={collapse_guard} pred1_thresh={collapse_guard_pred1} sep_thresh={collapse_guard_sep} freeze={collapse_guard_freeze_epochs}")

    from data.regression_targets import RegressionTargetGenerator
    reg_gen = RegressionTargetGenerator(horizon_periods=horizon)
    tb_horizon = horizon
    reg_horizon = reg_gen.horizon_periods
    if tb_horizon == reg_horizon == horizon:
        log.info(f"[HORIZON_CHECK] triple_barrier={tb_horizon} model={horizon} regression={reg_horizon} OK")
    else:
        log.error(f"[HORIZON_CHECK] FAIL - mismatch detected: triple_barrier={tb_horizon} model={horizon} regression={reg_horizon}")

    from data.pipeline import FeatureEngineer
    data_dir = Path("data_cache")
    sequence_length = config.data.sequence_length

    # === MULTI-ASSET SUPPORT ===
    if symbols and len(symbols) > 1:
        log.info(f"[DATA] Multi-asset training: symbols={symbols}")
        all_train_features = []
        all_train_enter = []
        all_train_side = []
        all_train_r = []
        all_train_sym_ids = []
        all_train_ysoft = []
        all_train_edge = []
        all_train_dir = []
        all_train_dir_conf = []
        all_train_htf = []
        all_val_features = []
        all_val_enter = []
        all_val_side = []
        all_val_outcomes = []
        all_val_r = []
        all_val_sym_ids = []
        all_val_ysoft = []
        all_val_edge = []
        all_val_dir = []
        all_val_dir_conf = []
        all_val_htf = []
        feature_columns_ref = None

        for sym_idx, sym in enumerate(symbols):
            sym_data_path = data_dir / f"{sym}_15m.parquet"
            if not sym_data_path.exists():
                log.error(f"[DATA] No data for {sym} at {sym_data_path} — skipping (use --download-missing-data to auto-fetch)")
                continue

            sym_df = pd.read_parquet(sym_data_path)
            log.info(f"[DATA] {sym} (sym_idx={sym_idx}): {len(sym_df)} candles loaded from {sym_data_path}")

            sym_engineer = FeatureEngineer()
            sym_features_df = sym_engineer.compute_all_features(sym_df)
            sym_features_df = sym_features_df.fillna(0)

            sym_funding_df = fetch_funding_rates(sym_df, data_dir)
            sym_funding_features = compute_funding_features(sym_df, sym_funding_df)
            sym_features_df = pd.concat([sym_features_df, sym_funding_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            sym_oi_df = fetch_open_interest_hist(sym_df, data_dir, symbol=sym)
            sym_oi_enabled = _oi_sanity_check(sym_df, sym_oi_df, symbol=sym)
            if sym_oi_enabled:
                sym_oi_features = compute_oi_features(sym_df, sym_oi_df)
            else:
                sym_oi_features = pd.DataFrame(
                    np.zeros((len(sym_df), OI_FEATURE_COUNT)),
                    columns=OI_FEATURE_NAMES,
                    index=sym_df.index,
                )
            sym_features_df = pd.concat([sym_features_df, sym_oi_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            try:
                sym_ls_df = fetch_ls_ratio_hist(sym_df, data_dir, symbol=sym)
                sym_ls_features = compute_ls_ratio_features(sym_df, sym_ls_df)
            except Exception as e:
                log.warning(f"[LS_RATIO] Failed for {sym}: {e} — using zeros")
                sym_ls_features = pd.DataFrame(
                    np.zeros((len(sym_df), LS_RATIO_FEATURE_COUNT)),
                    columns=LS_RATIO_FEATURE_NAMES,
                    index=sym_df.index,
                )
            sym_features_df = pd.concat([sym_features_df, sym_ls_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            if feature_columns_ref is None:
                feature_columns_ref = list(sym_features_df.columns)

            htf_cols = [c for c in sym_features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
            sym_htf_df = sym_features_df[htf_cols].copy()

            if use_v47_labels:
                from data.regression_targets import generate_v47_quality_targets
                sym_label_df = generate_v47_quality_targets(
                    sym_df, sym_htf_df,
                    horizon_periods=horizon,
                    tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
                    q_min_tp=q_min_tp,
                    r_min_expiry_strict=r_min_expiry_strict,
                    soft_label_temp=soft_label_temp,
                    auto_balance=auto_balance_enter_labels,
                    target_enter_rate=target_enter_rate,
                    target_enter_rate_min=target_enter_rate_min,
                    target_enter_rate_max=target_enter_rate_max,
                    balance_search_steps=balance_search_steps,
                )
            else:
                from data.regression_targets import generate_v46_quality_targets
                sym_label_df = generate_v46_quality_targets(
                    sym_df, sym_htf_df,
                    horizon_periods=horizon,
                    tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
                    r_min_expiry=r_min_expiry,
                    soft_label_temp=soft_label_temp,
                )

            sym_enter = sym_label_df['y_quality'].values.astype(np.float32)
            sym_side = sym_label_df['side_hint'].values.astype(np.int64)
            sym_outcomes = sym_label_df['outcome'].values
            sym_realized_r = sym_label_df['realized_r'].values.astype(np.float64)
            sym_ysoft = sym_label_df['y_soft'].values.astype(np.float32) if 'y_soft' in sym_label_df.columns else np.full(len(sym_label_df), 0.5, dtype=np.float32)
            sym_mfe = sym_label_df['mfe_r'].values.astype(np.float32) if 'mfe_r' in sym_label_df.columns else np.zeros(len(sym_label_df), dtype=np.float32)
            sym_mae = sym_label_df['mae_r'].values.astype(np.float32) if 'mae_r' in sym_label_df.columns else np.zeros(len(sym_label_df), dtype=np.float32)
            sym_edge_target = np.nan_to_num(sym_mfe - sym_mae, nan=0.0).astype(np.float32)
            sym_dir_target = sym_label_df['y_dir'].values.astype(np.float32)
            sym_dir_conf = sym_label_df['y_dir_conf'].values.astype(np.float32)
            sym_htf_target = sym_label_df['y_htf_score'].values.astype(np.int64)

            valid_start = sequence_length
            sym_feat_np = sym_features_df.values[valid_start:].astype(np.float32)
            sym_enter_np = sym_enter[valid_start:]
            sym_side_np = sym_side[valid_start:]
            sym_outcomes_np = sym_outcomes[valid_start:]
            sym_r_np = sym_realized_r[valid_start:]
            sym_ysoft_np = np.nan_to_num(sym_ysoft[valid_start:], nan=0.5).astype(np.float32)
            sym_edge_np = sym_edge_target[valid_start:]
            sym_dir_np = sym_dir_target[valid_start:]
            sym_dir_conf_np = sym_dir_conf[valid_start:]
            sym_htf_np = sym_htf_target[valid_start:]

            n_sym = len(sym_feat_np)

            train_end_sym = int(n_sym * 0.70)
            val_end_sym = int(n_sym * 0.85)

            log.info(f"[SPLIT] sym={sym} total={n_sym} train={train_end_sym} val={val_end_sym - train_end_sym} test={n_sym - val_end_sym}")

            all_train_features.append(sym_feat_np[:train_end_sym])
            all_train_enter.append(sym_enter_np[:train_end_sym])
            all_train_side.append(sym_side_np[:train_end_sym])
            all_train_r.append(sym_r_np[:train_end_sym])
            all_train_sym_ids.append(np.full(train_end_sym, sym_idx, dtype=np.int64))
            all_train_ysoft.append(sym_ysoft_np[:train_end_sym])
            all_train_edge.append(sym_edge_np[:train_end_sym])
            all_train_dir.append(sym_dir_np[:train_end_sym])
            all_train_dir_conf.append(sym_dir_conf_np[:train_end_sym])
            all_train_htf.append(sym_htf_np[:train_end_sym])

            val_size = val_end_sym - train_end_sym
            all_val_features.append(sym_feat_np[train_end_sym:val_end_sym])
            all_val_enter.append(sym_enter_np[train_end_sym:val_end_sym])
            all_val_side.append(sym_side_np[train_end_sym:val_end_sym])
            all_val_outcomes.append(sym_outcomes_np[train_end_sym:val_end_sym])
            all_val_r.append(sym_r_np[train_end_sym:val_end_sym])
            all_val_sym_ids.append(np.full(val_size, sym_idx, dtype=np.int64))
            all_val_ysoft.append(sym_ysoft_np[train_end_sym:val_end_sym])
            all_val_edge.append(sym_edge_np[train_end_sym:val_end_sym])
            all_val_dir.append(sym_dir_np[train_end_sym:val_end_sym])
            all_val_dir_conf.append(sym_dir_conf_np[train_end_sym:val_end_sym])
            all_val_htf.append(sym_htf_np[train_end_sym:val_end_sym])

        train_features_raw = np.concatenate(all_train_features, axis=0)
        train_enter = np.concatenate(all_train_enter, axis=0)
        train_side = np.concatenate(all_train_side, axis=0)
        train_r = np.concatenate(all_train_r, axis=0)
        train_sym_ids = np.concatenate(all_train_sym_ids, axis=0)
        train_ysoft = np.concatenate(all_train_ysoft, axis=0)
        train_edge = np.concatenate(all_train_edge, axis=0)
        train_dir = np.concatenate(all_train_dir, axis=0)
        train_dir_conf = np.concatenate(all_train_dir_conf, axis=0)
        train_htf = np.concatenate(all_train_htf, axis=0)

        val_features_raw = np.concatenate(all_val_features, axis=0)
        val_enter = np.concatenate(all_val_enter, axis=0)
        val_side = np.concatenate(all_val_side, axis=0)
        val_outcomes = np.concatenate(all_val_outcomes, axis=0)
        val_r = np.concatenate(all_val_r, axis=0)
        val_sym_ids = np.concatenate(all_val_sym_ids, axis=0)
        val_ysoft = np.concatenate(all_val_ysoft, axis=0)
        val_edge = np.concatenate(all_val_edge, axis=0)
        val_dir = np.concatenate(all_val_dir, axis=0)
        val_dir_conf = np.concatenate(all_val_dir_conf, axis=0)
        val_htf = np.concatenate(all_val_htf, axis=0)

        n_symbols = len(symbols)
        features_columns_list = feature_columns_ref
        input_dim = train_features_raw.shape[1]
        val_bars = len(val_enter)

        log.info(f"[DATA] symbols={symbols} total_samples={len(train_enter) + len(val_enter)} per_symbol=[see above]")
        log.info(f"[SCALER] fit_on=train only | features={input_dim} | symbols={n_symbols}")

        engineer = FeatureEngineer()
        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_columns_list)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_columns_list)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_multi(features, enter, side, outcomes, r_vals, sym_ids, ysoft, edge, dir_t, dir_c, htf_t, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return (features[valid], enter[valid], side[valid], outcomes[valid], r_vals[valid],
                    sym_ids[valid], ysoft[valid], edge[valid], dir_t[valid], dir_c[valid], htf_t[valid])

        train_outcomes_dummy = np.full(len(train_enter), "NO_CANDIDATE", dtype=object)
        train_scaled, train_enter, train_side, _, train_r, train_sym_ids, train_ysoft, train_edge, train_dir, train_dir_conf, train_htf = clean_multi(
            train_scaled, train_enter, train_side, train_outcomes_dummy, train_r, train_sym_ids, train_ysoft, train_edge, train_dir, train_dir_conf, train_htf, "Train")
        val_scaled, val_enter, val_side, val_outcomes, val_r, val_sym_ids, val_ysoft, val_edge, val_dir, val_dir_conf, val_htf = clean_multi(
            val_scaled, val_enter, val_side, val_outcomes, val_r, val_sym_ids, val_ysoft, val_edge, val_dir, val_dir_conf, val_htf, "Val")

        features_df_columns = features_columns_list

    else:
        n_symbols = 1
        df = pd.read_parquet(data_path)
        log.info(f"Loaded {len(df)} candles")

        engineer = FeatureEngineer()
        features_df = engineer.compute_all_features(df)
        features_df = features_df.fillna(0)
        log.info(f"Computed {len(features_df.columns)} base features ({engineer.STF_FEATURE_COUNT} STF + {engineer.HTF_FEATURE_COUNT} HTF)")

        funding_df = fetch_funding_rates(df, data_dir)
        funding_features = compute_funding_features(df, funding_df)
        features_df = pd.concat([features_df, funding_features], axis=1)
        features_df = features_df.fillna(0)

        oi_df = fetch_open_interest_hist(df, data_dir)
        oi_enabled = _oi_sanity_check(df, oi_df)
        if oi_enabled:
            oi_features = compute_oi_features(df, oi_df)
        else:
            oi_features = pd.DataFrame(
                np.zeros((len(df), OI_FEATURE_COUNT)),
                columns=OI_FEATURE_NAMES,
                index=df.index,
            )
        features_df = pd.concat([features_df, oi_features], axis=1)
        features_df = features_df.fillna(0)

        sym_for_ls = Path(data_path).stem.split("_15m")[0]
        ls_df_single = fetch_ls_ratio_hist(df, data_dir, symbol=sym_for_ls)
        ls_features_single = compute_ls_ratio_features(df, ls_df_single)
        features_df = pd.concat([features_df, ls_features_single], axis=1)
        features_df = features_df.fillna(0)

        total_features = FeatureEngineer.TOTAL_FEATURE_COUNT + FUNDING_FEATURE_COUNT + OI_FEATURE_COUNT + LS_RATIO_FEATURE_COUNT
        actual_cols = len(features_df.columns)
        log.info(f"Total features: {actual_cols} ({FeatureEngineer.TOTAL_FEATURE_COUNT} base + {FUNDING_FEATURE_COUNT} funding + {OI_FEATURE_COUNT} OI + {LS_RATIO_FEATURE_COUNT} LS ratio)")
        if actual_cols != total_features:
            log.error(f"FATAL: Feature count mismatch! Expected {total_features}, got {actual_cols}")
            log.error(f"Columns: {sorted(features_df.columns.tolist())}")
            sys.exit(1)

        htf_cols = [c for c in features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
        htf_features_df = features_df[htf_cols].copy()
        log.info(f"HTF features for labeling: {htf_cols}")

        if use_v47_labels:
            n_labelable = len(df) - horizon
            valid_start_offset = sequence_length
            n_after_valid = n_labelable - valid_start_offset if n_labelable > valid_start_offset else n_labelable
            purge_gap_est = horizon + sequence_length
            val_est = max(int(n_after_valid * 0.1), purge_gap_est)
            train_est = n_after_valid - purge_gap_est - val_est
            train_mask_v47 = np.zeros(len(df), dtype=bool)
            train_mask_v47[valid_start_offset:valid_start_offset + max(train_est, 1)] = True

            from data.regression_targets import generate_v47_quality_targets
            label_df = generate_v47_quality_targets(
                df, htf_features_df,
                horizon_periods=horizon,
                tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
                q_min_tp=q_min_tp,
                r_min_expiry_strict=r_min_expiry_strict,
                soft_label_temp=soft_label_temp,
                auto_balance=auto_balance_enter_labels,
                target_enter_rate=target_enter_rate,
                target_enter_rate_min=target_enter_rate_min,
                target_enter_rate_max=target_enter_rate_max,
                balance_search_steps=balance_search_steps,
                train_mask=train_mask_v47,
            )
        else:
            from data.regression_targets import generate_v46_quality_targets
            label_df = generate_v46_quality_targets(
                df, htf_features_df,
                horizon_periods=horizon,
                tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
                r_min_expiry=r_min_expiry,
                soft_label_temp=soft_label_temp,
            )

        enter_labels = label_df['y_quality'].values.astype(np.float32)
        side_hints = label_df['side_hint'].values.astype(np.int64)
        precomputed_outcomes = label_df['outcome'].values
        precomputed_r = label_df['realized_r'].values.astype(np.float64)
        precomputed_ysoft = label_df['y_soft'].values.astype(np.float32) if 'y_soft' in label_df.columns else np.full(len(label_df), 0.5, dtype=np.float32)
        precomputed_mfe = label_df['mfe_r'].values.astype(np.float32) if 'mfe_r' in label_df.columns else np.zeros(len(label_df), dtype=np.float32)
        precomputed_mae = label_df['mae_r'].values.astype(np.float32) if 'mae_r' in label_df.columns else np.zeros(len(label_df), dtype=np.float32)
        precomputed_edge = np.nan_to_num(precomputed_mfe - precomputed_mae, nan=0.0).astype(np.float32)
        precomputed_dir = label_df['y_dir'].values.astype(np.float32)
        precomputed_dir_conf = label_df['y_dir_conf'].values.astype(np.float32)
        precomputed_htf = label_df['y_htf_score'].values.astype(np.int64)

        valid_start = sequence_length
        features_np = features_df.values[valid_start:].astype(np.float32)
        enter_np = enter_labels[valid_start:].astype(np.float32)
        side_np = side_hints[valid_start:].astype(np.int64)
        outcomes_np = precomputed_outcomes[valid_start:]
        r_np = precomputed_r[valid_start:]
        ysoft_np = np.nan_to_num(precomputed_ysoft[valid_start:], nan=0.5).astype(np.float32)
        edge_np = precomputed_edge[valid_start:]
        dir_np = precomputed_dir[valid_start:]
        dir_conf_np = precomputed_dir_conf[valid_start:]
        htf_np = precomputed_htf[valid_start:]

        n_total = len(features_np)
        purge_gap = horizon + sequence_length
        val_samples = max(int(n_total * 0.1), purge_gap)
        train_samples = n_total - purge_gap - val_samples

        if train_samples < sequence_length * 3:
            log.error(f"Not enough data for training: {train_samples} samples")
            sys.exit(1)

        train_end = train_samples
        val_start_idx = train_end + purge_gap
        val_end = val_start_idx + val_samples

        log.info(f"Data split: train={train_samples}, purge={purge_gap}, val={val_samples}")

        train_features_raw = features_np[:train_end]
        train_enter = enter_np[:train_end]
        train_side = side_np[:train_end]
        train_r = r_np[:train_end]
        train_ysoft = ysoft_np[:train_end]
        train_edge = edge_np[:train_end]
        train_dir = dir_np[:train_end]
        train_dir_conf = dir_conf_np[:train_end]
        train_htf = htf_np[:train_end]

        val_features_raw = features_np[val_start_idx:val_end]
        val_enter = enter_np[val_start_idx:val_end]
        val_side = side_np[val_start_idx:val_end]
        val_outcomes = outcomes_np[val_start_idx:val_end]
        val_r = r_np[val_start_idx:val_end]
        val_ysoft = ysoft_np[val_start_idx:val_end]
        val_edge = edge_np[val_start_idx:val_end]
        val_dir = dir_np[val_start_idx:val_end]
        val_dir_conf = dir_conf_np[val_start_idx:val_end]
        val_htf = htf_np[val_start_idx:val_end]
        val_bars = val_samples

        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_df.columns)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_df.columns)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_enter(features, enter, side, outcomes, r_vals, ysoft, edge, dir_t, dir_c, htf_t, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return (features[valid], enter[valid], side[valid], outcomes[valid], r_vals[valid],
                    ysoft[valid], edge[valid], dir_t[valid], dir_c[valid], htf_t[valid])

        train_outcomes_dummy = np.full(len(train_enter), "NO_CANDIDATE", dtype=object)
        train_r_dummy = np.zeros(len(train_enter), dtype=np.float64)
        train_scaled, train_enter, train_side, _, _, train_ysoft, train_edge, train_dir, train_dir_conf, train_htf = clean_enter(
            train_scaled, train_enter, train_side, train_outcomes_dummy, train_r_dummy, train_ysoft, train_edge, train_dir, train_dir_conf, train_htf, "Train")
        val_scaled, val_enter, val_side, val_outcomes, val_r, val_ysoft, val_edge, val_dir, val_dir_conf, val_htf = clean_enter(
            val_scaled, val_enter, val_side, val_outcomes, val_r, val_ysoft, val_edge, val_dir, val_dir_conf, val_htf, "Val")

        train_sym_ids = np.zeros(len(train_enter), dtype=np.int64)
        val_sym_ids = np.zeros(len(val_enter), dtype=np.int64)
        features_df_columns = list(features_df.columns)
        input_dim = features_np.shape[1]

    # === VALUE TARGETS (net_r) ===
    train_value_targets = np.nan_to_num(train_r, nan=0.0).astype(np.float32)
    train_value_targets = np.clip(train_value_targets, -value_clip, value_clip)
    val_value_targets = np.nan_to_num(val_r, nan=0.0).astype(np.float32)
    val_value_targets = np.clip(val_value_targets, -value_clip, value_clip)

    # === EDGE TARGETS (mfe_r - mae_r) ===
    train_edge_targets = np.clip(train_edge, -5.0, 5.0).astype(np.float32)
    val_edge_targets = np.clip(val_edge, -5.0, 5.0).astype(np.float32)

    # === SOFT LABELS ===
    train_ysoft_targets = train_ysoft.astype(np.float32)
    val_ysoft_targets = val_ysoft.astype(np.float32)

    pos_count = train_enter.sum()
    neg_count = len(train_enter) - pos_count
    val_pos_count = val_enter.sum()

    if int(pos_count) == 0 or int(val_pos_count) == 0:
        max_train_best_r = float(np.nanmax(train_r)) if len(train_r) > 0 else 0.0
        max_val_best_r = float(np.nanmax(val_r)) if len(val_r) > 0 else 0.0
        effective_q_min = label_df.attrs.get('v47_diagnostics', {}).get('q_min_tp', q_min_tp) if hasattr(label_df, 'attrs') else q_min_tp
        raise ValueError(
            f"[LABEL_ERROR] ENTER positives are zero (train_pos={int(pos_count)}, val_pos={int(val_pos_count)}). "
            f"q_min_tp={effective_q_min} max_feasible_train={max_train_best_r:.4f} max_feasible_val={max_val_best_r:.4f}"
        )

    raw_pos_weight = neg_count / max(pos_count, 1)
    pos_weight = max(pos_weight_min, min(pos_weight_max, raw_pos_weight))
    log.info(f"ENTER label distribution: ENTER=1: {int(pos_count)} ({100*pos_count/len(train_enter):.1f}%), ENTER=0: {int(neg_count)} ({100*neg_count/len(train_enter):.1f}%)")
    log.info(f"[POS_WEIGHT] raw={raw_pos_weight:.2f} capped={pos_weight:.2f} range=[{pos_weight_min}, {pos_weight_max}]")

    class EnterDataset(Dataset):
        def __init__(self, features, enter_labels, side_hints, symbol_ids, value_targets, edge_targets, ysoft_targets, dir_targets, dir_conf_targets, htf_targets, seq_len):
            self.features = features.astype(np.float32)
            self.enter_labels = enter_labels.astype(np.float32)
            self.side_hints = side_hints.astype(np.int64)
            self.symbol_ids = symbol_ids.astype(np.int64)
            self.value_targets = value_targets.astype(np.float32)
            self.edge_targets = edge_targets.astype(np.float32)
            self.ysoft_targets = ysoft_targets.astype(np.float32)
            self.dir_targets = dir_targets.astype(np.float32)
            self.dir_conf_targets = dir_conf_targets.astype(np.float32)
            self.htf_targets = htf_targets.astype(np.int64)
            self.seq_len = seq_len
            self.valid_indices = list(range(seq_len, len(features)))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            actual_idx = self.valid_indices[idx]
            start = actual_idx - self.seq_len
            seq = self.features[start:actual_idx]
            return (
                torch.from_numpy(seq),
                torch.tensor(self.enter_labels[actual_idx], dtype=torch.float32),
                torch.tensor(self.side_hints[actual_idx], dtype=torch.long),
                torch.tensor(self.symbol_ids[actual_idx], dtype=torch.long),
                torch.tensor(self.value_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.edge_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.ysoft_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.dir_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.dir_conf_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.htf_targets[actual_idx], dtype=torch.long),
            )

    train_dataset = EnterDataset(train_scaled, train_enter, train_side, train_sym_ids, train_value_targets, train_edge_targets, train_ysoft_targets, train_dir, train_dir_conf, train_htf, sequence_length)
    val_dataset = EnterDataset(val_scaled, val_enter, val_side, val_sym_ids, val_value_targets, val_edge_targets, val_ysoft_targets, val_dir, val_dir_conf, val_htf, sequence_length)

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        enable_enter_head=True,
        enable_quantile_head=False,
        enable_vol_state_head=False,
        enable_mu_head=False,
        enable_sigma_head=False,
        enable_value_head=True,
        enable_edge_head=use_edge_head,
        enable_dir_head=True,
        enable_htf_head=True,
        n_symbols=n_symbols,
        symbol_embed_dim=symbol_embed_dim if n_symbols > 1 else 0,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "EnterQualityMLP"
    model.to(device)
    log.info(f"Model: EnterQualityMLP ({model.parameters_count():,} parameters)")
    log.info(f"Architecture: [512, 256, 128, 64] with residual connections")
    active_heads = "enter_head (binary) + value_head (regression)"
    if use_edge_head:
        active_heads += " + edge_head (regression)"
    log.info(f"Active heads: {active_heads} | n_symbols={n_symbols}")

    pos_rate = pos_count / max(len(train_enter), 1)
    if 0 < pos_rate < 1:
        bias_init_val = float(np.log(pos_rate / (1 - pos_rate)))
    else:
        bias_init_val = 0.0
    with torch.no_grad():
        enter_head_last = model.enter_head[-1]
        enter_head_last.bias.fill_(bias_init_val)
    log.info(f"[BIAS_INIT] pos_rate={pos_rate:.4f} bias={bias_init_val:.4f}")
    log.info(f"[POS_WEIGHT] final_pos_weight={pos_weight:.2f}")

    def focal_bce_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.sigmoid(logits)
        p_t = torch.where(targets >= 0.5, p_t, 1 - p_t)
        focal_weight = (1 - p_t) ** gamma
        alpha_t = torch.where(targets >= 0.5, alpha, 1 - alpha)
        return (alpha_t * focal_weight * bce).mean()

    configured_pos_weight = pos_weight
    warmup_pw = min(configured_pos_weight, warmup_pos_weight)
    transition_end_epoch = loss_warmup_epochs + transition_epochs

    def make_warmup_criterion(pw):
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))

    def make_transition_criterion(current_pw, current_gamma):
        if use_focal_loss and current_gamma > 0.01:
            def transition_focal(logits, targets, _gamma=current_gamma, _alpha=focal_alpha):
                bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                p_t = torch.sigmoid(logits)
                p_t = torch.where(targets >= 0.5, p_t, 1 - p_t)
                focal_weight = (1 - p_t) ** _gamma
                alpha_t = torch.where(targets >= 0.5, _alpha, 1 - _alpha)
                pw_weight = torch.where(targets >= 0.5, current_pw, 1.0)
                return (alpha_t * focal_weight * bce * pw_weight).mean()
            return transition_focal
        else:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([current_pw]).to(device))

    def make_full_criterion():
        if use_focal_loss:
            def full_focal(logits, targets, _gamma=focal_gamma_final, _alpha=focal_alpha):
                bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                p_t = torch.sigmoid(logits)
                p_t = torch.where(targets >= 0.5, p_t, 1 - p_t)
                focal_weight = (1 - p_t) ** _gamma
                alpha_t = torch.where(targets >= 0.5, _alpha, 1 - _alpha)
                pw_weight = torch.where(targets >= 0.5, configured_pos_weight, 1.0)
                return (alpha_t * focal_weight * bce * pw_weight).mean()
            return full_focal
        else:
            c = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([configured_pos_weight]).to(device))
            return lambda logits, targets: c(logits, targets)

    log.info(f"[LOSS] Three-stage schedule: WARMUP={loss_warmup_epochs}ep (BCE, pw={warmup_pw:.2f}) -> TRANSITION={transition_epochs}ep (ramp pw/gamma) -> FULL (focal={'on' if use_focal_loss else 'off'}, ohem={'on' if use_ohem else 'off'})")
    log.info(f"[LOSS] TRANSITION: pos_weight {warmup_pw:.2f}->{configured_pos_weight:.2f}, focal_gamma 0.00->{focal_gamma_final:.2f}, ohem={'off' if disable_ohem_during_transition else 'on'}")
    log.info(f"[LOSS] FULL stage: focal gamma={focal_gamma_final}, alpha={focal_alpha}, pos_weight={configured_pos_weight:.2f}")

    collapse_guard_consecutive_pred1 = 0
    collapse_guard_active_remaining = 0
    collapse_guard_pw_backup = configured_pos_weight
    lr_dropped_at_transition = False

    value_criterion = nn.HuberLoss(delta=1.0)
    edge_criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    effective_min_lr = min_lr if min_lr is not None else lr * 0.05
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=effective_min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    for pg in optimizer.param_groups:
        pg['lr'] = lr * 1e-3

    log.info(f"Training for {epochs} epochs (lr={lr}, batch={batch_size})")
    log.info(f"LR schedule: {warmup_epochs}-epoch warmup -> cosine annealing to {effective_min_lr:.2e}")
    log.info(f"Early stopping: patience=50, min_epochs=40")
    log.info(f"Barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR, horizon={horizon} bars")
    log.info("-" * 60)

    best_val_loss = float('inf')
    best_val_prauc = 0.0
    patience = 0
    max_patience = 50
    min_epochs = 40
    last_warmup_prauc = None
    history = {'train_loss': [], 'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_prauc': [], '_sep_history': []}

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        is_warmup_stage = (epoch < loss_warmup_epochs)
        is_transition_stage = (not is_warmup_stage) and (epoch < transition_end_epoch)
        is_full_stage = (epoch >= transition_end_epoch)

        if collapse_guard_active_remaining > 0:
            collapse_guard_active_remaining -= 1

        if is_warmup_stage:
            stage_name = "WARMUP"
            enter_criterion_fn = make_warmup_criterion(warmup_pw)
            epoch_use_ohem = False
            epoch_pw = warmup_pw
            epoch_focal_gamma = 0.0
            epoch_ohem_str = "off"
        elif is_transition_stage:
            stage_name = "TRANSITION"
            t_progress = (epoch - loss_warmup_epochs) / max(transition_epochs - 1, 1)
            t_progress = min(max(t_progress, 0.0), 1.0)
            epoch_pw = warmup_pw + t_progress * (configured_pos_weight - warmup_pw)
            epoch_focal_gamma = t_progress * focal_gamma_final

            if collapse_guard_active_remaining > 0:
                epoch_pw = min(epoch_pw, 2.5)
                epoch_use_ohem = False
                log.info(f"[COLLAPSE_GUARD] active ({collapse_guard_active_remaining} epochs left) — clamping pw to {epoch_pw:.2f}, ohem=off")
            else:
                epoch_use_ohem = use_ohem if not disable_ohem_during_transition else False

            enter_criterion_fn = make_transition_criterion(epoch_pw, epoch_focal_gamma)
            epoch_ohem_str = "on" if epoch_use_ohem else "off"

            if not lr_dropped_at_transition and epoch == loss_warmup_epochs:
                lr_dropped_at_transition = True
                for pg in optimizer.param_groups:
                    old_lr = pg['lr']
                    pg['lr'] = old_lr * lr_drop_on_transition
                log.info(f"[LR_DROP] Transition start: LR {old_lr:.2e} -> {pg['lr']:.2e} (x{lr_drop_on_transition})")
        else:
            stage_name = "FULL"
            epoch_pw = configured_pos_weight
            epoch_focal_gamma = focal_gamma_final

            if collapse_guard_active_remaining > 0:
                epoch_pw = min(epoch_pw, 2.5)
                epoch_use_ohem = False
                log.info(f"[COLLAPSE_GUARD] active ({collapse_guard_active_remaining} epochs left) — clamping pw to {epoch_pw:.2f}, ohem=off")
                enter_criterion_fn = make_transition_criterion(epoch_pw, epoch_focal_gamma)
            else:
                epoch_use_ohem = use_ohem
                enter_criterion_fn = make_full_criterion()

            epoch_ohem_str = "on" if epoch_use_ohem else "off"

        current_lr_log = optimizer.param_groups[0]['lr']
        log.info(f"[LOSS_STAGE] stage={stage_name} epoch={epoch+1} pos_weight={epoch_pw:.2f} "
                 f"focal_gamma={epoch_focal_gamma:.3f} ohem={epoch_ohem_str} lr={current_lr_log:.2e}")

        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            features_batch, enter_batch, side_batch, sym_id_batch, value_batch, edge_batch, ysoft_batch, dir_batch, dir_conf_batch, htf_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)
            sym_id_batch = sym_id_batch.to(device)
            value_batch = value_batch.to(device)
            edge_batch = edge_batch.to(device)
            ysoft_batch = ysoft_batch.to(device)
            dir_batch = dir_batch.to(device)
            dir_conf_batch = dir_conf_batch.to(device)
            htf_batch = htf_batch.to(device)

            optimizer.zero_grad()
            output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
            enter_logits = output.enter_logits.squeeze(-1)

            if is_warmup_stage:
                enter_targets = enter_batch
            elif use_soft_labels:
                enter_targets = ysoft_batch
            else:
                enter_targets = enter_batch

            if epoch_use_ohem:
                with torch.no_grad():
                    per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                        enter_logits, enter_targets, reduction='none'
                    )
                pos_mask = enter_batch >= 0.5
                neg_mask = ~pos_mask
                n_pos = pos_mask.sum().item()
                n_neg = neg_mask.sum().item()
                if n_neg > 0 and n_pos > 0:
                    k_neg = max(int(n_neg * ohem_neg_pct), n_pos)
                    k_neg = min(k_neg, n_neg)
                    neg_losses = per_sample_loss[neg_mask]
                    _, hard_neg_idx = torch.topk(neg_losses, k_neg)
                    neg_indices = torch.where(neg_mask)[0]
                    selected_neg = neg_indices[hard_neg_idx]
                    pos_indices = torch.where(pos_mask)[0]
                    keep_indices = torch.cat([pos_indices, selected_neg])
                    enter_logits_ohem = enter_logits[keep_indices]
                    enter_targets_ohem = enter_targets[keep_indices]
                else:
                    enter_logits_ohem = enter_logits
                    enter_targets_ohem = enter_targets
                enter_loss = enter_criterion_fn(enter_logits_ohem, enter_targets_ohem)
            else:
                enter_loss = enter_criterion_fn(enter_logits, enter_targets)

            loss = enter_loss

            # === v4.5.2 Directional Separation Fix ===
            pos_mask_sep = (enter_batch >= 0.5)
            neg_mask_sep = (enter_batch < 0.5)
            if pos_mask_sep.any() and neg_mask_sep.any():
                mean_pos_logit = enter_logits[pos_mask_sep].mean()
                mean_neg_logit = enter_logits[neg_mask_sep].mean()
                sep = mean_pos_logit - mean_neg_logit
                sep_loss = torch.relu(0.30 - sep)
                flip_penalty = torch.relu(mean_neg_logit - mean_pos_logit)
                loss = loss + 0.03 * sep_loss + 0.05 * flip_penalty

            if output.value_logits is not None:
                value_pred = output.value_logits.squeeze(-1)
                v_loss = value_criterion(value_pred, value_batch)
                loss = loss + value_loss_weight * v_loss

            if use_edge_head and output.edge_logits is not None:
                edge_pred = output.edge_logits.squeeze(-1)
                e_loss = edge_criterion(edge_pred, edge_batch)
                loss = loss + edge_loss_weight * e_loss

            if output.dir_logits is not None:
                dir_pred = output.dir_logits.squeeze(-1)
                dir_loss = nn.functional.binary_cross_entropy_with_logits(
                    dir_pred, dir_batch, weight=dir_conf_batch, reduction='mean'
                )
                loss = loss + w_dir * dir_loss

            if output.htf_logits is not None:
                htf_loss = nn.functional.cross_entropy(output.htf_logits, htf_batch, reduction='mean')
                loss = loss + w_htf * htf_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss_total = 0
        val_n = 0
        all_probs = []
        all_targets = []
        all_sides = []
        all_value_preds = []
        all_value_targets_list = []
        all_enter_logits_list = []
        all_edge_preds = []
        all_edge_targets_list = []
        all_dir_probs = []
        all_dir_targets = []
        all_htf_preds = []
        all_htf_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features_batch, enter_batch, side_batch, sym_id_batch, value_batch, edge_batch, ysoft_batch, dir_batch, dir_conf_batch, htf_batch = batch
                features_batch = features_batch.to(device)
                enter_batch = enter_batch.to(device)
                sym_id_batch = sym_id_batch.to(device)
                value_batch = value_batch.to(device)
                edge_batch = edge_batch.to(device)
                dir_batch = dir_batch.to(device)
                dir_conf_batch = dir_conf_batch.to(device)
                htf_batch = htf_batch.to(device)

                output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
                enter_logits = output.enter_logits.squeeze(-1)
                enter_loss = enter_criterion_fn(enter_logits, enter_batch)

                batch_loss = enter_loss
                if output.value_logits is not None:
                    value_pred = output.value_logits.squeeze(-1)
                    v_loss_val = value_criterion(value_pred, value_batch)
                    batch_loss = batch_loss + value_loss_weight * v_loss_val
                    all_value_preds.extend(value_pred.cpu().numpy())
                    all_value_targets_list.extend(value_batch.cpu().numpy())

                if use_edge_head and output.edge_logits is not None:
                    edge_pred = output.edge_logits.squeeze(-1)
                    e_loss_val = edge_criterion(edge_pred, edge_batch)
                    batch_loss = batch_loss + edge_loss_weight * e_loss_val
                    all_edge_preds.extend(edge_pred.cpu().numpy())
                    all_edge_targets_list.extend(edge_batch.cpu().numpy())

                if output.dir_logits is not None:
                    dir_pred = output.dir_logits.squeeze(-1)
                    dir_probs_batch = torch.sigmoid(dir_pred).cpu().numpy()
                    all_dir_probs.extend(dir_probs_batch)
                    all_dir_targets.extend(dir_batch.cpu().numpy())

                if output.htf_logits is not None:
                    htf_pred_classes = torch.argmax(output.htf_logits, dim=-1).cpu().numpy()
                    all_htf_preds.extend(htf_pred_classes)
                    all_htf_targets.extend(htf_batch.cpu().numpy())

                val_loss_total += batch_loss.item()
                val_n += 1

                probs = torch.sigmoid(enter_logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(enter_batch.cpu().numpy())
                all_sides.extend(side_batch.numpy())
                all_enter_logits_list.extend(enter_logits.cpu().numpy())

        avg_val_loss = val_loss_total / max(val_n, 1)

        all_logits_np = np.array(all_enter_logits_list)
        all_targets = np.array(all_targets)
        all_sides = np.array(all_sides)

        if np.any(np.isnan(all_logits_np)) or np.any(np.isinf(all_logits_np)):
            raise RuntimeError(f"[PENTER_AUDIT] FATAL: NaN/Inf detected in enter_logits at epoch {epoch+1}! "
                               f"nan_count={np.isnan(all_logits_np).sum()} inf_count={np.isinf(all_logits_np).sum()}")

        all_probs = 1.0 / (1.0 + np.exp(-all_logits_np))

        if np.any(np.isnan(all_probs)) or np.any(np.isinf(all_probs)):
            raise RuntimeError(f"[PENTER_AUDIT] FATAL: NaN/Inf detected in p_enter (sigmoid of logits) at epoch {epoch+1}! "
                               f"nan_count={np.isnan(all_probs).sum()} inf_count={np.isinf(all_probs).sum()}")

        log.info(f"[PENTER_AUDIT] p_min={all_probs.min():.6f} p_max={all_probs.max():.6f} p_mean={all_probs.mean():.6f} "
                 f"logits_min={all_logits_np.min():.4f} logits_max={all_logits_np.max():.4f}")

        threshold = 0.5
        preds = (all_probs >= threshold).astype(int)
        tp = ((preds == 1) & (all_targets == 1)).sum()
        fp = ((preds == 1) & (all_targets == 0)).sum()
        fn = ((preds == 0) & (all_targets == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        pos_rate = all_targets.mean()

        try:
            from sklearn.metrics import average_precision_score
            prauc = average_precision_score(all_targets, all_probs) if all_targets.sum() > 0 else 0.0
        except ImportError:
            prauc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_prauc'].append(prauc)

        if is_warmup_stage:
            last_warmup_prauc = prauc
        elif epoch == loss_warmup_epochs and last_warmup_prauc is not None:
            prauc_drop = last_warmup_prauc - prauc
            if prauc_drop > 0.06:
                log.warning(f"[ALERT] PR-AUC drop at TRANSITION start — warmup_prauc={last_warmup_prauc:.3f} "
                            f"transition_prauc={prauc:.3f} drop={prauc_drop:.3f} (expected small dip during ramp)")

        current_lr = optimizer.param_groups[0]['lr']

        pos_mask_s = all_targets == 1
        neg_mask_s = all_targets == 0
        ml_pos = float(all_logits_np[pos_mask_s].mean()) if pos_mask_s.any() else 0.0
        ml_neg = float(all_logits_np[neg_mask_s].mean()) if neg_mask_s.any() else 0.0
        sep_val = ml_pos - ml_neg
        pred_pct = preds.mean()

        log.info(
            f"Epoch {epoch+1}/{epochs} | Loss T:{avg_train_loss:.4f} V:{avg_val_loss:.4f} | "
            f"P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%} | PR-AUC:{prauc:.3f} | "
            f"Pos:{pos_rate:.1%} | Pred1:{pred_pct:.1%} | LR:{current_lr:.2e}"
        )
        log.info(
            f"[SAFETY] Pos%={pos_rate:.1%} Pred%={pred_pct:.1%} "
            f"mean_logit_pos={ml_pos:.3f} mean_logit_neg={ml_neg:.3f} sep={sep_val:.3f} PR-AUC={prauc:.3f}"
        )
        log.info(f"[SEP_CHECK] mean_pos={ml_pos:.3f} mean_neg={ml_neg:.3f} sep={sep_val:.3f}")
        history['_sep_history'].append(sep_val)

        if collapse_guard and not is_warmup_stage:
            if pred_pct > collapse_guard_pred1:
                collapse_guard_consecutive_pred1 += 1
            else:
                collapse_guard_consecutive_pred1 = 0

            triggered = False
            if collapse_guard_consecutive_pred1 >= 2:
                triggered = True
                log.warning(f"[COLLAPSE_GUARD] TRIGGERED: Pred1={pred_pct:.1%} for {collapse_guard_consecutive_pred1} consecutive epochs (thresh={collapse_guard_pred1:.0%})")
            elif sep_val < collapse_guard_sep and pred_pct > 0.95:
                triggered = True
                log.warning(f"[COLLAPSE_GUARD] TRIGGERED: sep={sep_val:.3f}<{collapse_guard_sep} AND Pred1={pred_pct:.1%}>95%")

            if triggered and collapse_guard_active_remaining <= 0:
                collapse_guard_active_remaining = collapse_guard_freeze_epochs + 1
                collapse_guard_consecutive_pred1 = 0
                log.warning(f"[COLLAPSE_GUARD] Disabling OHEM and clamping pos_weight<=2.5 for next {collapse_guard_freeze_epochs} epochs")

        p50 = np.percentile(all_probs, 50)
        p75 = np.percentile(all_probs, 75)
        p90 = np.percentile(all_probs, 90)
        p95 = np.percentile(all_probs, 95)
        p99 = np.percentile(all_probs, 99)
        log.info(f"[PENTER_PCTL] p50={p50:.4f} p75={p75:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}")

        if all_value_preds:
            all_vp = np.array(all_value_preds)
            all_vt = np.array(all_value_targets_list)
            value_mae = np.mean(np.abs(all_vp - all_vt))
            value_rmse = np.sqrt(np.mean((all_vp - all_vt)**2))
        else:
            value_mae = value_rmse = 0.0

        if all_edge_preds:
            all_ep = np.array(all_edge_preds)
            all_et = np.array(all_edge_targets_list)
            edge_mae_val = np.mean(np.abs(all_ep - all_et))
            edge_rmse_val = np.sqrt(np.mean((all_ep - all_et)**2))
        else:
            edge_mae_val = edge_rmse_val = 0.0

        dir_auc_val = 0.0
        dir_acc_val = 0.0
        if all_dir_probs:
            all_dp = np.array(all_dir_probs)
            all_dt = np.array(all_dir_targets)
            dir_preds_bin = (all_dp >= 0.5).astype(int)
            dir_acc_val = np.mean(dir_preds_bin == (all_dt >= 0.5).astype(int))
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique((all_dt >= 0.5).astype(int))) > 1:
                    dir_auc_val = roc_auc_score((all_dt >= 0.5).astype(int), all_dp)
            except Exception:
                pass

        htf_macro_f1 = 0.0
        htf_acc_val = 0.0
        if all_htf_preds:
            all_hp = np.array(all_htf_preds)
            all_ht = np.array(all_htf_targets)
            htf_acc_val = np.mean(all_hp == all_ht)
            try:
                from sklearn.metrics import f1_score
                htf_macro_f1 = f1_score(all_ht, all_hp, average='macro', zero_division=0)
            except Exception:
                pass

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"[METRIC] mean_logit_pos={ml_pos:.3f} mean_logit_neg={ml_neg:.3f}")
            log.info(f"[METRIC] value_mae={value_mae:.4f} value_rmse={value_rmse:.4f}")
            if use_edge_head:
                log.info(f"[METRIC] edge_mae={edge_mae_val:.4f} edge_rmse={edge_rmse_val:.4f}")
            log.info(f"[METRIC] dir_auc={dir_auc_val:.4f} dir_acc={dir_acc_val:.4f}")
            log.info(f"[METRIC] htf_macro_f1={htf_macro_f1:.4f} htf_acc={htf_acc_val:.4f}")
            log.info(f"[METRIC] p_enter percentiles (val): p50={p50:.4f} p75={p75:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}")

        ckpt_model_config = {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128, 64],
            'num_classes': 3,
            'dropout': 0.3,
            'use_layer_norm': True,
            'use_residual': True,
            'enable_enter_head': True,
            'enable_quantile_head': False,
            'enable_vol_state_head': False,
            'enable_mu_head': False,
            'enable_sigma_head': False,
            'enable_value_head': True,
            'enable_edge_head': use_edge_head,
            'enable_dir_head': True,
            'enable_htf_head': True,
            'n_symbols': n_symbols,
            'symbol_embed_dim': symbol_embed_dim if n_symbols > 1 else 0,
        }
        ckpt_train_config = {
            'use_focal_loss': use_focal_loss,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'use_ohem': use_ohem,
            'ohem_neg_pct': ohem_neg_pct,
            'use_edge_head': use_edge_head,
            'edge_loss_weight': edge_loss_weight,
            'use_soft_labels': use_soft_labels,
            'soft_label_temp': soft_label_temp,
            'loss_warmup_epochs': loss_warmup_epochs,
            'warmup_pos_weight': warmup_pos_weight,
            'transition_epochs': transition_epochs,
            'focal_gamma_final': focal_gamma_final,
            'lr_drop_on_transition': lr_drop_on_transition,
            'disable_ohem_during_transition': disable_ohem_during_transition,
            'collapse_guard': collapse_guard,
            'collapse_guard_pred1': collapse_guard_pred1,
            'collapse_guard_sep': collapse_guard_sep,
            'collapse_guard_freeze_epochs': collapse_guard_freeze_epochs,
            'w_quality': w_quality,
            'w_dir': w_dir,
            'w_htf': w_htf,
            'use_v47_labels': use_v47_labels,
            'q_min_tp': q_min_tp,
            'r_min_expiry_strict': r_min_expiry_strict,
            'auto_balance_enter_labels': auto_balance_enter_labels,
            'target_enter_rate': target_enter_rate,
            'pos_weight_min': pos_weight_min,
            'pos_weight_max': pos_weight_max,
        }

        if prauc > best_val_prauc:
            best_val_prauc = prauc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps, 'r_min_expiry': r_min_expiry},
                'best_prauc': best_val_prauc,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_prauc.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps, 'r_min_expiry': r_min_expiry},
                'best_val_loss': best_val_loss,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_loss.pt")
        else:
            patience += 1

        if epoch + 1 >= min_epochs and patience >= max_patience:
            log.info(f"Early stopping at epoch {epoch+1} (patience={max_patience})")
            break

        MONITORING_INTERVAL = 5
        if (epoch + 1) % MONITORING_INTERVAL == 0:
            sweep_outcomes = val_outcomes[sequence_length:]
            sweep_r = val_r[sequence_length:]
            n_sweep = min(len(all_probs), len(sweep_outcomes))
            if len(all_probs) != len(sweep_outcomes):
                log.warning(f"Sweep alignment: probs={len(all_probs)} vs outcomes={len(sweep_outcomes)}, using min={n_sweep}")
            _run_enter_trading_sweep(
                all_probs[:n_sweep], all_targets[:n_sweep], all_sides[:n_sweep],
                sweep_outcomes[:n_sweep], sweep_r[:n_sweep],
                val_bars, epoch + 1, tp_mult, sl_mult,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
            )

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) < epochs:
            log.info("=" * 60)
            log.info(f"  CHECKPOINT @ Epoch {epoch+1}/{epochs}")
            log.info("=" * 60)
            log.info(f"  Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
            log.info(f"  PR-AUC: {prauc:.3f} | Best: {best_val_prauc:.3f}")
            log.info(f"  Patience: {patience}/{max_patience}")
            log.info(f"  P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%}")
            try:
                resp = input("Continue training? (Y/n): ").strip().lower()
                if resp == 'n':
                    log.info("User stopped training at checkpoint")
                    break
            except EOFError:
                pass

    scaler_path = checkpoint_dir / "scaler.joblib"
    engineer.save_scalers(str(scaler_path))
    log.info(f"Scaler saved to {scaler_path}")

    best_ckpt = checkpoint_dir / "best_enter_prauc.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"Loaded best PR-AUC checkpoint (PR-AUC={best_val_prauc:.3f})")

    # === TEMPERATURE SCALING CALIBRATION ===
    log.info("[CALIB] temperature_fit start | collecting val logits...")
    model.eval()
    cal_logits = []
    cal_labels = []
    with torch.no_grad():
        for batch in val_loader:
            features_batch, enter_batch, side_batch, sym_id_batch, value_batch, edge_batch, ysoft_batch, dir_batch, dir_conf_batch, htf_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)
            sym_id_batch = sym_id_batch.to(device)
            output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
            cal_logits.append(output.enter_logits.squeeze(-1).cpu())
            cal_labels.append(enter_batch.cpu())

    cal_logits = torch.cat(cal_logits)
    cal_labels = torch.cat(cal_labels)
    n_cal = len(cal_logits)
    log.info(f"[CALIB] temperature_fit start | n={n_cal}")

    nll_before = nn.BCEWithLogitsLoss()(cal_logits, cal_labels).item()

    log_T = torch.nn.Parameter(torch.zeros(1))
    temp_optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)

    def temp_closure():
        temp_optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = nn.BCEWithLogitsLoss()(cal_logits / T, cal_labels)
        loss.backward()
        return loss

    temp_optimizer.step(temp_closure)
    temperature = float(torch.exp(log_T).item())
    nll_after = nn.BCEWithLogitsLoss()(cal_logits / temperature, cal_labels).item()

    log.info(f"[CALIB] temperature={temperature:.4f} | nll_before={nll_before:.4f} nll_after={nll_after:.4f}")

    # === ECE COMPUTATION (before and after calibration) ===
    def compute_ece(probs_np, labels_np, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_details = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (probs_np >= lo) & (probs_np < hi)
            if i == n_bins - 1:
                mask = (probs_np >= lo) & (probs_np <= hi)
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_conf = probs_np[mask].mean()
            avg_acc = labels_np[mask].mean()
            bin_ece = abs(avg_conf - avg_acc) * (n_in_bin / len(probs_np))
            ece += bin_ece
            bin_details.append({'bin': f'{lo:.2f}-{hi:.2f}', 'n': int(n_in_bin), 'conf': float(avg_conf), 'acc': float(avg_acc)})
        return float(ece), bin_details

    cal_labels_np = cal_labels.numpy()
    probs_before = torch.sigmoid(cal_logits).numpy()
    probs_after = torch.sigmoid(cal_logits / temperature).numpy()
    ece_before, _ = compute_ece(probs_before, cal_labels_np)
    ece_after, ece_bins = compute_ece(probs_after, cal_labels_np)
    log.info(f"[CALIB] ECE before={ece_before:.4f} | ECE after={ece_after:.4f}")

    temp_scale_path = checkpoint_dir / "temp_scale_v5.0.json"
    import json
    temp_data = {
        "temperature": temperature,
        "fitted_on": "val",
        "version": FEATURE_VERSION,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_bins": ece_bins,
        "n_samples": n_cal,
    }
    with open(temp_scale_path, 'w') as f:
        json.dump(temp_data, f, indent=2)
    log.info(f"Temperature scale saved to {temp_scale_path}")

    if smoke_infer:
        log.info("[SMOKE] Running single-row inference per symbol...")
        model.eval()
        syms = symbols if symbols and len(symbols) > 1 else ["BTCUSDT"]
        for si, sym in enumerate(syms):
            if n_symbols > 1:
                sym_mask = val_sym_ids == si
                if sym_mask.any():
                    last_feat = val_scaled[sym_mask][-1:]
                else:
                    continue
            else:
                last_feat = val_scaled[-1:]

            with torch.no_grad():
                x = torch.FloatTensor(last_feat).to(device)
                sym_tensor = torch.tensor([si], dtype=torch.long, device=device) if n_symbols > 1 else None
                output = model.forward_multihead(x, symbol_ids=sym_tensor)
                logit = float(output.enter_logits.cpu().item())
                p = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
                e_net = float(output.value_logits.cpu().item()) if output.value_logits is not None else 0.0

                z_vals = last_feat[0]
                z_min = float(z_vals.min())
                z_max = float(z_vals.max())
                clipped = int(((z_vals <= -5.0) | (z_vals >= 5.0)).sum())

                log.info(f"[SMOKE] sym={sym} logit={logit:.4f} T={temperature:.4f} p={p:.4f} e_net_pred={e_net:.4f}")
                log.info(f"[SMOKE] z_min={z_min:.4f} z_max={z_max:.4f} clipped={clipped}/{len(z_vals)}")

    if verify_enter_metrics:
        log.info("=" * 60)
        log.info("  VERIFY-ENTER-METRICS MODE")
        log.info("=" * 60)
        model.eval()
        pass_results = []
        for pass_i in range(3):
            v_logits_list = []
            v_targets_list = []
            with torch.no_grad():
                for batch in val_loader:
                    fb, eb, sb, si, vb, edb, ysb, db, dcb, hb = batch
                    fb = fb.to(device)
                    si = si.to(device)
                    out = model.forward_multihead(fb, symbol_ids=si if n_symbols > 1 else None)
                    v_logits_list.extend(out.enter_logits.squeeze(-1).cpu().numpy())
                    v_targets_list.extend(eb.numpy())

            v_logits = np.array(v_logits_list)
            v_targets = np.array(v_targets_list)
            v_probs = 1.0 / (1.0 + np.exp(-v_logits))

            pred_pct_v = (v_probs >= 0.5).mean()
            pos_m = v_targets == 1
            neg_m = v_targets == 0
            ml_p = float(v_logits[pos_m].mean()) if pos_m.any() else 0.0
            ml_n = float(v_logits[neg_m].mean()) if neg_m.any() else 0.0
            sep_v = ml_p - ml_n

            try:
                from sklearn.metrics import average_precision_score
                prauc_v = average_precision_score(v_targets, v_probs) if v_targets.sum() > 0 else 0.0
            except ImportError:
                prauc_v = 0.0

            p75_v = float(np.percentile(v_probs, 75))
            p99_v = float(np.percentile(v_probs, 99))

            pass_results.append({
                'pass': pass_i + 1,
                'pred_pct': float(pred_pct_v),
                'prauc': float(prauc_v),
                'sep': float(sep_v),
                'p75': p75_v,
                'p99': p99_v,
                'p_min': float(v_probs.min()),
                'p_max': float(v_probs.max()),
                'p_mean': float(v_probs.mean()),
                'logits_min': float(v_logits.min()),
                'logits_max': float(v_logits.max()),
            })
            log.info(f"[VERIFY] pass={pass_i+1} Pred%={pred_pct_v:.4f} PR-AUC={prauc_v:.3f} sep={sep_v:.3f} p75={p75_v:.4f} p99={p99_v:.4f}")

        assertions = []

        avg_pred_pct = np.mean([r['pred_pct'] for r in pass_results])
        a_pred = 0.02 <= avg_pred_pct <= 0.60
        assertions.append(('A', f"Pred% in [2%, 60%] (avg={avg_pred_pct:.4f})", 'PASS' if a_pred else 'FAIL'))

        sep_hist = history.get('_sep_history', [])
        if len(sep_hist) >= 10:
            sep_e1 = sep_hist[0]
            sep_e10 = sep_hist[9]
            b_sep = sep_e10 > sep_e1
            assertions.append(('B', f"sep increase e1->e10 ({sep_e1:.3f} -> {sep_e10:.3f})", 'PASS' if b_sep else 'FAIL'))
        else:
            assertions.append(('B', f"sep increase (not enough epochs: {len(sep_hist)})", 'SKIP'))

        avg_p75 = np.mean([r['p75'] for r in pass_results])
        avg_p99 = np.mean([r['p99'] for r in pass_results])
        c_mono = avg_p99 > avg_p75
        assertions.append(('C', f"p99 > p75 (p99={avg_p99:.4f} > p75={avg_p75:.4f})", 'PASS' if c_mono else 'FAIL'))

        report_lines = [
            "# ENTER Metrics Verification Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Epochs trained: {len(history['val_prauc'])}",
            "",
            "## Pass Results",
            "| Pass | Pred% | PR-AUC | sep | p75 | p99 |",
            "|------|-------|--------|-----|-----|-----|",
        ]
        for r in pass_results:
            report_lines.append(f"| {r['pass']} | {r['pred_pct']:.4f} | {r['prauc']:.3f} | {r['sep']:.3f} | {r['p75']:.4f} | {r['p99']:.4f} |")

        report_lines.extend(["", "## Assertions"])
        for aid, desc, result in assertions:
            report_lines.append(f"- **{aid}**: {desc} -> **{result}**")

        last_audit = pass_results[-1]
        report_lines.extend([
            "",
            "## Last PENTER_AUDIT",
            f"p_min={last_audit['p_min']:.6f} p_max={last_audit['p_max']:.6f} p_mean={last_audit['p_mean']:.6f} "
            f"logits_min={last_audit['logits_min']:.4f} logits_max={last_audit['logits_max']:.4f}",
        ])

        report_path = Path("verify_enter_metrics.md")
        report_path.write_text("\n".join(report_lines))
        log.info(f"[VERIFY] Report written to {report_path}")

        for aid, desc, result in assertions:
            log.info(f"[VERIFY] Assertion {aid}: {result} — {desc}")

    if verify_v46_separation:
        log.info("=" * 60)
        log.info("  VERIFY-V46-SEPARATION MODE")
        log.info("=" * 60)

        log.info(f"[V46_VERIFY] y_quality: total={len(train_enter)}, pos={int((train_enter >= 0.5).sum())}, "
                 f"neg={int((train_enter < 0.5).sum())}, "
                 f"pos%={100*(train_enter >= 0.5).mean():.1f}%")
        log.info(f"[V46_VERIFY] y_dir: total={len(train_dir)}, "
                 f"long={int((train_dir >= 0.5).sum())}, short={int((train_dir < 0.5).sum())}, "
                 f"balance={100*(train_dir >= 0.5).mean():.1f}% long")
        import collections
        htf_counts = collections.Counter(train_htf.tolist())
        log.info(f"[V46_VERIFY] y_htf_score class counts: {dict(sorted(htf_counts.items()))}")

        assert train_dir.min() >= 0.0 and train_dir.max() <= 1.0, f"y_dir out of range: [{train_dir.min()}, {train_dir.max()}]"
        assert train_dir_conf.min() >= 0.0, f"y_dir_conf negative: {train_dir_conf.min()}"
        assert train_htf.min() >= 0 and train_htf.max() <= 3, f"y_htf_score out of range: [{train_htf.min()}, {train_htf.max()}]"
        log.info("[V46_VERIFY] Target ranges: PASS")

        model.eval()
        with torch.no_grad():
            test_batch = next(iter(val_loader))
            fb = test_batch[0].to(device)
            si = test_batch[3].to(device)
            out = model.forward_multihead(fb, symbol_ids=si if n_symbols > 1 else None)
            assert out.dir_logits is not None, "dir_head output is None"
            assert out.htf_logits is not None, "htf_head output is None"
            assert out.dir_logits.shape[-1] == 1, f"dir_logits shape mismatch: {out.dir_logits.shape}"
            assert out.htf_logits.shape[-1] == 4, f"htf_logits shape mismatch: {out.htf_logits.shape}"
            log.info(f"[V46_VERIFY] Model outputs: dir_logits={out.dir_logits.shape}, htf_logits={out.htf_logits.shape}")

        log.info("[V46_VERIFY] All checks PASSED")

    if verify_v47_labels and use_v47_labels:
        log.info("=" * 60)
        log.info("  v4.7 LABEL VERIFICATION")
        log.info("=" * 60)

        train_pos = float(train_enter.sum())
        train_total = float(len(train_enter))
        enter_rate = train_pos / max(train_total, 1)
        log.info(f"[V47_VERIFY] train ENTER=1: {int(train_pos)} / {int(train_total)} = {100*enter_rate:.1f}%")

        v47_checks = []
        v47_passes = 0

        if target_enter_rate_min <= enter_rate <= target_enter_rate_max:
            log.info(f"  [PASS] enter_rate={enter_rate:.3f} in [{target_enter_rate_min}, {target_enter_rate_max}]")
            v47_passes += 1
            v47_checks.append(('enter_rate_in_range', True, f"{enter_rate:.3f}"))
        else:
            log.warning(f"  [FAIL] enter_rate={enter_rate:.3f} NOT in [{target_enter_rate_min}, {target_enter_rate_max}]")
            v47_checks.append(('enter_rate_in_range', False, f"{enter_rate:.3f}"))

        val_pos = float(val_enter.sum())
        val_total = float(len(val_enter))
        val_enter_rate = val_pos / max(val_total, 1)
        log.info(f"[V47_VERIFY] val ENTER=1: {int(val_pos)} / {int(val_total)} = {100*val_enter_rate:.1f}%")

        report_lines = [
            "# v4.7.1 Label Verification Report",
            "",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Label Version**: v4.7.1 (TP Quality Score Balancing)",
            "",
            "## Configuration",
            f"- q_min_tp: {q_min_tp}",
            f"- r_min_expiry_strict: {r_min_expiry_strict}",
            f"- auto_balance: {auto_balance_enter_labels}",
            f"- target_enter_rate: {target_enter_rate}",
            f"- target_range: [{target_enter_rate_min}, {target_enter_rate_max}]",
            f"- pos_weight_min: {pos_weight_min}",
            f"- pos_weight_max: {pos_weight_max}",
            "",
            "## Training Set Statistics",
            f"- Total bars: {int(train_total)}",
            f"- ENTER=1: {int(train_pos)} ({100*enter_rate:.1f}%)",
            f"- ENTER=0: {int(train_total - train_pos)} ({100*(1-enter_rate):.1f}%)",
            "",
            "## Validation Set Statistics",
            f"- Total bars: {int(val_total)}",
            f"- ENTER=1: {int(val_pos)} ({100*val_enter_rate:.1f}%)",
            "",
            "## Assertions",
        ]

        for check_name, passed, value in v47_checks:
            status = "PASS" if passed else "FAIL"
            report_lines.append(f"- [{status}] {check_name}: {value}")

        report_lines.extend([
            "",
            f"## Result: {v47_passes}/{len(v47_checks)} checks passed",
        ])

        report_path = Path("verify_v47_labels.md")
        report_path.write_text("\n".join(report_lines))
        log.info(f"[V47_VERIFY] Report written to {report_path}")

        if v47_passes == len(v47_checks):
            log.info("[V47_VERIFY] All checks PASSED")
        else:
            log.warning(f"[V47_VERIFY] {len(v47_checks) - v47_passes} checks FAILED")

    return model, engineer, features_df_columns, history


def _select_trades_with_cooldown(probs, sides, precomputed_outcomes, precomputed_r, threshold, cooldown):
    """Select trades using threshold + cooldown, return precomputed outcomes for selected trades.
    
    Uses PRECOMPUTED outcomes from labeling triple-barrier (single source of truth).
    Only selects indices where side_hint != 0 (candidates) and p_enter >= threshold.
    """
    import numpy as np
    
    candidate_mask = (probs >= threshold) & (sides != 0)
    
    selected_indices = []
    last_trade = -cooldown - 1
    for i in range(len(candidate_mask)):
        if candidate_mask[i] and (i - last_trade) > cooldown:
            selected_indices.append(i)
            last_trade = i
    
    if not selected_indices:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    sel = np.array(selected_indices)
    sel_outcomes = precomputed_outcomes[sel]
    sel_r = precomputed_r[sel]
    
    valid_mask = ~np.isnan(sel_r.astype(float))
    sel_outcomes = sel_outcomes[valid_mask]
    sel_r = sel_r[valid_mask]
    sel = sel[valid_mask]
    
    return sel_outcomes, sel_r, sel




def _compute_sweep_metrics(outcomes, r_values, val_bars):
    """Compute metrics from precomputed triple-barrier outcomes.
    
    Args:
        outcomes: Array of "TP", "SL", "EXP_WIN", "EXP_LOSS" strings
        r_values: Array of realized R-multiples
        val_bars: Total number of validation bars (for trades_per_day)
    """
    import numpy as np
    
    n_trades = len(outcomes)
    if n_trades == 0:
        return {
            'trades': 0, 'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0,
            'avg_win_r': 0, 'avg_loss_r': 0, 'median_r': 0,
            'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0, 'trades_per_day': 0,
        }
    
    r_values = np.array(r_values, dtype=np.float64)
    
    expect = float(np.mean(r_values))
    wins = (r_values > 0).sum()
    winrate = wins / n_trades
    
    pos_r = r_values[r_values > 0]
    neg_r = r_values[r_values < 0]
    gross_profit = float(pos_r.sum()) if len(pos_r) > 0 else 0.0
    gross_loss = float(abs(neg_r.sum())) if len(neg_r) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    avg_win_r = float(np.mean(pos_r)) if len(pos_r) > 0 else 0.0
    avg_loss_r = float(np.mean(neg_r)) if len(neg_r) > 0 else 0.0
    median_r = float(np.median(r_values))
    
    val_days = val_bars / 96.0
    trades_per_day = n_trades / val_days if val_days > 0 else 0.0
    trades_per_year = trades_per_day * 365.0
    
    std_r = float(np.std(r_values))
    if std_r > 1e-8 and n_trades > 1:
        sharpe = float(np.mean(r_values) / std_r * np.sqrt(max(trades_per_year, 1)))
    else:
        sharpe = 0.0
    
    outcomes_arr = np.array(outcomes)
    pct_tp = float((outcomes_arr == "TP").sum() / n_trades)
    pct_sl = float((outcomes_arr == "SL").sum() / n_trades)
    n_exp = ((outcomes_arr == "EXP_WIN") | (outcomes_arr == "EXP_LOSS")).sum()
    pct_exp = float(n_exp / n_trades)
    
    return {
        'trades': n_trades, 'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf,
        'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r, 'median_r': median_r,
        'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
        'trades_per_day': trades_per_day,
    }


def _run_enter_trading_sweep(probs, targets, sides, precomputed_outcomes, precomputed_r,
                              val_bars, epoch, tp_mult, sl_mult,
                              target_tpd=5.5, target_tpd_tol=1.5, min_trades=50):
    """ENTER trading sweep using PRECOMPUTED triple-barrier outcomes (parity with labeling).
    
    Args:
        probs: Model p_enter probabilities for val set
        targets: True enter labels for val set
        sides: side_hint values for val set
        precomputed_outcomes: Outcome strings from labeling ("TP","SL","EXP_WIN","EXP_LOSS","NO_CANDIDATE")
        precomputed_r: Realized R-multiples from labeling (NaN for non-candidates)
        val_bars: Number of validation bars (for trades_per_day)
        epoch: Current epoch number
        tp_mult/sl_mult: Barrier config (for display only)
        target_tpd: Target trades per day
        target_tpd_tol: Tolerance band around target
        min_trades: Minimum trades for a valid sweep row
    """
    import numpy as np
    THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    PERCENTILES = [90, 85, 80, 75, 70]
    COOLDOWN = 4

    safe_outcomes = np.where(
        np.isin(precomputed_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]),
        precomputed_outcomes,
        "NO_CANDIDATE"
    )
    safe_r = np.where(np.isnan(precomputed_r.astype(float)), 0.0, precomputed_r.astype(float))

    sweep_results = []

    for thresh in THRESHOLDS:
        sel_outcomes, sel_r, sel_idx = _select_trades_with_cooldown(
            probs, sides, safe_outcomes, safe_r, thresh, COOLDOWN
        )
        m = _compute_sweep_metrics(sel_outcomes, sel_r, val_bars)
        m['label'] = f"{thresh*100:.0f}%"
        m['thresh'] = thresh
        sweep_results.append(m)

    active_probs = probs[sides != 0]
    if len(active_probs) > 0:
        for pct in PERCENTILES:
            pct_thresh = float(np.percentile(active_probs, pct))
            sel_outcomes, sel_r, sel_idx = _select_trades_with_cooldown(
                probs, sides, safe_outcomes, safe_r, pct_thresh, COOLDOWN
            )
            m = _compute_sweep_metrics(sel_outcomes, sel_r, val_bars)
            m['label'] = f"top{100-pct}%"
            m['thresh'] = pct_thresh
            sweep_results.append(m)

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    best_freq_score = float('-inf')
    best_freq_label = ""
    best_any_score = float('-inf')
    best_any_label = ""

    for m in sweep_results:
        if m['trades'] < min_trades:
            continue
        if tpd_lo <= m['trades_per_day'] <= tpd_hi:
            if m['expect'] > best_freq_score:
                best_freq_score = m['expect']
                best_freq_label = m['label']
        if m['expect'] > best_any_score:
            best_any_score = m['expect']
            best_any_label = m['label']

    if best_freq_label:
        best_label = best_freq_label
        best_score = best_freq_score
    elif best_any_label:
        best_label = best_any_label
        best_score = best_any_score
    else:
        best_label = ""
        best_score = float('-inf')

    probs_arr = np.array(probs)
    p50 = float(np.percentile(probs_arr, 50))
    p75 = float(np.percentile(probs_arr, 75))
    p90 = float(np.percentile(probs_arr, 90))
    p95 = float(np.percentile(probs_arr, 95))
    p99 = float(np.percentile(probs_arr, 99))
    log.info("-" * 115)
    log.info("p_enter percentiles (val): p50=%.3f p75=%.3f p90=%.3f p95=%.3f p99=%.3f", p50, p75, p90, p95, p99)
    val_days = val_bars / 96.0
    log.info("ENTER TRADING SWEEP (epoch %d) | cooldown=%d | TP=%.1fx SL=%.1fx ATR | val_days=%.1f | target=%.1f±%.1f tpd",
             epoch, COOLDOWN, tp_mult, sl_mult, val_days, target_tpd, target_tpd_tol)
    log.info("%-8s %5s %8s %6s %6s %5s | %6s %6s %6s | %4s %4s %4s | %5s",
             "Select", "Trds", "Expect", "WR", "Shrpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "T/Day")
    log.info("-" * 115)
    for m in sweep_results:
        in_freq = tpd_lo <= m['trades_per_day'] <= tpd_hi
        marker = ""
        if m['label'] == best_label and m['trades'] >= min_trades and best_score > float('-inf'):
            marker = " << BEST" + (" (freq)" if best_freq_label else " (any)")
        log.info("%-8s %5d %+.4f %5.1f%% %+6.2f %5.2f | %+5.2f %+5.2f %+5.2f | %3.0f%% %3.0f%% %3.0f%% | %5.1f%s",
                 m['label'], m['trades'], m['expect'], m['winrate'] * 100, m['sharpe'], m['pf'],
                 m['avg_win_r'], m['avg_loss_r'], m['median_r'],
                 m['pct_tp'] * 100, m['pct_sl'] * 100, m['pct_exp'] * 100,
                 m['trades_per_day'],
                 marker)
    log.info("-" * 115)


def make_enter_prediction(model, engineer, feature_columns, data_path, device):
    import torch
    import numpy as np
    import pandas as pd

    log.info("Generating ENTER QUALITY prediction from latest data...")

    df = pd.read_parquet(data_path)
    from data.pipeline import FeatureEngineer

    is_v5 = getattr(model, '_is_v5', False)
    if not is_v5:
        expected_count = FeatureEngineer.TOTAL_FEATURE_COUNT + FUNDING_FEATURE_COUNT + OI_FEATURE_COUNT
        if len(feature_columns) != expected_count:
            raise RuntimeError(
                f"FATAL: feature_columns has {len(feature_columns)} cols, expected {expected_count} ({FeatureEngineer.TOTAL_FEATURE_COUNT} base + {FUNDING_FEATURE_COUNT} funding + {OI_FEATURE_COUNT} OI). "
                f"Checkpoint mismatch - retrain the model."
            )

    features_df = engineer.compute_all_features(df)
    features_df = features_df.fillna(0)

    data_dir = Path("data_cache")
    funding_df = fetch_funding_rates(df, data_dir)
    funding_features = compute_funding_features(df, funding_df)
    features_df = pd.concat([features_df, funding_features], axis=1)
    features_df = features_df.fillna(0)

    oi_df = fetch_open_interest_hist(df, data_dir)
    oi_features = compute_oi_features(df, oi_df)
    features_df = pd.concat([features_df, oi_features], axis=1)
    features_df = features_df.fillna(0)

    missing = set(feature_columns) - set(features_df.columns)
    extra = set(features_df.columns) - set(feature_columns)
    if missing or extra:
        log.error(f"FATAL: Feature column mismatch!")
        if missing:
            log.error(f"  Missing: {sorted(missing)}")
        if extra:
            log.error(f"  Extra: {sorted(extra)}")
        raise RuntimeError(f"Feature column mismatch: {len(missing)} missing, {len(extra)} extra. Retrain.")

    features_df = features_df.reindex(columns=feature_columns, fill_value=0)

    seq_len = 16
    last_features = features_df.iloc[-seq_len:].copy()
    if len(last_features) < seq_len:
        pad_rows = seq_len - len(last_features)
        pad_df = pd.DataFrame(
            np.zeros((pad_rows, len(feature_columns)), dtype=np.float32),
            columns=feature_columns,
        )
        last_features = pd.concat([pad_df, last_features], ignore_index=True)

    if hasattr(engineer, '_v5_global_scaler'):
        raw = last_features.values.astype(np.float32)
        last_scaled = engineer._v5_global_scaler.transform(raw).astype(np.float32)
        last_scaled = np.clip(last_scaled, -5.0, 5.0)
    else:
        last_scaled = engineer.transform_and_clip(
            pd.DataFrame(last_features.values, columns=feature_columns),
            clip_range=5.0
        ).values.astype(np.float32)
    last_scaled = np.where(np.isinf(last_scaled), 0, last_scaled)
    last_scaled = np.where(np.isnan(last_scaled), 0, last_scaled)

    model.eval()
    p_long_v5 = None
    p_short_v5 = None
    p_hold_v5 = None
    with torch.no_grad():
        if is_v5:
            x = torch.FloatTensor(last_scaled).unsqueeze(0).to(device)
            output = model(x)
            action_probs = torch.softmax(output['action_logits'], dim=-1).cpu().numpy().flatten()
            # V5 action head: index 0=HOLD, 1=LONG, 2=SHORT (from v5_forecaster.py)
            p_hold_v5 = float(action_probs[0])
            p_long_v5 = float(action_probs[1])
            p_short_v5 = float(action_probs[2])
            p_enter = 1.0 - p_hold_v5   # prob of non-HOLD (backward compat metric)
        else:
            x = torch.FloatTensor(last_scaled[-1:]).to(device)
            output = model.forward_multihead(x)
            p_enter = float(torch.sigmoid(output.enter_logits).cpu().item())

    last_row = features_df.iloc[-1]
    h1_trend = last_row.get('h1_trend_sign', 0)
    h4_trend = last_row.get('h4_trend_sign', 0)
    h1_slope = last_row.get('h1_sma20_slope', 0)
    h1_range_pos = last_row.get('h1_range_pos', 0.5)

    if is_v5 and p_long_v5 is not None and p_short_v5 is not None:
        # V5: direction comes from action head directly (not HTF trend)
        # p_long and p_short are the model's own directional convictions
        if p_long_v5 >= p_short_v5:
            side = "LONG"
        else:
            side = "SHORT"
        # HTF gates still apply as secondary confirmation
        trend_aligned = True   # V5 action head supersedes HTF direction
        slope_ok = True        # V5 uses internal features — slope gate redundant
        range_ok = True        # V5 uses internal features — range gate redundant
        log.info(f"V5 directional probabilities: p_hold={p_hold_v5:.4f} p_long={p_long_v5:.4f} "
                 f"p_short={p_short_v5:.4f} → V5 side={side}")
    else:
        # Legacy enter-quality model: use HTF trend for direction
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

    current_price = float(df.iloc[-1]['close'])

    atr_window = min(20, len(df) - 1)
    if atr_window < 2:
        atr = current_price * 0.005
    else:
        highs = df.iloc[-atr_window:]['high'].values
        lows = df.iloc[-atr_window:]['low'].values
        true_ranges = []
        for i in range(1, len(highs)):
            prev_close = float(df.iloc[-atr_window + i - 1]['close'])
            tr = max(float(highs[i]) - float(lows[i]),
                     abs(float(highs[i]) - prev_close),
                     abs(float(lows[i]) - prev_close))
            true_ranges.append(tr)
        atr = float(np.mean(true_ranges))

    log.info("=" * 70)
    log.info("INFERENCE DIAGNOSTICS")
    log.info("=" * 70)
    log.info(f"Current price: {current_price:.2f} | ATR(14): {atr:.2f} ({100*atr/current_price:.2f}% of price)")
    log.info(f"p_enter: {p_enter:.4f}")
    if is_v5 and p_long_v5 is not None:
        log.info(f"V5 action head: p_hold={p_hold_v5:.4f}  p_long={p_long_v5:.4f}  p_short={p_short_v5:.4f}")
        log.info(f"V5 direction (from action head): {side}")
    else:
        log.info(f"HTF gates:")
        log.info(f"  h1_trend_sign={h1_trend:+.0f}  h4_trend_sign={h4_trend:+.0f}  aligned={'YES' if trend_aligned else 'NO'}")
        log.info(f"  h1_sma20_slope={h1_slope:.4f}  |slope|>0.05={'YES' if slope_ok else 'NO'}")
        log.info(f"  h1_range_pos={h1_range_pos:.3f}  range_ok={'YES' if range_ok else 'NO'}")
        log.info(f"  HTF direction: {side}")
    
    feature_diagnostics = {}
    for col in ['rsi_14', 'rsi_7', 'macd', 'adx_14', 'bb_position', 'volume_ratio',
                'funding_rate', 'funding_rate_zscore_30d', 'open_interest', 'oi_delta_1h']:
        if col in features_df.columns:
            val = float(last_row.get(col, 0))
            feature_diagnostics[col] = val
    
    log.info(f"Key features: {' | '.join(f'{k}={v:.4f}' for k, v in feature_diagnostics.items())}")
    
    nan_count = int(np.isnan(last_scaled).sum()) + int(np.isinf(last_scaled).sum())
    zero_count = int((last_scaled == 0).sum())
    log.info(f"Scaled feature coverage: {len(feature_columns)} features | NaN/Inf={nan_count} | zeros={zero_count}")
    log.info("=" * 70)

    enter_threshold = 0.55
    if is_v5 and p_long_v5 is not None and p_short_v5 is not None:
        # V5 gate: signal fires when p_side > p_hold AND non-HOLD class wins
        p_side_v5 = p_long_v5 if side == "LONG" else p_short_v5
        # Minimum conviction: p_side must beat HOLD (>0.333 for balanced 3-class)
        v5_threshold = 0.40   # slightly above random (0.333) to avoid noise signals
        should_trade = p_side_v5 >= v5_threshold and p_enter >= 0.40
    else:
        should_trade = p_enter >= enter_threshold and trend_aligned and slope_ok and range_ok

    if should_trade:
        action = side
    else:
        action = "HOLD"

    if action == "LONG":
        sl_price = current_price - 1.5 * atr
        tp_price = current_price + 2.0 * atr
    elif action == "SHORT":
        sl_price = current_price + 1.5 * atr
        tp_price = current_price - 2.0 * atr
    else:
        sl_price = current_price - 1.0 * atr
        tp_price = current_price + 1.0 * atr

    sl_pct = abs(current_price - sl_price) / current_price
    tp_pct = abs(tp_price - current_price) / current_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    ACCOUNT_RISK_PER_TRADE = 0.02
    if sl_pct > 0:
        position_size = ACCOUNT_RISK_PER_TRADE / sl_pct * 100
    else:
        position_size = 1.0
    if p_enter > 0.75:
        position_size *= 1.25
    elif p_enter < 0.55:
        position_size *= 0.5
    position_size = min(max(position_size, 0.5), 5.0)

    if is_v5 and p_long_v5 is not None:
        confidence = max(p_long_v5, p_short_v5)   # highest directional conviction
        edge = confidence - 1.0 / 3.0              # above uniform 3-class baseline
        dir_probs = {"SHORT": round(p_short_v5, 4),
                     "HOLD": round(p_hold_v5, 4),
                     "LONG": round(p_long_v5, 4)}
    else:
        confidence = p_enter
        edge = p_enter - 0.5
        dir_probs = {"SHORT": round(1.0 if side == "SHORT" else 0.0, 4),
                     "HOLD": round(1.0 if action == "HOLD" else 0.0, 4),
                     "LONG": round(1.0 if side == "LONG" else 0.0, 4)}

    prediction = {
        "action": action,
        "confidence": round(confidence, 4),
        "direction_probs": dir_probs,
        "quantiles": {},
        "vol_state": "neutral",
        "vol_state_probs": {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33},
        "expected_return": round(edge, 6),
        "uncertainty": round(1.0 - confidence, 6),
        "edge": round(edge, 4),
        "entry_price": round(current_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "enter_quality_v3.3_funding_oi",
        "is_multihead": True,
        "urgency": "high" if p_enter > 0.7 and should_trade else ("medium" if should_trade else "low"),
        "suggested_order_type": "limit",
        "reasons": [],
    }

    reasons = []
    if should_trade:
        reasons.append(f"ENTER signal: p_enter={p_enter:.1%}")
        reasons.append(f"HTF trend: {side} (1H={h1_trend:+.0f}, 4H={h4_trend:+.0f})")
        if abs(h1_slope) > 0.1:
            reasons.append(f"Strong trend slope ({h1_slope:.2f})")
    else:
        if not trend_aligned:
            reasons.append("HTF trends not aligned")
        if not slope_ok:
            reasons.append(f"Weak slope ({h1_slope:.2f})")
        if not range_ok:
            reasons.append(f"Range position against trend ({h1_range_pos:.2f})")
        if p_enter < enter_threshold:
            reasons.append(f"p_enter {p_enter:.1%} < {enter_threshold:.0%} threshold")

    prediction["reasons"] = reasons if reasons else ["No signal"]

    return prediction


def push_prediction(replit_url: str, prediction: dict):
    import requests

    url = f"{replit_url.rstrip('/')}/api/gpu/push-prediction"
    log.info(f"Pushing prediction to dashboard...")
    log.info(f"  Action: {prediction['action']} | Confidence: {prediction['confidence']:.1%}")
    log.info(f"  Price: ${prediction['current_price']:,.2f}")
    log.info(f"  Entry: ${prediction['entry_price']:,.2f} | SL: ${prediction['stop_loss_price']:,.2f} | TP: ${prediction['take_profit_price']:,.2f}")

    try:
        resp = requests.post(url, json=prediction, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        log.info(f"  Pushed successfully! (id={result.get('id', '?')})")
        return True
    except Exception as e:
        log.error(f"  Failed to push: {e}")
        return False


def _compute_r_metrics(r_arr, regime_days):
    """Compute expectancy, win rate, PF, avg win/loss R, Sharpe from an R-multiple array."""
    import numpy as np
    if len(r_arr) == 0:
        return 0, 0, 0, 0, 0, 0
    e = float(np.mean(r_arr))
    w = float((r_arr > 0).sum() / len(r_arr))
    pos = r_arr[r_arr > 0]
    neg = r_arr[r_arr < 0]
    gp = float(pos.sum()) if len(pos) > 0 else 0.0
    gl = float(abs(neg.sum())) if len(neg) > 0 else 0.0
    pf = gp / gl if gl > 0 else 0.0
    aw = float(np.mean(pos)) if len(pos) > 0 else 0.0
    al = float(np.mean(neg)) if len(neg) > 0 else 0.0
    std = float(np.std(r_arr))
    tpy = (len(r_arr) / regime_days * 365.0) if regime_days > 0 else 0
    sh = float(np.mean(r_arr) / std * np.sqrt(max(tpy, 1))) if std > 1e-8 and len(r_arr) > 1 else 0.0
    return e, w, pf, aw, al, sh


def _empty_regime_result(regime_name, n_bars):
    return {
        'name': regime_name, 'bars': n_bars, 'trades': 0, 'trades_per_day': 0,
        'expect_gross': 0, 'expect_net': 0, 'expect_sized': 0,
        'winrate_gross': 0, 'winrate_net': 0,
        'sharpe_gross': 0, 'sharpe_net': 0,
        'pf_gross': 0, 'pf_net': 0, 'pf_sized': 0,
        'avg_win_r_gross': 0, 'avg_loss_r_gross': 0,
        'avg_win_r_net': 0, 'avg_loss_r_net': 0,
        'total_cost_r': 0, 'total_size_mult': 0,
        'avg_cost_r': 0, 'avg_size_mult': 0,
        'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0,
    }


def _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps):
    """Load model, compute features/inference, HTF gates. Returns shared context dict."""
    import torch
    import numpy as np
    import pandas as pd
    from training.triple_barrier import compute_atr_14

    checkpoint_path = None
    for candidate in [
        Path("checkpoints/best_enter_prauc.pt"),
        Path("checkpoints/best_v5_expectancy.pt"),
        Path("checkpoints/best_enter_loss.pt"),
        Path("checkpoints/best_v5_loss.pt"),
    ]:
        if candidate.exists():
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        log.error("No trained ENTER model found! Train first.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_version = checkpoint.get('feature_version', 'unknown')
    ACCEPTED_VERSIONS = {FEATURE_VERSION, "v5.0.1_forecaster"}
    if saved_version not in ACCEPTED_VERSIONS:
        log.error(f"FATAL: Feature version mismatch! Model: '{saved_version}', accepted: {ACCEPTED_VERSIONS}")
        sys.exit(1)
    log.info(f"Checkpoint: {checkpoint_path.name} | Feature version: {saved_version}")

    feature_columns = checkpoint.get('feature_columns', [])
    if not feature_columns:
        log.error("FATAL: No feature_columns in checkpoint - retrain.")
        sys.exit(1)

    cfg = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'legacy')

    if model_type == 'v5_forecaster':
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
    else:
        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        mlp_config = EnhancedMultiHeadMLP_Config(
            input_dim=cfg.get('input_dim', 63),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            num_classes=3, dropout=0.3, use_layer_norm=True, use_residual=True,
            enable_enter_head=True, enable_quantile_head=False,
            enable_vol_state_head=False, enable_mu_head=False, enable_sigma_head=False,
            enable_dir_head=cfg.get('enable_dir_head', False),
            enable_htf_head=cfg.get('enable_htf_head', False),
        )
        model = EnhancedMultiHeadMLP(mlp_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model._is_v5 = False

    model.to(device)
    model.eval()
    log.info(f"Model loaded: {model.parameters_count():,} parameters ({model_type})")

    from data.pipeline import FeatureEngineer
    engineer = FeatureEngineer()
    v5_global_scaler = None

    if model_type == 'v5_forecaster' and 'scaler_center' in checkpoint and 'scaler_scale' in checkpoint:
        from sklearn.preprocessing import RobustScaler
        v5_global_scaler = RobustScaler()
        v5_global_scaler.center_ = np.array(checkpoint['scaler_center'])
        v5_global_scaler.scale_ = np.array(checkpoint['scaler_scale'])
        log.info("Scaler loaded from checkpoint (embedded)")
    else:
        scaler_path = None
        for sname in ["per_symbol_scalers.joblib", "scaler.joblib"]:
            sp = Path(f"checkpoints/{sname}")
            if sp.exists():
                scaler_path = sp
                break
        if scaler_path:
            engineer.load_scalers(str(scaler_path))
            log.info(f"Scaler loaded from {scaler_path}")
        else:
            log.error("No saved scaler found! Predictions will be unreliable.")
            sys.exit(1)

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} candles")

    if 'timestamp' not in df.columns:
        log.error("Data must contain 'timestamp' column")
        sys.exit(1)

    features_df = engineer.compute_all_features(df)
    features_df = features_df.fillna(0)

    data_dir = Path("data_cache")
    funding_df = fetch_funding_rates(df, data_dir)
    funding_features = compute_funding_features(df, funding_df)
    features_df = pd.concat([features_df, funding_features], axis=1)
    features_df = features_df.fillna(0)

    oi_df = fetch_open_interest_hist(df, data_dir)
    oi_features = compute_oi_features(df, oi_df)
    features_df = pd.concat([features_df, oi_features], axis=1)
    features_df = features_df.fillna(0)

    missing = set(feature_columns) - set(features_df.columns)
    extra = set(features_df.columns) - set(feature_columns)
    if missing or extra:
        log.error(f"Feature column mismatch!")
        if missing:
            log.error(f"  Missing: {sorted(missing)}")
        if extra:
            log.error(f"  Extra: {sorted(extra)}")
        sys.exit(1)
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)
    log.info(f"Features: {len(feature_columns)} columns")

    if v5_global_scaler is not None:
        scaled_np = v5_global_scaler.transform(features_df.values).astype(np.float32)
        scaled_np = np.clip(scaled_np, -5.0, 5.0)
    else:
        scaled_df = engineer.transform_and_clip(features_df, clip_range=5.0)
        scaled_np = scaled_df.values.astype(np.float32)
    scaled_np = np.where(np.isinf(scaled_np), 0, scaled_np)
    scaled_np = np.where(np.isnan(scaled_np), 0, scaled_np)

    is_v5 = getattr(model, '_is_v5', False)
    log.info(f"Running {'V5' if is_v5 else 'legacy'} inference over {len(df)} bars...")
    all_p_enter = np.full(len(df), np.nan, dtype=np.float64)
    BATCH_SIZE = 1024
    valid_indices = list(range(len(scaled_np)))
    with torch.no_grad():
        for batch_start in range(0, len(valid_indices), BATCH_SIZE):
            batch_idx = valid_indices[batch_start:batch_start + BATCH_SIZE]
            batch_rows = scaled_np[batch_idx]
            batch_tensor = torch.FloatTensor(batch_rows).to(device)
            if is_v5:
                output = model(batch_tensor)
                action_probs = torch.softmax(output['action_logits'], dim=-1)
                p_batch = (1.0 - action_probs[:, 0]).cpu().numpy().flatten()
            else:
                output = model.forward_multihead(batch_tensor)
                p_batch = torch.sigmoid(output.enter_logits).cpu().numpy().flatten()
            for k, idx in enumerate(batch_idx):
                all_p_enter[idx] = p_batch[k]

    valid_predictions = int(np.sum(~np.isnan(all_p_enter)))
    log.info(f"Inference complete: {valid_predictions}/{len(df)} bars have predictions")

    h1_trend = features_df['h1_trend_sign'].values if 'h1_trend_sign' in features_df.columns else np.zeros(len(df))
    h4_trend = features_df['h4_trend_sign'].values if 'h4_trend_sign' in features_df.columns else np.zeros(len(df))
    h1_slope = features_df['h1_sma20_slope'].values if 'h1_sma20_slope' in features_df.columns else np.zeros(len(df))
    h1_range_pos = features_df['h1_range_pos'].values if 'h1_range_pos' in features_df.columns else np.full(len(df), 0.5)

    aligned = (h1_trend == h4_trend) & (h1_trend != 0)
    slope_ok_arr = np.abs(h1_slope) > slope_eps
    range_ok_arr = np.ones(len(df), dtype=bool)
    range_ok_arr[(h1_trend > 0) & (h1_range_pos < 0.2)] = False
    range_ok_arr[(h1_trend < 0) & (h1_range_pos > 0.8)] = False

    htf_pass = aligned & slope_ok_arr & range_ok_arr
    side_arr = np.where(h1_trend > 0, 1, np.where(h1_trend < 0, -1, 0)).astype(int)

    candidate_mask = htf_pass & (~np.isnan(all_p_enter))
    n_candidates = int(candidate_mask.sum())
    log.info(f"HTF-gated candidates: {n_candidates}/{valid_predictions} ({100*n_candidates/max(valid_predictions,1):.1f}%)")

    atr_full = compute_atr_14(df)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    timestamps_ms = df['timestamp'].values.astype(np.int64)

    regimes = []
    for regime_str in regimes_str.split(","):
        parts = regime_str.strip().split(":")
        start_str, end_str = parts[0], parts[1]
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + 86400 * 1000 - 1
        regimes.append((f"{start_str} to {end_str}", start_ms, end_ms))

    log.info(f"Regimes: {len(regimes)}")
    for name, s, e in regimes:
        mask = (timestamps_ms >= s) & (timestamps_ms <= e)
        log.info(f"  {name}: {mask.sum()} bars")

    return {
        'df': df, 'all_p_enter': all_p_enter, 'candidate_mask': candidate_mask,
        'side_arr': side_arr, 'atr_full': atr_full,
        'highs': highs, 'lows': lows, 'closes': closes,
        'timestamps_ms': timestamps_ms, 'regimes': regimes,
    }


def _eval_single_config(ctx, tp_mult, sl_mult, threshold, cooldown, horizon, r_min_expiry,
                         fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
                         verbose=True, debug_costs=False):
    """Evaluate a single (tp_mult, sl_mult, threshold, cooldown) config across all regimes.
    
    Returns (regime_results_list, overall_summary_dict).
    """
    import numpy as np
    from training.triple_barrier import triple_barrier_outcome_for_index, compute_trade_cost_r

    all_p_enter = ctx['all_p_enter']
    candidate_mask = ctx['candidate_mask']
    side_arr = ctx['side_arr']
    atr_full = ctx['atr_full']
    highs = ctx['highs']
    lows = ctx['lows']
    closes = ctx['closes']
    timestamps_ms = ctx['timestamps_ms']
    regimes = ctx['regimes']

    all_regime_results = []
    all_gross = []
    all_net = []
    all_sized = []

    for regime_name, start_ms, end_ms in regimes:
        regime_mask = (timestamps_ms >= start_ms) & (timestamps_ms <= end_ms)
        regime_indices = np.where(regime_mask)[0]

        if len(regime_indices) == 0:
            all_regime_results.append(_empty_regime_result(regime_name, 0))
            continue

        regime_candidates = candidate_mask[regime_indices]
        regime_p_enter = all_p_enter[regime_indices]

        trade_mask = regime_candidates & (regime_p_enter >= threshold)

        selected_local = []
        last_trade = -cooldown - 1
        for i in range(len(trade_mask)):
            if trade_mask[i] and (i - last_trade) > cooldown:
                selected_local.append(i)
                last_trade = i

        if not selected_local:
            all_regime_results.append(_empty_regime_result(regime_name, len(regime_indices)))
            continue

        global_indices = regime_indices[np.array(selected_local)]
        trade_p_enter = all_p_enter[global_indices]

        outcomes = []
        gross_r_values = []
        cost_r_values = []
        for gi in global_indices:
            side = int(side_arr[gi])
            atr_i = float(atr_full[gi])
            outcome, r = triple_barrier_outcome_for_index(
                highs, lows, closes, gi, side, atr_i,
                tp_mult, sl_mult, horizon, r_min_expiry,
            )
            cost_r = compute_trade_cost_r(
                float(closes[gi]), atr_i, sl_mult,
                fees_bps_entry, fees_bps_exit, spread_bps, slip_k,
            )
            outcomes.append(outcome)
            gross_r_values.append(r)
            cost_r_values.append(cost_r)

        outcomes = np.array(outcomes)
        gross_r = np.array(gross_r_values, dtype=np.float64)
        cost_r_arr = np.array(cost_r_values, dtype=np.float64)
        net_r = gross_r - cost_r_arr

        size_mults = np.ones(len(trade_p_enter), dtype=np.float64)
        if threshold < 1.0:
            for k in range(len(trade_p_enter)):
                p = trade_p_enter[k]
                if p > threshold:
                    raw = 1.0 + (p - threshold) / (1.0 - threshold) * (size_cap - 1.0)
                    size_mults[k] = min(raw, size_cap)
        sized_net_r = net_r * size_mults

        valid_mask = ~np.isnan(gross_r) & ~np.isnan(cost_r_arr) & ~np.isnan(net_r)
        outcomes = outcomes[valid_mask]
        gross_r = gross_r[valid_mask]
        net_r = net_r[valid_mask]
        cost_r_arr = cost_r_arr[valid_mask]
        sized_net_r = sized_net_r[valid_mask]
        size_mults = size_mults[valid_mask]

        n_trades = len(gross_r)
        n_bars_regime = len(regime_indices)
        regime_days = n_bars_regime / 96.0

        if n_trades == 0:
            all_regime_results.append(_empty_regime_result(regime_name, n_bars_regime))
            continue

        eg, wg, pfg, awg, alg, shg = _compute_r_metrics(gross_r, regime_days)
        en, wn, pfn, awn, aln, shn = _compute_r_metrics(net_r, regime_days)
        es, _, pfs, _, _, _ = _compute_r_metrics(sized_net_r, regime_days)

        trades_per_day = n_trades / regime_days if regime_days > 0 else 0

        pct_tp = float((outcomes == "TP").sum() / n_trades)
        pct_sl = float((outcomes == "SL").sum() / n_trades)
        pct_exp = float(((outcomes == "EXP_WIN") | (outcomes == "EXP_LOSS")).sum() / n_trades)

        if debug_costs and n_trades > 0:
            import random
            sample_indices = random.sample(range(n_trades), min(5, n_trades))
            log.info(f"  [DEBUG COSTS] {regime_name} — {min(5, n_trades)} random trades:")
            for si in sample_indices:
                g = gross_r[si]; c = cost_r_arr[si]; n = net_r[si]
                log.info(f"    gross_r={g:+.4f}  cost_r={c:.4f}  net_r={n:+.4f}  check={g - c:+.4f}")
                assert abs(n - (g - c)) < 1e-9, f"Cost accounting mismatch: net_r={n} != gross_r={g} - cost_r={c}"

        all_gross.extend(gross_r.tolist())
        all_net.extend(net_r.tolist())
        all_sized.extend(sized_net_r.tolist())

        all_regime_results.append({
            'name': regime_name, 'bars': n_bars_regime, 'trades': n_trades,
            'trades_per_day': trades_per_day,
            'expect_gross': eg, 'expect_net': en, 'expect_sized': es,
            'winrate_gross': wg, 'winrate_net': wn,
            'sharpe_gross': shg, 'sharpe_net': shn,
            'pf_gross': pfg, 'pf_net': pfn, 'pf_sized': pfs,
            'avg_win_r_gross': awg, 'avg_loss_r_gross': alg,
            'avg_win_r_net': awn, 'avg_loss_r_net': aln,
            'total_cost_r': float(np.sum(cost_r_arr)),
            'total_size_mult': float(np.sum(size_mults)),
            'avg_cost_r': float(np.sum(cost_r_arr) / n_trades),
            'avg_size_mult': float(np.sum(size_mults) / n_trades),
            'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
        })

    total_bars = sum(r['bars'] for r in all_regime_results)
    total_trades = sum(r['trades'] for r in all_regime_results)
    total_days = total_bars / 96.0

    def _overall(r_list):
        r_arr = np.array(r_list, dtype=np.float64)
        if len(r_arr) == 0:
            return 0, 0, 0, 0, 0, 0
        e = float(np.mean(r_arr))
        w = float((r_arr > 0).sum() / len(r_arr))
        tpd = total_trades / total_days if total_days > 0 else 0
        pos = r_arr[r_arr > 0]; neg = r_arr[r_arr < 0]
        gp = float(pos.sum()) if len(pos) > 0 else 0
        gl = float(abs(neg.sum())) if len(neg) > 0 else 0
        pf = gp / gl if gl > 0 else 0
        aw = float(np.mean(pos)) if len(pos) > 0 else 0
        al = float(np.mean(neg)) if len(neg) > 0 else 0
        std = float(np.std(r_arr))
        tpy = tpd * 365.0
        sh = float(np.mean(r_arr) / std * np.sqrt(max(tpy, 1))) if std > 1e-8 and len(r_arr) > 1 else 0
        return e, w, pf, aw, al, sh

    og_e, og_w, og_pf, og_aw, og_al, og_sh = _overall(all_gross)
    on_e, on_w, on_pf, on_aw, on_al, on_sh = _overall(all_net)
    os_e, _, os_pf, _, _, _ = _overall(all_sized)
    overall_tpd = total_trades / total_days if total_days > 0 else 0

    profitable_regimes = sum(1 for r in all_regime_results if r['trades'] > 0 and r['pf_net'] > 1.0)
    regimes_with_trades = sum(1 for r in all_regime_results if r['trades'] > 0)

    total_cost_r_sum = sum(r.get('total_cost_r', 0) for r in all_regime_results if r['trades'] > 0)
    total_sz_sum = sum(r.get('total_size_mult', 0) for r in all_regime_results if r['trades'] > 0)
    avg_cost_r = float(total_cost_r_sum / total_trades) if total_trades > 0 else 0
    avg_sz_mul = float(total_sz_sum / total_trades) if total_trades > 0 else 1.0

    summary = {
        'tp_mult': tp_mult, 'sl_mult': sl_mult, 'threshold': threshold, 'cooldown': cooldown,
        'overall_pf_net': on_pf, 'overall_e_net': on_e, 'overall_tpd': overall_tpd,
        'overall_pf_gross': og_pf, 'overall_e_gross': og_e,
        'overall_pf_sized': os_pf, 'overall_e_sized': os_e,
        'overall_wr_net': on_w, 'overall_wr_gross': og_w,
        'overall_win_r_net': on_aw, 'overall_loss_r_net': on_al,
        'overall_win_r_gross': og_aw, 'overall_loss_r_gross': og_al,
        'overall_sharpe_net': on_sh, 'overall_sharpe_gross': og_sh,
        'profitable_regimes': profitable_regimes,
        'regimes_with_trades': regimes_with_trades,
        'total_trades': total_trades, 'total_bars': total_bars,
        'avg_cost_r': avg_cost_r, 'avg_size_mult': avg_sz_mul,
        'regime_results': all_regime_results,
    }

    if verbose:
        _print_regime_table(all_regime_results, summary,
                            tp_mult, sl_mult, threshold, cooldown, horizon,
                            fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap)

    return all_regime_results, summary


def _print_regime_table(regime_results, summary, tp_mult, sl_mult, threshold, cooldown,
                         horizon, fees_entry, fees_exit, spread, slip_k, size_cap):
    """Print the detailed regime table for a single config."""
    log.info("")
    log.info("=" * 160)
    log.info("REGIME ROBUSTNESS RESULTS (GROSS / NET / SIZED)")
    log.info(f"Config: TP={tp_mult}x SL={sl_mult}x thr={threshold} cd={cooldown} | Horizon={horizon}")
    log.info(f"Costs: entry={fees_entry}bps exit={fees_exit}bps spread={spread}bps slip_k={slip_k} | Size cap={size_cap}x")
    log.info("=" * 160)

    hdr = "%-26s %6s %5s %5s | %8s %8s %8s | %5s %5s | %5s %5s | %5s %5s %5s | %5s %5s | %4s %4s %4s"
    log.info(hdr, "Regime", "Bars", "Trds", "T/Day",
             "E[gross]", "E[net]", "E[sized]",
             "WR_g", "WR_n",
             "Sh_g", "Sh_n",
             "PF_g", "PF_n", "PF_s",
             "CostR", "SzMul",
             "%TP", "%SL", "%EX")
    log.info("-" * 160)

    for r in regime_results:
        log.info("%-26s %6d %5d %5.1f | %+8.4f %+8.4f %+8.4f | %5.1f%% %5.1f%% | %+5.2f %+5.2f | %5.2f %5.2f %5.2f | %5.3f %5.2f | %3.0f%% %3.0f%% %3.0f%%",
                 r['name'], r['bars'], r['trades'], r['trades_per_day'],
                 r['expect_gross'], r['expect_net'], r['expect_sized'],
                 r['winrate_gross'] * 100, r['winrate_net'] * 100,
                 r['sharpe_gross'], r['sharpe_net'],
                 r['pf_gross'], r['pf_net'], r['pf_sized'],
                 r['avg_cost_r'], r['avg_size_mult'],
                 r['pct_tp'] * 100, r['pct_sl'] * 100, r['pct_exp'] * 100)

    log.info("-" * 160)
    s = summary
    log.info("%-26s %6d %5d %5.1f | %+8.4f %+8.4f %+8.4f | %5.1f%% %5.1f%% | %+5.2f %+5.2f | %5.2f %5.2f %5.2f | %5.3f %5.2f |",
             "OVERALL", s['total_bars'], s['total_trades'], s['overall_tpd'],
             s['overall_e_gross'], s['overall_e_net'], s['overall_e_sized'],
             s['overall_wr_gross'] * 100, s['overall_wr_net'] * 100,
             s['overall_sharpe_gross'], s['overall_sharpe_net'],
             s['overall_pf_gross'], s['overall_pf_net'], s['overall_pf_sized'],
             s['avg_cost_r'], s['avg_size_mult'])
    log.info("=" * 160)

    log.info("")
    log.info("Win/Loss R breakdown:")
    log.info("%-26s | %+6s %+6s | %+6s %+6s", "Regime", "WinR_g", "LosR_g", "WinR_n", "LosR_n")
    log.info("-" * 80)
    for r in regime_results:
        if r['trades'] > 0:
            log.info("%-26s | %+6.2f %+6.2f | %+6.2f %+6.2f",
                     r['name'], r['avg_win_r_gross'], r['avg_loss_r_gross'],
                     r['avg_win_r_net'], r['avg_loss_r_net'])
    log.info("%-26s | %+6.2f %+6.2f | %+6.2f %+6.2f",
             "OVERALL", s['overall_win_r_gross'], s['overall_loss_r_gross'],
             s['overall_win_r_net'], s['overall_loss_r_net'])

    if s['regimes_with_trades'] > 1:
        net_expects = [r['expect_net'] for r in regime_results if r['trades'] > 0]
        if len(net_expects) > 1:
            import numpy as np
            en_std = float(np.std(net_expects))
            en_mean = float(np.mean(net_expects))
            log.info("")
            log.info(f"Cross-regime consistency (NET): mean(E_net)={en_mean:+.4f} std={en_std:.4f} CV={en_std/abs(en_mean) if abs(en_mean)>1e-8 else float('inf'):.2f}")
            log.info(f"Net-profitable regimes: {s['profitable_regimes']}/{s['regimes_with_trades']}")


def run_regime_eval(data_path: Path, device: str, regimes_str: str, policy_str: str,
                    cooldown: int, tp_mult: float, sl_mult: float, horizon: int,
                    slope_eps: float, r_min_expiry: float,
                    fees_bps_entry: float = 5.0, fees_bps_exit: float = 5.0,
                    spread_bps: float = 1.0, slip_k: float = 0.10,
                    size_cap: float = 2.0):
    """Evaluate one fixed trading policy across multiple date regimes (backward-compatible)."""
    log.info("=" * 80)
    log.info("  REGIME ROBUSTNESS EVALUATION")
    log.info("=" * 80)

    ctx = _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps)

    policy_type, policy_value = policy_str.split(":")
    policy_type = policy_type.lower().strip()
    policy_value_clean = policy_value.lower().replace("top", "").strip()
    threshold = float(policy_value_clean)

    if policy_type == "percentile":
        import numpy as np
        active_p = ctx['all_p_enter'][ctx['candidate_mask']]
        active_p = active_p[~np.isnan(active_p)]
        if len(active_p) == 0:
            threshold = 1.0
        else:
            pct = 100.0 - threshold
            threshold = float(np.percentile(active_p, max(pct, 0)))

    log.info(f"Policy: {policy_str} (threshold={threshold:.4f}) | Cooldown: {cooldown} | TP={tp_mult}x SL={sl_mult}x | Horizon={horizon}")
    log.info(f"Costs: entry={fees_bps_entry}bps exit={fees_bps_exit}bps spread={spread_bps}bps slip_k={slip_k} | Size cap={size_cap}x")

    regime_results, summary = _eval_single_config(
        ctx, tp_mult, sl_mult, threshold, cooldown, horizon, r_min_expiry,
        fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
        verbose=True,
    )
    log.info("")
    return regime_results


def run_geometry_sweep(data_path: Path, device: str, regimes_str: str,
                       tp_sl_pairs: list, thresholds: list, cooldowns: list,
                       horizon: int, slope_eps: float, r_min_expiry: float,
                       fees_bps_entry: float = 5.0, fees_bps_exit: float = 5.0,
                       spread_bps: float = 1.0, slip_k: float = 0.10,
                       size_cap: float = 2.0, debug_costs: bool = False,
                       topn_list: list = None,
                       target_tpd: float = 2.5, target_tpd_tol: float = 1.0):
    """Geometry sweep: evaluate multiple (tp, sl, policy, cooldown) configs in one run.
    
    Supports both threshold and percentile (topN) policies.
    Uses paired TP/SL combos (not cartesian product).
    Selects the BEST config using NET-first priority rules and saves to best_policy.json.
    """
    import numpy as np

    log.info("=" * 80)
    log.info(f"  GEOMETRY SWEEP ({SYSTEM_VERSION})")
    log.info("=" * 80)

    ctx = _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps)

    active_p = ctx['all_p_enter'][ctx['candidate_mask']]
    active_p = active_p[~np.isnan(active_p)]

    percentile_thresholds = {}
    if topn_list:
        if len(active_p) > 0:
            for topn in topn_list:
                pct = 100.0 - topn
                pct_thresh = float(np.percentile(active_p, max(pct, 0)))
                percentile_thresholds[topn] = pct_thresh
                log.info(f"  Percentile top{topn}: p_enter >= {pct_thresh:.4f}")
        else:
            log.warning("WARNING: topn_list requested but no valid active p_enter values found! Percentile policies will be skipped.")
            topn_list = None

    configs = []
    for tp, sl in tp_sl_pairs:
        for thr in thresholds:
            for cd in cooldowns:
                configs.append({
                    'tp': tp, 'sl': sl, 'threshold': thr, 'cooldown': cd,
                    'policy_type': 'threshold', 'policy_value': thr,
                })
        if topn_list:
            for topn in topn_list:
                if topn not in percentile_thresholds:
                    continue
                thr = percentile_thresholds[topn]
                for cd in cooldowns:
                    configs.append({
                        'tp': tp, 'sl': sl, 'threshold': thr, 'cooldown': cd,
                        'policy_type': 'percentile', 'policy_value': topn,
                    })

    n_thr = len(thresholds)
    n_pct = len(topn_list) if topn_list else 0
    n_policies = n_thr + n_pct
    log.info(f"Sweep: {len(configs)} configurations ({len(tp_sl_pairs)} TP/SL pairs x {n_policies} policies x {len(cooldowns)} cd)")
    log.info(f"  Threshold policies: {thresholds}")
    if topn_list:
        log.info(f"  Percentile policies: top{topn_list}")
    log.info(f"Costs: entry={fees_bps_entry}bps exit={fees_bps_exit}bps spread={spread_bps}bps slip_k={slip_k} | Size cap={size_cap}x")
    log.info(f"Horizon={horizon} | r_min_expiry={r_min_expiry}")
    log.info(f"Target TPD: {target_tpd} +/- {target_tpd_tol}")

    all_summaries = []

    for cfg_idx, cfg in enumerate(configs):
        tp, sl, thr, cd = cfg['tp'], cfg['sl'], cfg['threshold'], cfg['cooldown']
        pol_label = f"thr={thr:.2f}" if cfg['policy_type'] == 'threshold' else f"top{int(cfg['policy_value'])}(={thr:.4f})"
        log.info("")
        log.info(f"--- Config {cfg_idx+1}/{len(configs)}: TP={tp} SL={sl} {pol_label} cd={cd} ---")

        regime_results, summary = _eval_single_config(
            ctx, tp, sl, thr, cd, horizon, r_min_expiry,
            fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
            verbose=True, debug_costs=debug_costs,
        )
        summary['policy_type'] = cfg['policy_type']
        summary['policy_value'] = cfg['policy_value']
        all_summaries.append(summary)

    log.info("")
    log.info("=" * 160)
    log.info("GEOMETRY SWEEP SUMMARY")
    log.info("=" * 160)

    hdr = "%-4s %-5s %-5s %-12s %-3s | %7s | %7s | %5s | %7s | %7s | %6s | %6s | %5s"
    log.info(hdr, "#", "TP", "SL", "Policy", "cd",
             "PF_n", "E_n", "TPD", "WinR_n", "LosR_n",
             "CostR", "SzMul", "Prof")
    log.info("-" * 130)

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    min_profitable = 2 if len(ctx['regimes']) >= 3 else 1

    best_idx = -1
    passing_indices = []
    for i, s in enumerate(all_summaries):
        pf_ok = s['overall_pf_net'] >= 1.05
        en_ok = s['overall_e_net'] > 0
        tpd_ok = tpd_lo <= s['overall_tpd'] <= tpd_hi
        prof_ok = s['profitable_regimes'] >= min_profitable
        passes = pf_ok and en_ok and tpd_ok and prof_ok
        if passes:
            passing_indices.append(i)

    if passing_indices:
        best_idx = max(passing_indices, key=lambda i: all_summaries[i]['overall_pf_net'])
    else:
        best_pf_net = -999
        for i, s in enumerate(all_summaries):
            if s['overall_tpd'] >= 1.5 and s['overall_e_net'] > 0 and s['overall_pf_net'] > best_pf_net:
                best_pf_net = s['overall_pf_net']
                best_idx = i
        if best_idx < 0:
            best_pf_net = -999
            for i, s in enumerate(all_summaries):
                if s['overall_tpd'] >= 1.5 and s['overall_pf_net'] > best_pf_net:
                    best_pf_net = s['overall_pf_net']
                    best_idx = i
        if best_idx < 0:
            best_pf_net = -999
            for i, s in enumerate(all_summaries):
                if s['overall_pf_net'] > best_pf_net:
                    best_pf_net = s['overall_pf_net']
                    best_idx = i

    for i, s in enumerate(all_summaries):
        is_best = (i == best_idx)
        tag = " << BEST" if is_best else ""
        prof_str = f"{s['profitable_regimes']}/{s['regimes_with_trades']}"
        pol_label = f"thr={s['threshold']:.2f}" if s['policy_type'] == 'threshold' else f"top{int(s['policy_value'])}"
        log.info("%-4d %-5.1f %-5.2f %-12s %-3d | %+7.2f | %+7.4f | %5.1f | %+7.2f | %+7.2f | %6.3f | %6.2f | %5s%s",
                 i+1, s['tp_mult'], s['sl_mult'], pol_label, s['cooldown'],
                 s['overall_pf_net'], s['overall_e_net'], s['overall_tpd'],
                 s['overall_win_r_net'], s['overall_loss_r_net'],
                 s['avg_cost_r'], s['avg_size_mult'], prof_str, tag)

    log.info("=" * 130)

    if best_idx >= 0:
        best = all_summaries[best_idx]
        pol_type = best['policy_type']
        pol_val = best['policy_value']
        pol_label = f"threshold:{pol_val}" if pol_type == 'threshold' else f"percentile:top{int(pol_val)}"

        log.info("")
        log.info("=" * 80)
        log.info(f"  << BEST CONFIG ({SYSTEM_VERSION}) >>")
        log.info("=" * 80)
        log.info(f"  Policy:     {pol_label}")
        log.info(f"  Threshold:  {best['threshold']:.4f}")
        log.info(f"  TP mult:    {best['tp_mult']}")
        log.info(f"  SL mult:    {best['sl_mult']}")
        log.info(f"  Cooldown:   {best['cooldown']}")
        log.info(f"  PF_net:     {best['overall_pf_net']:.2f}")
        log.info(f"  E[net]:     {best['overall_e_net']:+.4f}")
        log.info(f"  Trades/day: {best['overall_tpd']:.1f}")
        log.info(f"  WinR_net:   {best['overall_win_r_net']:+.2f}")
        log.info(f"  LossR_net:  {best['overall_loss_r_net']:+.2f}")
        log.info(f"  Avg CostR:  {best['avg_cost_r']:.3f}")
        log.info(f"  Avg SzMul:  {best['avg_size_mult']:.2f}")
        log.info(f"  Profitable: {best['profitable_regimes']}/{best['regimes_with_trades']} regimes")

        passed = best_idx in passing_indices
        criteria_desc = f"PF_net>=1.05, E[net]>0, TPD {tpd_lo:.1f}-{tpd_hi:.1f}, >={min_profitable} regimes profitable"
        if passed:
            log.info(f"  Status:     PASSED ({criteria_desc})")
        else:
            log.info(f"  Status:     BEST AVAILABLE (did not pass all criteria)")

        policy_json = {
            'version': SYSTEM_VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'policy_type': pol_type,
            'policy_value': pol_val,
            'tp_mult': best['tp_mult'],
            'sl_mult': best['sl_mult'],
            'threshold': best['threshold'],
            'cooldown': best['cooldown'],
            'horizon': horizon,
            'r_min_expiry': r_min_expiry,
            'fees_bps_entry': fees_bps_entry,
            'fees_bps_exit': fees_bps_exit,
            'spread_bps': spread_bps,
            'slip_k': slip_k,
            'size_cap': size_cap,
            'overall_pf_net': best['overall_pf_net'],
            'overall_e_net': best['overall_e_net'],
            'overall_tpd': best['overall_tpd'],
            'overall_win_r_net': best['overall_win_r_net'],
            'overall_loss_r_net': best['overall_loss_r_net'],
            'profitable_regimes': best['profitable_regimes'],
            'regimes_with_trades': best['regimes_with_trades'],
            'passed_all_criteria': passed,
            'costs': {
                'fees_bps_entry': fees_bps_entry,
                'fees_bps_exit': fees_bps_exit,
                'spread_bps': spread_bps,
                'slip_k': slip_k,
                'size_cap': size_cap,
            },
            'metrics': {
                'overall_pf_net': best['overall_pf_net'],
                'overall_e_net': best['overall_e_net'],
                'overall_tpd': best['overall_tpd'],
                'overall_win_r_net': best['overall_win_r_net'],
                'overall_loss_r_net': best['overall_loss_r_net'],
                'profitable_regimes': best['profitable_regimes'],
                'regimes_with_trades': best['regimes_with_trades'],
            },
        }

        out_path = Path("checkpoints/best_policy.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(policy_json, f, indent=2)
        log.info(f"  Saved to:   {out_path}")
        log.info("=" * 80)

    log.info("")
    return all_summaries


def _quantile_pinball_loss(preds, targets, taus):
    """Pinball (quantile) loss for multiple quantiles.

    Args:
        preds: [batch, n_quantiles] predicted quantile values
        targets: [batch] realized R values
        taus: list of quantile levels, e.g. [0.1, 0.5, 0.9]
    Returns:
        scalar loss averaged over batch and quantiles
    """
    import torch
    targets_expanded = targets.unsqueeze(-1).expand_as(preds)
    errors = targets_expanded - preds
    tau_tensor = torch.tensor(taus, device=preds.device, dtype=preds.dtype).unsqueeze(0)
    loss = torch.where(errors >= 0, tau_tensor * errors, (tau_tensor - 1.0) * errors)
    return loss.mean()


def _select_trades_by_score(scores, sides, precomputed_outcomes, precomputed_r, top_pct, cooldown=4):
    """Select top X% of CANDIDATE bars by score with cooldown, return precomputed outcomes.

    Ranks only candidate bars (side != 0 and valid outcome) by score,
    selects the top `top_pct` fraction, then applies cooldown.
    """
    import numpy as np

    n = len(scores)
    if n == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    candidate_mask = (sides != 0) & (~np.isnan(precomputed_r.astype(float)))
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    candidate_scores = scores[candidate_indices]
    n_to_select = max(int(len(candidate_indices) * top_pct), 1)
    top_in_candidates = np.argsort(-candidate_scores)[:n_to_select]
    top_indices = candidate_indices[top_in_candidates]
    top_indices = np.sort(top_indices)

    selected_indices = []
    last_trade = -cooldown - 1
    for idx in top_indices:
        if (idx - last_trade) > cooldown:
            selected_indices.append(idx)
            last_trade = idx

    if not selected_indices:
        return np.array([]), np.array([]), np.array([], dtype=int)

    sel = np.array(selected_indices)
    sel_outcomes = precomputed_outcomes[sel]
    sel_r = precomputed_r[sel]

    valid_mask = ~np.isnan(sel_r.astype(float))
    sel_outcomes = sel_outcomes[valid_mask]
    sel_r = sel_r[valid_mask]
    sel = sel[valid_mask]

    return sel_outcomes, sel_r, sel


def _run_distributional_sweep(scores, sides, precomputed_outcomes, precomputed_r,
                               val_bars, epoch, tp_mult, sl_mult,
                               target_tpd=6.5, target_tpd_tol=1.5, min_trades=30,
                               candidate_mask=None, risk_controls=None,
                               symbol_ids=None, horizon_bars=16):
    """Score-based sweep for distributional model: rank by score, evaluate top percentiles.

    If candidate_mask is provided, only candidate bars are considered for ranking.
    If risk_controls is provided, applies daily loss limit, max concurrent, symbol exposure caps.
    """
    import numpy as np
    from data.candidate_generator import apply_risk_controls, RiskControls

    TOP_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    COOLDOWN = 4

    safe_outcomes = np.where(
        np.isin(precomputed_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]),
        precomputed_outcomes,
        "NO_CANDIDATE"
    )
    safe_r = np.where(np.isnan(precomputed_r.astype(float)), 0.0, precomputed_r.astype(float))

    if candidate_mask is not None:
        non_cand = ~candidate_mask
        scores = scores.copy()
        scores[non_cand] = float('-inf')
        n_eligible = int(candidate_mask.sum())
        log.info(f"[SWEEP] Candidate mask applied: {n_eligible}/{len(candidate_mask)} bars eligible")
        if n_eligible == 0:
            log.warning("[SWEEP] WARNING: Zero candidate bars! Sweep will produce empty results. Consider relaxing candidate filters.")

    sweep_results = []

    for pct in TOP_PCTS:
        sel_outcomes, sel_r, sel_idx = _select_trades_by_score(
            scores, sides, safe_outcomes, safe_r, pct, COOLDOWN
        )

        if risk_controls is not None and len(sel_idx) > 0:
            sel_symbols = symbol_ids[sel_idx] if symbol_ids is not None else None
            sel_idx, sel_r = apply_risk_controls(
                sel_idx, sel_r, sel_symbols, risk_controls
            )
            sel_outcomes = safe_outcomes[sel_idx] if len(sel_idx) > 0 else np.array([])

        m = _compute_sweep_metrics(sel_outcomes, sel_r, val_bars)
        m['label'] = f"top{int(pct*100)}%"
        m['pct'] = pct
        sweep_results.append(m)

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    best_freq_score = float('-inf')
    best_freq_label = ""
    best_freq_pct = 0.0
    best_any_score = float('-inf')
    best_any_label = ""
    best_any_pct = 0.0

    for m in sweep_results:
        if m['trades'] < min_trades:
            continue
        if tpd_lo <= m['trades_per_day'] <= tpd_hi:
            if m['expect'] > best_freq_score:
                best_freq_score = m['expect']
                best_freq_label = m['label']
                best_freq_pct = m['pct']
        if m['expect'] > best_any_score:
            best_any_score = m['expect']
            best_any_label = m['label']
            best_any_pct = m['pct']

    if best_freq_label:
        best_label = best_freq_label
        best_score_val = best_freq_score
        best_pct = best_freq_pct
    elif best_any_label:
        best_label = best_any_label
        best_score_val = best_any_score
        best_pct = best_any_pct
    else:
        best_label = ""
        best_score_val = float('-inf')
        best_pct = 0.0

    score_arr = np.array(scores)
    if candidate_mask is not None:
        scores_eligible = score_arr[candidate_mask]
    else:
        scores_eligible = score_arr
    scores_finite = scores_eligible[np.isfinite(scores_eligible)]
    val_days = val_bars / 96.0
    log.info("-" * 120)
    log.info("[SCORE_DIAG] total=%d eligible=%d finite=%d nan_or_inf=%d",
             len(score_arr), len(scores_eligible), len(scores_finite),
             len(scores_eligible) - len(scores_finite))
    if len(scores_finite) > 0:
        sp50 = float(np.percentile(scores_finite, 50))
        sp75 = float(np.percentile(scores_finite, 75))
        sp90 = float(np.percentile(scores_finite, 90))
        sp95 = float(np.percentile(scores_finite, 95))
        sp99 = float(np.percentile(scores_finite, 99))
        log.info("score percentiles (val): p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f", sp50, sp75, sp90, sp95, sp99)
    else:
        log.info("score percentiles: EMPTY (no finite candidate scores)")
    log.info("DISTRIBUTIONAL SWEEP (epoch %d) | cooldown=%d | TP=%.1fx SL=%.1fx ATR | val_days=%.1f | target=%.1f±%.1f tpd",
             epoch, COOLDOWN, tp_mult, sl_mult, val_days, target_tpd, target_tpd_tol)
    log.info("%-8s %5s %8s %6s %6s %5s | %6s %6s %6s | %4s %4s %4s | %5s",
             "Select", "Trds", "Expect", "WR", "Shrpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "T/Day")
    log.info("-" * 120)
    for m in sweep_results:
        in_freq = tpd_lo <= m['trades_per_day'] <= tpd_hi
        marker = ""
        if m['label'] == best_label and m['trades'] >= min_trades and best_score_val > float('-inf'):
            marker = " <<< BEST"
        log.info("%-8s %5d %+8.3f %5.1f%% %6.2f %5.2f | %+6.3f %+6.3f %+6.3f | %3.0f%% %3.0f%% %3.0f%% | %5.1f%s",
                 m['label'], m['trades'], m['expect'], m['winrate']*100, m['sharpe'], m['pf'],
                 m['avg_win_r'], m['avg_loss_r'], m['median_r'],
                 m['pct_tp']*100, m['pct_sl']*100, m['pct_exp']*100, m['trades_per_day'], marker)
    log.info("-" * 120)

    return sweep_results, best_label, best_score_val, best_pct


def train_distributional_model(data_path, device, epochs, batch_size, lr,
                                checkpoint_interval=25, warmup_epochs=5, min_lr=None,
                                tp_mult=2.0, sl_mult=1.5, horizon=16,
                                symbols=None,
                                w_mse=1.0, w_quantile=0.5, w_bce=0.5, w_regime=0.0,
                                score_lambda=0.5, value_clip=3.0,
                                target_tpd=6.5, target_tpd_tol=1.5,
                                use_regime_head=False,
                                q_min_tp=0.3, r_min_expiry_strict=1.0,
                                auto_balance_enter_labels=True,
                                target_enter_rate=0.18,
                                target_enter_rate_min=0.12,
                                target_enter_rate_max=0.25,
                                balance_search_steps=30,
                                w_dir=0.3,
                                candidate_config=None,
                                multi_horizon_config=None,
                                preset_config=None,
                                use_money_score=False,
                                risk_controls=None,
                                use_kelly_sizing=False,
                                symbol_embed_dim=8):
    """v4.9.1 Distributional Trade Forecaster with Candidate Engine + Multi-Horizon + Multi-Preset.

    Replaces binary ENTER classification with distributional outputs:
    - value_head -> E[R] (Huber loss)
    - quantile_head -> q10, q50, q90 of realized R (pinball loss)
    - win_head -> p(R > 0) (BCE loss)
    - optional regime_head -> chop/trend/highvol (CE loss)

    v4.9.1 additions:
    - Candidate engine: filter bars by ATR/vol/breakout/fee gate
    - Multi-horizon: train separate models per horizon, select best
    - Multi-preset: evaluate multiple barrier presets, select best
    - Money-score: p_win*q50 - lambda*downside, with efficiency and regime weight
    - Kelly sizing + risk controls
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from config import config
    from data.candidate_generator import (
        CandidateConfig, generate_candidate_mask, compute_money_score,
        compute_kelly_size, apply_risk_controls, RiskControls,
        BARRIER_PRESETS, MultiHorizonConfig, PresetConfig,
    )

    if candidate_config is None:
        candidate_config = CandidateConfig(enabled=False)
    if risk_controls is None:
        risk_controls = RiskControls()

    use_multi_horizon = multi_horizon_config is not None
    use_multi_preset = preset_config is not None
    horizons = multi_horizon_config.horizons if use_multi_horizon else [horizon]
    presets = []
    if use_multi_preset:
        presets = [preset_config.get_preset_params(p) for p in preset_config.presets]
    else:
        presets = [{'tp_mult': tp_mult, 'sl_mult': sl_mult, 'label': 'default'}]

    log.info("=" * 60)
    log.info("  v4.9.1 DISTRIBUTIONAL TRADE FORECASTER - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {DIST_FEATURE_VERSION}")
    log.info(f"[DIST_CONFIG] w_mse={w_mse} w_quantile={w_quantile} w_bce={w_bce} w_regime={w_regime}")
    log.info(f"[DIST_CONFIG] score_lambda={score_lambda} value_clip={value_clip}")
    log.info(f"[DIST_CONFIG] target_tpd={target_tpd} tpd_tol={target_tpd_tol}")
    log.info(f"[DIST_CONFIG] use_regime_head={use_regime_head}")
    log.info(f"[DIST_CONFIG] quantile_taus=[0.10, 0.50, 0.90]")
    log.info(f"[DIST_CONFIG] candidates={candidate_config.enabled} multi_horizon={use_multi_horizon} multi_preset={use_multi_preset}")
    log.info(f"[DIST_CONFIG] money_score={use_money_score} kelly_sizing={use_kelly_sizing}")
    if use_multi_horizon:
        log.info(f"[DIST_CONFIG] horizons={horizons}")
    if use_multi_preset:
        preset_mode_str = preset_config.mode if preset_config else "fixed:standard"
        log.info(f"[DIST_CONFIG] presets={[p['label'] for p in presets]} mode={preset_mode_str}")

    from data.regression_targets import RegressionTargetGenerator
    reg_gen = RegressionTargetGenerator(horizon_periods=horizon)

    from data.pipeline import FeatureEngineer
    data_dir = Path("data_cache")
    sequence_length = config.data.sequence_length

    QUANTILE_TAUS = [0.10, 0.50, 0.90]

    primary_horizon = horizons[0] if use_multi_horizon else horizon
    primary_tp = presets[0]['tp_mult']
    primary_sl = presets[0]['sl_mult']

    # === DATA LOADING (reuse existing pipeline) ===
    if symbols and len(symbols) > 1:
        log.info(f"[DATA] Multi-asset distributional training: symbols={symbols}")
        all_train_features = []
        all_train_r = []
        all_train_win = []
        all_train_sym_ids = []
        all_train_dir = []
        all_train_dir_conf = []
        all_val_features = []
        all_val_r = []
        all_val_win = []
        all_val_outcomes = []
        all_val_sides = []
        all_val_sym_ids = []
        all_val_dir = []
        all_val_dir_conf = []
        all_val_cand = []
        all_train_cand = []
        feature_columns_ref = None

        for sym_idx, sym in enumerate(symbols):
            sym_data_path = data_dir / f"{sym}_15m.parquet"
            if not sym_data_path.exists():
                log.error(f"[DATA] No data for {sym} at {sym_data_path} — skipping")
                continue

            sym_df = pd.read_parquet(sym_data_path)
            log.info(f"[DATA] {sym} (sym_idx={sym_idx}): {len(sym_df)} candles loaded")

            sym_engineer = FeatureEngineer()
            sym_features_df = sym_engineer.compute_all_features(sym_df)
            sym_features_df = sym_features_df.fillna(0)

            sym_funding_df = fetch_funding_rates(sym_df, data_dir)
            sym_funding_features = compute_funding_features(sym_df, sym_funding_df)
            sym_features_df = pd.concat([sym_features_df, sym_funding_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            sym_oi_df = fetch_open_interest_hist(sym_df, data_dir, symbol=sym)
            sym_oi_enabled = _oi_sanity_check(sym_df, sym_oi_df, symbol=sym)
            if sym_oi_enabled:
                sym_oi_features = compute_oi_features(sym_df, sym_oi_df)
            else:
                sym_oi_features = pd.DataFrame(
                    np.zeros((len(sym_df), OI_FEATURE_COUNT)),
                    columns=OI_FEATURE_NAMES,
                    index=sym_df.index,
                )
            sym_features_df = pd.concat([sym_features_df, sym_oi_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            try:
                sym_ls_df = fetch_ls_ratio_hist(sym_df, data_dir, symbol=sym)
                sym_ls_features = compute_ls_ratio_features(sym_df, sym_ls_df)
            except Exception as e:
                log.warning(f"[LS_RATIO] Failed for {sym}: {e} — using zeros")
                sym_ls_features = pd.DataFrame(
                    np.zeros((len(sym_df), LS_RATIO_FEATURE_COUNT)),
                    columns=LS_RATIO_FEATURE_NAMES,
                    index=sym_df.index,
                )
            sym_features_df = pd.concat([sym_features_df, sym_ls_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            if feature_columns_ref is None:
                feature_columns_ref = list(sym_features_df.columns)

            htf_cols = [c for c in sym_features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
            sym_htf_df = sym_features_df[htf_cols].copy()

            sym_cand_mask = None
            if candidate_config.enabled:
                sym_cand_mask, sym_cand_diag = generate_candidate_mask(
                    sym_df, candidate_config, symbol=sym
                )

            if use_multi_preset and len(presets) > 1:
                from data.regression_targets import generate_multi_preset_targets
                sym_label_df = generate_multi_preset_targets(
                    sym_df, sym_htf_df,
                    presets=presets,
                    horizon_periods=primary_horizon,
                    q_min_tp=q_min_tp,
                    r_min_expiry_strict=r_min_expiry_strict,
                    soft_label_temp=1.0,
                    auto_balance=auto_balance_enter_labels,
                    target_enter_rate=target_enter_rate,
                    target_enter_rate_min=target_enter_rate_min,
                    target_enter_rate_max=target_enter_rate_max,
                    balance_search_steps=balance_search_steps,
                )
                sym_preset_targets = {}
                for pi, preset in enumerate(presets):
                    plabel = preset['label']
                    sym_preset_targets[plabel] = {
                        'realized_r': sym_label_df[f'realized_r_{plabel}'].values.astype(np.float32),
                        'outcome': sym_label_df[f'outcome_{plabel}'].values,
                        'side_hint': sym_label_df[f'side_hint_{plabel}'].values.astype(np.int64),
                    }

                if preset_config and preset_config.is_oracle:
                    sym_best_r = np.full(len(sym_df), np.nan, dtype=np.float32)
                    sym_best_side = np.zeros(len(sym_df), dtype=np.int64)
                    sym_best_outcome = np.full(len(sym_df), "NO_CANDIDATE", dtype=object)
                    for pi, preset in enumerate(presets):
                        plabel = preset['label']
                        pr = sym_label_df[f'realized_r_{plabel}'].values.astype(np.float32)
                        for j in range(len(sym_df)):
                            if not np.isnan(pr[j]) and (np.isnan(sym_best_r[j]) or pr[j] > sym_best_r[j]):
                                sym_best_r[j] = pr[j]
                                sym_best_outcome[j] = sym_label_df[f'outcome_{plabel}'].values[j]
                                sym_best_side[j] = sym_label_df[f'side_hint_{plabel}'].values[j]
                    sym_best_r = np.nan_to_num(sym_best_r, nan=0.0)
                    sym_realized_r = sym_best_r
                elif preset_config and preset_config.is_fixed:
                    fixed_name = preset_config.fixed_preset_name
                    if fixed_name not in sym_preset_targets:
                        fixed_name = 'standard'
                    sym_realized_r = np.nan_to_num(sym_preset_targets[fixed_name]['realized_r'], nan=0.0)
                    sym_best_outcome = sym_preset_targets[fixed_name]['outcome']
                    sym_best_side = sym_preset_targets[fixed_name]['side_hint']
                elif preset_config and preset_config.is_learnable:
                    fixed_name = 'standard'
                    if fixed_name not in sym_preset_targets:
                        fixed_name = list(sym_preset_targets.keys())[0]
                    sym_realized_r = np.nan_to_num(sym_preset_targets[fixed_name]['realized_r'], nan=0.0)
                    sym_best_outcome = sym_preset_targets[fixed_name]['outcome']
                    sym_best_side = sym_preset_targets[fixed_name]['side_hint']
                else:
                    fixed_name = 'standard'
                    sym_realized_r = np.nan_to_num(sym_preset_targets[fixed_name]['realized_r'], nan=0.0)
                    sym_best_outcome = sym_preset_targets[fixed_name]['outcome']
                    sym_best_side = sym_preset_targets[fixed_name]['side_hint']
                sym_win = (sym_realized_r > 0).astype(np.float32)
                sym_side = sym_best_side.astype(np.int64)
                sym_outcomes = sym_best_outcome
                sym_dir_target = sym_label_df['y_dir'].values.astype(np.float32)
                sym_dir_conf = sym_label_df['y_dir_conf'].values.astype(np.float32)
            else:
                from data.regression_targets import generate_v47_quality_targets
                sym_label_df = generate_v47_quality_targets(
                    sym_df, sym_htf_df,
                    horizon_periods=primary_horizon,
                    tp_atr_mult=primary_tp, sl_atr_mult=primary_sl,
                    q_min_tp=q_min_tp,
                    r_min_expiry_strict=r_min_expiry_strict,
                    soft_label_temp=1.0,
                    auto_balance=auto_balance_enter_labels,
                    target_enter_rate=target_enter_rate,
                    target_enter_rate_min=target_enter_rate_min,
                    target_enter_rate_max=target_enter_rate_max,
                    balance_search_steps=balance_search_steps,
                )
                sym_realized_r = sym_label_df['realized_r'].values.astype(np.float32)
                sym_win = (sym_realized_r > 0).astype(np.float32)
                sym_side = sym_label_df['side_hint'].values.astype(np.int64)
                sym_outcomes = sym_label_df['outcome'].values
                sym_dir_target = sym_label_df['y_dir'].values.astype(np.float32)
                sym_dir_conf = sym_label_df['y_dir_conf'].values.astype(np.float32)

            valid_start = sequence_length
            sym_feat_np = sym_features_df.values[valid_start:].astype(np.float32)
            sym_r_np = sym_realized_r[valid_start:]
            sym_win_np = sym_win[valid_start:]
            sym_side_np = sym_side[valid_start:]
            sym_outcomes_np = sym_outcomes[valid_start:]
            sym_dir_np = sym_dir_target[valid_start:]
            sym_dir_conf_np = sym_dir_conf[valid_start:]
            sym_cand_np = sym_cand_mask[valid_start:] if sym_cand_mask is not None else None

            n_sym = len(sym_feat_np)
            train_end_sym = int(n_sym * 0.70)
            val_end_sym = int(n_sym * 0.85)

            log.info(f"[SPLIT] sym={sym} total={n_sym} train={train_end_sym} val={val_end_sym - train_end_sym}")

            all_train_features.append(sym_feat_np[:train_end_sym])
            all_train_r.append(sym_r_np[:train_end_sym])
            all_train_win.append(sym_win_np[:train_end_sym])
            all_train_sym_ids.append(np.full(train_end_sym, sym_idx, dtype=np.int64))
            all_train_dir.append(sym_dir_np[:train_end_sym])
            all_train_dir_conf.append(sym_dir_conf_np[:train_end_sym])

            val_size = val_end_sym - train_end_sym
            all_val_features.append(sym_feat_np[train_end_sym:val_end_sym])
            all_val_r.append(sym_r_np[train_end_sym:val_end_sym])
            all_val_win.append(sym_win_np[train_end_sym:val_end_sym])
            all_val_outcomes.append(sym_outcomes_np[train_end_sym:val_end_sym])
            all_val_sides.append(sym_side_np[train_end_sym:val_end_sym])
            all_val_sym_ids.append(np.full(val_size, sym_idx, dtype=np.int64))
            all_val_dir.append(sym_dir_np[train_end_sym:val_end_sym])
            all_val_dir_conf.append(sym_dir_conf_np[train_end_sym:val_end_sym])

            if sym_cand_np is not None:
                all_train_cand.append(sym_cand_np[:train_end_sym])
                all_val_cand.append(sym_cand_np[train_end_sym:val_end_sym])

        train_features_raw = np.concatenate(all_train_features, axis=0)
        train_r = np.concatenate(all_train_r, axis=0)
        train_win = np.concatenate(all_train_win, axis=0)
        train_sym_ids = np.concatenate(all_train_sym_ids, axis=0)
        train_dir = np.concatenate(all_train_dir, axis=0)
        train_dir_conf = np.concatenate(all_train_dir_conf, axis=0)

        val_features_raw = np.concatenate(all_val_features, axis=0)
        val_r = np.concatenate(all_val_r, axis=0)
        val_win = np.concatenate(all_val_win, axis=0)
        val_outcomes = np.concatenate(all_val_outcomes, axis=0)
        val_sides = np.concatenate(all_val_sides, axis=0)
        val_sym_ids = np.concatenate(all_val_sym_ids, axis=0)
        val_dir = np.concatenate(all_val_dir, axis=0)
        val_dir_conf = np.concatenate(all_val_dir_conf, axis=0)

        cand_np = np.concatenate(all_val_cand, axis=0) if all_val_cand else None

        n_symbols = len(symbols)
        features_columns_list = feature_columns_ref
        input_dim = train_features_raw.shape[1]
        val_bars = len(val_r)

        engineer = FeatureEngineer()
        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_columns_list)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_columns_list)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_dist(features, r_vals, win_vals, sym_ids, dir_t, dir_c, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return features[valid], r_vals[valid], win_vals[valid], sym_ids[valid], dir_t[valid], dir_c[valid]

        train_scaled, train_r, train_win, train_sym_ids, train_dir, train_dir_conf = clean_dist(
            train_scaled, train_r, train_win, train_sym_ids, train_dir, train_dir_conf, "Train")

        val_r_raw = val_r.copy()
        val_outcomes_raw = val_outcomes.copy()
        val_sides_raw = val_sides.copy()

        val_clean_mask = ~np.isnan(np.where(np.isinf(val_scaled), np.nan, val_scaled)).any(axis=1)
        val_scaled = val_scaled[val_clean_mask]
        val_r = val_r[val_clean_mask]
        val_win = val_win[val_clean_mask]
        val_sym_ids = val_sym_ids[val_clean_mask]
        val_dir = val_dir[val_clean_mask]
        val_dir_conf = val_dir_conf[val_clean_mask]
        val_outcomes = val_outcomes[val_clean_mask]
        val_sides = val_sides[val_clean_mask]
        if cand_np is not None:
            cand_np = cand_np[val_clean_mask]

        features_df_columns = features_columns_list

    else:
        n_symbols = 1
        df = pd.read_parquet(data_path)
        log.info(f"Loaded {len(df)} candles")

        engineer = FeatureEngineer()
        features_df = engineer.compute_all_features(df)
        features_df = features_df.fillna(0)

        funding_df = fetch_funding_rates(df, data_dir)
        funding_features = compute_funding_features(df, funding_df)
        features_df = pd.concat([features_df, funding_features], axis=1)
        features_df = features_df.fillna(0)

        oi_df = fetch_open_interest_hist(df, data_dir)
        oi_enabled = _oi_sanity_check(df, oi_df)
        if oi_enabled:
            oi_features = compute_oi_features(df, oi_df)
        else:
            oi_features = pd.DataFrame(
                np.zeros((len(df), OI_FEATURE_COUNT)),
                columns=OI_FEATURE_NAMES,
                index=df.index,
            )
        features_df = pd.concat([features_df, oi_features], axis=1)
        features_df = features_df.fillna(0)

        htf_cols = [c for c in features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
        htf_features_df = features_df[htf_cols].copy()

        cand_mask_full = None
        if candidate_config.enabled:
            cand_mask_full, cand_diag = generate_candidate_mask(
                df, candidate_config, symbol=symbols[0] if symbols else "BTCUSDT"
            )

        if use_multi_preset and len(presets) > 1:
            from data.regression_targets import generate_multi_preset_targets
            preset_mode = preset_config.mode if preset_config else "fixed:standard"
            label_df = generate_multi_preset_targets(
                df, htf_features_df,
                presets=presets,
                horizon_periods=primary_horizon,
                q_min_tp=q_min_tp,
                r_min_expiry_strict=r_min_expiry_strict,
                soft_label_temp=1.0,
                auto_balance=auto_balance_enter_labels,
                target_enter_rate=target_enter_rate,
                target_enter_rate_min=target_enter_rate_min,
                target_enter_rate_max=target_enter_rate_max,
                balance_search_steps=balance_search_steps,
            )

            preset_targets = {}
            for pi, preset in enumerate(presets):
                plabel = preset['label']
                preset_targets[plabel] = {
                    'realized_r': label_df[f'realized_r_{plabel}'].values.astype(np.float32),
                    'outcome': label_df[f'outcome_{plabel}'].values,
                    'side_hint': label_df[f'side_hint_{plabel}'].values.astype(np.int64),
                }

            if preset_config and preset_config.is_oracle:
                best_r = np.full(len(df), np.nan, dtype=np.float32)
                best_preset_idx = np.zeros(len(df), dtype=np.int64)
                best_outcome = np.full(len(df), "NO_CANDIDATE", dtype=object)
                best_side = np.zeros(len(df), dtype=np.int64)
                for pi, preset in enumerate(presets):
                    plabel = preset['label']
                    pr = label_df[f'realized_r_{plabel}'].values.astype(np.float32)
                    for j in range(len(df)):
                        if not np.isnan(pr[j]) and (np.isnan(best_r[j]) or pr[j] > best_r[j]):
                            best_r[j] = pr[j]
                            best_preset_idx[j] = pi
                            best_outcome[j] = label_df[f'outcome_{plabel}'].values[j]
                            best_side[j] = label_df[f'side_hint_{plabel}'].values[j]
                best_r = np.nan_to_num(best_r, nan=0.0)
                realized_r = best_r
                log.info(f"[MULTI_PRESET] mode=oracle | {len(presets)} presets | WARNING: hindsight leakage, research only")
            elif preset_config and preset_config.is_fixed:
                fixed_name = preset_config.fixed_preset_name
                if fixed_name not in preset_targets:
                    log.warning(f"[MULTI_PRESET] fixed preset '{fixed_name}' not found, falling back to 'standard'")
                    fixed_name = 'standard'
                realized_r = np.nan_to_num(preset_targets[fixed_name]['realized_r'], nan=0.0)
                best_outcome = preset_targets[fixed_name]['outcome']
                best_side = preset_targets[fixed_name]['side_hint']
                best_preset_idx = np.full(len(df), list(preset_targets.keys()).index(fixed_name), dtype=np.int64)
                log.info(f"[MULTI_PRESET] mode=fixed:{fixed_name} | using single preset for labels (no leakage)")
            elif preset_config and preset_config.is_learnable:
                log.warning("[MULTI_PRESET] mode=learnable is not yet implemented (preset_head not wired). "
                            "Falling back to fixed:standard")
                fixed_name = 'standard'
                if fixed_name not in preset_targets:
                    fixed_name = list(preset_targets.keys())[0]
                realized_r = np.nan_to_num(preset_targets[fixed_name]['realized_r'], nan=0.0)
                best_outcome = preset_targets[fixed_name]['outcome']
                best_side = preset_targets[fixed_name]['side_hint']
                best_preset_idx = np.full(len(df), list(preset_targets.keys()).index(fixed_name), dtype=np.int64)
            else:
                fixed_name = 'standard'
                realized_r = np.nan_to_num(preset_targets[fixed_name]['realized_r'], nan=0.0)
                best_outcome = preset_targets[fixed_name]['outcome']
                best_side = preset_targets[fixed_name]['side_hint']
                best_preset_idx = np.zeros(len(df), dtype=np.int64)
                log.info(f"[MULTI_PRESET] mode=fixed:standard (default fallback)")

            win_labels = (realized_r > 0).astype(np.float32)
            side_hints = best_side.astype(np.int64) if hasattr(best_side, 'astype') else np.array(best_side, dtype=np.int64)
            precomputed_outcomes = best_outcome
            dir_target = label_df['y_dir'].values.astype(np.float32)
            dir_conf = label_df['y_dir_conf'].values.astype(np.float32)
        else:
            from data.regression_targets import generate_v47_quality_targets
            label_df = generate_v47_quality_targets(
                df, htf_features_df,
                horizon_periods=primary_horizon,
                tp_atr_mult=primary_tp, sl_atr_mult=primary_sl,
                q_min_tp=q_min_tp,
                r_min_expiry_strict=r_min_expiry_strict,
                soft_label_temp=1.0,
                auto_balance=auto_balance_enter_labels,
                target_enter_rate=target_enter_rate,
                target_enter_rate_min=target_enter_rate_min,
                target_enter_rate_max=target_enter_rate_max,
                balance_search_steps=balance_search_steps,
            )
            realized_r = label_df['realized_r'].values.astype(np.float32)
            win_labels = (realized_r > 0).astype(np.float32)
            side_hints = label_df['side_hint'].values.astype(np.int64)
            precomputed_outcomes = label_df['outcome'].values
            dir_target = label_df['y_dir'].values.astype(np.float32)
            dir_conf = label_df['y_dir_conf'].values.astype(np.float32)
            preset_targets = None

        valid_start = sequence_length
        features_np = features_df.values[valid_start:].astype(np.float32)
        r_np = realized_r[valid_start:]
        win_np = win_labels[valid_start:]
        side_np = side_hints[valid_start:]
        outcomes_np = precomputed_outcomes[valid_start:]
        dir_np = dir_target[valid_start:]
        dir_conf_np = dir_conf[valid_start:]
        cand_np = cand_mask_full[valid_start:] if cand_mask_full is not None else None

        n_total = len(features_np)
        max_horizon = max(horizons) if use_multi_horizon else horizon
        purge_gap = max_horizon + sequence_length
        val_samples = max(int(n_total * 0.1), purge_gap)
        train_samples = n_total - purge_gap - val_samples

        if train_samples < sequence_length * 3:
            log.error(f"Not enough data for training: {train_samples} samples")
            sys.exit(1)

        train_end = train_samples
        val_start_idx = train_end + purge_gap
        val_end = val_start_idx + val_samples

        log.info(f"Data split: train={train_samples}, purge={purge_gap}, val={val_samples}")

        train_features_raw = features_np[:train_end]
        train_r = r_np[:train_end]
        train_win = win_np[:train_end]
        train_dir = dir_np[:train_end]
        train_dir_conf = dir_conf_np[:train_end]

        val_features_raw = features_np[val_start_idx:val_end]
        val_r = r_np[val_start_idx:val_end]
        val_win = win_np[val_start_idx:val_end]
        val_outcomes = outcomes_np[val_start_idx:val_end]
        val_sides = side_np[val_start_idx:val_end]
        val_dir = dir_np[val_start_idx:val_end]
        val_dir_conf = dir_conf_np[val_start_idx:val_end]
        val_cand = cand_np[val_start_idx:val_end] if cand_np is not None else None
        val_bars = val_samples

        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_df.columns)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_df.columns)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_dist_single(features, r_vals, win_vals, dir_t, dir_c, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return features[valid], r_vals[valid], win_vals[valid], dir_t[valid], dir_c[valid]

        train_scaled, train_r, train_win, train_dir, train_dir_conf = clean_dist_single(
            train_scaled, train_r, train_win, train_dir, train_dir_conf, "Train")

        val_clean_mask = ~np.isnan(np.where(np.isinf(val_scaled), np.nan, val_scaled)).any(axis=1)
        val_scaled = val_scaled[val_clean_mask]
        val_r = val_r[val_clean_mask]
        val_win = val_win[val_clean_mask]
        val_outcomes = val_outcomes[val_clean_mask]
        val_sides = val_sides[val_clean_mask]
        val_dir = val_dir[val_clean_mask]
        val_dir_conf = val_dir_conf[val_clean_mask]
        if val_cand is not None:
            val_cand = val_cand[val_clean_mask]
        cand_np = val_cand

        train_sym_ids = np.zeros(len(train_r), dtype=np.int64)
        val_sym_ids = np.zeros(len(val_r), dtype=np.int64)
        features_df_columns = list(features_df.columns)
        input_dim = features_np.shape[1]

    # === TARGET PREPARATION ===
    train_value_targets = np.clip(np.nan_to_num(train_r, nan=0.0), -value_clip, value_clip).astype(np.float32)
    val_value_targets = np.clip(np.nan_to_num(val_r, nan=0.0), -value_clip, value_clip).astype(np.float32)
    train_quantile_targets = np.clip(np.nan_to_num(train_r, nan=0.0), -value_clip, value_clip).astype(np.float32)
    val_quantile_targets = np.clip(np.nan_to_num(val_r, nan=0.0), -value_clip, value_clip).astype(np.float32)
    train_win_targets = train_win.astype(np.float32)
    val_win_targets = val_win.astype(np.float32)

    win_rate = float(train_win_targets.mean())
    r_mean = float(train_value_targets.mean())
    r_std = float(train_value_targets.std())
    log.info(f"[DIST_DATA] train_samples={len(train_value_targets)} val_samples={len(val_value_targets)}")
    log.info(f"[DIST_DATA] win_rate={win_rate:.3f} R_mean={r_mean:.4f} R_std={r_std:.4f}")
    log.info(f"[DIST_DATA] R quantiles: q10={np.percentile(train_value_targets, 10):.4f} "
             f"q50={np.percentile(train_value_targets, 50):.4f} q90={np.percentile(train_value_targets, 90):.4f}")

    # === DATASET ===
    class DistDataset(Dataset):
        def __init__(self, features, r_targets, q_targets, win_targets, symbol_ids, dir_targets, dir_conf_targets, seq_len):
            self.features = features.astype(np.float32)
            self.r_targets = r_targets.astype(np.float32)
            self.q_targets = q_targets.astype(np.float32)
            self.win_targets = win_targets.astype(np.float32)
            self.symbol_ids = symbol_ids.astype(np.int64)
            self.dir_targets = dir_targets.astype(np.float32)
            self.dir_conf_targets = dir_conf_targets.astype(np.float32)
            self.seq_len = seq_len
            self.valid_indices = list(range(seq_len, len(features)))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            actual_idx = self.valid_indices[idx]
            start = actual_idx - self.seq_len
            seq = self.features[start:actual_idx]
            return (
                torch.from_numpy(seq),
                torch.tensor(self.r_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.q_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.win_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.symbol_ids[actual_idx], dtype=torch.long),
                torch.tensor(self.dir_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.dir_conf_targets[actual_idx], dtype=torch.float32),
            )

    train_dataset = DistDataset(train_scaled, train_value_targets, train_quantile_targets, train_win_targets,
                                train_sym_ids, train_dir, train_dir_conf, sequence_length)
    val_dataset = DistDataset(val_scaled, val_value_targets, val_quantile_targets, val_win_targets,
                              val_sym_ids, val_dir, val_dir_conf, sequence_length)

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # === MODEL ===
    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        enable_enter_head=False,
        enable_quantile_head=False,
        enable_vol_state_head=False,
        enable_mu_head=False,
        enable_sigma_head=False,
        enable_value_head=True,
        enable_edge_head=False,
        enable_dir_head=True,
        enable_htf_head=False,
        enable_win_head=True,
        enable_dist_quantile_head=True,
        enable_regime_head=use_regime_head,
        n_symbols=n_symbols,
        symbol_embed_dim=symbol_embed_dim if n_symbols > 1 else 0,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "DistributionalForecaster"
    model.to(device)
    log.info(f"Model: DistributionalForecaster ({model.parameters_count():,} parameters)")
    log.info(f"Active heads: value_head (E[R]) + dist_quantile_head (q10/q50/q90) + win_head (p_win) + dir_head")
    if use_regime_head:
        log.info(f"  + regime_head (chop/trend/highvol)")

    with torch.no_grad():
        win_bias = float(np.log(max(win_rate, 0.01) / max(1.0 - win_rate, 0.01)))
        model.win_head[-1].bias.fill_(win_bias)
        log.info(f"[BIAS_INIT] win_head bias={win_bias:.4f} (win_rate={win_rate:.3f})")

        model.value_head[-1].bias.fill_(r_mean)
        log.info(f"[BIAS_INIT] value_head bias={r_mean:.4f} (R_mean)")

        q10_init = float(np.percentile(train_value_targets, 10))
        q50_init = float(np.percentile(train_value_targets, 50))
        q90_init = float(np.percentile(train_value_targets, 90))
        model.dist_quantile_head[-1].bias.data.copy_(
            torch.tensor([q10_init, q50_init, q90_init])
        )
        log.info(f"[BIAS_INIT] dist_quantile_head bias=[{q10_init:.4f}, {q50_init:.4f}, {q90_init:.4f}]")

    # === LOSS FUNCTIONS ===
    value_criterion = nn.HuberLoss(delta=1.0)
    win_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    effective_min_lr = min_lr if min_lr is not None else lr * 0.05
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=effective_min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    for pg in optimizer.param_groups:
        pg['lr'] = lr * 1e-3

    log.info(f"Training for {epochs} epochs (lr={lr}, batch={batch_size})")
    log.info(f"Loss: {w_mse}*Huber(E_R) + {w_quantile}*Pinball(q10/q50/q90) + {w_bce}*BCE(p_win) + {w_dir}*BCE(dir)")
    log.info(f"Score: sigmoid(win) * E_R - {score_lambda} * max(0, -q10)")

    best_val_loss = float('inf')
    best_expectancy = float('-inf')
    best_expectancy_pct = 0.0
    patience = 0
    max_patience = 50
    min_epochs = 40
    history = {'train_loss': [], 'val_loss': [], 'val_value_mae': [], 'val_win_auc': [],
               'val_expectancy': [], 'val_best_tpd': []}

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"[DIST_EPOCH] epoch={epoch+1}/{epochs} lr={current_lr:.2e}")

        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            feat_batch, r_batch, q_batch, win_batch, sym_batch, dir_batch, dir_conf_batch = batch
            feat_batch = feat_batch.to(device)
            r_batch = r_batch.to(device)
            q_batch = q_batch.to(device)
            win_batch = win_batch.to(device)
            sym_batch = sym_batch.to(device)
            dir_batch = dir_batch.to(device)
            dir_conf_batch = dir_conf_batch.to(device)

            optimizer.zero_grad()
            output = model.forward_multihead(feat_batch, symbol_ids=sym_batch if n_symbols > 1 else None)

            value_pred = output.value_logits.squeeze(-1)
            value_loss = value_criterion(value_pred, r_batch)

            quantile_pred = output.dist_quantiles
            quantile_loss = _quantile_pinball_loss(quantile_pred, q_batch, QUANTILE_TAUS)

            win_pred = output.win_logits.squeeze(-1)
            win_loss = win_criterion(win_pred, win_batch)

            loss = w_mse * value_loss + w_quantile * quantile_loss + w_bce * win_loss

            if output.dir_logits is not None:
                dir_pred = output.dir_logits.squeeze(-1)
                dir_loss = nn.functional.binary_cross_entropy_with_logits(
                    dir_pred, dir_batch, weight=dir_conf_batch, reduction='mean'
                )
                loss = loss + w_dir * dir_loss

            if use_regime_head and output.regime_logits is not None:
                pass

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        # === VALIDATION ===
        model.eval()
        val_loss_total = 0
        val_n = 0
        all_value_preds = []
        all_value_tgts = []
        all_q_preds = []
        all_win_logits = []
        all_win_tgts = []
        all_dir_probs = []
        all_dir_tgts = []

        with torch.no_grad():
            for batch in val_loader:
                feat_batch, r_batch, q_batch, win_batch, sym_batch, dir_batch, dir_conf_batch = batch
                feat_batch = feat_batch.to(device)
                r_batch = r_batch.to(device)
                q_batch = q_batch.to(device)
                win_batch = win_batch.to(device)
                sym_batch = sym_batch.to(device)
                dir_batch = dir_batch.to(device)
                dir_conf_batch = dir_conf_batch.to(device)

                output = model.forward_multihead(feat_batch, symbol_ids=sym_batch if n_symbols > 1 else None)

                vp = output.value_logits.squeeze(-1)
                v_loss = value_criterion(vp, r_batch)
                qp = output.dist_quantiles
                q_loss = _quantile_pinball_loss(qp, q_batch, QUANTILE_TAUS)
                wp = output.win_logits.squeeze(-1)
                w_loss = win_criterion(wp, win_batch)

                batch_loss = w_mse * v_loss + w_quantile * q_loss + w_bce * w_loss

                if output.dir_logits is not None:
                    dp = output.dir_logits.squeeze(-1)
                    d_loss = nn.functional.binary_cross_entropy_with_logits(
                        dp, dir_batch, weight=dir_conf_batch, reduction='mean'
                    )
                    batch_loss = batch_loss + w_dir * d_loss
                    all_dir_probs.extend(torch.sigmoid(dp).cpu().numpy())
                    all_dir_tgts.extend(dir_batch.cpu().numpy())

                val_loss_total += batch_loss.item()
                val_n += 1

                all_value_preds.extend(vp.cpu().numpy())
                all_value_tgts.extend(r_batch.cpu().numpy())
                all_q_preds.extend(qp.cpu().numpy())
                all_win_logits.extend(wp.cpu().numpy())
                all_win_tgts.extend(win_batch.cpu().numpy())

        avg_val_loss = val_loss_total / max(val_n, 1)

        all_vp = np.array(all_value_preds)
        all_vt = np.array(all_value_tgts)
        value_mae = float(np.mean(np.abs(all_vp - all_vt)))
        value_rmse = float(np.sqrt(np.mean((all_vp - all_vt)**2)))

        all_qp = np.array(all_q_preds)
        q10_preds = all_qp[:, 0]
        q50_preds = all_qp[:, 1]
        q90_preds = all_qp[:, 2]
        q10_coverage = float((all_vt >= q10_preds).mean())
        q50_coverage = float((all_vt >= q50_preds).mean())
        q90_coverage = float((all_vt >= q90_preds).mean())

        all_wl = np.array(all_win_logits)
        all_wt = np.array(all_win_tgts)
        win_probs = 1.0 / (1.0 + np.exp(-all_wl))
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(all_wt)) > 1:
                win_auc = float(roc_auc_score(all_wt, win_probs))
            else:
                win_auc = 0.5
        except Exception:
            win_auc = 0.5

        # === COMPUTE SCORE ===
        if use_money_score:
            scores = compute_money_score(
                win_probs, all_vp, q10_preds, q50_preds, q90_preds,
                horizon_bars=primary_horizon if not use_multi_horizon else horizons[0],
                score_lambda=score_lambda,
                use_efficiency=True,
            )
        else:
            scores = win_probs * all_vp - score_lambda * np.maximum(0, -q10_preds)

        dir_auc = 0.0
        if all_dir_probs:
            all_dp = np.array(all_dir_probs)
            all_dt = np.array(all_dir_tgts)
            try:
                if len(np.unique((all_dt >= 0.5).astype(int))) > 1:
                    dir_auc = float(roc_auc_score((all_dt >= 0.5).astype(int), all_dp))
            except Exception:
                pass

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_value_mae'].append(value_mae)
        history['val_win_auc'].append(win_auc)

        log.info(
            f"Epoch {epoch+1}/{epochs} | Loss T:{avg_train_loss:.4f} V:{avg_val_loss:.4f} | "
            f"E[R]_MAE:{value_mae:.4f} E[R]_RMSE:{value_rmse:.4f} | "
            f"WinAUC:{win_auc:.3f} | DirAUC:{dir_auc:.3f}"
        )
        log.info(
            f"[QUANTILE_COV] q10={q10_coverage:.3f}(target=0.90) q50={q50_coverage:.3f}(target=0.50) "
            f"q90={q90_coverage:.3f}(target=0.10)"
        )
        log.info(
            f"[SCORE] mean={scores.mean():.4f} std={scores.std():.4f} "
            f"p50={np.percentile(scores, 50):.4f} p90={np.percentile(scores, 90):.4f} "
            f"p99={np.percentile(scores, 99):.4f}"
        )

        # === DISTRIBUTIONAL SWEEP ===
        SWEEP_INTERVAL = 5
        sweep_expect = 0.0
        sweep_pct = 0.0
        if (epoch + 1) % SWEEP_INTERVAL == 0 or epoch == 0:
            sweep_start = sequence_length
            sweep_scores = scores[sweep_start:]
            sweep_sides = val_sides[sweep_start:]
            sweep_outcomes = val_outcomes[sweep_start:]
            sweep_r = val_r[sweep_start:]

            n_sweep = min(len(sweep_scores), len(sweep_outcomes))

            sweep_cand_mask = None
            if cand_np is not None:
                sweep_cand_mask = cand_np[sweep_start:sweep_start + n_sweep] if len(cand_np) > sweep_start else None

            has_custom_risk = (risk_controls.daily_loss_limit_r != -3.0 or
                               risk_controls.max_concurrent_trades != 6 or
                               risk_controls.max_symbol_exposure != 3)

            sweep_sym_ids = None
            if has_custom_risk:
                sweep_sym_ids = val_sym_ids[sweep_start:sweep_start + n_sweep] if len(val_sym_ids) > sweep_start else None

            sweep_results, best_label, best_score_val, best_pct_val = _run_distributional_sweep(
                sweep_scores[:n_sweep], sweep_sides[:n_sweep],
                sweep_outcomes[:n_sweep], sweep_r[:n_sweep],
                val_bars, epoch + 1, primary_tp, primary_sl,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
                candidate_mask=sweep_cand_mask,
                risk_controls=risk_controls if has_custom_risk else None,
                symbol_ids=sweep_sym_ids,
                horizon_bars=primary_horizon,
            )
            sweep_expect = best_score_val if best_score_val > float('-inf') else 0.0
            sweep_pct = best_pct_val

        history['val_expectancy'].append(sweep_expect)
        history['val_best_tpd'].append(sweep_pct)

        # === CHECKPOINT SAVING ===
        ckpt_model_config = {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128, 64],
            'num_classes': 3,
            'dropout': 0.3,
            'use_layer_norm': True,
            'use_residual': True,
            'enable_enter_head': False,
            'enable_quantile_head': False,
            'enable_vol_state_head': False,
            'enable_mu_head': False,
            'enable_sigma_head': False,
            'enable_value_head': True,
            'enable_edge_head': False,
            'enable_dir_head': True,
            'enable_htf_head': False,
            'enable_win_head': True,
            'enable_dist_quantile_head': True,
            'enable_regime_head': use_regime_head,
            'n_symbols': n_symbols,
            'symbol_embed_dim': symbol_embed_dim if n_symbols > 1 else 0,
        }
        ckpt_train_config = {
            'model_type': 'distributional_trade_forecaster',
            'version': 'v4.9.1',
            'w_mse': w_mse,
            'w_quantile': w_quantile,
            'w_bce': w_bce,
            'w_regime': w_regime,
            'w_dir': w_dir,
            'score_lambda': score_lambda,
            'value_clip': value_clip,
            'quantile_taus': QUANTILE_TAUS,
            'target_tpd': target_tpd,
            'target_tpd_tol': target_tpd_tol,
            'v491_config': {
                'candidate_engine': candidate_config.enabled,
                'candidate_min_atr_pct': candidate_config.min_atr_pct,
                'multi_horizon': use_multi_horizon,
                'horizons': horizons,
                'multi_preset': use_multi_preset,
                'multi_preset_mode': preset_config.mode if preset_config else 'fixed:standard',
                'presets': [p['label'] for p in presets],
                'money_score': use_money_score,
                'kelly_sizing': use_kelly_sizing,
                'daily_loss_limit_r': risk_controls.daily_loss_limit_r,
                'max_concurrent_trades': risk_controls.max_concurrent_trades,
                'max_symbol_exposure': risk_controls.max_symbol_exposure,
            },
        }

        if sweep_expect > best_expectancy:
            best_expectancy = sweep_expect
            best_expectancy_pct = sweep_pct
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': DIST_FEATURE_VERSION,
                'model_type': 'distributional_trade_forecaster',
                'barrier_config': {
                    'tp_mult': primary_tp, 'sl_mult': primary_sl,
                    'horizon': primary_horizon,
                    'presets': [p['label'] for p in presets] if use_multi_preset else ['default'],
                    'horizons': horizons if use_multi_horizon else [primary_horizon],
                },
                'best_expectancy': best_expectancy,
                'best_expectancy_pct': best_expectancy_pct,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_dist_expectancy.pt")
            log.info(f"[CKPT] New best expectancy={best_expectancy:.4f} at top{int(best_expectancy_pct*100)}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': DIST_FEATURE_VERSION,
                'model_type': 'distributional_trade_forecaster',
                'barrier_config': {
                    'tp_mult': primary_tp, 'sl_mult': primary_sl,
                    'horizon': primary_horizon,
                    'presets': [p['label'] for p in presets] if use_multi_preset else ['default'],
                    'horizons': horizons if use_multi_horizon else [primary_horizon],
                },
                'best_val_loss': best_val_loss,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_dist_loss.pt")
        else:
            patience += 1

        if epoch + 1 >= min_epochs and patience >= max_patience:
            log.info(f"Early stopping at epoch {epoch+1} (patience={max_patience})")
            break

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) < epochs:
            log.info(f"  CHECKPOINT @ Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | Best Expect: {best_expectancy:.4f}")
            try:
                resp = input("Continue training? (Y/n): ").strip().lower()
                if resp == 'n':
                    log.info("User stopped training at checkpoint")
                    break
            except EOFError:
                pass

    # === POST-TRAINING: Load best checkpoint ===
    scaler_path = checkpoint_dir / "scaler_dist.joblib"
    engineer.save_scalers(str(scaler_path))
    log.info(f"Scaler saved to {scaler_path}")

    best_ckpt = checkpoint_dir / "best_dist_expectancy.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"Loaded best expectancy checkpoint (E={best_expectancy:.4f} at top{int(best_expectancy_pct*100)}%)")

    log.info("=" * 60)
    log.info("  DISTRIBUTIONAL TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"  Best val loss: {best_val_loss:.4f}")
    log.info(f"  Best expectancy: {best_expectancy:.4f} at top{int(best_expectancy_pct*100)}%")

    return model, engineer, features_df_columns, history


def main():
    parser = argparse.ArgumentParser(
        description=f"BTC Futures GPU Trainer - ENTER QUALITY Model ({SYSTEM_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A) Train fresh:
  python quick_start.py --url URL --epochs 300

  # B) Find best policy for 2-3 trades/day (NET):
  python quick_start.py --url URL --regime-eval --geometry-sweep \\
    --thresholds 0.80,0.85 --topn-list 8,10,12,15,18 \\
    --paired-tp-sl 3.0:1.25,3.0:1.5,3.5:1.5 --cooldowns 4,6,8 \\
    --target-tpd 2.5 --target-tpd-tol 1.0

  # C) Live run using saved best_policy.json (auto-loaded):
  python quick_start.py --url URL --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,XRPUSDT,ADAUSDT \\
    --interval 15m --enable-learning

  # Other:
  python quick_start.py --url URL --predict-only
  python quick_start.py --url URL --regime-eval --policy threshold:0.85 --cooldown 6
  python quick_start.py --url URL --live --dry-run --dry-run-candles 200
        """
    )
    parser.add_argument("--url", required=True, help="Your Replit dashboard URL")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (default: 6e-5)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs (default: 5)")
    parser.add_argument("--min-lr", type=float, default=None, help="Min LR for cosine annealing")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, predict from saved model")
    parser.add_argument("--no-push", action="store_true", help="Train but don't push prediction")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Pause every N epochs (0=no pausing)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR multiplier (default: 2.0)")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="SL ATR multiplier (default: 1.5)")
    parser.add_argument("--horizon", type=int, default=16, help="Horizon bars (default: 16)")
    parser.add_argument("--slope-eps", type=float, default=0.05, help="Min slope for trend gate (default: 0.05)")
    parser.add_argument("--r-min-expiry", type=float, default=1.0, help="Min R-multiple at expiry for ENTER=1 (default: 1.0)")
    parser.add_argument("--target-tpd", type=float, default=2.5, help="Target trades per day for BEST selection (default: 2.5)")
    parser.add_argument("--target-tpd-tol", type=float, default=1.0, help="Tolerance band for trades/day (default: 1.0)")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Explicitly trigger training mode")
    parser.add_argument("--value-loss-weight", type=float, default=0.5,
                        help="Weight for value head loss (default: 0.5)")
    parser.add_argument("--value-clip", type=float, default=3.0,
                        help="Clip value targets to [-clip, +clip] R (default: 3.0)")
    parser.add_argument("--smoke-calib", action="store_true", default=False,
                        help="Run temperature calibration smoke test after training")
    parser.add_argument("--smoke-infer", action="store_true", default=False,
                        help="Run single-row inference smoke test per symbol after training")
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use focal BCE loss (default: True)")
    parser.add_argument("--no-focal-loss", action="store_true", default=False,
                        help="Disable focal loss, use standard BCE")
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Focal loss gamma (default: 1.0)")
    parser.add_argument("--focal-alpha", type=float, default=0.45,
                        help="Focal loss alpha for ENTER=1 class (default: 0.45)")
    parser.add_argument("--use-ohem", action="store_true", default=False,
                        help="Use Online Hard Example Mining (default: False, temporarily disabled)")
    parser.add_argument("--no-ohem", action="store_true", default=False,
                        help="Disable OHEM")
    parser.add_argument("--ohem-neg-pct", type=float, default=0.15,
                        help="OHEM: keep top K%% hardest negatives (default: 0.08)")
    parser.add_argument("--use-edge-head", action="store_true", default=True,
                        help="Enable edge regression head (default: True)")
    parser.add_argument("--no-edge-head", action="store_true", default=False,
                        help="Disable edge head")
    parser.add_argument("--edge-loss-weight", type=float, default=0.15,
                        help="Weight for edge head loss (default: 0.15)")
    parser.add_argument("--use-soft-labels", action="store_true", default=True,
                        help="Use soft quality labels (default: True)")
    parser.add_argument("--no-soft-labels", action="store_true", default=False,
                        help="Disable soft quality labels")
    parser.add_argument("--soft-label-temp", type=float, default=1.2,
                        help="Soft label sigmoid temperature (default: 1.2)")
    parser.add_argument("--promote-min-pr-auc", type=float, default=0.42,
                        help="Min PR-AUC for promotion gate (default: 0.42)")
    parser.add_argument("--verify-pr-auc-upgrade", action="store_true", default=False,
                        help="Run verification: assert focal+OHEM+edge active, ECE computed, PR-AUC gate set")
    parser.add_argument("--min-enet-core", type=float, default=0.00,
                        help="Min E[net R] for CORE lane (default: 0.00)")
    parser.add_argument("--min-enet-flow", type=float, default=-0.05,
                        help="Min E[net R] for FLOW lane (default: -0.05)")
    parser.add_argument("--min-enet-scalp", type=float, default=-0.02,
                        help="Min E[net R] for SCALP lane (default: -0.02)")
    parser.add_argument("--budget-core", type=float, default=1.20,
                        help="Daily R budget for CORE lane per symbol (default: 1.20)")
    parser.add_argument("--budget-flow", type=float, default=0.60,
                        help="Daily R budget for FLOW lane per symbol (default: 0.60)")
    parser.add_argument("--budget-scalp", type=float, default=0.20,
                        help="Daily R budget for SCALP lane per symbol (default: 0.20)")
    parser.add_argument("--verify-separation", action="store_true", default=False,
                        help="Run 200-cycle dry-run verifying SCALP/CORE separation, budget bounds, exit logs")
    parser.add_argument("--gate-pf-net", type=float, default=1.05,
                        help="Promotion gate: min PF_net (default: 1.05)")
    parser.add_argument("--gate-enet", type=float, default=0.0,
                        help="Promotion gate: min E[net] (default: 0.0)")
    parser.add_argument("--gate-profitable-regimes", type=int, default=2,
                        help="Promotion gate: min profitable regimes (default: 2)")
    parser.add_argument("--gate-maxdd-r", type=float, default=6.0,
                        help="Promotion gate: max drawdown in R (default: 6.0)")
    parser.add_argument("--gate-p95-min", type=float, default=0.40,
                        help="Promotion gate: min p95 for calibration sanity (default: 0.40)")
    parser.add_argument("--gate-p95-max", type=float, default=0.98,
                        help="Promotion gate: max p95 for calibration sanity (default: 0.98)")
    parser.add_argument("--regime-eval", action="store_true", help="Run regime robustness evaluation (no training)")
    parser.add_argument("--regimes", type=str,
                        default="2019-01-01:2020-12-31,2021-01-01:2021-12-31,2022-01-01:2022-12-31,2023-01-01:2024-12-31",
                        help="Comma-separated date ranges as START:END (YYYY-MM-DD)")
    parser.add_argument("--policy", type=str, default="threshold:0.70",
                        help="Trade selection policy: 'threshold:0.70' or 'percentile:top20'")
    parser.add_argument("--cooldown", type=int, default=4, help="Cooldown bars after each trade (default: 4)")
    parser.add_argument("--fees-entry-bps", type=float, default=5.0, help="Entry fee in basis points (default: 5.0 = taker)")
    parser.add_argument("--fees-exit-bps", type=float, default=5.0, help="Exit fee in basis points (default: 5.0 = taker)")
    parser.add_argument("--spread-bps", type=float, default=1.0, help="Spread cost in basis points (default: 1.0)")
    parser.add_argument("--slip-k", type=float, default=0.10, help="Slippage factor as fraction of ATR (default: 0.10)")
    parser.add_argument("--size-cap", type=float, default=2.0, help="Max confidence size multiplier (default: 2.0)")
    parser.add_argument("--geometry-sweep", action="store_true",
                        help="Run geometry sweep with preset paired TP/SL combos")
    parser.add_argument("--tp-mults", type=str, default=None,
                        help="Comma-separated TP multipliers for non-sweep cartesian product (e.g. '2.0,2.5,3.0')")
    parser.add_argument("--sl-mults", type=str, default=None,
                        help="Comma-separated SL multipliers for non-sweep cartesian product (e.g. '1.25,1.5')")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds for sweep (e.g. '0.70,0.75')")
    parser.add_argument("--cooldowns", type=str, default=None,
                        help="Comma-separated cooldowns for sweep (e.g. '4,6,8')")
    parser.add_argument("--topn-list", type=str, default=None,
                        help="Comma-separated percentile topN values for sweep (e.g. '8,10,12,15,18')")
    parser.add_argument("--paired-tp-sl", type=str, default=None,
                        help="Comma-separated TP:SL pairs for sweep (e.g. '3.0:1.25,3.0:1.5,3.5:1.5')")
    parser.add_argument("--debug-costs", action="store_true",
                        help="Print 5 random trades per regime and assert cost accounting")

    parser.add_argument("--download-missing-data", action="store_true", default=False,
                        help="Auto-download missing 15m parquet data for all symbols before training")
    parser.add_argument("--allow-partial-data", action="store_true", default=False,
                        help="Allow training on subset of symbols if some data is missing (default: abort)")
    parser.add_argument("--symbol-balanced-sampling", action="store_true", default=True,
                        help="Cap per-symbol training samples to smallest symbol's count for balanced training (default: on)")
    parser.add_argument("--no-symbol-balanced-sampling", action="store_true", default=False,
                        help="Disable symbol-balanced sampling (allow BTC to dominate training)")
    parser.add_argument("--balanced-sampling-mode", type=str, choices=["cap", "weighted", "none"],
                        default="cap",
                        help="Symbol balancing mode: 'cap' truncates to min count, 'weighted' keeps all data with inverse-frequency loss weights, 'none' disables (default: cap)")
    parser.add_argument("--v5-symbol-embed-dim", type=int, default=8,
                        help="Symbol embedding dimension for multi-asset models (default: 8)")
    parser.add_argument("--per-symbol-scaler", action="store_true", default=False,
                        help="Fit/apply RobustScaler per symbol instead of global (default: off)")

    parser.add_argument("--v6", action="store_true", default=False,
                        help="Use V6Forecaster (Temporal-MoE-Attention) instead of V5 (default: off)")
    parser.add_argument("--v6-seq-len", type=int, default=16,
                        help="V6 sequence window length in bars (default: 16)")
    parser.add_argument("--v6-conv-channels", type=int, default=128,
                        help="V6 causal conv1d channel width (default: 128)")
    parser.add_argument("--v6-n-conv-layers", type=int, default=3,
                        help="V6 number of causal conv layers (default: 3)")
    parser.add_argument("--v6-attn-heads", type=int, default=4,
                        help="V6 number of self-attention heads (default: 4)")
    parser.add_argument("--v6-attn-layers", type=int, default=2,
                        help="V6 number of transformer blocks (default: 2)")
    parser.add_argument("--v6-n-experts", type=int, default=4,
                        help="V6 number of MoE experts (default: 4)")
    parser.add_argument("--v6-expert-top-k", type=int, default=2,
                        help="V6 top-k expert routing (default: 2)")
    parser.add_argument("--v6-feature-mask-ratio", type=float, default=0.15,
                        help="V6 random feature masking ratio during training (default: 0.15)")
    parser.add_argument("--v6-aux-weight", type=float, default=0.1,
                        help="V6 auxiliary next-bar prediction loss weight (default: 0.1)")
    parser.add_argument("--v6-confidence-weight", type=float, default=0.15,
                        help="V6 confidence calibration loss weight (default: 0.15)")
    parser.add_argument("--v6-moe-balance-weight", type=float, default=0.05,
                        help="V6 MoE load balancing loss weight (default: 0.05)")
    parser.add_argument("--loss-warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs using plain BCE before switching to focal/OHEM (default: 10)")
    parser.add_argument("--warmup-pos-weight", type=float, default=2.0,
                        help="Max pos_weight during loss warmup stage (default: 2.0)")
    parser.add_argument("--transition-epochs", type=int, default=15,
                        help="Number of transition epochs between WARMUP and FULL (default: 15)")
    parser.add_argument("--focal-gamma-final", type=float, default=0.5,
                        help="Final focal gamma after transition ramp (default: 0.5)")
    parser.add_argument("--lr-drop-on-transition", type=float, default=0.65,
                        help="LR multiplier at WARMUP->TRANSITION boundary (default: 0.65)")
    parser.add_argument("--disable-ohem-during-transition", action="store_true", default=True,
                        help="Keep OHEM off during transition (default: True)")
    parser.add_argument("--enable-ohem-during-transition", action="store_true", default=False,
                        help="Allow OHEM during transition stage")
    parser.add_argument("--collapse-guard", action="store_true", default=True,
                        help="Enable collapse guard (default: True)")
    parser.add_argument("--no-collapse-guard", action="store_true", default=False,
                        help="Disable collapse guard")
    parser.add_argument("--collapse-guard-pred1", type=float, default=0.98,
                        help="Pred1 threshold for collapse guard (default: 0.98)")
    parser.add_argument("--collapse-guard-sep", type=float, default=0.02,
                        help="Separation threshold for collapse guard (default: 0.02)")
    parser.add_argument("--collapse-guard-freeze-epochs", type=int, default=5,
                        help="Epochs to freeze OHEM and clamp pos_weight after collapse (default: 5)")
    parser.add_argument("--verify-enter-metrics", action="store_true", default=False,
                        help="Run 3-pass validation verification with assertions and generate report")
    parser.add_argument("--w-quality", type=float, default=1.0,
                        help="Weight for quality (enter) loss in composite loss (default: 1.0)")
    parser.add_argument("--w-dir", type=float, default=0.5,
                        help="Weight for direction loss in composite loss (default: 0.5)")
    parser.add_argument("--w-htf", type=float, default=0.5,
                        help="Weight for HTF supervision loss in composite loss (default: 0.5)")
    parser.add_argument("--verify-v46-separation", action="store_true", default=False,
                        help="Run v4.6 label/model verification checks without full training")

    parser.add_argument("--use-v47-labels", action="store_true", default=True,
                        help="Use v4.7 strict quality labeling (default: True)")
    parser.add_argument("--no-v47-labels", dest="use_v47_labels", action="store_false",
                        help="Disable v4.7 labels, fall back to v4.6")
    parser.add_argument("--q-min-tp", type=float, default=0.3,
                        help="Min TP quality score for ENTER=1 in v4.7.1 labeling (default: 0.3)")
    parser.add_argument("--r-min-expiry-strict", type=float, default=1.0,
                        help="Min R at expiry to count as positive in v4.7 (default: 1.0)")
    parser.add_argument("--auto-balance-enter-labels", action="store_true", default=True,
                        help="Auto-tune q_min_tp for target positive rate (default: True)")
    parser.add_argument("--no-auto-balance", dest="auto_balance_enter_labels", action="store_false",
                        help="Disable auto-balancing of enter labels")
    parser.add_argument("--target-enter-rate", type=float, default=0.18,
                        help="Target ENTER positive rate for auto-balance (default: 0.18)")
    parser.add_argument("--target-enter-rate-min", type=float, default=0.12,
                        help="Min acceptable ENTER positive rate (default: 0.12)")
    parser.add_argument("--target-enter-rate-max", type=float, default=0.25,
                        help="Max acceptable ENTER positive rate (default: 0.25)")
    parser.add_argument("--balance-search-steps", type=int, default=30,
                        help="Number of search steps for auto-balance (default: 30)")
    parser.add_argument("--pos-weight-min", type=float, default=0.5,
                        help="Min pos_weight guardrail (default: 0.5)")
    parser.add_argument("--pos-weight-max", type=float, default=6.0,
                        help="Max pos_weight guardrail (default: 6.0)")
    parser.add_argument("--verify-v47-labels", action="store_true", default=False,
                        help="Run v4.7 label verification and generate report")

    parser.add_argument("--train-distributional", action="store_true", default=False,
                        help="v4.9.0: Train distributional trade forecaster instead of binary ENTER model")
    parser.add_argument("--dist-w-mse", type=float, default=1.0,
                        help="Weight for E[R] Huber loss in distributional mode (default: 1.0)")
    parser.add_argument("--dist-w-quantile", type=float, default=0.5,
                        help="Weight for quantile pinball loss (default: 0.5)")
    parser.add_argument("--dist-w-bce", type=float, default=0.5,
                        help="Weight for p(win) BCE loss (default: 0.5)")
    parser.add_argument("--dist-w-regime", type=float, default=0.0,
                        help="Weight for regime CE loss (default: 0.0, disabled)")
    parser.add_argument("--dist-w-dir", type=float, default=0.3,
                        help="Weight for direction loss in distributional mode (default: 0.3)")
    parser.add_argument("--score-lambda", type=float, default=0.5,
                        help="Lambda for downside penalty in score: sigmoid(win)*E_R - lambda*max(0,-q10) (default: 0.5)")
    parser.add_argument("--use-regime-head", action="store_true", default=False,
                        help="Enable regime classification head (chop/trend/highvol) in distributional mode")
    parser.add_argument("--dist-target-tpd", type=float, default=6.5,
                        help="Target trades/day for distributional sweep (default: 6.5)")
    parser.add_argument("--dist-target-tpd-tol", type=float, default=1.5,
                        help="Tolerance for distributional target tpd (default: 1.5)")

    parser.add_argument("--use-candidates", action="store_true", default=False,
                        help="Enable candidate engine: filter bars by ATR/vol/breakout/fee gate")
    parser.add_argument("--no-candidates", dest="use_candidates", action="store_false")
    parser.add_argument("--cand-min-atr-pct", type=float, default=0.0015,
                        help="Min ATR%% for candidate filter (default: 0.0015)")
    parser.add_argument("--cand-breakout", action="store_true", default=True,
                        help="Enable breakout trigger in candidate filter")
    parser.add_argument("--cand-mean-reversion", action="store_true", default=True,
                        help="Enable mean-reversion trigger in candidate filter")
    parser.add_argument("--cand-fee-gate", action="store_true", default=True,
                        help="Enable fee/spread gate in candidate filter")
    parser.add_argument("--cand-round-trip-cost", type=float, default=0.0009,
                        help="Round-trip cost for fee gate (default: 0.0009)")
    parser.add_argument("--cand-target-rate", type=float, default=0.40,
                        help="Target candidate eligibility rate (default: 0.40)")
    parser.add_argument("--cand-min-rate", type=float, default=0.25,
                        help="Min candidate rate before auto-relax triggers (default: 0.25)")

    parser.add_argument("--multi-preset-mode", type=str, default="fixed:standard",
                        help="Multi-preset mode: 'oracle' (best-of hindsight, research only), "
                             "'fixed:<preset>' (e.g. fixed:standard), or 'learnable' (preset_head). "
                             "Default: fixed:standard")

    parser.add_argument("--train-v5", action="store_true", default=False,
                        help="v5.0: Train V5 Forecaster (continuous market predictions + decision layer)")
    parser.add_argument("--train-mythos", action="store_true", default=False,
                        help="Train MYTHOS stack (world model + expert council + router) with walk-forward evaluation")
    parser.add_argument("--mythos-profile", type=str, default="modern", choices=["modern", "legacy-stable"],
                        help="MYTHOS behavior profile: modern (full stack) or legacy-stable (pre-collapse compatibility)")
    parser.add_argument("--mythos-train-months", type=int, default=12,
                        help="MYTHOS walk-forward training window in months (default: 12)")
    parser.add_argument("--mythos-test-months", type=int, default=1,
                        help="MYTHOS walk-forward test window in months (default: 1)")
    parser.add_argument("--mythos-max-folds", type=int, default=None,
                        help="MYTHOS: optional max number of most recent folds to run")
    parser.add_argument("--mythos-n-regimes", type=int, default=4,
                        help="MYTHOS world-model latent regime count (default: 4)")
    parser.add_argument("--mythos-min-regime-confidence", type=float, default=0.45,
                        help="MYTHOS minimum regime posterior confidence to allow routing (default: 0.45)")
    parser.add_argument("--mythos-min-confidence", type=float, default=0.55,
                        help="MYTHOS minimum router confidence to allow trading (default: 0.55)")
    parser.add_argument("--mythos-min-expected-r", type=float, default=0.01,
                        help="MYTHOS minimum expected R per trade to allow execution (default: 0.01)")
    parser.add_argument("--mythos-edge-threshold", type=float, default=0.02,
                        help="MYTHOS minimum edge threshold for promotion gate (default: 0.02)")
    parser.add_argument("--mythos-daily-loss-cap", type=float, default=-4.0,
                        help="MYTHOS daily loss cap in R (default: -4.0)")
    parser.add_argument("--mythos-weekly-loss-cap", type=float, default=-12.0,
                        help="MYTHOS weekly loss cap in R (default: -12.0)")
    parser.add_argument("--mythos-emergency-stop-r", type=float, default=-18.0,
                        help="MYTHOS hard emergency stop: halt new trades once fold equity reaches this floor in R (default: -18.0)")
    parser.add_argument("--mythos-emergency-max-drawdown-r", type=float, default=12.0,
                        help="MYTHOS hard emergency stop: halt new trades once peak-to-trough drawdown reaches this R (default: 12.0)")
    parser.add_argument("--mythos-drawdown-size-start-r", type=float, default=6.0,
                        help="MYTHOS size throttle: drawdown level where position-size throttling starts (default: 6.0)")
    parser.add_argument("--mythos-dd-size-throttle-start-r", dest="mythos_drawdown_size_start_r", type=float,
                        help="Alias for --mythos-drawdown-size-start-r")
    parser.add_argument("--mythos-drawdown-size-full-r", type=float, default=14.0,
                        help="MYTHOS size throttle: drawdown level where minimum throttle is reached (default: 14.0)")
    parser.add_argument("--mythos-dd-size-throttle-end-r", dest="mythos_drawdown_size_full_r", type=float,
                        help="Alias for --mythos-drawdown-size-full-r")
    parser.add_argument("--mythos-drawdown-size-min-scale", type=float, default=0.35,
                        help="MYTHOS size throttle: minimum size scaling under deep drawdown (default: 0.35)")
    parser.add_argument("--mythos-dd-size-throttle-min", dest="mythos_drawdown_size_min_scale", type=float,
                        help="Alias for --mythos-drawdown-size-min-scale")
    parser.add_argument("--mythos-disable-conviction-boost-dd-r", type=float, default=6.0,
                        help="MYTHOS leverage safety: disable conviction boost once drawdown reaches this R (default: 6.0)")
    parser.add_argument("--mythos-disable-leverage-dd-r", type=float, default=6.0,
                        help="MYTHOS leverage safety: disable leverage once drawdown reaches this R (default: 6.0)")
    parser.add_argument("--mythos-dd-disable-leverage-r", dest="mythos_disable_leverage_dd_r", type=float,
                        help="Alias for --mythos-disable-leverage-dd-r")
    parser.add_argument("--mythos-dd-risk-recovery-r", type=float, default=3.0,
                        help="MYTHOS leverage safety: drawdown level to re-enable leverage after disable threshold (default: 3.0)")
    parser.add_argument("--mythos-cooldown-bars", type=int, default=4,
                        help="MYTHOS bars of cooldown after each executed trade (default: 4)")
    parser.add_argument("--mythos-max-trades-per-day", type=int, default=8,
                        help="MYTHOS max trades/day before throttling (default: 8)")
    parser.add_argument("--mythos-max-leverage", type=float, default=1.8,
                        help="MYTHOS maximum leverage multiplier (default: 1.8)")
    parser.add_argument("--mythos-vol-target", type=float, default=0.012,
                        help="MYTHOS daily volatility target for sizing (default: 0.012)")
    parser.add_argument("--mythos-min-trades", type=int, default=25,
                        help="MYTHOS minimum trades for confidence classification (default: 25)")
    parser.add_argument("--mythos-report-path", type=str, default="checkpoints/mythos_walkforward_report.json",
                        help="MYTHOS output report path (default: checkpoints/mythos_walkforward_report.json)")
    parser.add_argument("--mythos-save-best-model", action="store_true", default=True,
                        help="MYTHOS: persist best fold model artifact (default: enabled)")
    parser.add_argument("--mythos-no-save-best-model", dest="mythos_save_best_model", action="store_false",
                        help="MYTHOS: disable best-model artifact persistence")
    parser.add_argument("--mythos-best-model-metric", type=str, default="total_r",
                        choices=["total_r", "expectancy_r", "win_rate", "robust_score"],
                        help="MYTHOS: metric for selecting best fold model (default: total_r)")
    parser.add_argument("--mythos-model-output-dir", type=str, default="checkpoints/mythos_models",
                        help="MYTHOS: directory to save exported model artifacts")
    parser.add_argument("--mythos-analog-k", type=int, default=48,
                        help="MYTHOS v2: nearest analog memory neighbors (default: 48)")
    parser.add_argument("--mythos-analog-blend", type=float, default=0.35,
                        help="MYTHOS v2: blend weight for analog memory edge/confidence (default: 0.35)")
    parser.add_argument("--mythos-online-reliability-alpha", type=float, default=0.08,
                        help="MYTHOS v2: EMA speed for online reliability updates (default: 0.08)")
    parser.add_argument("--mythos-reliability-regime-window", type=int, default=80,
                        help="MYTHOS v2: rolling regime-specific reliability window (default: 80)")
    parser.add_argument("--mythos-robust-score-dd-penalty", type=float, default=0.35,
                        help="MYTHOS v2: drawdown penalty factor for robust_score metric (default: 0.35)")
    parser.add_argument("--mythos-side-balance-window", type=int, default=160,
                        help="MYTHOS v3: rolling window for side imbalance control (default: 160)")
    parser.add_argument("--mythos-side-imbalance-soft-cap", type=float, default=0.82,
                        help="MYTHOS v3: soft max side concentration before penalties (default: 0.82)")
    parser.add_argument("--mythos-side-imbalance-edge-penalty", type=float, default=0.015,
                        help="MYTHOS v3: edge penalty when one side dominates (default: 0.015)")
    parser.add_argument("--mythos-side-rebalance-enable", dest="mythos_side_rebalance_enable", action="store_true",
                        help="MYTHOS side rebalance: enable adaptive short/long rebalance nudges (default: enabled)")
    parser.add_argument("--mythos-no-side-rebalance-enable", dest="mythos_side_rebalance_enable", action="store_false",
                        help="MYTHOS side rebalance: disable adaptive short/long rebalance nudges")
    parser.set_defaults(mythos_side_rebalance_enable=True)
    parser.add_argument("--mythos-side-rebalance-warmup-trades", type=int, default=40,
                        help="MYTHOS side rebalance: warmup accepted trades before rebalance nudges (default: 40)")
    parser.add_argument("--mythos-side-rebalance-window", type=int, default=96,
                        help="MYTHOS side rebalance: rolling window for side-share estimation (default: 96)")
    parser.add_argument("--mythos-side-rebalance-short-target", type=float, default=0.32,
                        help="MYTHOS side rebalance: target short-side share in active window (default: 0.32)")
    parser.add_argument("--mythos-side-rebalance-short-boost", type=float, default=0.0035,
                        help="MYTHOS side rebalance: max short edge boost when underweight (default: 0.0035)")
    parser.add_argument("--mythos-side-rebalance-long-penalty", type=float, default=0.0030,
                        help="MYTHOS side rebalance: long edge penalty when short side is underweight (default: 0.0030)")
    parser.add_argument("--mythos-side-rebalance-conf-boost", type=float, default=0.02,
                        help="MYTHOS side rebalance: short confidence boost when underweight (default: 0.02)")
    parser.add_argument("--mythos-side-rebalance-quality-guard", type=float, default=0.06,
                        help="MYTHOS side rebalance: block short boost when short expectancy lags long by this guard (default: 0.06)")
    parser.add_argument("--mythos-side-rebalance-max-adjust", type=float, default=0.012,
                        help="MYTHOS side rebalance: cap on per-trade edge adjustment from rebalance nudges (default: 0.012)")
    parser.add_argument("--mythos-intelligence-enable", dest="mythos_intelligence_enable", action="store_true",
                        help="MYTHOS intelligence: enable online quality learner to recalibrate edge/confidence/uncertainty (default: enabled)")
    parser.add_argument("--mythos-no-intelligence-enable", dest="mythos_intelligence_enable", action="store_false",
                        help="MYTHOS intelligence: disable online quality learner recalibration")
    parser.set_defaults(mythos_intelligence_enable=True)
    parser.add_argument("--mythos-intelligence-min-samples", type=int, default=24,
                        help="MYTHOS intelligence: minimum samples per bucket before quality adjustments activate (default: 24)")
    parser.add_argument("--mythos-intelligence-ema-alpha", type=float, default=0.08,
                        help="MYTHOS intelligence: EMA alpha for online quality memory updates (default: 0.08)")
    parser.add_argument("--mythos-intelligence-hit-weight", type=float, default=0.55,
                        help="MYTHOS intelligence: score weight on hit-rate quality signal (default: 0.55)")
    parser.add_argument("--mythos-intelligence-expectancy-weight", type=float, default=0.45,
                        help="MYTHOS intelligence: score weight on expectancy quality signal (default: 0.45)")
    parser.add_argument("--mythos-intelligence-variance-penalty", type=float, default=0.18,
                        help="MYTHOS intelligence: penalty weight for unstable/high-variance quality buckets (default: 0.18)")
    parser.add_argument("--mythos-intelligence-edge-scale", type=float, default=0.010,
                        help="MYTHOS intelligence: positive score to edge boost scale (default: 0.010)")
    parser.add_argument("--mythos-intelligence-negative-edge-scale", type=float, default=0.012,
                        help="MYTHOS intelligence: negative score to edge penalty scale (default: 0.012)")
    parser.add_argument("--mythos-intelligence-conf-scale", type=float, default=0.06,
                        help="MYTHOS intelligence: score to confidence adjustment scale (default: 0.06)")
    parser.add_argument("--mythos-intelligence-uncertainty-scale", type=float, default=0.30,
                        help="MYTHOS intelligence: score to uncertainty adjustment scale (default: 0.30)")
    parser.add_argument("--mythos-intelligence-max-edge-adjust", type=float, default=0.020,
                        help="MYTHOS intelligence: cap on per-trade edge adjustment from quality learner (default: 0.020)")
    parser.add_argument("--mythos-intelligence-side-switch-enable", dest="mythos_intelligence_side_switch_enable", action="store_true",
                        help="MYTHOS intelligence: allow quality-driven side switch when opposite side has stronger evidence (default: enabled)")
    parser.add_argument("--mythos-no-intelligence-side-switch-enable", dest="mythos_intelligence_side_switch_enable", action="store_false",
                        help="MYTHOS intelligence: disable quality-driven side switch")
    parser.set_defaults(mythos_intelligence_side_switch_enable=True)
    parser.add_argument("--mythos-intelligence-side-switch-min-gap", type=float, default=0.30,
                        help="MYTHOS intelligence: minimum opposite-side quality score gap required to switch side (default: 0.30)")
    parser.add_argument("--mythos-intelligence-side-switch-min-analog-adv", type=float, default=0.0015,
                        help="MYTHOS intelligence: only allow side switch when chosen side analog edge is below this floor (default: 0.0015)")
    parser.add_argument("--mythos-intelligence-side-switch-conviction-guard", type=float, default=0.58,
                        help="MYTHOS intelligence: block side switching when conviction is above this threshold (default: 0.58)")
    parser.add_argument("--mythos-intelligence-side-switch-cooldown-bars", type=int, default=16,
                        help="MYTHOS intelligence: minimum bars between quality-driven side switches (default: 16)")
    parser.add_argument("--mythos-intelligence-side-switch-max-rate", dest="mythos_intelligence_side_switch_max_rate", type=float, default=0.08,
                        help="MYTHOS intelligence: maximum fraction of bars allowed to side-switch before throttling (default: 0.08)")
    parser.add_argument("--mythos-intelligence-max-switch-rate", dest="mythos_intelligence_side_switch_max_rate", type=float,
                        help="Alias for --mythos-intelligence-side-switch-max-rate")
    parser.add_argument("--mythos-intelligence-side-switch-min-samples", dest="mythos_intelligence_side_switch_min_samples", type=int, default=40,
                        help="MYTHOS intelligence: minimum intelligence-active bars before side switching can activate (default: 40)")
    parser.add_argument("--mythos-adaptive-side-target-strength", type=float, default=0.22,
                        help="MYTHOS adaptive: how strongly side health shifts long/short target mix (default: 0.22)")
    parser.add_argument("--mythos-adaptive-side-target-min", type=float, default=0.35,
                        help="MYTHOS adaptive: minimum target long share (default: 0.35)")
    parser.add_argument("--mythos-adaptive-side-target-max", type=float, default=0.65,
                        help="MYTHOS adaptive: maximum target long share (default: 0.65)")
    parser.add_argument("--mythos-side-health-penalty", type=float, default=0.02,
                        help="MYTHOS adaptive: extra edge penalty scale for overrepresented weak side (default: 0.02)")
    parser.add_argument("--mythos-side-health-boost", type=float, default=0.008,
                        help="MYTHOS adaptive: edge boost scale for underrepresented healthier side (default: 0.008)")
    parser.add_argument("--mythos-side-health-decay", type=float, default=0.97,
                        help="MYTHOS adaptive: decay factor for side health memory (default: 0.97)")
    parser.add_argument("--mythos-drawdown-edge-start-r", type=float, default=8.0,
                        help="MYTHOS v3: drawdown level where edge floor starts tightening (default: 8.0)")
    parser.add_argument("--mythos-drawdown-edge-step-r", type=float, default=4.0,
                        help="MYTHOS v3: drawdown step for incremental edge floor tightening (default: 4.0)")
    parser.add_argument("--mythos-drawdown-edge-boost", type=float, default=0.0025,
                        help="MYTHOS v3: added edge floor per drawdown step (default: 0.0025)")
    parser.add_argument("--mythos-loss-streak-trigger", type=int, default=4,
                        help="MYTHOS v3: consecutive losses before cooldown pause (default: 4)")
    parser.add_argument("--mythos-loss-streak-cooldown-bars", type=int, default=12,
                        help="MYTHOS v3: bars to pause after loss streak trigger (default: 12)")
    parser.add_argument("--mythos-side-fail-window", type=int, default=48,
                        help="MYTHOS v5: rolling trades window used to detect failing long/short side (default: 48)")
    parser.add_argument("--mythos-side-fail-min-trades", type=int, default=10,
                        help="MYTHOS v5: minimum side trades in window before side-fail cooldown can trigger (default: 10)")
    parser.add_argument("--mythos-side-fail-expectancy-r", type=float, default=-0.12,
                        help="MYTHOS v5: side expectancy threshold that triggers side cooldown (default: -0.12)")
    parser.add_argument("--mythos-side-fail-cooldown-bars", type=int, default=24,
                        help="MYTHOS v5: bars to pause entries for a failing side (default: 24)")
    parser.add_argument("--mythos-side-fail-hard-pause", dest="mythos_side_fail_hard_pause", action="store_true",
                        help="MYTHOS adaptive: hard-pause failing side instead of soft adaptive downweighting")
    parser.add_argument("--mythos-no-side-fail-hard-pause", dest="mythos_side_fail_hard_pause", action="store_false",
                        help="MYTHOS adaptive: keep trading both sides and adaptively rebalance (default)")
    parser.set_defaults(mythos_side_fail_hard_pause=False)
    parser.add_argument("--mythos-online-allocator-lr", type=float, default=0.06,
                        help="MYTHOS v4: online expert allocator learning rate (default: 0.06)")
    parser.add_argument("--mythos-online-allocator-min-mult", type=float, default=0.75,
                        help="MYTHOS v4: min multiplier for expert utility scaling (default: 0.75)")
    parser.add_argument("--mythos-online-allocator-max-mult", type=float, default=1.55,
                        help="MYTHOS v4: max multiplier for expert utility scaling (default: 1.55)")
    parser.add_argument("--mythos-change-detect-z-thresh", type=float, default=2.6,
                        help="MYTHOS v4: z-score trigger threshold for change detection (default: 2.6)")
    parser.add_argument("--mythos-change-detect-confirm-bars", type=int, default=2,
                        help="MYTHOS v4: consecutive bars required to confirm change mode (default: 2)")
    parser.add_argument("--mythos-change-detect-cooldown-bars", type=int, default=24,
                        help="MYTHOS v4: cooldown bars while change mode is active (default: 24)")
    parser.add_argument("--mythos-change-edge-floor-boost", type=float, default=0.004,
                        help="MYTHOS v4: additional edge floor during confirmed change mode (default: 0.004)")
    parser.add_argument("--mythos-change-confidence-boost", type=float, default=0.03,
                        help="MYTHOS v4: extra confidence requirement during change mode (default: 0.03)")
    parser.add_argument("--mythos-change-uncertainty-mult", type=float, default=1.15,
                        help="MYTHOS v4: uncertainty inflation during change mode (default: 1.15)")
    parser.add_argument("--mythos-regime-flip-window", type=int, default=24,
                        help="MYTHOS v5: bars used to track fast regime flip intensity (default: 24)")
    parser.add_argument("--mythos-regime-flip-trigger", type=float, default=0.35,
                        help="MYTHOS v5: flip intensity threshold to trigger instability mode (default: 0.35)")
    parser.add_argument("--mythos-instability-edge-mult", type=float, default=0.70,
                        help="MYTHOS v5: edge multiplier during instability mode (default: 0.70)")
    parser.add_argument("--mythos-instability-confidence-drop", type=float, default=0.08,
                        help="MYTHOS v5: confidence drop applied during instability mode (default: 0.08)")
    parser.add_argument("--mythos-instability-uncertainty-mult", type=float, default=1.35,
                        help="MYTHOS v5: uncertainty multiplier during instability mode (default: 1.35)")
    parser.add_argument("--mythos-transition-learn-rate", type=float, default=0.12,
                        help="MYTHOS v5: EMA learning rate for transition-state memory (default: 0.12)")
    parser.add_argument("--mythos-transition-min-samples", type=int, default=6,
                        help="MYTHOS v5: minimum observations before transition-state adjustments activate (default: 6)")
    parser.add_argument("--mythos-transition-edge-gain", type=float, default=0.35,
                        help="MYTHOS v5: transition-memory gain applied to edge scaling (default: 0.35)")
    parser.add_argument("--mythos-transition-confidence-gain", type=float, default=0.06,
                        help="MYTHOS v5: transition-memory gain applied to confidence adjustment (default: 0.06)")
    parser.add_argument("--mythos-transition-uncertainty-gain", type=float, default=0.30,
                        help="MYTHOS v5: transition-memory gain applied to uncertainty adjustment (default: 0.30)")
    parser.add_argument("--mythos-flip-harden-hold-bars", type=int, default=18,
                        help="MYTHOS v5: bars to keep strict guard after flip trigger (default: 18)")
    parser.add_argument("--mythos-counterfactual-min-advantage-r", type=float, default=0.006,
                        help="MYTHOS v5: minimum counterfactual edge advantage required to trade (default: 0.006)")
    parser.add_argument("--mythos-counterfactual-risk-penalty", type=float, default=0.6,
                        help="MYTHOS v5: uncertainty penalty in counterfactual gate (default: 0.6)")
    parser.add_argument("--mythos-counterfactual-margin", type=float, default=0.006,
                        help="MYTHOS v5: confidence margin bonus/penalty applied in counterfactual score (default: 0.006)")
    parser.add_argument("--mythos-counterfactual-uncertainty-weight", type=float, default=0.50,
                        help="MYTHOS v5: uncertainty penalty weight applied to alternative side score (default: 0.50)")
    parser.add_argument("--mythos-counterfactual-min-alt-hits", type=int, default=8,
                        help="MYTHOS v5: minimum analog hits per side before strict counterfactual filtering (default: 8)")
    parser.add_argument("--mythos-use-neural-expert", action="store_true", default=True,
                        help="MYTHOS v6: enable GPU neural expert in expert council (default: enabled)")
    parser.add_argument("--mythos-no-use-neural-expert", dest="mythos_use_neural_expert", action="store_false",
                        help="MYTHOS v6: disable GPU neural expert and use classic linear-only council")
    parser.add_argument("--mythos-neural-experts", dest="mythos_use_neural_expert", action="store_true",
                        help="Alias for --mythos-use-neural-expert")
    parser.add_argument("--mythos-no-neural-experts", dest="mythos_use_neural_expert", action="store_false",
                        help="Alias for --mythos-no-use-neural-expert")
    parser.add_argument("--mythos-neural-device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="MYTHOS v6: neural expert runtime device preference (default: auto)")
    parser.add_argument("--mythos-neural-hidden", type=int, default=64,
                        help="MYTHOS v6: hidden width for neural expert MLP (default: 64)")
    parser.add_argument("--mythos-neural-epochs", type=int, default=8,
                        help="MYTHOS v6: training epochs for neural expert per fold (default: 8)")
    parser.add_argument("--mythos-neural-lr", type=float, default=0.0015,
                        help="MYTHOS v6: learning rate for neural expert optimizer (default: 0.0015)")
    parser.add_argument("--mythos-neural-batch-size", type=int, default=512,
                        help="MYTHOS v6: batch size for neural expert training (default: 512)")
    parser.add_argument("--mythos-meta-learner", action="store_true", default=True,
                        help="MYTHOS v7: enable neural meta-learner for trade quality modulation (default: enabled)")
    parser.add_argument("--mythos-no-meta-learner", dest="mythos_meta_learner", action="store_false",
                        help="MYTHOS v7: disable neural meta-learner and use base decision stack")
    parser.add_argument("--mythos-use-deep-meta", dest="mythos_meta_learner", action="store_true",
                        help="Alias for --mythos-meta-learner")
    parser.add_argument("--mythos-no-use-deep-meta", dest="mythos_meta_learner", action="store_false",
                        help="Alias for --mythos-no-meta-learner")
    parser.add_argument("--mythos-meta-device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="MYTHOS v7: meta-learner runtime device preference (default: auto)")
    parser.add_argument("--mythos-meta-hidden", type=int, default=64,
                        help="MYTHOS v7: hidden width for meta-learner MLP (default: 64)")
    parser.add_argument("--mythos-meta-epochs", type=int, default=6,
                        help="MYTHOS v7: training epochs for meta-learner per fold (default: 6)")
    parser.add_argument("--mythos-meta-lr", type=float, default=0.001,
                        help="MYTHOS v7: learning rate for meta-learner optimizer (default: 0.001)")
    parser.add_argument("--mythos-meta-batch-size", type=int, default=1024,
                        help="MYTHOS v7: batch size for meta-learner training (default: 1024)")
    parser.add_argument("--mythos-meta-edge-gain", type=float, default=0.45,
                        help="MYTHOS v7: edge scaling gain from meta quality score (default: 0.45)")
    parser.add_argument("--mythos-meta-confidence-gain", type=float, default=0.08,
                        help="MYTHOS v7: confidence adjustment gain from meta quality score (default: 0.08)")
    parser.add_argument("--mythos-meta-uncertainty-gain", type=float, default=0.25,
                        help="MYTHOS v7: uncertainty damping gain from meta quality score (default: 0.25)")
    parser.add_argument("--mythos-meta-min-train-samples", type=int, default=512,
                        help="MYTHOS v7: minimum samples before meta-learner online updates activate (default: 512)")
    parser.add_argument("--mythos-meta-warmup-trades", type=int, default=300,
                        help="MYTHOS v7: trades before meta-learner gating starts affecting execution (default: 300)")
    parser.add_argument("--mythos-meta-warmup-samples", type=int, default=1024,
                        help="MYTHOS v7: warmup samples required before strict meta gating activates (default: 1024)")
    parser.add_argument("--mythos-meta-ready-prob-floor", type=float, default=0.42,
                        help="MYTHOS v7: lower probability bound considered neutral during meta warmup (default: 0.42)")
    parser.add_argument("--mythos-meta-ready-prob-ceiling", type=float, default=0.58,
                        help="MYTHOS v7: upper probability bound considered neutral during meta warmup (default: 0.58)")
    parser.add_argument("--mythos-meta-fallback", dest="mythos_meta_fallback", action="store_true",
                        help="MYTHOS v7: enable fallback online meta learner when torch/cuda meta model is unavailable (default: enabled)")
    parser.add_argument("--mythos-no-meta-fallback", dest="mythos_meta_fallback", action="store_false",
                        help="MYTHOS v7: disable fallback meta learner")
    parser.set_defaults(mythos_meta_fallback=True)
    parser.add_argument("--mythos-meta-fallback-lr", type=float, default=0.03,
                        help="MYTHOS v7: learning rate for fallback online meta learner (default: 0.03)")
    parser.add_argument("--mythos-precision-min-confidence", type=float, default=0.0,
                        help="MYTHOS precision: minimum confidence required before conviction filter (default: 0.0)")
    parser.add_argument("--mythos-precision-min-edge", type=float, default=0.0,
                        help="MYTHOS precision: minimum edge required before conviction filter (default: 0.0)")
    parser.add_argument("--mythos-precision-min-conviction", type=float, default=0.52,
                        help="MYTHOS precision: minimum conviction score required to execute a trade (default: 0.52)")
    parser.add_argument("--mythos-precision-high-conviction", type=float, default=0.72,
                        help="MYTHOS precision: conviction threshold counted as high-certainty trade (default: 0.72)")
    parser.add_argument("--mythos-precision-edge-weight", type=float, default=0.30,
                        help="MYTHOS precision: conviction score weight for edge strength (default: 0.30)")
    parser.add_argument("--mythos-precision-confidence-weight", type=float, default=0.30,
                        help="MYTHOS precision: conviction score weight for confidence strength (default: 0.30)")
    parser.add_argument("--mythos-precision-uncertainty-weight", type=float, default=0.25,
                        help="MYTHOS precision: conviction score weight for low uncertainty (default: 0.25)")
    parser.add_argument("--mythos-precision-meta-weight", type=float, default=0.15,
                        help="MYTHOS precision: conviction score weight for meta decisiveness (default: 0.15)")
    parser.add_argument("--mythos-conviction-score-threshold", type=float, default=0.62,
                        help="MYTHOS precision: conviction score threshold to activate leverage boost (default: 0.62)")
    parser.add_argument("--mythos-conviction-boost", type=float, default=0.35,
                        help="MYTHOS precision: leverage boost intensity on high-conviction setups (default: 0.35)")
    parser.add_argument("--mythos-conviction-max-size-mult", type=float, default=2.2,
                        help="MYTHOS precision: cap for leverage multiplier after conviction boost (default: 2.2)")
    parser.add_argument("--mythos-conviction-recent-window", type=int, default=64,
                        help="MYTHOS precision: recent high-conviction trades used to confirm leverage boost (default: 64)")
    parser.add_argument("--mythos-conviction-recent-min-trades", type=int, default=20,
                        help="MYTHOS precision: minimum recent high-conviction trades before leverage boost can activate (default: 20)")
    parser.add_argument("--mythos-conviction-recent-min-expectancy", type=float, default=0.05,
                        help="MYTHOS precision: minimum recent high-conviction expectancy required for leverage boost (default: 0.05)")
    parser.add_argument("--mythos-sure-min-analog-hits", type=int, default=12,
                        help="MYTHOS precision: minimum analog memory hits required for a signal to be considered sure (default: 12)")
    parser.add_argument("--mythos-sure-min-analog-ratio", type=float, default=0.20,
                        help="MYTHOS precision: minimum analog hit ratio vs analog-k for sure signals (default: 0.20)")
    parser.add_argument("--mythos-sure-meta-strength-min", type=float, default=0.10,
                        help="MYTHOS precision: minimum meta decisiveness for sure signals (default: 0.10)")
    parser.add_argument("--mythos-sure-edge-buffer", type=float, default=0.001,
                        help="MYTHOS precision: edge buffer above min edge threshold for sure signals (default: 0.001)")
    parser.add_argument("--mythos-sure-confidence-buffer", type=float, default=0.03,
                        help="MYTHOS precision: confidence buffer above min confidence for sure signals (default: 0.03)")
    parser.add_argument("--mythos-sure-recent-window", type=int, default=96,
                        help="MYTHOS precision: rolling window for sure-signal quality checks (default: 96)")
    parser.add_argument("--mythos-sure-recent-min-trades", type=int, default=24,
                        help="MYTHOS precision: min sure trades before sure quality gate becomes strict (default: 24)")
    parser.add_argument("--mythos-sure-recent-min-hit-rate", type=float, default=0.52,
                        help="MYTHOS precision: minimum recent sure hit-rate for continued sure qualification (default: 0.52)")
    parser.add_argument("--mythos-sure-recent-min-expectancy", type=float, default=0.04,
                        help="MYTHOS precision: minimum recent sure expectancy-R for sure qualification (default: 0.04)")
    parser.add_argument("--mythos-sure-cold-start-conviction-extra", type=float, default=0.04,
                        help="MYTHOS precision: extra conviction required during sure cold-start period (default: 0.04)")
    parser.add_argument("--mythos-sure-cold-start-meta-extra", type=float, default=0.06,
                        help="MYTHOS precision: extra meta decisiveness required during sure cold-start period (default: 0.06)")
    parser.add_argument("--mythos-leverage-edge-buffer", type=float, default=0.003,
                        help="MYTHOS leverage policy: edge buffer above min edge threshold for leverage activation (default: 0.003)")
    parser.add_argument("--mythos-leverage-confidence-buffer", type=float, default=0.04,
                        help="MYTHOS leverage policy: confidence buffer above min confidence for leverage activation (default: 0.04)")
    parser.add_argument("--mythos-leverage-conviction-buffer", type=float, default=0.04,
                        help="MYTHOS leverage policy: conviction buffer above score threshold for leverage activation (default: 0.04)")
    parser.add_argument("--mythos-leverage-recent-window", type=int, default=120,
                        help="MYTHOS leverage policy: rolling window for sure+leveraged quality checks (default: 120)")
    parser.add_argument("--mythos-leverage-recent-min-trades", type=int, default=20,
                        help="MYTHOS leverage policy: min sure+leveraged trades before strict leverage quality checks (default: 20)")
    parser.add_argument("--mythos-leverage-recent-min-hit-rate", type=float, default=0.53,
                        help="MYTHOS leverage policy: minimum recent sure+leveraged hit-rate (default: 0.53)")
    parser.add_argument("--mythos-leverage-recent-min-expectancy", type=float, default=0.05,
                        help="MYTHOS leverage policy: minimum recent sure+leveraged expectancy-R (default: 0.05)")
    parser.add_argument("--mythos-leverage-policy-window", type=int, default=160,
                        help="MYTHOS leverage policy: contextual policy window for leverage decisions (default: 160)")
    parser.add_argument("--mythos-leverage-policy-min-trades", type=int, default=24,
                        help="MYTHOS leverage policy: min trades for contextual policy reliability (default: 24)")
    parser.add_argument("--mythos-leverage-policy-min-hit-rate", type=float, default=0.54,
                        help="MYTHOS leverage policy: minimum blended hit-rate for leverage policy approval (default: 0.54)")
    parser.add_argument("--mythos-leverage-policy-min-expectancy", type=float, default=0.06,
                        help="MYTHOS leverage policy: minimum blended expectancy-R for leverage policy approval (default: 0.06)")
    parser.add_argument("--mythos-leverage-policy-context-weight", type=float, default=0.60,
                        help="MYTHOS leverage policy: weight on context score vs realized leverage history (default: 0.60)")
    parser.add_argument("--mythos-leverage-policy-cold-start-conviction-extra", type=float, default=0.08,
                        help="MYTHOS leverage policy: extra conviction needed before leverage during policy cold-start (default: 0.08)")
    parser.add_argument("--mythos-leverage-net-edge-floor", type=float, default=0.002,
                        help="MYTHOS leverage policy: minimum estimated net edge (after execution cost) to allow leverage boost (default: 0.002)")
    parser.add_argument("--mythos-leverage-side-policy-enable", dest="mythos_leverage_side_policy_enable", action="store_true",
                        help="MYTHOS leverage policy: require side-specific recent quality before leverage boost (default: enabled)")
    parser.add_argument("--mythos-no-leverage-side-policy-enable", dest="mythos_leverage_side_policy_enable", action="store_false",
                        help="MYTHOS leverage policy: disable side-specific leverage quality gating")
    parser.set_defaults(mythos_leverage_side_policy_enable=True)
    parser.add_argument("--mythos-leverage-side-min-trades", type=int, default=12,
                        help="MYTHOS leverage policy: minimum recent side trades before strict side leverage gating (default: 12)")
    parser.add_argument("--mythos-leverage-side-min-hit-rate", type=float, default=0.52,
                        help="MYTHOS leverage policy: minimum side-specific recent hit-rate to allow leverage boost (default: 0.52)")
    parser.add_argument("--mythos-leverage-side-min-expectancy", type=float, default=0.03,
                        help="MYTHOS leverage policy: minimum side-specific recent expectancy-R to allow leverage boost (default: 0.03)")
    parser.add_argument("--mythos-execution-fee-bps", type=float, default=4.0,
                        help="MYTHOS net intelligence: execution fee in bps applied in per-trade net-R accounting (default: 4.0)")
    parser.add_argument("--mythos-execution-slippage-bps", type=float, default=2.0,
                        help="MYTHOS net intelligence: slippage in bps applied in per-trade net-R accounting (default: 2.0)")
    parser.add_argument("--mythos-execution-cost-cap-r", type=float, default=0.35,
                        help="MYTHOS net intelligence: cap on deducted execution cost per trade in R-units (default: 0.35)")
    parser.add_argument("--mythos-bayes-quality-enable", dest="mythos_bayes_quality_enable", action="store_true",
                        help="MYTHOS Bayesian gate: enable online side/regime quality gating (default: enabled)")
    parser.add_argument("--mythos-no-bayes-quality-enable", dest="mythos_bayes_quality_enable", action="store_false",
                        help="MYTHOS Bayesian gate: disable online side/regime quality gating")
    parser.set_defaults(mythos_bayes_quality_enable=True)
    parser.add_argument("--mythos-bayes-quality-warmup-trades", type=int, default=20,
                        help="MYTHOS Bayesian gate: warmup trade count before strict quality rejects (default: 20)")
    parser.add_argument("--mythos-bayes-quality-decay", type=float, default=0.995,
                        help="MYTHOS Bayesian gate: EMA-style decay for online side/regime quality memory (default: 0.995)")
    parser.add_argument("--mythos-bayes-quality-prior-alpha", type=float, default=2.0,
                        help="MYTHOS Bayesian gate: beta prior alpha for hit-rate estimate (default: 2.0)")
    parser.add_argument("--mythos-bayes-quality-prior-beta", type=float, default=2.0,
                        help="MYTHOS Bayesian gate: beta prior beta for hit-rate estimate (default: 2.0)")
    parser.add_argument("--mythos-bayes-quality-regime-weight", type=float, default=0.45,
                        help="MYTHOS Bayesian gate: weight on regime-conditional quality vs side-global quality (default: 0.45)")
    parser.add_argument("--mythos-bayes-quality-min-win-prob", type=float, default=0.50,
                        help="MYTHOS Bayesian gate: minimum adjusted side win probability before reject pressure (default: 0.50)")
    parser.add_argument("--mythos-bayes-quality-min-expectancy", type=float, default=-0.01,
                        help="MYTHOS Bayesian gate: minimum adjusted side expectancy-R before reject pressure (default: -0.01)")
    parser.add_argument("--mythos-bayes-quality-edge-scale", type=float, default=0.22,
                        help="MYTHOS Bayesian gate: edge contribution scale to adjusted win-probability (default: 0.22)")
    parser.add_argument("--mythos-bayes-quality-confidence-scale", type=float, default=0.10,
                        help="MYTHOS Bayesian gate: confidence contribution scale to adjusted win-probability (default: 0.10)")
    parser.add_argument("--mythos-bayes-quality-uncertainty-scale", type=float, default=0.18,
                        help="MYTHOS Bayesian gate: uncertainty penalty scale on adjusted win-probability (default: 0.18)")
    parser.add_argument("--mythos-bayes-quality-reject-margin", type=float, default=0.05,
                        help="MYTHOS Bayesian gate: reject margin below min thresholds before skipping trade (default: 0.05)")
    parser.add_argument("--mythos-nonconformity-enable", dest="mythos_nonconformity_enable", action="store_true",
                        help="MYTHOS nonconformity gate: enable selective abstention on outlier decision contexts (default: enabled)")
    parser.add_argument("--mythos-no-nonconformity-enable", dest="mythos_nonconformity_enable", action="store_false",
                        help="MYTHOS nonconformity gate: disable selective abstention on outlier decision contexts")
    parser.set_defaults(mythos_nonconformity_enable=True)
    parser.add_argument("--mythos-nonconformity-warmup-trades", type=int, default=24,
                        help="MYTHOS nonconformity gate: warmup trades before nonconformity rejects can activate (default: 24)")
    parser.add_argument("--mythos-nonconformity-window", type=int, default=160,
                        help="MYTHOS nonconformity gate: rolling winner reference window size (default: 160)")
    parser.add_argument("--mythos-nonconformity-quantile", type=float, default=0.86,
                        help="MYTHOS nonconformity gate: accepted winner-score quantile ceiling (default: 0.86)")
    parser.add_argument("--mythos-nonconformity-margin", type=float, default=0.03,
                        help="MYTHOS nonconformity gate: additive margin above winner quantile threshold (default: 0.03)")
    parser.add_argument("--mythos-nonconformity-min-winners", type=int, default=16,
                        help="MYTHOS nonconformity gate: minimum winning references before strict filtering (default: 16)")
    parser.add_argument("--mythos-nonconformity-weight-uncertainty", type=float, default=0.36,
                        help="MYTHOS nonconformity gate: uncertainty component weight in outlier score (default: 0.36)")
    parser.add_argument("--mythos-nonconformity-weight-confidence", type=float, default=0.22,
                        help="MYTHOS nonconformity gate: inverse-confidence component weight in outlier score (default: 0.22)")
    parser.add_argument("--mythos-nonconformity-weight-edge", type=float, default=0.20,
                        help="MYTHOS nonconformity gate: inverse-edge component weight in outlier score (default: 0.20)")
    parser.add_argument("--mythos-nonconformity-weight-meta", type=float, default=0.14,
                        help="MYTHOS nonconformity gate: inverse-meta-decisiveness component weight in outlier score (default: 0.14)")
    parser.add_argument("--mythos-nonconformity-weight-analog", type=float, default=0.08,
                        help="MYTHOS nonconformity gate: inverse-analog-support component weight in outlier score (default: 0.08)")
    parser.add_argument("--mythos-nonconformity-override-conviction", type=float, default=0.88,
                        help="MYTHOS nonconformity gate: override conviction threshold for exceptionally strong trades (default: 0.88)")
    parser.add_argument("--mythos-nonconformity-override-edge-buffer", type=float, default=0.003,
                        help="MYTHOS nonconformity gate: edge buffer above base floor for override (default: 0.003)")
    parser.add_argument("--mythos-nonconformity-override-confidence-buffer", type=float, default=0.04,
                        help="MYTHOS nonconformity gate: confidence buffer above base floor for override (default: 0.04)")
    parser.add_argument("--mythos-nonconformity-target-reject-rate", type=float, default=0.48,
                        help="MYTHOS nonconformity adaptive: target reject rate for dynamic threshold relaxation (default: 0.48)")
    parser.add_argument("--mythos-nonconformity-reject-tolerance", type=float, default=0.12,
                        help="MYTHOS nonconformity adaptive: tolerance above target reject-rate before relaxation (default: 0.12)")
    parser.add_argument("--mythos-nonconformity-adaptive-relax", type=float, default=0.16,
                        help="MYTHOS nonconformity adaptive: relaxation gain when reject-rate overshoots target (default: 0.16)")
    parser.add_argument("--mythos-nonconformity-adaptive-max-relax", type=float, default=0.18,
                        help="MYTHOS nonconformity adaptive: max additional threshold relaxation (default: 0.18)")
    parser.add_argument("--mythos-nonconformity-soft-override-margin", type=float, default=0.04,
                        help="MYTHOS nonconformity adaptive: soft override margin for very strong conviction/context (default: 0.04)")
    parser.add_argument("--mythos-counterfactual-target-reject-rate", type=float, default=0.70,
                        help="MYTHOS counterfactual adaptive: target reject rate for dynamic relaxation (default: 0.70)")
    parser.add_argument("--mythos-counterfactual-reject-tolerance", type=float, default=0.10,
                        help="MYTHOS counterfactual adaptive: tolerance above target reject-rate before relaxation (default: 0.10)")
    parser.add_argument("--mythos-counterfactual-adaptive-relax", type=float, default=0.35,
                        help="MYTHOS counterfactual adaptive: relaxation gain when reject-rate overshoots target (default: 0.35)")
    parser.add_argument("--mythos-counterfactual-adaptive-min-adv-floor", type=float, default=0.25,
                        help="MYTHOS counterfactual adaptive: minimum retained fraction of base min-advantage under relaxation (default: 0.25)")
    parser.add_argument("--mythos-short-boost-enable", action="store_true", default=True,
                        help="MYTHOS adaptive: enable short-side edge boost when short side outperforms (default: enabled)")
    parser.add_argument("--mythos-no-short-boost-enable", dest="mythos_short_boost_enable", action="store_false",
                        help="MYTHOS adaptive: disable short-side outperformance boost")
    parser.add_argument("--mythos-short-boost-window", type=int, default=96,
                        help="MYTHOS adaptive: side performance window used for short outperformance boost (default: 96)")
    parser.add_argument("--mythos-short-boost-min-trades", type=int, default=20,
                        help="MYTHOS adaptive: minimum side trades in window before short boost can activate (default: 20)")
    parser.add_argument("--mythos-short-boost-threshold-r", type=float, default=0.08,
                        help="MYTHOS adaptive: required short expectancy edge over long to activate boost (default: 0.08)")
    parser.add_argument("--mythos-short-boost-edge", type=float, default=0.004,
                        help="MYTHOS adaptive: additive edge boost when short side outperforms (default: 0.004)")
    parser.add_argument("--mythos-short-boost-confidence", type=float, default=0.03,
                        help="MYTHOS adaptive: confidence boost when short side outperforms (default: 0.03)")
    parser.add_argument("--mythos-meta-bootstrap-samples", type=int, default=1024,
                        help="MYTHOS v7: bootstrap samples from training analog memory to pre-warm meta learner (default: 1024)")
    parser.add_argument("--mythos-meta-bootstrap-epochs", type=int, default=2,
                        help="MYTHOS v7: bootstrap passes over synthetic meta warmup samples (default: 2)")
    parser.add_argument("--v5-w-ret", type=float, default=6.0,
                        help="v5 weight for ret_h NLL loss (default: 6.0 — doubled from 3.0 to push return "
                             "signal from 2.4%% to ~67%% of gradient budget; Task #58)")
    parser.add_argument("--v5-w-mfe", type=float, default=0.15,
                        help="v5 weight for MFE Huber loss (default: 0.15 — reduced from 1.0 so MFE "
                             "is auxiliary; Task #58)")
    parser.add_argument("--v5-w-mae", type=float, default=0.15,
                        help="v5 weight for MAE Huber loss (default: 0.15 — reduced from 1.0 so MAE "
                             "is auxiliary; Task #58)")
    parser.add_argument("--v5-w-action", type=float, default=2.5,
                        help="v5 weight for action CE loss (default: 2.5 — raised from 0.5 to properly train direction head)")
    parser.add_argument("--v5-w-barrier", type=float, default=0.25,
                        help="v5 weight for barrier CE loss (default: 0.25)")
    parser.add_argument("--v5-w-regime", type=float, default=0.1,
                        help="v5 weight for regime CE loss (default: 0.1)")
    parser.add_argument("--v5-sigma-spread-reg", type=float, default=0.1,
                        help="v5 penalty on sigma>threshold to prevent NLL collapse (default: 0.1, set 0 to disable)")
    parser.add_argument("--v5-sigma-reg-threshold", type=float, default=0.40,
                        help="v5 sigma threshold above which penalty fires (default: 0.40; old hardcoded 1.5 "
                             "never fired at typical sigma=0.607 — Task #58)")
    parser.add_argument("--v5-phase1-epochs", type=int, default=50,
                        help="v5 two-phase curriculum: epochs in Phase 1 (return-only, no MFE/MAE/action "
                             "gradient); default: 50. Set 0 to disable Phase 1. Task #58")
    parser.add_argument("--v5-atr-normalize-risk-heads", action="store_true", default=True,
                        help="[API stub — no-op] mfe_R and mae_R are already R-units from build_v5_targets "
                             "(price_delta/ATR14). Dividing by ATR again would be price_delta/ATR^2 (unit error). "
                             "This flag is preserved for API compatibility and future use. Task #58")
    parser.add_argument("--v5-no-atr-normalize-risk-heads", dest="v5_atr_normalize_risk_heads",
                        action="store_false",
                        help="[API stub — no-op] See --v5-atr-normalize-risk-heads. Task #58")
    parser.add_argument("--v5-dynamic-action-labels", action="store_true", default=False,
                        help="v5 use mu_R>threshold to dynamically flip HOLD→LONG/SHORT action labels; "
                             "default: False (use dataset labels as-is). Task #58")
    parser.add_argument("--v5-loss-warmup-epochs", type=int, default=0,
                        help="v5 loss warmup: number of epochs with scaled weights (default: 0=disabled)")
    parser.add_argument("--v5-loss-warmup-ret-mult", type=float, default=1.0,
                        help="v5 loss warmup: multiplier for w_ret/mfe/mae during warmup epochs (default: 1.0)")
    parser.add_argument("--v5-loss-warmup-action-mult", type=float, default=1.0,
                        help="v5 loss warmup: multiplier for w_action during warmup epochs (default: 1.0)")
    parser.add_argument("--v5-score-lambda", type=float, default=0.30,
                        help="v5 downside penalty lambda in score formula (default: 0.30). "
                             "Task #56 A1: lowered 0.50→0.30. Break-even p_side drops from 0.333 to 0.231, "
                             "allowing ~88%% of bars to pass vs ~57%% at 0.50.")
    parser.add_argument("--v5-risk-proxy", type=str, default="mae", choices=["mae", "sigma"],
                        help="v5 risk denominator in score: 'mae' or 'sigma' (default: mae)")
    parser.add_argument("--v5-hold-target", type=float, default=0.30,
                        help="v5 target HOLD fraction for adaptive deadzone (default: 0.30)")
    parser.add_argument("--v5-mfe-min", type=float, default=0.05,
                        help="v5 min MFE in R-units for non-HOLD label (default: 0.05)")
    parser.add_argument("--v5-cand-warmup", type=int, default=3,
                        help="v5 epochs before enabling candidate mask in sweep (default: 3)")
    parser.add_argument("--v5-barrier-mode", type=str, default="fixed",
                        choices=["fixed", "oracle", "learnable"],
                        help="v5 barrier mode: fixed (single preset), oracle (hindsight, research), learnable (default: fixed)")

    parser.add_argument("--v5-sigma-max", type=float, default=1.0,
                        help="v5 quality gate: max predicted sigma (default: 1.0)")
    parser.add_argument("--v5-mae-max", type=float, default=1.0,
                        help="v5 quality gate: max predicted MAE in R-units (default: 1.0)")
    parser.add_argument("--v5-muR-min", type=float, default=0.05,
                        help="v5 quality gate: min |mu_R| edge (default: 0.05)")
    parser.add_argument("--v5-p-trade-min", type=float, default=0.40,
                        help="v5 quality gate: min p_trade = max(p_long, p_short) (default: 0.40)")
    parser.add_argument("--v5-enable-calib", action="store_true", default=False,
                        help="v5 enable 10-bin ECE calibration logging (default: disabled)")

    parser.add_argument("--v5-sweep-objective", choices=['quality', 'legacy'], default='quality',
                        help="v5 threshold sweep selection objective. "
                             "'quality' (default): composite=expect×sqrt(N), no TPD priority, PF>=1.10. "
                             "Picks the slice with the highest statistically significant per-trade edge "
                             "regardless of trade frequency. "
                             "'legacy': composite=expect×sharpe with TPD-window slice prioritized over "
                             "global best — historical default that crowned noisy high-TPD slices "
                             "(~0.05 ER) over high-quality low-TPD percentile slices (~0.30 ER).")
    parser.add_argument("--v5-target-tpd", type=float, default=6.5,
                        help="v5 TPD controller: target trades per day (default: 6.5)")
    parser.add_argument("--v5-tpd-tol", type=float, default=1.5,
                        help="v5 TPD controller: tolerance ± around target (default: 1.5)")
    parser.add_argument("--v5-target-trades-per-day", type=float, default=None,
                        help="Override --v5-target-tpd with intuitive trades/day target (e.g. 3.5)")
    parser.add_argument("--v5-target-trades-per-day-band", type=float, default=None,
                        help="Override --v5-tpd-tol with intuitive band (e.g. 0.8)")
    parser.add_argument("--v5-thr-warmup-epochs", type=int, default=3,
                        help="v5 TPD controller: epochs before adapting threshold (default: 3)")
    parser.add_argument("--v5-thr-step-mult", type=float, default=0.10,
                        help="v5 TPD controller: step multiplier on score_std (default: 0.10)")
    parser.add_argument("--v5-score-threshold", type=float, default=None,
                        help="v5 TPD controller: initial score threshold (default: auto p90)")
    parser.add_argument("--v5-mae-cap", type=float, default=2.0,
                        help="v5 score penalty: clamp MAE to this cap (default: 2.0)")
    parser.add_argument("--v5-side-bal-weight", type=float, default=0.05,
                        help="3-class side-balance KL loss weight added to L_action. "
                             "T5 fix: default 0.05 (was 0.15). 3-class targets include HOLD so "
                             "all-HOLD collapse is penalised. Lower weight reduces HOLD-bias pressure.")
    parser.add_argument("--v5-action-entropy-weight", type=float, default=0.12,
                        help="v5 action head entropy regularisation weight (default: 0.12). "
                             "Task #56 A3: raised 0.10→0.12 for slightly stronger diversity push. "
                             "Maximises entropy of the mean batch action distribution, preventing "
                             "direction collapse (100%% LONG or 100%% SHORT). Set 0.0 to disable. "
                             "Values 0.05-0.20 are typical. Task #54.")
    parser.add_argument("--v5-chop-hold-target", type=float, default=0.20,
                        help="v5 KL target HOLD fraction for chop-regime bars (default: 0.20). "
                             "Task #56 B1: old hardcoded value was 0.35. With 87.7%% bars in chop, "
                             "HOLD=0.35 biased model toward HOLD collapse. "
                             "HOLD=0.20 with 0.40/0.40 L/S teaches directionality. Task #56 B1.")
    parser.add_argument("--v5-ret-mag-ce-weight", action="store_true", default=False,
                        help="v5 upweight CE loss by return magnitude: "
                             "weight = 1 + clip(|ret_R|/median_ret, 0, 4) * --v5-ret-mag-scale. "
                             "Gives bull/bear bars up to 9× more CE gradient vs chop bars. "
                             "Default: off (opt-in). Task #56 B2.")
    parser.add_argument("--v5-ret-mag-scale", type=float, default=1.0,
                        help="v5 return-magnitude CE weight scaling factor (default: 1.0, optimal: 2.0). "
                             "Active only when --v5-ret-mag-ce-weight is set. Task #56 B2.")
    parser.add_argument("--v5-side-specialist", type=str, default='none',
                        choices=['none', 'short', 'long'],
                        help="Task #67: train a SINGLE direction specialist. "
                             "'short' → relabels LONG→HOLD, KL targets push p_short; "
                             "'long'  → relabels SHORT→HOLD, KL targets push p_long; "
                             "'none'  → standard bidirectional training (default). "
                             "See also --v5-dual-specialist for both directions per fold.")
    parser.add_argument("--v5-dual-specialist", action="store_true", default=False,
                        help="Task #67: train SHORT specialist then LONG specialist per walk-forward fold. "
                             "Each fold trains two models; their forward-test reports are merged. "
                             "Checkpoints saved as best_v5_short_fold{N}.pt and best_v5_long_fold{N}.pt. "
                             "Overrides --v5-side-specialist inside the walk-forward loop.")

    # Task #69: LONG specialist signal quality improvements
    parser.add_argument("--v5-min-mu-r-long", type=float, default=-1e9,
                        help="Task #69: LONG specialist hard gate — block LONG trades where mu_R < threshold. "
                             "0.0 = agree-only mode (return head must predict positive return). "
                             "Simulation shows +12%% total R and prevents all losses in ha=0%% bear folds. "
                             "Default: -1e9 (disabled). Recommended: 0.0 with --v5-dual-specialist.")
    parser.add_argument("--v5-long-disagree-mult", type=float, default=1.0,
                        help="Task #69: LONG specialist soft disagree multiplier — reduce score of LONG "
                             "trades where mu_R < 0 (return head disagrees with action head). "
                             "0.3 = 70%% score penalty, pushes agree trades higher in threshold sweep. "
                             "Default: 1.0 (disabled). Use 0.3 with --v5-min-mu-r-long for dual-gate effect.")
    parser.add_argument("--v5-specialist-align-weight", type=float, default=0.0,
                        help="Task #69: alignment loss weight — during LONG specialist training, penalise "
                             "negative mu_R on bars where the action head predicts LONG (p_long.detach() * relu(-mu_R)). "
                             "Pushes return head to agree with action head over training. "
                             "Recommended: 0.5. Default: 0.0 (disabled).")

    parser.add_argument("--v5-train-end-date", type=str, default=None,
                        help="v5 time-based split: train on data before this date (YYYY-MM-DD)")
    parser.add_argument("--v5-test-start-date", type=str, default=None,
                        help="v5 time-based split: test on data from this date (YYYY-MM-DD)")
    parser.add_argument("--v5-test-end-date", type=str, default=None,
                        help="v5 time-based split: test on data until this date (YYYY-MM-DD, default: end of data)")
    parser.add_argument("--v5-forward-test", action="store_true", default=False,
                        help="v5: run forward test on test set after training (frozen decision layer)")
    parser.add_argument("--v5-freeze-decision", action="store_true", default=True,
                        help="v5 forward test: freeze decision layer (no TPD adaptation, default: True)")
    parser.add_argument("--v5-diagnostics", action="store_true", default=False,
                        help="v5: run leakage/overfitting diagnostics after training")
    parser.add_argument("--v5-feature-report", action="store_true", default=False,
                        help="v5: generate feature importance ranking (permutation) and correlation cleanup report after training. Outputs to checkpoints/v5_feature_report.json")
    parser.add_argument("--v5-mu-debias", action="store_true", default=False,
                        help="v5: per-symbol mu_R EMA debiasing in forward test (opt-in, default: False). Enable with --v5-mu-debias when the model shows persistent directional drift.")
    parser.add_argument("--v5-no-mu-debias", dest="v5_mu_debias", action="store_false",
                        help="v5: disable mu_R debiasing")
    parser.add_argument("--no-mu-debias", dest="v5_mu_debias", action="store_false",
                        help="alias for --v5-no-mu-debias: disable mu_R debiasing")
    parser.add_argument("--v5-mu-debias-alpha", type=float, default=0.01,
                        help="v5.3.1+: EMA alpha for mu_R debiasing (default: 0.01)")
    parser.add_argument("--v5-ema200-regime-gate", action="store_true", default=False,
                        help="v5: EMA200 regime gate - LONG only when close>EMA200, SHORT only when close<EMA200")
    parser.add_argument("--v5-score-side-mode", type=str, default="action_head",
                        choices=["action_head", "mu_sign"],
                        help="v5.0.6: side selection mode. 'action_head' (default) uses |mu_R| with p_long/p_short for direction (fixes zero-SHORT bug). 'mu_sign' is legacy mode where mu_R sign drives direction.")
    parser.add_argument("--v5-rr-weight", type=float, default=0.0,
                        help="v5.0.6: risk/reward ratio bonus weight. When >0, adds mfe/mae ratio bonus to score (default: 0.0)")
    parser.add_argument("--v5-weekly-loss-cap", type=float, default=None,
                        help="v5.0.7: weekly loss cap in R-units. When weekly cumulative R drops below this, stop trading for remainder of week (e.g. -5.0). Default: None (disabled)")
    parser.add_argument("--v5-warmup-skip-bars", type=int, default=0,
                        help="v5.0.7: skip trades in first N bars of forward test window to avoid cold-start losses (e.g. 96 = 1 day). Default: 0 (disabled)")
    parser.add_argument("--v5-corr-block", action="store_true", default=True,
                        help="v5.0.8: enable smart cross-asset correlation blocking (default: on)")
    parser.add_argument("--v5-no-corr-block", dest="v5_corr_block", action="store_false",
                        help="v5.0.8: disable smart cross-asset correlation blocking")
    parser.add_argument("--v5-corr-window-days", type=int, default=30,
                        help="v5.0.8: rolling window in days for correlation computation (default: 30)")
    parser.add_argument("--v5-corr-thresh", type=float, default=0.70,
                        help="v5.0.8: abs correlation threshold to block entry (default: 0.70)")
    parser.add_argument("--v5-corr-same-side-only", action="store_true", default=True,
                        help="v5.0.8: only block when new trade matches open position side (default: on)")
    parser.add_argument("--v5-corr-any-side", dest="v5_corr_same_side_only", action="store_false",
                        help="v5.0.8: block correlated entries regardless of side")
    parser.add_argument("--v5-log-corr-matrix", action="store_true", default=True,
                        help="v5.0.8: log correlation matrix at fold end (default: on)")
    parser.add_argument("--v5-corr-max-block", type=int, default=5,
                        help="v5.4: max symbols one open position can block via correlation (default: 5, 0=unlimited)")

    parser.add_argument("--v5-adaptive-sizing", action="store_true", default=False,
                        help="v5.0.8+: enable adaptive position sizing via fractional Kelly criterion (default: off)")
    parser.add_argument("--v5-kelly-fraction", type=float, default=0.25,
                        help="v5.0.8+: Kelly fraction for position sizing — 0.25 = quarter-Kelly (default: 0.25)")
    parser.add_argument("--v5-max-size-mult", type=float, default=2.5,
                        help="v5.0.8+: maximum position size multiplier cap (default: 2.5)")
    parser.add_argument("--v5-min-size-mult", type=float, default=0.25,
                        help="v5.0.8+: minimum position size multiplier floor (default: 0.25)")

    parser.add_argument("--v5-regime-scaling", action="store_true", default=False,
                        help="v5.0.8+: enable dynamic risk scaling by regime — scales risk up in trending, down in choppy (default: off)")
    parser.add_argument("--v5-regime-bull-mult", type=float, default=1.5,
                        help="v5.0.8+: risk multiplier in favorable (bull/trending) regime (default: 1.5)")
    parser.add_argument("--v5-regime-bear-mult", type=float, default=0.5,
                        help="v5.0.8+: risk multiplier in hostile (bear/choppy) regime (default: 0.5)")
    parser.add_argument("--v5-regime-lookback", type=int, default=20,
                        help="v5.0.8+: lookback trades for rolling equity regime signal (default: 20)")

    parser.add_argument("--v5-daily-loss-cap", type=float, default=None,
                        help="v5.0.8+: daily loss cap in R-units. Stops trading for rest of day when hit (e.g. -3.0). Default: None (disabled)")
    parser.add_argument("--v5-trailing-equity-stop", type=float, default=None,
                        help="v5.0.8+: trailing equity stop in R-units. Pauses trading when equity drops this far from peak (e.g. 15.0). Default: None (disabled)")
    parser.add_argument("--v5-per-symbol-daily-r", type=float, default=None,
                        help="v5.0.8+: per-symbol daily R budget cap. Stops trading a symbol for rest of day when hit (e.g. -2.0). Default: None (disabled)")
    parser.add_argument("--v5-per-symbol-r-kill", type=float, default=None,
                        help="Per-symbol cumulative R kill switch. When a symbol's cumulative R drops below this floor (e.g. -8.0), stop trading it for the rest of the fold. Default: None (disabled)")
    parser.add_argument("--v5-per-symbol-threshold", action="store_true", default=False,
                        help="Enable per-symbol threshold sweep. Finds optimal threshold per symbol during training; symbols with no edge get threshold=inf (never traded). Default: off")
    parser.add_argument("--v5-short-oversample", action="store_true", default=False,
                        help="v5.3.0: oversample SHORT labels to reach min fraction of LONG+SHORT. Fixes persistent LONG bias in training data. Default: off")
    parser.add_argument("--v5-short-min-fraction", type=float, default=0.40,
                        help="v5.3.0: minimum SHORT fraction of LONG+SHORT after oversampling (default: 0.40 = 40%%). "
                             "Task #56 B3: raised 0.35→0.40 so the model develops a stronger p_short signal.")
    parser.add_argument("--v5-ema200-soft-mult", type=float, default=None,
                        help="v5.3.0: EMA200 soft gate multiplier. When set, replaces hard EMA200 block with size reduction (e.g. 0.50 = half size). Default: None (hard block)")
    parser.add_argument("--v5-per-side-threshold", action="store_true", default=False,
                        help="v5.3.0: per-side threshold in forward test. Uses separate LONG and SHORT thresholds from per-symbol sweep. Requires --v5-per-symbol-threshold. Default: off")
    parser.add_argument("--v5-min-threshold", type=float, default=None,
                        help="v5.0.8+: minimum score threshold floor. Prevents calibrated threshold from dropping too low (e.g. 0.05). Default: None (disabled)")
    parser.add_argument("--v5-per-sym-no-edge-fallback", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="v5.1+: when ALL per-symbol thresholds are inf (HIGH_BAR), fall back to global effective_threshold instead of producing 0 trades. Default: True")
    parser.add_argument("--v5-max-threshold", type=float, default=None,
                        help="v5.0.9+: maximum score threshold ceiling. Caps calibrated threshold from above to prevent sweep from setting it too high (e.g. 0.10). Default: None (disabled)")
    parser.add_argument("--v5-min-threshold-pct", type=float, default=None,
                        help="v5.0.8+: adaptive minimum threshold as percentile of test score distribution (e.g. 70 = use 70th percentile as floor). Adapts to each fold's score range. Default: None (disabled)")
    parser.add_argument("--v5-max-trades-per-day", type=int, default=8,
                        help="v5.0.8+: maximum trades per day across all symbols. Blocks further entries once reached (e.g. 6). Default: 8")

    parser.add_argument("--v5-trailing-sl", action="store_true", default=False,
                        help="v5.0.8+: enable trailing stop-loss. Moves SL to breakeven then trails behind best price. Default: disabled")
    parser.add_argument("--v5-trail-activation", type=float, default=1.5,
                        help="v5.0.8+: ATR multiples of favorable move before trailing SL activates (default: 1.5)")
    parser.add_argument("--v5-trail-distance", type=float, default=1.0,
                        help="v5.0.8+: ATR multiples behind best price for trailing stop (default: 1.0)")
    parser.add_argument("--v5-allow-runner", action="store_true", default=False,
                        help="v5.0.8+: don't exit at TP, let winning trades run with tighter trail. Default: disabled")

    parser.add_argument("--v5-conviction-sizing", action="store_true", default=False,
                        help="v5.0.8+: enable score-tiered position sizing. Top signals get larger size, weak signals get smaller. Default: disabled")
    parser.add_argument("--v5-conviction-top-pct", type=float, default=5.0,
                        help="v5.0.8+: top percentile tier for conviction sizing (default: 5.0)")
    parser.add_argument("--v5-conviction-top-mult", type=float, default=2.5,
                        help="v5.0.8+: size multiplier for top tier trades (default: 2.5)")
    parser.add_argument("--v5-conviction-high-pct", type=float, default=20.0,
                        help="v5.0.8+: high percentile tier boundary (default: 20.0)")
    parser.add_argument("--v5-conviction-high-mult", type=float, default=1.5,
                        help="v5.0.8+: size multiplier for high tier trades (default: 1.5)")
    parser.add_argument("--v5-conviction-conf-thresh", type=float, default=0.65,
                        help="v5.0.8+: directional confidence threshold for boost (default: 0.65)")
    parser.add_argument("--v5-conviction-conf-boost", type=float, default=1.3,
                        help="v5.0.8+: confidence boost multiplier when p_dir exceeds threshold (default: 1.3)")

    parser.add_argument("--v5-adx-gate", action="store_true", default=False,
                        help="v5.0.9+: enable ADX regime gate — blocks trades when ADX < min (choppy market)")
    parser.add_argument("--v5-adx-period", type=int, default=14,
                        help="v5.0.9+: ADX indicator period (default: 14)")
    parser.add_argument("--v5-adx-min", type=float, default=18.0,
                        help="v5.0.9+: minimum ADX value to allow trades (default: 18.0)")
    parser.add_argument("--v5-adx-exception-top-pct", type=float, default=10.0,
                        help="v5.0.9+: top N%% of scores bypass ADX gate (default: 10.0)")

    parser.add_argument("--v5-ultra-conviction", action="store_true", default=False,
                        help="v5.0.9+: enable ultra-conviction risk tier — up to 5%% risk for rare high-conviction setups")
    parser.add_argument("--v5-ultra-risk-cap", type=float, default=0.05,
                        help="v5.0.9+: max risk per ultra trade as fraction (default: 0.05 = 5%%)")
    parser.add_argument("--v5-ultra-score-pct", type=float, default=0.95,
                        help="v5.0.9+: score percentile gate for ultra tier (default: 0.95 = top 5%%)")
    parser.add_argument("--v5-ultra-adx-min", type=float, default=25.0,
                        help="v5.0.9+: minimum ADX for ultra tier (default: 25.0)")
    parser.add_argument("--v5-ultra-edge-min", type=float, default=0.03,
                        help="v5.0.9+: minimum expected edge for ultra tier (default: 0.03)")
    parser.add_argument("--v5-ultra-dd-max", type=float, default=0.10,
                        help="v5.0.9+: max drawdown before ultra tier is suspended (default: 0.10 = 10%%)")
    parser.add_argument("--v5-ultra-max-per-day", type=int, default=1,
                        help="v5.0.9+: max ultra trades per day (default: 1)")
    parser.add_argument("--v5-ultra-mult", type=float, default=3.0,
                        help="v5.0.9+: size multiplier for ultra trades (default: 3.0)")

    parser.add_argument("--v5-ddt-enable", action="store_true", default=False,
                        help="v5 DDT: enable Drawdown-Adaptive Throttle (gradually tightens entry & sizing during drawdowns)")
    parser.add_argument("--v5-ddt-lookback-trades", type=int, default=60,
                        help="v5 DDT: rolling window of closed trades for throttle (default: 60)")
    parser.add_argument("--v5-ddt-bad-rollr", type=float, default=6.0,
                        help="v5 DDT: rolling sum R <= -bad_rollr => max throttle (default: 6.0)")
    parser.add_argument("--v5-ddt-thr-k", type=float, default=0.60,
                        help="v5 DDT: threshold multiplier thr_eff = thr_base * (1 + thr_k*throttle) (default: 0.60)")
    parser.add_argument("--v5-ddt-thr-min", type=float, default=0.08,
                        help="v5 DDT: minimum effective threshold (default: 0.08)")
    parser.add_argument("--v5-ddt-thr-max", type=float, default=0.25,
                        help="v5 DDT: maximum effective threshold hard cap (default: 0.25)")
    parser.add_argument("--v5-ddt-size-k", type=float, default=0.70,
                        help="v5 DDT: size dampening size_mult = 1 - size_k*throttle (default: 0.70)")
    parser.add_argument("--v5-ddt-min-size-mult", type=float, default=0.25,
                        help="v5 DDT: minimum position size multiplier (default: 0.25)")
    parser.add_argument("--v5-ddt-alpha-down", type=float, default=0.30,
                        help="v5 DDT: EMA alpha for tightening (fast) (default: 0.30)")
    parser.add_argument("--v5-ddt-alpha-up", type=float, default=0.05,
                        help="v5 DDT: EMA alpha for loosening (slow) (default: 0.05)")
    parser.add_argument("--v5-ddt-warmup-trades", type=int, default=20,
                        help="v5 DDT: don't throttle until >= N closed trades (default: 20)")

    parser.add_argument("--v5-multi-regime", action="store_true", default=False,
                        help="v5.0.9+: enable multi-regime classifier (trending_up/down, choppy, high_vol, low_vol)")
    parser.add_argument("--v5-regime-adx-trending", type=float, default=25.0,
                        help="v5.0.9+: ADX threshold for trending regime (default: 25.0)")
    parser.add_argument("--v5-regime-adx-choppy", type=float, default=20.0,
                        help="v5.0.9+: ADX below this = choppy regime (default: 20.0)")
    parser.add_argument("--v5-regime-atr-high-vol", type=float, default=1.3,
                        help="v5.0.9+: ATR ratio above this = high_vol regime (default: 1.3)")
    parser.add_argument("--v5-regime-atr-low-vol", type=float, default=0.7,
                        help="v5.0.9+: ATR ratio below this = low_vol regime (default: 0.7)")
    parser.add_argument("--v5-regime-atr-window", type=int, default=96,
                        help="v5.0.9+: rolling ATR window for regime classification (default: 96 bars)")
    parser.add_argument("--v5-regime-ema-slope-window", type=int, default=10,
                        help="v5.0.9+: EMA200 slope lookback bars (default: 10)")
    parser.add_argument("--v5-regime-ema-buffer", type=float, default=0.005,
                        help="v5.0.9+: price vs EMA200 buffer for trend direction (default: 0.005 = 0.5%%)")

    parser.add_argument("--v5-edge-first", action="store_true", default=False,
                        help="v5.1: Edge-First mode - only take trades with real edge (filters low-edge noise)")
    parser.add_argument("--v5-edge-min", type=float, default=0.03,
                        help="v5.1: minimum edge score to consider a trade (default: 0.03)")
    parser.add_argument("--v5-edge-pct-floor", type=int, default=70,
                        help="v5.1: percentile floor for edge scores (default: 70)")
    parser.add_argument("--v5-edge-topn-per-day", type=int, default=4,
                        help="v5.1: max trades per day under edge-first mode (default: 4)")
    parser.add_argument("--v5-regime-side-map", type=str, default=None,
                        help="v5.1: regime-conditional side filtering, e.g. 'trending_up=LONG,trending_down=SHORT,choppy=NONE'")
    parser.add_argument("--v5-size-floor", type=float, default=0.0,
                        help="v5.1: minimum sizing multiplier floor (only when not in drawdown, default: 0 = disabled)")

    parser.add_argument("--v5-head-disagree-gate", action="store_true", default=False,
                        help="v5.2: block trades when model heads disagree (action side vs mu_R sign)")
    parser.add_argument("--slippage-base-bps", type=float, default=0.0,
                        help="v5.2: slippage deduction in basis points before edge calc (default: 0 = disabled)")
    parser.add_argument("--v5-per-symbol-cooldown", action="store_true", default=True,
                        help="v5.4: per-symbol cooldown instead of global (default: on)")
    parser.add_argument("--v5-global-cooldown", dest="v5_per_symbol_cooldown", action="store_false",
                        help="v5.4: revert to global cooldown across all symbols")
    parser.add_argument("--v5-regime-soft", action="store_true", default=True,
                        help="v5.5: soft regime gate — disagree trades get reduced size instead of blocked (default: on)")
    parser.add_argument("--v5-regime-hard", dest="v5_regime_soft", action="store_false",
                        help="v5.5: revert to hard regime blocking")
    parser.add_argument("--v5-regime-disagree-mult", type=float, default=0.3,
                        help="v5.5: size multiplier when regime disagrees with side (default: 0.3)")
    parser.add_argument("--v5-regime-none-mult", type=float, default=0.2,
                        help="v5.5: size multiplier when regime is NONE (default: 0.2)")
    parser.add_argument("--v5-per-symbol-soft-kill", action="store_true", default=True,
                        help="v5.5: soft kill — bad symbols get reduced size instead of being killed (default: on)")
    parser.add_argument("--v5-per-symbol-hard-kill", dest="v5_per_symbol_soft_kill", action="store_false",
                        help="v5.5: revert to permanent symbol kill")
    parser.add_argument("--v5-edge-topn-soft", action="store_true", default=True,
                        help="v5.5: soft edge topn — trades above cap get decaying size instead of blocked (default: on)")
    parser.add_argument("--v5-edge-topn-hard", dest="v5_edge_topn_soft", action="store_false",
                        help="v5.5: revert to hard edge topn cap")
    parser.add_argument("--v5-edge-topn-decay", type=float, default=0.7,
                        help="v5.5: decay factor per excess trade above topn cap (default: 0.7)")
    parser.add_argument("--v5-sigma-discount", action="store_true", default=True,
                        help="v5.3: apply sigma sharpness multiplier 1/(1+sigma) to scores (default: enabled)")
    parser.add_argument("--v5-no-sigma-discount", action="store_true", default=False,
                        help="v5.3: disable sigma discount")
    parser.add_argument("--v5-min-p-side", type=float, default=0.45,
                        help="v5.3: minimum p_side conviction to allow a trade (default: 0.45, 0=disabled)")
    parser.add_argument("--v5-min-p-short", type=float, default=0.0,
                        help="v5.3: minimum p_short to allow SHORT trades (default: 0=disabled, e.g. 0.55)")
    parser.add_argument("--v5-side-aware-scoring", action="store_true", default=False,
                        help="v5.3: require mu_R direction to agree with side (shorts need mu_R<0, longs need mu_R>0)")
    parser.add_argument("--v5-mae-asym-weight", type=float, default=2.0,
                        help="v5.3: asymmetric MAE loss penalty for underestimation (default: 2.0, 1.0=symmetric)")

    parser.add_argument("--v5-soft-gate-floor", action="store_true", default=True,
                        help="v5.6: clamp soft gate mult to size_floor minimum so position sizers can amplify (default: on)")
    parser.add_argument("--v5-no-soft-gate-floor", dest="v5_soft_gate_floor", action="store_false",
                        help="v5.6: disable soft gate floor clamping")
    parser.add_argument("--v5-weekly-cap-dynamic", action="store_true", default=False,
                        help="v5.6: scale weekly cap based on rolling 4-week performance (default: off)")
    parser.add_argument("--v5-weekly-cap-scale", type=float, default=2.0,
                        help="v5.6: multiplier for weekly cap when rolling performance is positive (default: 2.0)")
    parser.add_argument("--v5-quality-gate", action="store_true", default=False,
                        help="v5.6: rolling quality gate — reduce sizing when action accuracy drops below threshold (default: off)")
    parser.add_argument("--v5-quality-gate-window", type=int, default=50,
                        help="v5.6: number of recent trades to track for quality gate (default: 50)")
    parser.add_argument("--v5-direction-balance-cap", action="store_true", default=False,
                        help="v5.6: reduce sizing on dominant direction when balance exceeds threshold (default: off)")
    parser.add_argument("--v5-direction-balance-threshold", type=float, default=0.75,
                        help="v5.6: direction balance threshold for 0.5x sizing reduction (default: 0.75)")
    parser.add_argument("--v5-rolling-er-gate", action="store_true", default=False,
                        help="Rolling E[R] gate: block a symbol when trailing E[R] over last N trades falls below min_er (default: off)")
    parser.add_argument("--v5-rolling-er-window", type=int, default=20,
                        help="Rolling E[R] gate: number of recent trades per symbol for E[R] computation (default: 20)")
    parser.add_argument("--v5-rolling-er-min", type=float, default=-0.05,
                        help="Rolling E[R] gate: minimum acceptable trailing E[R]; symbol blocked when below this (default: -0.05)")
    parser.add_argument("--v5-gate-mode", type=str, default="ref_magnitude",
                        choices=["ref_magnitude", "percentile_top15"],
                        help="v5.7: forward-test trade-selection gate strategy. "
                             "ref_magnitude (default): current adaptive ref-magnitude gate with relax loop. "
                             "percentile_top15: selects top-15%% of quality-masked scores each fold (diagnostic/experimental).")
    parser.add_argument("--v5-live-threshold", type=float, default=None,
                        help="Live mode v5_score threshold override (default: V5_SCORE_THRESHOLD=0.5). "
                             "Signals with score below this are blocked. Range: 0.1–5.0")
    parser.add_argument("--v5-live-mae-floor", type=float, default=0.5,
                        help="Live mode MAE floor to prevent score explosion in low-vol markets "
                             "(default: 0.5). Was 0.001 which caused scores of 5000+)")
    parser.add_argument("--v5-predictive-sltp", action="store_true", default=False,
                        help="Use model MFE/MAE head predictions to set dynamic SL/TP. "
                             "SL widens when model expects large adverse excursion. "
                             "TP widens when model expects large favorable excursion. "
                             "Never tightens SL below base (sl_mult×ATR). Recommended with V5+ models.")

    parser.add_argument("--v5-temp-scale", action="store_true", default=False,
                        help="v5.0.9+: enable post-training temperature scaling calibration")

    parser.add_argument("--v5-promote-metric", type=str, default="expectancy",
                        choices=["expectancy", "pf", "val_loss"],
                        help="v5.0.9+: metric for checkpoint promotion (default: expectancy)")

    parser.add_argument("--v5-staged-training", action="store_true", default=False,
                        help="v5.0.9+: enable staged training schedule (3 phases)")

    parser.add_argument("--v5-walk-forward", action="store_true", default=False,
                        help="v5: run walk-forward analysis with rolling train/test windows")
    parser.add_argument("--v5-wf-train-months", type=int, default=9,
                        help="v5 walk-forward: training window in months (default: 9). "
                             "Task #56 C2: lowered 12→9. 9mo gives 29 folds vs 23 over 3yr history, "
                             "more fold samples for better threshold calibration. "
                             "9mo still covers 2+ full crypto market cycles.")
    parser.add_argument("--v5-wf-test-months", type=int, default=1,
                        help="v5 walk-forward: test window in months (default: 1)")
    parser.add_argument("--v5-wf-threshold-ema", action="store_true", default=True,
                        help="v5.3.1+: carry-forward threshold EMA across walk-forward folds (default: True)")
    parser.add_argument("--v5-no-wf-threshold-ema", action="store_true", default=False,
                        help="v5.3.1+: disable threshold carry-forward EMA")
    parser.add_argument("--v5-wf-threshold-ema-alpha", type=float, default=0.5,
                        help="v5.3.1+: EMA blending weight for threshold carry-forward (default: 0.5 = 50%% new, 50%% prior)")
    parser.add_argument("--v5-wf-threshold-decay", type=float, default=0.5,
                        help="v5.5+: decay factor for threshold EMA on dead folds (0 trades). Range 0-1. Halves threshold per dead fold (default: 0.5)")
    parser.add_argument("--v5-min-trades", type=int, default=20,
                        help="v5.3.1+: minimum trades for valid fold. Below this → LOW_CONF, trades kept but threshold EMA skips (default: 20)")

    parser.add_argument("--v5-recency-weight", action="store_true", default=False,
                        help="v5.7+: exponential recency weighting — recent samples get higher loss weight (default: False)")
    parser.add_argument("--v5-recency-half-life", type=float, default=60,
                        help="v5.7+: half-life in days for recency decay. Data this many days old gets 50%% weight (default: 60). "
                             "Task #56 C1: lowered 90→60. Crypto regimes shift fast; 60-day half-life weights recent 2 months 2× more.")
    parser.add_argument("--v5-finetune-months", type=int, default=0,
                        help="v5.7+: fine-tune on last N months after main training (0=disabled, default: 0)")
    parser.add_argument("--v5-finetune-epochs", type=int, default=5,
                        help="v5.7+: epochs for fine-tuning phase (default: 5)")
    parser.add_argument("--v5-finetune-lr-mult", type=float, default=0.1,
                        help="v5.7+: LR multiplier for fine-tuning phase (default: 0.1)")
    parser.add_argument("--v5-wf-warm-start", action="store_true", default=False,
                        help="v5.7+: warm-start each walk-forward fold from previous fold model (default: False)")
    parser.add_argument("--v5-wf-warm-start-lr-mult", type=float, default=0.3,
                        help="v5.7+: LR multiplier for warm-start first epoch (default: 0.3, not yet used — reserved)")

    parser.add_argument("--multi-horizon", action="store_true", default=False,
                        help="Train multiple horizons (8,16,32) and select best per bar")
    parser.add_argument("--multi-horizons", type=str, default="8,16,32",
                        help="Comma-separated horizon bars (default: 8,16,32)")
    parser.add_argument("--cooldown-per-horizon", action="store_true", default=False,
                        help="Use separate cooldown per horizon instead of unified")

    parser.add_argument("--multi-preset", action="store_true", default=False,
                        help="Train with multiple barrier presets and select best per bar")
    parser.add_argument("--barrier-presets", type=str, default="tight,standard,wide,asymmetric",
                        help="Comma-separated barrier preset names (default: tight,standard,wide,asymmetric)")

    parser.add_argument("--money-score", action="store_true", default=False,
                        help="Use enhanced money-score formula with efficiency and regime weight")
    parser.add_argument("--kelly-sizing", action="store_true", default=False,
                        help="Enable Kelly-like position sizing")
    parser.add_argument("--daily-loss-limit", type=float, default=-3.0,
                        help="Daily loss limit in R-units (default: -3.0)")
    parser.add_argument("--max-concurrent", type=int, default=6,
                        help="Max concurrent trades (default: 6)")
    parser.add_argument("--max-symbol-exposure", type=int, default=3,
                        help="Max trades per symbol (default: 3)")

    parser.add_argument("--live", action="store_true",
                        help="Run continuous live multi-asset inference loop")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,LTCUSDT,NEARUSDT,PEPEUSDT,SUIUSDT,AAVEUSDT,ARBUSDT,DOTUSDT,MATICUSDT,FILUSDT,APTUSDT,OPUSDT",
                        help="Comma-separated symbols to monitor (default: all 20 symbols)")
    parser.add_argument("--interval", type=str, default="15m",
                        help="Signal timeframe interval (default: 15m)")
    parser.add_argument("--paper", action="store_true", default=False,
                        help="Paper mode — simulate positions + record trades (default: off)")
    parser.add_argument("--record-trades", action="store_true", default=False,
                        help="Enable trade recording (POST to /api/live/trade). Default: off unless --paper or --live")
    parser.add_argument("--execution-mode", type=str, default=None,
                        choices=["signal_only", "paper", "live"],
                        help="Explicit execution mode override (default: derived from --paper/--live flags)")
    parser.add_argument("--enter-threshold", type=float, default=0.85,
                        help="p_enter threshold for live signals (default: 0.85)")
    parser.add_argument("--max-pos-total", type=int, default=2,
                        help="Max total open positions (default: 2)")
    parser.add_argument("--max-pos-symbol", type=int, default=1,
                        help="Max open positions per symbol (default: 1)")
    parser.add_argument("--risk-cap-total", type=float, default=10.0,
                        help="Max total portfolio risk %% (default: 10.0)")
    parser.add_argument("--risk-cap-symbol", type=float, default=5.0,
                        help="Max per-symbol risk %% (default: 5.0)")
    parser.add_argument("--exec-tf", type=str, default="3m", choices=["1m", "3m", "5m"],
                        help="Execution timeframe for improved fills (default: 3m)")
    parser.add_argument("--exec-window-min", type=int, default=15,
                        help="Execution window in minutes (default: 15)")
    parser.add_argument("--pullback-atr", type=float, default=0.20,
                        help="Pullback ATR fraction for exec (default: 0.20)")
    parser.add_argument("--confirm-indicator", type=str, default="vwap", choices=["vwap", "ema20"],
                        help="Confirmation indicator for exec (default: vwap)")
    parser.add_argument("--allow-market-fallback", action="store_true", default=False,
                        help="Allow market entry if exec window expires (default: off)")
    parser.add_argument("--no-exec", action="store_true", default=False,
                        help="Disable lower-TF execution (enter at market on signal)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replay cached candles instead of fetching live data")
    parser.add_argument("--dry-run-candles", type=int, default=200,
                        help="Number of bars to replay in dry-run mode (default: 200)")
    parser.add_argument("--no-correlation-block", action="store_true", default=False,
                        help="[DEPRECATED: use --v5-no-corr-block] Disable same-direction correlation blocking for live mode (default: on)")
    parser.add_argument("--per-symbol-models", action="store_true", default=False,
                        help="Use per-symbol deployed models from checkpoints/deployed/{symbol}/")
    parser.add_argument("--enable-learning", action="store_true", default=False,
                        help="Enable scheduled retraining + safe promotion during live run")
    parser.add_argument("--retrain-hour", type=int, default=4,
                        help="UTC hour to trigger daily retrain (default: 4)")
    parser.add_argument("--retrain-interval", type=int, default=24,
                        help="Hours between retrains (default: 24)")
    parser.add_argument("--retrain-epochs", type=int, default=300,
                        help="Training epochs for scheduled retrain (default: 300)")
    parser.add_argument("--min-prauc", type=float, default=0.35,
                        help="Minimum PR-AUC for model promotion (default: 0.35)")
    parser.add_argument("--min-pf-net", type=float, default=1.05,
                        help="Minimum PF_net for model promotion (default: 1.05)")
    parser.add_argument("--min-profitable-regimes", type=int, default=3,
                        help="Minimum profitable regimes for promotion (default: 3)")
    parser.add_argument("--no-auto-promote", action="store_true", default=False,
                        help="Disable auto-promotion (train + evaluate only)")
    parser.add_argument("--no-geometry-sweep-retrain", action="store_true", default=False,
                        help="Skip geometry sweep during scheduled retrain")
    parser.add_argument("--limit-15m", type=int, default=800,
                        help="Number of 15m candles to fetch per symbol (default: 800, ~50 H4 bars)")
    parser.add_argument("--direct-htf", action="store_true", default=False,
                        help="Fetch 1H/4H candles directly from exchange instead of resampling")
    parser.add_argument("--verify-system", action="store_true", default=False,
                        help="Run system verification mode: N cycles of assertions on lane routing, CROSS, quota, payloads")
    parser.add_argument("--cycles", type=int, default=30,
                        help="Number of cycles for --verify-system/--verify-separation mode (default: 30)")

    args = parser.parse_args()
    profile_changes = _apply_mythos_profile_overrides(args)
    if getattr(args, "train_mythos", False) and profile_changes:
        log.info("[MYTHOS PROFILE] %s applied with %s overrides", args.mythos_profile, len(profile_changes))
        for name, old, new in profile_changes:
            log.info("[MYTHOS PROFILE]   %s: %s -> %s", name, old, new)

    print()
    print("=" * 60)
    print(f"  BTC FUTURES - ENTER QUALITY MODEL {SYSTEM_VERSION}")
    print("=" * 60)
    print()

    device = check_gpu()
    data_dir = Path("data_cache")

    if args.live:
        from portfolio import PortfolioManager
        from execution import ExecutionModule
        from live_runner import LiveRunner

        symbols = [s.strip().upper() for s in args.symbols.split(",")]

        live_tp = args.tp_mult
        live_sl = args.sl_mult
        live_threshold = args.enter_threshold
        live_cooldown = args.cooldown
        policy_source = "CLI defaults"

        policy_path = Path("checkpoints/best_policy.json")
        cli_policy_set = '--policy' in sys.argv or '--enter-threshold' in sys.argv
        if not cli_policy_set and policy_path.exists():
            try:
                with open(policy_path) as f:
                    bp = json.load(f)
                live_tp = bp.get('tp_mult', live_tp)
                live_sl = bp.get('sl_mult', live_sl)
                live_threshold = bp.get('threshold', live_threshold)
                live_cooldown = bp.get('cooldown', live_cooldown)
                pol_type = bp.get('policy_type', 'threshold')
                pol_val = bp.get('policy_value', live_threshold)
                pf_net = bp.get('metrics', {}).get('overall_pf_net', bp.get('overall_pf_net', '?'))
                e_net = bp.get('metrics', {}).get('overall_e_net', bp.get('overall_e_net', '?'))
                tpd = bp.get('metrics', {}).get('overall_tpd', bp.get('overall_tpd', '?'))
                pol_label = f"threshold:{pol_val}" if pol_type == 'threshold' else f"percentile:top{int(pol_val)}"
                policy_source = f"best_policy.json ({pol_label})"
                print(f"  Loaded BEST policy from {policy_path}")
                print(f"    Policy: {pol_label} (threshold={live_threshold:.4f})")
                print(f"    TP={live_tp}x SL={live_sl}x Cooldown={live_cooldown}")
                print(f"    PF_net={pf_net} E[net]={e_net} TPD={tpd}")
            except Exception as e:
                log.warning(f"Failed to load best_policy.json: {e}, using CLI defaults")

        if '--tp-mult' in sys.argv:
            live_tp = args.tp_mult
        if '--sl-mult' in sys.argv:
            live_sl = args.sl_mult
        if '--cooldown' in sys.argv:
            live_cooldown = args.cooldown

        print(f"  Policy source: {policy_source}")

        portfolio = PortfolioManager(
            max_positions_total=args.max_pos_total,
            max_positions_per_symbol=args.max_pos_symbol,
            risk_cap_total_pct=args.risk_cap_total,
            risk_cap_symbol_pct=args.risk_cap_symbol,
            cooldown_bars=live_cooldown,
            block_correlated_same_dir=not args.no_correlation_block,
        )

        execution = None
        if not args.no_exec:
            execution = ExecutionModule(
                exec_tf=args.exec_tf,
                exec_window_minutes=args.exec_window_min,
                pullback_atr_frac=args.pullback_atr,
                confirm_indicator=args.confirm_indicator,
                allow_market_fallback=args.allow_market_fallback,
            )

        learning_mgr = None
        if args.enable_learning:
            from learning import LearningManager, LearningConfig
            learning_config = LearningConfig(
                retrain_hour_utc=args.retrain_hour,
                retrain_interval_hours=args.retrain_interval,
                training_epochs=args.retrain_epochs,
                min_prauc_threshold=args.promote_min_pr_auc,
                min_pf_net=args.min_pf_net,
                min_profitable_regimes=args.min_profitable_regimes,
                auto_promote=not args.no_auto_promote,
                geometry_sweep_on_retrain=not args.no_geometry_sweep_retrain,
                gate_pf_net=args.gate_pf_net,
                gate_enet=args.gate_enet,
                gate_profitable_regimes=args.gate_profitable_regimes,
                gate_maxdd_r=args.gate_maxdd_r,
                gate_p95_min=args.gate_p95_min,
                gate_p95_max=args.gate_p95_max,
            )
            learning_mgr = LearningManager(
                replit_url=args.url,
                device=device,
                symbols=symbols,
                config=learning_config,
            )
            print(f"  Learning system: ON (retrain @ {args.retrain_hour}:00 UTC)")

        if args.execution_mode:
            exec_mode = args.execution_mode
        elif args.live and not args.paper:
            exec_mode = "live"
        elif args.paper:
            exec_mode = "paper"
        else:
            exec_mode = "signal_only"

        record_trades = args.record_trades or args.paper or (exec_mode in ("paper", "live"))
        log.info(f"[MODE] execution_mode={exec_mode} record_trades={record_trades} paper={args.paper} live={exec_mode == 'live'}")

        runner = LiveRunner(
            replit_url=args.url,
            symbols=symbols,
            device=device,
            interval=args.interval,
            enter_threshold=live_threshold,
            tp_mult=live_tp,
            sl_mult=live_sl,
            cooldown_bars=live_cooldown,
            paper=args.paper,
            execution_mode=exec_mode,
            record_trades=record_trades,
            portfolio_manager=portfolio,
            execution_module=execution,
            dry_run=args.dry_run,
            dry_run_candles=args.dry_run_candles,
            per_symbol_models=args.per_symbol_models,
            limit_15m=args.limit_15m,
            direct_htf=args.direct_htf,
            budget_core=args.budget_core,
            budget_flow=args.budget_flow,
            budget_scalp=args.budget_scalp,
            side_aware_scoring=getattr(args, 'v5_side_aware_scoring', False),
            direction_balance_cap=getattr(args, 'v5_direction_balance_cap', False),
            direction_balance_threshold=getattr(args, 'v5_direction_balance_threshold', 0.75),
            v5_live_threshold=getattr(args, 'v5_live_threshold', None),
            v5_mae_floor=getattr(args, 'v5_live_mae_floor', None),
            predictive_sltp=getattr(args, 'v5_predictive_sltp', False),
        )
        runner.learning_manager = learning_mgr

        if args.verify_separation:
            from verify_system import SeparationVerifier, run_static_audit
            cycles = args.cycles if args.cycles != 30 else 200
            sep_verifier = SeparationVerifier(max_cycles=cycles)
            runner.separation_verifier = sep_verifier
            print(f"\n  SEPARATION VERIFICATION MODE (v4.5): Running {cycles} cycles")
            print(f"  Checks: SCALP gate enforcement, router priority, budget bounds, exit resolve")
            print(f"  Running static code audit first...\n")
            static_report = run_static_audit()
            print(static_report)
            runner.run()
            report = sep_verifier.generate_report()
            full_report = static_report + "\n\n" + report
            with open("verify_report_v4.5.md", 'w') as f:
                f.write(full_report)
            print(f"\n{'='*60}")
            print(report)
            print(f"{'='*60}")
            print(f"\nFull report saved to verify_report_v4.5.md")
        elif args.verify_system:
            from verify_system import SystemVerifier, run_static_audit
            verifier = SystemVerifier(max_cycles=args.cycles)
            runner.verifier = verifier
            portfolio.verifier = verifier
            print(f"\n  VERIFICATION MODE: Running {args.cycles} cycles with assertions")
            print(f"  Running static code audit first...\n")
            static_report = run_static_audit()
            print(static_report)
            runner.run()
            report = verifier.generate_report()
            full_report = static_report + "\n\n" + report
            with open("verify_report.md", 'w') as f:
                f.write(full_report)
            print(f"\n{'='*60}")
            print(report)
            print(f"{'='*60}")
            print(f"\nFull report saved to verify_report.md")
        else:
            runner.run()
        return

    if args.regime_eval:
        data_path = download_data(args.url, data_dir)

        is_sweep = args.geometry_sweep or args.tp_mults or args.sl_mults or args.thresholds or args.cooldowns or args.topn_list or args.paired_tp_sl
        if is_sweep:
            PAIRED_TP_SL = [
                (3.0, 1.25),
                (3.0, 1.5),
                (3.5, 1.5),
            ]

            if args.paired_tp_sl:
                tp_sl_pairs = []
                for pair_str in args.paired_tp_sl.split(","):
                    tp_str, sl_str = pair_str.strip().split(":")
                    tp_sl_pairs.append((float(tp_str), float(sl_str)))
            elif args.geometry_sweep:
                tp_sl_pairs = PAIRED_TP_SL
            elif args.tp_mults or args.sl_mults:
                tp_mults = [float(x) for x in args.tp_mults.split(",")] if args.tp_mults else [args.tp_mult]
                sl_mults = [float(x) for x in args.sl_mults.split(",")] if args.sl_mults else [args.sl_mult]
                tp_sl_pairs = [(tp, sl) for tp in tp_mults for sl in sl_mults]
            else:
                tp_sl_pairs = PAIRED_TP_SL

            thresholds = [float(x) for x in args.thresholds.split(",")] if args.thresholds else [0.80, 0.85]
            cooldowns = [int(x) for x in args.cooldowns.split(",")] if args.cooldowns else [4, 6, 8]
            topn_list = [int(x) for x in args.topn_list.split(",")] if args.topn_list else None

            run_geometry_sweep(
                data_path, device, args.regimes,
                tp_sl_pairs, thresholds, cooldowns,
                args.horizon, args.slope_eps, args.r_min_expiry,
                fees_bps_entry=args.fees_entry_bps, fees_bps_exit=args.fees_exit_bps,
                spread_bps=args.spread_bps, slip_k=args.slip_k,
                size_cap=args.size_cap,
                debug_costs=args.debug_costs,
                topn_list=topn_list,
                target_tpd=args.target_tpd, target_tpd_tol=args.target_tpd_tol,
            )
        else:
            run_regime_eval(
                data_path, device, args.regimes, args.policy,
                args.cooldown, args.tp_mult, args.sl_mult, args.horizon,
                args.slope_eps, args.r_min_expiry,
                fees_bps_entry=args.fees_entry_bps, fees_bps_exit=args.fees_exit_bps,
                spread_bps=args.spread_bps, slip_k=args.slip_k,
                size_cap=args.size_cap,
            )
        return

    if args.verify_pr_auc_upgrade:
        log.info("=" * 60)
        log.info("  PR-AUC STABLE PATCH VERIFICATION (v4.5.1)")
        log.info("=" * 60)
        checks_passed = 0
        checks_total = 0
        failures = []

        use_focal = args.use_focal_loss and not args.no_focal_loss
        use_ohem_flag = args.use_ohem and not args.no_ohem
        use_edge = args.use_edge_head and not args.no_edge_head
        use_soft = args.use_soft_labels and not getattr(args, 'no_soft_labels', False)

        checks_total += 1
        if use_focal:
            log.info(f"  [PASS] Focal loss ACTIVE (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
            checks_passed += 1
        else:
            log.warning("  [FAIL] Focal loss DISABLED")
            failures.append("focal_loss disabled")

        checks_total += 1
        if not use_ohem_flag:
            log.info(f"  [PASS] OHEM DISABLED (prevents all-positive collapse)")
            checks_passed += 1
        else:
            log.warning("  [WARN] OHEM ACTIVE — may cause all-positive collapse with focal loss")
            checks_passed += 1

        checks_total += 1
        if args.ohem_neg_pct <= 0.20:
            log.info(f"  [PASS] OHEM neg_pct={args.ohem_neg_pct} (<= 0.20)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] OHEM neg_pct={args.ohem_neg_pct} (expected <= 0.20)")
            failures.append(f"ohem_neg_pct={args.ohem_neg_pct}")

        checks_total += 1
        if use_edge:
            log.info(f"  [PASS] Edge head ACTIVE (weight={args.edge_loss_weight})")
            checks_passed += 1
        else:
            log.warning("  [FAIL] Edge head DISABLED")
            failures.append("edge_head disabled")

        checks_total += 1
        if use_soft:
            log.info(f"  [PASS] Soft labels ON (temp={args.soft_label_temp})")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Soft labels OFF (expected ON by default)")
            failures.append("soft_labels off")

        checks_total += 1
        if args.soft_label_temp <= 1.2:
            log.info(f"  [PASS] Soft label temp={args.soft_label_temp} (<= 1.2)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Soft label temp={args.soft_label_temp} (expected <= 1.2)")
            failures.append(f"soft_label_temp={args.soft_label_temp}")

        checks_total += 1
        if args.r_min_expiry >= 1.0:
            log.info(f"  [PASS] r_min_expiry={args.r_min_expiry} (>= 1.0)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] r_min_expiry={args.r_min_expiry} (expected >= 1.0)")
            failures.append(f"r_min_expiry={args.r_min_expiry}")

        checks_total += 1
        if args.horizon == 16:
            log.info(f"  [PASS] Horizon={args.horizon} (== 16)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Horizon={args.horizon} (expected 16)")
            failures.append(f"horizon={args.horizon}")

        checks_total += 1
        from data.regression_targets import RegressionTargetGenerator
        reg_gen = RegressionTargetGenerator(horizon_periods=args.horizon)
        if reg_gen.horizon_periods == args.horizon:
            log.info(f"  [PASS] Horizon alignment: CLI={args.horizon} regression={reg_gen.horizon_periods}")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Horizon mismatch: CLI={args.horizon} regression={reg_gen.horizon_periods}")
            failures.append("horizon mismatch")

        checks_total += 1
        if args.promote_min_pr_auc >= 0.42:
            log.info(f"  [PASS] PR-AUC promotion gate = {args.promote_min_pr_auc} (>= 0.42)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] PR-AUC promotion gate = {args.promote_min_pr_auc} (< 0.42)")
            failures.append(f"promote_min_pr_auc={args.promote_min_pr_auc}")

        checks_total += 1
        from models.simple_mlp import EnhancedMultiHeadMLP_Config
        test_cfg = EnhancedMultiHeadMLP_Config(input_dim=63, enable_edge_head=True)
        if hasattr(test_cfg, 'enable_edge_head') and test_cfg.enable_edge_head:
            log.info("  [PASS] EnhancedMultiHeadMLP supports edge_head")
            checks_passed += 1
        else:
            log.warning("  [FAIL] EnhancedMultiHeadMLP missing edge_head support")
            failures.append("edge_head not in model config")

        checks_total += 1
        ece_temp_path = Path("checkpoints/temp_scale_v5.0.json")
        if ece_temp_path.exists():
            with open(ece_temp_path) as f:
                td = json.load(f)
            if 'ece_before' in td and 'ece_after' in td:
                log.info(f"  [PASS] ECE metrics in temp_scale: before={td['ece_before']:.4f} after={td['ece_after']:.4f}")
                checks_passed += 1
            else:
                log.warning("  [WARN] temp_scale exists but missing ECE fields (retrain to populate)")
                checks_passed += 1
        else:
            log.info("  [INFO] No temp_scale yet (run training first to generate ECE data)")
            checks_passed += 1

        checks_total += 1
        VERSION = FEATURE_VERSION
        if VERSION == "v4.5.2_directional_sep_fix":
            log.info(f"  [PASS] Version = {VERSION}")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Version = {VERSION} (expected v4.5.2_directional_sep_fix)")
            failures.append(f"version={VERSION}")

        checks_total += 1
        import inspect
        sig = inspect.signature(_oi_sanity_check)
        returns_bool = sig.return_annotation in (bool, inspect.Parameter.empty)
        has_threshold = 'coverage_threshold' in sig.parameters
        threshold_default = sig.parameters.get('coverage_threshold')
        threshold_val = threshold_default.default if threshold_default else None
        if has_threshold and threshold_val == 80.0:
            log.info(f"  [PASS] OI auto-disable: coverage_threshold={threshold_val}%, returns bool")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] OI auto-disable: threshold={threshold_val} (expected 80.0)")
            failures.append(f"oi_threshold={threshold_val}")

        log.info(f"\n  RESULT: {checks_passed}/{checks_total} checks passed")
        if checks_passed == checks_total:
            log.info("  [VERIFY_PR_AUC] ALL CHECKS PASSED")
        else:
            log.warning(f"  [VERIFY_PR_AUC] FAIL: {', '.join(failures)}")
        return

    if not args.predict_only:
        symbols_list = [s.strip().upper() for s in args.symbols.split(",")]

        if args.download_missing_data:
            log.info(f"[DOWNLOAD] Auto-downloading data for {symbols_list}...")
            dl_results = download_missing_data(args.url, data_dir, symbols_list)
            for sym, count in dl_results.items():
                log.info(f"[DOWNLOAD] {sym}: {count} candles available")
        else:
            data_path = download_data(args.url, data_dir, symbol=symbols_list[0])

        symbols_list = preflight_data_check(
            data_dir, symbols_list,
            allow_partial=args.allow_partial_data,
            min_bars=MIN_BARS_FOR_TRAINING
        )

        data_path = data_dir / f"{symbols_list[0]}_15m.parquet"

        if args.train_mythos:
            log.info("[MODE] MYTHOS Walk-Forward Intelligence Stack")
            from mythos.config import MythosConfig
            from mythos.walkforward import run_mythos_walk_forward

            mythos_cfg = MythosConfig(
                train_months=args.mythos_train_months,
                test_months=args.mythos_test_months,
                tp_mult=args.tp_mult,
                sl_mult=args.sl_mult,
                n_regimes=args.mythos_n_regimes,
                min_regime_confidence=args.mythos_min_regime_confidence,
                min_router_confidence=args.mythos_min_confidence,
                min_expected_r=args.mythos_min_expected_r,
                min_edge_threshold=args.mythos_edge_threshold,
                daily_loss_cap_r=args.mythos_daily_loss_cap,
                weekly_loss_cap_r=args.mythos_weekly_loss_cap,
                emergency_stop_r=args.mythos_emergency_stop_r,
                emergency_max_drawdown_r=args.mythos_emergency_max_drawdown_r,
                drawdown_size_start_r=args.mythos_drawdown_size_start_r,
                drawdown_size_full_r=args.mythos_drawdown_size_full_r,
                drawdown_size_min_scale=args.mythos_drawdown_size_min_scale,
                disable_conviction_boost_drawdown_r=args.mythos_disable_conviction_boost_dd_r,
                disable_leverage_drawdown_r=args.mythos_disable_leverage_dd_r,
                dd_size_throttle_start_r=args.mythos_drawdown_size_start_r,
                dd_size_throttle_end_r=args.mythos_drawdown_size_full_r,
                dd_size_throttle_min=args.mythos_drawdown_size_min_scale,
                dd_disable_leverage_r=args.mythos_disable_leverage_dd_r,
                dd_risk_recovery_r=args.mythos_dd_risk_recovery_r,
                cooldown_bars=args.mythos_cooldown_bars,
                max_trades_per_day=args.mythos_max_trades_per_day,
                max_leverage=args.mythos_max_leverage,
                vol_target=args.mythos_vol_target,
                min_trades_for_confidence=args.mythos_min_trades,
                max_folds=args.mythos_max_folds,
                save_best_model=args.mythos_save_best_model,
                best_model_metric=args.mythos_best_model_metric,
                model_output_dir=args.mythos_model_output_dir,
                analog_k=args.mythos_analog_k,
                analog_blend=args.mythos_analog_blend,
                online_reliability_alpha=args.mythos_online_reliability_alpha,
                reliability_regime_window=args.mythos_reliability_regime_window,
                robust_score_dd_penalty=args.mythos_robust_score_dd_penalty,
                side_balance_window=args.mythos_side_balance_window,
                side_imbalance_soft_cap=args.mythos_side_imbalance_soft_cap,
                side_imbalance_edge_penalty=args.mythos_side_imbalance_edge_penalty,
                side_rebalance_enable=args.mythos_side_rebalance_enable,
                side_rebalance_warmup_trades=args.mythos_side_rebalance_warmup_trades,
                side_rebalance_window=args.mythos_side_rebalance_window,
                side_rebalance_short_target=args.mythos_side_rebalance_short_target,
                side_rebalance_short_boost=args.mythos_side_rebalance_short_boost,
                side_rebalance_long_penalty=args.mythos_side_rebalance_long_penalty,
                side_rebalance_conf_boost=args.mythos_side_rebalance_conf_boost,
                side_rebalance_quality_guard=args.mythos_side_rebalance_quality_guard,
                side_rebalance_max_adjust=args.mythos_side_rebalance_max_adjust,
                intelligence_enable=args.mythos_intelligence_enable,
                intelligence_min_samples=args.mythos_intelligence_min_samples,
                intelligence_ema_alpha=args.mythos_intelligence_ema_alpha,
                intelligence_hit_weight=args.mythos_intelligence_hit_weight,
                intelligence_expectancy_weight=args.mythos_intelligence_expectancy_weight,
                intelligence_variance_penalty=args.mythos_intelligence_variance_penalty,
                intelligence_edge_scale=args.mythos_intelligence_edge_scale,
                intelligence_negative_edge_scale=args.mythos_intelligence_negative_edge_scale,
                intelligence_conf_scale=args.mythos_intelligence_conf_scale,
                intelligence_uncertainty_scale=args.mythos_intelligence_uncertainty_scale,
                intelligence_max_edge_adjust=args.mythos_intelligence_max_edge_adjust,
                intelligence_side_switch_enable=args.mythos_intelligence_side_switch_enable,
                intelligence_side_switch_min_gap=args.mythos_intelligence_side_switch_min_gap,
                intelligence_side_switch_min_analog_adv=args.mythos_intelligence_side_switch_min_analog_adv,
                intelligence_side_switch_conviction_guard=args.mythos_intelligence_side_switch_conviction_guard,
                intelligence_side_switch_cooldown_bars=args.mythos_intelligence_side_switch_cooldown_bars,
                intelligence_side_switch_max_rate=args.mythos_intelligence_side_switch_max_rate,
                intelligence_side_switch_min_samples=args.mythos_intelligence_side_switch_min_samples,
                adaptive_side_target_strength=args.mythos_adaptive_side_target_strength,
                adaptive_side_target_min=args.mythos_adaptive_side_target_min,
                adaptive_side_target_max=args.mythos_adaptive_side_target_max,
                side_health_penalty=args.mythos_side_health_penalty,
                side_health_boost=args.mythos_side_health_boost,
                side_health_decay=args.mythos_side_health_decay,
                precision_min_edge=args.mythos_precision_min_edge,
                precision_min_confidence=args.mythos_precision_min_confidence,
                precision_min_conviction=args.mythos_precision_min_conviction,
                precision_high_conviction=args.mythos_precision_high_conviction,
                conviction_weight_edge=args.mythos_precision_edge_weight,
                conviction_weight_confidence=args.mythos_precision_confidence_weight,
                conviction_weight_uncertainty=args.mythos_precision_uncertainty_weight,
                conviction_weight_meta=args.mythos_precision_meta_weight,
                conviction_score_threshold=args.mythos_conviction_score_threshold,
                conviction_boost=args.mythos_conviction_boost,
                conviction_max_size_mult=args.mythos_conviction_max_size_mult,
                conviction_recent_window=args.mythos_conviction_recent_window,
                conviction_recent_min_trades=args.mythos_conviction_recent_min_trades,
                conviction_recent_min_expectancy=args.mythos_conviction_recent_min_expectancy,
                sure_min_analog_hits=args.mythos_sure_min_analog_hits,
                sure_min_analog_ratio=args.mythos_sure_min_analog_ratio,
                sure_meta_strength_min=args.mythos_sure_meta_strength_min,
                sure_edge_buffer=args.mythos_sure_edge_buffer,
                sure_confidence_buffer=args.mythos_sure_confidence_buffer,
                sure_recent_window=args.mythos_sure_recent_window,
                sure_recent_min_trades=args.mythos_sure_recent_min_trades,
                sure_recent_min_hit_rate=args.mythos_sure_recent_min_hit_rate,
                sure_recent_min_expectancy=args.mythos_sure_recent_min_expectancy,
                sure_cold_start_conviction_extra=args.mythos_sure_cold_start_conviction_extra,
                sure_cold_start_meta_extra=args.mythos_sure_cold_start_meta_extra,
                leverage_edge_buffer=args.mythos_leverage_edge_buffer,
                leverage_confidence_buffer=args.mythos_leverage_confidence_buffer,
                leverage_conviction_buffer=args.mythos_leverage_conviction_buffer,
                leverage_recent_window=args.mythos_leverage_recent_window,
                leverage_recent_min_trades=args.mythos_leverage_recent_min_trades,
                leverage_recent_min_hit_rate=args.mythos_leverage_recent_min_hit_rate,
                leverage_recent_min_expectancy=args.mythos_leverage_recent_min_expectancy,
                leverage_policy_window=args.mythos_leverage_policy_window,
                leverage_policy_min_trades=args.mythos_leverage_policy_min_trades,
                leverage_policy_min_hit_rate=args.mythos_leverage_policy_min_hit_rate,
                leverage_policy_min_expectancy=args.mythos_leverage_policy_min_expectancy,
                leverage_policy_context_weight=args.mythos_leverage_policy_context_weight,
                leverage_policy_cold_start_conviction_extra=args.mythos_leverage_policy_cold_start_conviction_extra,
                leverage_net_edge_floor=args.mythos_leverage_net_edge_floor,
                leverage_side_policy_enable=args.mythos_leverage_side_policy_enable,
                leverage_side_min_trades=args.mythos_leverage_side_min_trades,
                leverage_side_min_hit_rate=args.mythos_leverage_side_min_hit_rate,
                leverage_side_min_expectancy=args.mythos_leverage_side_min_expectancy,
                execution_fee_bps=args.mythos_execution_fee_bps,
                execution_slippage_bps=args.mythos_execution_slippage_bps,
                execution_cost_cap_r=args.mythos_execution_cost_cap_r,
                bayes_quality_enable=args.mythos_bayes_quality_enable,
                bayes_quality_warmup_trades=args.mythos_bayes_quality_warmup_trades,
                bayes_quality_decay=args.mythos_bayes_quality_decay,
                bayes_quality_prior_alpha=args.mythos_bayes_quality_prior_alpha,
                bayes_quality_prior_beta=args.mythos_bayes_quality_prior_beta,
                bayes_quality_regime_weight=args.mythos_bayes_quality_regime_weight,
                bayes_quality_min_win_prob=args.mythos_bayes_quality_min_win_prob,
                bayes_quality_min_expectancy=args.mythos_bayes_quality_min_expectancy,
                bayes_quality_edge_scale=args.mythos_bayes_quality_edge_scale,
                bayes_quality_confidence_scale=args.mythos_bayes_quality_confidence_scale,
                bayes_quality_uncertainty_scale=args.mythos_bayes_quality_uncertainty_scale,
                bayes_quality_reject_margin=args.mythos_bayes_quality_reject_margin,
                nonconformity_enable=args.mythos_nonconformity_enable,
                nonconformity_warmup_trades=args.mythos_nonconformity_warmup_trades,
                nonconformity_window=args.mythos_nonconformity_window,
                nonconformity_quantile=args.mythos_nonconformity_quantile,
                nonconformity_margin=args.mythos_nonconformity_margin,
                nonconformity_min_winners=args.mythos_nonconformity_min_winners,
                nonconformity_weight_uncertainty=args.mythos_nonconformity_weight_uncertainty,
                nonconformity_weight_confidence=args.mythos_nonconformity_weight_confidence,
                nonconformity_weight_edge=args.mythos_nonconformity_weight_edge,
                nonconformity_weight_meta=args.mythos_nonconformity_weight_meta,
                nonconformity_weight_analog=args.mythos_nonconformity_weight_analog,
                nonconformity_override_conviction=args.mythos_nonconformity_override_conviction,
                nonconformity_override_edge_buffer=args.mythos_nonconformity_override_edge_buffer,
                nonconformity_override_confidence_buffer=args.mythos_nonconformity_override_confidence_buffer,
                nonconformity_target_reject_rate=args.mythos_nonconformity_target_reject_rate,
                nonconformity_reject_tolerance=args.mythos_nonconformity_reject_tolerance,
                nonconformity_adaptive_relax=args.mythos_nonconformity_adaptive_relax,
                nonconformity_adaptive_max_relax=args.mythos_nonconformity_adaptive_max_relax,
                nonconformity_soft_override_margin=args.mythos_nonconformity_soft_override_margin,
                counterfactual_target_reject_rate=args.mythos_counterfactual_target_reject_rate,
                counterfactual_reject_tolerance=args.mythos_counterfactual_reject_tolerance,
                counterfactual_adaptive_relax=args.mythos_counterfactual_adaptive_relax,
                counterfactual_adaptive_min_adv_floor=args.mythos_counterfactual_adaptive_min_adv_floor,
                short_boost_enable=args.mythos_short_boost_enable,
                short_boost_window=args.mythos_short_boost_window,
                short_boost_min_trades=args.mythos_short_boost_min_trades,
                short_boost_threshold_r=args.mythos_short_boost_threshold_r,
                short_boost_edge=args.mythos_short_boost_edge,
                short_boost_confidence=args.mythos_short_boost_confidence,
                drawdown_edge_start_r=args.mythos_drawdown_edge_start_r,
                drawdown_edge_step_r=args.mythos_drawdown_edge_step_r,
                drawdown_edge_boost=args.mythos_drawdown_edge_boost,
                loss_streak_trigger=args.mythos_loss_streak_trigger,
                loss_streak_cooldown_bars=args.mythos_loss_streak_cooldown_bars,
                side_fail_window=args.mythos_side_fail_window,
                side_fail_min_trades=args.mythos_side_fail_min_trades,
                side_fail_expectancy_r=args.mythos_side_fail_expectancy_r,
                side_fail_cooldown_bars=args.mythos_side_fail_cooldown_bars,
                side_fail_hard_pause=args.mythos_side_fail_hard_pause,
                online_allocator_lr=args.mythos_online_allocator_lr,
                online_allocator_min_mult=args.mythos_online_allocator_min_mult,
                online_allocator_max_mult=args.mythos_online_allocator_max_mult,
                change_detect_z_thresh=args.mythos_change_detect_z_thresh,
                change_detect_confirm_bars=args.mythos_change_detect_confirm_bars,
                change_detect_cooldown_bars=args.mythos_change_detect_cooldown_bars,
                change_edge_floor_boost=args.mythos_change_edge_floor_boost,
                change_confidence_boost=args.mythos_change_confidence_boost,
                change_uncertainty_mult=args.mythos_change_uncertainty_mult,
                flip_intensity_trigger=args.mythos_regime_flip_trigger,
                flip_harden_hold_bars=args.mythos_flip_harden_hold_bars,
                instability_edge_mult=args.mythos_instability_edge_mult,
                instability_confidence_drop=args.mythos_instability_confidence_drop,
                instability_uncertainty_mult=args.mythos_instability_uncertainty_mult,
                transition_learn_rate=args.mythos_transition_learn_rate,
                transition_min_samples=args.mythos_transition_min_samples,
                transition_edge_gain=args.mythos_transition_edge_gain,
                transition_confidence_gain=args.mythos_transition_confidence_gain,
                transition_uncertainty_gain=args.mythos_transition_uncertainty_gain,
                counterfactual_min_advantage_r=args.mythos_counterfactual_min_advantage_r,
                counterfactual_risk_penalty=args.mythos_counterfactual_risk_penalty,
                counterfactual_margin=args.mythos_counterfactual_margin,
                counterfactual_uncertainty_weight=args.mythos_counterfactual_uncertainty_weight,
                counterfactual_min_alt_hits=args.mythos_counterfactual_min_alt_hits,
                enable_gpu_neural_experts=args.mythos_use_neural_expert,
                neural_expert_device=args.mythos_neural_device,
                neural_expert_hidden=args.mythos_neural_hidden,
                neural_expert_epochs=args.mythos_neural_epochs,
                neural_expert_lr=args.mythos_neural_lr,
                neural_expert_batch_size=args.mythos_neural_batch_size,
                use_meta_learner=args.mythos_meta_learner,
                meta_learner_device=args.mythos_meta_device,
                meta_learner_hidden=args.mythos_meta_hidden,
                meta_learner_epochs=args.mythos_meta_epochs,
                meta_learner_lr=args.mythos_meta_lr,
                meta_learner_batch_size=args.mythos_meta_batch_size,
                meta_learner_edge_blend=args.mythos_meta_edge_gain,
                meta_learner_conf_blend=args.mythos_meta_confidence_gain,
                meta_learner_uncertainty_penalty=args.mythos_meta_uncertainty_gain,
                meta_learner_fallback=args.mythos_meta_fallback,
                meta_learner_fallback_lr=args.mythos_meta_fallback_lr,
                meta_bootstrap_samples=args.mythos_meta_bootstrap_samples,
                meta_bootstrap_epochs=args.mythos_meta_bootstrap_epochs,
                meta_learner_min_train_samples=args.mythos_meta_min_train_samples,
                meta_learner_warmup_samples=args.mythos_meta_warmup_samples,
                meta_learner_ready_prob_floor=args.mythos_meta_ready_prob_floor,
                meta_learner_ready_prob_ceiling=args.mythos_meta_ready_prob_ceiling,
            )
            mythos_report = run_mythos_walk_forward(
                data_dir=data_dir,
                symbols=symbols_list,
                train_months=args.mythos_train_months,
                test_months=args.mythos_test_months,
                config=mythos_cfg,
                output_path=Path(args.mythos_report_path),
            )
            agg = mythos_report.get("aggregate", {})
            log.info(
                "[MYTHOS] Complete: trades=%s totalR=%s expectancy=%s win_rate=%s pf=%s avgDD=%s robust=%s change_rate=%s cf_reject_rate=%s active_folds=%s/%s",
                agg.get("total_trades"), agg.get("total_r"), agg.get("expectancy_r"),
                agg.get("win_rate"), agg.get("profit_factor"),
                agg.get("avg_max_drawdown_r"), agg.get("avg_robust_score"), agg.get("change_mode_rate"),
                agg.get("counterfactual_reject_rate"),
                agg.get("active_folds"), agg.get("folds"),
            )
            log.info(
                "[MYTHOS] Side stats: long trades=%s win=%s totalR=%s | short trades=%s win=%s totalR=%s",
                agg.get("long_trades"), agg.get("long_win_rate"), agg.get("long_total_r"),
                agg.get("short_trades"), agg.get("short_win_rate"), agg.get("short_total_r"),
            )
            log.info(
                "[MYTHOS] High-conviction stats: trades=%s win=%s totalR=%s",
                agg.get("high_conviction_trades"), agg.get("high_conviction_win_rate"), agg.get("high_conviction_total_r"),
            )
            log.info(
                "[MYTHOS] Intelligence certainty: sure trades=%s hits=%s win=%s totalR=%s",
                agg.get("sure_trades"),
                agg.get("sure_hits"),
                agg.get("sure_win_rate"),
                agg.get("sure_total_r"),
            )
            log.info(
                "[MYTHOS] Leverage execution: leveraged trades=%s hits=%s hit_rate=%s totalR=%s | sure+leveraged=%s hits=%s hit_rate=%s totalR=%s",
                agg.get("leveraged_trades"),
                agg.get("leveraged_hits"),
                agg.get("leveraged_hit_rate"),
                agg.get("leveraged_total_r"),
                agg.get("sure_leveraged_trades"),
                agg.get("sure_leveraged_hits"),
                agg.get("sure_leveraged_hit_rate"),
                agg.get("sure_leveraged_total_r"),
            )
            log.info(
                "[MYTHOS] Leverage gate: approved=%s blocked_candidates=%s",
                agg.get("leverage_boost_approved"),
                agg.get("leverage_blocked_candidates"),
            )
            log.info(
                "[MYTHOS] Net execution: cost_totalR=%s net_totalR=%s net_expectancy=%s net_win_rate=%s net_sure_lev_hit=%s",
                agg.get("execution_cost_total_r"),
                agg.get("net_total_r"),
                agg.get("net_expectancy_r"),
                agg.get("net_win_rate"),
                agg.get("net_sure_leveraged_hit_rate"),
            )
            log.info(
                "[MYTHOS] Bayesian quality gate: rejects=%s reject_rate=%s ready_checks=%s",
                agg.get("bayes_quality_rejects"),
                agg.get("bayes_quality_reject_rate"),
                agg.get("bayes_quality_ready_checks"),
            )
            log.info(
                "[MYTHOS] Nonconformity gate: rejects=%s reject_rate=%s overrides=%s ready_checks=%s winners_ref=%s",
                agg.get("nonconformity_rejects"),
                agg.get("nonconformity_reject_rate"),
                agg.get("nonconformity_overrides"),
                agg.get("nonconformity_ready_checks"),
                agg.get("nonconformity_winner_ref_count"),
            )
            log.info(
                "[MYTHOS] Intelligence engine: mode_bars=%s mode_rate=%s side_switches=%s avg_score=%s",
                agg.get("intelligence_mode_bars"),
                agg.get("intelligence_mode_rate"),
                agg.get("intelligence_side_switches"),
                agg.get("intelligence_avg_score"),
            )
            if agg.get("best_model_path"):
                log.info(
                    "[MYTHOS] Best model: metric=%s value=%s path=%s",
                    agg.get("best_model_metric"),
                    agg.get("best_model_metric_value"),
                    agg.get("best_model_path"),
                )
            return

        if args.train_v5:
            log.info("[MODE] v5.0 Forecaster training (continuous predictions + decision layer)")
            from data.candidate_generator import (
                CandidateConfig, RiskControls,
            )
            from train.v5_train import (
                train_v5_model, V5QualityGateConfig, V5TPDControllerConfig,
                run_v5_forward_test, run_v5_walk_forward,
            )

            cand_cfg = CandidateConfig.from_cli_args(args) if args.use_candidates else CandidateConfig(enabled=False)
            risk_cfg = RiskControls.from_cli_args(args)

            v5_barrier_presets = None
            if args.v5_barrier_mode != 'fixed':
                v5_barrier_presets = [p.strip() for p in args.barrier_presets.split(",")]

            qual_cfg = V5QualityGateConfig(
                sigma_max=args.v5_sigma_max,
                mae_max=args.v5_mae_max,
                mu_R_min=args.v5_muR_min,
                p_trade_min=args.v5_p_trade_min,
                enable_calib=args.v5_enable_calib,
            )
            v5_effective_tpd = args.v5_target_tpd
            v5_effective_tol = args.v5_tpd_tol
            if args.v5_target_trades_per_day is not None:
                v5_effective_tpd = args.v5_target_trades_per_day
                log.info(f"[V5] --v5-target-trades-per-day={v5_effective_tpd} overrides --v5-target-tpd")
            if args.v5_target_trades_per_day_band is not None:
                v5_effective_tol = args.v5_target_trades_per_day_band
                log.info(f"[V5] --v5-target-trades-per-day-band={v5_effective_tol} overrides --v5-tpd-tol")

            tpd_cfg = V5TPDControllerConfig(
                target_tpd=v5_effective_tpd,
                tpd_tol=v5_effective_tol,
                thr_warmup_epochs=args.v5_thr_warmup_epochs,
                thr_step_mult=args.v5_thr_step_mult,
                score_threshold=args.v5_score_threshold,
                score_lambda=args.v5_score_lambda,
                mae_cap=args.v5_mae_cap,
                side_mode=args.v5_score_side_mode,
                rr_weight=args.v5_rr_weight,
                min_threshold_floor=args.v5_min_threshold if args.v5_min_threshold is not None else 0.001,
            )
            log.info(f"[V5] side_mode={args.v5_score_side_mode} rr_weight={args.v5_rr_weight}")

            train_end_date = args.v5_train_end_date
            test_start_date = args.v5_test_start_date
            test_end_date = args.v5_test_end_date

            if train_end_date and test_start_date:
                from datetime import datetime as dt
                te = dt.strptime(train_end_date, "%Y-%m-%d")
                ts = dt.strptime(test_start_date, "%Y-%m-%d")
                if te > ts:
                    log.error(f"[V5] train_end_date ({train_end_date}) must be <= test_start_date ({test_start_date})")
                    sys.exit(1)
                log.info(f"[V5] Time-based split: train < {train_end_date}, test >= {test_start_date}"
                         + (f" to {test_end_date}" if test_end_date else ""))

            v5_regime_side_map = None
            if args.v5_regime_side_map:
                v5_regime_side_map = {}
                for pair in args.v5_regime_side_map.split(','):
                    k, v = pair.strip().split('=')
                    v5_regime_side_map[k.strip()] = v.strip().upper()
                log.info(f"[V5] Regime side map: {v5_regime_side_map}")

            _effective_balanced_mode = args.balanced_sampling_mode
            if args.no_symbol_balanced_sampling:
                _effective_balanced_sampling = False
                _effective_balanced_mode = 'none'
            elif _effective_balanced_mode == 'none':
                _effective_balanced_sampling = False
            else:
                _effective_balanced_sampling = True
            log.info(f"[BALANCE] balanced_sampling={_effective_balanced_sampling} mode={_effective_balanced_mode}")

            if args.v5_walk_forward:
                log.info("[MODE] V5 Walk-Forward Analysis")
                run_v5_walk_forward(
                    data_dir=data_dir,
                    device=device,
                    symbols=symbols_list,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    train_months=args.v5_wf_train_months,
                    test_months=args.v5_wf_test_months,
                    horizon=args.horizon,
                    tp_mult=args.tp_mult,
                    sl_mult=args.sl_mult,
                    score_lambda=args.v5_score_lambda,
                    risk_proxy=args.v5_risk_proxy,
                    quality_gate_cfg=qual_cfg,
                    tpd_ctrl_cfg=tpd_cfg,
                    candidate_config=cand_cfg,
                    risk_controls=risk_cfg,
                    hold_target=args.v5_hold_target,
                    mfe_min=args.v5_mfe_min,
                    w_ret=args.v5_w_ret, w_mfe=args.v5_w_mfe,
                    w_mae=args.v5_w_mae, w_action=args.v5_w_action,
                    w_barrier=args.v5_w_barrier, w_regime=args.v5_w_regime,
                    sigma_spread_reg=args.v5_sigma_spread_reg,
                    sigma_reg_threshold=args.v5_sigma_reg_threshold,
                    phase1_epochs=args.v5_phase1_epochs,
                    atr_normalize_risk_heads=args.v5_atr_normalize_risk_heads,
                    dynamic_action_labels=args.v5_dynamic_action_labels,
                    loss_warmup_epochs=args.v5_loss_warmup_epochs,
                    loss_warmup_ret_mult=args.v5_loss_warmup_ret_mult,
                    loss_warmup_action_mult=args.v5_loss_warmup_action_mult,
                    warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
                    barrier_mode=args.v5_barrier_mode,
                    barrier_presets=v5_barrier_presets,
                    use_regime_head=args.use_regime_head,
                    cand_warmup_epochs=args.v5_cand_warmup,
                    ema200_regime_gate=args.v5_ema200_regime_gate,
                    weekly_loss_cap=args.v5_weekly_loss_cap,
                    warmup_skip_bars=args.v5_warmup_skip_bars,
                    corr_block=args.v5_corr_block and not args.no_correlation_block,
                    corr_window_days=args.v5_corr_window_days,
                    corr_thresh=args.v5_corr_thresh,
                    corr_same_side_only=args.v5_corr_same_side_only,
                    corr_log_matrix=args.v5_log_corr_matrix,
                    corr_max_block=args.v5_corr_max_block,
                    adaptive_sizing=args.v5_adaptive_sizing,
                    kelly_fraction=args.v5_kelly_fraction,
                    max_size_mult=args.v5_max_size_mult,
                    min_size_mult=args.v5_min_size_mult,
                    regime_scaling=args.v5_regime_scaling,
                    regime_bull_mult=args.v5_regime_bull_mult,
                    regime_bear_mult=args.v5_regime_bear_mult,
                    regime_lookback=args.v5_regime_lookback,
                    daily_loss_cap=args.v5_daily_loss_cap,
                    trailing_equity_stop=args.v5_trailing_equity_stop,
                    per_symbol_daily_r_budget=args.v5_per_symbol_daily_r,
                    min_threshold=args.v5_min_threshold,
                    max_threshold=args.v5_max_threshold,
                    min_threshold_pct=args.v5_min_threshold_pct,
                    max_trades_per_day=args.v5_max_trades_per_day,
                    trailing_sl=args.v5_trailing_sl,
                    trail_activation=args.v5_trail_activation,
                    trail_distance=args.v5_trail_distance,
                    allow_runner=args.v5_allow_runner,
                    conviction_sizing=args.v5_conviction_sizing,
                    conviction_tier_top_pct=args.v5_conviction_top_pct,
                    conviction_tier_top_mult=args.v5_conviction_top_mult,
                    conviction_tier_high_pct=args.v5_conviction_high_pct,
                    conviction_tier_high_mult=args.v5_conviction_high_mult,
                    conviction_confidence_threshold=args.v5_conviction_conf_thresh,
                    conviction_confidence_boost=args.v5_conviction_conf_boost,
                    adx_gate=args.v5_adx_gate,
                    adx_period=args.v5_adx_period,
                    adx_min=args.v5_adx_min,
                    adx_exception_top_pct=args.v5_adx_exception_top_pct,
                    temp_scale=args.v5_temp_scale,
                    promote_metric=args.v5_promote_metric,
                    stage_a_epochs=10 if args.v5_staged_training else 0,
                    balanced_sampling=_effective_balanced_sampling,
                    balanced_sampling_mode=_effective_balanced_mode,
                    symbol_embed_dim=args.v5_symbol_embed_dim,
                    per_symbol_scaler=args.per_symbol_scaler,
                    ultra_conviction=args.v5_ultra_conviction,
                    ultra_risk_cap=args.v5_ultra_risk_cap,
                    ultra_score_pct=args.v5_ultra_score_pct,
                    ultra_adx_min=args.v5_ultra_adx_min,
                    ultra_edge_min=args.v5_ultra_edge_min,
                    ultra_dd_max=args.v5_ultra_dd_max,
                    ultra_max_per_day=args.v5_ultra_max_per_day,
                    ultra_mult=args.v5_ultra_mult,
                    ddt_enable=args.v5_ddt_enable,
                    ddt_lookback_trades=args.v5_ddt_lookback_trades,
                    ddt_bad_rollr=args.v5_ddt_bad_rollr,
                    ddt_thr_k=args.v5_ddt_thr_k,
                    ddt_thr_min=args.v5_ddt_thr_min,
                    ddt_thr_max=args.v5_ddt_thr_max,
                    ddt_size_k=args.v5_ddt_size_k,
                    ddt_min_size_mult=args.v5_ddt_min_size_mult,
                    ddt_alpha_down=args.v5_ddt_alpha_down,
                    ddt_alpha_up=args.v5_ddt_alpha_up,
                    ddt_warmup_trades=args.v5_ddt_warmup_trades,
                    multi_regime=args.v5_multi_regime,
                    regime_adx_trending=args.v5_regime_adx_trending,
                    regime_adx_choppy=args.v5_regime_adx_choppy,
                    regime_atr_high_vol=args.v5_regime_atr_high_vol,
                    regime_atr_low_vol=args.v5_regime_atr_low_vol,
                    regime_atr_window=args.v5_regime_atr_window,
                    regime_ema_slope_window=args.v5_regime_ema_slope_window,
                    regime_ema_buffer=args.v5_regime_ema_buffer,
                    edge_first=args.v5_edge_first,
                    edge_min=args.v5_edge_min,
                    edge_pct_floor=args.v5_edge_pct_floor,
                    edge_topn_per_day=args.v5_edge_topn_per_day,
                    regime_side_map=v5_regime_side_map,
                    regime_soft=args.v5_regime_soft,
                    regime_disagree_mult=args.v5_regime_disagree_mult,
                    regime_none_mult=args.v5_regime_none_mult,
                    per_symbol_soft_kill=args.v5_per_symbol_soft_kill,
                    edge_topn_soft=args.v5_edge_topn_soft,
                    edge_topn_decay=args.v5_edge_topn_decay,
                    size_floor=args.v5_size_floor,
                    soft_gate_floor=args.v5_soft_gate_floor,
                    weekly_cap_dynamic=args.v5_weekly_cap_dynamic,
                    weekly_cap_scale=args.v5_weekly_cap_scale,
                    quality_gate_enabled=getattr(args, 'v5_quality_gate', False),
                    quality_gate_window=getattr(args, 'v5_quality_gate_window', 50),
                    direction_balance_cap=getattr(args, 'v5_direction_balance_cap', False),
                    direction_balance_threshold=getattr(args, 'v5_direction_balance_threshold', 0.75),
                    recency_weight=args.v5_recency_weight,
                    recency_half_life=args.v5_recency_half_life,
                    finetune_months=args.v5_finetune_months,
                    finetune_epochs=args.v5_finetune_epochs,
                    finetune_lr_mult=args.v5_finetune_lr_mult,
                    warm_start=args.v5_wf_warm_start,
                    warm_start_lr_mult=args.v5_wf_warm_start_lr_mult,
                    head_disagreement_gate=getattr(args, 'v5_head_disagree_gate', False),
                    slippage_base_bps=getattr(args, 'slippage_base_bps', 0.0),
                    sigma_discount=args.v5_sigma_discount and not args.v5_no_sigma_discount,
                    min_p_side=args.v5_min_p_side,
                    min_p_short=args.v5_min_p_short,
                    side_aware_scoring=args.v5_side_aware_scoring,
                    mae_asym_weight=args.v5_mae_asym_weight,
                    per_symbol_cooldown=args.v5_per_symbol_cooldown,
                    cooldown=args.cooldown,
                    min_trades=args.v5_min_trades,
                    wf_threshold_ema=args.v5_wf_threshold_ema and not args.v5_no_wf_threshold_ema,
                    wf_threshold_ema_alpha=args.v5_wf_threshold_ema_alpha,
                    wf_threshold_decay=args.v5_wf_threshold_decay,
                    mu_debias=args.v5_mu_debias,
                    mu_debias_alpha=args.v5_mu_debias_alpha,
                    per_symbol_r_kill=args.v5_per_symbol_r_kill,
                    per_symbol_threshold=args.v5_per_symbol_threshold,
                    sweep_objective=args.v5_sweep_objective,
                    short_oversample=args.v5_short_oversample,
                    short_min_fraction=args.v5_short_min_fraction,
                    ema200_soft_mult=args.v5_ema200_soft_mult,
                    per_side_threshold=args.v5_per_side_threshold,
                    per_sym_no_edge_fallback=args.v5_per_sym_no_edge_fallback,
                    rolling_er_gate=getattr(args, 'v5_rolling_er_gate', False),
                    rolling_er_window=getattr(args, 'v5_rolling_er_window', 20),
                    rolling_er_min=getattr(args, 'v5_rolling_er_min', -0.05),
                    gate_mode=getattr(args, 'v5_gate_mode', 'ref_magnitude'),
                    replit_url=getattr(args, 'url', None),
                    model_version='v6' if args.v6 else 'v5',
                    v6_seq_len=args.v6_seq_len,
                    v6_conv_channels=args.v6_conv_channels,
                    v6_n_conv_layers=args.v6_n_conv_layers,
                    v6_attn_heads=args.v6_attn_heads,
                    v6_attn_layers=args.v6_attn_layers,
                    v6_n_experts=args.v6_n_experts,
                    v6_expert_top_k=args.v6_expert_top_k,
                    v6_feature_mask_ratio=args.v6_feature_mask_ratio,
                    v6_aux_weight=args.v6_aux_weight,
                    v6_confidence_weight=args.v6_confidence_weight,
                    v6_moe_balance_weight=args.v6_moe_balance_weight,
                    side_bal_weight=args.v5_side_bal_weight,
                    action_entropy_weight=args.v5_action_entropy_weight,
                    chop_hold_target=args.v5_chop_hold_target,
                    ret_mag_ce_weight=args.v5_ret_mag_ce_weight,
                    ret_mag_scale=args.v5_ret_mag_scale,
                    specialist_mode=getattr(args, 'v5_side_specialist', 'none'),
                    dual_specialist=getattr(args, 'v5_dual_specialist', False),
                    min_mu_r_long=getattr(args, 'v5_min_mu_r_long', -1e9),              # Task #69
                    long_disagree_mult=getattr(args, 'v5_long_disagree_mult', 1.0),     # Task #69
                    specialist_align_weight=getattr(args, 'v5_specialist_align_weight', 0.0),  # Task #69
                )
                return

            replit_url = getattr(args, 'url', None)
            _single_pusher = None
            if replit_url:
                import train.v5_train as _v5mod
                from train.training_push import TrainingProgressPusher
                _single_pusher = TrainingProgressPusher(replit_url=replit_url)
                try:
                    import torch as _torch
                    _gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU"
                except Exception:
                    _gpu_name = "Unknown"
                _single_pusher.session_start(
                    session_type="single_train",
                    total_folds=1,
                    total_epochs=args.epochs,
                    symbols=symbols_list,
                    config={
                        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                        "horizon": args.horizon, "tp_mult": args.tp_mult, "sl_mult": args.sl_mult,
                        "model_version": "v6" if args.v6 else "v5",
                    },
                    gpu_name=_gpu_name,
                )
                _single_pusher.fold_start(fold_num=1, train_start="", train_end="",
                                           test_start="", test_end="")
                _v5mod._active_pusher = _single_pusher
                _v5mod._active_fold_num = 1
                _v5mod._active_total_folds = 1

            try:
              train_v5_model(
                data_path, device, args.epochs, args.batch_size, args.lr,
                checkpoint_interval=args.checkpoint_interval,
                warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
                tp_mult=args.tp_mult, sl_mult=args.sl_mult,
                horizon=args.horizon,
                symbols=symbols_list,
                w_ret=args.v5_w_ret,
                w_mfe=args.v5_w_mfe,
                w_mae=args.v5_w_mae,
                w_action=args.v5_w_action,
                w_barrier=args.v5_w_barrier,
                w_regime=args.v5_w_regime,
                sigma_spread_reg=args.v5_sigma_spread_reg,
                sigma_reg_threshold=args.v5_sigma_reg_threshold,
                phase1_epochs=args.v5_phase1_epochs,
                atr_normalize_risk_heads=args.v5_atr_normalize_risk_heads,
                dynamic_action_labels=args.v5_dynamic_action_labels,
                loss_warmup_epochs=args.v5_loss_warmup_epochs,
                loss_warmup_ret_mult=args.v5_loss_warmup_ret_mult,
                loss_warmup_action_mult=args.v5_loss_warmup_action_mult,
                score_lambda=args.v5_score_lambda,
                risk_proxy=args.v5_risk_proxy,
                target_tpd=v5_effective_tpd,
                target_tpd_tol=v5_effective_tol,
                hold_target=args.v5_hold_target,
                mfe_min=args.v5_mfe_min,
                cand_warmup_epochs=args.v5_cand_warmup,
                barrier_mode=args.v5_barrier_mode,
                barrier_presets=v5_barrier_presets,
                use_regime_head=args.use_regime_head,
                candidate_config=cand_cfg,
                risk_controls=risk_cfg,
                q_min_tp=args.q_min_tp,
                r_min_expiry_strict=args.r_min_expiry_strict,
                auto_balance_enter_labels=args.auto_balance_enter_labels,
                target_enter_rate=args.target_enter_rate,
                target_enter_rate_min=args.target_enter_rate_min,
                target_enter_rate_max=args.target_enter_rate_max,
                balance_search_steps=args.balance_search_steps,
                quality_gate_cfg=qual_cfg,
                tpd_ctrl_cfg=tpd_cfg,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                run_forward_test=args.v5_forward_test,
                freeze_decision=args.v5_freeze_decision,
                run_diagnostics=args.v5_diagnostics,
                ema200_regime_gate=args.v5_ema200_regime_gate,
                weekly_loss_cap=args.v5_weekly_loss_cap,
                warmup_skip_bars=args.v5_warmup_skip_bars,
                corr_block=args.v5_corr_block and not args.no_correlation_block,
                corr_window_days=args.v5_corr_window_days,
                corr_thresh=args.v5_corr_thresh,
                corr_same_side_only=args.v5_corr_same_side_only,
                corr_log_matrix=args.v5_log_corr_matrix,
                adaptive_sizing=args.v5_adaptive_sizing,
                kelly_fraction=args.v5_kelly_fraction,
                max_size_mult=args.v5_max_size_mult,
                min_size_mult=args.v5_min_size_mult,
                regime_scaling=args.v5_regime_scaling,
                regime_bull_mult=args.v5_regime_bull_mult,
                regime_bear_mult=args.v5_regime_bear_mult,
                regime_lookback=args.v5_regime_lookback,
                daily_loss_cap=args.v5_daily_loss_cap,
                trailing_equity_stop=args.v5_trailing_equity_stop,
                per_symbol_daily_r_budget=args.v5_per_symbol_daily_r,
                min_threshold=args.v5_min_threshold,
                max_threshold=args.v5_max_threshold,
                min_threshold_pct=args.v5_min_threshold_pct,
                max_trades_per_day=args.v5_max_trades_per_day,
                trailing_sl=args.v5_trailing_sl,
                trail_activation=args.v5_trail_activation,
                trail_distance=args.v5_trail_distance,
                allow_runner=args.v5_allow_runner,
                conviction_sizing=args.v5_conviction_sizing,
                conviction_tier_top_pct=args.v5_conviction_top_pct,
                conviction_tier_top_mult=args.v5_conviction_top_mult,
                conviction_tier_high_pct=args.v5_conviction_high_pct,
                conviction_tier_high_mult=args.v5_conviction_high_mult,
                conviction_confidence_threshold=args.v5_conviction_conf_thresh,
                conviction_confidence_boost=args.v5_conviction_conf_boost,
                adx_gate=args.v5_adx_gate,
                adx_period=args.v5_adx_period,
                adx_min=args.v5_adx_min,
                adx_exception_top_pct=args.v5_adx_exception_top_pct,
                temp_scale=args.v5_temp_scale,
                promote_metric=args.v5_promote_metric,
                stage_a_epochs=10 if args.v5_staged_training else 0,
                balanced_sampling=_effective_balanced_sampling,
                balanced_sampling_mode=_effective_balanced_mode,
                symbol_embed_dim=args.v5_symbol_embed_dim,
                per_symbol_scaler=args.per_symbol_scaler,
                ultra_conviction=args.v5_ultra_conviction,
                ultra_risk_cap=args.v5_ultra_risk_cap,
                ultra_score_pct=args.v5_ultra_score_pct,
                ultra_adx_min=args.v5_ultra_adx_min,
                ultra_edge_min=args.v5_ultra_edge_min,
                ultra_dd_max=args.v5_ultra_dd_max,
                ultra_max_per_day=args.v5_ultra_max_per_day,
                ultra_mult=args.v5_ultra_mult,
                ddt_enable=args.v5_ddt_enable,
                ddt_lookback_trades=args.v5_ddt_lookback_trades,
                ddt_bad_rollr=args.v5_ddt_bad_rollr,
                ddt_thr_k=args.v5_ddt_thr_k,
                ddt_thr_min=args.v5_ddt_thr_min,
                ddt_thr_max=args.v5_ddt_thr_max,
                ddt_size_k=args.v5_ddt_size_k,
                ddt_min_size_mult=args.v5_ddt_min_size_mult,
                ddt_alpha_down=args.v5_ddt_alpha_down,
                ddt_alpha_up=args.v5_ddt_alpha_up,
                ddt_warmup_trades=args.v5_ddt_warmup_trades,
                multi_regime=args.v5_multi_regime,
                regime_adx_trending=args.v5_regime_adx_trending,
                regime_adx_choppy=args.v5_regime_adx_choppy,
                regime_atr_high_vol=args.v5_regime_atr_high_vol,
                regime_atr_low_vol=args.v5_regime_atr_low_vol,
                regime_atr_window=args.v5_regime_atr_window,
                regime_ema_slope_window=args.v5_regime_ema_slope_window,
                regime_ema_buffer=args.v5_regime_ema_buffer,
                edge_first=args.v5_edge_first,
                edge_min=args.v5_edge_min,
                edge_pct_floor=args.v5_edge_pct_floor,
                edge_topn_per_day=args.v5_edge_topn_per_day,
                regime_side_map=v5_regime_side_map,
                regime_soft=args.v5_regime_soft,
                regime_disagree_mult=args.v5_regime_disagree_mult,
                regime_none_mult=args.v5_regime_none_mult,
                per_symbol_soft_kill=args.v5_per_symbol_soft_kill,
                edge_topn_soft=args.v5_edge_topn_soft,
                edge_topn_decay=args.v5_edge_topn_decay,
                size_floor=args.v5_size_floor,
                soft_gate_floor=args.v5_soft_gate_floor,
                weekly_cap_dynamic=args.v5_weekly_cap_dynamic,
                weekly_cap_scale=args.v5_weekly_cap_scale,
                quality_gate_enabled=getattr(args, 'v5_quality_gate', False),
                quality_gate_window=getattr(args, 'v5_quality_gate_window', 50),
                direction_balance_cap=getattr(args, 'v5_direction_balance_cap', False),
                direction_balance_threshold=getattr(args, 'v5_direction_balance_threshold', 0.75),
                head_disagreement_gate=args.v5_head_disagree_gate,
                slippage_base_bps=args.slippage_base_bps,
                sigma_discount=args.v5_sigma_discount and not args.v5_no_sigma_discount,
                min_p_side=args.v5_min_p_side,
                min_p_short=args.v5_min_p_short,
                side_aware_scoring=args.v5_side_aware_scoring,
                mae_asym_weight=args.v5_mae_asym_weight,
                per_symbol_cooldown=args.v5_per_symbol_cooldown,
                cooldown=args.cooldown,
                mu_debias=args.v5_mu_debias,
                mu_debias_alpha=args.v5_mu_debias_alpha,
                min_trades=args.v5_min_trades,
                feature_report=args.v5_feature_report,
                per_symbol_r_kill=args.v5_per_symbol_r_kill,
                per_symbol_threshold=args.v5_per_symbol_threshold,
                sweep_objective=args.v5_sweep_objective,
                short_oversample=args.v5_short_oversample,
                short_min_fraction=args.v5_short_min_fraction,
                ema200_soft_mult=args.v5_ema200_soft_mult,
                per_side_threshold=args.v5_per_side_threshold,
                per_sym_no_edge_fallback=args.v5_per_sym_no_edge_fallback,
                rolling_er_gate=getattr(args, 'v5_rolling_er_gate', False),
                rolling_er_window=getattr(args, 'v5_rolling_er_window', 20),
                rolling_er_min=getattr(args, 'v5_rolling_er_min', -0.05),
                gate_mode=getattr(args, 'v5_gate_mode', 'ref_magnitude'),
                model_version='v6' if args.v6 else 'v5',
                v6_seq_len=args.v6_seq_len,
                v6_conv_channels=args.v6_conv_channels,
                v6_n_conv_layers=args.v6_n_conv_layers,
                v6_attn_heads=args.v6_attn_heads,
                v6_attn_layers=args.v6_attn_layers,
                v6_n_experts=args.v6_n_experts,
                v6_expert_top_k=args.v6_expert_top_k,
                v6_feature_mask_ratio=args.v6_feature_mask_ratio,
                v6_aux_weight=args.v6_aux_weight,
                v6_confidence_weight=args.v6_confidence_weight,
                v6_moe_balance_weight=args.v6_moe_balance_weight,
                side_bal_weight=args.v5_side_bal_weight,
                action_entropy_weight=args.v5_action_entropy_weight,
                chop_hold_target=args.v5_chop_hold_target,
                ret_mag_ce_weight=args.v5_ret_mag_ce_weight,
                ret_mag_scale=args.v5_ret_mag_scale,
                specialist_mode=getattr(args, 'v5_side_specialist', 'none'),
                dual_specialist=getattr(args, 'v5_dual_specialist', False),
                min_mu_r_long=getattr(args, 'v5_min_mu_r_long', -1e9),              # Task #69
                long_disagree_mult=getattr(args, 'v5_long_disagree_mult', 1.0),     # Task #69
                specialist_align_weight=getattr(args, 'v5_specialist_align_weight', 0.0),  # Task #69
            )

              if _single_pusher is not None:
                _single_pusher.fold_end(fold_num=1, completed_folds=1)
                _single_pusher.session_end(status="completed", completed_folds=1)
                _v5mod._active_pusher = None
            except Exception as _train_err:
              if _single_pusher is not None:
                _single_pusher.session_end(status="failed", error_message=str(_train_err))
                _v5mod._active_pusher = None
              raise

            tag = "V6" if args.v6 else "V5"
            log.info("=" * 60)
            log.info(f"  {tag} TRAINING COMPLETE")
            log.info("=" * 60)
            return

        elif args.train_distributional:
            log.info("[MODE] v4.9.0 Distributional Trade Forecaster training")
            from data.candidate_generator import (
                CandidateConfig, MultiHorizonConfig, PresetConfig, RiskControls,
            )
            cand_cfg = CandidateConfig.from_cli_args(args) if args.use_candidates else CandidateConfig(enabled=False)
            mh_cfg = MultiHorizonConfig.from_cli_args(args) if getattr(args, 'multi_horizon', False) else None
            preset_cfg = PresetConfig.from_cli_args(args) if getattr(args, 'multi_preset', False) else None
            risk_cfg = RiskControls.from_cli_args(args)

            model, engineer, feature_columns, history = train_distributional_model(
                data_path, device, args.epochs, args.batch_size, args.lr,
                checkpoint_interval=args.checkpoint_interval,
                warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
                tp_mult=args.tp_mult, sl_mult=args.sl_mult,
                horizon=args.horizon,
                symbols=symbols_list,
                w_mse=args.dist_w_mse,
                w_quantile=args.dist_w_quantile,
                w_bce=args.dist_w_bce,
                w_regime=args.dist_w_regime,
                score_lambda=args.score_lambda,
                target_tpd=args.dist_target_tpd,
                target_tpd_tol=args.dist_target_tpd_tol,
                use_regime_head=args.use_regime_head,
                q_min_tp=args.q_min_tp,
                r_min_expiry_strict=args.r_min_expiry_strict,
                auto_balance_enter_labels=args.auto_balance_enter_labels,
                target_enter_rate=args.target_enter_rate,
                target_enter_rate_min=args.target_enter_rate_min,
                target_enter_rate_max=args.target_enter_rate_max,
                balance_search_steps=args.balance_search_steps,
                w_dir=args.dist_w_dir,
                candidate_config=cand_cfg,
                multi_horizon_config=mh_cfg,
                preset_config=preset_cfg,
                use_money_score=getattr(args, 'money_score', False),
                risk_controls=risk_cfg,
                use_kelly_sizing=getattr(args, 'kelly_sizing', False),
                symbol_embed_dim=args.v5_symbol_embed_dim,
            )
        else:
            use_focal = args.use_focal_loss and not args.no_focal_loss
            use_ohem = args.use_ohem and not args.no_ohem
            use_edge = args.use_edge_head and not args.no_edge_head
            use_soft = args.use_soft_labels and not args.no_soft_labels

            model, engineer, feature_columns, history = train_enter_model(
                data_path, device, args.epochs, args.batch_size, args.lr,
                checkpoint_interval=args.checkpoint_interval,
                warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
                tp_mult=args.tp_mult, sl_mult=args.sl_mult,
                horizon=args.horizon, slope_eps=args.slope_eps,
                r_min_expiry=args.r_min_expiry,
                target_tpd=args.target_tpd, target_tpd_tol=args.target_tpd_tol,
                symbols=symbols_list, value_loss_weight=args.value_loss_weight,
                value_clip=args.value_clip,
                smoke_calib=args.smoke_calib, smoke_infer=args.smoke_infer,
                use_focal_loss=use_focal, focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha, use_ohem=use_ohem,
                ohem_neg_pct=args.ohem_neg_pct, use_edge_head=use_edge,
                edge_loss_weight=args.edge_loss_weight,
                use_soft_labels=use_soft,
                soft_label_temp=args.soft_label_temp,
                loss_warmup_epochs=args.loss_warmup_epochs,
                warmup_pos_weight=args.warmup_pos_weight,
                transition_epochs=args.transition_epochs,
                focal_gamma_final=args.focal_gamma_final,
                lr_drop_on_transition=args.lr_drop_on_transition,
                disable_ohem_during_transition=args.disable_ohem_during_transition and not args.enable_ohem_during_transition,
                collapse_guard=args.collapse_guard and not args.no_collapse_guard,
                collapse_guard_pred1=args.collapse_guard_pred1,
                collapse_guard_sep=args.collapse_guard_sep,
                collapse_guard_freeze_epochs=args.collapse_guard_freeze_epochs,
                verify_enter_metrics=args.verify_enter_metrics,
                w_quality=args.w_quality,
                w_dir=args.w_dir,
                w_htf=args.w_htf,
                verify_v46_separation=args.verify_v46_separation,
                use_v47_labels=args.use_v47_labels,
                q_min_tp=args.q_min_tp,
                r_min_expiry_strict=args.r_min_expiry_strict,
                auto_balance_enter_labels=args.auto_balance_enter_labels,
                target_enter_rate=args.target_enter_rate,
                target_enter_rate_min=args.target_enter_rate_min,
                target_enter_rate_max=args.target_enter_rate_max,
                balance_search_steps=args.balance_search_steps,
                pos_weight_min=args.pos_weight_min,
                pos_weight_max=args.pos_weight_max,
                verify_v47_labels=args.verify_v47_labels,
                symbol_embed_dim=args.v5_symbol_embed_dim,
            )

        print()
        log.info("=" * 60)
        log.info("  TRAINING COMPLETE")
        log.info("=" * 60)

        if history['val_loss']:
            best_loss = min(history['val_loss'])
            best_prauc = max(history['val_prauc']) if history['val_prauc'] else 0
            log.info(f"  Best val loss: {best_loss:.4f}")
            log.info(f"  Best PR-AUC: {best_prauc:.3f}")
            log.info(f"  Epochs trained: {len(history['val_loss'])}")
    else:
        import torch
        checkpoint_path = None
        for candidate in [
            Path("checkpoints/best_enter_prauc.pt"),
            Path("checkpoints/best_v5_expectancy.pt"),
            Path("checkpoints/best_enter_loss.pt"),
            Path("checkpoints/best_v5_loss.pt"),
        ]:
            if candidate.exists():
                checkpoint_path = candidate
                break
        if checkpoint_path is None:
            log.error("No trained ENTER model found! Run without --predict-only first.")
            sys.exit(1)

        log.info("Downloading fresh data for prediction...")
        data_path = download_data(args.url, data_dir, force_fresh=True)

        log.info("Loading saved model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        saved_version = checkpoint.get('feature_version', 'unknown')
        ACCEPTED_VERSIONS = {FEATURE_VERSION, "v5.0.1_forecaster"}
        if saved_version not in ACCEPTED_VERSIONS:
            log.error(f"FATAL: Feature version mismatch! Model: '{saved_version}', accepted: {ACCEPTED_VERSIONS}")
            sys.exit(1)
        log.info(f"Feature version: {saved_version} (accepted)")

        feature_columns = checkpoint.get('feature_columns', [])
        if not feature_columns:
            log.error("FATAL: No feature_columns in checkpoint - retrain.")
            sys.exit(1)

        cfg = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', 'legacy')

        if model_type == 'v5_forecaster':
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
        else:
            from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
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
                enable_edge_head=cfg.get('enable_edge_head', False),
                enable_dir_head=cfg.get('enable_dir_head', False),
                enable_htf_head=cfg.get('enable_htf_head', False),
            )
            model = EnhancedMultiHeadMLP(mlp_config)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model._is_v5 = False
        model.to(device)

        from data.pipeline import FeatureEngineer
        engineer = FeatureEngineer()
        if model_type == 'v5_forecaster' and 'scaler_center' in checkpoint and 'scaler_scale' in checkpoint:
            from sklearn.preprocessing import RobustScaler
            v5_scaler = RobustScaler()
            v5_scaler.center_ = np.array(checkpoint['scaler_center'])
            v5_scaler.scale_ = np.array(checkpoint['scaler_scale'])
            engineer._v5_global_scaler = v5_scaler
            log.info("Scaler loaded from checkpoint (embedded)")
        else:
            scaler_path = None
            for sname in ["per_symbol_scalers.joblib", "scaler.joblib"]:
                sp = Path(f"checkpoints/{sname}")
                if sp.exists():
                    scaler_path = sp
                    break
            if scaler_path:
                engineer.load_scalers(str(scaler_path))
            else:
                log.warning("No saved scaler found - prediction quality may be reduced")

    if not args.no_push:
        prediction = make_enter_prediction(model, engineer, feature_columns, data_path, device)

        print()
        log.info("=" * 60)
        log.info(f"  SIGNAL: {prediction['action']} | p_enter: {prediction['confidence']:.1%}")
        log.info("=" * 60)
        log.info(f"  Price: ${prediction['current_price']:,.2f}")
        log.info(f"  Entry: ${prediction['entry_price']:,.2f}")
        log.info(f"  SL:    ${prediction['stop_loss_price']:,.2f} ({prediction['stop_loss_pct']:.2%})")
        log.info(f"  TP:    ${prediction['take_profit_price']:,.2f} ({prediction['take_profit_pct']:.2%})")
        log.info(f"  R:R = {prediction['risk_reward_ratio']:.1f} | Position: {prediction['position_size_pct']:.1f}%")
        if prediction.get('reasons'):
            log.info(f"  Reasons: {', '.join(prediction['reasons'])}")
        log.info("=" * 60)

        is_hold = prediction['action'] == "HOLD"
        if is_hold:
            log.info("Signal: HOLD - pushing to dashboard (no trade)")
        else:
            log.info(f"Signal: {prediction['action']} PASSED - pushing to dashboard")
        push_prediction(args.url, prediction)
    else:
        log.info("Skipping prediction push (--no-push)")

    print()
    log.info("Done! Check your dashboard to see the prediction.")
    print()


if __name__ == "__main__":
    main()
