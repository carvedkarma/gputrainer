#!/usr/bin/env python3
"""Bulk download 15m candle data for all 20 symbols.

Strategy:
1. For symbols already in DB (BTC/ETH/SOL/BNB) - export directly from dashboard API (fast)
2. For new symbols (AVAX/XRP/ADA) - fetch from Binance API directly

Usage:
    python3 gpu_trainer/bulk_download.py
    python3 gpu_trainer/bulk_download.py --force
"""

import json
import sys
import time
import os
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

import pandas as pd

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "PEPEUSDT", "SUIUSDT",
    "AAVEUSDT", "ARBUSDT", "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
]
DATA_DIR = Path(__file__).parent / "data_cache"
INTERVAL = "15m"
MS_15M = 15 * 60 * 1000
CANDLES_PER_REQUEST = 1000
DAYS_BACK = 1826
MIN_CANDLES = 20000
DASHBOARD_URL = "http://localhost:5000"

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
]


def fetch_from_dashboard(symbol: str) -> pd.DataFrame:
    url = f"{DASHBOARD_URL}/api/data/export-csv?symbol={symbol}&timeframe=15m"
    print(f"  [{symbol}] Fetching from dashboard API...")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read().decode()
        
        lines = data.strip().split('\n')
        if len(lines) < 2:
            return pd.DataFrame()
        
        import io
        df = pd.read_csv(io.StringIO(data))
        return df
    except Exception as e:
        print(f"  [{symbol}] Dashboard fetch failed: {e}")
        return pd.DataFrame()


def fetch_klines_batch(symbol: str, start_ms: int, end_ms: int) -> list:
    params = f"symbol={symbol}&interval={INTERVAL}&startTime={start_ms}&endTime={end_ms}&limit={CANDLES_PER_REQUEST}"
    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            continue
    return []


def download_from_binance(symbol: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    all_candles = []
    cursor = start_ms
    batch_num = 0
    expected_batches = (now_ms - start_ms) // (CANDLES_PER_REQUEST * MS_15M) + 1
    
    print(f"  [{symbol}] Downloading {days_back} days from Binance...")
    
    while cursor < now_ms:
        batch_end = min(cursor + CANDLES_PER_REQUEST * MS_15M, now_ms)
        
        data = []
        for attempt in range(3):
            data = fetch_klines_batch(symbol, cursor, batch_end)
            if data:
                break
            time.sleep(0.5 * (attempt + 1))
        
        if data:
            for k in data:
                all_candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
        
        cursor = batch_end
        batch_num += 1
        
        if batch_num % 20 == 0:
            pct = min(99, int(batch_num / expected_batches * 100))
            print(f"  [{symbol}] {pct}% - {len(all_candles):,} candles...", flush=True)
        
        time.sleep(0.005)
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


STALE_THRESHOLD_HOURS = 48


def refresh_existing_data(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    last_ts = int(df["timestamp"].max())
    now_ms = int(time.time() * 1000)
    hours_stale = (now_ms - last_ts) / (3600 * 1000)

    if hours_stale <= STALE_THRESHOLD_HOURS:
        return df

    print(f"  [{symbol}] Data is {hours_stale:.0f}h stale (last: {datetime.utcfromtimestamp(last_ts/1000).strftime('%Y-%m-%d %H:%M')}), fetching new candles...")

    start_ms = last_ts + MS_15M
    new_candles = []
    cursor = start_ms
    batch_num = 0

    while cursor < now_ms:
        batch_end = min(cursor + CANDLES_PER_REQUEST * MS_15M, now_ms)
        data = []
        for attempt in range(3):
            data = fetch_klines_batch(symbol, cursor, batch_end)
            if data:
                break
            time.sleep(0.5 * (attempt + 1))

        if data:
            for k in data:
                new_candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

        cursor = batch_end
        batch_num += 1
        if batch_num % 20 == 0:
            print(f"  [{symbol}] Refreshing... {len(new_candles):,} new candles so far", flush=True)
        time.sleep(0.02)

    if new_candles:
        df_new = pd.DataFrame(new_candles)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"  [{symbol}] Added {len(new_candles):,} new candles (total: {len(df):,})")
    else:
        print(f"  [{symbol}] No new candles available from Binance")

    return df


def process_symbol(symbol: str, force: bool = False, refresh: bool = False) -> dict:
    parquet_path = DATA_DIR / f"{symbol}_15m.parquet"

    if parquet_path.exists() and not force:
        df = pd.read_parquet(parquet_path)
        if len(df) >= MIN_CANDLES:
            if refresh:
                df = refresh_existing_data(symbol, df)
                df.to_parquet(parquet_path, index=False)
                date_min = datetime.utcfromtimestamp(df["timestamp"].min() / 1000).strftime("%Y-%m-%d")
                date_max = datetime.utcfromtimestamp(df["timestamp"].max() / 1000).strftime("%Y-%m-%d")
                return {"symbol": symbol, "candles": len(df), "status": "refreshed", "date_range": f"{date_min} to {date_max}"}
            else:
                last_ts = int(df["timestamp"].max())
                now_ms = int(time.time() * 1000)
                hours_stale = (now_ms - last_ts) / (3600 * 1000)
                if hours_stale > STALE_THRESHOLD_HOURS:
                    print(f"  [{symbol}] Cached but stale ({hours_stale:.0f}h old), auto-refreshing...")
                    df = refresh_existing_data(symbol, df)
                    df.to_parquet(parquet_path, index=False)
                    date_min = datetime.utcfromtimestamp(df["timestamp"].min() / 1000).strftime("%Y-%m-%d")
                    date_max = datetime.utcfromtimestamp(df["timestamp"].max() / 1000).strftime("%Y-%m-%d")
                    return {"symbol": symbol, "candles": len(df), "status": "refreshed", "date_range": f"{date_min} to {date_max}"}
                print(f"  [{symbol}] Already cached: {len(df):,} candles")
                return {"symbol": symbol, "candles": len(df), "status": "cached"}

    df = fetch_from_dashboard(symbol)

    if len(df) < MIN_CANDLES:
        print(f"  [{symbol}] Dashboard only has {len(df)} candles, fetching from Binance...")
        df_binance = download_from_binance(symbol)
        if len(df_binance) > len(df):
            df = df_binance

    if df.empty:
        return {"symbol": symbol, "candles": 0, "status": "failed"}

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    date_min = datetime.utcfromtimestamp(df["timestamp"].min() / 1000).strftime("%Y-%m-%d")
    date_max = datetime.utcfromtimestamp(df["timestamp"].max() / 1000).strftime("%Y-%m-%d")

    df.to_parquet(parquet_path, index=False)

    status = "ok" if len(df) >= MIN_CANDLES else f"low"
    print(f"  [{symbol}] SAVED: {len(df):,} candles [{date_min} to {date_max}]")
    return {"symbol": symbol, "candles": len(df), "status": status, "date_range": f"{date_min} to {date_max}"}


def main():
    force = "--force" in sys.argv
    refresh = "--refresh" in sys.argv
    only = None
    for arg in sys.argv[1:]:
        if arg.endswith("USDT"):
            only = [arg]
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = only or SYMBOLS
    
    print("=" * 60)
    print("  BULK DOWNLOAD: 15m Candle Data for GPU Training")
    print("=" * 60)
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Output:  {DATA_DIR}")
    if refresh:
        print(f"  Mode:    REFRESH (force update stale data)")
    print()
    
    results = []
    for sym in symbols:
        result = process_symbol(sym, force=force, refresh=refresh)
        results.append(result)
        sys.stdout.flush()
    
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    total = 0
    all_ok = True
    for r in results:
        ok = r["candles"] >= MIN_CANDLES
        icon = "OK" if ok else "LOW" if r["candles"] > 0 else "FAIL"
        print(f"  [{icon:4s}] {r['symbol']}: {r['candles']:>9,} candles  {r.get('date_range', '')}")
        total += r["candles"]
        if not ok:
            all_ok = False
    
    print(f"\n  Total: {total:,} candles")
    if all_ok:
        print("  All symbols ready for GPU training!")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
