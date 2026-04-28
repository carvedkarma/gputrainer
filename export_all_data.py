#!/usr/bin/env python3
"""Export all 20 symbol datasets from the dashboard API to parquet files for GPU training."""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "PEPEUSDT", "SUIUSDT",
    "AAVEUSDT", "ARBUSDT", "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
]
DATA_DIR = Path(__file__).parent / "data_cache"
MIN_CANDLES = 20000

def get_replit_url():
    url = os.environ.get("REPLIT_URL", "").rstrip("/")
    if not url:
        url = "http://localhost:5000"
    return url

def export_symbol(base_url: str, symbol: str, force: bool = False) -> dict:
    parquet_path = DATA_DIR / f"{symbol}_15m.parquet"
    
    if parquet_path.exists() and not force:
        df = pd.read_parquet(parquet_path)
        print(f"  {symbol}: already cached ({len(df)} candles)")
        return {"symbol": symbol, "candles": len(df), "status": "cached"}
    
    url = f"{base_url}/api/data/export-csv?symbol={symbol}&timeframe=15m"
    print(f"  {symbol}: downloading from {url}...")
    
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  {symbol}: FAILED - {e}")
        return {"symbol": symbol, "candles": 0, "status": f"error: {e}"}
    
    csv_path = DATA_DIR / f"{symbol}_15m.csv"
    total_bytes = 0
    with open(csv_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total_bytes += len(chunk)
    
    print(f"  {symbol}: downloaded {total_bytes / 1024:.0f} KB")
    
    df = pd.read_csv(csv_path)
    if len(df) < 100:
        print(f"  {symbol}: INSUFFICIENT DATA ({len(df)} candles)")
        csv_path.unlink(missing_ok=True)
        return {"symbol": symbol, "candles": len(df), "status": "insufficient"}
    
    date_min = pd.Timestamp(df['timestamp'].min(), unit='ms').strftime('%Y-%m-%d')
    date_max = pd.Timestamp(df['timestamp'].max(), unit='ms').strftime('%Y-%m-%d')
    
    df.to_parquet(parquet_path, index=False)
    csv_path.unlink(missing_ok=True)
    
    status = "ok" if len(df) >= MIN_CANDLES else f"low ({len(df)} < {MIN_CANDLES})"
    print(f"  {symbol}: {len(df)} candles [{date_min} to {date_max}] -> {parquet_path.name} [{status}]")
    return {"symbol": symbol, "candles": len(df), "status": status, "date_range": f"{date_min} to {date_max}"}

def check_backfill_progress(base_url: str) -> dict:
    try:
        resp = requests.get(f"{base_url}/api/historical/backfill/progress", timeout=10)
        return resp.json()
    except:
        return {"inProgress": False}

def main():
    force = "--force" in sys.argv
    wait = "--wait" not in sys.argv or True
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_url = get_replit_url()
    print(f"Dashboard URL: {base_url}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Force re-download: {force}")
    print()
    
    progress = check_backfill_progress(base_url)
    if progress.get("inProgress"):
        print(f"Backfill in progress: {progress.get('message', 'unknown')}")
        print("Waiting for backfill to complete before exporting...")
        while True:
            time.sleep(10)
            progress = check_backfill_progress(base_url)
            msg = progress.get("message", "")
            pct = progress.get("progress", 0)
            print(f"  Backfill: {pct}% - {msg}")
            if not progress.get("inProgress"):
                print("Backfill complete!")
                break
        print()
    
    print("=" * 60)
    print("  EXPORTING DATA FOR GPU TRAINER")
    print("=" * 60)
    
    results = []
    for sym in SYMBOLS:
        result = export_symbol(base_url, sym, force=force)
        results.append(result)
    
    print()
    print("=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for r in results:
        status_icon = "OK" if r["candles"] >= MIN_CANDLES else "LOW" if r["candles"] > 0 else "FAIL"
        print(f"  [{status_icon:4s}] {r['symbol']}: {r['candles']:>7,} candles - {r.get('status', 'unknown')}")
        if r["candles"] < MIN_CANDLES:
            all_ok = False
    
    ready_count = sum(1 for r in results if r["candles"] >= MIN_CANDLES)
    print()
    print(f"  Ready for training: {ready_count}/{len(SYMBOLS)} symbols")
    
    if all_ok:
        print("  All symbols have sufficient data for GPU training!")
    else:
        low_syms = [r["symbol"] for r in results if r["candles"] < MIN_CANDLES]
        print(f"  WARNING: {', '.join(low_syms)} need more data. Run backfill first.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
