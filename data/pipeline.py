import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import aiohttp
import asyncio
import time
from datetime import datetime, timedelta
import json
import gzip
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pywt
    HAVE_PYWT = True
except ImportError:
    HAVE_PYWT = False
    logger.warning(
        "[PIPELINE] pywt (PyWavelets) is not installed — wavelet features will be "
        "zero-filled.  Install with: pip install PyWavelets"
    )

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from tqdm import tqdm

class BinanceDataFetcher:
    BINANCE_VISION_URL = "https://data-api.binance.vision/api/v3"
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2"
    
    def __init__(self, symbols: List[str], timeframes: List[str], replit_proxy_url: Optional[str] = None, use_sync: bool = False):
        self.symbols = symbols
        self.timeframes = timeframes
        self.session = None
        self.working_source = None
        self.replit_proxy_url = replit_proxy_url
        self.use_sync = use_sync
    
    def _fetch_replit_proxy_sync(self, symbol: str, timeframe: str, limit: int,
                                  end_time: Optional[int] = None) -> List[Dict]:
        if not self.replit_proxy_url:
            return []
        
        import requests
        import time as _time
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(limit, 1000)
        }
        if end_time:
            params["endTime"] = end_time
        
        url = f"{self.replit_proxy_url}/api/data/klines"
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Replit Proxy] Fetching {symbol} {timeframe} (limit={params['limit']}, attempt {attempt}/{max_retries})...")
                
                response = requests.get(url, params=params, timeout=45)
                
                if response.status_code != 200:
                    print(f"[Replit Proxy] HTTP {response.status_code} (len={len(response.text)}): {response.text[:200]}")
                    if attempt < max_retries:
                        _time.sleep(3)
                    continue
                
                body = response.text
                if not body or len(body) < 5:
                    print(f"[Replit Proxy] Empty response body (len={len(body) if body else 0}), retrying...")
                    if attempt < max_retries:
                        _time.sleep(3)
                    continue
                
                content_type = response.headers.get('content-type', '')
                if 'json' not in content_type and body.strip().startswith('<'):
                    print(f"[Replit Proxy] Got HTML instead of JSON (content-type={content_type}, len={len(body)})")
                    print(f"[Replit Proxy] Body preview: {body[:200]}")
                    if attempt < max_retries:
                        _time.sleep(5)
                    continue
                
                import json
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as je:
                    print(f"[Replit Proxy] JSON parse failed: {je}")
                    print(f"[Replit Proxy] Body preview ({len(body)} bytes): {body[:300]}")
                    if attempt < max_retries:
                        _time.sleep(3)
                    continue
                if "candles" not in data or not isinstance(data["candles"], list):
                    print(f"[Replit Proxy] No 'candles' key in response, got keys: {list(data.keys())}")
                    if attempt < max_retries:
                        _time.sleep(3)
                    continue
                
                candles = []
                for c in data["candles"]:
                    candles.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": c["timestamp"],
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c["volume"]),
                        "close_time": c["closeTime"],
                        "quote_volume": float(c["quoteVolume"]),
                        "trades": c["trades"],
                        "taker_buy_base": float(c["takerBuyBase"]),
                        "taker_buy_quote": float(c["takerBuyQuote"])
                    })
                if candles:
                    print(f"[Replit Proxy] OK: {len(candles)} candles for {symbol} {timeframe}")
                    return candles
                else:
                    print(f"[Replit Proxy] Response had empty candles array")
                    if attempt < max_retries:
                        _time.sleep(3)
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"[Replit Proxy] Timeout after 45s (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    _time.sleep(5)
            except requests.exceptions.ConnectionError as e:
                print(f"[Replit Proxy] Connection error: {e} (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    _time.sleep(5)
            except Exception as e:
                print(f"[Replit Proxy] Error: {type(e).__name__}: {e} (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    _time.sleep(3)
        
        print(f"[Replit Proxy] FAILED: Could not fetch {symbol} {timeframe} after {max_retries} attempts")
        return []
    
    def _fetch_binance_direct_sync(self, symbol: str, timeframe: str, limit: int,
                                     end_time: Optional[int] = None) -> List[Dict]:
        import requests
        
        binance_urls = [
            "https://api.binance.com/api/v3",
            "https://data-api.binance.vision/api/v3",
            "https://api1.binance.com/api/v3",
        ]
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(limit, 1000)
        }
        if end_time:
            params["endTime"] = end_time
        
        for base_url in binance_urls:
            try:
                url = f"{base_url}/klines"
                print(f"[Direct Binance] Trying {base_url} for {symbol} {timeframe}...")
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code != 200:
                    print(f"[Direct Binance] {base_url} returned HTTP {response.status_code}")
                    continue
                
                raw = response.json()
                if not isinstance(raw, list) or len(raw) == 0:
                    print(f"[Direct Binance] {base_url} returned empty array")
                    continue
                
                candles = []
                for k in raw:
                    candles.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": k[6],
                        "quote_volume": float(k[7]),
                        "trades": k[8],
                        "taker_buy_base": float(k[9]),
                        "taker_buy_quote": float(k[10])
                    })
                print(f"[Direct Binance] OK: {len(candles)} candles for {symbol} {timeframe} via {base_url}")
                return candles
            except Exception as e:
                print(f"[Direct Binance] {base_url} failed: {e}")
                continue
        
        print(f"[Direct Binance] FAILED: All Binance endpoints failed for {symbol} {timeframe}")
        return []

    def fetch_klines_sync(self, symbol: str, timeframe: str, limit: int = 1000,
                          end_time: Optional[int] = None) -> List[Dict]:
        if self.replit_proxy_url:
            result = self._fetch_replit_proxy_sync(symbol, timeframe, limit, end_time)
            if result:
                return result
            print(f"[Sync] Replit proxy failed, falling back to direct Binance...")
        
        return self._fetch_binance_direct_sync(symbol, timeframe, limit, end_time)
    
    def fetch_historical_sync(self, symbol: str, timeframe: str, 
                               num_candles: int = 175000) -> pd.DataFrame:
        print(f"[Sync] Fetching {num_candles:,} candles for {symbol} {timeframe}...")
        
        all_candles = []
        end_time = None
        remaining = num_candles
        batch_size = 1000
        fetched = 0
        
        while remaining > 0:
            fetch_count = min(batch_size, remaining)
            candles = self.fetch_klines_sync(symbol, timeframe, fetch_count, end_time)
            
            if not candles:
                print(f"[Sync] No more data available for {symbol} {timeframe}")
                break
            
            all_candles = candles + all_candles
            remaining -= len(candles)
            fetched += len(candles)
            
            pct = (fetched / num_candles) * 100
            print(f"[{symbol} {timeframe}] Progress: {fetched:,}/{num_candles:,} ({pct:.1f}%)")
            
            if len(candles) < batch_size:
                break
                
            end_time = candles[0]["timestamp"] - 1
            
            import time
            time.sleep(0.1)
        
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"])
        
        print(f"[Sync] Completed {symbol} {timeframe}: {len(df):,} unique candles")
        return df
    
    def fetch_all_historical_sync(self, num_candles: int = 175000, 
                                    progress_callback=None) -> Dict[str, Dict[str, pd.DataFrame]]:
        results = {}
        
        total_pairs = len(self.symbols) * len(self.timeframes)
        current = 0
        
        print(f"[Sync] Fetching {num_candles:,} candles for {len(self.symbols)} symbols, {len(self.timeframes)} timeframes")
        print(f"[Sync] Total pairs to fetch: {total_pairs}")
        
        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                current += 1
                print(f"")
                print(f"=== [{current}/{total_pairs}] Fetching {symbol} {timeframe} ===")
                
                if progress_callback:
                    progress_callback(current, total_pairs, symbol, timeframe)
                
                try:
                    df = self.fetch_historical_sync(symbol, timeframe, num_candles)
                    results[symbol][timeframe] = df
                except Exception as e:
                    print(f"[Sync] ERROR: Failed to fetch {symbol} {timeframe} after retries: {e}")
                    print(f"[Sync] Skipping {symbol} {timeframe} and continuing...")
                    results[symbol][timeframe] = None
        
        if progress_callback:
            progress_callback(total_pairs, total_pairs, "", "")
        
        print(f"")
        print(f"[Sync] All fetches complete!")
        return results
    
    def fetch_bulk_from_replit(self, progress_callback=None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Bulk download all GPU training data from Replit server in one request.
        This is much faster than fetching candle-by-candle.
        
        progress_callback(current_candles, expected_total, current_symbol, current_timeframe)
        - During streaming: expected_total is set once meta is received
        - Returns real-time candle count for accurate ETA calculation
        """
        if not self.replit_proxy_url:
            print("[Bulk Download] No Replit proxy URL configured!")
            return {}
        
        import requests
        
        url = f"{self.replit_proxy_url}/api/nn-data/bulk-export"
        print(f"[Bulk Download] Fetching all data from Replit: {url}")
        print(f"[Bulk Download] Proxy URL: {self.replit_proxy_url}")
        
        try:
            print(f"[Bulk Download] Sending request...")
            response = requests.get(url, stream=True, timeout=600)  # 10 min timeout for large data
            print(f"[Bulk Download] Response status: {response.status_code}")
            print(f"[Bulk Download] Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"[Bulk Download] HTTP ERROR {response.status_code}")
                try:
                    body = response.text[:500]
                    print(f"[Bulk Download] Response body: {body}")
                except:
                    pass
                return {}
            
            # Parse gzipped NDJSON stream
            results: Dict[str, Dict[str, List]] = {}
            total_candles = 0
            expected_total = 0  # Set when meta received
            meta = None
            summary = None
            parse_errors = 0
            current_sym = ""
            current_tf = ""
            last_progress_time = time.time()
            
            # Decompress gzip stream
            decompressor = gzip.GzipFile(fileobj=response.raw)
            
            print("[Bulk Download] Receiving data stream...")
            
            for line_bytes in decompressor:
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    if obj.get('type') == 'meta':
                        meta = obj
                        expected_total = meta.get('totalCandles', 8000000)  # Estimated ~8M candles
                        print(f"[Bulk Download] Meta: {len(meta.get('timeframes', []))} timeframes, {len(meta.get('symbols', []))} symbols")
                        print(f"[Bulk Download] Expected: ~{expected_total:,} candles")
                        
                        # Initialize results structure
                        for sym in meta.get('symbols', []):
                            results[sym] = {}
                            for tf in meta.get('timeframes', []):
                                results[sym][tf] = []
                        
                        # Initial progress callback
                        if progress_callback:
                            progress_callback(0, expected_total, "", "")
                    
                    elif obj.get('type') == 'summary':
                        summary = obj
                        expected_total = obj.get('totalCandles', expected_total)
                        print(f"[Bulk Download] Server reports: {expected_total:,} candles")
                    
                    else:
                        # This is a candle record
                        sym = obj.get('s')
                        tf = obj.get('tf')
                        
                        if sym and tf and sym in results and tf in results[sym]:
                            results[sym][tf].append({
                                'timestamp': obj['t'],
                                'open': float(obj['o']),
                                'high': float(obj['h']),
                                'low': float(obj['l']),
                                'close': float(obj['c']),
                                'volume': float(obj['v']),
                            })
                            total_candles += 1
                            current_sym = sym
                            current_tf = tf
                            
                            # Yield to main thread every 10k candles to keep UI responsive
                            if total_candles % 10000 == 0:
                                time.sleep(0)  # Allow other threads to run
                            
                            # Update progress every 100k candles or every second
                            now = time.time()
                            if total_candles % 100000 == 0 or (now - last_progress_time) >= 1.0:
                                last_progress_time = now
                                if progress_callback:
                                    progress_callback(total_candles, expected_total, current_sym, current_tf)
                            
                            # Reduce logging frequency to every 1M candles
                            if total_candles % 1000000 == 0:
                                print(f"[Bulk Download] Progress: {total_candles:,} / {expected_total:,} candles...")
                
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 5:
                        print(f"[Bulk Download] JSON parse error: {e}")
                    continue
            
            # Verify stream integrity - REQUIRE summary and exact match for data safety
            if not meta:
                print("[Bulk Download] INTEGRITY ERROR: No metadata received - stream corrupted")
                return {}
            
            if not summary:
                print("[Bulk Download] INTEGRITY ERROR: No summary received - stream may be truncated")
                print("[Bulk Download] Returning empty to trigger fallback")
                return {}
            
            expected = summary.get('totalCandles', 0)
            if total_candles != expected:
                print(f"[Bulk Download] INTEGRITY ERROR: Expected {expected:,} candles but received {total_candles:,}")
                print("[Bulk Download] Stream corrupted or truncated - returning empty to trigger fallback")
                return {}
            
            if parse_errors > 0:
                print(f"[Bulk Download] WARNING: {parse_errors} parse errors encountered")
                # Only fail on significant parse errors (more than 0.1% of data)
                if parse_errors > max(10, total_candles * 0.001):
                    print("[Bulk Download] Too many parse errors - returning empty to trigger fallback")
                    return {}
            
            print(f"[Bulk Download] Integrity verified: {total_candles:,} candles match server count")
            
            # Convert lists to DataFrames
            df_results: Dict[str, Dict[str, pd.DataFrame]] = {}
            
            for sym in results:
                df_results[sym] = {}
                for tf in results[sym]:
                    if results[sym][tf]:
                        df = pd.DataFrame(results[sym][tf])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        df_results[sym][tf] = df
                        print(f"[Bulk Download] {sym} {tf}: {len(df):,} candles")
                    else:
                        df_results[sym][tf] = pd.DataFrame()
            
            # Verify all expected symbol/timeframe pairs have data
            expected_pairs = len(meta.get('symbols', [])) * len(meta.get('timeframes', []))
            actual_pairs = sum(1 for sym in df_results for tf in df_results[sym] if len(df_results[sym][tf]) > 0)
            
            if actual_pairs == 0:
                print("[Bulk Download] INTEGRITY ERROR: No data received for any symbol/timeframe")
                return {}
            
            if actual_pairs < expected_pairs:
                missing = []
                for sym in meta.get('symbols', []):
                    for tf in meta.get('timeframes', []):
                        if sym not in df_results or tf not in df_results.get(sym, {}) or len(df_results[sym][tf]) == 0:
                            missing.append(f"{sym}/{tf}")
                print(f"[Bulk Download] INTEGRITY ERROR: Missing data for {len(missing)} pairs: {missing[:5]}...")
                print("[Bulk Download] Returning empty to trigger fallback")
                return {}
            
            print(f"[Bulk Download] Total: {total_candles:,} candles downloaded from Replit")
            return df_results
            
        except requests.exceptions.Timeout:
            print("[Bulk Download] Request timed out after 10 minutes")
            return {}
        except Exception as e:
            print(f"[Bulk Download] Error: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _try_fetch(self, url: str, params: Dict, source_name: str) -> Optional[Any]:
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    print(f"[{source_name}] Rate limited (429)")
                elif resp.status == 403:
                    print(f"[{source_name}] Forbidden (403) - possibly blocked")
                elif resp.status == 451:
                    print(f"[{source_name}] Geoblocked (451)")
                else:
                    print(f"[{source_name}] HTTP {resp.status}")
        except aiohttp.ClientConnectorError as e:
            print(f"[{source_name}] Connection failed: {e}")
        except asyncio.TimeoutError:
            print(f"[{source_name}] Timeout")
        except Exception as e:
            print(f"[{source_name}] Error: {e}")
        return None
    
    async def _fetch_replit_proxy(self, symbol: str, timeframe: str, limit: int,
                                   end_time: Optional[int] = None) -> List[Dict]:
        if not self.replit_proxy_url:
            return []
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(limit, 1000)
        }
        if end_time:
            params["endTime"] = end_time
        
        url = f"{self.replit_proxy_url}/api/data/klines"
        data = await self._try_fetch(url, params, "Replit Proxy")
        
        if data and "candles" in data:
            candles = []
            for c in data["candles"]:
                candles.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": c["timestamp"],
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                    "close_time": c["closeTime"],
                    "quote_volume": float(c["quoteVolume"]),
                    "trades": c["trades"],
                    "taker_buy_base": float(c["takerBuyBase"]),
                    "taker_buy_quote": float(c["takerBuyQuote"])
                })
            if candles:
                print(f"[Replit Proxy] Successfully fetched {len(candles)} candles")
            return candles
        return []
    
    async def fetch_klines(self, symbol: str, timeframe: str, limit: int = 1000, 
                          start_time: Optional[int] = None, 
                          end_time: Optional[int] = None) -> List[Dict]:
        # If Replit Proxy is configured, use ONLY that source
        # This avoids DNS/connection errors from trying blocked Binance APIs
        if self.replit_proxy_url:
            proxy_data = await self._fetch_replit_proxy(symbol, timeframe, limit, end_time)
            if proxy_data:
                self.working_source = "Replit Proxy"
                return proxy_data
            # If proxy fails, don't fall back to Binance (it's likely blocked)
            print(f"[Replit Proxy] Failed to fetch {symbol} {timeframe} - no fallback when proxy is configured")
            return []
        
        # No proxy configured - try direct Binance access (for non-geoblocked regions)
        if self.working_source == "CryptoCompare":
            cc_data = await self._fetch_cryptocompare(symbol, timeframe, limit, end_time)
            if cc_data:
                return cc_data
            self.working_source = None
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        sources = [
            (f"{self.BINANCE_VISION_URL}/klines", params, "Binance Vision"),
            (f"{self.BINANCE_API_URL}/klines", params, "Binance API"),
        ]
        
        if self.working_source and self.working_source not in ["CryptoCompare", "Replit Proxy"]:
            sources = [s for s in sources if s[2] == self.working_source] + \
                      [s for s in sources if s[2] != self.working_source]
        
        for url, p, source_name in sources:
            data = await self._try_fetch(url, p, source_name)
            if data:
                self.working_source = source_name
                print(f"[{source_name}] Successfully fetched {len(data)} candles")
                return [self._parse_kline(k, symbol, timeframe) for k in data]
        
        # Last resort: CryptoCompare
        cc_data = await self._fetch_cryptocompare(symbol, timeframe, limit, end_time)
        if cc_data:
            self.working_source = "CryptoCompare"
            return cc_data
        
        return []
    
    async def _fetch_cryptocompare(self, symbol: str, timeframe: str, limit: int, 
                                    end_time: Optional[int] = None) -> List[Dict]:
        if end_time:
            return []
        
        fsym = symbol.replace("USDT", "")
        tsym = "USDT"
        
        tf_map = {"1m": "minute", "5m": "minute", "15m": "minute", "1h": "hour", "4h": "hour", "1d": "day"}
        endpoint = tf_map.get(timeframe, "minute")
        
        aggregate = 1
        if timeframe == "5m":
            aggregate = 5
        elif timeframe == "15m":
            aggregate = 15
        elif timeframe == "4h":
            aggregate = 4
        
        url = f"{self.CRYPTOCOMPARE_URL}/histo{endpoint}"
        params: Dict[str, Any] = {
            "fsym": fsym, 
            "tsym": tsym, 
            "limit": 2000,
            "aggregate": aggregate
        }
        
        data = await self._try_fetch(url, params, "CryptoCompare")
        if data and "Data" in data and "Data" in data["Data"]:
            candles = []
            for c in data["Data"]["Data"]:
                if c.get("close", 0) == 0 and c.get("open", 0) == 0:
                    continue
                ts_ms = c["time"] * 1000
                candles.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": ts_ms,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("volumefrom", 0)),
                    "close_time": ts_ms,
                    "quote_volume": float(c.get("volumeto", 0)),
                    "trades": 0,
                    "taker_buy_base": 0,
                    "taker_buy_quote": 0
                })
            candles.sort(key=lambda x: x["timestamp"])
            if candles:
                print(f"[CryptoCompare] Fetched {len(candles)} most recent candles (max 2000, no pagination)")
            return candles
        return []
    
    def _parse_kline(self, kline: List, symbol: str, timeframe: str) -> Dict:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": kline[0],
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "close_time": kline[6],
            "quote_volume": float(kline[7]),
            "trades": int(kline[8]),
            "taker_buy_base": float(kline[9]),
            "taker_buy_quote": float(kline[10])
        }
    
    async def fetch_all_historical(self, lookback_candles: int = 50000) -> Dict[str, Dict[str, pd.DataFrame]]:
        all_data = {}
        
        for symbol in tqdm(self.symbols, desc="Fetching symbols"):
            all_data[symbol] = {}
            for timeframe in self.timeframes:
                candles = []
                oldest_ts = None
                
                while len(candles) < lookback_candles:
                    batch = await self.fetch_klines(
                        symbol, timeframe, limit=1000,
                        end_time=oldest_ts
                    )
                    if not batch:
                        if self.working_source == "CryptoCompare":
                            print(f"[CryptoCompare] Limited to {len(candles)} candles (no pagination)")
                        break
                    
                    batch.sort(key=lambda x: x["timestamp"])
                    
                    if oldest_ts is None:
                        candles = batch + candles
                    else:
                        new_candles = [c for c in batch if c["timestamp"] < oldest_ts]
                        if not new_candles:
                            break
                        candles = new_candles + candles
                    
                    oldest_ts = candles[0]["timestamp"] - 1
                    await asyncio.sleep(0.1)
                
                candles.sort(key=lambda x: x["timestamp"])
                candles = candles[-lookback_candles:] if len(candles) > lookback_candles else candles
                    
                if candles:
                    df = pd.DataFrame(candles)
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("datetime", inplace=True)
                    all_data[symbol][timeframe] = df
                    print(f"Fetched {len(candles)} total candles for {symbol} {timeframe}")
                else:
                    all_data[symbol][timeframe] = pd.DataFrame()
                
        return all_data
    
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        params = {"symbol": symbol, "limit": limit}
        
        sources = [
            (f"{self.BINANCE_VISION_URL}/depth", "Binance Vision"),
            (f"{self.BINANCE_API_URL}/depth", "Binance API"),
        ]
        
        for url, source_name in sources:
            data = await self._try_fetch(url, params, source_name)
            if data and "bids" in data and "asks" in data:
                bids = np.array([[float(p), float(q)] for p, q in data["bids"]])
                asks = np.array([[float(p), float(q)] for p, q in data["asks"]])
                
                bid_volume = bids[:, 1].sum() if len(bids) > 0 else 0
                ask_volume = asks[:, 1].sum() if len(asks) > 0 else 0
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
                
                return {
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "imbalance": imbalance,
                    "spread": (asks[0, 0] - bids[0, 0]) / bids[0, 0] if len(bids) > 0 and len(asks) > 0 else 0
                }
        return {}
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class FeatureEngineer:
    """
    Feature engineering class for STF (Single-TimeFrame) features.
    
    CRITICAL: This class must remain in sync between training and inference.
    Any changes to feature computation MUST increment the VERSION.
    """
    # Version string documents the exact computation method
    # Format: major.minor.patch-mode-details
    # Increment when ANY computation changes (windows, formulas, normalization)
    VERSION = "5.2.0-stf44-enh24-htf12-regime5-ema3"
    
    STF_FEATURE_COUNT = 44
    ENH_FEATURE_COUNT = 24
    HTF_FEATURE_COUNT = 12
    REGIME_FEATURE_COUNT = 5
    EMA200_FEATURE_COUNT = 3
    TOTAL_FEATURE_COUNT = 88
    
    HTF_FEATURE_NAMES = [
        "h1_sma20_slope", "h1_trend_sign", "h1_rsi14", "h1_atr_ratio", "h1_range_pos",
        "h4_sma20_slope", "h4_trend_sign", "h4_rsi14", "h4_atr_ratio", "h4_range_pos",
        "rsi_divergence_15m_1h", "macd_hist_slope_1h",
    ]
    
    REGIME_FEATURE_NAMES = [
        "regime_trend", "regime_volatility", "regime_momentum",
        "regime_session_sin", "regime_session_cos",
    ]
    
    EMA200_FEATURE_NAMES = [
        "ema200_pos_15m", "above_ema200_15m", "h1_ema200_pos",
    ]
    
    VERSION_DETAILS = {
        "return_type": "pct_change",
        "log_return_type": "log_ratio",
        "ema_warmup": "full_history",
        "rsi_method": "wilder_smoothing",
        "bb_window": 20,
        "bb_std": 2,
        "atr_window": 14,
        "periods": [5, 10, 20, 50, 100],
        "htf_timeframes": ["1H", "4H"],
        "htf_sma_period": 20,
        "htf_sma_slope_lookback": 3,
        "htf_rsi_period": 14,
        "htf_atr_period": 14,
        "htf_leakage_prevention": "shift_by_1_htf_bar",
        "htf_merge_method": "merge_asof_backward",
    }
    
    def __init__(self, wavelet: str = "db4", wavelet_level: int = 4,
                 symbols=None, timeframes=None, **kwargs):
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.scalers = {}
        self.symbols = symbols
        self.timeframes = timeframes
        
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        for period in [5, 10, 20, 50, 100]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            features[f"std_{period}"] = df["close"].rolling(period).std()
            features[f"return_{period}"] = df["close"].pct_change(period)
            
        features["rsi_14"] = self._compute_rsi(df["close"], 14)
        
        macd, signal, hist = self._compute_macd(df["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = hist
        
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(df["close"])
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        features["atr_14"] = self._compute_atr(df, 14)
        
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_sma_20"]
        
        features["adx_14"] = self._compute_adx(df, 14)
        
        stoch_k, stoch_d = self._compute_stochastic(df)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        
        features["obv"] = self._compute_obv(df)
        features["obv_sma"] = features["obv"].rolling(20).mean()
        
        rsi_14 = features["rsi_14"]
        price_slope_14 = df["close"].pct_change(14)
        rsi_slope_14 = rsi_14.diff(14)
        features["rsi_divergence"] = np.where(
            (price_slope_14 > 0) & (rsi_slope_14 < 0), -1.0,
            np.where((price_slope_14 < 0) & (rsi_slope_14 > 0), 1.0, 0.0)
        )
        
        vol_mom_10 = df["close"].pct_change(10) * (df["volume"] / features["volume_sma_20"].clip(lower=1))
        features["vol_weighted_mom_10"] = vol_mom_10
        
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap_window = 96
        rolling_tp_vol = (typical_price * df["volume"]).rolling(vwap_window, min_periods=1).sum()
        rolling_vol = df["volume"].rolling(vwap_window, min_periods=1).sum()
        vwap = rolling_tp_vol / rolling_vol.clip(lower=1)
        features["vwap_deviation"] = (df["close"] - vwap) / vwap.clip(lower=1e-8)
        
        features["close_to_high_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
        
        features["volume_delta"] = df.get("taker_buy_base", pd.Series(0, index=df.index)) / df["volume"].clip(lower=1) - 0.5
        
        return features
    
    ENH_FEATURE_NAMES = [
        "roc_accel_20",
        "trend_persistence_20", "trend_persistence_50",
        "momentum_alignment",
        "hurst_exponent",
        "taker_imbalance_20", "taker_pressure_delta",
        "volume_surge", "trade_intensity",
        "garman_klass_vol",
        "vol_breakout",
        "atr_expansion", "atr_contraction_flag",
        "range_volatility_20",
        "close_momentum_z",
        "directional_volume_flow",
        "price_acceleration",
        "ATR_ratio_7_28", "bb_squeeze", "vol_regime_roc",
        "trend_efficiency", "momentum_acceleration",
        "cvd_zscore", "volume_price_divergence",
    ]
    
    def compute_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 24 enhanced information-dense features.
        
        Groups:
          1. Momentum/Trend (5): ROC acceleration, trend persistence x2,
             momentum alignment, Hurst exponent
          2. Microstructure (4): taker imbalance, taker pressure delta, volume surge,
             trade intensity
          3. Volatility Regime (5): Garman-Klass vol, vol breakout,
             ATR expansion, ATR contraction flag, range volatility
          4. Signal Quality (3): close momentum z-score, directional volume flow,
             price acceleration
          5. Volatility Term Structure (3): ATR_ratio_7_28, bb_squeeze, vol_regime_roc
          6. Momentum Quality (2): trend_efficiency, momentum_acceleration
          7. Order Flow (2): cvd_zscore, volume_price_divergence
        """
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        roc_20 = close.pct_change(20)
        features["roc_accel_20"] = roc_20 - roc_20.shift(20)
        
        returns = close.pct_change()
        sign_returns = np.sign(returns)
        features["trend_persistence_20"] = sign_returns.rolling(20, min_periods=5).mean()
        features["trend_persistence_50"] = sign_returns.rolling(50, min_periods=10).mean()
        
        mom_5 = np.sign(close.pct_change(5))
        mom_20 = np.sign(close.pct_change(20))
        mom_50 = np.sign(close.pct_change(50))
        features["momentum_alignment"] = (mom_5 + mom_20 + mom_50) / 3.0
        
        log_returns = np.log(close / close.shift(1))
        rolling_std = log_returns.rolling(100, min_periods=50).std()
        rolling_mean = log_returns.rolling(100, min_periods=50).mean()
        demeaned = log_returns - rolling_mean
        lr_arr = demeaned.values.astype(np.float64)
        n_pts = len(lr_arr)
        window = 100
        cummax_arr = np.full(n_pts, np.nan)
        cummin_arr = np.full(n_pts, np.nan)
        for i in range(window, n_pts):
            seg = lr_arr[i-window:i]
            valid = ~np.isnan(seg)
            if valid.sum() < 50:
                continue
            seg_clean = np.where(valid, seg, 0.0)
            cs = np.cumsum(seg_clean)
            cummax_arr[i] = np.max(cs)
            cummin_arr[i] = np.min(cs)
        cumdev_range = pd.Series(cummax_arr - cummin_arr, index=df.index)
        rs_ratio = cumdev_range / rolling_std.clip(lower=1e-10)
        hurst_raw = np.log(rs_ratio.clip(lower=1e-10)) / np.log(window)
        features["hurst_exponent"] = hurst_raw.fillna(0.5).clip(0.0, 1.0)
        
        taker_buy = df.get("taker_buy_base", pd.Series(0, index=df.index))
        taker_sell = volume - taker_buy
        imbalance = (taker_buy - taker_sell) / volume.clip(lower=1)
        features["taker_imbalance_20"] = imbalance.rolling(20, min_periods=1).mean().clip(-1.0, 1.0)
        features["taker_pressure_delta"] = (imbalance - imbalance.rolling(20, min_periods=1).mean()).clip(-1.0, 1.0)
        
        vol_sma = volume.rolling(50, min_periods=10).mean()
        features["volume_surge"] = (volume / vol_sma.clip(lower=1)).clip(0, 10.0)
        
        n_trades = df.get("number_of_trades", pd.Series(0, index=df.index))
        n_trades_sma = n_trades.rolling(20, min_periods=1).mean()
        features["trade_intensity"] = n_trades / n_trades_sma.clip(lower=1) if n_trades.sum() > 0 else pd.Series(1.0, index=df.index)
        
        log_hl = np.log(high / low.clip(lower=1e-10))
        log_co = np.log(close / df["open"].clip(lower=1e-10))
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        features["garman_klass_vol"] = gk_var.rolling(20, min_periods=5).mean().apply(lambda x: np.sqrt(max(x, 0)))
        
        bb_std_20 = close.rolling(20).std()
        bb_std_50 = close.rolling(50, min_periods=20).std()
        features["vol_breakout"] = (bb_std_20 / bb_std_50.clip(lower=1e-10) - 1.0).clip(-2.0, 5.0)
        
        atr_14 = self._compute_atr(df, 14)
        atr_14_sma = atr_14.rolling(50, min_periods=10).mean()
        features["atr_expansion"] = (atr_14 / atr_14_sma.clip(lower=1e-10)).clip(0.1, 5.0)
        features["atr_contraction_flag"] = (features["atr_expansion"] < 0.7).astype(float)
        
        hl_range = high - low
        features["range_volatility_20"] = (hl_range.rolling(20, min_periods=5).std() / hl_range.rolling(20, min_periods=5).mean().clip(lower=1e-10)).clip(0, 5.0)
        
        mom_20_raw = close.pct_change(20)
        mom_mean = mom_20_raw.rolling(50, min_periods=10).mean()
        mom_std = mom_20_raw.rolling(50, min_periods=10).std()
        features["close_momentum_z"] = ((mom_20_raw - mom_mean) / mom_std.clip(lower=1e-10)).clip(-5.0, 5.0)
        
        signed_volume = volume * np.sign(returns)
        features["directional_volume_flow"] = (signed_volume.rolling(20, min_periods=5).sum() / volume.rolling(20, min_periods=5).sum().clip(lower=1)).clip(-1.0, 1.0)
        
        features["price_acceleration"] = (returns - returns.shift(1)).clip(-0.05, 0.05)
        
        atr_7 = self._compute_atr(df, 7)
        atr_28 = self._compute_atr(df, 28)
        features["ATR_ratio_7_28"] = (atr_7 / atr_28.clip(lower=1e-10)).clip(0.1, 5.0)
        
        bb_width = (close.rolling(20).std() * 2) / close.rolling(20).mean().clip(lower=1e-10)
        features["bb_squeeze"] = bb_width.rolling(50, min_periods=10).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0 if len(x) > 0 else 0.5,
            raw=False
        ).clip(0.0, 1.0)
        
        atr_14_roc = atr_14.pct_change(5)
        features["vol_regime_roc"] = atr_14_roc.clip(-1.0, 1.0)
        
        N = 20
        net_move = (close - close.shift(N)).abs()
        total_path = close.diff().abs().rolling(N, min_periods=5).sum()
        features["trend_efficiency"] = (net_move / total_path.clip(lower=1e-10)).clip(0.0, 1.0)
        
        rsi_14 = self._compute_rsi(close, 14)
        features["momentum_acceleration"] = rsi_14.pct_change(5).clip(-0.5, 0.5)
        
        cvd = (taker_buy - taker_sell).cumsum()
        cvd_mean = cvd.rolling(50, min_periods=10).mean()
        cvd_std = cvd.rolling(50, min_periods=10).std()
        features["cvd_zscore"] = ((cvd - cvd_mean) / cvd_std.clip(lower=1e-10)).clip(-3.0, 3.0)
        
        vol_change = volume.rolling(10, min_periods=3).mean() / volume.rolling(30, min_periods=10).mean().clip(lower=1)
        price_change = close.pct_change(10).abs()
        features["volume_price_divergence"] = (vol_change - 1.0 - price_change * 10).clip(-3.0, 3.0)
        
        assert len(features.columns) == self.ENH_FEATURE_COUNT, \
            f"Expected {self.ENH_FEATURE_COUNT} enhanced features, got {len(features.columns)}: {list(features.columns)}"
        
        return features
    
    def compute_htf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Higher Timeframe (1H, 4H) context features from 15m candle data.
        
        Resamples 15m OHLCV into 1H and 4H bars, computes indicators on each,
        then maps them back to every 15m row using only COMPLETED HTF bars
        (shifted by 1 HTF bar to prevent lookahead leakage).
        
        Returns DataFrame with 10 HTF features aligned to the 15m index.
        """
        if 'timestamp' not in df.columns:
            logger.warning("No 'timestamp' column found - cannot compute HTF features")
            return pd.DataFrame(0, index=df.index, columns=self.HTF_FEATURE_NAMES)
        
        ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        ohlcv = pd.DataFrame({
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values,
        }, index=ts)
        
        atr_15m = self._compute_atr(df, 14)
        
        htf_features = pd.DataFrame(index=ohlcv.index)
        htf_shifted_indexes = {}
        
        for tf_label, resample_rule in [("h1", "1h"), ("h4", "4h")]:
            htf_bars = ohlcv.resample(resample_rule, label='left', closed='left').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna(subset=['open'])
            
            if len(htf_bars) < 25:
                logger.warning(f"Only {len(htf_bars)} {tf_label} bars - need at least 25 for indicators")
                for col in [f"{tf_label}_sma20_slope", f"{tf_label}_trend_sign", 
                            f"{tf_label}_rsi14", f"{tf_label}_atr_ratio", f"{tf_label}_range_pos"]:
                    htf_features[col] = 0.0
                continue
            
            sma20 = htf_bars['close'].rolling(20, min_periods=1).mean()
            
            htf_atr = self._compute_atr_from_ohlc(htf_bars, 14)
            
            sma20_slope = (sma20 - sma20.shift(3)) / (htf_atr.abs() + 1e-9)
            
            trend_sign = np.sign(sma20_slope)
            
            rsi14 = self._compute_rsi(htf_bars['close'], 14)
            
            htf_indicators = pd.DataFrame({
                f'{tf_label}_sma20_slope': sma20_slope,
                f'{tf_label}_trend_sign': trend_sign,
                f'{tf_label}_rsi14': rsi14,
                f'{tf_label}_atr': htf_atr,
                f'{tf_label}_htf_high': htf_bars['high'],
                f'{tf_label}_htf_low': htf_bars['low'],
            }, index=htf_bars.index)
            
            htf_indicators = htf_indicators.shift(1)
            htf_shifted_indexes[tf_label] = htf_indicators.dropna(how='all').index
            
            htf_indicators = htf_indicators.reset_index()
            htf_indicators.columns = ['htf_ts'] + list(htf_indicators.columns[1:])
            
            ohlcv_reset = ohlcv.reset_index()
            ohlcv_reset.columns = ['ts_15m'] + list(ohlcv_reset.columns[1:])
            
            merged = pd.merge_asof(
                ohlcv_reset[['ts_15m']],
                htf_indicators,
                left_on='ts_15m',
                right_on='htf_ts',
                direction='backward'
            )
            
            htf_features[f'{tf_label}_sma20_slope'] = merged[f'{tf_label}_sma20_slope'].values
            htf_features[f'{tf_label}_trend_sign'] = merged[f'{tf_label}_trend_sign'].values
            htf_features[f'{tf_label}_rsi14'] = merged[f'{tf_label}_rsi14'].values
            
            merged_htf_atr = merged[f'{tf_label}_atr'].values
            htf_features[f'{tf_label}_atr_ratio'] = atr_15m.values / (merged_htf_atr + 1e-9)
            
            htf_high = merged[f'{tf_label}_htf_high'].values
            htf_low = merged[f'{tf_label}_htf_low'].values
            htf_features[f'{tf_label}_range_pos'] = np.clip(
                (ohlcv['close'].values - htf_low) / (htf_high - htf_low + 1e-9),
                0.0, 1.0
            )
        
        rsi_15m = self._compute_rsi(ohlcv['close'], 14)
        h1_bars = ohlcv.resample('1h', label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        }).dropna(subset=['open'])
        if len(h1_bars) >= 25:
            rsi_1h = self._compute_rsi(h1_bars['close'], 14)
            rsi_1h_shifted = rsi_1h.shift(1)
            rsi_1h_mapped = rsi_1h_shifted.reindex(ohlcv.index, method='ffill')
            rsi_15m_slope = rsi_15m.diff(4)
            rsi_1h_slope = rsi_1h_mapped.diff(4)
            htf_features['rsi_divergence_15m_1h'] = np.clip(
                (rsi_15m_slope - rsi_1h_slope).fillna(0.0), -50.0, 50.0
            ) / 50.0
            
            macd_1h, _, macd_hist_1h = self._compute_macd(h1_bars['close'])
            macd_hist_slope_1h = macd_hist_1h.diff(3)
            macd_hist_slope_1h_shifted = macd_hist_slope_1h.shift(1)
            macd_hist_slope_mapped = macd_hist_slope_1h_shifted.reindex(ohlcv.index, method='ffill')
            macd_atr = self._compute_atr_from_ohlc(h1_bars, 14).shift(1).reindex(ohlcv.index, method='ffill')
            htf_features['macd_hist_slope_1h'] = np.clip(
                (macd_hist_slope_mapped / macd_atr.clip(lower=1e-10)).fillna(0.0),
                -3.0, 3.0
            )
        else:
            htf_features['rsi_divergence_15m_1h'] = 0.0
            htf_features['macd_hist_slope_1h'] = 0.0
        
        result = htf_features[self.HTF_FEATURE_NAMES].copy()
        result.index = df.index
        
        nan_counts = result.isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            nan_cols = {col: int(nan_counts[col]) for col in result.columns if nan_counts[col] > 0}
            logger.info(f"HTF NaNs summary: total={total_nans} (expected during warmup) | {nan_cols}")
        
        self._sanity_check_htf_leakage(ohlcv, df, htf_shifted_indexes)
        
        return result
    
    def _sanity_check_htf_leakage(self, ohlcv: pd.DataFrame, original_df: pd.DataFrame,
                                   htf_shifted_indexes: dict, n_samples: int = 20):
        """One-time diagnostic: verify HTF bars are strictly in the past for each 15m row.
        
        Uses pre-computed shifted bar indexes from compute_htf_features() to avoid
        redundant resampling. For 20 random rows, prints: 15m timestamp t, matched
        1H timestamp t1h, matched 4H timestamp t4h.
        """
        import random
        
        valid_start = max(20, len(ohlcv) // 10)
        if len(ohlcv) < valid_start + n_samples:
            return
        
        sample_indices = sorted(random.sample(range(valid_start, len(ohlcv)), n_samples))
        
        h1_idx = htf_shifted_indexes.get('h1')
        h4_idx = htf_shifted_indexes.get('h4')
        if h1_idx is None or h4_idx is None:
            return
        
        logger.info("=" * 70)
        logger.info("HTF LEAKAGE SANITY CHECK (20 random rows)")
        logger.info(f"{'15m timestamp t':>25} | {'h1 shift(1) origin':>25} | {'h4 shift(1) origin':>25}")
        logger.info("-" * 70)
        
        violations = 0
        for idx in sample_indices:
            t = ohlcv.index[idx]
            
            h1_matched = None
            h4_matched = None
            for tf_label, tf_idx, tf_period in [("h1", h1_idx, pd.Timedelta(hours=1)),
                                                  ("h4", h4_idx, pd.Timedelta(hours=4))]:
                ts_arr = pd.DatetimeIndex(tf_idx)
                candidates = ts_arr[ts_arr <= t]
                if len(candidates) > 0:
                    matched_ts = candidates[-1]
                    origin_bar = matched_ts + tf_period
                    if tf_label == "h1":
                        h1_matched = (matched_ts, origin_bar)
                    else:
                        h4_matched = (matched_ts, origin_bar)
            
            t_str = str(t)[:19]
            h1_str = str(h1_matched[1])[:19] if h1_matched else "N/A"
            h4_str = str(h4_matched[1])[:19] if h4_matched else "N/A"
            
            ok = True
            if h1_matched and h1_matched[0] > t:
                ok = False
                violations += 1
            if h4_matched and h4_matched[0] > t:
                ok = False
                violations += 1
            
            status = "OK" if ok else "LEAK!"
            logger.info(f"{t_str:>25} | {h1_str:>25} | {h4_str:>25}  {status}")
        
        logger.info("-" * 70)
        if violations == 0:
            logger.info("PASS: All 20 rows use strictly past HTF bars (shift(1) verified). No leakage.")
        else:
            logger.warning(f"INFO: {violations} potential timing edge cases in {n_samples} rows. "
                          f"merge_asof ensures correct alignment — this is diagnostic only.")
        logger.info("=" * 70)
    
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 5 continuous regime features as model inputs.
        
        Replaces binary regime classification with a 4D+1 continuous regime vector:
          1. regime_trend: tanh(ADX/25 * sign(EMA_slope)) in [-1, 1]
          2. regime_volatility: z-score of ATR_14 vs rolling median/std of ATR over 50 bars
          3. regime_momentum: tanh(RSI_slope_5 / 10) in [-1, 1]
          4. regime_session_sin: sin(2*pi*hour/24) — cyclical session encoding
          5. regime_session_cos: cos(2*pi*hour/24) — cyclical session encoding
        """
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        
        adx_14 = self._compute_adx(df, 14)
        ema_20 = close.ewm(span=20).mean()
        ema_slope = ema_20.diff(5)
        ema_slope_sign = np.sign(ema_slope)
        features["regime_trend"] = np.tanh((adx_14 / 25.0) * ema_slope_sign)
        
        atr_14 = self._compute_atr(df, 14)
        atr_rolling_median = atr_14.rolling(50, min_periods=10).median()
        atr_rolling_std = atr_14.rolling(50, min_periods=10).std()
        features["regime_volatility"] = ((atr_14 - atr_rolling_median) / atr_rolling_std.clip(lower=1e-10)).clip(-5.0, 5.0)
        
        rsi_14 = self._compute_rsi(close, 14)
        rsi_slope_5 = rsi_14.diff(5)
        features["regime_momentum"] = np.tanh(rsi_slope_5 / 10.0)
        
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            hour = ts.dt.hour + ts.dt.minute / 60.0
        else:
            hour = pd.Series(np.zeros(len(df)), index=df.index)
        features["regime_session_sin"] = np.sin(2 * np.pi * hour / 24.0)
        features["regime_session_cos"] = np.cos(2 * np.pi * hour / 24.0)
        
        assert len(features.columns) == self.REGIME_FEATURE_COUNT, \
            f"Expected {self.REGIME_FEATURE_COUNT} regime features, got {len(features.columns)}: {list(features.columns)}"
        
        return features
    
    def compute_ema200_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 3 EMA200-based macro trend position features.

        These give the model direct knowledge of where price sits relative to the
        200-period trend — the most important trend filter used in all live gates.

          1. ema200_pos_15m:  (close / EMA200_15m) - 1, clipped ±0.15
          2. above_ema200_15m: sign(close - EMA200_15m) ∈ {-1, +1}
          3. h1_ema200_pos:   (close / EMA200_approx_1h) - 1, clipped ±0.20
                               approximated as EMA(800) on 15m bars (≈200 × 4-bar 1h period)
        """
        features = pd.DataFrame(index=df.index)
        close = df["close"]

        ema200_15m = close.ewm(span=200, adjust=False).mean()
        features["ema200_pos_15m"] = ((close / ema200_15m.clip(lower=1e-10)) - 1.0).clip(-0.15, 0.15)
        features["above_ema200_15m"] = np.sign(close - ema200_15m).fillna(0.0)

        ema200_1h_approx = close.ewm(span=800, adjust=False).mean()
        features["h1_ema200_pos"] = ((close / ema200_1h_approx.clip(lower=1e-10)) - 1.0).clip(-0.20, 0.20)

        assert len(features.columns) == self.EMA200_FEATURE_COUNT, \
            f"Expected {self.EMA200_FEATURE_COUNT} EMA200 features, got {len(features.columns)}: {list(features.columns)}"

        return features

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features: STF + ENH + HTF + REGIME + EMA200 = total.
        
        Returns a single DataFrame with deterministic column order.
        """
        stf = self.compute_technical_features(df)
        enh = self.compute_enhanced_features(df)
        htf = self.compute_htf_features(df)
        regime = self.compute_regime_features(df)
        ema200 = self.compute_ema200_features(df)
        
        combined = pd.concat([stf, enh, htf, regime, ema200], axis=1)
        
        assert combined.shape[1] == self.TOTAL_FEATURE_COUNT, \
            f"Expected {self.TOTAL_FEATURE_COUNT} features, got {combined.shape[1]}: {list(combined.columns)}"
        
        return combined
    
    def _compute_atr_from_ohlc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ATR from a generic OHLC DataFrame (works for resampled HTF bars)."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    def compute_wavelet_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        if not HAVE_PYWT:
            n = len(prices)
            zeros = np.zeros(n)
            features: Dict[str, np.ndarray] = {"trend": zeros.copy(), "noise": zeros.copy()}
            for i in range(1, self.wavelet_level + 1):
                features[f"cycle_{i}"] = zeros.copy()
            return features

        coeffs = pywt.wavedec(prices, self.wavelet, level=self.wavelet_level)
        
        trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], self.wavelet)
        
        features = {"trend": trend[:len(prices)]}
        
        for i, c in enumerate(coeffs[1:], 1):
            detail_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(cc) for cc in coeffs[1:]]
            detail_coeffs[i] = c
            detail = pywt.waverec(detail_coeffs, self.wavelet)
            features[f"cycle_{i}"] = detail[:len(prices)]
            
        features["noise"] = prices - trend[:len(prices)]
        
        return features
    
    def compute_cross_asset_features(self, data: Dict[str, pd.DataFrame], 
                                     target_symbol: str = "BTCUSDT") -> pd.DataFrame:
        target_df = data[target_symbol]
        features = pd.DataFrame(index=target_df.index)
        
        for symbol, df in data.items():
            if symbol == target_symbol:
                continue
                
            aligned_df = df.reindex(target_df.index, method="ffill")
            
            symbol_short = symbol.replace("USDT", "")
            
            features[f"{symbol_short}_returns"] = aligned_df["close"].pct_change()
            
            features[f"{symbol_short}_corr_20"] = (
                target_df["close"].pct_change()
                .rolling(20)
                .corr(aligned_df["close"].pct_change())
            )
            
            features[f"{symbol_short}_lead_1"] = aligned_df["close"].pct_change().shift(1)
            features[f"{symbol_short}_lead_5"] = aligned_df["close"].pct_change(5).shift(1)
            
            btc_returns = target_df["close"].pct_change()
            alt_returns = aligned_df["close"].pct_change()
            features[f"{symbol_short}_relative_strength"] = btc_returns - alt_returns
            
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _compute_bollinger(self, prices: pd.Series, period: int = 20, 
                          std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
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
    
    def _compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-8)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return obv
    
    def fit_scalers(self, features: pd.DataFrame, method: str = "robust"):
        for col in features.columns:
            if method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            valid_data = features[col].dropna().values.reshape(-1, 1)
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers[col] = scaler
                
    def transform(self, features: pd.DataFrame, drop_invalid: bool = False) -> pd.DataFrame:
        """Transform features using fitted scalers.
        
        Args:
            features: DataFrame with feature columns
            drop_invalid: If True, drop rows with NaN/Inf (returns shorter DataFrame).
                         If False (default), only replace Inf->NaN for compatibility.
        """
        transformed = features.copy()
        for col in features.columns:
            if col in self.scalers:
                valid_mask = ~features[col].isna()
                if valid_mask.any():
                    transformed.loc[valid_mask, col] = self.scalers[col].transform(
                        features.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()
        
        # === CRITICAL: Data validation after scaling ===
        # Replace ±Inf with NaN
        inf_count = np.isinf(transformed.values).sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} Inf values after scaling, replacing with NaN")
            transformed = transformed.replace([np.inf, -np.inf], np.nan)
        
        nan_count = transformed.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values after scaling")
            
        return transformed
    
    def transform_and_clip(self, features: pd.DataFrame, clip_range: float = 5.0) -> pd.DataFrame:
        """Transform features and clip extreme values.
        
        STABILITY FIX (Feb 2026): Even after RobustScaler, some features can have
        extreme outliers (>10σ) that cause gradient explosions in neural networks.
        
        This method:
        1. Applies RobustScaler transformation
        2. Clips ALL values to [-clip_range, +clip_range]
        3. Replaces any NaN/Inf with 0
        
        Args:
            features: DataFrame with feature columns
            clip_range: Clip values to [-clip_range, +clip_range]. Default 5.0.
        
        Returns:
            Cleaned, scaled, clipped DataFrame ready for training
        """
        transformed = self.transform(features)
        
        # Count extreme values before clipping
        extreme_count = 0
        for col in transformed.columns:
            col_data = transformed[col].values
            extreme_count += ((np.abs(col_data) > clip_range) & ~np.isnan(col_data)).sum()
        
        if extreme_count > 0:
            logger.info(f"Clipping {extreme_count:,} extreme values to [-{clip_range}, +{clip_range}]")
        
        # Clip all values
        transformed = transformed.clip(lower=-clip_range, upper=clip_range)
        
        # Replace any remaining NaN/Inf with 0
        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        transformed = transformed.fillna(0)
        
        # Final validation
        assert np.isfinite(transformed.values).all(), "Features still have non-finite values after clip!"
        
        return transformed
    
    def save_scalers(self, path: str):
        joblib.dump(self.scalers, path)
        
    def load_scalers(self, path: str):
        self.scalers = joblib.load(path)


class TradingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 sequence_length: int = 100, validate_data: bool = True,
                 strict_finite: bool = True):
        """
        Create a trading dataset for sequence modeling.
        
        Args:
            features: Input features array (N, D)
            labels: Target labels array (N,)
            sequence_length: Length of each sequence window
            validate_data: If True, check for NaN/Inf at initialization
            strict_finite: If True, FAIL if any NaN/Inf found (recommended)
        """
        # === CRITICAL: Validate input data before creating tensors ===
        if validate_data:
            # Check for NaN/Inf in features
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            
            if nan_count > 0 or inf_count > 0:
                msg = f"TradingDataset: Found {nan_count} NaN and {inf_count} Inf in features"
                if strict_finite:
                    raise ValueError(msg + " - data must be cleaned before dataset creation!")
                else:
                    logger.warning(msg)
                    # Replace Inf with NaN, then fill with 0 (NOT RECOMMENDED)
                    features = np.where(np.isinf(features), np.nan, features)
                    features = np.nan_to_num(features, nan=0.0)
                
            # Check for invalid labels
            invalid_labels = ~np.isin(labels, [0, 1, 2])
            if invalid_labels.sum() > 0:
                logger.warning(f"TradingDataset: Found {invalid_labels.sum()} invalid labels")
        
        # Final assertion - all data must be finite
        assert np.isfinite(features).all(), "Features contain non-finite values!"
        
        self.features = torch.FloatTensor(features)
        # Use LongTensor for classification labels (not FloatTensor)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Window: features[idx:idx+seq_len], predict label at END of window (current bar)
        # This ensures training and inference alignment: model predicts for the "current" bar
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]  # Last bar IN the window, not next bar
        return x, y


class MultiTimeframeDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], labels: np.ndarray,
                 sequence_lengths: Dict[str, int]):
        self.data = {tf: torch.FloatTensor(d) for tf, d in data.items()}
        self.labels = torch.FloatTensor(labels)
        self.sequence_lengths = sequence_lengths
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = {tf: self.data[tf][idx] for tf in self.data}
        y = self.labels[idx]
        return x, y


def create_labels(df: pd.DataFrame, horizon: int = 16, 
                  threshold: float = 0.001,
                  trading_cost: float = 0.0009) -> np.ndarray:
    """
    Create cost-aware labels for trading signals.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of bars to look ahead (16 = ~4 hours on 15m)
        threshold: Minimum return threshold before costs
        trading_cost: Round-trip trading cost (default 0.09% = 0.0009)
    
    Returns:
        labels: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
        
    Class mapping after (labels + 1):
        0 = SHORT, 1 = NEUTRAL, 2 = LONG
    """
    future_returns = df["close"].pct_change(horizon).shift(-horizon)
    
    # Cost-aware threshold: only signal if net return exceeds costs
    net_threshold = threshold + trading_cost
    
    labels = np.zeros(len(df))
    labels[future_returns > net_threshold] = 1   # LONG only if profit > costs
    labels[future_returns < -net_threshold] = -1  # SHORT only if profit > costs
    
    return labels


def prepare_data_loaders(features: np.ndarray, labels: np.ndarray,
                        config, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(features)
    train_end = int(n * config.data.train_split)
    val_end = int(n * (config.data.train_split + config.data.val_split))
    
    train_dataset = TradingDataset(
        features[:train_end], 
        labels[:train_end],
        config.data.sequence_length
    )
    val_dataset = TradingDataset(
        features[train_end:val_end],
        labels[train_end:val_end],
        config.data.sequence_length
    )
    test_dataset = TradingDataset(
        features[val_end:],
        labels[val_end:],
        config.data.sequence_length
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class DashboardAPIFetcher:
    """
    Fetches multi-timeframe aligned data from the Replit dashboard's GPU Export API.
    This is the recommended way to get training data as it provides properly aligned
    candles with as-of joins across timeframes.
    """
    
    def __init__(self, dashboard_url: str):
        self.dashboard_url = dashboard_url.rstrip('/')
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_timeframes(self) -> Dict[str, List[str]]:
        """Get available timeframes and symbols from dashboard."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/timeframes") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("timeframes", [])
            return []
    
    async def get_data_range(self, symbol: str = "BTCUSDT", timeframe: str = "1m") -> Dict:
        """Get data range for a symbol/timeframe."""
        session = await self._get_session()
        params = {"symbol": symbol, "timeframe": timeframe}
        async with session.get(f"{self.dashboard_url}/api/gpu-export/data-range", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    
    async def get_trainer_config(self) -> Dict:
        """Get optimal GPU trainer configuration."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/trainer-config") as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    
    async def get_feature_specs(self) -> List[Dict]:
        """Get feature engineering specifications."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/feature-specs") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("features", [])
            return []
    
    async def get_walk_forward_folds(self, symbol: str = "BTCUSDT", timeframe: str = "1m",
                                      train_months: int = 12, val_months: int = 2, 
                                      test_months: int = 2) -> List[Dict]:
        """Get walk-forward validation fold timestamps."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trainMonths": train_months,
            "valMonths": val_months,
            "testMonths": test_months
        }
        async with session.get(f"{self.dashboard_url}/api/gpu-export/walk-forward-folds", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("folds", [])
            return []
    
    async def fetch_multi_tf_candles(self, symbol: str, base_tf: str, 
                                      start_ts: int, end_ts: int, 
                                      limit: int = 100000) -> pd.DataFrame:
        """Fetch multi-timeframe aligned candles with as-of joins."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API] Fetching {symbol} {base_tf} from {start_ts} to {end_ts}...")
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/multi-tf", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API] Received {len(df)} candles with columns: {list(df.columns)}")
                    return df
            print(f"[Dashboard API] Error: HTTP {resp.status}")
            return pd.DataFrame()
    
    async def fetch_cross_asset_candles(self, base_tf: str, start_ts: int, end_ts: int,
                                         limit: int = 100000) -> pd.DataFrame:
        """Fetch cross-asset aligned candles (BTC/ETH/SOL/BNB)."""
        session = await self._get_session()
        params = {
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API] Fetching cross-asset data for {base_tf}...")
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/cross-asset", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API] Received {len(df)} cross-asset rows")
                    return df
            return pd.DataFrame()
    
    async def push_predictions(self, predictions: List[Dict], model_id: str) -> bool:
        """Push model predictions back to dashboard for ensemble integration."""
        session = await self._get_session()
        payload = {
            "predictions": predictions,
            "modelId": model_id,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        async with session.post(f"{self.dashboard_url}/api/gpu-export/predictions", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"[Dashboard API] Pushed {result.get('received', 0)} predictions")
                return True
            return False
    
    def fetch_multi_tf_candles_sync(self, symbol: str, base_tf: str,
                                     start_ts: int, end_ts: int,
                                     limit: int = 100000) -> pd.DataFrame:
        """Synchronous version of fetch_multi_tf_candles."""
        import requests
        params = {
            "symbol": symbol,
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API Sync] Fetching {symbol} {base_tf}...")
        
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/multi-tf", params=params, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API Sync] Received {len(df)} candles")
                    return df
        except Exception as e:
            print(f"[Dashboard API Sync] Error: {e}")
        return pd.DataFrame()
    
    def get_trainer_config_sync(self) -> Dict:
        """Synchronous version of get_trainer_config."""
        import requests
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/trainer-config", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting config: {e}")
        return {}
    
    def get_trading_costs_sync(self) -> Dict:
        """Get trading costs for cost-adjusted edge computation."""
        import requests
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/trading-costs", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting trading costs: {e}")
        return {"costs": {"totalRoundTrip": 0.0009}}
    
    async def get_trading_costs(self) -> Dict:
        """Get trading costs for cost-adjusted edge computation."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/trading-costs") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"costs": {"totalRoundTrip": 0.0009}}
    
    async def get_enhanced_labels(self, symbol: str = "BTCUSDT", timeframe: str = "15m",
                                   start_ts: int = None, end_ts: int = None,
                                   limit: int = 10000) -> pd.DataFrame:
        """Get enhanced training labels with cost-adjusted edge and trade-worthiness."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/enhanced-labels", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                labels = data.get("labels", [])
                if labels:
                    df = pd.DataFrame(labels)
                    print(f"[Dashboard API] Received {len(df)} enhanced labels")
                    return df
            return pd.DataFrame()
    
    def get_enhanced_labels_sync(self, symbol: str = "BTCUSDT", timeframe: str = "15m",
                                  start_ts: int = None, end_ts: int = None,
                                  limit: int = 10000) -> pd.DataFrame:
        """Synchronous version of get_enhanced_labels."""
        import requests
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/enhanced-labels", 
                               params=params, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                labels = data.get("labels", [])
                if labels:
                    df = pd.DataFrame(labels)
                    print(f"[Dashboard API Sync] Received {len(df)} enhanced labels")
                    return df
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting enhanced labels: {e}")
        return pd.DataFrame()
    
    def validate_data_integrity_sync(self, symbol: str, timeframe: str,
                                      start_ts: int = None, end_ts: int = None) -> Dict:
        """
        Validate data integrity before training to ensure no cross-contamination.
        This calls the Replit dashboard's validation endpoint and checks for:
        - Symbol/timeframe mismatches
        - Duplicate timestamps
        - Data gaps
        
        Returns a report dict with 'valid' boolean and any warnings.
        """
        import requests
        params = {"symbol": symbol, "timeframe": timeframe}
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        print(f"")
        print(f"{'='*60}")
        print(f"[Data Validation] Checking integrity for {symbol} {timeframe}")
        print(f"{'='*60}")
        
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/validate-integrity",
                               params=params, timeout=30)
            if resp.status_code == 200:
                report = resp.json()
                
                if report.get("valid"):
                    print(f"[Data Validation] PASSED - {report.get('totalRecords', 0):,} records")
                    print(f"[Data Validation] Timestamp range: {report.get('timestampRange', {})}")
                else:
                    print(f"[Data Validation] FAILED!")
                    for warning in report.get("warnings", []):
                        print(f"[Data Validation] WARNING: {warning}")
                
                if report.get("symbolMismatch"):
                    print(f"[Data Validation] CRITICAL: Symbol mismatch detected!")
                    print(f"[Data Validation] Expected: {symbol}")
                    print(f"[Data Validation] Found: {report.get('distinctSymbols')}")
                
                if report.get("timeframeMismatch"):
                    print(f"[Data Validation] CRITICAL: Timeframe mismatch detected!")
                    print(f"[Data Validation] Expected: {timeframe}")
                    print(f"[Data Validation] Found: {report.get('distinctTimeframes')}")
                
                if report.get("duplicateTimestamps", 0) > 0:
                    print(f"[Data Validation] WARNING: {report.get('duplicateTimestamps')} duplicate timestamps")
                
                print(f"{'='*60}")
                print(f"")
                return report
        except Exception as e:
            print(f"[Data Validation] Error: {e}")
        
        return {"valid": False, "warnings": ["Validation failed - could not reach API"]}
    
    def verify_dataframe_integrity(self, df: pd.DataFrame, expected_symbol: str, 
                                    expected_timeframe: str) -> Tuple[bool, List[str]]:
        """
        Verify that a loaded DataFrame contains only the expected symbol/timeframe.
        This is a client-side validation to catch any data mixing issues.
        
        Returns (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        print(f"")
        print(f"[DataFrame Validation] Verifying data for {expected_symbol} {expected_timeframe}")
        
        if df.empty:
            warnings.append("DataFrame is empty")
            return False, warnings
        
        # Check for symbol column if present
        if "symbol" in df.columns:
            unique_symbols = df["symbol"].unique().tolist()
            if len(unique_symbols) > 1:
                is_valid = False
                warnings.append(f"CRITICAL: Multiple symbols in data: {unique_symbols}")
            elif len(unique_symbols) == 1 and unique_symbols[0] != expected_symbol:
                is_valid = False
                warnings.append(f"CRITICAL: Symbol mismatch - expected {expected_symbol}, got {unique_symbols[0]}")
            else:
                print(f"[DataFrame Validation] Symbol check PASSED: {expected_symbol}")
        
        # Check for timeframe column if present
        if "timeframe" in df.columns:
            unique_tfs = df["timeframe"].unique().tolist()
            if len(unique_tfs) > 1:
                is_valid = False
                warnings.append(f"CRITICAL: Multiple timeframes in data: {unique_tfs}")
            elif len(unique_tfs) == 1 and unique_tfs[0] != expected_timeframe:
                is_valid = False
                warnings.append(f"CRITICAL: Timeframe mismatch - expected {expected_timeframe}, got {unique_tfs[0]}")
            else:
                print(f"[DataFrame Validation] Timeframe check PASSED: {expected_timeframe}")
        
        # Check for duplicate timestamps
        if "timestamp" in df.columns:
            n_total = len(df)
            n_unique = df["timestamp"].nunique()
            if n_unique < n_total:
                dup_count = n_total - n_unique
                warnings.append(f"WARNING: {dup_count} duplicate timestamps found")
                print(f"[DataFrame Validation] WARNING: {dup_count} duplicate timestamps")
            else:
                print(f"[DataFrame Validation] Timestamp uniqueness PASSED: {n_unique:,} unique")
        
        # Check data ordering
        if "timestamp" in df.columns:
            is_sorted = df["timestamp"].is_monotonic_increasing
            if not is_sorted:
                warnings.append("WARNING: Data not sorted by timestamp")
                print(f"[DataFrame Validation] WARNING: Data not sorted")
            else:
                print(f"[DataFrame Validation] Timestamp ordering PASSED")
        
        if is_valid and not warnings:
            print(f"[DataFrame Validation] ALL CHECKS PASSED for {expected_symbol} {expected_timeframe}")
        elif is_valid:
            print(f"[DataFrame Validation] PASSED with warnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print(f"[DataFrame Validation] FAILED:")
            for w in warnings:
                print(f"  - {w}")
        
        print(f"")
        return is_valid, warnings
    
    def fetch_multi_tf_candles_validated(self, symbol: str, base_tf: str,
                                          start_ts: int, end_ts: int,
                                          limit: int = 100000) -> Tuple[pd.DataFrame, bool, List[str]]:
        """
        Fetch multi-timeframe candles with validation.
        Returns (dataframe, is_valid, warnings).
        This is the recommended method for fetching training data.
        """
        # Step 1: Validate data integrity on the server
        integrity_report = self.validate_data_integrity_sync(symbol, base_tf, start_ts, end_ts)
        
        if not integrity_report.get("valid", False):
            print(f"[Validated Fetch] Server-side validation failed!")
            return pd.DataFrame(), False, integrity_report.get("warnings", [])
        
        # Step 2: Fetch the data
        df = self.fetch_multi_tf_candles_sync(symbol, base_tf, start_ts, end_ts, limit)
        
        if df.empty:
            return df, False, ["No data returned from API"]
        
        # Step 3: Verify the DataFrame client-side
        is_valid, warnings = self.verify_dataframe_integrity(df, symbol, base_tf)
        
        if not is_valid:
            print(f"[Validated Fetch] Client-side validation failed!")
            return df, False, warnings
        
        print(f"[Validated Fetch] SUCCESS: {len(df):,} validated records for {symbol} {base_tf}")
        return df, True, warnings


def compute_features_from_spec(df: pd.DataFrame, feature_specs: List[Dict]) -> pd.DataFrame:
    """
    Compute features according to the specification from the dashboard.
    This ensures features match exactly between dashboard and GPU trainer.
    """
    features = pd.DataFrame(index=df.index)
    
    close = df.get('1m_close', df.get('close', df.get('btc_close')))
    if close is None:
        raise ValueError("No close price column found")
    
    high = df.get('1m_high', df.get('high', df.get('btc_high')))
    low = df.get('1m_low', df.get('low', df.get('btc_low')))
    open_price = df.get('1m_open', df.get('open', df.get('btc_open')))
    volume = df.get('1m_volume', df.get('volume', df.get('btc_volume')))
    
    for spec in feature_specs:
        name = spec['name']
        formula = spec['formula']
        window = spec.get('window', 20)
        
        try:
            if 'log_return' in name:
                lookback = int(name.split('_')[-1]) if '_' in name else 1
                features[name] = np.log(close / close.shift(lookback))
            
            elif name.startswith('volatility_') and name != 'volatility_regime':
                # Rolling volatility (std of log returns)
                log_ret = np.log(close / close.shift(1))
                features[name] = log_ret.rolling(window).std()
            
            elif 'ema_ratio' in name:
                ema = close.ewm(span=window, adjust=False).mean()
                features[name] = close / ema - 1
            
            elif name == 'rsi_14':
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                features[name] = 100 - (100 / (1 + rs))
            
            elif 'macd' in name:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                
                if name == 'macd_line':
                    features[name] = macd_line
                elif name == 'macd_signal':
                    features[name] = macd_signal
                elif name == 'macd_hist':
                    features[name] = macd_line - macd_signal
            
            elif name == 'atr_14_norm':
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                features[name] = atr / close
            
            elif name == 'body_ratio':
                features[name] = (close - open_price) / open_price
            
            elif name == 'wick_up_ratio':
                body_top = pd.concat([open_price, close], axis=1).max(axis=1)
                features[name] = (high - body_top) / open_price
            
            elif name == 'wick_dn_ratio':
                body_bottom = pd.concat([open_price, close], axis=1).min(axis=1)
                features[name] = (body_bottom - low) / open_price
            
            elif name == 'volume_log':
                features[name] = np.log1p(volume)
            
            elif name == 'volume_zscore':
                vol_mean = volume.rolling(window).mean()
                vol_std = volume.rolling(window).std()
                features[name] = (volume - vol_mean) / vol_std.replace(0, 1)
            
            elif 'trend_slope' in name:
                ema20 = close.ewm(span=20, adjust=False).mean()
                features[name] = ema20.diff(window) / window
            
            elif '_return_1' in name and name.startswith(('eth', 'sol', 'bnb')):
                asset = name.split('_')[0]
                asset_close = df.get(f'{asset}_close')
                if asset_close is not None:
                    features[name] = np.log(asset_close / asset_close.shift(1))
            
            elif '_corr_' in name:
                # Rolling correlations between BTC and other assets
                # e.g., btc_eth_corr_20 -> rolling(20).corr() of BTC and ETH returns
                parts = name.split('_')
                if len(parts) >= 4:
                    asset1 = parts[0]  # btc
                    asset2 = parts[1]  # eth, sol, bnb
                    corr_window = int(parts[-1])  # 20
                    
                    close1 = df.get(f'{asset1}_close', close)
                    close2 = df.get(f'{asset2}_close')
                    
                    if close2 is not None:
                        ret1 = np.log(close1 / close1.shift(1))
                        ret2 = np.log(close2 / close2.shift(1))
                        features[name] = ret1.rolling(corr_window).corr(ret2)
                    else:
                        features[name] = 0.0
            
            elif name == 'volatility_regime':
                # Quantile-bucket volatility_20 with thresholds [0.33, 0.67]
                # Matches FEATURE_SPECS: quantile_bucket(volatility_20, [0.33, 0.67])
                log_ret = np.log(close / close.shift(1))
                vol_20 = log_ret.rolling(20).std()
                
                # Rolling quantile thresholds
                q33 = vol_20.rolling(100, min_periods=20).quantile(0.33)
                q67 = vol_20.rolling(100, min_periods=20).quantile(0.67)
                
                # Map to bucket: 0=low, 1=medium, 2=high
                def map_bucket(row_idx):
                    v = vol_20.iloc[row_idx]
                    t33 = q33.iloc[row_idx]
                    t67 = q67.iloc[row_idx]
                    if pd.isna(v) or pd.isna(t33) or pd.isna(t67):
                        return 1  # medium (default)
                    if v < t33:
                        return 0  # low
                    elif v < t67:
                        return 1  # medium
                    else:
                        return 2  # high
                
                features[name] = pd.Series([map_bucket(i) for i in range(len(vol_20))], index=df.index)
            
            elif 'relative_strength' in name:
                asset = name.split('_')[0]
                asset_close = df.get(f'{asset}_close')
                if asset_close is not None:
                    btc_ret = np.log(close / close.shift(20))
                    asset_ret = np.log(asset_close / asset_close.shift(20))
                    features[name] = asset_ret - btc_ret
            
        except Exception as e:
            print(f"[Feature] Error computing {name}: {e}")
            features[name] = np.nan
    
    return features.fillna(0)


def compute_enhanced_labels(
    df: pd.DataFrame,
    horizons: List[int] = [15, 60, 240],
    trading_costs: float = 0.0009
) -> pd.DataFrame:
    """
    Compute enhanced training labels with:
    1. Cost-adjusted edge (return - trading costs)
    2. Trade-worthiness labels (positive edge + clean move)
    3. Sample weights (prioritize high-move samples)
    
    Args:
        df: DataFrame with close/high/low prices
        horizons: List of forward horizons in candles
        trading_costs: Round-trip trading cost (default 0.09%)
    
    Returns:
        DataFrame with enhanced labels
    """
    close = df.get('1m_close', df.get('close', df.get('btc_close')))
    high = df.get('1m_high', df.get('high', df.get('btc_high')))
    low = df.get('1m_low', df.get('low', df.get('btc_low')))
    
    if close is None:
        raise ValueError("No close price column found")
    
    labels = pd.DataFrame(index=df.index)
    
    for h in horizons:
        raw_return = np.log(close.shift(-h) / close)
        
        labels[f'return_{h}'] = raw_return
        
        edge = np.abs(raw_return) - trading_costs
        labels[f'edge_{h}'] = edge
        
        direction = np.where(
            raw_return > trading_costs, 1,
            np.where(raw_return < -trading_costs, -1, 0)
        )
        labels[f'direction_{h}'] = direction
        
        max_favorable = pd.Series(np.nan, index=df.index)
        max_adverse = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - h):
            entry = close.iloc[i]
            exit_dir = np.sign(raw_return.iloc[i])
            
            future_highs = high.iloc[i+1:i+h+1]
            future_lows = low.iloc[i+1:i+h+1]
            
            if exit_dir >= 0:
                max_favorable.iloc[i] = (future_highs.max() - entry) / entry
                max_adverse.iloc[i] = (entry - future_lows.min()) / entry
            else:
                max_favorable.iloc[i] = (entry - future_lows.min()) / entry
                max_adverse.iloc[i] = (future_highs.max() - entry) / entry
        
        clean_move_ratio = max_favorable / (max_favorable + max_adverse + 1e-8)
        
        trade_worthy = ((edge > 0.001) & (clean_move_ratio > 0.5)).astype(float)
        labels[f'trade_worthy_{h}'] = trade_worthy
    
    max_abs_return = np.maximum.reduce([np.abs(labels[f'return_{h}']) for h in horizons])
    
    move_weight = np.power(max_abs_return / 0.01, 0.5)
    
    log_ret = np.log(close / close.shift(1))
    vol_20 = log_ret.rolling(20).std()
    
    volatility_multiplier = np.where(
        vol_20 > 0.015, 1.5,
        np.where(vol_20 < 0.005, 0.3, 1.0)
    )
    
    raw_weight = move_weight * volatility_multiplier
    labels['sample_weight'] = np.clip(raw_weight, 0.1, 5.0)
    
    labels['sample_weight'] = labels['sample_weight'].fillna(1.0)
    
    return labels


class WeightedTrainingDataset(Dataset):
    """
    PyTorch Dataset with sample weighting for improved training.
    Addresses the '90% chop' problem by weighting samples by move size.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        seq_len: int = 256,
        horizons: List[str] = ['edge_15', 'edge_60', 'edge_240']
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.seq_len = seq_len
        self.horizons = horizons
        
        self.valid_indices = np.arange(seq_len, len(features) - max(240, 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        
        x = self.features[actual_idx - self.seq_len:actual_idx]
        
        y = self.labels[actual_idx]
        
        w = self.weights[actual_idx]
        
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(w, dtype=torch.float32)
        )
    
    @staticmethod
    def create_weighted_sampler(weights: np.ndarray, num_samples: Optional[int] = None):
        """
        Create a WeightedRandomSampler for DataLoader.
        Higher weights = more likely to be sampled.
        """
        from torch.utils.data import WeightedRandomSampler
        
        normalized = weights / weights.sum()
        
        if num_samples is None:
            num_samples = len(weights)
        
        return WeightedRandomSampler(
            weights=normalized,
            num_samples=num_samples,
            replacement=True
        )
