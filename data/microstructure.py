"""
Market Microstructure Data Pipeline

Fetches and processes institutional-grade market data:
- Funding rates (historical)
- Open interest (with change tracking)
- Liquidation data
- Order book depth/imbalance
- Taker buy/sell flow
- Long/short ratio

This data provides the "why" behind price moves, not just the moves themselves.
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

class MicrostructureDataFetcher:
    """Fetches market microstructure data from Binance Futures API."""
    
    BINANCE_FUTURES_URL = "https://fapi.binance.com"
    
    def __init__(self, symbol: str = "BTCUSDT", replit_proxy_url: Optional[str] = None):
        self.symbol = symbol
        self.replit_proxy_url = replit_proxy_url
        
    def _fetch_with_retry(self, url: str, params: Dict = None, retries: int = 3) -> Optional[Any]:
        """Fetch with exponential backoff retry."""
        for i in range(retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 2 ** i
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text[:100]}")
            except Exception as e:
                logger.warning(f"Request error (attempt {i+1}): {e}")
                if i < retries - 1:
                    time.sleep(2 ** i)
        return None
    
    def fetch_funding_rate_history(self, limit: int = 1000, 
                                    start_time: Optional[int] = None,
                                    end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Returns DataFrame with columns:
        - timestamp: Funding time in ms
        - funding_rate: The funding rate
        - mark_price: Mark price at funding time
        """
        url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        params = {"symbol": self.symbol, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            records.append({
                "timestamp": item["fundingTime"],
                "funding_rate": float(item["fundingRate"]),
                "mark_price": float(item.get("markPrice", 0))
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_open_interest_history(self, period: str = "5m", limit: int = 500,
                                     start_time: Optional[int] = None,
                                     end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical open interest.
        
        Args:
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            
        Returns DataFrame with columns:
        - timestamp: Time in ms
        - open_interest: Open interest in contracts
        - open_interest_value: Open interest in USDT
        """
        url = f"{self.BINANCE_FUTURES_URL}/futures/data/openInterestHist"
        params = {
            "symbol": self.symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            records.append({
                "timestamp": item["timestamp"],
                "open_interest": float(item["sumOpenInterest"]),
                "open_interest_value": float(item["sumOpenInterestValue"])
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_long_short_ratio_history(self, period: str = "5m", limit: int = 500,
                                        start_time: Optional[int] = None,
                                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical long/short account ratio.
        
        Returns DataFrame with columns:
        - timestamp
        - long_short_ratio: Ratio of long to short accounts
        - long_account: % of accounts that are long
        - short_account: % of accounts that are short
        """
        url = f"{self.BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
        params = {
            "symbol": self.symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            records.append({
                "timestamp": item["timestamp"],
                "long_short_ratio": float(item["longShortRatio"]),
                "long_account": float(item["longAccount"]),
                "short_account": float(item["shortAccount"])
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_top_trader_ratio_history(self, period: str = "5m", limit: int = 500,
                                        account_type: str = "position",
                                        start_time: Optional[int] = None,
                                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical top trader long/short ratio.
        
        Args:
            account_type: "position" for position ratio, "account" for account ratio
            
        Returns DataFrame with columns:
        - timestamp
        - top_long_short_ratio
        - top_long_account
        - top_short_account
        """
        if account_type == "position":
            url = f"{self.BINANCE_FUTURES_URL}/futures/data/topLongShortPositionRatio"
        else:
            url = f"{self.BINANCE_FUTURES_URL}/futures/data/topLongShortAccountRatio"
            
        params = {
            "symbol": self.symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            records.append({
                "timestamp": item["timestamp"],
                "top_long_short_ratio": float(item["longShortRatio"]),
                "top_long_account": float(item["longAccount"]),
                "top_short_account": float(item["shortAccount"])
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_taker_buy_sell_volume(self, period: str = "5m", limit: int = 500,
                                     start_time: Optional[int] = None,
                                     end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch taker buy/sell volume (aggressive order flow).
        
        Returns DataFrame with columns:
        - timestamp
        - buy_volume: Taker buy volume
        - sell_volume: Taker sell volume  
        - buy_sell_ratio: buy_volume / sell_volume
        - net_taker_flow: buy_volume - sell_volume
        """
        url = f"{self.BINANCE_FUTURES_URL}/futures/data/takerlongshortRatio"
        params = {
            "symbol": self.symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            buy_vol = float(item["buyVol"])
            sell_vol = float(item["sellVol"])
            records.append({
                "timestamp": item["timestamp"],
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "buy_sell_ratio": float(item["buySellRatio"]),
                "net_taker_flow": buy_vol - sell_vol
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_order_book_depth(self, depth: int = 50) -> Dict:
        """
        Fetch current order book depth.
        
        Returns dict with:
        - bids: List of [price, qty]
        - asks: List of [price, qty]
        - spread: Bid-ask spread
        - imbalance: (bid_volume - ask_volume) / total_volume
        - depth_ratio: Volume at best bid/ask
        """
        url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/depth"
        params = {"symbol": self.symbol, "limit": depth}
        
        data = self._fetch_with_retry(url, params)
        if not data:
            return {}
        
        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
        
        if not bids or not asks:
            return {}
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        spread_pct = spread / best_bid if best_bid > 0 else 0
        
        bid_volume = sum(b[1] for b in bids[:10])
        ask_volume = sum(a[1] for a in asks[:10])
        total_volume = bid_volume + ask_volume
        
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "bid_volume_10": bid_volume,
            "ask_volume_10": ask_volume,
            "imbalance": imbalance,
            "depth_ratio": bid_volume / ask_volume if ask_volume > 0 else 1.0
        }
    
    def fetch_recent_liquidations(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent liquidation orders.
        
        Returns DataFrame with columns:
        - timestamp
        - side: "BUY" (short liquidation) or "SELL" (long liquidation)
        - price
        - quantity
        - value: price * quantity
        """
        url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/forceOrders"
        params = {"symbol": self.symbol, "limit": min(limit, 1000)}
        
        data = self._fetch_with_retry(url, params)
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            price = float(item["price"])
            qty = float(item["origQty"])
            records.append({
                "timestamp": item["time"],
                "side": item["side"],
                "price": price,
                "quantity": qty,
                "value": price * qty
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def fetch_mark_index_prices(self) -> Dict:
        """
        Fetch current mark price, index price, and basis.
        
        Returns dict with:
        - mark_price
        - index_price
        - basis: (mark - index) / index (premium/discount)
        - last_funding_rate
        - next_funding_time
        """
        url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"
        params = {"symbol": self.symbol}
        
        data = self._fetch_with_retry(url, params)
        if not data:
            return {}
        
        mark = float(data["markPrice"])
        index = float(data["indexPrice"])
        
        return {
            "mark_price": mark,
            "index_price": index,
            "basis": (mark - index) / index if index > 0 else 0,
            "last_funding_rate": float(data["lastFundingRate"]),
            "next_funding_time": data["nextFundingTime"],
            "interest_rate": float(data.get("interestRate", 0))
        }


class MicrostructureFeatureEngine:
    """
    Computes microstructure features for ML training.
    
    Features computed:
    - Funding rate features (current, momentum, deviation from mean)
    - Open interest features (level, change, momentum)
    - Long/short ratio features (level, extreme detection)
    - Taker flow features (imbalance, momentum)
    - Order book features (spread, imbalance, depth)
    - Liquidation features (volume, ratio, cascade detection)
    """
    
    def __init__(self):
        self.lookback_periods = [5, 15, 60, 240]  # 5m, 15m, 1h, 4h in candles
        
    def compute_funding_features(self, funding_df: pd.DataFrame, 
                                  candle_timestamps: pd.Series) -> pd.DataFrame:
        """
        Compute funding rate features aligned to candle timestamps.
        
        Returns DataFrame with columns:
        - funding_rate: Current funding rate
        - funding_rate_8h_avg: 8-hour average
        - funding_rate_deviation: Current vs 8h avg
        - funding_rate_momentum: Rate of change
        - funding_rate_extreme: Boolean for extreme funding
        """
        if funding_df.empty:
            return pd.DataFrame(index=candle_timestamps)
        
        funding_df = funding_df.sort_values("timestamp")
        
        features = []
        for ts in candle_timestamps:
            mask = funding_df["timestamp"] <= ts
            recent = funding_df[mask].tail(8)  # Last 8 funding periods = 1 day
            
            if len(recent) == 0:
                features.append({
                    "funding_rate": 0,
                    "funding_rate_8h_avg": 0,
                    "funding_rate_deviation": 0,
                    "funding_rate_momentum": 0,
                    "funding_rate_extreme": 0
                })
                continue
            
            current = recent["funding_rate"].iloc[-1]
            avg_8h = recent["funding_rate"].tail(3).mean()  # 3 periods = 24h
            avg_daily = recent["funding_rate"].mean()
            
            features.append({
                "funding_rate": current * 100,  # Scale to percentage
                "funding_rate_8h_avg": avg_8h * 100,
                "funding_rate_deviation": (current - avg_daily) / max(abs(avg_daily), 1e-6),
                "funding_rate_momentum": (current - recent["funding_rate"].iloc[0]) * 100 if len(recent) > 1 else 0,
                "funding_rate_extreme": 1 if abs(current) > 0.001 else 0  # >0.1% is extreme
            })
        
        return pd.DataFrame(features, index=candle_timestamps)
    
    def compute_oi_features(self, oi_df: pd.DataFrame,
                             candle_timestamps: pd.Series) -> pd.DataFrame:
        """
        Compute open interest features.
        
        Returns DataFrame with columns:
        - oi_value: Current OI in USDT
        - oi_change_1h: 1-hour % change
        - oi_change_4h: 4-hour % change
        - oi_momentum: Rate of change
        - oi_relative: Current vs 24h avg
        """
        if oi_df.empty:
            return pd.DataFrame(index=candle_timestamps)
        
        oi_df = oi_df.sort_values("timestamp")
        
        features = []
        for ts in candle_timestamps:
            mask = oi_df["timestamp"] <= ts
            recent = oi_df[mask].tail(288)  # 24h of 5m data
            
            if len(recent) == 0:
                features.append({
                    "oi_value": 0,
                    "oi_change_1h": 0,
                    "oi_change_4h": 0,
                    "oi_momentum": 0,
                    "oi_relative": 1
                })
                continue
            
            current = recent["open_interest_value"].iloc[-1]
            avg_24h = recent["open_interest_value"].mean()
            
            oi_1h_ago = recent["open_interest_value"].iloc[-12] if len(recent) >= 12 else current
            oi_4h_ago = recent["open_interest_value"].iloc[-48] if len(recent) >= 48 else current
            
            features.append({
                "oi_value": current / 1e9,  # Scale to billions
                "oi_change_1h": (current - oi_1h_ago) / oi_1h_ago if oi_1h_ago > 0 else 0,
                "oi_change_4h": (current - oi_4h_ago) / oi_4h_ago if oi_4h_ago > 0 else 0,
                "oi_momentum": (current - recent["open_interest_value"].iloc[0]) / recent["open_interest_value"].iloc[0] if len(recent) > 1 else 0,
                "oi_relative": current / avg_24h if avg_24h > 0 else 1
            })
        
        return pd.DataFrame(features, index=candle_timestamps)
    
    def compute_flow_features(self, taker_df: pd.DataFrame,
                               candle_timestamps: pd.Series) -> pd.DataFrame:
        """
        Compute taker flow features (aggressive order imbalance).
        
        Returns DataFrame with columns:
        - taker_ratio: Buy/sell ratio
        - net_taker_flow: Normalized net flow
        - flow_momentum: Change in flow
        - flow_extreme: Extreme flow detection
        """
        if taker_df.empty:
            return pd.DataFrame(index=candle_timestamps)
        
        taker_df = taker_df.sort_values("timestamp")
        
        features = []
        for ts in candle_timestamps:
            mask = taker_df["timestamp"] <= ts
            recent = taker_df[mask].tail(48)  # 4 hours of 5m data
            
            if len(recent) == 0:
                features.append({
                    "taker_ratio": 1,
                    "net_taker_flow": 0,
                    "flow_momentum": 0,
                    "flow_extreme": 0
                })
                continue
            
            current = recent.iloc[-1]
            avg_ratio = recent["buy_sell_ratio"].mean()
            std_ratio = recent["buy_sell_ratio"].std()
            
            z_score = (current["buy_sell_ratio"] - avg_ratio) / max(std_ratio, 0.01)
            
            features.append({
                "taker_ratio": current["buy_sell_ratio"],
                "net_taker_flow": current["net_taker_flow"] / 1e6,  # Scale to millions
                "flow_momentum": (current["buy_sell_ratio"] - recent["buy_sell_ratio"].iloc[0]) if len(recent) > 1 else 0,
                "flow_extreme": 1 if abs(z_score) > 2 else 0
            })
        
        return pd.DataFrame(features, index=candle_timestamps)
    
    def compute_ls_ratio_features(self, ls_df: pd.DataFrame,
                                   candle_timestamps: pd.Series) -> pd.DataFrame:
        """
        Compute long/short ratio features.
        
        Returns DataFrame with columns:
        - ls_ratio: Current long/short ratio
        - ls_deviation: Deviation from mean
        - ls_extreme: Extreme ratio detection
        - crowd_sentiment: Simplified sentiment score
        """
        if ls_df.empty:
            return pd.DataFrame(index=candle_timestamps)
        
        ls_df = ls_df.sort_values("timestamp")
        
        features = []
        for ts in candle_timestamps:
            mask = ls_df["timestamp"] <= ts
            recent = ls_df[mask].tail(288)  # 24h of 5m data
            
            if len(recent) == 0:
                features.append({
                    "ls_ratio": 1,
                    "ls_deviation": 0,
                    "ls_extreme": 0,
                    "crowd_sentiment": 0
                })
                continue
            
            current = recent["long_short_ratio"].iloc[-1]
            avg = recent["long_short_ratio"].mean()
            std = recent["long_short_ratio"].std()
            
            z_score = (current - avg) / max(std, 0.01)
            
            sentiment = 0
            if current > 2:
                sentiment = -1  # Crowd too long, contrarian bearish
            elif current < 0.5:
                sentiment = 1   # Crowd too short, contrarian bullish
            
            features.append({
                "ls_ratio": current,
                "ls_deviation": z_score,
                "ls_extreme": 1 if abs(z_score) > 2 else 0,
                "crowd_sentiment": sentiment
            })
        
        return pd.DataFrame(features, index=candle_timestamps)
    
    def compute_liquidation_features(self, liq_df: pd.DataFrame,
                                      candle_timestamps: pd.Series,
                                      window_ms: int = 300000) -> pd.DataFrame:
        """
        Compute liquidation features.
        
        Args:
            window_ms: Time window in milliseconds (default 5 minutes)
        
        Returns DataFrame with columns:
        - liq_long_volume: Long liquidation volume
        - liq_short_volume: Short liquidation volume
        - liq_imbalance: (long - short) / total
        - liq_cascade_risk: Cascade liquidation risk score
        """
        features = []
        
        for ts in candle_timestamps:
            if liq_df.empty:
                features.append({
                    "liq_long_volume": 0,
                    "liq_short_volume": 0,
                    "liq_imbalance": 0,
                    "liq_cascade_risk": 0
                })
                continue
            
            mask = (liq_df["timestamp"] >= ts - window_ms) & (liq_df["timestamp"] <= ts)
            window = liq_df[mask]
            
            if len(window) == 0:
                features.append({
                    "liq_long_volume": 0,
                    "liq_short_volume": 0,
                    "liq_imbalance": 0,
                    "liq_cascade_risk": 0
                })
                continue
            
            long_liqs = window[window["side"] == "SELL"]["value"].sum()
            short_liqs = window[window["side"] == "BUY"]["value"].sum()
            total = long_liqs + short_liqs
            
            imbalance = (long_liqs - short_liqs) / total if total > 0 else 0
            
            cascade_risk = min(total / 1e7, 1.0)  # Normalize to [0, 1]
            
            features.append({
                "liq_long_volume": long_liqs / 1e6,
                "liq_short_volume": short_liqs / 1e6,
                "liq_imbalance": imbalance,
                "liq_cascade_risk": cascade_risk
            })
        
        return pd.DataFrame(features, index=candle_timestamps)
    
    def compute_all_microstructure_features(
        self,
        candle_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        taker_df: pd.DataFrame,
        ls_df: pd.DataFrame,
        liq_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute all microstructure features for a candle DataFrame.
        
        Returns DataFrame with all microstructure features aligned to candle timestamps.
        """
        timestamps = candle_df["timestamp"]
        
        funding_features = self.compute_funding_features(funding_df, timestamps)
        oi_features = self.compute_oi_features(oi_df, timestamps)
        flow_features = self.compute_flow_features(taker_df, timestamps)
        ls_features = self.compute_ls_ratio_features(ls_df, timestamps)
        liq_features = self.compute_liquidation_features(liq_df, timestamps)
        
        all_features = pd.concat([
            funding_features.reset_index(drop=True),
            oi_features.reset_index(drop=True),
            flow_features.reset_index(drop=True),
            ls_features.reset_index(drop=True),
            liq_features.reset_index(drop=True)
        ], axis=1)
        
        all_features = all_features.fillna(0)
        
        return all_features


def fetch_all_microstructure_data(
    symbol: str = "BTCUSDT",
    lookback_days: int = 30,
    replit_proxy_url: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all microstructure data for a symbol.
    
    Returns dict with:
    - funding: Funding rate history
    - oi: Open interest history
    - taker: Taker buy/sell volume
    - ls_ratio: Long/short ratio
    - top_trader: Top trader ratio
    - liquidations: Recent liquidations
    """
    fetcher = MicrostructureDataFetcher(symbol, replit_proxy_url)
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
    
    logger.info(f"Fetching microstructure data for {symbol}, {lookback_days} days")
    
    data = {}
    
    logger.info("Fetching funding rate history...")
    data["funding"] = fetcher.fetch_funding_rate_history(
        limit=1000, start_time=start_time, end_time=end_time
    )
    
    logger.info("Fetching open interest history...")
    data["oi"] = fetcher.fetch_open_interest_history(
        period="5m", limit=500, start_time=start_time, end_time=end_time
    )
    
    logger.info("Fetching taker volume history...")
    data["taker"] = fetcher.fetch_taker_buy_sell_volume(
        period="5m", limit=500, start_time=start_time, end_time=end_time
    )
    
    logger.info("Fetching long/short ratio history...")
    data["ls_ratio"] = fetcher.fetch_long_short_ratio_history(
        period="5m", limit=500, start_time=start_time, end_time=end_time
    )
    
    logger.info("Fetching top trader ratio history...")
    data["top_trader"] = fetcher.fetch_top_trader_ratio_history(
        period="5m", limit=500, start_time=start_time, end_time=end_time
    )
    
    logger.info("Fetching recent liquidations...")
    data["liquidations"] = fetcher.fetch_recent_liquidations(limit=1000)
    
    for key, df in data.items():
        if not df.empty:
            logger.info(f"  {key}: {len(df)} records")
        else:
            logger.warning(f"  {key}: No data")
    
    return data
