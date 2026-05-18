"""Lower-timeframe execution module.

After a 15m ENTER signal qualifies, this module attempts to improve fill price
using 1m or 3m microstructure data. It does NOT change TP/SL geometry — only
the entry price.

Default behavior: if pullback + confirmation never occur within the execution
window, the trade is SKIPPED (no market fallback unless explicitly enabled).
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

log = logging.getLogger("Execution")


@dataclass
class ExecutionResult:
    executed: bool
    entry_price: float
    signal_price: float
    side: str
    symbol: str
    pullback_hit: bool
    confirmation_hit: bool
    method: str
    cost_improvement_bps: float
    elapsed_seconds: float
    reason: str


class ExecutionModule:
    def __init__(
        self,
        exec_tf: str = "3m",
        exec_window_minutes: int = 15,
        pullback_atr_frac: float = 0.20,
        confirm_indicator: str = "vwap",
        allow_market_fallback: bool = False,
        blocking_window: bool = False,
        fetcher=None,
    ):
        self.exec_tf = exec_tf
        self.exec_window_minutes = exec_window_minutes
        self.pullback_atr_frac = pullback_atr_frac
        self.confirm_indicator = confirm_indicator
        self.allow_market_fallback = allow_market_fallback
        self.blocking_window = bool(blocking_window)
        self.fetcher = fetcher

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical = (df['high'] + df['low'] + df['close']) / 3.0
        cum_tp_vol = (typical * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        return cum_tp_vol / cum_vol.clip(lower=1e-12)

    def _compute_ema(self, series: pd.Series, period: int = 20) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _check_pullback_long(self, candle: dict, signal_price: float, pullback_level: float) -> bool:
        return candle['low'] <= pullback_level

    def _check_pullback_short(self, candle: dict, signal_price: float, pullback_level: float) -> bool:
        return candle['high'] >= pullback_level

    def _check_confirmation_long(self, candle: dict, indicator_value: float) -> bool:
        return candle['close'] > indicator_value and candle['close'] > candle['open']

    def _check_confirmation_short(self, candle: dict, indicator_value: float) -> bool:
        return candle['close'] < indicator_value and candle['close'] < candle['open']

    def attempt_entry(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        atr: float,
        p_enter: float,
        dry_run_candles: Optional[pd.DataFrame] = None,
    ) -> ExecutionResult:
        log.info(f"[EXEC] {symbol} {side} — attempting improved entry on {self.exec_tf}")
        log.info(f"  Signal price: {signal_price:.2f} | ATR: {atr:.2f} | p_enter: {p_enter:.4f}")
        log.info(f"  Pullback target: {self.pullback_atr_frac:.2f} ATR = {self.pullback_atr_frac * atr:.2f}")
        log.info(f"  Confirm indicator: {self.confirm_indicator} | Window: {self.exec_window_minutes}min")
        log.info(f"  Mode: {'BLOCKING_WINDOW' if self.blocking_window else 'SNAPSHOT_NON_BLOCKING'}")

        pullback_dist = self.pullback_atr_frac * atr
        if side == "LONG":
            pullback_level = signal_price - pullback_dist
        else:
            pullback_level = signal_price + pullback_dist

        log.info(f"  Pullback level: {pullback_level:.2f}")

        start_time = time.time()
        pullback_hit = False
        confirmation_hit = False
        entry_price = signal_price

        if dry_run_candles is not None:
            result = self._simulate_on_candles(
                dry_run_candles, symbol, side, signal_price, atr,
                pullback_level, start_time
            )
            return result

        result = self._live_execution(
            symbol, side, signal_price, atr, pullback_level, start_time
        )
        return result

    def _simulate_on_candles(
        self,
        candles: pd.DataFrame,
        symbol: str,
        side: str,
        signal_price: float,
        atr: float,
        pullback_level: float,
        start_time: float,
    ) -> ExecutionResult:
        pullback_hit = False
        confirmation_hit = False
        entry_price = signal_price

        if self.confirm_indicator == "vwap":
            indicator = self._compute_vwap(candles)
        else:
            indicator = self._compute_ema(candles['close'], 20)

        max_candles = self.exec_window_minutes // self._tf_minutes()

        for i in range(min(max_candles, len(candles))):
            row = candles.iloc[i]
            candle = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
            }
            ind_val = float(indicator.iloc[i]) if i < len(indicator) else signal_price

            if not pullback_hit:
                if side == "LONG":
                    pullback_hit = self._check_pullback_long(candle, signal_price, pullback_level)
                else:
                    pullback_hit = self._check_pullback_short(candle, signal_price, pullback_level)

            if pullback_hit and not confirmation_hit:
                if side == "LONG":
                    confirmation_hit = self._check_confirmation_long(candle, ind_val)
                else:
                    confirmation_hit = self._check_confirmation_short(candle, ind_val)

                if confirmation_hit:
                    entry_price = float(candle['close'])
                    break

        elapsed = time.time() - start_time
        executed = confirmation_hit

        if not executed and self.allow_market_fallback:
            executed = True
            entry_price = float(candles.iloc[min(max_candles - 1, len(candles) - 1)]['close'])
            method = "market_fallback"
        elif executed:
            method = "pullback_confirm"
        else:
            method = "missed"

        if side == "LONG":
            improvement_bps = (signal_price - entry_price) / signal_price * 10000
        else:
            improvement_bps = (entry_price - signal_price) / signal_price * 10000

        reason = self._build_reason(pullback_hit, confirmation_hit, method)
        method_label = "entry_improvement_missed" if method == "missed" else method
        log.info(f"  [EXEC RESULT] {method_label} | pullback={'HIT' if pullback_hit else 'MISS'} | "
                 f"confirm={'HIT' if confirmation_hit else 'MISS'} | "
                 f"entry={entry_price:.2f} vs signal={signal_price:.2f} | "
                 f"improvement={improvement_bps:+.1f} bps")

        return ExecutionResult(
            executed=executed, entry_price=entry_price, signal_price=signal_price,
            side=side, symbol=symbol, pullback_hit=pullback_hit,
            confirmation_hit=confirmation_hit, method=method,
            cost_improvement_bps=improvement_bps, elapsed_seconds=elapsed,
            reason=reason,
        )

    def _live_execution(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        atr: float,
        pullback_level: float,
        start_time: float,
    ) -> ExecutionResult:
        if self.fetcher is None:
            log.warning("No fetcher configured for live execution — skipping")
            return ExecutionResult(
                executed=False, entry_price=signal_price, signal_price=signal_price,
                side=side, symbol=symbol, pullback_hit=False, confirmation_hit=False,
                method="no_fetcher", cost_improvement_bps=0, elapsed_seconds=0,
                reason="No data fetcher configured for live execution",
            )

        # Non-blocking snapshot mode (default): evaluate a single fresh lower-TF
        # snapshot and return immediately so one symbol cannot stall the entire
        # multi-asset run loop for exec_window_minutes.
        if not self.blocking_window:
            try:
                candles_raw = self.fetcher.fetch_klines_sync(symbol, self.exec_tf, limit=30)
                if not candles_raw:
                    return ExecutionResult(
                        executed=False, entry_price=signal_price, signal_price=signal_price,
                        side=side, symbol=symbol, pullback_hit=False, confirmation_hit=False,
                        method="no_data", cost_improvement_bps=0.0,
                        elapsed_seconds=max(time.time() - start_time, 0.0),
                        reason="No lower-TF data available for snapshot execution check",
                    )

                df = pd.DataFrame(candles_raw)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                if len(df) == 0:
                    return ExecutionResult(
                        executed=False, entry_price=signal_price, signal_price=signal_price,
                        side=side, symbol=symbol, pullback_hit=False, confirmation_hit=False,
                        method="no_data", cost_improvement_bps=0.0,
                        elapsed_seconds=max(time.time() - start_time, 0.0),
                        reason="Lower-TF dataframe is empty in snapshot execution check",
                    )

                if self.confirm_indicator == "vwap":
                    indicator = self._compute_vwap(df)
                else:
                    indicator = self._compute_ema(df['close'], 20)

                latest = df.iloc[-1]
                candle = {
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest.get('volume', 0)),
                }
                ind_val = float(indicator.iloc[-1]) if len(indicator) > 0 else float(candle['close'])

                if side == "LONG":
                    pullback_hit = self._check_pullback_long(candle, signal_price, pullback_level)
                else:
                    pullback_hit = self._check_pullback_short(candle, signal_price, pullback_level)

                confirmation_hit = False
                if pullback_hit:
                    if side == "LONG":
                        confirmation_hit = self._check_confirmation_long(candle, ind_val)
                    else:
                        confirmation_hit = self._check_confirmation_short(candle, ind_val)

                entry_price = signal_price
                if confirmation_hit:
                    entry_price = float(candle['close'])
                    method = "pullback_confirm"
                    executed = True
                elif self.allow_market_fallback:
                    entry_price = float(candle['close'])
                    method = "market_fallback"
                    executed = True
                else:
                    method = "missed"
                    executed = False

                if side == "LONG":
                    improvement_bps = (signal_price - entry_price) / signal_price * 10000
                else:
                    improvement_bps = (entry_price - signal_price) / signal_price * 10000

                elapsed = time.time() - start_time
                reason = self._build_reason(pullback_hit, confirmation_hit, method)
                method_label = "entry_improvement_missed" if method == "missed" else method
                log.info(
                    "  [EXEC RESULT] %s | pullback=%s | confirm=%s | "
                    "entry=%.2f vs signal=%.2f | improvement=%+.1f bps | elapsed=%.0fs",
                    method_label,
                    "HIT" if pullback_hit else "MISS",
                    "HIT" if confirmation_hit else "MISS",
                    entry_price,
                    signal_price,
                    improvement_bps,
                    elapsed,
                )
                return ExecutionResult(
                    executed=executed, entry_price=entry_price, signal_price=signal_price,
                    side=side, symbol=symbol, pullback_hit=bool(pullback_hit),
                    confirmation_hit=bool(confirmation_hit), method=method,
                    cost_improvement_bps=float(improvement_bps), elapsed_seconds=float(elapsed),
                    reason=reason,
                )
            except Exception as e:
                log.warning(f"  Execution snapshot error: {e}")
                return ExecutionResult(
                    executed=False, entry_price=signal_price, signal_price=signal_price,
                    side=side, symbol=symbol, pullback_hit=False, confirmation_hit=False,
                    method="snapshot_error", cost_improvement_bps=0.0,
                    elapsed_seconds=max(time.time() - start_time, 0.0),
                    reason=f"Snapshot execution error: {e}",
                )

        pullback_hit = False
        confirmation_hit = False
        entry_price = signal_price
        poll_interval = self._tf_minutes() * 60
        deadline = start_time + self.exec_window_minutes * 60

        all_candles = []

        while time.time() < deadline:
            try:
                candles_raw = self.fetcher.fetch_klines_sync(symbol, self.exec_tf, limit=30)
                if not candles_raw:
                    time.sleep(poll_interval / 2)
                    continue

                df = pd.DataFrame(candles_raw)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].astype(float)

                if self.confirm_indicator == "vwap":
                    indicator = self._compute_vwap(df)
                else:
                    indicator = self._compute_ema(df['close'], 20)

                latest = df.iloc[-1]
                candle = {
                    'open': float(latest['open']), 'high': float(latest['high']),
                    'low': float(latest['low']), 'close': float(latest['close']),
                    'volume': float(latest.get('volume', 0)),
                }
                ind_val = float(indicator.iloc[-1])

                if not pullback_hit:
                    if side == "LONG":
                        pullback_hit = self._check_pullback_long(candle, signal_price, pullback_level)
                    else:
                        pullback_hit = self._check_pullback_short(candle, signal_price, pullback_level)

                if pullback_hit and not confirmation_hit:
                    if side == "LONG":
                        confirmation_hit = self._check_confirmation_long(candle, ind_val)
                    else:
                        confirmation_hit = self._check_confirmation_short(candle, ind_val)

                    if confirmation_hit:
                        entry_price = float(candle['close'])
                        break

            except Exception as e:
                log.warning(f"  Execution poll error: {e}")

            time.sleep(poll_interval)

        elapsed = time.time() - start_time
        executed = confirmation_hit

        if not executed and self.allow_market_fallback:
            executed = True
            method = "market_fallback"
        elif executed:
            method = "pullback_confirm"
        else:
            method = "missed"

        if side == "LONG":
            improvement_bps = (signal_price - entry_price) / signal_price * 10000
        else:
            improvement_bps = (entry_price - signal_price) / signal_price * 10000

        reason = self._build_reason(pullback_hit, confirmation_hit, method)
        method_label = "entry_improvement_missed" if method == "missed" else method
        log.info(f"  [EXEC RESULT] {method_label} | pullback={'HIT' if pullback_hit else 'MISS'} | "
                 f"confirm={'HIT' if confirmation_hit else 'MISS'} | "
                 f"entry={entry_price:.2f} vs signal={signal_price:.2f} | "
                 f"improvement={improvement_bps:+.1f} bps | elapsed={elapsed:.0f}s")

        return ExecutionResult(
            executed=executed, entry_price=entry_price, signal_price=signal_price,
            side=side, symbol=symbol, pullback_hit=pullback_hit,
            confirmation_hit=confirmation_hit, method=method,
            cost_improvement_bps=improvement_bps, elapsed_seconds=elapsed,
            reason=reason,
        )

    def _tf_minutes(self) -> int:
        if self.exec_tf == "1m":
            return 1
        elif self.exec_tf == "3m":
            return 3
        elif self.exec_tf == "5m":
            return 5
        return 3

    def _build_reason(self, pullback_hit: bool, confirmation_hit: bool, method: str) -> str:
        parts = []
        if method == "pullback_confirm":
            parts.append(f"Pullback {self.pullback_atr_frac:.0%} ATR hit")
            parts.append(f"{self.confirm_indicator.upper()} reclaim confirmed")
            parts.append("Entered on improved fill")
        elif method == "market_fallback":
            if pullback_hit:
                parts.append("Pullback hit but no confirmation")
            else:
                parts.append("No pullback within window")
            parts.append("Market fallback entry")
        elif method == "missed":
            if not pullback_hit:
                parts.append(f"No pullback ({self.pullback_atr_frac:.0%} ATR) within {self.exec_window_minutes}min")
            else:
                parts.append(f"Pullback hit but {self.confirm_indicator.upper()} not reclaimed")
            parts.append("Trade skipped")
        return " | ".join(parts)
