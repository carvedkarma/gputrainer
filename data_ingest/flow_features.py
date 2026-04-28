"""1-minute klines (CDN) -> 15-minute order-flow features.

See docstring in original task spec. Aggregator computes per-bar:
  cvd_delta, aggressor_ratio, total_volume, taker_buy_volume,
  trade_count, trade_intensity, large_trade_count, large_trade_imbalance,
  liquidation_proxy.

`large_trade_count` and `liquidation_proxy` are 1m-kline proxies, NOT
true tick-level aggTrades counts. The proxy is documented in the audit
report.
"""
from __future__ import annotations

import logging
import time
from collections import deque

from . import binance_api, db

log = logging.getLogger(__name__)

BAR_MS = 15 * 60 * 1000
WINDOW_1M = 1440  # 24h
WARMUP_MIN = 240  # 4h
LARGE_VOL_MULT = 5.0
LIQ_RANGE_MULT = 4.0
FLUSH = 5000


def _median(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def backfill_flow(symbol: str, start_ts_ms: int | None = None) -> tuple[int, int]:
    existing_min, existing_max, existing_count = db.get_existing_flow_range(symbol)
    if start_ts_ms is None:
        if existing_max is None:
            mn15, _, _ = db.get_existing_15m_range(symbol)
            start_ts_ms = mn15
        else:
            start_ts_ms = existing_max + BAR_MS
    if start_ts_ms is None:
        start_ts_ms = 0  # take everything CDN provides

    log.info("%s flow: starting from %s",
             symbol,
             time.strftime("%Y-%m-%d", time.gmtime(start_ts_ms / 1000)) if start_ts_ms else "earliest")

    vol_buf: deque[float] = deque(maxlen=WINDOW_1M)
    rng_buf: deque[float] = deque(maxlen=WINDOW_1M)

    bar_open_ts = -1
    bar_taker_buy = 0.0
    bar_total_vol = 0.0
    bar_trades = 0
    bar_large_count = 0
    bar_large_imbalance = 0.0
    bar_liq_count = 0
    bar_subbar_count = 0

    cur_med_vol = 0.0
    cur_med_rng = 0.0
    minutes_since_med = 60

    pending: list[tuple] = []
    inserted_total = 0
    minute_count = 0

    def _flush_bar():
        nonlocal bar_open_ts, bar_taker_buy, bar_total_vol, bar_trades
        nonlocal bar_large_count, bar_large_imbalance, bar_liq_count, bar_subbar_count
        if bar_open_ts < 0 or bar_subbar_count == 0:
            return
        if len(vol_buf) < WARMUP_MIN:
            bar_open_ts = -1
            bar_taker_buy = 0.0; bar_total_vol = 0.0; bar_trades = 0
            bar_large_count = 0; bar_large_imbalance = 0.0; bar_liq_count = 0
            bar_subbar_count = 0
            return
        taker_sell = bar_total_vol - bar_taker_buy
        cvd_delta = bar_taker_buy - taker_sell
        agg = bar_taker_buy / bar_total_vol if bar_total_vol > 0 else 0.5
        intensity = bar_trades / 15.0
        pending.append((
            symbol, bar_open_ts,
            cvd_delta, agg, bar_total_vol, bar_taker_buy,
            bar_trades, intensity,
            bar_large_count, bar_large_imbalance, bar_liq_count,
        ))
        bar_open_ts = -1
        bar_taker_buy = 0.0; bar_total_vol = 0.0; bar_trades = 0
        bar_large_count = 0; bar_large_imbalance = 0.0; bar_liq_count = 0
        bar_subbar_count = 0

    for (ts, o, h, l, c, v, trades, taker_buy) in binance_api.iter_klines(
            symbol, "1m", min_ts_ms=start_ts_ms):
        rng = h - l

        minutes_since_med += 1
        if minutes_since_med >= 60 and len(vol_buf) >= WARMUP_MIN:
            cur_med_vol = _median(list(vol_buf))
            cur_med_rng = _median(list(rng_buf))
            minutes_since_med = 0

        is_large = cur_med_vol > 0 and v > LARGE_VOL_MULT * cur_med_vol
        is_liq = (cur_med_rng > 0 and rng > LIQ_RANGE_MULT * cur_med_rng
                  and cur_med_vol > 0 and v > LARGE_VOL_MULT * cur_med_vol)

        bar_ts = ts - (ts % BAR_MS)
        if bar_open_ts < 0:
            bar_open_ts = bar_ts
        elif bar_ts != bar_open_ts:
            _flush_bar()
            bar_open_ts = bar_ts

        bar_total_vol += v
        bar_taker_buy += taker_buy
        bar_trades += trades
        bar_subbar_count += 1
        if is_large:
            bar_large_count += 1
            bar_large_imbalance += (taker_buy - (v - taker_buy))
        if is_liq:
            bar_liq_count += 1

        vol_buf.append(v)
        rng_buf.append(rng)
        minute_count += 1

        if len(pending) >= FLUSH:
            inserted_total += db.upsert_flow_features(pending)
            pending.clear()
        if minute_count % 200000 == 0:
            log.info("%s flow: %dM 1m bars processed, last=%s",
                     symbol, minute_count // 1_000_000,
                     time.strftime("%Y-%m-%d", time.gmtime(ts / 1000)))

    _flush_bar()
    if pending:
        inserted_total += db.upsert_flow_features(pending)
        pending.clear()

    _, _, total_after = db.get_existing_flow_range(symbol)
    log.info("%s flow: inserted=%d total=%d", symbol, inserted_total, total_after)
    return inserted_total, total_after
