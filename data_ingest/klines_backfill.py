"""15-minute kline backfill into the existing `candles` table (CDN-sourced)."""
from __future__ import annotations

import logging

from . import binance_api, db

log = logging.getLogger(__name__)

BAR_MS = 15 * 60 * 1000
FLUSH = 5000


def backfill_symbol(symbol: str, start_ts_ms: int | None = None,
                    historical: bool = True) -> tuple[int, int]:
    """Backfill 15m candles. By default (historical=True) iterates from 2019-09
    forward, letting ON CONFLICT DO NOTHING dedupe. Set historical=False to
    only fetch bars after the existing max."""
    existing_min, existing_max, existing_count = db.get_existing_15m_range(symbol)
    if start_ts_ms is None and not historical:
        start_ts_ms = (existing_max + BAR_MS) if existing_max else None

    inserted_total = 0
    buf: list[tuple] = []
    for (ts, o, h, l, c, v, _cnt, _tbv) in binance_api.iter_klines(
            symbol, "15m", min_ts_ms=start_ts_ms):
        buf.append((ts, o, h, l, c, v))
        if len(buf) >= FLUSH:
            inserted_total += db.upsert_candles_15m(symbol, buf)
            buf.clear()
    if buf:
        inserted_total += db.upsert_candles_15m(symbol, buf)

    _, _, total_after = db.get_existing_15m_range(symbol)
    log.info("%s 15m: inserted=%d total=%d", symbol, inserted_total, total_after)
    return inserted_total, total_after
