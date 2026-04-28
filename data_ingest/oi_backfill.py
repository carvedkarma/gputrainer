"""Open-interest backfill via data.binance.vision metrics archives.

The metrics CSV provides 5-minute cadence OI going back roughly to the
symbol's listing date — far better than the live REST endpoint's 30-day
limit. Stored in `open_interest_history` with period='5m'.
"""
from __future__ import annotations

import logging

from . import binance_api, db

log = logging.getLogger(__name__)
FLUSH = 5000


def backfill_oi(symbol: str, start_year: int = 2020, start_month: int = 1) -> int:
    inserted_total = 0
    buf: list[tuple] = []
    for ts, oi, _ratio in binance_api.iter_metrics(symbol,
                                                   start_year=start_year,
                                                   start_month=start_month):
        buf.append((ts, oi))
        if len(buf) >= FLUSH:
            inserted_total += db.upsert_oi(symbol, buf, period="5m")
            buf.clear()
    if buf:
        inserted_total += db.upsert_oi(symbol, buf, period="5m")
    log.info("%s OI: inserted=%d", symbol, inserted_total)
    return inserted_total
