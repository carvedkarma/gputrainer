"""Funding-rate backfill via data.binance.vision archives."""
from __future__ import annotations

import logging

from . import binance_api, db

log = logging.getLogger(__name__)
FLUSH = 1000


def backfill_funding(symbol: str, start_ts_ms: int | None = None) -> tuple[int, int]:
    existing_min, existing_max, existing_count = db.get_existing_funding_range(symbol)
    if start_ts_ms is None:
        start_ts_ms = (existing_max + 1) if existing_max else None

    inserted_total = 0
    buf: list[tuple] = []
    for ts, fr in binance_api.iter_funding(symbol, min_ts_ms=start_ts_ms):
        buf.append((ts, fr))
        if len(buf) >= FLUSH:
            inserted_total += db.upsert_funding(symbol, buf)
            buf.clear()
    if buf:
        inserted_total += db.upsert_funding(symbol, buf)

    _, _, total_after = db.get_existing_funding_range(symbol)
    log.info("%s funding: inserted=%d total=%d", symbol, inserted_total, total_after)
    return inserted_total, total_after
