"""Postgres helpers for V7 ingest. Uses psycopg2 with batched execute_values."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Sequence

import psycopg2
from psycopg2.extras import execute_values


def _dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return dsn


@contextmanager
def conn():
    c = psycopg2.connect(_dsn())
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def upsert_candles_15m(symbol: str, rows: Sequence[tuple]) -> int:
    """rows: (timestamp, open, high, low, close, volume)."""
    if not rows:
        return 0
    payload = [(symbol, ts, "15m", o, h, l, c, v) for (ts, o, h, l, c, v) in rows]
    with conn() as cn, cn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO candles (symbol, timestamp, timeframe, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timestamp, timeframe) DO NOTHING
            """,
            payload,
            page_size=2000,
        )
        return cur.rowcount


def upsert_flow_features(rows: Sequence[tuple]) -> int:
    """rows: (symbol, ts, cvd_delta, agg_ratio, total_vol, taker_buy_vol, trade_count,
              trade_intensity, large_trade_count, large_trade_imbalance, liq_proxy)."""
    if not rows:
        return 0
    with conn() as cn, cn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO flow_features_15m
              (symbol, timestamp, cvd_delta, aggressor_ratio, total_volume,
               taker_buy_volume, trade_count, trade_intensity,
               large_trade_count, large_trade_imbalance, liquidation_proxy)
            VALUES %s
            ON CONFLICT (symbol, timestamp) DO NOTHING
            """,
            rows,
            page_size=2000,
        )
        return cur.rowcount


def upsert_funding(symbol: str, rows: Sequence[tuple]) -> int:
    """rows: (timestamp, funding_rate)."""
    if not rows:
        return 0
    payload = [(symbol, ts, fr) for (ts, fr) in rows]
    with conn() as cn, cn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO funding_history (symbol, timestamp, funding_rate)
            VALUES %s
            ON CONFLICT (symbol, timestamp) DO NOTHING
            """,
            payload,
            page_size=2000,
        )
        return cur.rowcount


def upsert_oi(symbol: str, rows: Sequence[tuple], period: str = "5m") -> int:
    """rows: (timestamp, sum_open_interest)."""
    if not rows:
        return 0
    payload = [(symbol, ts, period, oi) for (ts, oi) in rows]
    with conn() as cn, cn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO open_interest_history (symbol, timestamp, period, sum_open_interest)
            VALUES %s
            ON CONFLICT (symbol, timestamp, period) DO NOTHING
            """,
            payload,
            page_size=2000,
        )
        return cur.rowcount


def get_existing_15m_range(symbol: str) -> tuple[int | None, int | None, int]:
    """Return (min_ts, max_ts, count) for 15m candles of symbol."""
    with conn() as cn, cn.cursor() as cur:
        cur.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM candles WHERE symbol=%s AND timeframe='15m'",
            (symbol,),
        )
        row = cur.fetchone()
        return (row[0], row[1], row[2] or 0)


def get_existing_flow_range(symbol: str) -> tuple[int | None, int | None, int]:
    with conn() as cn, cn.cursor() as cur:
        cur.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM flow_features_15m WHERE symbol=%s",
            (symbol,),
        )
        row = cur.fetchone()
        return (row[0], row[1], row[2] or 0)


def get_existing_funding_range(symbol: str) -> tuple[int | None, int | None, int]:
    with conn() as cn, cn.cursor() as cur:
        cur.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM funding_history WHERE symbol=%s",
            (symbol,),
        )
        row = cur.fetchone()
        return (row[0], row[1], row[2] or 0)
