"""Binance archive client (data.binance.vision).

The live Binance Futures REST API (`fapi.binance.com`) is geo-blocked from the
Replit datacentre (HTTP 451). The historical archive CDN at
`data.binance.vision` is reachable, well-structured, and contains everything
we need for the V7 truth-discovery audit:

* USDM-futures klines at any interval (monthly + daily ZIPs)
* Funding rate history (monthly ZIPs)
* "Metrics" series: open interest, top-trader L/S ratio, taker buy/sell ratio
  at 5-minute cadence (daily ZIPs)

This module exposes streaming iterators that yield in time-ascending order
across whichever (monthly, daily) chunks exist.
"""
from __future__ import annotations

import csv
import io
import logging
import threading
import time
import zipfile
from datetime import datetime, timezone
from typing import Iterator

import requests

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um"
USER_AGENT = "v7-truth-discovery/1.0"

# Polite throttling: 12 req/sec is well within CDN tolerance.
_LOCK = threading.Lock()
_LAST = [0.0]
_MIN_INT = 1.0 / 12.0


def _throttle():
    with _LOCK:
        now = time.monotonic()
        wait = _LAST[0] + _MIN_INT - now
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.monotonic()


def _fetch(url: str, retries: int = 4) -> bytes | None:
    """Return body bytes, or None if 404. Retries on transient errors."""
    last_err: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_err = e
            wait = 2 ** attempt
            log.debug("retry %d/%d after %ds: %s", attempt + 1, retries, wait, e)
            time.sleep(wait)
    log.warning("CDN fetch failed: %s (%s)", url, last_err)
    return None


def _csv_rows_from_zip(blob: bytes) -> Iterator[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            first = True
            for row in reader:
                if first:
                    first = False
                    # Skip header if it looks like one
                    if row and not row[0].lstrip("-").replace(".", "").isdigit():
                        continue
                yield row


def _iter_months(start: tuple[int, int], end: tuple[int, int]) -> Iterator[tuple[int, int]]:
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _iter_days(year: int, month: int) -> Iterator[tuple[int, int, int]]:
    """Yield (y, m, d) for every day in the given month, capped by today."""
    if month == 12:
        next_y, next_m = year + 1, 1
    else:
        next_y, next_m = year, month + 1
    today = datetime.now(timezone.utc).date()
    d = 1
    while True:
        try:
            cur = datetime(year, month, d, tzinfo=timezone.utc).date()
        except ValueError:
            return
        if cur >= datetime(next_y, next_m, 1, tzinfo=timezone.utc).date():
            return
        if cur > today:
            return
        yield year, month, d
        d += 1


# ---------- klines ----------

def _kline_url(symbol: str, interval: str, year: int, month: int,
               day: int | None = None) -> str:
    if day is None:
        return (f"{BASE}/monthly/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")
    return (f"{BASE}/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{year:04d}-{month:02d}-{day:02d}.zip")


def iter_klines(symbol: str, interval: str,
                start_year: int = 2019, start_month: int = 9,
                min_ts_ms: int | None = None) -> Iterator[tuple]:
    """Stream kline rows in time order from the CDN archives.

    Yields tuples: (open_ts_ms, open, high, low, close, volume, count, taker_buy_vol).
    Skips months that have no monthly archive AND no daily archives.
    """
    if min_ts_ms is not None:
        dt = datetime.fromtimestamp(min_ts_ms / 1000, tz=timezone.utc)
        if (dt.year, dt.month) > (start_year, start_month):
            start_year, start_month = dt.year, dt.month
    now = datetime.now(timezone.utc)
    end_year, end_month = now.year, now.month
    months_seen = 0
    rows_seen = 0
    for y, m in _iter_months((start_year, start_month), (end_year, end_month)):
        # Try monthly first.
        blob = _fetch(_kline_url(symbol, interval, y, m))
        if blob is not None:
            for row in _csv_rows_from_zip(blob):
                ts = int(row[0])
                if min_ts_ms is not None and ts < min_ts_ms:
                    continue
                yield (ts, float(row[1]), float(row[2]), float(row[3]),
                       float(row[4]), float(row[5]),
                       int(float(row[8])) if len(row) > 8 else 0,
                       float(row[9]) if len(row) > 9 else 0.0)
                rows_seen += 1
            months_seen += 1
            continue
        # Fall back to daily.
        any_day = False
        for yy, mm, dd in _iter_days(y, m):
            blob = _fetch(_kline_url(symbol, interval, yy, mm, dd))
            if blob is None:
                continue
            any_day = True
            for row in _csv_rows_from_zip(blob):
                ts = int(row[0])
                if min_ts_ms is not None and ts < min_ts_ms:
                    continue
                yield (ts, float(row[1]), float(row[2]), float(row[3]),
                       float(row[4]), float(row[5]),
                       int(float(row[8])) if len(row) > 8 else 0,
                       float(row[9]) if len(row) > 9 else 0.0)
                rows_seen += 1
        if any_day:
            months_seen += 1
    log.info("%s %s: %d months yielded, %d rows", symbol, interval, months_seen, rows_seen)


# ---------- funding ----------

def _funding_url(symbol: str, year: int, month: int) -> str:
    return (f"{BASE}/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip")


def iter_funding(symbol: str, start_year: int = 2019, start_month: int = 9,
                 min_ts_ms: int | None = None) -> Iterator[tuple]:
    """Yields (calc_time_ms, funding_rate)."""
    if min_ts_ms is not None:
        dt = datetime.fromtimestamp(min_ts_ms / 1000, tz=timezone.utc)
        if (dt.year, dt.month) > (start_year, start_month):
            start_year, start_month = dt.year, dt.month
    now = datetime.now(timezone.utc)
    end_year, end_month = now.year, now.month
    for y, m in _iter_months((start_year, start_month), (end_year, end_month)):
        blob = _fetch(_funding_url(symbol, y, m))
        if blob is None:
            continue
        for row in _csv_rows_from_zip(blob):
            ts = int(row[0])
            if min_ts_ms is not None and ts < min_ts_ms:
                continue
            yield ts, float(row[2])


# ---------- metrics (open interest etc) ----------

def _metrics_url(symbol: str, year: int, month: int, day: int) -> str:
    return (f"{BASE}/daily/metrics/{symbol}/"
            f"{symbol}-metrics-{year:04d}-{month:02d}-{day:02d}.zip")


def _parse_dt(s: str) -> int:
    # "2024-06-01 00:05:00" -> ms UTC
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def iter_metrics(symbol: str, start_year: int = 2020, start_month: int = 1,
                 min_ts_ms: int | None = None) -> Iterator[tuple]:
    """Yields (ts_ms, sum_open_interest, sum_taker_long_short_vol_ratio)."""
    if min_ts_ms is not None:
        dt = datetime.fromtimestamp(min_ts_ms / 1000, tz=timezone.utc)
        if (dt.year, dt.month) > (start_year, start_month):
            start_year, start_month = dt.year, dt.month
    now = datetime.now(timezone.utc)
    end_year, end_month = now.year, now.month
    for y, m in _iter_months((start_year, start_month), (end_year, end_month)):
        for yy, mm, dd in _iter_days(y, m):
            blob = _fetch(_metrics_url(symbol, yy, mm, dd))
            if blob is None:
                continue
            for row in _csv_rows_from_zip(blob):
                if len(row) < 8:
                    continue
                try:
                    ts = _parse_dt(row[0])
                except ValueError:
                    continue
                if min_ts_ms is not None and ts < min_ts_ms:
                    continue
                try:
                    yield ts, float(row[2]), float(row[7])
                except (ValueError, IndexError):
                    continue


# ---------- thin compatibility shim (unused by V7 ingest, kept for any legacy callers) ----------

def klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    raise NotImplementedError("Live REST klines is geo-blocked; use iter_klines() from the CDN")


def funding_rate(symbol: str, start_ms=None, end_ms=None, limit=1000):
    raise NotImplementedError("Live REST funding is geo-blocked; use iter_funding() from the CDN")
