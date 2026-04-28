"""Per-symbol reconciliation: row count, date continuity, gaps."""
from __future__ import annotations

import time
from . import db

BAR_MS = 15 * 60 * 1000


def reconcile_15m(symbol: str) -> dict:
    mn, mx, cnt = db.get_existing_15m_range(symbol)
    if cnt == 0 or mn is None or mx is None:
        return {"symbol": symbol, "ok": False, "reason": "no_data", "count": 0}
    expected = (mx - mn) // BAR_MS + 1
    gaps = expected - cnt
    days = (mx - mn) / 86400_000.0
    return {
        "symbol": symbol,
        "ok": gaps < expected * 0.001,  # tolerate <0.1% gaps
        "count": cnt,
        "expected": int(expected),
        "gaps": int(gaps),
        "first": time.strftime("%Y-%m-%d", time.gmtime(mn / 1000)),
        "last": time.strftime("%Y-%m-%d", time.gmtime(mx / 1000)),
        "days": round(days, 1),
    }


def reconcile_flow(symbol: str) -> dict:
    mn, mx, cnt = db.get_existing_flow_range(symbol)
    if cnt == 0 or mn is None or mx is None:
        return {"symbol": symbol, "ok": False, "reason": "no_data", "count": 0}
    expected = (mx - mn) // BAR_MS + 1
    return {
        "symbol": symbol,
        "ok": (expected - cnt) < expected * 0.01,
        "count": cnt,
        "expected": int(expected),
        "gaps": int(expected - cnt),
        "first": time.strftime("%Y-%m-%d", time.gmtime(mn / 1000)),
        "last": time.strftime("%Y-%m-%d", time.gmtime(mx / 1000)),
    }


def reconcile_funding(symbol: str) -> dict:
    mn, mx, cnt = db.get_existing_funding_range(symbol)
    if cnt == 0 or mn is None or mx is None:
        return {"symbol": symbol, "ok": False, "reason": "no_data", "count": 0}
    return {
        "symbol": symbol,
        "ok": cnt > 30,
        "count": cnt,
        "first": time.strftime("%Y-%m-%d", time.gmtime(mn / 1000)),
        "last": time.strftime("%Y-%m-%d", time.gmtime(mx / 1000)),
    }
