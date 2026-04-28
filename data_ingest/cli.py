"""V7 ingest orchestrator.

Usage:
  python -m gpu_trainer.data_ingest.cli all
  python -m gpu_trainer.data_ingest.cli klines     # 13 short-history symbols
  python -m gpu_trainer.data_ingest.cli flow       # 7 full-history symbols
  python -m gpu_trainer.data_ingest.cli funding    # 20 symbols
  python -m gpu_trainer.data_ingest.cli oi         # 20 symbols (last 30d)
  python -m gpu_trainer.data_ingest.cli reconcile
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import (
    flow_features,
    funding_backfill,
    klines_backfill,
    oi_backfill,
    reconcile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Symbol classes for V7 truth-discovery audit
FULL_HISTORY = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
SHORT_HISTORY = ["DOGEUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "PEPEUSDT", "SUIUSDT",
                 "AAVEUSDT", "ARBUSDT", "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT"]
ALL_SYMBOLS = FULL_HISTORY + SHORT_HISTORY


def cmd_klines(args):
    symbols = args.symbols or SHORT_HISTORY
    summary = []
    for s in symbols:
        try:
            inserted, total = klines_backfill.backfill_symbol(s)
            summary.append({"symbol": s, "inserted": inserted, "total": total})
        except Exception as e:
            summary.append({"symbol": s, "error": str(e)})
    print(json.dumps(summary, indent=2))


def cmd_flow(args):
    symbols = args.symbols or FULL_HISTORY
    summary = []
    for s in symbols:
        try:
            inserted, total = flow_features.backfill_flow(s)
            summary.append({"symbol": s, "inserted": inserted, "total": total})
        except Exception as e:
            summary.append({"symbol": s, "error": str(e)})
            logging.exception("flow %s failed", s)
    print(json.dumps(summary, indent=2))


def cmd_funding(args):
    symbols = args.symbols or ALL_SYMBOLS
    summary = []
    for s in symbols:
        try:
            inserted, total = funding_backfill.backfill_funding(s)
            summary.append({"symbol": s, "inserted": inserted, "total": total})
        except Exception as e:
            summary.append({"symbol": s, "error": str(e)})
    print(json.dumps(summary, indent=2))


def cmd_oi(args):
    symbols = args.symbols or ALL_SYMBOLS
    summary = []
    for s in symbols:
        try:
            inserted = oi_backfill.backfill_oi(s)
            summary.append({"symbol": s, "inserted": inserted})
        except Exception as e:
            summary.append({"symbol": s, "error": str(e)})
    print(json.dumps(summary, indent=2))


def cmd_reconcile(args):
    out = {"15m": [], "flow": [], "funding": []}
    for s in ALL_SYMBOLS:
        out["15m"].append(reconcile.reconcile_15m(s))
        out["flow"].append(reconcile.reconcile_flow(s))
        out["funding"].append(reconcile.reconcile_funding(s))
    print(json.dumps(out, indent=2))


def cmd_all(args):
    cmd_klines(args)
    cmd_flow(args)
    cmd_funding(args)
    cmd_oi(args)
    cmd_reconcile(args)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("klines", cmd_klines), ("flow", cmd_flow),
                     ("funding", cmd_funding), ("oi", cmd_oi),
                     ("reconcile", cmd_reconcile), ("all", cmd_all)]:
        p = sub.add_parser(name)
        p.add_argument("--symbols", nargs="*", default=None)
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
