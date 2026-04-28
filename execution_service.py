"""
Execution Service — Bybit Position/Balance Push Loop

Runs alongside the GPU trainer FastAPI server. Polls Bybit for positions and
balance every few seconds, then pushes the data to the Replit dashboard so
it can display live trading state without calling Bybit directly.

Usage:
    from gpu_trainer.execution_service import start_execution_service
    start_execution_service(replit_url="https://your-app.replit.app", gpu_self_url="https://xxx.ngrok.io")

Or standalone:
    python -m gpu_trainer.execution_service --replit-url https://your-app.replit.app --gpu-url https://xxx.ngrok.io
"""

import os
import time
import hmac
import hashlib
import json
import logging
import threading
import requests
from typing import Optional, Dict, Any

log = logging.getLogger("ExecutionService")

BYBIT_BASE_URL = "https://api.bybit.com"
PUSH_INTERVAL = 5
RECV_WINDOW = "5000"


def _bybit_signature(api_key: str, api_secret: str, timestamp: str, params_str: str) -> str:
    payload = f"{timestamp}{api_key}{RECV_WINDOW}{params_str}"
    return hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _bybit_get(endpoint: str, api_key: str, api_secret: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    ts = str(int(time.time() * 1000))
    query_string = ""
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())

    sign = _bybit_signature(api_key, api_secret, ts, query_string)

    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": sign,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "Content-Type": "application/json",
    }

    url = f"{BYBIT_BASE_URL}{endpoint}"
    if query_string:
        url += f"?{query_string}"

    resp = requests.get(url, headers=headers, timeout=10)
    return resp.json()


def fetch_positions(api_key: str, api_secret: str) -> list:
    try:
        data = _bybit_get("/v5/position/list", api_key, api_secret, {
            "category": "linear",
            "settleCoin": "USDT",
        })
        if data.get("retCode") != 0:
            log.warning(f"Bybit positions error: {data.get('retMsg')}")
            return []
        positions = data.get("result", {}).get("list", [])
        return [p for p in positions if float(p.get("size", "0")) != 0]
    except Exception as e:
        log.error(f"Failed to fetch positions: {e}")
        return []


def fetch_balance(api_key: str, api_secret: str) -> Optional[Dict]:
    try:
        data = _bybit_get("/v5/account/wallet-balance", api_key, api_secret, {
            "accountType": "UNIFIED",
        })
        if data.get("retCode") != 0:
            log.warning(f"Bybit balance error: {data.get('retMsg')}")
            return None
        accounts = data.get("result", {}).get("list", [])
        if not accounts:
            return None
        acct = accounts[0]
        coins = acct.get("coin", [])
        usdt = next((c for c in coins if c.get("coin") == "USDT"), {})
        return {
            "equity": acct.get("totalEquity", "0"),
            "walletBalance": acct.get("totalWalletBalance", "0"),
            "availableToWithdraw": acct.get("totalAvailableBalance", "0"),
            "unrealisedPnl": usdt.get("unrealisedPnl", "0"),
            "totalMarginBalance": acct.get("totalMarginBalance", "0"),
        }
    except Exception as e:
        log.error(f"Failed to fetch balance: {e}")
        return None


def push_to_replit(replit_url: str, positions: list, balance: Optional[Dict], gpu_self_url: Optional[str] = None, exec_secret: Optional[str] = None):
    payload: Dict[str, Any] = {
        "positions": positions,
        "balance": balance,
        "timestamp": int(time.time() * 1000),
    }
    if gpu_self_url:
        payload["gpu_callback_url"] = gpu_self_url

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if exec_secret:
        headers["X-Exec-Secret"] = exec_secret

    try:
        resp = requests.post(
            f"{replit_url}/api/execution/push-state",
            json=payload,
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning(f"Push failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        log.error(f"Failed to push state to Replit: {e}")


def _push_loop(replit_url: str, api_key: str, api_secret: str, gpu_self_url: Optional[str] = None, interval: int = PUSH_INTERVAL, exec_secret: Optional[str] = None):
    log.info(f"[Execution Service] Started — pushing Bybit state to {replit_url} every {interval}s")
    consecutive_errors = 0

    while True:
        try:
            positions = fetch_positions(api_key, api_secret)
            balance = fetch_balance(api_key, api_secret)
            push_to_replit(replit_url, positions, balance, gpu_self_url, exec_secret)
            consecutive_errors = 0

            pos_count = len(positions)
            bal_str = balance.get("equity", "?") if balance else "N/A"
            log.debug(f"Pushed: {pos_count} positions, equity=${bal_str}")

        except Exception as e:
            consecutive_errors += 1
            log.error(f"Push loop error ({consecutive_errors}): {e}")
            if consecutive_errors > 10:
                log.warning("Too many consecutive errors, backing off to 30s")
                time.sleep(30)
                continue

        time.sleep(interval)


def start_execution_service(
    replit_url: str,
    gpu_self_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    exec_secret: Optional[str] = None,
    interval: int = PUSH_INTERVAL,
    daemon: bool = True,
):
    key = api_key or os.environ.get("BYBIT_API_KEY", "")
    secret = api_secret or os.environ.get("BYBIT_API_SECRET", "")
    auth = exec_secret or os.environ.get("EXEC_SECRET", "") or os.environ.get("SESSION_SECRET", "")

    if not key or not secret:
        log.error("[Execution Service] BYBIT_API_KEY and BYBIT_API_SECRET must be set")
        return None

    if not replit_url:
        log.error("[Execution Service] replit_url is required")
        return None

    t = threading.Thread(
        target=_push_loop,
        args=(replit_url, key, secret, gpu_self_url, interval, auth or None),
        daemon=daemon,
        name="ExecutionService",
    )
    t.start()
    log.info(f"[Execution Service] Thread started (daemon={daemon}, auth={'yes' if auth else 'no'})")
    return t


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Bybit Execution Service Push Loop")
    parser.add_argument("--replit-url", required=True, help="Replit dashboard URL")
    parser.add_argument("--gpu-url", default=None, help="GPU trainer ngrok URL (for auto-registration)")
    parser.add_argument("--exec-secret", default=None, help="Shared secret for authenticating with Replit (SESSION_SECRET)")
    parser.add_argument("--interval", type=int, default=5, help="Push interval in seconds")
    args = parser.parse_args()

    start_execution_service(
        replit_url=args.replit_url,
        gpu_self_url=args.gpu_url,
        exec_secret=args.exec_secret,
        interval=args.interval,
        daemon=False,
    )
