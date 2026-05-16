import asyncio
import unittest
from unittest.mock import patch

import api.server as api_server
from live_runner import LiveRunner
from portfolio import PortfolioManager, Position


class _FakeResp:
    def __init__(self, status_code: int, payload):
        self.status_code = int(status_code)
        self._payload = payload

    def json(self):
        return self._payload


class LiveApiSyncIntegrationTests(unittest.TestCase):
    def setUp(self):
        api_server._dashboard_sessions.clear()
        api_server._MARKET_PRICE_CACHE.clear()

    def _create_open_trade(self, session_id: str, symbol: str = "BTCUSDT") -> int:
        payload = {
            "symbol": symbol,
            "side": "LONG",
            "entry_price": 100.0,
            "stop_loss": 99.0,
            "take_profit": 105.0,
            "p_enter": 0.75,
            "lane": "MYTHOS",
            "status": "open",
            "model_name": "mythos_runtime_live",
        }
        result = asyncio.run(api_server.create_live_trade(payload, session_id=session_id))
        return int(result["id"])

    def test_open_positions_summary_runs_auto_close_before_return(self):
        sid = "int-auto-close"
        trade_id = self._create_open_trade(sid)

        async def _fake_market_prices(symbols, force_refresh=False):
            return {"BTCUSDT": {"price": 98.9, "ts": 1, "source": "test"}}

        with patch("api.server._resolve_market_prices", side_effect=_fake_market_prices):
            summary = asyncio.run(api_server.open_positions_summary(session_id=sid))

        self.assertEqual(summary["count"], 0)
        state = api_server._get_dashboard_session(sid)
        tr = state.trades[int(trade_id)]
        self.assertEqual(str(tr.get("status", "")).lower(), "closed")
        self.assertEqual(str(tr.get("outcome", "")), "SL")

    def test_manual_close_sync_removes_runner_position_and_tm_state(self):
        sid = "int-manual-sync"
        trade_id = self._create_open_trade(sid)

        portfolio = PortfolioManager(cooldown_bars=0)
        with patch.object(LiveRunner, "_detect_gpu_self_url", return_value=None):
            runner = LiveRunner(
                replit_url="http://fake.local",
                symbols=["BTCUSDT"],
                device="cpu",
                paper=True,
                execution_mode="paper",
                record_trades=True,
                portfolio_manager=portfolio,
                paper_session_id=sid,
                dashboard_engine="mythos",
                live_model="mythos",
            )

        pos = Position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            entry_time=0.0,
            atr=1.0,
            tp_price=105.0,
            sl_price=99.0,
            p_enter=0.75,
            size_mult=1.0,
            risk_pct=1.0,
            bar_index=1,
            lane="MYTHOS",
            horizon=24,
            htf_score=0,
            threshold_used=0.6,
        )
        pos.dashboard_trade_id = int(trade_id)
        runner.portfolio.open_positions["BTCUSDT"] = pos
        runner.trade_manager.position_state["BTCUSDT"] = {"risk_per_unit": 1.0}

        asyncio.run(
            api_server.manual_close_paper_trade(
                int(trade_id),
                {"note": "integration_manual_close"},
                session_id=sid,
            )
        )
        summary = asyncio.run(api_server.open_positions_summary(session_id=sid))
        self.assertEqual(summary["count"], 0)

        def _fake_get(url, params=None, timeout=0):
            self.assertIn("/api/paper/open-positions-summary", url)
            return _FakeResp(200, summary)

        with patch("live_runner.requests.get", side_effect=_fake_get):
            runner._sync_portfolio_from_web(source="integration_test")

        self.assertNotIn("BTCUSDT", runner.portfolio.open_positions)
        self.assertIsNone(runner.trade_manager.get_state("BTCUSDT"))

    def test_partial_close_creates_child_and_reduces_parent_risk(self):
        sid = "int-partial"
        trade_id = self._create_open_trade(sid)
        state = api_server._get_dashboard_session(sid)
        before_risk = float(state.trades[int(trade_id)].get("risk_usd_used", 0.0))

        result = asyncio.run(
            api_server.paper_trade_manager_action(
                int(trade_id),
                {"action": "partial_close", "close_pct": 50.0, "note": "integration_partial"},
                session_id=sid,
            )
        )
        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(str(result.get("action", "")), "partial_close")
        child_id = int(result.get("partial_trade_id", 0))
        self.assertGreater(child_id, 0)

        parent = state.trades[int(trade_id)]
        child = state.trades[int(child_id)]
        self.assertEqual(str(child.get("status", "")).lower(), "closed")
        self.assertEqual(str(child.get("outcome", "")), "PARTIAL_CLOSE")
        after_risk = float(parent.get("risk_usd_used", 0.0))
        self.assertLess(after_risk, before_risk)

    def test_cycle_log_payload_includes_stage_and_execution_status(self):
        sid = "int-cycle-stage"
        portfolio = PortfolioManager(cooldown_bars=0)
        with patch.object(LiveRunner, "_detect_gpu_self_url", return_value=None):
            runner = LiveRunner(
                replit_url="http://fake.local",
                symbols=["BTCUSDT"],
                device="cpu",
                paper=True,
                execution_mode="paper",
                record_trades=True,
                portfolio_manager=portfolio,
                paper_session_id=sid,
                dashboard_engine="mythos",
                live_model="mythos",
            )

        captured = []

        def _fake_retry(method, url, **kwargs):
            captured.append(dict(kwargs.get("json") or {}))
            return _FakeResp(200, {"ok": True})

        with patch("live_runner._retry_request", side_effect=_fake_retry):
            runner._push_cycle_log(
                symbol="BTCUSDT",
                price=100.0,
                p_enter=0.8,
                htf={"h1_trend": 0, "h4_trend": 0},
                direction="LONG",
                decision="ENTER_EXECUTED",
                reasons=["integration_exec"],
                decision_stage="execution",
                execution_status="executed",
            )

        self.assertEqual(len(captured), 1)
        self.assertEqual(str(captured[0].get("decision_stage")), "execution")
        self.assertEqual(str(captured[0].get("execution_status")), "executed")

    def test_adaptive_spread_limit_tightens_when_slippage_is_high(self):
        portfolio = PortfolioManager(cooldown_bars=0)
        with patch.object(LiveRunner, "_detect_gpu_self_url", return_value=None):
            runner = LiveRunner(
                replit_url="http://fake.local",
                symbols=["BTCUSDT"],
                device="cpu",
                paper=True,
                execution_mode="paper",
                record_trades=True,
                portfolio_manager=portfolio,
                exec_max_spread_bps=14.0,
                exec_spread_adaptive_enable=True,
                exec_spread_window=20,
                exec_spread_target_slippage_bps=2.0,
                exec_spread_min_bps=4.0,
            )
        runner._recent_entry_slippage_bps = [4.0] * 20
        limit_bps = runner._adaptive_spread_limit_bps()
        self.assertLess(limit_bps, 14.0)
        self.assertGreaterEqual(limit_bps, 4.0)

    def test_execute_candidate_rejects_when_spread_gate_fails(self):
        sid = "int-spread-gate"
        portfolio = PortfolioManager(cooldown_bars=0)
        with patch.object(LiveRunner, "_detect_gpu_self_url", return_value=None):
            runner = LiveRunner(
                replit_url="http://fake.local",
                symbols=["BTCUSDT"],
                device="cpu",
                paper=True,
                execution_mode="paper",
                record_trades=True,
                portfolio_manager=portfolio,
                paper_session_id=sid,
                exec_max_spread_bps=10.0,
                exec_spread_adaptive_enable=False,
            )
        candidate = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "current_price": 100.0,
            "atr": 1.0,
            "p_enter": 0.9,
            "htf": {"h1_trend": 0, "h4_trend": 0},
            "v5_info": {"v5_score": 1.0, "threshold_used": 0.5, "lane": "MYTHOS"},
            "df_candles": None,
        }
        captured = []
        with patch.object(
            runner,
            "_fetch_symbol_microstructure",
            return_value={"bid": 99.8, "ask": 100.2, "spread_bps": 40.0},
        ), patch.object(runner, "_push_cycle_log", side_effect=lambda **kwargs: captured.append(kwargs)):
            runner._execute_candidate(candidate)
        self.assertEqual(len(runner.portfolio.open_positions), 0)
        self.assertTrue(any(str(x.get("decision")) == "EXECUTION_SKIPPED" for x in captured))


if __name__ == "__main__":
    unittest.main()
