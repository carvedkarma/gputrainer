"""Smart Trade Manager v5 — adaptive exits with policy presets.

Evaluates open positions each cycle and returns actions:
HOLD | MOVE_SL | TRAIL_SL | CLOSE_PARTIAL | CLOSE_FULL
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger("TradeMgr")

ADVERSE_FLIP_R = -0.60
ADVERSE_HTF_DROP = 2
ADVERSE_P_ENTER_DECAY = 0.80
MAX_SL_UPDATES = 6
MIN_BARS_BETWEEN_SL = 1

_POLICIES: Dict[str, Dict[str, object]] = {
    "defensive": {
        "breakeven_trigger_r": 0.20,
        "trail_trigger_r": 0.45,
        "trail_distance_r": 0.28,
        "trail_min_r": 0.14,
        "trail_drawdown_tighten": 0.18,
        "trail_max_tighten": 0.14,
        "stall_trigger_r": 0.55,
        "stall_bars": 1,
        "scale_out_plan": ((0.55, 35.0), (1.05, 35.0)),
        "scale_out_min_bars": 2,
        "time_min_mult": 0.45,
        "time_max_mult": 1.20,
        "time_winner_bonus": 0.25,
        "time_loser_penalty": 0.35,
    },
    "balanced": {
        "breakeven_trigger_r": 0.35,
        "trail_trigger_r": 0.60,
        "trail_distance_r": 0.40,
        "trail_min_r": 0.20,
        "trail_drawdown_tighten": 0.16,
        "trail_max_tighten": 0.18,
        "stall_trigger_r": 0.70,
        "stall_bars": 2,
        "scale_out_plan": ((0.90, 25.0), (1.70, 25.0)),
        "scale_out_min_bars": 2,
        "time_min_mult": 0.50,
        "time_max_mult": 1.55,
        "time_winner_bonus": 0.40,
        "time_loser_penalty": 0.30,
    },
    "aggressive": {
        "breakeven_trigger_r": 0.45,
        "trail_trigger_r": 0.75,
        "trail_distance_r": 0.52,
        "trail_min_r": 0.28,
        "trail_drawdown_tighten": 0.12,
        "trail_max_tighten": 0.16,
        "stall_trigger_r": 1.20,
        "stall_bars": 3,
        "scale_out_plan": ((1.10, 20.0), (2.00, 25.0), (3.00, 25.0)),
        "scale_out_min_bars": 2,
        "time_min_mult": 0.60,
        "time_max_mult": 2.20,
        "time_winner_bonus": 0.60,
        "time_loser_penalty": 0.22,
    },
    "max": {
        "breakeven_trigger_r": 0.30,
        "trail_trigger_r": 0.55,
        "trail_distance_r": 0.34,
        "trail_min_r": 0.16,
        "trail_drawdown_tighten": 0.20,
        "trail_max_tighten": 0.24,
        "stall_trigger_r": 0.95,
        "stall_bars": 2,
        "scale_out_plan": ((0.70, 22.0), (1.35, 23.0), (2.25, 25.0)),
        "scale_out_min_bars": 1,
        "time_min_mult": 0.45,
        "time_max_mult": 2.60,
        "time_winner_bonus": 0.75,
        "time_loser_penalty": 0.35,
    },
}


@dataclass
class TMAction:
    action: str
    reason: str
    new_sl: Optional[float] = None
    close_price: Optional[float] = None
    close_pct: Optional[float] = None


class TradeManager:
    """Manages dynamic exits for open positions."""

    def __init__(
        self,
        policy_mode: str = "balanced",
        enable_scale_out: bool = True,
        enable_regime_time_stop: bool = True,
        enable_volatility_trailing: bool = True,
    ):
        mode = str(policy_mode or "balanced").strip().lower()
        self.policy_mode = mode if mode in _POLICIES else "balanced"
        self.enable_scale_out = bool(enable_scale_out)
        self.enable_regime_time_stop = bool(enable_regime_time_stop)
        self.enable_volatility_trailing = bool(enable_volatility_trailing)
        self.position_state: Dict[str, dict] = {}

    def _policy(self) -> Dict[str, object]:
        return _POLICIES[self.policy_mode]

    def _ensure_state(self, symbol: str, pos) -> dict:
        if symbol not in self.position_state:
            risk_per_unit = abs(pos.entry_price - pos.original_sl) if hasattr(pos, "original_sl") else abs(pos.entry_price - pos.sl_price)
            self.position_state[symbol] = {
                "peak_price": pos.entry_price,
                "trough_price": pos.entry_price,
                "max_favorable_r": 0.0,
                "max_adverse_r": 0.0,
                "stall_counter": 0,
                "last_mfe": 0.0,
                "breakeven_moved": False,
                "risk_per_unit": risk_per_unit,
                "original_sl": pos.sl_price,
                "entry_htf_score": getattr(pos, "htf_score", None),
                "sl_update_count": 0,
                "last_sl_update_bar": -999,
                "effective_horizon": int(max(getattr(pos, "horizon", 24), 1)),
                "partial_steps_done": set(),
                "partial_closed_pct": 0.0,
                "last_partial_bar": -999,
            }
        return self.position_state[symbol]

    def _effective_horizon(
        self,
        pos,
        state: dict,
        unrealized_r: float,
        current_htf_score: Optional[int],
        current_p_enter: Optional[float],
    ) -> int:
        base_horizon = int(max(getattr(pos, "horizon", 24), 1))
        if not self.enable_regime_time_stop:
            state["effective_horizon"] = base_horizon
            return base_horizon

        policy = self._policy()
        horizon_mult = 1.0

        if unrealized_r >= 0.35:
            horizon_mult += float(policy["time_winner_bonus"]) * min(unrealized_r / 2.0, 1.0)
        elif unrealized_r <= -0.20:
            horizon_mult -= float(policy["time_loser_penalty"]) * min(abs(unrealized_r) / 1.5, 1.0)

        if current_htf_score is not None and state.get("entry_htf_score") is not None:
            delta = float(current_htf_score - state.get("entry_htf_score", 0))
            horizon_mult += max(min(delta * 0.06, 0.25), -0.30)

        threshold = float(max(getattr(pos, "threshold_used", 0.0), 0.0))
        if current_p_enter is not None and threshold > 0.0:
            conf_ratio = float(current_p_enter / threshold)
            horizon_mult += max(min((conf_ratio - 1.0) * 0.30, 0.20), -0.30)

        horizon_mult = max(float(policy["time_min_mult"]), min(float(policy["time_max_mult"]), horizon_mult))
        resolved = int(max(1, round(base_horizon * horizon_mult)))
        state["effective_horizon"] = resolved
        return resolved

    def _resolve_trail_distance(self, state: dict, unrealized_r: float) -> float:
        policy = self._policy()
        trail_r = float(policy["trail_distance_r"])
        if not self.enable_volatility_trailing:
            return trail_r

        drawdown_from_peak = max(float(state["max_favorable_r"]) - float(unrealized_r), 0.0)
        tighten = min(
            drawdown_from_peak * float(policy["trail_drawdown_tighten"]),
            float(policy["trail_max_tighten"]),
        )
        trail_r -= tighten
        trail_r = max(float(policy["trail_min_r"]), trail_r)
        return trail_r

    def _maybe_scale_out(
        self,
        symbol: str,
        pos,
        state: dict,
        current_bar: int,
        unrealized_r: float,
    ) -> Optional[TMAction]:
        if not self.enable_scale_out:
            return None
        if float(state.get("partial_closed_pct", 0.0)) >= 95.0:
            return None

        policy = self._policy()
        plan = policy.get("scale_out_plan") or ()
        min_gap = int(max(policy.get("scale_out_min_bars", 1), 1))
        if (current_bar - int(state.get("last_partial_bar", -999))) < min_gap:
            return None

        for step_idx, step in enumerate(plan):
            trigger_r, close_pct = float(step[0]), float(step[1])
            if step_idx in state["partial_steps_done"]:
                continue
            if state["max_favorable_r"] < trigger_r:
                continue
            if unrealized_r < max(0.10, trigger_r * 0.30):
                continue
            state["partial_steps_done"].add(step_idx)
            state["last_partial_bar"] = current_bar
            state["partial_closed_pct"] = min(
                100.0,
                float(state.get("partial_closed_pct", 0.0)) + close_pct,
            )
            self._log_action(symbol, pos, unrealized_r, state, "CLOSE_PARTIAL", f"SCALE_OUT_{step_idx + 1}")
            return TMAction(action="CLOSE_PARTIAL", reason=f"SCALE_OUT_{step_idx + 1}", close_pct=close_pct)
        return None

    def _maybe_move_to_breakeven(self, symbol: str, pos, state: dict, current_bar: int) -> Optional[TMAction]:
        policy = self._policy()
        if state["max_favorable_r"] < float(policy["breakeven_trigger_r"]) or state["breakeven_moved"]:
            return None

        sl_allowed = (
            state["sl_update_count"] < MAX_SL_UPDATES
            and current_bar - state["last_sl_update_bar"] >= MIN_BARS_BETWEEN_SL
        )
        if not sl_allowed:
            return None

        if pos.is_long and pos.sl_price < pos.entry_price:
            state["breakeven_moved"] = True
            state["sl_update_count"] += 1
            state["last_sl_update_bar"] = current_bar
            self._log_action(symbol, pos, state["max_favorable_r"], state, "MOVE_SL", "BREAKEVEN")
            return TMAction(action="MOVE_SL", reason="BREAKEVEN", new_sl=pos.entry_price)
        if (not pos.is_long) and pos.sl_price > pos.entry_price:
            state["breakeven_moved"] = True
            state["sl_update_count"] += 1
            state["last_sl_update_bar"] = current_bar
            self._log_action(symbol, pos, state["max_favorable_r"], state, "MOVE_SL", "BREAKEVEN")
            return TMAction(action="MOVE_SL", reason="BREAKEVEN", new_sl=pos.entry_price)
        return None

    def update_position(
        self,
        symbol: str,
        pos,
        current_price: float,
        current_bar: int,
        current_htf_score: Optional[int] = None,
        current_p_enter: Optional[float] = None,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
    ) -> TMAction:
        """Evaluate one open position and return the recommended action."""
        state = self._ensure_state(symbol, pos)
        risk_per_unit = float(state["risk_per_unit"])
        if risk_per_unit <= 0:
            return TMAction(action="HOLD", reason="zero_risk_per_unit")

        policy = self._policy()
        use_high = candle_high if candle_high is not None else current_price
        use_low = candle_low if candle_low is not None else current_price

        if pos.is_long:
            if use_high > state["peak_price"]:
                state["peak_price"] = use_high
            if use_low < state["trough_price"]:
                state["trough_price"] = use_low
            favorable_r = (state["peak_price"] - pos.entry_price) / risk_per_unit
            adverse_r = (state["trough_price"] - pos.entry_price) / risk_per_unit
            unrealized_r = (current_price - pos.entry_price) / risk_per_unit
        else:
            if use_low < state["peak_price"]:
                state["peak_price"] = use_low
            if use_high > state["trough_price"]:
                state["trough_price"] = use_high
            favorable_r = (pos.entry_price - state["peak_price"]) / risk_per_unit
            adverse_r = (pos.entry_price - state["trough_price"]) / risk_per_unit
            unrealized_r = (pos.entry_price - current_price) / risk_per_unit

        state["max_favorable_r"] = max(float(state["max_favorable_r"]), float(favorable_r))
        state["max_adverse_r"] = min(float(state["max_adverse_r"]), float(adverse_r))
        bars_held = current_bar - pos.bar_index if pos.bar_index > 0 else 0
        effective_horizon = self._effective_horizon(
            pos=pos,
            state=state,
            unrealized_r=unrealized_r,
            current_htf_score=current_htf_score,
            current_p_enter=current_p_enter,
        )

        if bars_held >= effective_horizon:
            self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", "TIME_EXIT")
            return TMAction(action="CLOSE_FULL", reason="TIME_EXIT", close_price=current_price)

        if unrealized_r <= ADVERSE_FLIP_R:
            htf_dropped = False
            p_decayed = False
            if current_htf_score is not None and state.get("entry_htf_score") is not None:
                htf_drop = state["entry_htf_score"] - current_htf_score
                if htf_drop >= ADVERSE_HTF_DROP:
                    htf_dropped = True
            if current_p_enter is not None and getattr(pos, "threshold_used", 0.0) > 0:
                if current_p_enter < pos.threshold_used * ADVERSE_P_ENTER_DECAY:
                    p_decayed = True
            if htf_dropped or p_decayed:
                self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", "ADVERSE_FLIP")
                return TMAction(action="CLOSE_FULL", reason="ADVERSE_FLIP", close_price=current_price)

        scale_out_action = self._maybe_scale_out(
            symbol=symbol,
            pos=pos,
            state=state,
            current_bar=current_bar,
            unrealized_r=unrealized_r,
        )
        if scale_out_action is not None:
            return scale_out_action

        stall_trigger_r = float(policy["stall_trigger_r"])
        if unrealized_r >= stall_trigger_r:
            if state["max_favorable_r"] <= state["last_mfe"] + 0.01:
                state["stall_counter"] += 1
            else:
                state["stall_counter"] = 0
            state["last_mfe"] = state["max_favorable_r"]

            if state["stall_counter"] >= int(policy["stall_bars"]):
                self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", "STALL_TAKEPROFIT")
                return TMAction(action="CLOSE_FULL", reason="STALL_TAKEPROFIT", close_price=current_price)
        else:
            state["stall_counter"] = 0
            state["last_mfe"] = state["max_favorable_r"]

        sl_allowed = (
            state["sl_update_count"] < MAX_SL_UPDATES
            and current_bar - state["last_sl_update_bar"] >= MIN_BARS_BETWEEN_SL
        )

        if state["max_favorable_r"] >= float(policy["trail_trigger_r"]) and sl_allowed:
            trail_r = self._resolve_trail_distance(state=state, unrealized_r=unrealized_r)
            if pos.is_long:
                new_sl = state["peak_price"] - trail_r * risk_per_unit
                if new_sl > pos.sl_price:
                    state["sl_update_count"] += 1
                    state["last_sl_update_bar"] = current_bar
                    return TMAction(action="TRAIL_SL", reason="TRAIL_LOCK", new_sl=new_sl)
            else:
                new_sl = state["peak_price"] + trail_r * risk_per_unit
                if new_sl < pos.sl_price:
                    state["sl_update_count"] += 1
                    state["last_sl_update_bar"] = current_bar
                    return TMAction(action="TRAIL_SL", reason="TRAIL_LOCK", new_sl=new_sl)

        be_action = self._maybe_move_to_breakeven(
            symbol=symbol,
            pos=pos,
            state=state,
            current_bar=current_bar,
        )
        if be_action is not None:
            return be_action

        return TMAction(action="HOLD", reason="no_exit_trigger")

    def get_state(self, symbol: str) -> Optional[dict]:
        return self.position_state.get(symbol)

    def clear_position(self, symbol: str):
        self.position_state.pop(symbol, None)

    def _log_action(self, symbol: str, pos, unrealized_r: float, state: dict, action: str, reason: str):
        log.info(
            "[TM] sym=%s lane=%s policy=%s uR=%+.2f mfe=%+.2f mae=%+.2f "
            "h=%s action=%s reason=%s",
            symbol,
            getattr(pos, "lane", "NA"),
            self.policy_mode,
            unrealized_r,
            float(state.get("max_favorable_r", 0.0)),
            float(state.get("max_adverse_r", 0.0)),
            int(state.get("effective_horizon", getattr(pos, "horizon", 0))),
            action,
            reason,
        )

    def check_intrabar_exit(self, pos, candle_high: float, candle_low: float) -> Optional[str]:
        """Resolve TP/SL using high/low (not close) for paper mode accuracy."""
        if pos.is_long:
            sl_hit = candle_low <= pos.sl_price
            tp_hit = candle_high >= pos.tp_price
        else:
            sl_hit = candle_high >= pos.sl_price
            tp_hit = candle_low <= pos.tp_price

        if sl_hit and tp_hit:
            return "SL"
        if sl_hit:
            return "SL"
        if tp_hit:
            return "TP"
        return None
