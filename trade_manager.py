"""Smart Trade Manager v4.4.1 — Dynamic exit intelligence.

Evaluates open positions each cycle and returns actions:
HOLD | MOVE_SL | TRAIL_SL | CLOSE_PARTIAL | CLOSE_FULL

Safety limits:
  MAX_SL_UPDATES (6)  — hard cap on SL moves per trade to prevent flip-flop.
  MIN_BARS_BETWEEN_SL (1) — cooldown between SL updates to prevent micro-jitter.

Uses existing lanes + HTF score + p_enter — no model retraining needed.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

log = logging.getLogger("TradeMgr")

BREAKEVEN_TRIGGER_R = 0.35
TRAIL_TRIGGER_R = 0.60
TRAIL_DISTANCE_R = 0.40
STALL_TRIGGER_R = 0.70
STALL_BARS = 2
ADVERSE_FLIP_R = -0.60
ADVERSE_HTF_DROP = 2
ADVERSE_P_ENTER_DECAY = 0.80
MAX_SL_UPDATES = 6
MIN_BARS_BETWEEN_SL = 1


@dataclass
class TMAction:
    action: str
    reason: str
    new_sl: Optional[float] = None
    close_price: Optional[float] = None


class TradeManager:
    """Manages dynamic exits for open positions."""

    def __init__(self):
        self.position_state: Dict[str, dict] = {}

    def _ensure_state(self, symbol: str, pos) -> dict:
        if symbol not in self.position_state:
            risk_per_unit = abs(pos.entry_price - pos.original_sl) if hasattr(pos, 'original_sl') else abs(pos.entry_price - pos.sl_price)
            self.position_state[symbol] = {
                'peak_price': pos.entry_price,
                'trough_price': pos.entry_price,
                'max_favorable_r': 0.0,
                'max_adverse_r': 0.0,
                'stall_counter': 0,
                'last_mfe': 0.0,
                'breakeven_moved': False,
                'risk_per_unit': risk_per_unit,
                'original_sl': pos.sl_price,
                'entry_htf_score': pos.htf_score,
                'sl_update_count': 0,
                'last_sl_update_bar': -999,
            }
        return self.position_state[symbol]

    def update_position(self, symbol: str, pos, current_price: float,
                        current_bar: int, current_htf_score: Optional[int] = None,
                        current_p_enter: Optional[float] = None,
                        candle_high: Optional[float] = None,
                        candle_low: Optional[float] = None) -> TMAction:
        """Evaluate one open position and return the recommended action.

        Priority order: TIME_EXIT > ADVERSE_FLIP > STALL_TAKEPROFIT > TRAIL_SL > BREAKEVEN > HOLD
        """
        state = self._ensure_state(symbol, pos)
        risk_per_unit = state['risk_per_unit']
        if risk_per_unit <= 0:
            return TMAction(action="HOLD", reason="zero_risk_per_unit")

        use_high = candle_high if candle_high is not None else current_price
        use_low = candle_low if candle_low is not None else current_price

        if pos.is_long:
            if use_high > state['peak_price']:
                state['peak_price'] = use_high
            if use_low < state['trough_price']:
                state['trough_price'] = use_low
            favorable_r = (state['peak_price'] - pos.entry_price) / risk_per_unit
            adverse_r = (state['trough_price'] - pos.entry_price) / risk_per_unit
            unrealized_r = (current_price - pos.entry_price) / risk_per_unit
        else:
            if use_low < state['peak_price']:
                state['peak_price'] = use_low
            if use_high > state['trough_price']:
                state['trough_price'] = use_high
            favorable_r = (pos.entry_price - state['peak_price']) / risk_per_unit
            adverse_r = (pos.entry_price - state['trough_price']) / risk_per_unit
            unrealized_r = (pos.entry_price - current_price) / risk_per_unit

        state['max_favorable_r'] = max(state['max_favorable_r'], favorable_r)
        state['max_adverse_r'] = min(state['max_adverse_r'], adverse_r)

        bars_held = current_bar - pos.bar_index if pos.bar_index > 0 else 0

        if bars_held >= pos.horizon:
            self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", "TIME_EXIT")
            return TMAction(action="CLOSE_FULL", reason="TIME_EXIT", close_price=current_price)

        if unrealized_r <= ADVERSE_FLIP_R:
            htf_dropped = False
            p_decayed = False
            if current_htf_score is not None:
                htf_drop = state['entry_htf_score'] - current_htf_score
                if htf_drop >= ADVERSE_HTF_DROP:
                    htf_dropped = True
            if current_p_enter is not None and pos.threshold_used > 0:
                if current_p_enter < pos.threshold_used * ADVERSE_P_ENTER_DECAY:
                    p_decayed = True
            if htf_dropped or p_decayed:
                reason_detail = []
                if htf_dropped:
                    reason_detail.append(f"htf_drop={state['entry_htf_score']}->{current_htf_score}")
                if p_decayed:
                    reason_detail.append(f"p_decay={current_p_enter:.4f}<{pos.threshold_used*ADVERSE_P_ENTER_DECAY:.4f}")
                reason = f"ADVERSE_FLIP({','.join(reason_detail)})"
                self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", reason)
                return TMAction(action="CLOSE_FULL", reason="ADVERSE_FLIP", close_price=current_price)

        if unrealized_r >= STALL_TRIGGER_R:
            if state['max_favorable_r'] <= state['last_mfe'] + 0.01:
                state['stall_counter'] += 1
            else:
                state['stall_counter'] = 0
            state['last_mfe'] = state['max_favorable_r']

            if state['stall_counter'] >= STALL_BARS:
                self._log_action(symbol, pos, unrealized_r, state, "CLOSE_FULL", "STALL_TAKEPROFIT")
                return TMAction(action="CLOSE_FULL", reason="STALL_TAKEPROFIT", close_price=current_price)
        else:
            state['stall_counter'] = 0
            state['last_mfe'] = state['max_favorable_r']

        sl_allowed = (state['sl_update_count'] < MAX_SL_UPDATES and
                      current_bar - state['last_sl_update_bar'] >= MIN_BARS_BETWEEN_SL)

        if state['max_favorable_r'] >= TRAIL_TRIGGER_R:
            trail_r = TRAIL_DISTANCE_R
            if pos.is_long:
                new_sl = state['peak_price'] - trail_r * risk_per_unit
                if new_sl > pos.sl_price:
                    if not sl_allowed:
                        log.debug(f"[TM_SL] sym={symbol} TRAIL_SL suppressed "
                                  f"(updates={state['sl_update_count']}/{MAX_SL_UPDATES}, "
                                  f"bars_since_last={current_bar - state['last_sl_update_bar']}/{MIN_BARS_BETWEEN_SL})")
                    else:
                        state['sl_update_count'] += 1
                        state['last_sl_update_bar'] = current_bar
                        log.info(f"[TM_SL] sym={symbol} action=TRAIL_SL new_sl={new_sl:.2f} "
                                 f"old_sl={pos.sl_price:.2f} peak={state['peak_price']:.2f} "
                                 f"mfe={state['max_favorable_r']:.2f}R "
                                 f"(update {state['sl_update_count']}/{MAX_SL_UPDATES})")
                        return TMAction(action="TRAIL_SL", reason="TRAIL_LOCK", new_sl=new_sl)
            else:
                new_sl = state['peak_price'] + trail_r * risk_per_unit
                if new_sl < pos.sl_price:
                    if not sl_allowed:
                        log.debug(f"[TM_SL] sym={symbol} TRAIL_SL suppressed "
                                  f"(updates={state['sl_update_count']}/{MAX_SL_UPDATES}, "
                                  f"bars_since_last={current_bar - state['last_sl_update_bar']}/{MIN_BARS_BETWEEN_SL})")
                    else:
                        state['sl_update_count'] += 1
                        state['last_sl_update_bar'] = current_bar
                        log.info(f"[TM_SL] sym={symbol} action=TRAIL_SL new_sl={new_sl:.2f} "
                                 f"old_sl={pos.sl_price:.2f} peak={state['peak_price']:.2f} "
                                 f"mfe={state['max_favorable_r']:.2f}R "
                                 f"(update {state['sl_update_count']}/{MAX_SL_UPDATES})")
                        return TMAction(action="TRAIL_SL", reason="TRAIL_LOCK", new_sl=new_sl)

        if state['max_favorable_r'] >= BREAKEVEN_TRIGGER_R and not state['breakeven_moved']:
            if pos.is_long and pos.sl_price < pos.entry_price:
                if not sl_allowed:
                    log.debug(f"[TM_SL] sym={symbol} BREAKEVEN suppressed "
                              f"(updates={state['sl_update_count']}/{MAX_SL_UPDATES}, "
                              f"bars_since_last={current_bar - state['last_sl_update_bar']}/{MIN_BARS_BETWEEN_SL})")
                else:
                    state['breakeven_moved'] = True
                    state['sl_update_count'] += 1
                    state['last_sl_update_bar'] = current_bar
                    log.info(f"[TM_SL] sym={symbol} action=BREAKEVEN new_sl={pos.entry_price:.2f} "
                             f"old_sl={pos.sl_price:.2f} mfe={state['max_favorable_r']:.2f}R "
                             f"(update {state['sl_update_count']}/{MAX_SL_UPDATES})")
                    return TMAction(action="MOVE_SL", reason="BREAKEVEN", new_sl=pos.entry_price)
            elif not pos.is_long and pos.sl_price > pos.entry_price:
                if not sl_allowed:
                    log.debug(f"[TM_SL] sym={symbol} BREAKEVEN suppressed "
                              f"(updates={state['sl_update_count']}/{MAX_SL_UPDATES}, "
                              f"bars_since_last={current_bar - state['last_sl_update_bar']}/{MIN_BARS_BETWEEN_SL})")
                else:
                    state['breakeven_moved'] = True
                    state['sl_update_count'] += 1
                    state['last_sl_update_bar'] = current_bar
                    log.info(f"[TM_SL] sym={symbol} action=BREAKEVEN new_sl={pos.entry_price:.2f} "
                             f"old_sl={pos.sl_price:.2f} mfe={state['max_favorable_r']:.2f}R "
                             f"(update {state['sl_update_count']}/{MAX_SL_UPDATES})")
                    return TMAction(action="MOVE_SL", reason="BREAKEVEN", new_sl=pos.entry_price)

        return TMAction(action="HOLD", reason="no_exit_trigger")

    def get_state(self, symbol: str) -> Optional[dict]:
        return self.position_state.get(symbol)

    def clear_position(self, symbol: str):
        self.position_state.pop(symbol, None)

    def _log_action(self, symbol: str, pos, unrealized_r: float, state: dict,
                     action: str, reason: str):
        log.info(f"[TM] sym={symbol} lane={pos.lane} uR={unrealized_r:+.2f} "
                 f"mfe={state['max_favorable_r']:+.2f} mae={state['max_adverse_r']:+.2f} "
                 f"action={action} reason={reason}")

    def check_intrabar_exit(self, pos, candle_high: float, candle_low: float) -> Optional[str]:
        """Resolve TP/SL using high/low (not close) for paper mode accuracy.

        Returns: 'TP', 'SL', or None.
        If both hit in same bar, assume worst-case (SL first) unless 1m data available.
        """
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
