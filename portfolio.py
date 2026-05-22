"""Portfolio manager for multi-asset live trading.

Tracks open positions, enforces cooldowns, risk caps, and correlation rules.
Paper mode by default — no exchange orders, only logging.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("Portfolio")


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    entry_time: float
    atr: float
    tp_price: float
    sl_price: float
    p_enter: float
    size_mult: float
    risk_pct: float
    bar_index: int = 0
    lane: str = "CORE"
    horizon: int = 24
    htf_score: int = 0
    threshold_used: float = 0.85
    dashboard_trade_id: Optional[int] = None
    original_sl: Optional[float] = None
    breakeven_moved: bool = False

    def __post_init__(self):
        if self.original_sl is None:
            self.original_sl = self.sl_price

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"

    @property
    def bars_open(self) -> int:
        return 0

    def check_exit(self, current_price: float, current_bar: int = 0,
                   candle_high: Optional[float] = None,
                   candle_low: Optional[float] = None) -> Optional[str]:
        use_high = candle_high if candle_high is not None else current_price
        use_low = candle_low if candle_low is not None else current_price
        intrabar = candle_high is not None or candle_low is not None

        if self.is_long:
            sl_hit = use_low <= self.sl_price
            tp_hit = use_high >= self.tp_price
        else:
            sl_hit = use_high >= self.sl_price
            tp_hit = use_low <= self.tp_price

        outcome = None
        if sl_hit and tp_hit:
            outcome = "SL"
        elif sl_hit:
            outcome = "SL"
        elif tp_hit:
            outcome = "TP"

        if outcome is not None:
            log.info(f"[EXIT_RESOLVE] sym={self.symbol} intrabar={intrabar} "
                     f"tp_hit={tp_hit} sl_hit={sl_hit} outcome={outcome} "
                     f"lane={self.lane} bars_held={current_bar - self.bar_index if current_bar > 0 and self.bar_index > 0 else 0}")
            return outcome

        if current_bar > 0 and self.bar_index > 0:
            bars_held = current_bar - self.bar_index
            if bars_held >= self.horizon:
                log.info(f"[EXIT_RESOLVE] sym={self.symbol} intrabar={intrabar} "
                         f"tp_hit=False sl_hit=False outcome=TIME_EXIT "
                         f"lane={self.lane} bars_held={bars_held}")
                return "TIME_EXIT"
        return None


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    outcome: str
    gross_r: float
    p_enter: float


class PortfolioManager:
    def __init__(
        self,
        max_positions_total: int = 2,
        max_positions_per_symbol: int = 1,
        risk_cap_total_pct: float = 10.0,
        risk_cap_symbol_pct: float = 5.0,
        cooldown_bars: int = 8,
        block_correlated_same_dir: bool = True,
        correlated_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.max_positions_total = max_positions_total
        self.max_positions_per_symbol = max_positions_per_symbol
        self.risk_cap_total_pct = risk_cap_total_pct
        self.risk_cap_symbol_pct = risk_cap_symbol_pct
        self.cooldown_bars = max(int(cooldown_bars), 0)
        self.block_correlated_same_dir = block_correlated_same_dir
        self.correlated_pairs = correlated_pairs or [("BTCUSDT", "ETHUSDT")]

        self.on_close_callback: Optional[callable] = None
        self.verifier = None
        self.open_positions: Dict[str, Position] = {}
        self.last_trade_bar: Dict[str, int] = {}
        self.trade_history: List[TradeRecord] = []
        self.current_bar: int = 0

    def set_bar(self, bar_index: int):
        self.current_bar = bar_index

    def total_open(self) -> int:
        return len(self.open_positions)

    def symbol_open(self, symbol: str) -> bool:
        return symbol in self.open_positions

    def total_risk_pct(self) -> float:
        return sum(p.risk_pct for p in self.open_positions.values())

    def symbol_risk_pct(self, symbol: str) -> float:
        if symbol in self.open_positions:
            return self.open_positions[symbol].risk_pct
        return 0.0

    def cooldown_ok(self, symbol: str) -> bool:
        last = self.last_trade_bar.get(symbol, -self.cooldown_bars - 1)
        return (self.current_bar - last) >= self.cooldown_bars

    def _correlated_block(self, symbol: str, side: str) -> Optional[str]:
        if not self.block_correlated_same_dir:
            return None
        for a, b in self.correlated_pairs:
            partner = None
            if symbol == a and b in self.open_positions:
                partner = b
            elif symbol == b and a in self.open_positions:
                partner = a
            if partner and self.open_positions[partner].side == side:
                return partner
        return None

    def can_enter(self, symbol: str, side: str, risk_pct: float) -> Tuple[bool, str]:
        if self.total_open() >= self.max_positions_total:
            return False, f"max total positions ({self.max_positions_total}) reached"

        if self.symbol_open(symbol):
            return False, f"position already open for {symbol}"

        if not self.cooldown_ok(symbol):
            bars_left = max(
                self.cooldown_bars - (self.current_bar - self.last_trade_bar.get(symbol, 0)),
                0,
            )
            return False, f"cooldown active for {symbol} ({bars_left} bars left)"

        if self.total_risk_pct() + risk_pct > self.risk_cap_total_pct:
            return False, f"total risk cap {self.risk_cap_total_pct}% would be exceeded ({self.total_risk_pct():.1f}% + {risk_pct:.1f}%)"

        if risk_pct > self.risk_cap_symbol_pct:
            return False, f"per-symbol risk cap {self.risk_cap_symbol_pct}% exceeded ({risk_pct:.1f}%)"

        blocked_by = self._correlated_block(symbol, side)
        if blocked_by:
            log.info(f"[CROSS_BLOCK] sym={symbol} blocked_with={blocked_by} dir={side} "
                     f"reason=same_direction_correlated_exposure")
            if self.verifier:
                fs = self.verifier.verify_cross_blocking(
                    self.verifier.stats.total_cycles, symbol, side,
                    True, blocked_by, self.open_positions)
                self.verifier.add_failures(fs)
            return False, f"correlated pair {blocked_by} already has same-direction ({side}) position"

        if self.verifier:
            fs = self.verifier.verify_cross_blocking(
                self.verifier.stats.total_cycles, symbol, side,
                False, None, self.open_positions)
            self.verifier.add_failures(fs)

        return True, "OK"

    def open_position(self, pos: Position):
        self.open_positions[pos.symbol] = pos
        self.last_trade_bar[pos.symbol] = self.current_bar
        log.info(f"[OPEN] {pos.symbol} {pos.side} @ {pos.entry_price:.2f} | "
                 f"SL={pos.sl_price:.2f} TP={pos.tp_price:.2f} | "
                 f"p_enter={pos.p_enter:.4f} risk={pos.risk_pct:.1f}%")

    def close_position(self, symbol: str, exit_price: float, outcome: str):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions.pop(symbol)
        initial_sl = pos.original_sl if pos.original_sl is not None else pos.sl_price
        original_risk_abs = abs(pos.entry_price - initial_sl)
        if pos.is_long:
            gross_r = (exit_price - pos.entry_price) / original_risk_abs if original_risk_abs > 0 else 0
        else:
            gross_r = (pos.entry_price - exit_price) / original_risk_abs if original_risk_abs > 0 else 0

        record = TradeRecord(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=time.time(),
            outcome=outcome, gross_r=gross_r, p_enter=pos.p_enter,
        )
        self.trade_history.append(record)
        log.info(f"[CLOSE] {symbol} {pos.side} @ {exit_price:.2f} | "
                 f"outcome={outcome} R={gross_r:+.2f} lane={pos.lane}")

        if self.on_close_callback and pos.dashboard_trade_id:
            try:
                self.on_close_callback(pos, exit_price, outcome, gross_r)
            except Exception as e:
                log.warning(f"on_close_callback failed for {symbol}: {e}")

    def check_exits(self, prices: Dict[str, float],
                     highs: Optional[Dict[str, float]] = None,
                     lows: Optional[Dict[str, float]] = None):
        to_close = []
        highs = highs or {}
        lows = lows or {}
        for symbol, pos in self.open_positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            candle_high = highs.get(symbol)
            candle_low = lows.get(symbol)
            outcome = pos.check_exit(price, current_bar=self.current_bar,
                                      candle_high=candle_high, candle_low=candle_low)
            if outcome:
                exit_price = price
                if outcome == "SL":
                    exit_price = pos.sl_price
                elif outcome == "TP":
                    exit_price = pos.tp_price
                to_close.append((symbol, exit_price, outcome))
        for symbol, price, outcome in to_close:
            self.close_position(symbol, price, outcome)

    def rank_candidates(self, candidates: List[dict]) -> List[dict]:
        return sorted(candidates, key=lambda c: (-c['p_enter'], -c.get('expected_net_r', 0)))

    def filter_and_rank(self, candidates: List[dict]) -> List[dict]:
        """Stateful per-cycle selection: tentatively reserves slots/risk so caps
        and correlation rules apply across same-cycle candidates."""
        ranked = self.rank_candidates(candidates)
        accepted = []
        tentative_positions: Dict[str, str] = {}
        tentative_risk = 0.0

        for c in ranked:
            symbol = c['symbol']
            side = c['side']
            risk = c.get('risk_pct', 2.0)

            ok, reason = self.can_enter(symbol, side, risk)
            if not ok:
                c['reject_reason'] = reason
                log.info(f"[SKIP] {symbol} {side} p_enter={c['p_enter']:.4f} — {reason}")
                continue

            if self.total_open() + len(tentative_positions) >= self.max_positions_total:
                c['reject_reason'] = "tentative max total positions reached this cycle"
                log.info(f"[SKIP] {symbol} {side} p_enter={c['p_enter']:.4f} — max total positions (tentative)")
                continue

            if symbol in tentative_positions:
                c['reject_reason'] = "tentative duplicate symbol this cycle"
                log.info(f"[SKIP] {symbol} {side} — already selected this cycle")
                continue

            if self.total_risk_pct() + tentative_risk + risk > self.risk_cap_total_pct:
                c['reject_reason'] = "tentative risk cap exceeded"
                log.info(f"[SKIP] {symbol} {side} — tentative risk cap exceeded")
                continue

            if self.block_correlated_same_dir:
                blocked = False
                for a, b in self.correlated_pairs:
                    partner = None
                    if symbol == a:
                        partner = b
                    elif symbol == b:
                        partner = a
                    if partner and partner in tentative_positions and tentative_positions[partner] == side:
                        c['reject_reason'] = f"tentative correlated same-direction with {partner}"
                        log.info(f"[SKIP] {symbol} {side} — correlated with tentative {partner} {side}")
                        blocked = True
                        break
                if blocked:
                    continue

            tentative_positions[symbol] = side
            tentative_risk += risk
            accepted.append(c)
            c['accept_reason'] = "OK"

        return accepted

    def summary(self) -> dict:
        wins = [t for t in self.trade_history if t.gross_r > 0]
        losses = [t for t in self.trade_history if t.gross_r <= 0]
        total = len(self.trade_history)
        return {
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / total if total > 0 else 0,
            'avg_r': sum(t.gross_r for t in self.trade_history) / total if total > 0 else 0,
            'open_positions': len(self.open_positions),
            'total_risk_pct': self.total_risk_pct(),
        }
