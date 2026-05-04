# Mythos-First Real-Time Paper Trading Dashboard Architecture

## 1) Objective

Build a localhost-ready, animated, modern dashboard where **Mythos is the base decision engine** for:

- real-time signal generation,
- paper order simulation and lifecycle management,
- signal/trade history, and
- model observability and controls.

This architecture is designed to keep your current trainer stack and `LiveRunner` while adding a robust app layer around it.

---

## 2) What is wired now (backend baseline)

The live/paper runner now supports selecting backend model runtime with:

- `--live-model v5` (existing behavior)
- `--live-model mythos` (new Mythos runtime path)

Implemented runtime path:

- Loads `mythos_best_<SYMBOL>.json` artifacts (default search in `checkpoints/mythos_models/`).
- Reconstructs world model + experts + router for inference.
- Produces Mythos signal decisions (`side`, `edge`, `confidence`, `uncertainty`, `regime`, `expert`).
- Routes into existing paper/live execution path, logging lane as `MYTHOS`.

This gives us a stable foundation for dashboard integration without replacing your execution and trade-management internals.

---

## 3) System Architecture (high level)

```text
Binance/Market Data
        |
        v
[Data Ingest + Feature Builder]
        |
        v
[Mythos Runtime Inference Service]
        |
        +----> [Signal Bus/Event Stream] ----> [Dashboard WS/API]
        |
        v
[Paper Execution Engine]
        |
        v
[Trade State Store + Analytics Store]
        |
        +----> [Frontend: Animated Dashboard]
```

### Core components

1. **Market Data Adapter**
   - Pulls 15m candles + optional 1h/4h context.
   - Normalizes timestamps and health checks (staleness/error counters).

2. **Mythos Runtime Service**
   - Loads artifact per symbol.
   - Recomputes Mythos features on rolling window.
   - Emits decision packet:
     - side/abstain
     - edge/confidence/uncertainty
     - regime id
     - selected expert
     - reason code

3. **Paper Trading Engine**
   - Accepts only Mythos-approved entries.
   - Simulates fills, SL/TP exits, cooldown, and risk caps.
   - Emits position lifecycle events.

4. **Event + State Layer**
   - Append-only signal/trade events (for audit + replay).
   - Materialized current state:
     - open positions
     - PnL/equity curve
     - per-side win metrics
     - per-regime metrics

5. **Dashboard API Layer**
   - REST for historical queries.
   - WebSocket for live animation streams (ticks, signals, fills, PnL deltas).

6. **Frontend Dashboard**
   - Modern animated UI.
   - Real-time trade blotter + signal timeline + model internals.

---

## 4) Recommended frontend architecture (animated modern style)

- **Framework**: Next.js + TypeScript
- **Styling**: TailwindCSS + CSS variables for theme
- **Animation**: Framer Motion
- **Charts**: Lightweight Charts + ECharts (for dense performance views)
- **State**: Zustand (UI state) + React Query (API cache)
- **Realtime**: WebSocket client with reconnect + backfill

### Frontend screens

1. **Overview**
   - Equity curve, daily PnL, exposure, win rate.
   - Live status pills (Data OK / Mythos Ready / Paper Engine Running).

2. **Signals Monitor**
   - Streaming cards/table of latest Mythos decisions.
   - Filters: symbol, side, regime, confidence band.
   - Visual confidence meter and edge sparkline.

3. **Trade Manager**
   - Open positions with live R-multiple and SL/TP distance.
   - Closed trades with outcomes + reasons.
   - Manual controls (paper-only): close, disable symbol, pause strategy.

4. **Mythos Intelligence Panel**
   - Regime map over time.
   - Expert selection frequency.
   - Abstain reasons distribution.
   - LONG vs SHORT performance split.

5. **History & Analytics**
   - Per-symbol, per-regime, per-session analytics.
   - Fold-derived comparison views and recent drift panels.

---

## 5) Backend API contract (minimum)

### REST

- `GET /api/dashboard/status`
- `GET /api/dashboard/positions/open`
- `GET /api/dashboard/trades?from=&to=&symbol=&side=`
- `GET /api/dashboard/signals?from=&to=&symbol=&regime=`
- `GET /api/dashboard/metrics/summary`
- `GET /api/dashboard/metrics/long-short`
- `GET /api/dashboard/metrics/regimes`
- `POST /api/dashboard/control/pause`
- `POST /api/dashboard/control/resume`

### WebSocket (`/ws/dashboard`)

Event envelopes:

- `signal.created`
- `trade.opened`
- `trade.updated`
- `trade.closed`
- `portfolio.updated`
- `health.updated`

Each event should include `event_id`, `ts`, `symbol`, and `source` (`MYTHOS`).

---

## 6) Data model (minimum tables/collections)

1. `signals`
   - id, ts, symbol, side, edge, confidence, uncertainty, regime, expert, abstain, reason, source

2. `trades`
   - id, symbol, side, entry_ts, exit_ts, entry_price, exit_price, sl, tp, pnl_r, pnl_usd, status, source_signal_id

3. `positions_open`
   - symbol, side, entry_price, current_price, unrealized_r, stop_loss, take_profit, age_bars

4. `portfolio_snapshots`
   - ts, equity_r, equity_usd, drawdown_r, exposure_pct, open_positions

5. `model_health`
   - ts, symbol, data_staleness_s, api_error_count, artifact_version, runtime_mode

---

## 7) Execution/risk architecture

- Global kill-switch for entries (paper/live independent).
- Per-symbol enable/disable.
- Max concurrent positions + per-side caps.
- Daily/weekly loss guards.
- Trade cooldown and stale-data halt.
- Every blocked entry logs explicit reason (for dashboard debugging).

---

## 8) Rollout plan

### Phase A (complete now)
- Mythos runtime selected by `--live-model mythos`.
- Signals flow through existing runner + paper executor.

### Phase B
- Add dedicated dashboard API namespace + WS stream.
- Persist normalized signal/trade/model-health entities.

### Phase C
- Build animated frontend shell (overview, signals, trades).
- Wire live updates + playback mode.

### Phase D
- Add intelligence analytics (regime/expert/side decomposition).
- Add controls and operational safety panels.

---

## 9) Localhost run model

1. Train/save Mythos best artifact (already in training workflow).
2. Start API/backend services.
3. Run paper loop with Mythos backend:

```bash
python3 quick_start.py --live --paper --live-model mythos --symbols BTCUSDT --interval 15m --execution-mode paper --record-trades --url http://localhost:3000
```

4. Start frontend app (`localhost`) connected to REST + WS API.

---

## 10) Notes for future production hardening

- Add artifact versioning + checksum validation.
- Add deterministic replay mode for postmortems.
- Add queue-backed event delivery for resilience.
- Add per-component health probes and alerting.
