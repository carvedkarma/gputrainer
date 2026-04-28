#!/bin/bash
# V7 ingest orchestrator. Mostly serial but launches several parallel
# subprocesses for the slow steps (klines + OI + flow). Idempotent / resumable.
set -u
cd "$(dirname "$0")/../.."
mkdir -p /tmp/v7_logs
LOG=/tmp/v7_logs/run_all.log

echo "=== $(date -u) START funding (all 20) ===" | tee -a "$LOG"
python3 -m gpu_trainer.data_ingest.cli funding 2>&1 | tee -a "$LOG"

echo "=== $(date -u) LAUNCH klines + OI + flow in parallel ===" | tee -a "$LOG"
python3 -m gpu_trainer.data_ingest.cli klines > /tmp/v7_logs/klines.log 2>&1 &
PID_K=$!
python3 -m gpu_trainer.data_ingest.cli oi > /tmp/v7_logs/oi.log 2>&1 &
PID_O=$!
python3 -m gpu_trainer.data_ingest.cli flow > /tmp/v7_logs/flow.log 2>&1 &
PID_F=$!

echo "PIDs: klines=$PID_K oi=$PID_O flow=$PID_F" | tee -a "$LOG"
wait $PID_K; echo "=== $(date -u) klines DONE rc=$? ===" | tee -a "$LOG"
wait $PID_O; echo "=== $(date -u) oi DONE rc=$? ===" | tee -a "$LOG"
wait $PID_F; echo "=== $(date -u) flow DONE rc=$? ===" | tee -a "$LOG"

echo "=== $(date -u) ALL DONE ===" | tee -a "$LOG"
