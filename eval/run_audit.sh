#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
mkdir -p .local/reports /tmp/v7_logs
LOG=/tmp/v7_logs/audit.log
echo "=== $(date -u) START AUDIT ===" | tee -a "$LOG"
python3 -W ignore -m gpu_trainer.eval.v7_signal_audit_augmented 2>&1 | tee -a "$LOG"
echo "=== $(date -u) DONE ===" | tee -a "$LOG"
