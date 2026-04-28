import json
import time
import uuid
import logging
import urllib.request
import urllib.error

log = logging.getLogger("v5")


class TrainingProgressPusher:
    def __init__(self, replit_url: str = None):
        self.replit_url = replit_url.rstrip('/') if replit_url else None
        self.session_id = None
        self.enabled = replit_url is not None
        self._last_epoch_push = 0
        self._epoch_push_interval = 1  # push every epoch for live monitor accuracy
        self._fold_start_times = {}
        self._session_start_time = None
        self._push_timeout = 30         # seconds — Replit can be slow under load
        self._push_max_retries = 2      # retry once before giving up

    def _push_event(self, event_type: str, payload: dict) -> dict:
        if not self.enabled or not self.replit_url:
            return {}
        url = f"{self.replit_url}/api/ingest/event"
        event = {
            "event_id": f"train_{uuid.uuid4().hex[:12]}",
            "type": event_type,
            "payload": payload,
            "ts": int(time.time() * 1000),
        }
        data = json.dumps(event).encode('utf-8')
        last_err = None
        for attempt in range(self._push_max_retries + 1):
            if attempt > 0:
                time.sleep(2 ** attempt)  # 2s, 4s back-off
                # Refresh event_id so the server treats it as a new event on retry
                event["event_id"] = f"train_{uuid.uuid4().hex[:12]}_r{attempt}"
                data = json.dumps(event).encode('utf-8')
                log.info(f"[TrainingPush] Retry {attempt}/{self._push_max_retries} for {event_type}")
            try:
                req = urllib.request.Request(url, data=data, headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'GPUTrainer/1.0',
                })
                with urllib.request.urlopen(req, timeout=self._push_timeout) as resp:
                    result = json.loads(resp.read().decode())
                    return result
            except urllib.error.HTTPError as e:
                try:
                    body = json.loads(e.read().decode())
                    if body.get("status") == "blocked":
                        log.warning(f"[TrainingPush] BLOCKED: {body.get('reason', 'unknown')}")
                        return body
                    log.warning(f"[TrainingPush] HTTP {e.code} for {event_type}: {body}")
                    return body  # don't retry HTTP errors (4xx/5xx)
                except Exception:
                    log.warning(f"[TrainingPush] HTTP {e.code} for {event_type}")
                    return {}
            except Exception as e:
                last_err = e
                log.warning(f"[TrainingPush] Attempt {attempt + 1} failed for {event_type}: {e}")
        log.warning(f"[TrainingPush] All retries exhausted for {event_type}: {last_err}")
        return {}

    def session_start(self, session_type: str, total_folds: int, total_epochs: int,
                      symbols: list, config: dict, gpu_name: str = None,
                      train_months: int = None, test_months: int = None):
        self._session_start_time = time.time()
        payload = {
            "session_type": session_type,
            "total_folds": total_folds,
            "total_epochs": total_epochs,
            "symbols": symbols,
            "config": config,
            "gpu_name": gpu_name,
            "train_months": train_months,
            "test_months": test_months,
        }
        result = self._push_event("TRAINING_SESSION_START", payload)
        if result.get("status") == "blocked":
            log.error(f"[TrainingPush] Cannot start: {result.get('reason', 'Previous sessions not cleared')}")
            log.error("[TrainingPush] Clear previous training sessions from the Training Monitor page before starting new training.")
            self.enabled = False
            return
        if result.get("status") == "accepted":
            self.session_id = result.get("session_id")
            log.info(f"[TrainingPush] Session accepted by server (id={self.session_id})")
        if not self.session_id:
            try:
                url = f"{self.replit_url}/api/training/active"
                req = urllib.request.Request(url, headers={'User-Agent': 'GPUTrainer/1.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("active"):
                        self.session_id = data["active"]["id"]
            except Exception:
                pass
        if self.session_id:
            log.info(f"[TrainingPush] Session started (id={self.session_id})")
        else:
            log.warning("[TrainingPush] Session started but session_id is None — push connectivity may have failed")

    def session_end(self, status: str = "completed", completed_folds: int = 0,
                    aggregate_metrics: dict = None, error_message: str = None):
        if not self.session_id:
            return
        self._push_event("TRAINING_SESSION_END", {
            "session_id": self.session_id,
            "status": status,
            "completed_at": int(time.time() * 1000),
            "completed_folds": completed_folds,
            "aggregate_metrics": aggregate_metrics,
            "error_message": error_message,
        })
        log.info(f"[TrainingPush] Session ended (status={status})")

    def fold_start(self, fold_num: int, train_start: str, train_end: str,
                   test_start: str, test_end: str):
        if not self.session_id:
            return
        self._fold_start_times[fold_num] = time.time()
        self._last_epoch_push = 0
        self._push_event("TRAINING_FOLD_START", {
            "session_id": self.session_id,
            "fold_num": fold_num,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    def fold_end(self, fold_num: int, completed_folds: int, report: dict = None,
                 aggregate_metrics: dict = None):
        if not self.session_id:
            return
        payload = {
            "session_id": self.session_id,
            "fold_num": fold_num,
            "completed_folds": completed_folds,
            "status": "completed",
            "aggregate_metrics": aggregate_metrics,
        }
        if report:
            payload.update({
                "trades": report.get("total_trades", 0),
                "win_rate": report.get("win_rate"),
                "expectancy": report.get("expectancy_r"),
                "profit_factor": report.get("profit_factor"),
                "sharpe": report.get("sharpe"),
                "max_drawdown": report.get("max_drawdown_r"),
                "total_r": report.get("total_r"),
                "long_short_ratio": f"{report.get('n_long', 0)}/{report.get('n_short', 0)}",
                "per_symbol": report.get("per_symbol_r"),
                "final_threshold": report.get("score_threshold"),
                "trail_win_pct": (report.get("n_trail_win", 0) / max(report.get("total_trades", 1), 1)),
                "trail_be_pct": (report.get("n_trail_be", 0) / max(report.get("total_trades", 1), 1)),
            })
        self._push_event("TRAINING_FOLD_END", payload)

    def epoch_update(self, fold_num: int, epoch: int, total_epochs: int,
                     train_loss: float = None, val_loss: float = None,
                     loss_breakdown: dict = None, action_accuracy: float = None,
                     learning_rate: float = None, sweep_metrics: dict = None,
                     total_folds: int = 1):
        if not self.session_id:
            return
        if epoch % self._epoch_push_interval != 0 and epoch != total_epochs - 1 and epoch != 1:
            return

        elapsed = time.time() - self._session_start_time if self._session_start_time else 0
        fold_elapsed = time.time() - self._fold_start_times.get(fold_num, time.time())
        if epoch > 0 and fold_elapsed > 0:
            epoch_time = fold_elapsed / epoch
            remaining_epochs = total_epochs - epoch
            remaining_folds = total_folds - fold_num
            eta_seconds = remaining_epochs * epoch_time + (remaining_folds - 1) * total_epochs * epoch_time
            estimated_completion = int((time.time() + eta_seconds) * 1000)
        else:
            estimated_completion = None

        payload = {
            "session_id": self.session_id,
            "fold_num": fold_num,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_breakdown": loss_breakdown,
            "action_accuracy": action_accuracy,
            "learning_rate": learning_rate,
            "estimated_completion_ts": estimated_completion,
        }
        if sweep_metrics:
            payload.update({
                "expectancy": sweep_metrics.get("expectancy"),
                "profit_factor": sweep_metrics.get("profit_factor"),
                "win_rate": sweep_metrics.get("win_rate"),
                "max_drawdown": sweep_metrics.get("max_drawdown"),
                "trades_per_day": sweep_metrics.get("trades_per_day"),
                "threshold": sweep_metrics.get("threshold"),
                "score_diag": sweep_metrics.get("score_diag"),
            })
        self._push_event("TRAINING_EPOCH", payload)

    def session_update(self, current_fold: int = None, completed_folds: int = None,
                       current_epoch: int = None, aggregate_metrics: dict = None):
        if not self.session_id:
            return
        self._push_event("TRAINING_SESSION_UPDATE", {
            "session_id": self.session_id,
            "current_fold": current_fold,
            "completed_folds": completed_folds,
            "current_epoch": current_epoch,
            "aggregate_metrics": aggregate_metrics,
        })
