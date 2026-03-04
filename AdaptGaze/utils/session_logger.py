"""
AdaptGaze - Session Logger
============================
Logs session events to a timestamped CSV file in the logs/ directory.

Logged events include:
  - session_start / session_end
  - blink (single / double / long)
  - fatigue_level changes
  - rest_reminder
  - model_loaded / calibration_done
  - error messages
"""

import os
import csv
import time
import threading
from datetime import datetime
from config.settings import LOG_DIR, LOG_FLUSH_INTERVAL


class SessionLogger:
    """Thread-safe CSV session event logger."""

    FIELDS = ["timestamp", "elapsed_s", "event", "detail"]

    def __init__(self):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"session_{session_id}.csv")
        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._log_path = log_path

        # Periodic flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

        self.log("session_start", f"log={log_path}")
        print(f"[SessionLogger] Logging to {log_path}")

    # ------------------------------------------------------------------
    def log(self, event: str, detail: str = ""):
        """Append one event row."""
        now = time.time()
        row = {
            "timestamp": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "elapsed_s": round(now - self._start_time, 2),
            "event":     event,
            "detail":    detail,
        }
        with self._lock:
            self._writer.writerow(row)

    # ------------------------------------------------------------------
    def _periodic_flush(self):
        while not self._stop_event.wait(LOG_FLUSH_INTERVAL):
            with self._lock:
                self._file.flush()

    # ------------------------------------------------------------------
    def close(self):
        self.log("session_end",
                 f"duration_s={round(time.time() - self._start_time, 1)}")
        self._stop_event.set()
        with self._lock:
            self._file.flush()
            self._file.close()

    @property
    def log_path(self) -> str:
        return self._log_path
