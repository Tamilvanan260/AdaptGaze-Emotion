"""
AdaptGaze - Fatigue Monitor
=============================
Monitors blink rate within a rolling time window and classifies
the user's fatigue level as NORMAL, STRAINING, or DROWSY.

Also tracks total session duration and emits a rest reminder every 20 min.
"""

import time
from collections import deque
from enum import Enum, auto

from config.settings import (
    FATIGUE_WINDOW_SECONDS,
    FATIGUE_HIGH_BLINK_RATE,
    FATIGUE_LOW_BLINK_RATE,
)

REST_REMINDER_INTERVAL = 20 * 60  # 20 minutes in seconds


class FatigueLevel(Enum):
    NORMAL    = auto()
    STRAINING = auto()  # high blink rate – eye strain
    DROWSY    = auto()  # low blink rate  – drowsiness


class FatigueMonitor:
    """Rolling-window blink-rate fatigue classifier."""

    def __init__(self):
        self._blink_times: deque = deque()  # timestamps of recent blinks
        self._session_start = time.time()
        self._last_reminder_time = self._session_start
        self.fatigue_level = FatigueLevel.NORMAL
        self.blink_rate = 0.0          # blinks per minute
        self.session_elapsed = 0.0     # seconds

    # ------------------------------------------------------------------
    def record_blink(self):
        """Call this every time a blink is detected."""
        self._blink_times.append(time.time())

    # ------------------------------------------------------------------
    def update(self) -> dict:
        """
        Call once per frame to get current fatigue state.

        Returns
        -------
        dict:
            'blink_rate'    : float – blinks per minute
            'fatigue_level' : FatigueLevel
            'rest_reminder' : bool – True if user should take a break
            'elapsed_min'   : float – session duration in minutes
        """
        now = time.time()
        self.session_elapsed = now - self._session_start

        # Prune old blink timestamps outside the rolling window
        cutoff = now - FATIGUE_WINDOW_SECONDS
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        # Compute blink rate (blinks per minute)
        self.blink_rate = len(self._blink_times) * (60.0 / FATIGUE_WINDOW_SECONDS)

        # Classify fatigue level
        if self.blink_rate > FATIGUE_HIGH_BLINK_RATE:
            self.fatigue_level = FatigueLevel.STRAINING
        elif self.blink_rate < FATIGUE_LOW_BLINK_RATE:
            self.fatigue_level = FatigueLevel.DROWSY
        else:
            self.fatigue_level = FatigueLevel.NORMAL

        # Check rest reminder
        rest_reminder = (now - self._last_reminder_time) >= REST_REMINDER_INTERVAL
        if rest_reminder:
            self._last_reminder_time = now

        return {
            "blink_rate":    round(self.blink_rate, 1),
            "fatigue_level": self.fatigue_level,
            "rest_reminder": rest_reminder,
            "elapsed_min":   round(self.session_elapsed / 60.0, 1),
        }

    # ------------------------------------------------------------------
    def reset(self):
        self._blink_times.clear()
        self._session_start = time.time()
        self._last_reminder_time = self._session_start
        self.fatigue_level = FatigueLevel.NORMAL
        self.blink_rate = 0.0
        self.session_elapsed = 0.0
