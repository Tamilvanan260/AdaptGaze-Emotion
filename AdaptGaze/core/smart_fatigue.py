"""
AdaptGaze – Smart Fatigue AI
==============================
Predicts fatigue BEFORE it becomes critical by analysing trends in:
  - Blink duration (blinks get longer as eyes tire)
  - Blink interval (spacing becomes irregular when drowsy)
  - EAR trend      (average eye openness drops over time)
  - Gaze stability (gaze drifts more when fatigued)

Uses a lightweight online linear regression to detect deteriorating trends
without needing a pre-trained model — works from the first minute of use.

Fatigue Risk Levels:
  LOW      – user is fresh, no signs of fatigue
  MODERATE – early signs detected, monitor closely
  HIGH     – fatigue confirmed, recommend break
  CRITICAL – urgent break needed
"""

import time
import numpy as np
from collections import deque
from enum import Enum, auto
from typing import Optional

# ── Parameters ────────────────────────────────────────────────────────────────
HISTORY_WINDOW   = 120    # seconds of history to keep
TREND_WINDOW     = 60     # seconds used for trend computation
MIN_SAMPLES      = 10     # minimum blink samples before prediction
EAR_HISTORY_LEN  = 300    # frames of EAR to keep for trend


class FatigueRisk(Enum):
    LOW      = auto()
    MODERATE = auto()
    HIGH     = auto()
    CRITICAL = auto()


class SmartFatigueAI:
    """
    Online fatigue predictor using multi-signal trend analysis.
    No pre-training required — learns from the current session.
    """

    def __init__(self):
        # Blink event records: (timestamp, duration_seconds)
        self._blink_events: deque = deque()

        # EAR history: (timestamp, avg_ear)
        self._ear_history: deque = deque()

        # Gaze stability: (timestamp, dispersion)
        self._gaze_stability: deque = deque()

        self._session_start = time.time()
        self.risk_level = FatigueRisk.LOW

        # Computed signals
        self.ear_trend           = 0.0   # slope of EAR over time (negative = drooping)
        self.blink_duration_trend = 0.0  # slope of blink duration (positive = longer blinks)
        self.blink_interval_cv   = 0.0   # coefficient of variation of blink intervals
        self.predicted_fatigue   = 0.0   # composite score 0-1

    # ── Public API ────────────────────────────────────────────────────────────
    def record_blink(self, duration_seconds: float):
        """Call when a blink completes. Pass how long the eye was closed."""
        self._blink_events.append((time.time(), max(0.0, duration_seconds)))
        self._prune_history()

    def record_ear(self, left_ear: float, right_ear: float):
        """Call every frame with current EAR values."""
        avg = (left_ear + right_ear) / 2.0
        self._ear_history.append((time.time(), avg))
        if len(self._ear_history) > EAR_HISTORY_LEN:
            self._ear_history.popleft()

    def record_gaze_stability(self, dispersion: float):
        """Call every frame with gaze dispersion from AttentionTracker."""
        self._gaze_stability.append((time.time(), dispersion))
        self._prune_old(self._gaze_stability)

    def update(self) -> dict:
        """
        Compute current fatigue prediction.

        Returns
        -------
        dict:
            'risk_level'          : FatigueRisk
            'predicted_fatigue'   : float 0-1
            'ear_trend'           : float (negative = eye drooping)
            'blink_duration_trend': float (positive = longer blinks)
            'blink_interval_cv'   : float (high = irregular blinking)
            'recommendation'      : str
        """
        self._prune_history()

        # Compute each signal
        self.ear_trend            = self._compute_ear_trend()
        self.blink_duration_trend = self._compute_blink_duration_trend()
        self.blink_interval_cv    = self._compute_blink_interval_cv()
        gaze_drift                = self._compute_gaze_drift()

        # Normalise signals to [0, 1] fatigue contributions
        ear_fatigue      = float(np.clip(-self.ear_trend * 200, 0, 1))
        blink_fatigue    = float(np.clip(self.blink_duration_trend * 50, 0, 1))
        interval_fatigue = float(np.clip(self.blink_interval_cv, 0, 1))
        gaze_fatigue     = float(np.clip(gaze_drift * 10, 0, 1))

        # Weighted composite score
        self.predicted_fatigue = (
            0.35 * ear_fatigue +
            0.30 * blink_fatigue +
            0.20 * interval_fatigue +
            0.15 * gaze_fatigue
        )

        # Classify risk level
        self.risk_level = self._classify_risk()

        return {
            "risk_level":           self.risk_level,
            "predicted_fatigue":    round(self.predicted_fatigue, 3),
            "ear_trend":            round(self.ear_trend, 5),
            "blink_duration_trend": round(self.blink_duration_trend, 5),
            "blink_interval_cv":    round(self.blink_interval_cv, 3),
            "recommendation":       self._get_recommendation(),
        }

    # ── Signal computation ────────────────────────────────────────────────────
    def _compute_ear_trend(self) -> float:
        """Linear regression slope of EAR over time. Negative = drooping eyes."""
        if len(self._ear_history) < 30:
            return 0.0
        data = list(self._ear_history)
        t0 = data[0][0]
        X = np.array([d[0] - t0 for d in data])
        y = np.array([d[1] for d in data])
        return float(self._linear_slope(X, y))

    def _compute_blink_duration_trend(self) -> float:
        """Linear regression slope of blink duration. Positive = longer blinks = fatigue."""
        events = list(self._blink_events)
        if len(events) < MIN_SAMPLES:
            return 0.0
        t0 = events[0][0]
        X = np.array([e[0] - t0 for e in events])
        y = np.array([e[1] for e in events])
        return float(self._linear_slope(X, y))

    def _compute_blink_interval_cv(self) -> float:
        """
        Coefficient of variation of inter-blink intervals.
        High CV means irregular blinking – sign of drowsiness.
        """
        events = list(self._blink_events)
        if len(events) < MIN_SAMPLES:
            return 0.0
        times = np.array([e[0] for e in events])
        intervals = np.diff(times)
        if intervals.mean() < 1e-6:
            return 0.0
        return float(intervals.std() / intervals.mean())

    def _compute_gaze_drift(self) -> float:
        """Mean gaze dispersion over recent window."""
        data = list(self._gaze_stability)
        if len(data) < 5:
            return 0.0
        return float(np.mean([d[1] for d in data]))

    # ── Risk classification ───────────────────────────────────────────────────
    def _classify_risk(self) -> FatigueRisk:
        f = self.predicted_fatigue
        if f < 0.25:
            return FatigueRisk.LOW
        elif f < 0.50:
            return FatigueRisk.MODERATE
        elif f < 0.75:
            return FatigueRisk.HIGH
        else:
            return FatigueRisk.CRITICAL

    def _get_recommendation(self) -> str:
        recommendations = {
            FatigueRisk.LOW:      "No action needed. Eyes are fresh.",
            FatigueRisk.MODERATE: "Consider blinking more consciously.",
            FatigueRisk.HIGH:     "Take a short break. Look away from screen.",
            FatigueRisk.CRITICAL: "URGENT: Rest your eyes immediately (20-20-20 rule).",
        }
        return recommendations[self.risk_level]

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _linear_slope(X: np.ndarray, y: np.ndarray) -> float:
        """Compute slope of best-fit line using least squares."""
        if len(X) < 2:
            return 0.0
        X_mean = X.mean()
        y_mean = y.mean()
        denom = ((X - X_mean) ** 2).sum()
        if denom < 1e-10:
            return 0.0
        return float(((X - X_mean) * (y - y_mean)).sum() / denom)

    def _prune_history(self):
        cutoff = time.time() - HISTORY_WINDOW
        self._prune_old(self._blink_events, cutoff)
        self._prune_old(self._gaze_stability, cutoff)

    @staticmethod
    def _prune_old(dq: deque, cutoff: Optional[float] = None):
        if cutoff is None:
            cutoff = time.time() - HISTORY_WINDOW
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def reset(self):
        self._blink_events.clear()
        self._ear_history.clear()
        self._gaze_stability.clear()
        self._session_start = time.time()
        self.risk_level = FatigueRisk.LOW
        self.predicted_fatigue = 0.0
