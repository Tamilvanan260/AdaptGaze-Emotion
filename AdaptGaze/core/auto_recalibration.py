"""
AdaptGaze – Auto-Recalibration Monitor
========================================
Continuously monitors gaze prediction quality and triggers a recalibration
alert when accuracy degrades beyond acceptable thresholds.

Accuracy is estimated by:
  1. Gaze consistency score  – how stable/repeatable the gaze signal is
     when the user appears to be fixating (from AttentionTracker)
  2. Prediction confidence   – variance of model output over a short window
  3. Head pose drift         – detects if user has moved significantly
     since last calibration (yaw/pitch shift)

When accuracy drops:
  - A visual warning is shown in the debug overlay
  - The session logger records the event
  - An optional callback triggers the calibration UI
"""

import time
import numpy as np
from collections import deque
from enum import Enum, auto
from typing import Callable, Optional

# ── Parameters ────────────────────────────────────────────────────────────────
CONSISTENCY_WINDOW   = 5.0    # seconds to measure gaze consistency
CONSISTENCY_THRESHOLD = 0.06  # max acceptable gaze variance during fixation
POSE_DRIFT_THRESHOLD  = 18.0  # degrees – head moved too far from calibration pose
CONFIDENCE_WINDOW    = 3.0    # seconds for prediction variance check
LOW_CONFIDENCE_THRESH = 0.08  # max prediction variance before warning
COOLDOWN_SECONDS     = 120    # min seconds between recalibration alerts


class CalibrationStatus(Enum):
    GOOD       = auto()   # accuracy is acceptable
    DEGRADING  = auto()   # early warning – slight drop
    POOR       = auto()   # recalibration recommended
    CRITICAL   = auto()   # recalibration required


class AutoRecalibrationMonitor:
    """
    Monitors gaze quality and raises alerts when recalibration is needed.
    """

    def __init__(self, on_recalibrate_needed: Optional[Callable] = None):
        """
        Parameters
        ----------
        on_recalibrate_needed : callable – called when status becomes CRITICAL
        """
        self._on_recalibrate = on_recalibrate_needed or (lambda: None)

        # Gaze prediction history during fixations
        self._fixation_gazes: deque = deque()   # (timestamp, x, y)
        self._prediction_history: deque = deque()  # (timestamp, x, y)

        # Reference head pose from calibration (set after first calibration)
        self._ref_yaw:   Optional[float] = None
        self._ref_pitch: Optional[float] = None

        self._last_alert_time = 0.0
        self.status = CalibrationStatus.GOOD

        # Computed metrics
        self.consistency_score = 1.0   # 1 = perfect, 0 = very inconsistent
        self.pose_drift        = 0.0   # degrees from calibration pose
        self.confidence_score  = 1.0   # 1 = high confidence

    # ── Public API ────────────────────────────────────────────────────────────
    def set_calibration_pose(self, yaw: float, pitch: float):
        """
        Call after calibration completes to record the reference head pose.
        """
        self._ref_yaw   = yaw
        self._ref_pitch = pitch
        self.status = CalibrationStatus.GOOD
        print(f"[AutoRecal] Reference pose set: yaw={yaw:.1f} pitch={pitch:.1f}")

    def update(self, gaze_norm: Optional[tuple],
               head_pose: Optional[tuple],
               is_fixating: bool) -> dict:
        """
        Call once per frame.

        Parameters
        ----------
        gaze_norm  : (nx, ny) current predicted gaze
        head_pose  : (yaw, pitch) current head pose or None
        is_fixating: bool – True when AttentionTracker says FOCUSED

        Returns
        -------
        dict:
            'status'           : CalibrationStatus
            'consistency_score': float 0-1
            'pose_drift'       : float degrees
            'confidence_score' : float 0-1
            'needs_recalibration': bool
            'message'          : str
        """
        now = time.time()

        # Record gaze prediction for confidence estimation
        if gaze_norm is not None:
            self._prediction_history.append((now, gaze_norm[0], gaze_norm[1]))

        # During fixation, record for consistency check
        if is_fixating and gaze_norm is not None:
            self._fixation_gazes.append((now, gaze_norm[0], gaze_norm[1]))

        # Prune old data
        self._prune(self._fixation_gazes,    now - CONSISTENCY_WINDOW)
        self._prune(self._prediction_history, now - CONFIDENCE_WINDOW)

        # Compute metrics
        self.consistency_score = self._compute_consistency()
        self.confidence_score  = self._compute_confidence()
        self.pose_drift        = self._compute_pose_drift(head_pose)

        # Classify status
        prev_status = self.status
        self.status = self._classify()

        # Trigger callback if critical and cooldown passed
        needs_recal = False
        if (self.status == CalibrationStatus.CRITICAL
                and (now - self._last_alert_time) > COOLDOWN_SECONDS):
            self._last_alert_time = now
            needs_recal = True
            self._on_recalibrate()

        return {
            "status":             self.status,
            "consistency_score":  round(self.consistency_score, 3),
            "pose_drift":         round(self.pose_drift, 1),
            "confidence_score":   round(self.confidence_score, 3),
            "needs_recalibration": needs_recal,
            "message":            self._get_message(),
        }

    # ── Metric computation ────────────────────────────────────────────────────
    def _compute_consistency(self) -> float:
        """
        During fixation, gaze should be stable.
        Returns 1 - normalised_variance (1 = perfectly consistent).
        """
        if len(self._fixation_gazes) < 5:
            return 1.0
        pts = np.array([[x, y] for _, x, y in self._fixation_gazes])
        variance = float(np.var(pts))
        score = 1.0 - min(variance / CONSISTENCY_THRESHOLD, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_confidence(self) -> float:
        """
        Low variance in predictions = high confidence.
        High variance = model is uncertain.
        """
        if len(self._prediction_history) < 5:
            return 1.0
        pts = np.array([[x, y] for _, x, y in self._prediction_history])
        variance = float(np.var(pts))
        score = 1.0 - min(variance / LOW_CONFIDENCE_THRESH, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_pose_drift(self, head_pose: Optional[tuple]) -> float:
        """Euclidean distance from calibration pose in degrees."""
        if head_pose is None or self._ref_yaw is None:
            return 0.0
        yaw, pitch = head_pose
        drift = np.sqrt((yaw - self._ref_yaw) ** 2 + (pitch - self._ref_pitch) ** 2)
        return float(drift)

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(self) -> CalibrationStatus:
        # Pose drift is most reliable signal
        if self.pose_drift > POSE_DRIFT_THRESHOLD:
            return CalibrationStatus.CRITICAL
        if self.pose_drift > POSE_DRIFT_THRESHOLD * 0.6:
            return CalibrationStatus.POOR

        # Consistency during fixation
        if self.consistency_score < 0.3:
            return CalibrationStatus.CRITICAL
        if self.consistency_score < 0.55:
            return CalibrationStatus.POOR
        if self.consistency_score < 0.75:
            return CalibrationStatus.DEGRADING

        # Prediction confidence
        if self.confidence_score < 0.3:
            return CalibrationStatus.POOR
        if self.confidence_score < 0.6:
            return CalibrationStatus.DEGRADING

        return CalibrationStatus.GOOD

    def _get_message(self) -> str:
        messages = {
            CalibrationStatus.GOOD:      "Calibration accurate.",
            CalibrationStatus.DEGRADING: "Slight accuracy drop detected.",
            CalibrationStatus.POOR:      "Recalibration recommended.",
            CalibrationStatus.CRITICAL:  "Please recalibrate now.",
        }
        return messages[self.status]

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _prune(dq: deque, cutoff: float):
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def reset(self):
        self._fixation_gazes.clear()
        self._prediction_history.clear()
        self.status = CalibrationStatus.GOOD
        self.consistency_score = 1.0
        self.confidence_score  = 1.0
        self.pose_drift        = 0.0
