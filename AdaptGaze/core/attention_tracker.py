"""
AdaptGaze – Attention Tracker
==============================
Analyses gaze patterns in real time to determine whether the user is:

  FOCUSED    – gaze stable, within a region, consistent fixations
  DISTRACTED – gaze jumping rapidly across screen
  AWAY       – no face detected for several seconds
  TRANSITIONING – gaze moving between regions (natural reading/scanning)

Uses a sliding window of gaze positions and computes:
  - Gaze velocity       (how fast gaze is moving)
  - Fixation ratio      (% of time gaze is stable)
  - Dispersion score    (spread of gaze points)
  - Coverage area       (how much of screen is being looked at)
"""

import time
import numpy as np
from collections import deque
from enum import Enum, auto
from typing import Optional

# ── Parameters ──────────────────────────────────────────────────────────────
WINDOW_SECONDS       = 3.0    # rolling window for analysis
VELOCITY_THRESHOLD   = 0.04   # normalised units/sec – above = saccade
FIXATION_THRESHOLD   = 0.02   # gaze dispersion radius for fixation
MIN_FIXATION_RATIO   = 0.55   # fraction of window that must be fixated
AWAY_TIMEOUT_SECONDS = 2.5    # seconds without face → AWAY


class AttentionState(Enum):
    FOCUSED       = auto()
    DISTRACTED    = auto()
    TRANSITIONING = auto()
    AWAY          = auto()


class AttentionTracker:
    """
    Real-time attention state estimator based on gaze movement patterns.
    """

    def __init__(self):
        # Circular buffer: (timestamp, norm_x, norm_y)
        self._gaze_history: deque = deque()
        self._last_face_time: float = time.time()
        self._state = AttentionState.AWAY

        # Metrics (updated each call)
        self.velocity        = 0.0
        self.fixation_ratio  = 0.0
        self.dispersion      = 0.0
        self.coverage        = 0.0

        # Fixation cluster centre (where user is currently looking)
        self.fixation_centre: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────────────
    def update(self, gaze_norm: Optional[tuple], face_detected: bool) -> dict:
        """
        Call once per frame.

        Parameters
        ----------
        gaze_norm    : (nx, ny) normalised gaze in [0,1] or None
        face_detected: bool – whether a face was found this frame

        Returns
        -------
        dict with keys:
            'state'         : AttentionState
            'velocity'      : float – current gaze speed
            'fixation_ratio': float – fraction of time gaze is stable
            'dispersion'    : float – spread of recent gaze points
            'coverage'      : float – screen area covered (0-1)
            'focus_score'   : float – 0 (distracted) to 1 (focused)
        """
        now = time.time()

        # ── Handle face absence ───────────────────────────────────────────
        if not face_detected or gaze_norm is None:
            if (now - self._last_face_time) > AWAY_TIMEOUT_SECONDS:
                self._state = AttentionState.AWAY
            return self._result()

        self._last_face_time = now

        # ── Record gaze point ─────────────────────────────────────────────
        self._gaze_history.append((now, gaze_norm[0], gaze_norm[1]))

        # Prune old entries outside the rolling window
        cutoff = now - WINDOW_SECONDS
        while self._gaze_history and self._gaze_history[0][0] < cutoff:
            self._gaze_history.popleft()

        if len(self._gaze_history) < 5:
            self._state = AttentionState.TRANSITIONING
            return self._result()

        # ── Compute metrics ───────────────────────────────────────────────
        pts = np.array([[x, y] for _, x, y in self._gaze_history],
                       dtype=np.float32)
        times = np.array([t for t, _, _ in self._gaze_history])

        self.velocity       = self._compute_velocity(pts, times)
        self.fixation_ratio = self._compute_fixation_ratio(pts)
        self.dispersion     = self._compute_dispersion(pts)
        self.coverage       = self._compute_coverage(pts)

        # ── Classify attention state ──────────────────────────────────────
        self._state = self._classify()

        return self._result()

    # ── State classification ─────────────────────────────────────────────────
    def _classify(self) -> AttentionState:
        if self.fixation_ratio >= MIN_FIXATION_RATIO and self.velocity < VELOCITY_THRESHOLD:
            return AttentionState.FOCUSED
        elif self.velocity > VELOCITY_THRESHOLD * 2.5 or self.coverage > 0.45:
            return AttentionState.DISTRACTED
        else:
            return AttentionState.TRANSITIONING

    # ── Metric computations ──────────────────────────────────────────────────
    @staticmethod
    def _compute_velocity(pts: np.ndarray, times: np.ndarray) -> float:
        """Mean gaze speed in normalised units/second."""
        if len(pts) < 2:
            return 0.0
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dts   = np.diff(times)
        dts   = np.where(dts < 1e-6, 1e-6, dts)
        speeds = dists / dts
        return float(np.mean(speeds))

    @staticmethod
    def _compute_fixation_ratio(pts: np.ndarray) -> float:
        """
        Fraction of consecutive point-pairs where gaze moved less than
        FIXATION_THRESHOLD (i.e., gaze was stable = fixating).
        """
        if len(pts) < 2:
            return 0.0
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        fixated = (dists < FIXATION_THRESHOLD).sum()
        return float(fixated / len(dists))

    @staticmethod
    def _compute_dispersion(pts: np.ndarray) -> float:
        """Standard deviation of gaze positions (spread)."""
        return float(np.std(pts))

    @staticmethod
    def _compute_coverage(pts: np.ndarray) -> float:
        """
        Fraction of screen area covered by convex hull of gaze points.
        Approximated as bounding-box area.
        """
        if len(pts) < 3:
            return 0.0
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        return float(min(x_range * y_range, 1.0))

    # ── Result builder ───────────────────────────────────────────────────────
    def _result(self) -> dict:
        focus_score = self._compute_focus_score()
        return {
            "state":          self._state,
            "velocity":       round(self.velocity, 4),
            "fixation_ratio": round(self.fixation_ratio, 3),
            "dispersion":     round(self.dispersion, 4),
            "coverage":       round(self.coverage, 4),
            "focus_score":    round(focus_score, 3),
        }

    def _compute_focus_score(self) -> float:
        """
        Composite focus score in [0, 1].
        Higher = more focused.
        """
        if self._state == AttentionState.AWAY:
            return 0.0
        # Weighted combination of fixation ratio and low velocity
        vel_score = max(0.0, 1.0 - self.velocity / (VELOCITY_THRESHOLD * 3))
        score = 0.6 * self.fixation_ratio + 0.4 * vel_score
        return float(np.clip(score, 0.0, 1.0))

    @property
    def state(self) -> AttentionState:
        return self._state

    def reset(self):
        self._gaze_history.clear()
        self._last_face_time = time.time()
        self._state = AttentionState.AWAY
        self.velocity = 0.0
        self.fixation_ratio = 0.0
        self.dispersion = 0.0
        self.coverage = 0.0
