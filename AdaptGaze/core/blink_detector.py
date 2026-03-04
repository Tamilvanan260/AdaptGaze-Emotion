"""
AdaptGaze - Blink Detector
============================
Detects three blink gestures using Eye Aspect Ratio (EAR):

  SINGLE blink → left click
  DOUBLE blink → right click
  LONG   blink → drag toggle (press-and-hold / release)

State machine approach:
  OPEN → CLOSING → CLOSED → OPENING → OPEN (single/long classified on re-open)
  Double blink detected by two singles within DOUBLE_BLINK_MAX_GAP seconds.
"""

import time
from enum import Enum, auto
from typing import Callable, Optional
from config.settings import (
    EAR_THRESHOLD,
    BLINK_CONSEC_FRAMES,
    DOUBLE_BLINK_MAX_GAP,
    LONG_BLINK_MIN_SECONDS,
)


class BlinkGesture(Enum):
    NONE   = auto()
    SINGLE = auto()
    DOUBLE = auto()
    LONG   = auto()


class BlinkDetector:
    """
    Stateful blink gesture classifier.

    Parameters
    ----------
    on_single : callable  invoked on single blink
    on_double : callable  invoked on double blink
    on_long   : callable  invoked when long blink starts (drag begin)
    on_drag_end: callable invoked when long blink ends (drag release)
    """

    def __init__(
        self,
        on_single: Optional[Callable] = None,
        on_double: Optional[Callable] = None,
        on_long:   Optional[Callable] = None,
        on_drag_end: Optional[Callable] = None,
    ):
        self._on_single   = on_single   or (lambda: None)
        self._on_double   = on_double   or (lambda: None)
        self._on_long     = on_long     or (lambda: None)
        self._on_drag_end = on_drag_end or (lambda: None)

        # EAR frame counters
        self._closed_frames = 0        # consecutive frames below threshold
        self._is_closed     = False    # eye currently closed?

        # Timing
        self._close_start_time: Optional[float] = None
        self._last_blink_time:  Optional[float] = None

        # Drag state
        self._dragging = False

        # Blink history for double-blink detection
        self._pending_single = False
        self._pending_time: Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, left_ear: float, right_ear: float) -> BlinkGesture:
        """
        Call once per frame.

        Parameters
        ----------
        left_ear, right_ear : Eye Aspect Ratio values from FaceMeshDetector

        Returns
        -------
        BlinkGesture – the gesture detected this frame (may be NONE).
        """
        avg_ear = (left_ear + right_ear) / 2.0
        now = time.time()
        gesture = BlinkGesture.NONE

        # ── Eye closing ──────────────────────────────────────────────────
        if avg_ear < EAR_THRESHOLD:
            self._closed_frames += 1
            if not self._is_closed and self._closed_frames >= BLINK_CONSEC_FRAMES:
                # Transition: OPEN → CLOSED
                self._is_closed = True
                self._close_start_time = now
        else:
            # ── Eye opened ───────────────────────────────────────────────
            if self._is_closed:
                closed_duration = now - (self._close_start_time or now)
                self._is_closed = False
                self._closed_frames = 0

                if self._dragging:
                    # Long blink ended → release drag
                    self._dragging = False
                    self._on_drag_end()
                    gesture = BlinkGesture.LONG
                elif closed_duration >= LONG_BLINK_MIN_SECONDS:
                    # New long blink → start drag
                    self._dragging = True
                    self._on_long()
                    gesture = BlinkGesture.LONG
                else:
                    # Short blink – check for double
                    if (self._pending_single
                            and self._pending_time is not None
                            and (now - self._pending_time) <= DOUBLE_BLINK_MAX_GAP):
                        # Double blink confirmed
                        self._pending_single = False
                        self._pending_time   = None
                        self._on_double()
                        gesture = BlinkGesture.DOUBLE
                    else:
                        # Might be start of double – defer single
                        self._pending_single = True
                        self._pending_time   = now
            else:
                self._closed_frames = 0

        # ── Resolve deferred single blink ────────────────────────────────
        if (self._pending_single
                and self._pending_time is not None
                and (now - self._pending_time) > DOUBLE_BLINK_MAX_GAP
                and not self._is_closed):
            self._pending_single = False
            self._pending_time   = None
            self._on_single()
            if gesture == BlinkGesture.NONE:
                gesture = BlinkGesture.SINGLE

        return gesture

    # ------------------------------------------------------------------
    @property
    def is_dragging(self) -> bool:
        return self._dragging

    def reset(self):
        self._closed_frames  = 0
        self._is_closed      = False
        self._close_start_time = None
        self._last_blink_time  = None
        self._dragging       = False
        self._pending_single = False
        self._pending_time   = None
