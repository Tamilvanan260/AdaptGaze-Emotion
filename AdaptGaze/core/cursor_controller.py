"""
AdaptGaze - Cursor Controller
================================
Translates normalised gaze coordinates (0..1) to screen pixel positions
and dispatches PyAutoGUI mouse events.

Applies:
  1. Kalman filter smoothing (handled upstream in GazeController)
  2. Exponential moving average (EMA) for additional smoothness
  3. Deadzone – ignore tiny movements to prevent micro-jitter
"""

import pyautogui
import numpy as np
from config.settings import CURSOR_SMOOTHING, CURSOR_DEADZONE

# Prevent PyAutoGUI from moving mouse to corner and raising FailSafeException
# when cursor approaches screen edge (set False for production, True for dev)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0          # remove built-in pause between calls


class CursorController:
    """Maps gaze positions to screen coordinates and controls the mouse."""

    def __init__(self):
        self._screen_w, self._screen_h = pyautogui.size()
        self._last_x: float = self._screen_w / 2
        self._last_y: float = self._screen_h / 2
        self._dragging = False

    # ------------------------------------------------------------------
    def move(self, norm_x: float, norm_y: float, speed_factor: float = 1.0):
        """
        Move the cursor to the screen position corresponding to (norm_x, norm_y).

        Parameters
        ----------
        norm_x, norm_y : float in [0, 1]
        speed_factor   : float – emotion-based speed multiplier (0.5 = slow, 1.0 = normal)
        """
        # Clamp to valid range
        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        # Convert to pixel coordinates
        target_x = norm_x * self._screen_w
        target_y = norm_y * self._screen_h

        # Apply emotion speed factor – slow emotions reduce EMA step size
        effective_smoothing = CURSOR_SMOOTHING * float(np.clip(speed_factor, 0.1, 1.0))

        # EMA smoothing
        smooth_x = self._last_x + effective_smoothing * (target_x - self._last_x)
        smooth_y = self._last_y + effective_smoothing * (target_y - self._last_y)

        # Deadzone – skip tiny movements
        dx = abs(smooth_x - self._last_x)
        dy = abs(smooth_y - self._last_y)
        if dx < CURSOR_DEADZONE and dy < CURSOR_DEADZONE:
            return

        self._last_x = smooth_x
        self._last_y = smooth_y

        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
        except Exception:
            pass  # silently skip if move fails (e.g. permission error)

    # ------------------------------------------------------------------
    def left_click(self):
        try:
            pyautogui.click(_pause=False)
        except Exception:
            pass

    def right_click(self):
        try:
            pyautogui.rightClick(_pause=False)
        except Exception:
            pass

    def drag_start(self):
        """Begin a drag at the current cursor position."""
        if not self._dragging:
            self._dragging = True
            try:
                pyautogui.mouseDown(_pause=False)
            except Exception:
                pass

    def drag_end(self):
        """Release the drag."""
        if self._dragging:
            self._dragging = False
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass

    @property
    def screen_size(self):
        return self._screen_w, self._screen_h

    @property
    def is_dragging(self) -> bool:
        return self._dragging
