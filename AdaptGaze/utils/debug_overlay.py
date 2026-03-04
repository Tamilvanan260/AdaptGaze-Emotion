"""
AdaptGaze - Debug Overlay
===========================
Draws real-time diagnostic information on the camera preview frame.

Information shown:
  - Iris landmarks (dots)
  - Left / right EAR values
  - Head yaw / pitch
  - Blink gesture indicator
  - Fatigue level and blink rate
  - Predicted gaze position (crosshair on mini-map)
  - FPS counter
"""

import cv2
import numpy as np
import time
from core.fatigue_monitor import FatigueLevel

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.55
FONT_MED   = 0.7
LINE_H     = 24   # vertical spacing for text rows
TEXT_COLOR = (220, 220, 220)
WARN_COLOR = (0, 180, 255)
ERR_COLOR  = (0, 80, 255)
OK_COLOR   = (80, 255, 80)


class DebugOverlay:
    """Renders diagnostic HUD onto a copy of the camera frame."""

    def __init__(self, frame_width: int, frame_height: int):
        self._fw = frame_width
        self._fh = frame_height
        self._prev_time = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, face_data: dict | None,
             gaze_norm: tuple | None, gesture_label: str,
             fatigue_info: dict, is_dragging: bool) -> np.ndarray:
        """
        Draw HUD on frame and return the annotated copy.

        Parameters
        ----------
        frame        : BGR camera frame
        face_data    : dict from FaceMeshDetector or None
        gaze_norm    : (nx, ny) normalised gaze or None
        gesture_label: string label for last gesture
        fatigue_info : dict from FatigueMonitor.update()
        is_dragging  : bool
        """
        out = frame.copy()
        self._update_fps()

        # Semi-transparent info panel
        overlay = out.copy()
        cv2.rectangle(overlay, (5, 5), (300, 220), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        row = 30
        # FPS
        cv2.putText(out, f"FPS: {self._fps:.1f}", (12, row),
                    FONT, FONT_SMALL, TEXT_COLOR, 1)
        row += LINE_H

        if face_data:
            # EAR
            l_ear = face_data.get("left_ear", 0)
            r_ear = face_data.get("right_ear", 0)
            cv2.putText(out, f"EAR  L:{l_ear:.3f}  R:{r_ear:.3f}",
                        (12, row), FONT, FONT_SMALL, TEXT_COLOR, 1)
            row += LINE_H

            # Head pose
            hp = face_data.get("head_pose")
            if hp:
                cv2.putText(out, f"Yaw:{hp[0]:+.1f}  Pitch:{hp[1]:+.1f}",
                            (12, row), FONT, FONT_SMALL, TEXT_COLOR, 1)
            row += LINE_H

            # Iris dots
            li = face_data.get("left_iris")
            ri = face_data.get("right_iris")
            if li is not None:
                cv2.circle(out, (int(li[0]), int(li[1])), 4, (0, 255, 255), -1)
            if ri is not None:
                cv2.circle(out, (int(ri[0]), int(ri[1])), 4, (0, 255, 255), -1)

        # Gaze coordinate
        if gaze_norm:
            nx, ny = gaze_norm
            cv2.putText(out, f"Gaze ({nx:.2f}, {ny:.2f})",
                        (12, row), FONT, FONT_SMALL, OK_COLOR, 1)
        else:
            cv2.putText(out, "Gaze: no prediction",
                        (12, row), FONT, FONT_SMALL, WARN_COLOR, 1)
        row += LINE_H

        # Gesture
        g_color = WARN_COLOR if gesture_label != "—" else TEXT_COLOR
        cv2.putText(out, f"Gesture: {gesture_label}",
                    (12, row), FONT, FONT_SMALL, g_color, 1)
        row += LINE_H

        # Drag indicator
        if is_dragging:
            cv2.putText(out, "DRAGGING", (12, row), FONT, FONT_MED, (0, 100, 255), 2)
            row += LINE_H

        # Fatigue
        fl = fatigue_info.get("fatigue_level", FatigueLevel.NORMAL)
        br = fatigue_info.get("blink_rate", 0.0)
        fl_color = {
            FatigueLevel.NORMAL:    OK_COLOR,
            FatigueLevel.STRAINING: WARN_COLOR,
            FatigueLevel.DROWSY:    ERR_COLOR,
        }.get(fl, TEXT_COLOR)
        cv2.putText(out, f"Blinks/min: {br:.1f}  [{fl.name}]",
                    (12, row), FONT, FONT_SMALL, fl_color, 1)
        row += LINE_H

        # Session time
        elapsed = fatigue_info.get("elapsed_min", 0.0)
        cv2.putText(out, f"Session: {elapsed:.1f} min",
                    (12, row), FONT, FONT_SMALL, TEXT_COLOR, 1)

        # Gaze mini-map (bottom-right corner)
        if gaze_norm:
            self._draw_minimap(out, gaze_norm)

        return out

    # ------------------------------------------------------------------
    def _draw_minimap(self, out: np.ndarray, gaze_norm: tuple):
        """Draw a small screen-position indicator in the bottom-right."""
        map_w, map_h = 120, 70
        mx = self._fw - map_w - 10
        my = self._fh - map_h - 10
        cv2.rectangle(out, (mx, my), (mx + map_w, my + map_h), (60, 60, 60), -1)
        cv2.rectangle(out, (mx, my), (mx + map_w, my + map_h), (160, 160, 160), 1)
        gx = int(mx + gaze_norm[0] * map_w)
        gy = int(my + gaze_norm[1] * map_h)
        cv2.drawMarker(out, (gx, gy), (0, 255, 100),
                       cv2.MARKER_CROSS, 12, 2)

    # ------------------------------------------------------------------
    def _update_fps(self):
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._prev_time = now
