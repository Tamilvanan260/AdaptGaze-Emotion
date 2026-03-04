"""
AdaptGaze – Enhanced Debug Overlay
=====================================
Extended HUD that displays all AI Intelligence metrics:
  - Original: FPS, EAR, head pose, gesture, blink rate
  - NEW: Attention state + focus score
  - NEW: Smart fatigue risk level + prediction
  - NEW: Calibration status + pose drift
  - NEW: Active user profile name
  - NEW: Gaze heatmap trail (last N positions)
"""

import cv2
import numpy as np
import time
from collections import deque

from core.fatigue_monitor import FatigueLevel
from core.attention_tracker import AttentionState
from core.smart_fatigue import FatigueRisk
from core.auto_recalibration import CalibrationStatus
from core.emotion_detector import Emotion

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.52
FONT_MED   = 0.68
LINE_H     = 23

# Colour palette
C_WHITE  = (220, 220, 220)
C_GREEN  = (80,  255, 80)
C_YELLOW = (0,   220, 255)
C_ORANGE = (0,   165, 255)
C_RED    = (0,   80,  255)
C_CYAN   = (255, 220, 0)
C_PURPLE = (255, 80,  200)

# Gaze trail
TRAIL_LEN = 40


class EnhancedOverlay:
    """Renders the full AI-enhanced diagnostic HUD onto a camera frame."""

    def __init__(self, frame_width: int, frame_height: int):
        self._fw = frame_width
        self._fh = frame_height
        self._prev_time = time.time()
        self._fps = 0.0
        self._gaze_trail: deque = deque(maxlen=TRAIL_LEN)

    # ── Main draw method ──────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray,
             face_data,
             gaze_norm,
             gesture_label: str,
             fatigue_info: dict,
             is_dragging: bool,
             attention_info: dict,
             smart_fatigue_info: dict,
             recal_info: dict,
             emotion_info: dict,
             active_profile: str = "default") -> np.ndarray:

        out = frame.copy()
        self._update_fps()

        # ── Gaze trail ────────────────────────────────────────────────────
        if gaze_norm:
            self._gaze_trail.append(gaze_norm)
        self._draw_gaze_trail(out)

        # ── Emotion banner at top of frame ────────────────────────────────
        self._draw_emotion_banner(out, emotion_info)

        # ── Left panel (original metrics) ─────────────────────────────────
        self._draw_panel(out, x=5, y=55, w=295, h=340)
        row = 78
        self._text(out, f"FPS: {self._fps:.1f}   Profile: {active_profile}",
                   12, row, C_WHITE)
        row += LINE_H

        if face_data:
            l_ear = face_data.get("left_ear", 0)
            r_ear = face_data.get("right_ear", 0)
            self._text(out, f"EAR  L:{l_ear:.3f}  R:{r_ear:.3f}", 12, row, C_WHITE)
            row += LINE_H
            hp = face_data.get("head_pose")
            if hp:
                self._text(out, f"Yaw:{hp[0]:+.1f}  Pitch:{hp[1]:+.1f}",
                           12, row, C_WHITE)
            row += LINE_H
            li = face_data.get("left_iris")
            ri = face_data.get("right_iris")
            if li is not None:
                cv2.circle(out, (int(li[0]), int(li[1])), 4, C_CYAN, -1)
            if ri is not None:
                cv2.circle(out, (int(ri[0]), int(ri[1])), 4, C_CYAN, -1)

        if gaze_norm:
            self._text(out, f"Gaze ({gaze_norm[0]:.2f}, {gaze_norm[1]:.2f})",
                       12, row, C_GREEN)
        else:
            self._text(out, "Gaze: no prediction", 12, row, C_YELLOW)
        row += LINE_H

        g_color = C_YELLOW if gesture_label != "—" else C_WHITE
        self._text(out, f"Gesture: {gesture_label}", 12, row, g_color)
        row += LINE_H

        if is_dragging:
            self._text(out, "DRAGGING", 12, row, C_ORANGE)
            row += LINE_H

        # Blink fatigue
        fl     = fatigue_info.get("fatigue_level", FatigueLevel.NORMAL)
        br     = fatigue_info.get("blink_rate", 0.0)
        fl_col = {FatigueLevel.NORMAL: C_GREEN,
                  FatigueLevel.STRAINING: C_YELLOW,
                  FatigueLevel.DROWSY:    C_RED}.get(fl, C_WHITE)
        self._text(out, f"Blinks/min: {br:.1f}  [{fl.name}]", 12, row, fl_col)
        row += LINE_H

        elapsed = fatigue_info.get("elapsed_min", 0.0)
        self._text(out, f"Session: {elapsed:.1f} min", 12, row, C_WHITE)
        row += LINE_H + 6

        # ── Divider ───────────────────────────────────────────────────────
        cv2.line(out, (12, row), (288, row), (80, 80, 80), 1)
        row += 10

        # ── AI Metrics section ────────────────────────────────────────────
        self._text(out, "── AI Intelligence ──", 12, row, C_PURPLE)
        row += LINE_H

        # Attention
        att_state = attention_info.get("state", AttentionState.AWAY)
        focus_sc  = attention_info.get("focus_score", 0.0)
        att_col   = {AttentionState.FOCUSED:       C_GREEN,
                     AttentionState.TRANSITIONING:  C_YELLOW,
                     AttentionState.DISTRACTED:     C_ORANGE,
                     AttentionState.AWAY:           C_RED}.get(att_state, C_WHITE)
        self._text(out, f"Attention: {att_state.name}  ({focus_sc:.2f})",
                   12, row, att_col)
        row += LINE_H

        # Smart fatigue
        risk    = smart_fatigue_info.get("risk_level", FatigueRisk.LOW)
        fat_sc  = smart_fatigue_info.get("predicted_fatigue", 0.0)
        risk_col = {FatigueRisk.LOW:      C_GREEN,
                    FatigueRisk.MODERATE: C_YELLOW,
                    FatigueRisk.HIGH:     C_ORANGE,
                    FatigueRisk.CRITICAL: C_RED}.get(risk, C_WHITE)
        self._text(out, f"Fatigue Risk: {risk.name}  ({fat_sc:.2f})",
                   12, row, risk_col)
        row += LINE_H

        # Draw fatigue bar
        self._draw_bar(out, 12, row, 276, 12, fat_sc, risk_col)
        row += 20

        # Calibration status
        cal_status  = recal_info.get("status", CalibrationStatus.GOOD)
        pose_drift  = recal_info.get("pose_drift", 0.0)
        cal_col     = {CalibrationStatus.GOOD:      C_GREEN,
                       CalibrationStatus.DEGRADING:  C_YELLOW,
                       CalibrationStatus.POOR:       C_ORANGE,
                       CalibrationStatus.CRITICAL:   C_RED}.get(cal_status, C_WHITE)
        self._text(out, f"Calibration: {cal_status.name}  drift:{pose_drift:.1f}°",
                   12, row, cal_col)
        row += LINE_H

        # Recalibration warning banner
        if cal_status in (CalibrationStatus.POOR, CalibrationStatus.CRITICAL):
            self._draw_warning_banner(out, recal_info.get("message", ""))

        # Smart fatigue recommendation
        if risk in (FatigueRisk.HIGH, FatigueRisk.CRITICAL):
            rec = smart_fatigue_info.get("recommendation", "")
            self._draw_recommendation(out, rec)

        # ── Mini gaze map ─────────────────────────────────────────────────
        if gaze_norm:
            self._draw_minimap(out, gaze_norm)

        return out

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _draw_emotion_banner(self, out: np.ndarray, emotion_info: dict):
        """Draw a prominent emotion banner at the top of the frame."""
        emoji_label  = emotion_info.get("emoji_label",  "😐 NEUTRAL")
        color        = emotion_info.get("color",         C_WHITE)
        speed_factor = emotion_info.get("speed_factor",  1.0)
        rest_alert   = emotion_info.get("rest_alert",    False)
        mouth_curve  = emotion_info.get("mouth_curve",   0.0)

        # Banner background
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], 50), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # Emotion label
        cv2.putText(out, f"Emotion: {emoji_label}",
                    (12, 32), FONT, FONT_MED, color, 2)

        # Speed indicator
        speed_text = f"Speed: {int(speed_factor * 100)}%"
        speed_col  = C_GREEN if speed_factor == 1.0 else C_ORANGE
        cv2.putText(out, speed_text,
                    (out.shape[1] - 160, 32), FONT, FONT_SMALL, speed_col, 1)

        # Sad rest alert flash
        if rest_alert:
            h, w = out.shape[:2]
            overlay2 = out.copy()
            cv2.rectangle(overlay2, (0, h - 90), (w, h - 55), (0, 0, 150), -1)
            cv2.addWeighted(overlay2, 0.8, out, 0.2, 0, out)
            cv2.putText(out, "SAD detected - Please take a break and rest your eyes",
                        (10, h - 65), FONT, 0.52, (255, 255, 255), 1)
    def _draw_gaze_trail(self, out: np.ndarray):
        """Draw fading trail of recent gaze positions."""
        trail = list(self._gaze_trail)
        for i, (nx, ny) in enumerate(trail):
            px = int(nx * self._fw)
            py = int(ny * self._fh)
            alpha = int(255 * (i / len(trail)))
            radius = 2 + int(3 * (i / len(trail)))
            color = (0, alpha, 255 - alpha)
            cv2.circle(out, (px, py), radius, color, -1)

    def _draw_panel(self, out, x, y, w, h):
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    def _draw_minimap(self, out: np.ndarray, gaze_norm: tuple):
        map_w, map_h = 130, 80
        mx = self._fw - map_w - 10
        my = self._fh - map_h - 10
        cv2.rectangle(out, (mx, my), (mx + map_w, my + map_h), (50, 50, 50), -1)
        cv2.rectangle(out, (mx, my), (mx + map_w, my + map_h), (150, 150, 150), 1)
        gx = int(mx + gaze_norm[0] * map_w)
        gy = int(my + gaze_norm[1] * map_h)
        cv2.drawMarker(out, (gx, gy), C_GREEN, cv2.MARKER_CROSS, 14, 2)
        self._text(out, "GAZE MAP", mx + 3, my - 5, C_WHITE, scale=0.4)

    def _draw_bar(self, out, x, y, w, h, value, color):
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 60, 60), -1)
        bar_w = int(w * np.clip(value, 0, 1))
        if bar_w > 0:
            cv2.rectangle(out, (x, y), (x + bar_w, y + h), color, -1)
        cv2.rectangle(out, (x, y), (x + w, y + h), (140, 140, 140), 1)

    def _draw_warning_banner(self, out, message: str):
        h, w = out.shape[:2]
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - 55), (w, h - 30), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.putText(out, f"⚠ {message}", (10, h - 37),
                    FONT, 0.55, (255, 255, 255), 1)

    def _draw_recommendation(self, out, message: str):
        h, w = out.shape[:2]
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - 85), (w, h - 60), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.putText(out, message, (10, h - 67),
                    FONT, 0.48, (200, 255, 200), 1)

    @staticmethod
    def _text(out, text, x, y, color, scale=FONT_SMALL, thickness=1):
        cv2.putText(out, text, (x, y), FONT, scale, color, thickness)

    def _update_fps(self):
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._prev_time = now
