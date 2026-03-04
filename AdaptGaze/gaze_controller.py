"""
AdaptGaze – Enhanced Gaze Controller (AI Upgraded)
====================================================
Orchestrates all subsystems including new AI Intelligence modules:

Original:
  - FaceMeshDetector, FeatureExtractor, GazePredictor
  - KalmanFilter2D, CursorController
  - BlinkDetector, FatigueMonitor
  - DebugOverlay, SessionLogger

NEW AI Modules:
  - AttentionTracker    → real-time focus/distraction detection
  - SmartFatigueAI      → predictive fatigue with trend analysis
  - AutoRecalibration   → detects and alerts accuracy degradation
  - ProfileManager      → multi-user model management
  - EnhancedOverlay     → full AI metrics HUD

Press 'q' or ESC to quit.
Press 'r' to manually trigger recalibration.
Press 'p' to switch user profile.
"""

import os
import sys
import cv2
import time
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    GAZE_MODEL_PATH, SHOW_DEBUG_WINDOW, DEBUG_WINDOW_SCALE,
    EAR_THRESHOLD,
)

# Core modules
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from core.gaze_model import GazePredictor
from core.kalman_filter import KalmanFilter2D
from core.cursor_controller import CursorController
from core.blink_detector import BlinkDetector, BlinkGesture
from core.fatigue_monitor import FatigueMonitor, FatigueLevel

# AI Intelligence modules
from core.attention_tracker import AttentionTracker, AttentionState
from core.smart_fatigue import SmartFatigueAI, FatigueRisk
from core.auto_recalibration import AutoRecalibrationMonitor, CalibrationStatus
from core.user_profile import ProfileManager
from core.emotion_detector import EmotionDetector, Emotion

# Utils
from utils.enhanced_overlay import EnhancedOverlay
from utils.session_logger import SessionLogger


class GazeController:
    """Top-level controller – gaze tracking + full AI intelligence pipeline."""

    def __init__(self, profile_name: str = "default"):
        print("[GazeController] Initialising AI-enhanced subsystems...")
        self._logger = SessionLogger()
        self._logger.log("profile", f"loading={profile_name}")

        # Camera
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera. Check CAMERA_INDEX in settings.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera.")
        h, w = frame.shape[:2]

        # Core modules
        self._detector  = FaceMeshDetector()
        self._extractor = FeatureExtractor(w, h)
        self._predictor = GazePredictor()
        self._kalman    = KalmanFilter2D()
        self._cursor    = CursorController()
        self._fatigue   = FatigueMonitor()

        # AI modules
        self._attention = AttentionTracker()
        self._smart_fat = SmartFatigueAI()
        self._recal_mon = AutoRecalibrationMonitor(
            on_recalibrate_needed=self._on_recalibration_needed
        )
        self._profiles  = ProfileManager()

        # Overlay
        self._overlay = EnhancedOverlay(w, h)

        # Blink close time for duration tracking
        self._blink_close_time = 0.0

        # Blink detector with callbacks
        self._blink = BlinkDetector(
            on_single=self._on_single_blink,
            on_double=self._on_double_blink,
            on_long=self._on_long_blink,
            on_drag_end=self._on_drag_end,
        )

        # Load model
        self._active_profile = profile_name
        loaded = False
        if self._profiles.profile_exists(profile_name):
            model = self._profiles.load_profile(profile_name)
            if model is not None:
                self._predictor.model = model
                self._predictor._trained = True
                loaded = True
                self._logger.log("model_loaded", f"profile={profile_name}")

        if not loaded:
            if self._predictor.load(GAZE_MODEL_PATH):
                self._logger.log("model_loaded", "default_path")
            else:
                print("[GazeController] WARNING: No trained model found.")
                print("  -> Run: python main.py --calibrate")
                self._logger.log("warning", "no_model_found")

        # State variables
        self._last_gesture_label = "-"
        self._last_gaze_norm     = None
        self._recal_needed       = False
        self._running            = False

        # Default AI info dicts
        self._attention_info = {
            "state": AttentionState.AWAY, "focus_score": 0.0,
            "velocity": 0.0, "fixation_ratio": 0.0,
            "dispersion": 0.0, "coverage": 0.0
        }
        self._smart_fat_info = {
            "risk_level": FatigueRisk.LOW,
            "predicted_fatigue": 0.0, "recommendation": ""
        }
        self._recal_info = {
            "status": CalibrationStatus.GOOD,
            "pose_drift": 0.0, "message": ""
        }

        self._emotion_detector = EmotionDetector()
        self._emotion_info = {
            "emotion": Emotion.NEUTRAL,
            "emoji_label": "😐 NEUTRAL",
            "speed_factor": 1.0,
            "color": (200, 200, 200),
            "rest_alert": False,
            "mouth_curve": 0.0,
            "brow_ratio": 0.0,
        }

        print(f"[GazeController] Ready. Profile: '{self._active_profile}'")
        print("  Keys: Q/ESC=quit  R=recalibrate  P=switch profile")

    # Blink callbacks
    def _on_single_blink(self):
        self._cursor.left_click()
        self._last_gesture_label = "SINGLE (left click)"
        self._logger.log("blink", "single_left_click")
        self._fatigue.record_blink()
        self._smart_fat.record_blink(time.time() - self._blink_close_time)

    def _on_double_blink(self):
        self._cursor.right_click()
        self._last_gesture_label = "DOUBLE (right click)"
        self._logger.log("blink", "double_right_click")
        self._fatigue.record_blink()
        self._smart_fat.record_blink(time.time() - self._blink_close_time)

    def _on_long_blink(self):
        self._cursor.drag_start()
        self._last_gesture_label = "LONG (drag start)"
        self._logger.log("blink", "long_drag_start")
        self._fatigue.record_blink()
        self._smart_fat.record_blink(time.time() - self._blink_close_time)

    def _on_drag_end(self):
        self._cursor.drag_end()
        self._last_gesture_label = "LONG (drag end)"
        self._logger.log("blink", "long_drag_end")

    def _on_recalibration_needed(self):
        self._recal_needed = True
        self._logger.log("auto_recal", "recalibration_needed")
        print("[AutoRecal] Recalibration recommended! Press R to recalibrate.")

    # Main loop
    def run(self):
        self._running = True
        win_name = "AdaptGaze - AI Enhanced"
        prev_fatigue_level = None
        prev_attention     = None
        prev_risk          = None
        frame_count        = 0

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    print("[GazeController] Frame read failed - exiting.")
                    break

                frame = cv2.flip(frame, 1)
                frame_count += 1

                # Face detection
                face_data = self._detector.process(frame)
                feat = self._extractor.extract(face_data)
                face_detected = face_data is not None

                # Gaze prediction
                gaze_norm = None
                if feat is not None and self._predictor.is_trained:
                    raw = self._predictor.predict(feat)
                    kx, ky = self._kalman.update(float(raw[0]), float(raw[1]))
                    gaze_norm = (kx, ky)
                    self._cursor.move(kx, ky,
                        speed_factor=self._emotion_info.get("speed_factor", 1.0))
                    self._last_gaze_norm = gaze_norm

                # Blink detection + EAR tracking
                if face_data:
                    l_ear = face_data["left_ear"]
                    r_ear = face_data["right_ear"]
                    self._blink.update(l_ear, r_ear)
                    avg_ear = (l_ear + r_ear) / 2.0
                    if avg_ear < EAR_THRESHOLD:
                        if self._blink_close_time == 0.0:
                            self._blink_close_time = time.time()
                    else:
                        self._blink_close_time = 0.0
                    self._smart_fat.record_ear(l_ear, r_ear)

                # AI: Emotion detection (every 3 frames)
                if frame_count % 3 == 0 and face_data:
                    lm = face_data.get("landmarks")
                    if lm is not None:
                        prev_emotion = self._emotion_info.get("emotion", Emotion.NEUTRAL)
                        self._emotion_info = self._emotion_detector.update(lm)
                        new_emotion = self._emotion_info["emotion"]
                        if new_emotion != prev_emotion:
                            self._logger.log("emotion",
                                f"{new_emotion.name} speed={self._emotion_info['speed_factor']}")
                        if self._emotion_info.get("rest_alert"):
                            self._logger.log("emotion_alert", "SAD_rest_reminder")
                            print(f"[Emotion] SAD detected – rest reminder triggered.")

                # AI: Attention tracking (every 2 frames)
                is_fixating = False
                if frame_count % 2 == 0:
                    self._attention_info = self._attention.update(
                        gaze_norm, face_detected
                    )
                    is_fixating = (
                        self._attention_info["state"] == AttentionState.FOCUSED
                    )
                    att_state = self._attention_info["state"]
                    if att_state != prev_attention:
                        self._logger.log("attention", att_state.name)
                        prev_attention = att_state
                    self._smart_fat.record_gaze_stability(
                        self._attention_info.get("dispersion", 0.0)
                    )

                # AI: Smart fatigue (every 15 frames)
                if frame_count % 15 == 0:
                    self._smart_fat_info = self._smart_fat.update()
                    risk = self._smart_fat_info["risk_level"]
                    if risk != prev_risk:
                        self._logger.log("smart_fatigue",
                            f"{risk.name} score={self._smart_fat_info['predicted_fatigue']:.3f}")
                        prev_risk = risk

                # AI: Auto-recalibration monitor (every 10 frames)
                if frame_count % 10 == 0:
                    hp = face_data.get("head_pose") if face_data else None
                    self._recal_info = self._recal_mon.update(
                        gaze_norm, hp, is_fixating
                    )

                # Standard fatigue monitor
                fatigue_info = self._fatigue.update()
                fl = fatigue_info["fatigue_level"]
                if fl != prev_fatigue_level:
                    self._logger.log("fatigue", fl.name)
                    prev_fatigue_level = fl
                if fatigue_info["rest_reminder"]:
                    self._logger.log("rest_reminder",
                        f"elapsed={fatigue_info['elapsed_min']}min")

                # Debug display
                if SHOW_DEBUG_WINDOW:
                    annotated = self._overlay.draw(
                        frame, face_data, gaze_norm,
                        self._last_gesture_label, fatigue_info,
                        self._cursor.is_dragging,
                        self._attention_info,
                        self._smart_fat_info,
                        self._recal_info,
                        self._emotion_info,
                        active_profile=self._active_profile,
                    )
                    if DEBUG_WINDOW_SCALE != 1.0:
                        dh, dw = annotated.shape[:2]
                        annotated = cv2.resize(
                            annotated,
                            (int(dw * DEBUG_WINDOW_SCALE),
                             int(dh * DEBUG_WINDOW_SCALE))
                        )
                    cv2.imshow(win_name, annotated)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    print("[GazeController] Manual recalibration triggered.")
                    self._logger.log("recal", "manual_triggered")
                    self._trigger_recalibration()
                elif key == ord("p"):
                    self._prompt_profile_switch()

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _trigger_recalibration(self):
        self._cap.release()
        cv2.destroyAllWindows()
        self._detector.close()

        from calibration.calibration import CalibrationSystem
        cal = CalibrationSystem()
        success = cal.run()

        if success:
            self._predictor.load(GAZE_MODEL_PATH)
            self._recal_mon.reset()
            self._kalman.reset()
            self._logger.log("recal", "completed")
            print("[GazeController] Recalibration complete.")
        else:
            print("[GazeController] Recalibration aborted.")

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._detector = FaceMeshDetector()

    def _prompt_profile_switch(self):
        profiles = self._profiles.list_profiles()
        if not profiles:
            print("[Profile] No saved profiles found.")
            return
        print("\n[Profile] Available profiles:")
        for i, p in enumerate(profiles):
            print(f"  {i+1}. {p['username']}  (last used: {p.get('last_used','?')})")
        name = input("Enter profile name to load (or Enter to cancel): ").strip()
        if name and self._profiles.profile_exists(name):
            model = self._profiles.load_profile(name)
            if model:
                self._predictor.model = model
                self._predictor._trained = True
                self._active_profile = name
                self._kalman.reset()
                self._logger.log("profile_switch", f"to={name}")
                print(f"[Profile] Switched to '{name}'")
        elif name:
            print(f"[Profile] Profile '{name}' not found.")

    def _shutdown(self):
        print("[GazeController] Shutting down...")
        self._cursor.drag_end()
        self._cap.release()
        cv2.destroyAllWindows()
        self._detector.close()
        self._logger.close()
        print("[GazeController] Session ended.")
