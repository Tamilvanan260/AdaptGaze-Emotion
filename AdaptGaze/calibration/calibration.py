"""
AdaptGaze - 9-Point Calibration System
=========================================
Displays calibration targets on a full-screen OpenCV window.
The user looks at each dot while the system collects gaze feature vectors.
After all points, a gaze regression model is trained and saved.

Calibration grid (3×3):

  TL   TC   TR
  ML   MC   MR
  BL   BC   BR
"""

import os
import time
import cv2
import numpy as np
from typing import List, Tuple

from config.settings import (
    CALIBRATION_POINTS,
    SAMPLES_PER_POINT,
    CALIBRATION_WAIT_SECONDS,
    CALIBRATION_DATA_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_INDEX,
)
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from core.gaze_model import GazePredictor


# ── Calibration point positions (normalised, 3×3 grid) ──────────────────────
GRID = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

# Display parameters
DOT_RADIUS = 18
DOT_COLOR_IDLE   = (255, 255, 255)
DOT_COLOR_ACTIVE = (0, 255, 100)
DOT_COLOR_DONE   = (80, 80, 80)
BG_COLOR = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class CalibrationSystem:
    """Runs the full 9-point calibration pipeline."""

    def __init__(self):
        self._cap = None
        self._detector = FaceMeshDetector()
        self._extractor = None
        self._predictor = GazePredictor()

        # Storage for collected samples
        self._features: List[np.ndarray] = []
        self._targets:  List[np.ndarray] = []

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """
        Execute calibration.

        Returns True if calibration completed and model was trained,
        False if user aborted or no camera available.
        """
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            print("[Calibration] ERROR: Could not open camera.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Get actual resolution for extractor
        ret, frame = self._cap.read()
        if not ret:
            print("[Calibration] ERROR: Could not read frame.")
            return False
        h, w = frame.shape[:2]
        self._extractor = FeatureExtractor(w, h)

        # Full-screen calibration window
        screen_w, screen_h = self._get_screen_size()
        win_name = "AdaptGaze Calibration"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        success = False
        try:
            success = self._calibration_loop(win_name, screen_w, screen_h)
        finally:
            cv2.destroyWindow(win_name)
            self._cap.release()
            self._detector.close()

        if success and len(self._features) > 0:
            self._train_and_save()
            return True
        return False

    # ------------------------------------------------------------------
    def _calibration_loop(self, win_name: str, sw: int, sh: int) -> bool:
        """Main calibration interaction loop. Returns True on completion."""

        # ── Introduction screen ──────────────────────────────────────────
        self._show_intro(win_name, sw, sh)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            return False

        # ── Iterate over calibration points ─────────────────────────────
        for pt_idx, (nx, ny) in enumerate(GRID):
            px = int(nx * sw)
            py = int(ny * sh)

            # Countdown before collecting
            deadline = time.time() + CALIBRATION_WAIT_SECONDS
            while time.time() < deadline:
                ret, frame = self._cap.read()
                if not ret:
                    continue
                canvas = self._make_canvas(sw, sh, GRID, pt_idx, state="waiting")
                remaining = deadline - time.time()
                msg = f"Look at the dot – starting in {remaining:.1f}s"
                cv2.putText(canvas, msg, (sw // 2 - 220, sh - 40),
                            FONT, 0.8, (200, 200, 200), 2)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return False

            # Collect SAMPLES_PER_POINT frames
            collected = 0
            while collected < SAMPLES_PER_POINT:
                ret, frame = self._cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                face_data = self._detector.process(frame)
                feat = self._extractor.extract(face_data)
                if feat is not None:
                    self._features.append(feat)
                    self._targets.append(np.array([nx, ny], dtype=np.float32))
                    collected += 1

                # Draw progress
                canvas = self._make_canvas(sw, sh, GRID, pt_idx, state="collecting")
                progress_pct = collected / SAMPLES_PER_POINT
                bar_w = int(300 * progress_pct)
                cv2.rectangle(canvas, (sw // 2 - 150, sh - 60),
                              (sw // 2 - 150 + bar_w, sh - 40), DOT_COLOR_ACTIVE, -1)
                cv2.rectangle(canvas, (sw // 2 - 150, sh - 60),
                              (sw // 2 + 150, sh - 40), (180, 180, 180), 2)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return False

            print(f"[Calibration] Point {pt_idx + 1}/{len(GRID)} done "
                  f"({len(self._features)} total samples)")

        # ── Training screen ──────────────────────────────────────────────
        canvas = np.full((sh, sw, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(canvas, "Training personalised model – please wait…",
                    (sw // 2 - 320, sh // 2), FONT, 1.0, (255, 255, 255), 2)
        cv2.imshow(win_name, canvas)
        cv2.waitKey(1)

        return True

    # ------------------------------------------------------------------
    def _train_and_save(self):
        X = np.array(self._features, dtype=np.float32)
        y = np.array(self._targets,  dtype=np.float32)
        print(f"[Calibration] Training on {len(X)} samples…")
        history = self._predictor.train(X, y, verbose=0)
        final_loss = history["loss"][-1]
        print(f"[Calibration] Training complete. Final MSE: {final_loss:.5f}")
        self._predictor.save()
        np.savez(CALIBRATION_DATA_PATH, X=X, y=y)
        print(f"[Calibration] Data saved to {CALIBRATION_DATA_PATH}")

    # ------------------------------------------------------------------
    @staticmethod
    def _make_canvas(sw: int, sh: int, grid: list,
                     active_idx: int, state: str) -> np.ndarray:
        """Build the calibration display frame."""
        canvas = np.full((sh, sw, 3), BG_COLOR, dtype=np.uint8)
        for i, (nx, ny) in enumerate(grid):
            px, py = int(nx * sw), int(ny * sh)
            if i < active_idx:
                color = DOT_COLOR_DONE
                radius = DOT_RADIUS // 2
            elif i == active_idx:
                color = DOT_COLOR_ACTIVE if state == "collecting" else DOT_COLOR_IDLE
                radius = DOT_RADIUS
            else:
                color = DOT_COLOR_IDLE
                radius = DOT_RADIUS
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius + 3, (180, 180, 180), 1)
        return canvas

    @staticmethod
    def _show_intro(win_name: str, sw: int, sh: int):
        canvas = np.full((sh, sw, 3), BG_COLOR, dtype=np.uint8)
        lines = [
            "AdaptGaze – Calibration",
            "",
            "You will see 9 dots appear one at a time.",
            "Look directly at each dot until the progress bar fills.",
            "Keep your head still and blink naturally.",
            "",
            "Press SPACE or any key to begin  |  Q / ESC to quit",
        ]
        y0 = sh // 2 - len(lines) * 25
        for i, line in enumerate(lines):
            scale = 1.0 if i == 0 else 0.75
            thickness = 2 if i == 0 else 1
            color = (0, 255, 100) if i == 0 else (220, 220, 220)
            cv2.putText(canvas, line, (sw // 2 - 340, y0 + i * 50),
                        FONT, scale, color, thickness)
        cv2.imshow(win_name, canvas)

    @staticmethod
    def _get_screen_size() -> Tuple[int, int]:
        """Try to determine the display resolution via xrandr or fallback."""
        try:
            import subprocess
            out = subprocess.check_output(["xrandr"]).decode()
            for line in out.splitlines():
                if " connected" in line and "primary" in line:
                    parts = line.split()
                    for p in parts:
                        if "x" in p and "+" in p:
                            res = p.split("+")[0]
                            w, h = res.split("x")
                            return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080  # fallback
