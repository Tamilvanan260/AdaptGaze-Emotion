"""
AdaptGaze - Global Configuration Settings
==========================================
Central configuration for all system parameters.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CALIBRATION_DATA_PATH = os.path.join(MODEL_DIR, "calibration_data.npz")
GAZE_MODEL_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# ─── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_POINTS = 9          # 3x3 grid
SAMPLES_PER_POINT = 30          # frames collected per calibration point
CALIBRATION_WAIT_SECONDS = 1.5  # pause before collecting samples

# ─── Gaze Model ────────────────────────────────────────────────────────────────
GAZE_INPUT_DIM = 18             # iris + head pose features
GAZE_HIDDEN_UNITS = [128, 64, 32]
GAZE_OUTPUT_DIM = 2             # (x, y) normalised screen coordinates
GAZE_EPOCHS = 300
GAZE_BATCH_SIZE = 16
GAZE_LEARNING_RATE = 1e-3

# ─── Kalman Filter ─────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2
KALMAN_ERROR_COV = 1.0

# ─── Cursor ────────────────────────────────────────────────────────────────────
CURSOR_SMOOTHING = 0.35         # EMA coefficient (lower = smoother)
CURSOR_DEADZONE = 4             # pixels – ignore movements smaller than this

# ─── Blink Detection ───────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.21            # Eye Aspect Ratio threshold for closed eye
BLINK_CONSEC_FRAMES = 2         # consecutive frames for a blink
DOUBLE_BLINK_MAX_GAP = 0.7      # seconds between two blinks for double-blink
LONG_BLINK_MIN_SECONDS = 1.2    # seconds held closed for long blink / drag

# ─── Fatigue Detection ─────────────────────────────────────────────────────────
FATIGUE_WINDOW_SECONDS = 60     # sliding window for blink rate
FATIGUE_HIGH_BLINK_RATE = 25    # blinks/min  → eyes straining
FATIGUE_LOW_BLINK_RATE = 8      # blinks/min  → drowsy

# ─── Session Logger ────────────────────────────────────────────────────────────
LOG_FLUSH_INTERVAL = 30         # seconds between log flushes

# ─── Display ───────────────────────────────────────────────────────────────────
SHOW_DEBUG_WINDOW = True
DEBUG_WINDOW_SCALE = 1.0
