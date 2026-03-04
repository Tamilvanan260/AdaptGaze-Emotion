# AdaptGaze 👁️ – Adaptive Multimodal Gaze-Based Assistive Computing System

> **Final-year engineering project** — enables users with motor disabilities to control
> a computer entirely through eye gaze and blink gestures, using only a standard webcam.

---

## Table of Contents

1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Running the System](#running-the-system)
6. [Calibration Guide](#calibration-guide)
7. [Blink Gestures](#blink-gestures)
8. [Configuration](#configuration)
9. [Architecture Overview](#architecture-overview)
10. [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|---|---|
| Real-time iris tracking | MediaPipe Face Mesh (478 landmarks, iris refinement) |
| Head pose estimation | `cv2.solvePnP` → yaw & pitch in degrees |
| Personalised gaze model | Small TensorFlow Dense network trained per-user |
| 9-point calibration | Interactive full-screen calibration grid |
| Kalman filter smoothing | Eliminates jitter without adding lag |
| Blink gestures | Single / double / long blink → click / right-click / drag |
| Fatigue detection | Rolling blink-rate monitor with rest reminders |
| Session logger | CSV event log saved to `logs/` |
| Debug HUD | Real-time overlay with EAR, head pose, FPS, gaze map |

---

## System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.10.x (exactly)
- **Webcam**: Any USB or built-in webcam (720p+ recommended)
- **RAM**: ≥ 4 GB
- **Disk**: ≥ 500 MB (for dependencies)
- **Display**: Must be connected (calibration uses full-screen window)

---

## Project Structure

```
AdaptGaze/
├── main.py                     # Entry point
├── gaze_controller.py          # Main runtime orchestrator
├── requirements.txt            # Pinned dependencies
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py             # All tunable parameters
│
├── core/
│   ├── __init__.py
│   ├── face_mesh.py            # MediaPipe Face Mesh wrapper
│   ├── feature_extractor.py    # Raw landmarks → feature vector (18-D)
│   ├── gaze_model.py           # TF Dense regression model
│   ├── kalman_filter.py        # 2-D Kalman filter for cursor smoothing
│   ├── cursor_controller.py    # PyAutoGUI mouse control
│   ├── blink_detector.py       # EAR-based blink gesture state machine
│   └── fatigue_monitor.py      # Rolling blink-rate fatigue classifier
│
├── calibration/
│   ├── __init__.py
│   └── calibration.py          # 9-point interactive calibration
│
├── utils/
│   ├── __init__.py
│   ├── debug_overlay.py        # OpenCV HUD renderer
│   └── session_logger.py       # CSV session event logger
│
├── models/                     # Auto-created – stores trained model & data
│   └── (gaze_model.keras, calibration_data.npz)
│
└── logs/                       # Auto-created – session CSV logs
    └── (session_YYYYMMDD_HHMMSS.csv)
```

---

## Installation

### Step 1 – Clone / Download the Project

```bash
git clone https://github.com/your-repo/AdaptGaze.git
cd AdaptGaze
```

### Step 2 – Create a Python 3.10 Virtual Environment

```bash
# Linux / macOS
python3.10 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 – Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4 – Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note (Linux):** If PyAutoGUI fails with a display error, install:
> ```bash
> sudo apt-get install python3-tk python3-dev scrot
> ```

> **Note (macOS):** Grant Terminal / VS Code **Accessibility** permission in:
> System Preferences → Privacy & Security → Accessibility

> **Note (Windows):** If `opencv-python` fails, try:
> ```bash
> pip install opencv-python==4.8.1.78 --no-deps
> pip install numpy==1.24.3
> ```

### Step 5 – Verify Installation

```bash
python -c "import cv2, mediapipe, tensorflow, pyautogui, numpy; print('All OK')"
```

---

## Running the System

### First-Time Setup (Calibration required)

```bash
python main.py --calibrate
```

This will:
1. Open the 9-point calibration window
2. Collect gaze samples for each dot
3. Train your personal gaze model
4. Automatically start gaze control

### Subsequent Runs (after calibration)

```bash
python main.py
```

### Calibration Only (no gaze control after)

```bash
python main.py --calibrate-only
```

### Quit

Press **`q`** or **`ESC`** in the debug window, or **`Ctrl+C`** in the terminal.

---

## Calibration Guide

1. Sit **40–70 cm** from your webcam in good, even lighting.
2. Run `python main.py --calibrate`.
3. A dark full-screen window appears with 9 white dots.
4. **Look directly at each dot** as it turns green.
5. Hold your gaze steady while the progress bar fills (≈1 second per dot).
6. After all 9 dots, the model trains automatically (≈10–30 seconds).
7. Gaze control begins immediately.

**Tips for best accuracy:**
- Keep your head position consistent during calibration.
- Avoid strong backlighting behind you.
- Clean the camera lens if image is blurry.
- Re-calibrate if you move your chair/monitor significantly.

---

## Blink Gestures

| Gesture | Action | How to perform |
|---|---|---|
| **Single blink** | Left click | Close both eyes briefly (< 1.2 s), open once |
| **Double blink** | Right click | Two quick blinks within 0.4 seconds |
| **Long blink** | Drag start | Hold eyes closed ≥ 1.2 seconds, then open |
| **Long blink (again)** | Drag end | Same gesture again to release drag |

> EAR threshold and timing are tunable in `config/settings.py`.

---

## Configuration

Edit `config/settings.py` to customise behaviour:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `EAR_THRESHOLD` | `0.21` | Eye closeness threshold |
| `LONG_BLINK_MIN_SECONDS` | `1.2` | Seconds for drag gesture |
| `DOUBLE_BLINK_MAX_GAP` | `0.4` | Max gap between double blink |
| `CURSOR_SMOOTHING` | `0.35` | EMA factor (0=no move, 1=raw) |
| `CURSOR_DEADZONE` | `4` | Min pixels before cursor moves |
| `SAMPLES_PER_POINT` | `30` | Calibration samples per dot |
| `GAZE_EPOCHS` | `300` | Max training epochs |
| `SHOW_DEBUG_WINDOW` | `True` | Show camera feed with HUD |

---

## Architecture Overview

```
Webcam Frame
    │
    ▼
FaceMeshDetector (MediaPipe)
    │   iris centres, EAR, head pose, landmarks
    ▼
FeatureExtractor
    │   18-D normalised vector
    ▼
GazePredictor (TF Dense network)
    │   raw (x, y) ∈ [0,1]
    ▼
KalmanFilter2D + EMA
    │   smooth (x, y)
    ▼
CursorController (PyAutoGUI)
    │   moveTo / click / drag
    ▼
Screen

Side threads:
  BlinkDetector  → EAR → gesture → CursorController
  FatigueMonitor → blink rate → warnings
  SessionLogger  → CSV log
  DebugOverlay   → OpenCV HUD
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open camera` | Change `CAMERA_INDEX` in settings.py (try 1, 2…) |
| Cursor jittery | Increase `KALMAN_MEASUREMENT_NOISE` or decrease `CURSOR_SMOOTHING` |
| Cursor drifts | Re-run calibration; ensure stable lighting |
| Blinks not detected | Adjust `EAR_THRESHOLD` (lower if eyes aren't being detected as closed) |
| `ModuleNotFoundError` | Ensure virtual environment is activated and `pip install -r requirements.txt` succeeded |
| `PyAutoGUI FailSafeException` | Move mouse away from corner; failsafe is disabled by default in code |
| Gaze model not found | Run `python main.py --calibrate` first |
| Slow FPS | Set `SHOW_DEBUG_WINDOW = False` in settings.py |

---

## Session Logs

Each session creates a CSV file in `logs/`:

```
timestamp, elapsed_s, event, detail
2024-01-15T10:30:00, 0.0, session_start, log=logs/session_...csv
2024-01-15T10:30:05, 5.1, blink, single_left_click
2024-01-15T10:30:10, 10.2, fatigue, NORMAL
...
2024-01-15T10:50:00, 1200.0, rest_reminder, elapsed=20.0min
2024-01-15T10:55:00, 1500.0, session_end, duration_s=1500.0
```

---

## License

MIT License – free to use for academic and personal projects.
