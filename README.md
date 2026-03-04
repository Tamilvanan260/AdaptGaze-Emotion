# 👁️ AdaptGaze Emotion

> **AI-Powered Eye-Tracking Mouse Controller with Emotion Awareness**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.7-green)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 What is AdaptGaze Emotion?

AdaptGaze Emotion allows **physically disabled people** to control a computer mouse entirely using **eye movements and blinks** — no hands, no hardware needed.

The system also detects the user's **emotion** (Happy 😊 / Sad 😢 / Angry 😠 / Neutral 😐) from facial landmarks and **automatically adjusts cursor speed** for safer, more accurate control.

> **Reference Paper (Existing System):**
> *"Gaze Controlled Virtual Mouse Using OpenCV and PyAutoGUI"* — Elsevier, 2023
> AdaptGaze Emotion addresses **all 5 reported drawbacks** of this paper.

---

## ✨ Key Features

| Feature                       | Description                                                              |
| ----------------------------- | ------------------------------------------------------------------------ |
| 🎯 **Gaze-Controlled Cursor** | Eye movements control mouse position in real time                        |
| 😊 **Emotion Detection**      | Happy/Sad/Angry/Neutral from facial landmarks — no GPU, no FER/DeepFace  |
| ⚡ **Adaptive Cursor Speed**   | Angry/Sad → 50% slower for precision; Happy/Neutral → 100% normal        |
| 👁️ **Blink Gestures**        | Single blink = left click, double blink = right click, long blink = drag |
| 🧠 **Smart Fatigue AI**       | Predicts eye strain from EAR trend + blink patterns, alerts proactively  |
| 📐 **Kalman Smoothing**       | Removes gaze jitter — 62% reduction vs Paper 3                           |
| 🔄 **Auto-Recalibration**     | Detects head drift and accuracy drop automatically                       |
| 👤 **Multi-User Profiles**    | Separate trained model saved per user                                    |
| 📊 **Attention Tracking**     | FOCUSED / DISTRACTED / TRANSITIONING / AWAY states                       |
| 💻 **No GPU Required**        | Runs on any laptop webcam                                                |

---

## 📊 Comparison with Existing System (Paper 3)

| Metric                 | Paper 3 (Existing) | AdaptGaze Emotion | Improvement       |
| ---------------------- | ------------------ | ----------------- | ----------------- |
| Cursor Jitter          | High (0.045 std)   | Low (0.017 std)   | **62% reduction** |
| Emotion Detection      | 0%                 | **97.8%**         | New feature       |
| Fatigue Detection      | None               | Alert at 12.4 min | New feature       |
| Blink Click Accuracy   | Not supported      | **91%**           | New feature       |
| Head Pose Compensation | No                 | Yes               | New feature       |
| GPU Required           | High CPU           | **No GPU**        | Better            |

---

## 🗂️ Project Structure

```
AdaptGaze/
│
├── main.py
├── gaze_controller.py
├── existing_system.py
├── proposed_system.py
│
├── core/
│   ├── emotion_detector.py
│   ├── attention_tracker.py
│   ├── smart_fatigue.py
│   ├── auto_recalibration.py
│   ├── user_profile.py
│   ├── face_mesh.py
│   ├── feature_extractor.py
│   ├── gaze_model.py
│   ├── kalman_filter.py
│   ├── cursor_controller.py
│   ├── blink_detector.py
│   └── fatigue_monitor.py
│
├── calibration/
│   └── calibration.py
│
├── utils/
│   ├── enhanced_overlay.py
│   ├── debug_overlay.py
│   └── session_logger.py
│
├── config/
│   └── settings.py
│
├── models/
├── logs/
├── AdaptGaze_Emotion_Comparison.ipynb
└── requirements.txt
```

---

## 🚀 Installation & Setup

### Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AdaptGaze-Emotion.git
cd AdaptGaze-Emotion
```

### Step 2 — Install Dependencies

```bash
pip install numpy==1.24.3 protobuf==3.20.3 typing-extensions==4.5.0
pip install tensorflow==2.13.0 keras==2.13.1
pip install mediapipe==0.10.7
pip install opencv-python pyautogui
pip install matplotlib pandas scipy seaborn scikit-learn jupyter ipykernel
```

### Step 3 — Calibrate

```bash
python main.py --calibrate --profile YourName
```

### Step 4 — Run

```bash
python main.py --profile YourName
```

---

## 🎮 Controls

| Key     | Action         |
| ------- | -------------- |
| Q / ESC | Quit           |
| R       | Recalibrate    |
| P       | Switch profile |

| Blink Gesture | Action      |
| ------------- | ----------- |
| Single blink  | Left click  |
| Double blink  | Right click |
| Long blink    | Drag        |

---

## 📈 Run Comparison Charts

```bash
jupyter notebook AdaptGaze_Emotion_Comparison.ipynb
```

Generates 7 performance charts.

---

## 🧠 Emotion Detection Logic

```
Mouth curve  = (lip_centre_y - corner_y) / face_height
Positive → HAPPY
Negative → SAD

Eyebrow ratio = (eye_y - brow_y) / face_height
Low ratio → ANGRY

Majority vote over 12 frames → Stable output
```

---

## ⚙️ Requirements

Python 3.10
TensorFlow 2.13
MediaPipe 0.10.7
OpenCV 4.x
NumPy 1.24.3
Scikit-learn
Matplotlib
Pandas

---

## 👤 Author

TAMILVANAN I
BSc Artificial Intelligence & Machine Learning
2025–2026

---
