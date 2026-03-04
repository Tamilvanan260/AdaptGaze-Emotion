"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXISTING SYSTEM – Paper 3 Recreation                               ║
║  "Gaze Controlled Virtual Mouse Using OpenCV and PyAutoGUI"                 ║
║  Journal: Elsevier, 2023                                                    ║
║                                                                             ║
║  What this system does (exactly like Paper 3):                             ║
║    - Detects iris using MediaPipe Face Mesh                                 ║
║    - Maps iris position directly to cursor (no smoothing)                   ║
║    - No emotion detection                                                   ║
║    - No fatigue monitoring                                                  ║
║    - No blink gestures (cannot click)                                       ║
║    - No head pose compensation                                              ║
║                                                                             ║
║  Drawbacks visible when running:                                            ║
║    ✗ Cursor is jerky and unstable                                           ║
║    ✗ Head movement confuses the cursor                                      ║
║    ✗ No clicking ability                                                    ║
║    ✗ Eye strain after long use – no warning                                 ║
║    ✗ No emotion awareness                                                   ║
║                                                                             ║
║  Student : TAMILVANAN I  |  Reg: 2328C0353                                 ║
║  Guide   : MRS.M.MANIMEKALAI                                               ║
║  Purpose : Final Year Project – Existing System Comparison                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
  powershell:
    cd D:\\AdaptGaze_AI_Enhanced\\AdaptGaze
    python existing_system.py

  Press Q or ESC to quit.
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# ── Safety ────────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

# ── Screen size ───────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# Iris landmark indices (MediaPipe 478-point model)
LEFT_IRIS  = [474, 475, 476, 477]   # left iris centre landmarks
RIGHT_IRIS = [469, 470, 471, 472]   # right iris centre landmarks

# ── Camera setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 60)
print("  EXISTING SYSTEM (Paper 3) – Running")
print("  Raw iris tracking, NO smoothing, NO emotion")
print("  Press Q or ESC to quit")
print("=" * 60)

# ── FPS tracking ──────────────────────────────────────────────────────────────
prev_time  = time.time()
fps        = 0.0
frame_count = 0

# ── Jitter measurement (for comparison) ───────────────────────────────────────
cursor_positions = []   # stores (cx, cy) to measure jitter at end

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame       = cv2.flip(frame, 1)
        frame_count += 1
        h, w        = frame.shape[:2]

        # ── FPS ───────────────────────────────────────────────────────────
        now  = time.time()
        dt   = now - prev_time
        fps  = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else fps)
        prev_time = now

        # ── Convert to RGB for MediaPipe ──────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        cursor_x = cursor_y = None

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            # ── Get iris centre (average of 4 iris landmarks) ─────────────
            # Paper 3 method: use iris X,Y directly → map to screen
            # RIGHT iris controls cursor (dominant eye)
            iris_x = np.mean([landmarks[i].x for i in RIGHT_IRIS])
            iris_y = np.mean([landmarks[i].y for i in RIGHT_IRIS])

            # ── Direct mapping to screen (NO smoothing – Paper 3 method) ──
            cursor_x = int(iris_x * SCREEN_W)
            cursor_y = int(iris_y * SCREEN_H)

            # Move cursor directly – raw, unfiltered
            pyautogui.moveTo(cursor_x, cursor_y)
            cursor_positions.append((cursor_x, cursor_y))

            # ── Draw iris dots on frame ───────────────────────────────────
            for idx in RIGHT_IRIS + LEFT_IRIS:
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

            # ── Gaze position on frame ────────────────────────────────────
            gaze_px = int(iris_x * w)
            gaze_py = int(iris_y * h)
            cv2.circle(frame, (gaze_px, gaze_py), 8, (0, 0, 255), 2)

        # ── HUD overlay (simple – like Paper 3) ──────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "EXISTING SYSTEM – Paper 3 (Elsevier 2023)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}   |   Raw Iris Tracking (No Smoothing)",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        if cursor_x:
            cv2.putText(frame, f"Cursor: ({cursor_x}, {cursor_y})",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

        # Drawback warnings
        cv2.putText(frame, "NO Emotion | NO Fatigue | NO Click | NO Smoothing",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)

        # ── Show ──────────────────────────────────────────────────────────
        cv2.imshow("Existing System – Paper 3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

# ── Session summary ───────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("  EXISTING SYSTEM – Session Summary")
print("=" * 60)
print(f"  Frames processed : {frame_count}")
if len(cursor_positions) > 10:
    xs = [p[0] for p in cursor_positions]
    ys = [p[1] for p in cursor_positions]
    jitter_x = np.std(np.diff(xs))
    jitter_y = np.std(np.diff(ys))
    print(f"  Cursor jitter X  : {jitter_x:.2f} px (std of frame-to-frame change)")
    print(f"  Cursor jitter Y  : {jitter_y:.2f} px")
    print(f"  → High jitter = jerky movement (Paper 3 drawback)")
print("  Emotion detected : NONE (not implemented)")
print("  Fatigue checked  : NONE (not implemented)")
print("  Click support    : NONE (needs physical mouse)")
print("=" * 60)
print("  Compare this output with proposed_system.py")
print("=" * 60)
