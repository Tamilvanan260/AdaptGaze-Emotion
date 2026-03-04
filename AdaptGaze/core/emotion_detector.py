"""
AdaptGaze – Emotion Detector
==============================
Detects basic emotions from MediaPipe Face Mesh landmarks WITHOUT
any external ML library (no FER, no DeepFace).

Method: Facial Action Unit (FAU) geometry measurement
  - Mouth curve      → Happy / Sad
  - Eyebrow height   → Angry / Surprised
  - Eye openness     → combined with above for confirmation

Emotions detected:
  HAPPY     → cursor speed NORMAL  (1.0x)
  NEUTRAL   → cursor speed NORMAL  (1.0x)
  ANGRY     → cursor speed SLOW    (0.5x)
  SAD       → cursor speed SLOW    (0.5x) + rest alert

Landmark indices used (MediaPipe 478-point model):
  Mouth corners : 61 (left), 291 (right)
  Mouth top     : 13
  Mouth bottom  : 14
  Left eyebrow  : 70 (inner), 105 (outer)
  Right eyebrow : 300 (inner), 334 (outer)
  Left eye top  : 159
  Left eye bot  : 145
  Right eye top : 386
  Right eye bot : 374
  Nose tip      : 1
  Chin          : 152
"""

import numpy as np
import time
from enum import Enum, auto
from collections import deque
from typing import Optional

# ── Emotion definitions ───────────────────────────────────────────────────────
class Emotion(Enum):
    NEUTRAL = auto()
    HAPPY   = auto()
    SAD     = auto()
    ANGRY   = auto()

# Emoji labels for HUD display
EMOTION_EMOJI = {
    Emotion.NEUTRAL: "😐 NEUTRAL",
    Emotion.HAPPY:   "😊 HAPPY",
    Emotion.SAD:     "😢 SAD",
    Emotion.ANGRY:   "😠 ANGRY",
}

# Cursor speed multiplier per emotion
EMOTION_SPEED = {
    Emotion.NEUTRAL: 1.0,
    Emotion.HAPPY:   1.0,
    Emotion.SAD:     0.5,
    Emotion.ANGRY:   0.5,
}

# HUD colours (BGR) per emotion
EMOTION_COLOR = {
    Emotion.NEUTRAL: (200, 200, 200),
    Emotion.HAPPY:   (80,  255, 80),
    Emotion.SAD:     (255, 180, 80),
    Emotion.ANGRY:   (60,  60,  255),
}

# ── Landmark indices ──────────────────────────────────────────────────────────
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

L_EYEBROW_INNER = 70
L_EYEBROW_OUTER = 105
R_EYEBROW_INNER = 300
R_EYEBROW_OUTER = 334

L_EYE_TOP = 159
L_EYE_BOT = 145
R_EYE_TOP = 386
R_EYE_BOT = 374

NOSE_TIP = 1
CHIN     = 152

# ── Thresholds (tuned empirically) ───────────────────────────────────────────
MOUTH_CURVE_HAPPY_THRESH  =  0.012   # corners above centre line
MOUTH_CURVE_SAD_THRESH    = -0.010   # corners below centre line
EYEBROW_LOW_THRESH        =  0.18    # brow-to-eye ratio – low = angry
SMOOTHING_WINDOW          = 12       # frames for majority-vote smoothing


class EmotionDetector:
    """
    Geometry-based facial emotion detector using MediaPipe landmarks.
    No external ML library required.
    """

    def __init__(self):
        self._history: deque = deque(maxlen=SMOOTHING_WINDOW)
        self.emotion       = Emotion.NEUTRAL
        self.speed_factor  = 1.0
        self.emoji_label   = EMOTION_EMOJI[Emotion.NEUTRAL]
        self.color         = EMOTION_COLOR[Emotion.NEUTRAL]

        # Sad rest alert tracking
        self._sad_start_time: Optional[float] = None
        self._sad_alert_sent  = False
        self.rest_alert        = False   # True for one update cycle when alert triggers

    # ── Public API ────────────────────────────────────────────────────────────
    def update(self, landmarks: np.ndarray) -> dict:
        """
        Detect emotion from face landmark array.

        Parameters
        ----------
        landmarks : np.ndarray shape (478, 3) – pixel-space landmarks
                    from FaceMeshDetector

        Returns
        -------
        dict:
            'emotion'      : Emotion enum
            'emoji_label'  : str  e.g. "😊 HAPPY"
            'speed_factor' : float – cursor speed multiplier
            'color'        : (B,G,R) tuple for HUD
            'rest_alert'   : bool – True when SAD held > 30 seconds
            'mouth_curve'  : float – raw metric for debugging
            'brow_ratio'   : float – raw metric for debugging
        """
        self.rest_alert = False

        if landmarks is None or len(landmarks) < 478:
            return self._result(0.0, 0.0)

        # ── Compute geometry metrics ──────────────────────────────────────
        mouth_curve = self._mouth_curve(landmarks)
        brow_ratio  = self._eyebrow_ratio(landmarks)

        # ── Raw emotion classification ────────────────────────────────────
        raw = self._classify(mouth_curve, brow_ratio)

        # ── Temporal smoothing (majority vote) ───────────────────────────
        self._history.append(raw)
        self.emotion = self._majority_vote()

        # ── Derived values ────────────────────────────────────────────────
        self.speed_factor = EMOTION_SPEED[self.emotion]
        self.emoji_label  = EMOTION_EMOJI[self.emotion]
        self.color        = EMOTION_COLOR[self.emotion]

        # ── SAD rest alert logic ──────────────────────────────────────────
        if self.emotion == Emotion.SAD:
            if self._sad_start_time is None:
                self._sad_start_time = time.time()
            elif (time.time() - self._sad_start_time > 30.0
                  and not self._sad_alert_sent):
                self.rest_alert    = True
                self._sad_alert_sent = True
        else:
            self._sad_start_time = None
            self._sad_alert_sent  = False

        return self._result(mouth_curve, brow_ratio)

    # ── Geometry measurements ─────────────────────────────────────────────────
    def _mouth_curve(self, lm: np.ndarray) -> float:
        """
        Measure how much the mouth corners curve up (positive) or down (negative).

        Method:
          mouth_centre_y = midpoint of top & bottom lip
          corner_avg_y   = average of left & right corners
          curve = (mouth_centre_y - corner_avg_y) / face_height
          Positive → corners ABOVE centre → HAPPY
          Negative → corners BELOW centre → SAD
        """
        face_height = float(np.linalg.norm(
            lm[NOSE_TIP, :2] - lm[CHIN, :2]
        ))
        if face_height < 1e-3:
            return 0.0

        centre_y = (lm[MOUTH_TOP, 1] + lm[MOUTH_BOTTOM, 1]) / 2.0
        corner_y = (lm[MOUTH_LEFT, 1] + lm[MOUTH_RIGHT, 1]) / 2.0

        # In image coords Y increases downward, so corners above = smaller Y
        curve = (centre_y - corner_y) / face_height
        return float(curve)

    def _eyebrow_ratio(self, lm: np.ndarray) -> float:
        """
        Eyebrow-to-eye distance ratio.
        Low ratio → eyebrows pulled down → ANGRY
        High ratio → eyebrows raised → SURPRISED (not used here but available)
        """
        face_height = float(np.linalg.norm(
            lm[NOSE_TIP, :2] - lm[CHIN, :2]
        ))
        if face_height < 1e-3:
            return 0.25  # neutral default

        # Left side
        l_brow_y = (lm[L_EYEBROW_INNER, 1] + lm[L_EYEBROW_OUTER, 1]) / 2.0
        l_eye_y  = (lm[L_EYE_TOP, 1]  + lm[L_EYE_BOT, 1])  / 2.0
        l_dist   = l_eye_y - l_brow_y   # positive = brow above eye

        # Right side
        r_brow_y = (lm[R_EYEBROW_INNER, 1] + lm[R_EYEBROW_OUTER, 1]) / 2.0
        r_eye_y  = (lm[R_EYE_TOP, 1]  + lm[R_EYE_BOT, 1])  / 2.0
        r_dist   = r_eye_y - r_brow_y

        avg_dist = (l_dist + r_dist) / 2.0
        return float(avg_dist / face_height)

    # ── Classification ────────────────────────────────────────────────────────
    @staticmethod
    def _classify(mouth_curve: float, brow_ratio: float) -> Emotion:
        """
        Rule-based emotion classification from geometry metrics.
        Priority: ANGRY > HAPPY > SAD > NEUTRAL
        """
        # ANGRY: eyebrows pulled very low
        if brow_ratio < EYEBROW_LOW_THRESH:
            return Emotion.ANGRY

        # HAPPY: mouth corners clearly raised
        if mouth_curve > MOUTH_CURVE_HAPPY_THRESH:
            return Emotion.HAPPY

        # SAD: mouth corners clearly lowered
        if mouth_curve < MOUTH_CURVE_SAD_THRESH:
            return Emotion.SAD

        return Emotion.NEUTRAL

    def _majority_vote(self) -> Emotion:
        """Return the most frequent emotion in the recent history window."""
        if not self._history:
            return Emotion.NEUTRAL
        counts = {}
        for e in self._history:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    # ── Result builder ────────────────────────────────────────────────────────
    def _result(self, mouth_curve: float, brow_ratio: float) -> dict:
        return {
            "emotion":      self.emotion,
            "emoji_label":  self.emoji_label,
            "speed_factor": self.speed_factor,
            "color":        self.color,
            "rest_alert":   self.rest_alert,
            "mouth_curve":  round(mouth_curve, 4),
            "brow_ratio":   round(brow_ratio,  4),
        }

    def reset(self):
        self._history.clear()
        self.emotion      = Emotion.NEUTRAL
        self.speed_factor = 1.0
        self._sad_start_time = None
        self._sad_alert_sent  = False
        self.rest_alert       = False
