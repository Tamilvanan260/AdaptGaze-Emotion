"""
AdaptGaze - Feature Extractor
==============================
Converts raw face-mesh output into a flat feature vector suitable for
the gaze regression model.

Feature vector (18 values):
  [0:2]  – left iris (x, y) normalised to [0, 1]
  [2:4]  – right iris (x, y) normalised to [0, 1]
  [4:6]  – left iris relative to left eye corner (dx, dy)
  [6:8]  – right iris relative to right eye corner (dx, dy)
  [8:10] – head yaw, pitch (degrees, clamped to [-45, 45])
  [10:14]– left eye bounding box centre & size (cx, cy, w, h) normalised
  [14:18]– right eye bounding box centre & size (cx, cy, w, h) normalised
"""

import numpy as np
from typing import Optional

# Eye corner landmark indices used for relative iris position
LEFT_INNER_CORNER  = 362
LEFT_OUTER_CORNER  = 263
RIGHT_INNER_CORNER = 33
RIGHT_OUTER_CORNER = 133

# Eye bounding box landmarks
LEFT_EYE_BOX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_BOX = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


class FeatureExtractor:
    """Builds normalised feature vectors from face-mesh data dictionaries."""

    def __init__(self, frame_width: int, frame_height: int):
        self.fw = float(frame_width)
        self.fh = float(frame_height)

    def extract(self, face_data: dict) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        face_data : dict returned by FaceMeshDetector.process()

        Returns
        -------
        np.ndarray of shape (18,) or None if data is incomplete.
        """
        if face_data is None:
            return None

        pts = face_data["landmarks"]
        li = face_data["left_iris"]
        ri = face_data["right_iris"]
        hp = face_data["head_pose"]

        # ── Normalise iris positions ──────────────────────────────────────
        li_norm = li / np.array([self.fw, self.fh])
        ri_norm = ri / np.array([self.fw, self.fh])

        # ── Iris relative to inner eye corner ────────────────────────────
        l_corner  = pts[LEFT_INNER_CORNER,  :2]
        r_corner  = pts[RIGHT_INNER_CORNER, :2]
        li_rel = (li - l_corner) / np.array([self.fw, self.fh])
        ri_rel = (ri - r_corner) / np.array([self.fw, self.fh])

        # ── Head pose ─────────────────────────────────────────────────────
        if hp is not None:
            yaw, pitch = hp
            yaw   = float(np.clip(yaw,   -45, 45)) / 45.0
            pitch = float(np.clip(pitch, -45, 45)) / 45.0
        else:
            yaw, pitch = 0.0, 0.0
        head_feats = np.array([yaw, pitch], dtype=np.float32)

        # ── Eye bounding boxes ────────────────────────────────────────────
        l_bbox = self._eye_bbox(pts, LEFT_EYE_BOX)
        r_bbox = self._eye_bbox(pts, RIGHT_EYE_BOX)

        feature_vec = np.concatenate([
            li_norm.astype(np.float32),
            ri_norm.astype(np.float32),
            li_rel.astype(np.float32),
            ri_rel.astype(np.float32),
            head_feats,
            l_bbox,
            r_bbox,
        ])

        # Sanity check – any NaN means something went wrong
        if np.any(np.isnan(feature_vec)):
            return None

        return feature_vec  # shape (18,)

    def _eye_bbox(self, pts: np.ndarray, indices: list) -> np.ndarray:
        """Return (cx, cy, w, h) normalised for the given eye landmark indices."""
        eye_pts = pts[indices, :2]
        x_min, y_min = eye_pts.min(axis=0)
        x_max, y_max = eye_pts.max(axis=0)
        cx = ((x_min + x_max) / 2.0) / self.fw
        cy = ((y_min + y_max) / 2.0) / self.fh
        w  = (x_max - x_min) / self.fw
        h  = (y_max - y_min) / self.fh
        return np.array([cx, cy, w, h], dtype=np.float32)
