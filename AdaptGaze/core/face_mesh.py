"""
AdaptGaze - Face Mesh Module
=============================
Wraps MediaPipe Face Mesh to extract:
  - Iris landmark positions (left & right)
  - Eye contour landmarks for EAR calculation
  - 3-D facial landmarks for head-pose estimation
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

# ─── MediaPipe landmark indices ────────────────────────────────────────────────
# Iris centres (MediaPipe 478-point model)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye contour points for EAR (Eye Aspect Ratio)
# Format: [p1,p2,p3,p4,p5,p6] – horizontal + vertical pairs
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

# 3-D head-pose reference landmarks
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]


class FaceMeshDetector:
    """Real-time face mesh + iris landmark detector."""

    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,          # enables iris landmarks (468-477)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Process a BGR frame and return extracted features.

        Returns
        -------
        dict with keys:
            'left_iris'   : (x, y) normalised centre of left iris
            'right_iris'  : (x, y) normalised centre of right iris
            'left_ear'    : float – left Eye Aspect Ratio
            'right_ear'   : float – right Eye Aspect Ratio
            'landmarks'   : full (478, 3) array of normalised landmarks
            'head_pose'   : (yaw, pitch) in degrees – may be None
        or None if no face detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        # Convert normalised landmarks to pixel coords array
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)

        left_iris_centre = self._iris_centre(pts, LEFT_IRIS)
        right_iris_centre = self._iris_centre(pts, RIGHT_IRIS)
        left_ear = self._eye_aspect_ratio(pts, LEFT_EYE_EAR_IDX)
        right_ear = self._eye_aspect_ratio(pts, RIGHT_EYE_EAR_IDX)
        head_pose = self._head_pose(pts, frame_bgr.shape)

        return {
            "left_iris": left_iris_centre,
            "right_iris": right_iris_centre,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "landmarks": pts,
            "head_pose": head_pose,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _iris_centre(pts: np.ndarray, indices: list) -> np.ndarray:
        """Return mean (x, y) of the four iris boundary points."""
        selected = pts[indices, :2]
        return selected.mean(axis=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, idx: list) -> float:
        """
        Compute Eye Aspect Ratio.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        p1, p2, p3, p4, p5, p6 = [pts[i, :2] for i in idx]
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    # ------------------------------------------------------------------
    @staticmethod
    def _head_pose(pts: np.ndarray, shape: Tuple) -> Optional[Tuple[float, float]]:
        """
        Estimate head yaw and pitch (degrees) using solvePnP.

        Uses a generic 3-D face model whose metric coordinates match the
        MediaPipe canonical face geometry.
        """
        # Generic 3-D model points (mm) – nose, left/right eye corners, mouth corners, chin
        model_points = np.array([
            [0.0,    0.0,    0.0],      # nose tip         (lm 1)
            [-30.0, -125.0, -30.0],     # left eye corner  (lm 33)
            [30.0, -125.0, -30.0],      # right eye corner (lm 263)
            [-25.0,  170.0, -50.0],     # left mouth       (lm 61)
            [25.0,  170.0, -50.0],      # right mouth      (lm 291)
            [0.0,   200.0, -100.0],     # chin             (lm 199)
        ], dtype=np.float64)

        h, w = shape[:2]
        focal = w
        cam_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1   ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = pts[HEAD_POSE_POINTS, :2].astype(np.float64)

        try:
            success, rvec, _ = cv2.solvePnP(
                model_points, image_points, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return None
            rot_mat, _ = cv2.Rodrigues(rvec)
            # Decompose rotation matrix to Euler angles
            sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            if sy > 1e-6:
                pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
                yaw   = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            else:
                pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
                yaw   = 0.0
            return float(yaw), float(pitch)
        except cv2.error:
            return None

    # ------------------------------------------------------------------
    def close(self):
        self.face_mesh.close()
