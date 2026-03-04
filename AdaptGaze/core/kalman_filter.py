"""
AdaptGaze - Kalman Filter for Cursor Smoothing
================================================
1-D Kalman filter applied independently to X and Y cursor coordinates.
Reduces jitter from noisy gaze predictions without adding significant lag.
"""

import numpy as np
from config.settings import (
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_ERROR_COV,
)


class KalmanFilter1D:
    """Scalar Kalman filter (constant-velocity model, velocity not tracked)."""

    def __init__(self,
                 process_noise: float = KALMAN_PROCESS_NOISE,
                 measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
                 error_cov: float = KALMAN_ERROR_COV):
        self.Q = process_noise       # process noise covariance
        self.R = measurement_noise   # measurement noise covariance
        self.P = error_cov           # estimation error covariance
        self.x = 0.0                 # state estimate
        self._initialised = False

    def update(self, measurement: float) -> float:
        """Feed a new measurement and return the filtered estimate."""
        if not self._initialised:
            self.x = measurement
            self._initialised = True
            return self.x

        # Prediction step
        self.P = self.P + self.Q

        # Update step
        K = self.P / (self.P + self.R)   # Kalman gain
        self.x = self.x + K * (measurement - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

    def reset(self):
        self.P = KALMAN_ERROR_COV
        self.x = 0.0
        self._initialised = False


class KalmanFilter2D:
    """Applies independent 1-D Kalman filters to X and Y axes."""

    def __init__(self):
        self.kf_x = KalmanFilter1D()
        self.kf_y = KalmanFilter1D()

    def update(self, x: float, y: float) -> tuple:
        """Return (filtered_x, filtered_y)."""
        return self.kf_x.update(x), self.kf_y.update(y)

    def reset(self):
        self.kf_x.reset()
        self.kf_y.reset()
