"""
AdaptGaze - Gaze Regression Model
===================================
Small dense neural network that maps a feature vector (18 values) to
normalised screen coordinates (x, y) in [0, 1].

Architecture:
  Input(18) → Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(2, sigmoid)
"""

import os
import numpy as np

# Silence TF INFO / WARNING logs before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config.settings import (
    GAZE_INPUT_DIM, GAZE_HIDDEN_UNITS, GAZE_OUTPUT_DIM,
    GAZE_EPOCHS, GAZE_BATCH_SIZE, GAZE_LEARNING_RATE,
    GAZE_MODEL_PATH,
)


def build_gaze_model() -> keras.Model:
    """Construct and compile the gaze regression network."""
    inp = keras.Input(shape=(GAZE_INPUT_DIM,), name="features")
    x = inp
    for i, units in enumerate(GAZE_HIDDEN_UNITS):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{i}",
        )(x)
        x = layers.Dropout(0.2, name=f"drop_{i}")(x)
    out = layers.Dense(GAZE_OUTPUT_DIM, activation="sigmoid", name="gaze_output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="GazeNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=GAZE_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


class GazePredictor:
    """Wraps the keras model for training and inference."""

    def __init__(self):
        self.model: keras.Model = build_gaze_model()
        self._trained = False

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> dict:
        """
        Train the model on calibration data.

        Parameters
        ----------
        X : (N, 18) feature array
        y : (N, 2)  normalised screen positions in [0, 1]
        """
        cb_early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        )
        history = self.model.fit(
            X, y,
            epochs=GAZE_EPOCHS,
            batch_size=GAZE_BATCH_SIZE,
            validation_split=0.15,
            callbacks=[cb_early],
            verbose=verbose,
        )
        self._trained = True
        return history.history

    # ------------------------------------------------------------------
    def predict(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Predict normalised (x, y) from a single feature vector.

        Parameters
        ----------
        feature_vec : shape (18,)

        Returns
        -------
        np.ndarray shape (2,) – predicted (x, y) in [0, 1]
        """
        if feature_vec is None:
            return np.array([0.5, 0.5])
        x = feature_vec.reshape(1, -1).astype(np.float32)
        pred = self.model.predict(x, verbose=0)
        return pred.flatten()

    # ------------------------------------------------------------------
    def save(self, path: str = GAZE_MODEL_PATH):
        self.model.save(path)
        print(f"[GazePredictor] Model saved to {path}")

    # ------------------------------------------------------------------
    def load(self, path: str = GAZE_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            self.model = keras.models.load_model(path)
            self._trained = True
            print(f"[GazePredictor] Model loaded from {path}")
            return True
        except Exception as e:
            print(f"[GazePredictor] Failed to load model: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained
