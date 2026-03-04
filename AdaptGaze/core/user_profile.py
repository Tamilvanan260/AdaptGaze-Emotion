"""
AdaptGaze – Multi-User Profile Manager
=========================================
Saves and loads per-user gaze models and calibration data.
Each user gets their own directory under models/profiles/.

Profile contents:
  models/profiles/<username>/
      gaze_model.keras          – trained gaze model
      calibration_data.npz      – raw calibration features & targets
      profile.json              – metadata (name, date, accuracy stats)

Usage:
  pm = ProfileManager()
  pm.save_profile("Alice", model, cal_data)
  model = pm.load_profile("Alice")
  profiles = pm.list_profiles()
"""

import os
import json
import shutil
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

# Silence TF before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

from config.settings import MODEL_DIR

PROFILES_DIR = os.path.join(MODEL_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)


class ProfileManager:
    """Manages per-user gaze model profiles."""

    def __init__(self):
        self._active_profile: Optional[str] = None

    # ── Save ─────────────────────────────────────────────────────────────────
    def save_profile(self, username: str, model: keras.Model,
                     X: np.ndarray, y: np.ndarray,
                     extra_meta: Optional[dict] = None) -> str:
        """
        Save a trained model and calibration data for a user.

        Parameters
        ----------
        username  : str – profile name (alphanumeric + underscores)
        model     : trained keras Model
        X         : calibration feature array
        y         : calibration target array
        extra_meta: optional additional metadata to store

        Returns
        -------
        Path to the profile directory.
        """
        username = self._sanitise_name(username)
        profile_dir = os.path.join(PROFILES_DIR, username)
        os.makedirs(profile_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(profile_dir, "gaze_model.keras")
        model.save(model_path)

        # Save calibration data
        cal_path = os.path.join(profile_dir, "calibration_data.npz")
        np.savez(cal_path, X=X, y=y)

        # Save metadata
        meta = {
            "username":         username,
            "created_at":       datetime.now().isoformat(),
            "last_used":        datetime.now().isoformat(),
            "num_cal_samples":  int(len(X)),
            "model_input_dim":  int(X.shape[1]) if len(X.shape) > 1 else 0,
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = os.path.join(profile_dir, "profile.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._active_profile = username
        print(f"[ProfileManager] Profile saved: {username} ({len(X)} samples)")
        return profile_dir

    # ── Load ─────────────────────────────────────────────────────────────────
    def load_profile(self, username: str) -> Optional[keras.Model]:
        """
        Load a user's gaze model.

        Returns the keras Model or None if profile doesn't exist.
        """
        username = self._sanitise_name(username)
        profile_dir = os.path.join(PROFILES_DIR, username)
        model_path  = os.path.join(profile_dir, "gaze_model.keras")

        if not os.path.exists(model_path):
            print(f"[ProfileManager] Profile not found: {username}")
            return None

        try:
            model = keras.models.load_model(model_path)
            self._active_profile = username
            self._update_last_used(profile_dir)
            print(f"[ProfileManager] Profile loaded: {username}")
            return model
        except Exception as e:
            print(f"[ProfileManager] Failed to load profile '{username}': {e}")
            return None

    def load_calibration_data(self, username: str):
        """
        Load raw calibration arrays for a user.

        Returns (X, y) tuple or (None, None).
        """
        username = self._sanitise_name(username)
        cal_path = os.path.join(PROFILES_DIR, username, "calibration_data.npz")
        if not os.path.exists(cal_path):
            return None, None
        data = np.load(cal_path)
        return data["X"], data["y"]

    # ── List / Delete ─────────────────────────────────────────────────────────
    def list_profiles(self) -> List[Dict]:
        """
        Return a list of all saved profiles with metadata.

        Returns
        -------
        List of dicts with keys: username, created_at, last_used, num_cal_samples
        """
        profiles = []
        if not os.path.exists(PROFILES_DIR):
            return profiles

        for name in sorted(os.listdir(PROFILES_DIR)):
            profile_dir = os.path.join(PROFILES_DIR, name)
            meta_path   = os.path.join(profile_dir, "profile.json")
            model_path  = os.path.join(profile_dir, "gaze_model.keras")

            if not os.path.isdir(profile_dir) or not os.path.exists(model_path):
                continue

            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {"username": name, "created_at": "unknown",
                        "last_used": "unknown", "num_cal_samples": 0}

            profiles.append(meta)

        return profiles

    def delete_profile(self, username: str) -> bool:
        """Delete a user profile. Returns True on success."""
        username = self._sanitise_name(username)
        profile_dir = os.path.join(PROFILES_DIR, username)
        if os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)
            print(f"[ProfileManager] Profile deleted: {username}")
            if self._active_profile == username:
                self._active_profile = None
            return True
        return False

    def profile_exists(self, username: str) -> bool:
        username = self._sanitise_name(username)
        model_path = os.path.join(PROFILES_DIR, username, "gaze_model.keras")
        return os.path.exists(model_path)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _sanitise_name(name: str) -> str:
        """Keep only alphanumeric, underscores, hyphens. Max 32 chars."""
        clean = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
        return clean[:32] or "default"

    @staticmethod
    def _update_last_used(profile_dir: str):
        meta_path = os.path.join(profile_dir, "profile.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["last_used"] = datetime.now().isoformat()
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    @property
    def active_profile(self) -> Optional[str]:
        return self._active_profile
