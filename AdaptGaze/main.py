"""
AdaptGaze - Entry Point (AI Enhanced)
=======================================
Usage:
  python main.py                        # Run with default profile
  python main.py --profile Alice        # Run with named user profile
  python main.py --calibrate            # Calibrate then run (default profile)
  python main.py --calibrate --profile Alice   # Calibrate and save as Alice
  python main.py --calibrate-only       # Calibrate only
  python main.py --list-profiles        # Show all saved user profiles
  python main.py --help                 # Show this help
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def print_banner():
    print(r"""
 ___  ____   __   ____  ____  ___    __   ____  ____
/ __)( ___) (  ) ( ___)( ___)/ __)  / _\ (  _ \( ___)
\__ \ )__)   )(  )__)  ) __) \__ \ /    \ )___/ )__)
(___/(____) (__)(____)(____)(___//_/\/\_\(__)  (____)

   Adaptive Multimodal Gaze-Based Assistive Computing System
   AI Intelligence Edition
   -----------------------------------------------------------
""")


def get_arg(args, flag, default=None):
    """Get value of --flag VALUE from args list."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def run_calibration(profile_name: str = "default") -> bool:
    from calibration.calibration import CalibrationSystem
    print(f"[Main] Starting calibration for profile: '{profile_name}'")
    cal = CalibrationSystem()
    success = cal.run()
    if success:
        # Save to named profile if not default
        if profile_name != "default":
            try:
                from core.user_profile import ProfileManager
                from core.gaze_model import GazePredictor
                from config.settings import CALIBRATION_DATA_PATH, GAZE_MODEL_PATH
                import numpy as np
                from tensorflow import keras

                pm = ProfileManager()
                predictor = GazePredictor()
                if predictor.load(GAZE_MODEL_PATH):
                    if os.path.exists(CALIBRATION_DATA_PATH):
                        data = np.load(CALIBRATION_DATA_PATH)
                        pm.save_profile(profile_name, predictor.model,
                                        data["X"], data["y"])
                        print(f"[Main] Profile '{profile_name}' saved.")
            except Exception as e:
                print(f"[Main] Could not save profile: {e}")
        print("[Main] Calibration complete.")
    else:
        print("[Main] Calibration was aborted or failed.")
    return success


def list_profiles():
    from core.user_profile import ProfileManager
    pm = ProfileManager()
    profiles = pm.list_profiles()
    if not profiles:
        print("[Main] No saved profiles found.")
        print("       Run: python main.py --calibrate --profile YourName")
        return
    print(f"\n[Main] Saved profiles ({len(profiles)}):")
    print(f"  {'Name':<20} {'Samples':<10} {'Last Used'}")
    print(f"  {'-'*20} {'-'*10} {'-'*20}")
    for p in profiles:
        print(f"  {p.get('username','?'):<20} "
              f"{p.get('num_cal_samples','?'):<10} "
              f"{p.get('last_used','?')[:19]}")
    print()


def run_gaze_control(profile_name: str = "default"):
    from gaze_controller import GazeController
    ctrl = GazeController(profile_name=profile_name)
    ctrl.run()


def main():
    print_banner()
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    if "--list-profiles" in args:
        list_profiles()
        sys.exit(0)

    profile_name   = get_arg(args, "--profile", "default")
    calibrate      = "--calibrate"      in args
    calibrate_only = "--calibrate-only" in args

    if calibrate or calibrate_only:
        success = run_calibration(profile_name)
        if calibrate_only or not success:
            sys.exit(0 if success else 1)

    # Check model exists
    from config.settings import GAZE_MODEL_PATH
    from core.user_profile import ProfileManager
    pm = ProfileManager()
    has_model = os.path.exists(GAZE_MODEL_PATH) or pm.profile_exists(profile_name)

    if not has_model:
        print("[Main] No trained model found.")
        ans = input("       Run calibration now? [y/N]: ").strip().lower()
        if ans == "y":
            success = run_calibration(profile_name)
            if not success:
                print("[Main] Calibration failed. Exiting.")
                sys.exit(1)
        else:
            print("[Main] Exiting. Run --calibrate first.")
            sys.exit(0)

    run_gaze_control(profile_name)


if __name__ == "__main__":
    main()
