#!/usr/bin/env python3
"""
run_app.py - cross-platform launcher.

Works on Windows, macOS, and Linux.  On Windows the recommended entry point is
still launch.bat, which also creates the virtual environment and installs
dependencies.
"""
import os
import subprocess
import sys

# Windows consoles often default to cp1252; make the Unicode status output safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None:
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

import config  # noqa: E402

BANNER = "=" * 60


def main() -> int:
    print(BANNER)
    print("  NutriVision AI - Food Recognition & Calorie Counter")
    print(BANNER)
    print()

    mobilenet = os.path.exists(config.TL_MODEL_PATH)
    ann = os.path.exists(config.ANN_MODEL_PATH)
    ood = os.path.exists(config.OOD_CALIBRATION_PATH)

    print("Model status:")
    print(f"  MobileNetV2 classifier : {'OK' if mobilenet else 'MISSING'}")
    print(f"  Nutritional ANN        : {'OK' if ann else 'MISSING'}")
    print(f"  Non-food calibration   : {'OK' if ood else 'MISSING (gate disabled)'}")
    print()

    if not mobilenet:
        print("WARNING: the food classifier checkpoint is missing.")
        print("The app will open, but recognition falls back to a weaker path.")
        print("To build it on this machine:")
        print("  python training/download_dataset.py")
        print("  python training/train_mobilenet_linear.py")
        print()

    try:
        import streamlit  # noqa: F401
    except Exception:
        print("Streamlit is not installed in this Python environment.")
        print("Install dependencies with:")
        print("  python -m pip install -r requirements.txt")
        return 1

    print("Launching Streamlit...")
    print("URL: http://localhost:8501")
    print("Press Ctrl+C to stop the app.")
    print()
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
