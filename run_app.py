#!/usr/bin/env python3
"""
run_app.py - Cross-platform launcher
"""
import os, sys, subprocess

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

import config

print("=" * 60)
print("  FoodAI — Food Recognition & Calorie Estimator")
print("=" * 60)

# Check if models exist, train if not
ann_exists = os.path.exists(config.ANN_MODEL_PATH)
tl_exists  = os.path.exists(config.TL_MODEL_PATH)
cnn_exists = os.path.exists(config.CNN_MODEL_PATH)

print(f"\nModel Status:")
print(f"  ANN       : {'✅ Found' if ann_exists else '❌ Missing'}")
print(f"  MobileNetV2: {'✅ Found' if tl_exists  else '❌ Missing'}")
print(f"  CNN       : {'✅ Found' if cnn_exists else '❌ Missing'}")

if not ann_exists:
    print("\n⚠️  No models found. Running bootstrap training...")
    subprocess.run([sys.executable, "bootstrap_models.py"], check=True)

print("\n🚀 Launching Streamlit app...")
print("   URL: http://localhost:8501\n")
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.headless", "false",
                "--theme.base", "dark"])
