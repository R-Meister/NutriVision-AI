"""
bootstrap_models.py
Quick-train all models using synthetic data so the app works immediately.
Run this BEFORE launching app.py.

Usage:
    python bootstrap_models.py

Time estimate on CPU:
    ANN  training: ~30 seconds
    CNN  training: ~3-5 minutes
    MobileNetV2:   ~5-10 minutes
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import config

print("=" * 70)
print("  FoodAI — Bootstrap Model Training")
print("=" * 70)
print(f"Models will be saved to: {config.SAVED_MODELS}")
print()

# ── 1. Train ANN (fast, always works) ────────────────────────────────────────
print("STEP 1/3 — Training Nutritional ANN…")
t0 = time.time()
from training.train_ann import train_ann
train_ann(epochs=80, batch_size=32)
print(f"  ✅ ANN done in {time.time()-t0:.0f}s\n")

# ── 2. Train Custom CNN ───────────────────────────────────────────────────────
print("STEP 2/3 — Training Custom CNN (synthetic data, quick epochs)…")
t0 = time.time()
from training.dataset_loader import load_dataset
from training.train_cnn import train_cnn_model
Xtr, ytr, Xv, yv, _, _ = load_dataset("synthetic", n_per_class=30)
train_cnn_model(Xtr, ytr, Xv, yv, epochs=10, batch_size=16)
print(f"  ✅ CNN done in {time.time()-t0:.0f}s\n")

# ── 3. Train MobileNetV2 ──────────────────────────────────────────────────────
print("STEP 3/3 — Training MobileNetV2 (feature extraction on synthetic data)…")
t0 = time.time()
from training.train_cnn import train_mobilenet_model
train_mobilenet_model(Xtr, ytr, Xv, yv, epochs=10, fine_tune=False, batch_size=16)
print(f"  ✅ MobileNetV2 done in {time.time()-t0:.0f}s\n")

print("=" * 70)
print("  🎉 All models trained and saved!")
print("  Launch the app with:  streamlit run app.py")
print("=" * 70)
