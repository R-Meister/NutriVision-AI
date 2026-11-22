"""
bootstrap_models.py
One-time setup helper.

Trains the nutritional ANN (fast, deterministic) and — only when the real
image dataset exists — trains the image classifiers on real photographs.

⚠️  Earlier versions of this script trained the image models on random noise,
which is why the checked-in models had chance-level accuracy.  That fallback
has been removed on purpose: a model trained on noise must never overwrite a
real checkpoint.

Usage:
    python bootstrap_models.py

Prepare real image data first:
    python training/download_dataset.py
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import config

print("=" * 70)
print("  NutriVision AI — Model Bootstrap")
print("=" * 70)
print(f"Models will be saved to: {config.SAVED_MODELS}")
print()

# ── 1. Train ANN (tabular nutrition regression) ───────────────────────────────
print("STEP 1/2 — Training Nutritional ANN…")
t0 = time.time()
from training.train_ann import train_ann
train_ann(epochs=80, batch_size=32)
print(f"  ✅ ANN done in {time.time()-t0:.0f}s\n")

# ── 2. Train image models only on REAL data ──────────────────────────────────
from training.dataset_loader import has_real_dataset

if not has_real_dataset():
    print("STEP 2/2 — Image classifiers skipped.")
    print("  ❌ No real dataset found at data/food_images.")
    print("  Run the command below, then re-run this script:")
    print("     python training/download_dataset.py")
    print("\n  (Refusing to train on synthetic noise — it produces a")
    print("   classifier that is only right by coincidence.)")
    sys.exit(0)

print("STEP 2/2 — Training the image classifier on REAL photographs…")
from training.train_mobilenet_linear import train_linear_model

train_linear_model()

print("=" * 70)
print("  🎉 Models trained and saved!")
print("  The custom from-scratch CNN is optional (CPU training is slow):")
print("     python training/train_cnn.py --model cnn --dataset real --epochs 5")
print("  Launch the app with:  streamlit run app.py")
print("=" * 70)
