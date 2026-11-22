"""
train_ann.py
Train the Nutritional ANN on synthetically generated data.
Runs quickly (< 60 seconds on CPU) and always succeeds.
"""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.ann_model import build_ann, compile_ann, build_training_data, get_ann_callbacks


def train_ann(epochs: int = config.ANN_EPOCHS,
              batch_size: int = 32,
              save_path: str = config.ANN_MODEL_PATH) -> dict:
    """
    Train the Nutritional ANN and save it to `save_path`.

    Returns
    -------
    dict: training history metrics
    """
    print("=" * 60)
    print("  Training Nutritional ANN")
    print("=" * 60)

    # ── Build dataset ─────────────────────────────────────────────────────
    X, y = build_training_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features → {y.shape[1]} outputs")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )

    # ── Build & compile ───────────────────────────────────────────────────
    model = build_ann()
    compile_ann(model)
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_ann_callbacks(),
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_loss = min(history.history["val_loss"])
    print(f"\n✅ ANN training complete. Best val_mae = {val_loss:.4f}")

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = os.path.join(config.RESULTS_DIR, "ann_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"History saved → {hist_path}")

    return history.history


if __name__ == "__main__":
    train_ann()
