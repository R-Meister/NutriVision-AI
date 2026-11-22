#!/usr/bin/env python3
"""
Train the production food classifier: frozen MobileNetV2 features + linear head.

Why this instead of fine-tuning the whole head?
-----------------------------------------------
With ~200 real photos per class, a small softmax head trained with heavy
augmentation underfits/overfits noisily.  The ImageNet backbone already
contains strong food features, so a regularised linear probe on the frozen
GAP features generalises much better on this dataset size:

    MobileNetV2 (frozen, ImageNet) → GAP(1280) → StandardScaler
    → LogisticRegression(12 classes) → softmax

The logistic-regression coefficients are folded into a Keras Dense layer, so
the saved model remains a single self-contained `.keras` file that the app
loads exactly like before.

Usage:
    python training/train_mobilenet_linear.py
    python training/train_mobilenet_linear.py --force-features
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.dataset_loader import load_real_dataset

FEATURE_CACHE = os.path.join(config.DATA_DIR, "features", "mobilenetv2_gap.npz")


def build_feature_extractor() -> Model:
    """Frozen MobileNetV2 with the same [0,1] → [-1,1] preprocessing as the app."""
    base = MobileNetV2(input_shape=(*config.IMG_SIZE, 3), include_top=False,
                       weights="imagenet")
    base.trainable = False

    inputs = layers.Input((*config.IMG_SIZE, 3), name="input_image")
    x = layers.Rescaling(2.0, -1.0, name="rescale")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    return Model(inputs, x, name="MobileNetV2_FeatureExtractor")


def extract_features(model: Model, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=0).astype(np.float32)


def _load_or_extract(force: bool = False):
    if os.path.exists(FEATURE_CACHE) and not force:
        print(f"[features] Loading cache {FEATURE_CACHE}")
        data = np.load(FEATURE_CACHE)
        return (data["Ftr"], data["Fv"], data["Fte"],
                data["ytr"], data["yv"], data["yte"])

    print("[features] Loading real images …")
    Xtr, ytr, Xv, yv, Xte, yte = load_real_dataset()
    print(f"[features] Extracting MobileNetV2 features for "
          f"{len(Xtr) + len(Xv) + len(Xte)} images …")
    encoder = build_feature_extractor()
    t0 = time.time()
    Ftr = extract_features(encoder, Xtr)
    Fv = extract_features(encoder, Xv)
    Fte = extract_features(encoder, Xte)
    print(f"[features] Done in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, Ftr=Ftr, Fv=Fv, Fte=Fte,
                        ytr=ytr, yv=yv, yte=yte)
    print(f"[features] Cached → {FEATURE_CACHE}")
    return Ftr, Fv, Fte, ytr, yv, yte


def build_inference_model(coef: np.ndarray, intercept: np.ndarray,
                          mean: np.ndarray, scale: np.ndarray) -> Model:
    """
    Fold StandardScaler + LogisticRegression into a Keras Dense layer.

    Standardised features: z = (f - mean) / scale
    Logits:                z @ coef.T + intercept
    Dense equivalent:      f @ W + b
        W = coef.T / scale[:, None]
        b = intercept - (mean / scale) @ coef.T
    """
    W = (coef.T / scale[:, None]).astype(np.float32)
    b = (intercept - (mean / scale) @ coef.T).astype(np.float32)

    base = MobileNetV2(input_shape=(*config.IMG_SIZE, 3), include_top=False,
                       weights="imagenet")
    base.trainable = False

    inputs = layers.Input((*config.IMG_SIZE, 3), name="input_image")
    x = layers.Rescaling(2.0, -1.0, name="rescale")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(len(config.FOOD_CLASSES), activation="softmax",
                           name="predictions")(x)

    model = Model(inputs, outputs, name="MobileNetV2_FoodClassifier_LinearHead")
    model.get_layer("predictions").set_weights([W, b])
    return model


def train_linear_model(force_features: bool = False,
                       C: float = 0.1) -> dict:
    """Train the linear probe and save the self-contained Keras model."""
    Ftr, Fv, Fte, ytr, yv, yte = _load_or_extract(force_features)
    ytr_i, yv_i, yte_i = ytr.argmax(1), yv.argmax(1), yte.argmax(1)

    print(f"[train] LogisticRegression(C={C}) on {Ftr.shape[1]}-D features …")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000),
    )
    clf.fit(Ftr, ytr_i)

    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]

    val_pred = clf.predict(Fv)
    test_proba = clf.predict_proba(Fte)
    test_pred = test_proba.argmax(1)
    top3 = np.argsort(test_proba, axis=1)[:, -3:]
    top3_acc = float(np.any(top3 == yte_i[:, None], axis=1).mean())

    val_acc = float((val_pred == yv_i).mean())
    test_acc = float((test_pred == yte_i).mean())
    print(f"[train] Validation accuracy : {val_acc:.4f}")
    print(f"[train] Test accuracy       : {test_acc:.4f}")
    print(f"[train] Test top-3 accuracy : {top3_acc:.4f}")

    model = build_inference_model(lr.coef_, lr.intercept_,
                                  scaler.mean_, scaler.scale_)
    model.save(config.TL_MODEL_PATH)
    print(f"[save] {config.TL_MODEL_PATH} "
          f"({os.path.getsize(config.TL_MODEL_PATH) / 1e6:.1f} MB)")

    metrics = {
        "method": "MobileNetV2 frozen GAP features + StandardScaler + LogisticRegression",
        "C": C,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "top3_accuracy": top3_acc,
        "train_samples": int(len(Ftr)),
        "val_samples": int(len(Fv)),
        "test_samples": int(len(Fte)),
    }
    out = os.path.join(config.RESULTS_DIR, "mobilenet_linear_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {out}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true",
                        help="Re-extract frozen backbone features")
    parser.add_argument("--C", type=float, default=0.1,
                        help="Inverse regularisation strength for logistic regression")
    args = parser.parse_args()
    train_linear_model(args.force_features, args.C)
    return 0


if __name__ == "__main__":
    sys.exit(main())
