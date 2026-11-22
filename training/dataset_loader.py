"""
dataset_loader.py
Handles dataset preparation for the food classifier.

Modes
-----
  real       – real photographs in ``data/food_images/{train,val,test}/<class>``
               (build them with ``python training/download_dataset.py``)
  synthetic  – random-noise images; ONLY for pipeline smoke tests.  A model
               trained on this data cannot recognise food.

The loader always returns (X_train, y_train, X_val, y_val, X_test, y_test)
as NumPy arrays with one-hot labels, matching the training scripts.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMG_H, IMG_W = config.IMG_SIZE
NUM_CLASSES  = config.NUM_CLASSES
SEED         = config.RANDOM_SEED


# ── Real dataset: data/food_images/{train,val,test}/<class>/*.jpg ────────────

def _load_split(split: str, n_per_class: int | None = None) -> tuple:
    split_dir = os.path.join(config.REAL_DATA_DIR, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")

    images, labels = [], []
    for class_idx, food in enumerate(config.FOOD_CLASSES):
        folder = os.path.join(split_dir, food)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing class folder {folder}. "
                "Run: python training/download_dataset.py"
            )
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        if n_per_class is not None:
            files = files[:n_per_class]
        if not files:
            raise RuntimeError(f"No images found in {folder}")

        for path in files:
            try:
                img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception as exc:
                print(f"[dataset] Skipping unreadable image {path}: {exc}")
                continue
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(class_idx)

    if not images:
        raise RuntimeError(f"No images could be loaded from {split_dir}")

    X = np.stack(images).astype(np.float32)
    y = tf.keras.utils.to_categorical(
        np.asarray(labels, dtype=np.int32), NUM_CLASSES
    ).astype(np.float32)
    return X, y


def load_real_dataset(n_per_class: int | None = None) -> tuple:
    """Load the curated real-image dataset from ``data/food_images``."""
    print(f"[dataset] Loading real images from {config.REAL_DATA_DIR} …")
    X_tr, y_tr   = _load_split("train", n_per_class)
    X_val, y_val = _load_split("val", n_per_class)
    X_te, y_te   = _load_split("test", n_per_class)
    print(
        f"[dataset] Real dataset: train={len(X_tr)} | "
        f"val={len(X_val)} | test={len(X_te)}"
    )
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def has_real_dataset() -> bool:
    return all(
        os.path.isdir(os.path.join(config.REAL_DATA_DIR, split))
        for split in ("train", "val", "test")
    )


def load_test_dataset(n_per_class: int | None = None) -> tuple:
    """Load only the held-out test split (used by training/evaluate.py)."""
    if not os.path.isdir(os.path.join(config.REAL_DATA_DIR, "test")):
        raise FileNotFoundError(
            "Missing data/food_images/test. Run training/download_dataset.py."
        )
    return _load_split("test", n_per_class)


# ── Synthetic dataset (pipeline smoke test only) ─────────────────────────────

def make_synthetic_dataset(n_per_class: int = 50) -> tuple:
    """
    Generate random-noise images and labels.

    ⚠️  This data contains no visual signal.  Models trained on it perform at
    chance level.  It exists only to verify that the training code runs.
    """
    rng = np.random.default_rng(SEED)
    n   = n_per_class * NUM_CLASSES

    X = rng.random((n, IMG_H, IMG_W, 3), dtype=np.float32)
    y_int = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    y = tf.keras.utils.to_categorical(y_int, NUM_CLASSES).astype(np.float32)

    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    t = int(0.70 * n)
    v = int(0.85 * n)
    return X[:t], y[:t], X[t:v], y[t:v], X[v:], y[v:]


# ── Main entry point ──────────────────────────────────────────────────────────

def load_dataset(mode: str = "auto", n_per_class: int = 100) -> tuple:
    """
    Load the dataset according to mode.

    Parameters
    ----------
    mode : "auto"       – real images if available, else synthetic
           "real"       – force the real dataset (raises if missing)
           "synthetic"  – random-noise smoke-test data
           "food101"    – legacy alias for "real"
    n_per_class : optional cap on images per class
    """
    mode = mode.lower()
    if mode == "food101":
        print("[dataset] 'food101' is now assembled by training/download_dataset.py "
              "→ using the real dataset.")
        mode = "real"

    if mode == "real":
        return load_real_dataset(n_per_class)

    if mode == "auto":
        if has_real_dataset():
            return load_real_dataset(n_per_class)
        raise RuntimeError(
            "No real dataset found at data/food_images. Build it with:\n"
            "  python training/download_dataset.py\n"
            "Only use --dataset synthetic for pipeline smoke tests."
        )

    if mode == "synthetic":
        print("[dataset] ⚠️  Using SYNTHETIC noise data — accuracy will be chance.")
        return make_synthetic_dataset(n_per_class)

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    Xtr, ytr, Xv, yv, Xt, yt = load_dataset("synthetic", 2)
    print(f"Train={Xtr.shape}, Val={Xv.shape}, Test={Xt.shape}")
