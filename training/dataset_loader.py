"""
dataset_loader.py
Handles dataset preparation with three tiers:
  Tier 1  – Full Food-101 download (requires internet + ~5 GB disk)
  Tier 2  – Small synthetic dataset using random noise images (always works)
  Tier 3  – TensorFlow flowers dataset as a training proxy (small, fast)

The loader always returns (X_train, y_train, X_val, y_val, X_test, y_test)
as NumPy arrays.
"""
import os, sys, shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


IMG_H, IMG_W = config.IMG_SIZE
NUM_CLASSES  = config.NUM_CLASSES
SEED         = config.RANDOM_SEED


# ── Tier 2 — Synthetic dataset (always available, used as fallback) ───────────

def make_synthetic_dataset(n_per_class: int = 50) -> tuple:
    """
    Generate (n_per_class × NUM_CLASSES) random images and one-hot labels.
    Images are random noise — purely for verifying training pipeline integrity.
    """
    rng = np.random.default_rng(SEED)
    n   = n_per_class * NUM_CLASSES

    X = rng.random((n, IMG_H, IMG_W, 3), dtype=np.float32)
    y_int = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    y = tf.keras.utils.to_categorical(y_int, NUM_CLASSES).astype(np.float32)

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    # 70 / 15 / 15 split
    t  = int(0.70 * n)
    v  = int(0.85 * n)
    return X[:t], y[:t], X[t:v], y[t:v], X[v:], y[v:]


# ── Tier 3 — TF Datasets: food101 (subset) ───────────────────────────────────

def load_food101_subset(n_per_class: int = 100) -> tuple:
    """
    Load a small subset of Food-101 via TensorFlow Datasets.
    Downloads only the first time; subsequent runs use cache.

    Uses only the classes present in config.FOOD_CLASSES (12 classes).
    Food-101 class names are all lowercase with underscores — matching ours.
    """
    print("[dataset] Attempting Food-101 subset via TensorFlow Datasets…")
    try:
        # Map our class names to Food-101 indices
        # Food-101 has 101 classes; we select our 12
        builder = tfds.builder("food101")
        builder.download_and_prepare()

        # Get the integer class labels list from the builder
        all_labels = builder.info.features["label"].names
        our_indices = []
        for cls in config.FOOD_CLASSES:
            # food101 uses names like "pizza", "apple_pie" etc.
            # direct match first, then partial
            match = cls if cls in all_labels else None
            if match is None:
                for al in all_labels:
                    if cls in al or al in cls:
                        match = al
                        break
            if match:
                our_indices.append(all_labels.index(match))
            else:
                our_indices.append(None)

        valid_map = {old: new for new, old in enumerate(our_indices) if old is not None}

        ds = builder.as_dataset(split="train")
        ds = ds.filter(lambda ex: tf.reduce_any(
            [ex["label"] == i for i in valid_map.keys() if i is not None]
        ))
        ds = ds.map(lambda ex: (
            tf.image.resize(tf.cast(ex["image"], tf.float32) / 255.0, [IMG_H, IMG_W]),
            ex["label"]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(5000, seed=SEED)

        images, labels = [], []
        counts = {i: 0 for i in valid_map.keys()}
        for img, lbl in tfds.as_numpy(ds):
            li = int(lbl)
            if li in valid_map and counts[li] < n_per_class:
                images.append(img)
                labels.append(valid_map[li])
                counts[li] += 1
            if all(c >= n_per_class for c in counts.values()):
                break

        if len(images) < 10:
            raise ValueError("Too few images loaded.")

        X = np.array(images, dtype=np.float32)
        y_int = np.array(labels)
        y = tf.keras.utils.to_categorical(y_int, NUM_CLASSES).astype(np.float32)

        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        t = int(0.70 * len(X))
        v = int(0.85 * len(X))
        print(f"[dataset] Food-101 subset loaded: {len(X)} samples.")
        return X[:t], y[:t], X[t:v], y[t:v], X[v:], y[v:]

    except Exception as e:
        print(f"[dataset] Food-101 load failed: {e}")
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def load_dataset(mode: str = "auto", n_per_class: int = 100) -> tuple:
    """
    Load the dataset according to mode.

    Parameters
    ----------
    mode : "auto"       – try Food-101 → synthetic fallback
           "food101"    – force Food-101 (fails if unavailable)
           "synthetic"  – always use synthetic data

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if mode == "synthetic":
        print("[dataset] Using synthetic dataset.")
        return make_synthetic_dataset(n_per_class)

    if mode in ("auto", "food101"):
        result = load_food101_subset(n_per_class)
        if result is not None:
            return result
        if mode == "food101":
            raise RuntimeError("Food-101 load failed and mode='food101'.")
        print("[dataset] Falling back to synthetic dataset.")
        return make_synthetic_dataset(n_per_class)

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    Xtr, ytr, Xv, yv, Xt, yt = load_dataset("synthetic")
    print(f"Train={Xtr.shape}, Val={Xv.shape}, Test={Xt.shape}")
