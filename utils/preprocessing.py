"""
preprocessing.py
Image loading, resizing, normalisation and data-augmentation helpers
used by training scripts AND inference.
"""
import numpy as np
from PIL import Image
import io, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Single-image preprocessing ────────────────────────────────────────────────

def load_and_preprocess(image_source, target_size=None) -> np.ndarray:
    """
    Load an image from a file path, bytes, or PIL Image and return a
    normalised float32 NumPy array of shape (H, W, 3).

    Steps
    -----
    1. Decode / open the image.
    2. Resize to target_size (default config.IMG_SIZE).
    3. Convert to RGB (3 channels).
    4. Normalise pixel values to [0, 1].
    """
    if target_size is None:
        target_size = config.IMG_SIZE

    if isinstance(image_source, (str, os.PathLike)):
        img = Image.open(image_source)
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source.astype(np.uint8))
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    img = img.convert("RGB")                     # ensure 3 channels
    img = img.resize(target_size, Image.LANCZOS) # resize
    arr = np.array(img, dtype=np.float32) / 255.0  # normalise
    return arr                                   # shape (H, W, 3)


def preprocess_for_model(image_source, target_size=None) -> np.ndarray:
    """
    Return a batch-ready array of shape (1, H, W, 3) for model.predict().
    """
    arr = load_and_preprocess(image_source, target_size)
    return np.expand_dims(arr, axis=0)           # shape (1, H, W, 3)


def preprocess_mobilenet(image_source, target_size=None) -> np.ndarray:
    """
    MobileNetV2-specific preprocessing:
    pixel values scaled to [-1, 1] via tf.keras.applications.mobilenet_v2.
    Returns batch shape (1, H, W, 3).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    arr = load_and_preprocess(image_source, target_size)         # [0,1]
    arr = arr * 255.0                                             # back to [0,255]
    arr = preprocess_input(arr)                                   # → [-1,1]
    return np.expand_dims(arr, axis=0)


# ── Keras data-augmentation layer (used during training) ─────────────────────

def build_augmentation_layer():
    """
    Returns a tf.keras Sequential augmentation model (applied only during
    training).  Covers rotation, flip, zoom and brightness jitter.
    """
    import tensorflow as tf
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.AUG_ROTATION / 360.0),
        tf.keras.layers.RandomZoom(config.AUG_ZOOM),
        tf.keras.layers.RandomBrightness(
            factor=(config.AUG_BRIGHTNESS[0] - 1, config.AUG_BRIGHTNESS[1] - 1)
        ),
    ], name="data_augmentation")
    return aug


# ── Batch generator (NumPy, no TF Dataset required) ──────────────────────────

def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int,
                    shuffle: bool = True):
    """
    Minimal NumPy-based batch generator — used when tf.data is unavailable.

    Yields
    ------
    (X_batch, y_batch) : np.ndarray pairs
    """
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = indices[start:end]
        yield X[batch], y[batch]


# ── Utility: decode prediction array into labelled dict ──────────────────────

def decode_predictions(probs: np.ndarray, top_k: int = 3) -> list[dict]:
    """
    Convert a softmax probability array into a list of top-k predictions.

    Parameters
    ----------
    probs : 1-D array of shape (NUM_CLASSES,)
    top_k : number of top predictions to return

    Returns
    -------
    List of dicts: [{"label": str, "confidence": float}, ...]
    """
    from utils.nutrition_database import get_display_name
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        label = config.FOOD_CLASSES[idx]
        results.append({
            "label":        label,
            "display_name": get_display_name(label),
            "confidence":   float(probs[idx]),
        })
    return results
