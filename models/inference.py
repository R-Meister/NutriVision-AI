"""
inference.py
Unified Deep Learning Inference Engine for NutriVision AI.
Features:
- Cached model loader (MobileNetV2, Custom CNN, Nutrition ANN)
- Multi-food region detection & individual item decomposition
- Genuine top-3 confidence scores & low-confidence uncertainty handling
- Complete nutritional computation (Calories, Protein, Carbs, Fat, Fiber, Sugar, Sodium)
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.nutrition_database import (get_nutrition, get_display_name,
                                      get_typical_portion, get_database)
from utils.portion_estimation import (estimate_portion, detect_multi_food_regions,
                                      draw_bounding_boxes)
from utils.preprocessing import decode_predictions

# ── Global Model Cache ────────────────────────────────────────────────────────
_MODEL_CACHE = {}
_FEATURE_MODEL_CACHE = {}
_OOD_CACHE = {"loaded": False, "features": None, "threshold": None}

def _load_model(path: str, model_type: str):
    """Load model from disk once and cache in memory."""
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        _MODEL_CACHE[path] = model
        print(f"[inference] Successfully loaded {model_type} from {path}")
        return model
    except Exception as e:
        print(f"[inference] Error loading {model_type}: {e}")
        return None


def get_cnn_model():
    return _load_model(config.CNN_MODEL_PATH, "CNN")


def get_mobilenet_model():
    return _load_model(config.TL_MODEL_PATH, "MobileNetV2")


def get_ann_model():
    return _load_model(config.ANN_MODEL_PATH, "ANN")


# ── Out-of-distribution / "is this one of our foods?" check ──────────────────
#
# A 12-class softmax always returns one of the 12 labels, even for a face or a
# landscape.  To avoid showing fake calories, compare the image's MobileNetV2
# GAP feature with the training features.  If the closest training photo is
# still far away (cosine similarity below the calibrated threshold), the image
# is reported as "not one of the supported foods".

def _load_ood_calibration() -> dict:
    if _OOD_CACHE["loaded"]:
        return _OOD_CACHE
    _OOD_CACHE["loaded"] = True

    path = config.OOD_CALIBRATION_PATH
    if not os.path.exists(path):
        print("[inference] OOD calibration not found; non-food gate disabled.")
        return _OOD_CACHE
    try:
        data = np.load(path)
        features = data["features"].astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _OOD_CACHE["features"] = features / norms
        _OOD_CACHE["threshold"] = float(data["threshold"])
        print(
            f"[inference] OOD gate loaded: {len(features)} reference features, "
            f"threshold={_OOD_CACHE['threshold']:.2f}"
        )
    except Exception as exc:
        print(f"[inference] Could not load OOD calibration: {exc}")
    return _OOD_CACHE


def _get_feature_model(model):
    key = id(model)
    if key in _FEATURE_MODEL_CACHE:
        return _FEATURE_MODEL_CACHE[key]
    try:
        feature_model = tf.keras.Model(
            model.input, model.get_layer("gap").output, name="gap_features"
        )
        _FEATURE_MODEL_CACHE[key] = feature_model
        return feature_model
    except Exception:
        return None


def _food_match_score(model, batch: np.ndarray):
    """Return (mean top-5 cosine similarity to training food features, threshold)."""
    cal = _load_ood_calibration()
    if cal["features"] is None:
        return None, None
    feature_model = _get_feature_model(model)
    if feature_model is None:
        return None, None
    feat = feature_model.predict(batch, verbose=0)[0].astype(np.float32)
    norm = float(np.linalg.norm(feat))
    if norm == 0:
        return 0.0, cal["threshold"]
    sims = cal["features"] @ (feat / norm)
    k = min(5, len(sims))
    score = float(np.sort(sims)[-k:].mean())
    return score, cal["threshold"]


# ── Single Item Prediction ────────────────────────────────────────────────────

def predict_single_food(image_crop: np.ndarray,
                        portion_override: float = None,
                        area_ratio: float = None,
                        top_k: int = 3,
                        prefer_model: str = "mobilenet") -> dict:
    """
    Classify a single food image or cropped region.
    """
    model = None
    model_name = "none"

    if prefer_model == "mobilenet":
        model = get_mobilenet_model()
        model_name = "MobileNetV2 (Transfer Learning)"
    else:
        model = get_cnn_model()
        model_name = "Custom CNN"

    # If the requested checkpoint is missing, fall back to the other model
    # instead of silently switching to the colour heuristic.
    if model is None:
        if prefer_model == "mobilenet":
            model = get_cnn_model()
            model_name = "Custom CNN (fallback)"
        else:
            model = get_mobilenet_model()
            model_name = "MobileNetV2 (fallback)"

    # Preprocess image crop to (1, 224, 224, 3)
    img = image_crop
    if img.max() <= 1.0:
        img = img
    else:
        img = img / 255.0

    if img.shape[:2] != config.IMG_SIZE:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize(config.IMG_SIZE, Image.LANCZOS)
        img = np.array(pil, dtype=np.float32) / 255.0

    batch = np.expand_dims(img.astype(np.float32), 0)

    # Run model prediction
    if model is not None:
        probs = model.predict(batch, verbose=0)[0]
    else:
        probs = _heuristic_color_distribution(img)
        model_name = "Heuristic Fallback"

    # Is this one of the 12 supported foods, or an unrelated photo?
    match_score, ood_threshold = _food_match_score(model, batch) if model is not None else (None, None)
    supported_food = (match_score is None) or (match_score >= ood_threshold)

    # Top-K predictions
    top_preds = decode_predictions(probs, top_k=top_k)
    top1 = top_preds[0]
    food_name = top1["label"]
    confidence = float(top1["confidence"])
    uncertain = (confidence < config.CONFIDENCE_THRESHOLD) or (not supported_food)

    # Portion estimation
    if portion_override is not None:
        portion_g = float(portion_override)
        portion_method = "user_manual_override"
    else:
        est = estimate_portion(food_name, image_array=image_crop, area_ratio=area_ratio)
        portion_g = float(est["estimated_grams"])
        portion_method = est["method"]

    # Nutrition calculation
    nutr = get_nutrition(food_name, portion_g)

    return {
        "food_name": food_name,
        "display_name": get_display_name(food_name),
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "uncertain": uncertain,
        "supported_food": bool(supported_food),
        "match_score": None if match_score is None else round(float(match_score), 3),
        "ood_threshold": None if ood_threshold is None else round(float(ood_threshold), 3),
        "portion_grams": portion_g,
        "portion_method": portion_method,
        "nutrition": nutr,
        "top_predictions": top_preds,
        "model_used": model_name
    }


# ── Full Meal / Multi-Food Analysis ──────────────────────────────────────────

def analyze_meal(image_array: np.ndarray,
                 enable_multi_item: bool = True,
                 prefer_model: str = "mobilenet") -> dict:
    """
    Complete Meal Analysis Pipeline:
    1. Detect if multiple food items exist (OpenCV segmentation)
    2. Run classification and portion estimation for each item
    3. Calculate total meal nutrition
    4. Annotate image with bounding boxes
    """
    if enable_multi_item:
        regions = detect_multi_food_regions(image_array, max_regions=4)
    else:
        H, W = image_array.shape[:2]
        regions = [{
            "box": (0, 0, W, H),
            "crop": image_array,
            "area_ratio": 1.0,
            "region_id": 1
        }]

    analyzed_items = []
    labels = []

    for reg in regions:
        crop = reg["crop"]
        area_r = reg["area_ratio"]
        pred = predict_single_food(crop, area_ratio=area_r, prefer_model=prefer_model)
        pred["box"] = reg["box"]
        pred["region_id"] = reg["region_id"]
        analyzed_items.append(pred)
        if pred.get("supported_food", True):
            labels.append(f"{pred['display_name']} ({pred['confidence_pct']}%)")
        else:
            labels.append(f"Not recognized ({pred['confidence_pct']}%)")

    # Annotate image
    annotated_img = draw_bounding_boxes(image_array, regions, labels)

    supported_items = [it for it in analyzed_items if it.get("supported_food", True)]
    rejected_items = [it for it in analyzed_items if not it.get("supported_food", True)]

    # Aggregate only accepted food items; never invent calories for non-food.
    total_cal  = round(sum(it["nutrition"]["calories"]  for it in supported_items), 1)
    total_prot = round(sum(it["nutrition"]["protein"]   for it in supported_items), 1)
    total_carb = round(sum(it["nutrition"]["carbs"]     for it in supported_items), 1)
    total_fat  = round(sum(it["nutrition"]["fat"]       for it in supported_items), 1)
    total_fib  = round(sum(it["nutrition"]["fiber"]     for it in supported_items), 1)
    total_sug  = round(sum(it["nutrition"]["sugar"]     for it in supported_items), 1)
    total_sod  = round(sum(it["nutrition"]["sodium_mg"] for it in supported_items), 1)
    total_wt   = round(sum(it["portion_grams"]          for it in supported_items), 1)

    primary = supported_items[0] if supported_items else (
        analyzed_items[0] if analyzed_items else None
    )

    return {
        "is_multi_food": len(analyzed_items) > 1,
        "is_supported_meal": bool(supported_items),
        "supported_items": supported_items,
        "rejected_items": rejected_items,
        "items": analyzed_items,
        "primary_item": primary,
        "annotated_image": annotated_img,
        "total_calories": total_cal,
        "total_protein": total_prot,
        "total_carbs": total_carb,
        "total_fat": total_fat,
        "total_fiber": total_fib,
        "total_sugar": total_sug,
        "total_sodium_mg": total_sod,
        "total_weight_g": total_wt
    }


def _heuristic_color_distribution(img: np.ndarray) -> np.ndarray:
    """Demo probability distribution when no weights are available."""
    mean_r = float(np.mean(img[:, :, 0]))
    mean_g = float(np.mean(img[:, :, 1]))
    mean_b = float(np.mean(img[:, :, 2]))

    weights = np.ones(config.NUM_CLASSES, dtype=np.float32) * 0.05
    if mean_r > 0.6 and mean_g < 0.4:
        weights[0] += 0.6  # pizza
    elif mean_g > 0.5:
        weights[8] += 0.5  # apple
    elif mean_r > 0.7 and mean_g > 0.6:
        weights[9] += 0.5  # banana
    else:
        weights[6] += 0.5  # biryani

    return np.exp(weights) / np.sum(np.exp(weights))
