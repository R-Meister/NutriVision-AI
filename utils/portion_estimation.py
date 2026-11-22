"""
portion_estimation.py
Advanced portion size and multi-food region estimation using OpenCV.
Features:
- Contours & adaptive saliency region segmentation for multiple items
- Food area coverage calculation relative to plate / frame
- Transparent approximation warnings and manual override support
"""
import numpy as np
import cv2
from PIL import Image
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.nutrition_database import get_typical_portion


def estimate_portion(food_name: str, image_array: np.ndarray = None, area_ratio: float = None) -> dict:
    """
    Estimate food portion weight in grams.

    Strategy:
    1. Base weight = typical serving size for food item from verified DB.
    2. Area scale factor = derived from food pixel coverage (or explicit area_ratio).
    3. Clamped to [MIN_PORTION, MAX_PORTION].
    """
    typical = get_typical_portion(food_name)
    method  = "verified_typical_portion"
    scale   = 1.0

    if area_ratio is not None:
        # Scale typical by ratio relative to expected median coverage (0.4)
        scale = 0.65 + (float(area_ratio) / 0.45) * 0.55
        scale = float(np.clip(scale, 0.5, 1.8))
        method = "region_contour_area"
    elif image_array is not None:
        try:
            cov = _food_region_ratio(image_array)
            # Map coverage 0.25 - 0.75 -> 0.7 - 1.4 of typical
            scale = 0.7 + (cov - 0.25) / 0.50 * 0.7
            scale = float(np.clip(scale, 0.6, 1.6))
            method = "image_foreground_coverage"
        except Exception:
            scale = 1.0

    est = int(np.clip(round(typical * scale), config.MIN_PORTION, config.MAX_PORTION))

    return {
        "estimated_grams": est,
        "method": method,
        "note": "⚠️ Estimated from 2D pixel area and typical serving size. Adjust using slider for exact weight."
    }


def _food_region_ratio(image_array: np.ndarray) -> float:
    """Compute foreground food pixel fraction vs background."""
    img = image_array
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Convert to HSV for robust food segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Non-neutral (food) pixels usually have saturation > 35 and value between 25 and 245
    mask = (saturation > 35) & (value > 30) & (value < 245)
    return float(mask.mean())


def detect_multi_food_regions(image_array: np.ndarray, max_regions: int = 4) -> list[dict]:
    """
    Segment multiple food items from a single image (e.g. thali, plate with sides).

    Returns
    -------
    list of dict:
        [{
            "box": (x, y, w, h),
            "crop": np.ndarray (H, W, 3),
            "area_ratio": float,
            "region_id": int
        }, ...]
    """
    img = image_array
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    H, W = img.shape[:2]
    total_area = H * W

    # Preprocess with Gaussian blur and adaptive threshold in LAB/HSV
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to join food components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_regions = []
    min_area = total_area * 0.05   # at least 5% of plate
    max_area = total_area * 0.85   # less than 85% to be considered multi-region

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Add padding
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                valid_regions.append({
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "crop": crop.astype(np.float32) / 255.0,
                    "area_ratio": float(area / total_area),
                    "region_id": len(valid_regions) + 1
                })

        if len(valid_regions) >= max_regions:
            break

    # If no distinct multiple regions found or only 1 small region, return entire image
    if len(valid_regions) < 2:
        return [{
            "box": (0, 0, W, H),
            "crop": (img.astype(np.float32) / 255.0),
            "area_ratio": _food_region_ratio(img),
            "region_id": 1
        }]

    return valid_regions


def draw_bounding_boxes(image_array: np.ndarray, regions: list[dict], labels: list[str] = None) -> np.ndarray:
    """Draw bounding boxes and item tags on the meal image."""
    img = image_array.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    colors = [(16, 185, 129), (59, 130, 246), (245, 158, 11), (239, 68, 68)]  # Emerald, Blue, Amber, Red

    for i, reg in enumerate(regions):
        x, y, w, h = reg["box"]
        if w == img.shape[1] and h == img.shape[0]:
            # Whole image, no box needed
            continue
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        label_text = f"Item {i+1}"
        if labels and i < len(labels):
            label_text = labels[i]

        # Tag banner
        cv2.putText(img, label_text, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return img
