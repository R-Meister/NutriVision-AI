"""Meal analyzer page: pick a photo, run the model, review and log nutrition."""
import os

import numpy as np
import streamlit as st
from PIL import Image

import config
from models.inference import analyze_meal
from ui.theme import page_header, section_title, set_flash, clean_name
from utils.database import log_meal
from utils.nutrition_database import get_nutrition

UPLOAD_LABEL = "Upload a photo"
CAMERA_LABEL = "Use camera"

SAMPLES = [
    ("Pizza Margherita", "pizza"),
    ("Chicken Biryani", "chicken_biryani"),
    ("Masala Dosa", "masala_dosa"),
    ("Classic Burger", "burger"),
    ("Crispy Samosa", "samosa"),
]

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks / Other"]

RESULT_KEY = "meal_analysis_result"


def _clear_analysis() -> None:
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop("current_analyzed_img", None)


def tab_analyzer():
    page_header(
        "Analyze a meal",
        "Upload or capture a meal photo to estimate calories and macronutrients.",
    )

    # ── Input ────────────────────────────────────────────────────────────────
    with st.container():
        input_type = st.radio(
            "Input", [UPLOAD_LABEL, CAMERA_LABEL], horizontal=True,
            label_visibility="collapsed",
        )

        pil_image = None
        if input_type == UPLOAD_LABEL:
            uploaded_file = st.file_uploader(
                "Food photo", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                pil_image = Image.open(uploaded_file).convert("RGB")
        else:
            cam_file = st.camera_input("Take a photo", label_visibility="collapsed")
            if cam_file:
                pil_image = Image.open(cam_file).convert("RGB")

        # Sample selection is kept in session state so it survives reruns
        # (for example when the Analyze button is pressed).
        if pil_image is not None:
            st.session_state.pop("active_sample", None)
        elif "active_sample" in st.session_state:
            pil_image = _generate_sample_image(st.session_state["active_sample"])

        st.caption("Samples:")
        sample_cols = st.columns(len(SAMPLES))
        for col, (label, key) in zip(sample_cols, SAMPLES):
            with col:
                if st.button(label, key=f"sample_{key}", width="stretch"):
                    _clear_analysis()
                    st.session_state["active_sample"] = key
                    st.rerun()

    if pil_image is None:
        st.caption("Choose a photo above or try a sample to get started.")
        return

    with st.expander("Analysis settings"):
        model_options = ["MobileNetV2 (Transfer Learning)"]
        if os.path.exists(config.CNN_MODEL_PATH):
            model_options.append("Custom 4-Block CNN")
        model_choice = st.selectbox(
            "Model",
            model_options,
            help="MobileNetV2 uses frozen ImageNet features with a linear food classifier.",
        )
        prefer_model = "mobilenet" if "MobileNetV2" in model_choice else "cnn"

        multi_item_mode = st.checkbox(
            "Detect multiple food regions",
            value=True,
            help="Segment plates into distinct food regions using OpenCV contour analysis "
                 "(for example a thali, combo platter or sides).",
        )
        st.caption("Portion weights are estimated automatically from image area.")

    # ── Preview and run ──────────────────────────────────────────────────────
    preview_col, action_col = st.columns([1, 1], gap="large")
    with preview_col:
        st.image(pil_image, width="stretch")

    with action_col:
        analyze_btn = st.button(
            "Analyze meal", type="primary", width="stretch",
        )
        if st.button("Clear image", width="stretch"):
            _clear_analysis()
            st.session_state.pop("active_sample", None)
            st.rerun()

        if analyze_btn:
            with st.spinner("Analyzing meal…"):
                result = analyze_meal(
                    np.array(pil_image),
                    enable_multi_item=multi_item_mode,
                    prefer_model=prefer_model,
                )
                # Reset per-item portion sliders from any previous analysis.
                for key in [k for k in st.session_state if str(k).startswith(("portion_", "item_portion_"))]:
                    del st.session_state[key]
                st.session_state[RESULT_KEY] = result
                st.session_state["current_analyzed_img"] = pil_image

        res = st.session_state.get(RESULT_KEY)
        if res and res.get("supported_items"):
            st.success("Analysis complete.")
        elif res:
            st.warning("No supported food was found in this photo.")

    if res:
        st.divider()
        _render_analysis_results(res)


# ── Results ──────────────────────────────────────────────────────────────────
def _render_analysis_results(res: dict):
    supported_items = res.get("supported_items") or []
    rejected_items = res.get("rejected_items") or []

    # A 12-class model always returns a food label; reject unrelated photos
    # (faces, landscapes, objects) instead of inventing calories for them.
    if not supported_items:
        _render_no_food(res, rejected_items)
        return

    if rejected_items:
        st.warning(
            f"{len(rejected_items)} region(s) did not match a supported food and were "
            "left out of the nutrition totals."
        )

    section_title("Result")

    # ── Totals ───────────────────────────────────────────────────────────────
    updated_items = _render_items(supported_items)
    totals = _aggregate(updated_items)

    _render_total_calorie_banner(totals)

    # ── Classification detail ────────────────────────────────────────────────
    primary = updated_items[0]
    with st.expander("Top 3 classification results"):
        st.caption(f"Confidence distribution for {clean_name(primary['display_name'])}")
        for pred in primary["top_predictions"]:
            pct = pred["confidence"] * 100
            st.markdown(f"**{clean_name(pred['display_name'])}**: {pct:.1f}%")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))

    # ── Save to diary ────────────────────────────────────────────────────────
    _render_save_to_diary(updated_items, totals)


def _render_no_food(res: dict, rejected_items: list):
    st.error(
        "This photo doesn't look like one of the supported foods, so no calories "
        "were estimated. Try a clear, close-up meal photo."
    )

    img = res.get("annotated_image")
    if img is not None:
        st.image(img, caption="Image that was checked", width="stretch")

    if rejected_items:
        best = max(rejected_items, key=lambda it: it.get("match_score") or 0.0)
        score = best.get("match_score")
        threshold = best.get("ood_threshold")
        if score is not None and threshold is not None:
            st.caption(
                f"Closest guess: {clean_name(best['display_name'])} "
                f"({best['confidence_pct']}% confidence, match score {score} "
                f"vs. required {threshold})."
            )
        else:
            st.caption(
                f"Closest guess: {clean_name(best['display_name'])} "
                f"({best['confidence_pct']}% confidence)."
            )

    supported = ", ".join(c.replace("_", " ").title() for c in config.FOOD_CLASSES)
    st.caption(f"Supported foods: {supported}")


def _render_items(items: list) -> list:
    """Render the portion adjuster for each item; return items with new portions."""
    is_multi = len(items) > 1

    if is_multi and st.session_state.get("meal_analysis_result", {}).get("annotated_image") is not None:
        st.image(
            st.session_state["meal_analysis_result"]["annotated_image"],
            caption="Detected food regions",
            width="stretch",
        )

    section_title("Items and portions")
    st.caption("Adjust portion weights to recalculate everything below.")

    updated_items = []
    for idx, item in enumerate(items):
        with st.container(border=True):
            name_col, conf_col = st.columns([3, 2])
            with name_col:
                st.markdown(f"**{clean_name(item['display_name'])}**")
                st.caption(
                    f"{item['nutrition'].get('category', 'General')} · {item['model_used']}"
                )
            with conf_col:
                st.metric("Confidence", f"{item['confidence_pct']}%")
                if item["uncertain"]:
                    st.caption("Low confidence — check the food type and portion.")

            serving_unit = item["nutrition"].get("serving_unit", "g")
            new_portion = st.slider(
                f"Portion ({serving_unit})",
                min_value=config.MIN_PORTION,
                max_value=config.MAX_PORTION,
                value=int(item["portion_grams"]),
                step=5,
                key=f"item_portion_{idx}_{item['food_name']}",
            )

            nutr = get_nutrition(item["food_name"], new_portion)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Calories", f"{nutr['calories']:.0f} kcal")
            m2.metric("Protein", f"{nutr['protein']:.1f} g")
            m3.metric("Carbs", f"{nutr['carbs']:.1f} g")
            m4.metric("Fat", f"{nutr['fat']:.1f} g")
            m5.metric("Fiber", f"{nutr['fiber']:.1f} g")
            m6.metric("Sugar", f"{nutr['sugar']:.1f} g")

        item_copy = dict(item)
        item_copy["portion_grams"] = new_portion
        item_copy["nutrition"] = nutr
        updated_items.append(item_copy)

    return updated_items


def _aggregate(items: list) -> dict:
    return {
        "calories": round(sum(it["nutrition"]["calories"] for it in items), 1),
        "protein": round(sum(it["nutrition"]["protein"] for it in items), 1),
        "carbs": round(sum(it["nutrition"]["carbs"] for it in items), 1),
        "fat": round(sum(it["nutrition"]["fat"] for it in items), 1),
        "fiber": round(sum(it["nutrition"]["fiber"] for it in items), 1),
        "sugar": round(sum(it["nutrition"]["sugar"] for it in items), 1),
        "sodium_mg": round(sum(it["nutrition"]["sodium_mg"] for it in items), 1),
        "weight_g": round(sum(it["portion_grams"] for it in items), 1),
    }


def _render_total_calorie_banner(totals: dict):
    with st.container(border=True):
        cal_col, weight_col = st.columns([2, 1])
        with cal_col:
            st.metric("Total calories", f"{totals['calories']:.0f} kcal")
        with weight_col:
            st.metric("Total weight", f"{totals['weight_g']:.0f} g")
        st.caption("Estimated from nutrition density and image-based portion sizing.")
    st.caption("Macronutrients (adjusted portions)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Protein", f"{totals['protein']:.1f} g")
    c2.metric("Carbs", f"{totals['carbs']:.1f} g")
    c3.metric("Fat", f"{totals['fat']:.1f} g")
    c4.metric("Fiber", f"{totals['fiber']:.1f} g")
    c5.metric("Sugar", f"{totals['sugar']:.1f} g")
    c6.metric("Sodium", f"{totals['sodium_mg']:.0f} mg")


def _render_save_to_diary(items: list, totals: dict):
    primary = items[0]
    if len(items) == 1:
        display_name = clean_name(primary["display_name"])
    else:
        display_name = f"{clean_name(primary['display_name'])} + {len(items) - 1} more"

    with st.form("save_meal_form", border=True):
        st.markdown("**Save to diary**")
        meal_type = st.selectbox("Meal", MEAL_TYPES)
        submitted = st.form_submit_button("Save meal", type="primary", width="stretch")

    if submitted:
        log_meal(
            food_name=primary["food_name"],
            display_name=display_name,
            confidence=primary["confidence"],
            portion_g=totals["weight_g"],
            nutrition={
                "calories": totals["calories"],
                "protein": totals["protein"],
                "carbs": totals["carbs"],
                "fat": totals["fat"],
                "fiber": totals["fiber"],
                "sugar": totals["sugar"],
                "sodium_mg": totals["sodium_mg"],
            },
            items=[{
                "food_name": it["food_name"],
                "display_name": clean_name(it["display_name"]),
                "portion_g": it["portion_grams"],
                "calories": it["nutrition"]["calories"],
                "protein": it["nutrition"]["protein"],
                "carbs": it["nutrition"]["carbs"],
                "fat": it["nutrition"]["fat"],
                "fiber": it["nutrition"]["fiber"],
                "sugar": it["nutrition"]["sugar"],
                "sodium_mg": it["nutrition"]["sodium_mg"],
            } for it in items],
            meal_type=meal_type,
            top3=primary["top_predictions"],
        )
        set_flash(f"Saved {meal_type} ({totals['calories']:.0f} kcal) to your diary.")
        st.rerun()


# ── Sample loading ───────────────────────────────────────────────────────────
def _generate_sample_image(food_key: str) -> Image.Image:
    """Return a real sample photo for the quick-test buttons."""
    sample_map = {
        "pizza": "pizza",
        "chicken_biryani": "biryani",
        "masala_dosa": "dosa",
        "burger": "burger",
        "samosa": "samosa",
    }
    sample_path = os.path.join(
        config.ASSETS_DIR, "samples", f"{sample_map.get(food_key, food_key)}.jpg"
    )
    if os.path.exists(sample_path):
        return Image.open(sample_path).convert("RGB")

    # Fallback illustration (used only if sample photos are absent).
    H, W = 300, 300
    img = np.ones((H, W, 3), dtype=np.uint8) * 245

    # Center circle for plate
    import cv2
    cv2.circle(img, (150, 150), 125, (230, 230, 230), -1)
    cv2.circle(img, (150, 150), 120, (255, 255, 255), -1)

    # Food color palettes
    if "pizza" in food_key:
        cv2.circle(img, (150, 150), 95, (30, 140, 230), -1)   # Crust/Sauce
        cv2.circle(img, (150, 150), 85, (50, 210, 255), -1)   # Cheese
        # Pepperoni / toppings
        for pt in [(120, 110), (180, 120), (140, 170), (175, 175)]:
            cv2.circle(img, pt, 14, (35, 45, 180), -1)
    elif "biryani" in food_key:
        cv2.circle(img, (150, 150), 90, (40, 180, 235), -1)   # Saffron rice
        cv2.circle(img, (135, 140), 25, (25, 75, 140), -1)    # Chicken piece
        cv2.circle(img, (165, 160), 20, (25, 75, 140), -1)
    elif "dosa" in food_key:
        cv2.ellipse(img, (150, 150), (100, 45), 30, 0, 360, (40, 170, 210), -1) # Golden dosa roll
        cv2.circle(img, (80, 120), 22, (50, 120, 200), -1)   # Sambar
        cv2.circle(img, (220, 180), 18, (240, 240, 240), -1) # Coconut chutney
    elif "burger" in food_key:
        cv2.ellipse(img, (150, 130), (70, 40), 0, 0, 360, (50, 160, 215), -1) # Bun
        cv2.rectangle(img, (85, 140), (215, 160), (30, 70, 110), -1)          # Patty
        cv2.rectangle(img, (80, 135), (220, 142), (40, 200, 50), -1)          # Lettuce
        cv2.ellipse(img, (150, 170), (68, 30), 0, 0, 360, (50, 160, 215), -1) # Bottom Bun
    else: # samosa
        pts = np.array([[150, 80], [80, 200], [220, 200]], np.int32)
        cv2.fillPoly(img, [pts], (30, 145, 210))
        cv2.circle(img, (220, 100), 20, (50, 170, 50), -1) # Green chutney

    return Image.fromarray(img)
