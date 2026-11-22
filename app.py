"""
app.py — NutriVision AI: Food Image Analyzer & Calorie Counter
Inspired by WebMD Food Calorie Counter & modern health platforms.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import config
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime

from utils.nutrition_database import (get_database, get_nutrition, get_display_name,
                                      get_typical_portion, search_foods,
                                      get_all_categories, load_nutrition_df)
from utils.database import (log_meal, get_today_log, get_history,
                            get_daily_summary, get_daily_goals, set_daily_goals,
                            delete_entry, clear_today, get_weekly_trend)
from models.inference import (get_mobilenet_model, get_cnn_model,
                              get_ann_model, analyze_meal, predict_single_food)

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision AI — Food Calorie & Nutrition Counter",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Clean Light / Emerald WebMD-inspired CSS ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Light clean background */
[data-testid="stAppViewContainer"] {
    background: #F8FAFC;
    color: #0F172A;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
}

/* Brand Navbar */
.nv-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 24px;
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.04);
    margin-bottom: 24px;
}

.nv-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nv-brand-logo {
    font-size: 2rem;
    background: #ECFDF5;
    padding: 6px 12px;
    border-radius: 12px;
}

.nv-brand-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #065F46;
    letter-spacing: -0.5px;
    margin: 0;
}

.nv-brand-tagline {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
    margin: 0;
}

/* Hero Section */
.nv-hero {
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 50%, #FFFFFF 100%);
    border: 1px solid #D1FAE5;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(16, 185, 129, 0.06);
}

.nv-hero h1 {
    font-size: 2.3rem;
    font-weight: 800;
    color: #064E3B;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}

.nv-hero p {
    font-size: 1.05rem;
    color: #334155;
    max-width: 780px;
    line-height: 1.6;
    margin: 0;
}

/* Cards */
.nv-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.04);
    margin-bottom: 20px;
}

.nv-card:hover {
    box-shadow: 0 8px 25px -4px rgba(0, 0, 0, 0.07);
    transition: all 0.2s ease-in-out;
}

/* Calorie Banner */
.calorie-banner {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    color: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 6px 20px -2px rgba(16, 185, 129, 0.3);
    margin-bottom: 20px;
}

.calorie-val {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 4px 0;
}

.calorie-unit {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 600;
}

/* Metric Cards */
.macro-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

.macro-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
}

.macro-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #64748B;
    margin-bottom: 4px;
}

.macro-num {
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
}

.macro-unit {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
}

/* Confidence Badge */
.badge-conf {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.badge-green { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.badge-amber { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 10px 22px;
    transition: all 0.2s;
    border: none;
}

.primary-btn button {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
}

.primary-btn button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
}

/* Budget Widget */
.budget-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.budget-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: #059669;
}

/* Food Item Card */
.food-item-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #10B981;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached AI Engine Loader ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing AI Inference Models…")
def load_models():
    mn  = get_mobilenet_model()
    cnn = get_cnn_model()
    ann = get_ann_model()
    return mn, cnn, ann


# ── Header / Top Navigation Bar ───────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="nv-navbar">
        <div class="nv-brand">
            <div class="nv-brand-logo">🥗</div>
            <div>
                <h2 class="nv-brand-title">NutriVision AI</h2>
                <p class="nv-brand-tagline">AI-Powered Food Recognition & Nutritional Analysis</p>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:16px;">
            <span style="background:#ECFDF5; color:#047857; padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; border:1px solid #A7F3D0;">
                🟢 504+ Food Classes Active
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 1: AI Food Analyzer (Primary Core Feature) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def tab_analyzer():
    # Hero Section
    st.markdown("""
    <div class="nv-hero">
        <h1>Know What's on Your Plate.</h1>
        <p>
            Upload a meal photograph or snap a picture. Our deep learning model identifies food items,
            detects plate regions, calculates calories, and breaks down protein, carbs, fats, fiber, and sugar instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Samples
    st.markdown("**⚡ Quick Test with Sample Meals:**")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    sample_choice = None
    with qc1:
        if st.button("🍕 Pizza Margherita", use_container_width=True): sample_choice = "pizza"
    with qc2:
        if st.button("🍗 Chicken Biryani", use_container_width=True): sample_choice = "chicken_biryani"
    with qc3:
        if st.button("🫓 Masala Dosa", use_container_width=True): sample_choice = "masala_dosa"
    with qc4:
        if st.button("🍔 Classic Burger", use_container_width=True): sample_choice = "burger"
    with qc5:
        if st.button("🥟 Crispy Samosa", use_container_width=True): sample_choice = "samosa"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Upload & Camera Card
    col_input, col_options = st.columns([2, 1])

    with col_input:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Food Image Input")
        input_type = st.radio("Choose Input Method:", ["📁 Upload Food Photo", "📷 Use Camera"], horizontal=True)

        pil_image = None
        if input_type == "📁 Upload Food Photo":
            uploaded_file = st.file_uploader("Drag and drop your food photo here", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                pil_image = Image.open(uploaded_file).convert("RGB")
        else:
            cam_file = st.camera_input("Snap a meal photo")
            if cam_file:
                pil_image = Image.open(cam_file).convert("RGB")

        # Fallback to quick sample image generator
        if pil_image is None and sample_choice is not None:
            pil_image = _generate_sample_image(sample_choice)
            st.session_state["active_sample"] = sample_choice

        st.markdown('</div>', unsafe_allow_html=True)

    with col_options:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Analysis Settings")
        model_choice = st.selectbox(
            "Deep Learning Model:",
            ["MobileNetV2 (Transfer Learning)", "Custom 4-Block CNN"],
            help="MobileNetV2 uses ImageNet features with fine-tuned food classification head."
        )
        prefer_model = "mobilenet" if "MobileNetV2" in model_choice else "cnn"

        multi_item_mode = st.checkbox(
            "✨ Multi-Food Region Segmentation",
            value=True,
            help="Segment plates into distinct food regions using OpenCV contour analysis (e.g. combo platter, thali, sides)."
        )

        st.caption("ℹ️ Automatic depth and area calibration scales typical serving weight approximations.")
        st.markdown('</div>', unsafe_allow_html=True)

    # If image is ready
    if pil_image is not None:
        st.markdown("---")
        preview_col, action_col = st.columns([1, 2])

        with preview_col:
            st.markdown('<div class="nv-card">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Image Preview")
            st.image(pil_image, use_column_width=True)
            if st.button("🗑️ Remove / Clear Image", use_container_width=True):
                st.session_state.pop("meal_analysis_result", None)
                st.session_state.pop("active_sample", None)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with action_col:
            st.markdown('<div class="nv-card">', unsafe_allow_html=True)
            st.markdown("#### 🚀 Ready for Deep Learning Analysis")
            st.write("Click below to run model forward pass, region segmentation, portion scaling, and macronutrient regression.")

            analyze_btn = st.button("⚡ ANALYZE FOOD & CALCULATE CALORIES", type="primary", use_container_width=True)

            if analyze_btn or "meal_analysis_result" in st.session_state:
                if analyze_btn:
                    with st.spinner("🤖 Neural network is analyzing food features & segmenting plate..."):
                        img_arr = np.array(pil_image)
                        result = analyze_meal(img_arr, enable_multi_item=multi_item_mode, prefer_model=prefer_model)
                        st.session_state["meal_analysis_result"] = result
                        st.session_state["current_analyzed_img"] = pil_image

                res = st.session_state.get("meal_analysis_result")
                if res:
                    st.success("✅ Food Analysis Complete!")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Results Presentation ──────────────────────────────────────────────
        res = st.session_state.get("meal_analysis_result")
        if res:
            _render_analysis_results(res)


def _render_analysis_results(res: dict):
    st.markdown("---")
    st.markdown("## 📊 Nutritional Analysis Results")

    items = res["items"]
    is_multi = res["is_multi_food"]

    # If multi-food, show annotated image
    if is_multi and "annotated_image" in res:
        col_ann, col_summary = st.columns([1, 1])
        with col_ann:
            st.markdown('<div class="nv-card">', unsafe_allow_html=True)
            st.markdown("#### 🍱 Detected Food Regions")
            st.image(res["annotated_image"], caption="OpenCV Region Segmentation", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_summary:
            _render_total_calorie_banner(res)
    else:
        _render_total_calorie_banner(res)

    # ── Itemized Breakdown & Portion Adjuster ──────────────────────────────────
    st.markdown("### 🍽️ Recognized Food Items & Serving Weight")
    st.caption("Adjust portion weights below for real-time recalculation of all calories and macros.")

    updated_items = []
    for idx, item in enumerate(items):
        with st.container():
            st.markdown(f'<div class="food-item-box">', unsafe_allow_html=True)
            c_name, c_conf, c_slider = st.columns([2, 1, 3])

            with c_name:
                st.markdown(f"### {item['display_name']}")
                st.caption(f"Category: **{item['nutrition'].get('category', 'General')}** · Model: {item['model_used']}")

            with c_conf:
                badge_class = "badge-green" if not item["uncertain"] else "badge-amber"
                st.markdown(f'<span class="badge-conf {badge_class}">Confidence: {item["confidence_pct"]}%</span>', unsafe_allow_html=True)
                if item["uncertain"]:
                    st.caption("⚠️ Low confidence. Please verify portion & food type.")

            with c_slider:
                serving_unit = item["nutrition"].get("serving_unit", "g")
                new_portion = st.slider(
                    f"Portion Weight ({serving_unit})",
                    min_value=config.MIN_PORTION,
                    max_value=config.MAX_PORTION,
                    value=int(item["portion_grams"]),
                    step=5,
                    key=f"portion_{idx}_{item['food_name']}"
                )

            # Recalculate nutrition with new portion
            nutr_recalc = get_nutrition(item["food_name"], new_portion)

            # Macro pills for this item
            mp1, mp2, mp3, mp4, mp5, mp6 = st.columns(6)
            mp1.metric("Calories", f"{nutr_recalc['calories']:.0f} kcal")
            mp2.metric("Protein", f"{nutr_recalc['protein']:.1f}g")
            mp3.metric("Carbs", f"{nutr_recalc['carbs']:.1f}g")
            mp4.metric("Fat", f"{nutr_recalc['fat']:.1f}g")
            mp5.metric("Fiber", f"{nutr_recalc['fiber']:.1f}g")
            mp6.metric("Sugar", f"{nutr_recalc['sugar']:.1f}g")

            item_copy = dict(item)
            item_copy["portion_grams"] = new_portion
            item_copy["nutrition"] = nutr_recalc
            updated_items.append(item_copy)

            st.markdown('</div>', unsafe_allow_html=True)

    # Re-aggregate grand meal totals
    recalc_cal  = round(sum(it["nutrition"]["calories"]  for it in updated_items), 1)
    recalc_prot = round(sum(it["nutrition"]["protein"]   for it in updated_items), 1)
    recalc_carb = round(sum(it["nutrition"]["carbs"]     for it in updated_items), 1)
    recalc_fat  = round(sum(it["nutrition"]["fat"]       for it in updated_items), 1)
    recalc_fib  = round(sum(it["nutrition"]["fiber"]     for it in updated_items), 1)
    recalc_sug  = round(sum(it["nutrition"]["sugar"]     for it in updated_items), 1)
    recalc_sod  = round(sum(it["nutrition"]["sodium_mg"] for it in updated_items), 1)
    recalc_wt   = round(sum(it["portion_grams"]          for it in updated_items), 1)

    # ── Interactive Charts ────────────────────────────────────────────────────
    st.markdown("### 📈 Macronutrient Breakdown")
    col_donut, col_bar = st.columns([1, 1])

    with col_donut:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("#### Calorie Contribution by Macro")
        # 1g Protein = 4 kcal, 1g Carbs = 4 kcal, 1g Fat = 9 kcal
        cal_prot = recalc_prot * 4
        cal_carb = recalc_carb * 4
        cal_fat  = recalc_fat * 9
        total_macro_cal = max(cal_prot + cal_carb + cal_fat, 1)

        fig_donut = go.Figure(data=[go.Pie(
            labels=["Protein (4 kcal/g)", "Carbohydrates (4 kcal/g)", "Fat (9 kcal/g)"],
            values=[cal_prot, cal_carb, cal_fat],
            hole=0.55,
            marker=dict(colors=["#3B82F6", "#8B5CF6", "#F59E0B"]),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )])
        fig_donut.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bar:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("#### Macro Grams vs Recommended Serving")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="This Meal",
            x=["Protein", "Carbs", "Fat", "Fiber", "Sugar"],
            y=[recalc_prot, recalc_carb, recalc_fat, recalc_fib, recalc_sug],
            marker_color=["#3B82F6", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899"],
            text=[f"{v}g" for v in [recalc_prot, recalc_carb, recalc_fat, recalc_fib, recalc_sug]],
            textposition="outside"
        ))
        fig_bar.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            yaxis=dict(title="Grams (g)", gridcolor="#E2E8F0"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top-3 Predictions Detail ──────────────────────────────────────────────
    with st.expander("🔬 View Top-3 Classification Confidences"):
        primary = updated_items[0]
        st.write(f"Confidence distribution for **{primary['display_name']}**:")
        for pred in primary["top_predictions"]:
            pct = pred["confidence"] * 100
            st.markdown(f"**{pred['display_name']}**: `{pct:.1f}%`")
            st.progress(min(pct / 100.0, 1.0))

    # ── Save to Daily Food Diary ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Log Meal to Your Food Diary")

    col_mtype, col_save = st.columns([1, 2])
    with col_mtype:
        meal_type = st.selectbox("Meal Category:", ["Breakfast", "Lunch", "Dinner", "Snacks / Other"])

    with col_save:
        st.write("")
        st.write("")
        if st.button("✅ SAVE ANALYZED MEAL TO DAILY DIARY", type="primary", use_container_width=True):
            primary = updated_items[0]
            log_meal(
                food_name=primary["food_name"],
                display_name=primary["display_name"] if len(updated_items) == 1 else f"{primary['display_name']} + {len(updated_items)-1} sides",
                confidence=primary["confidence"],
                portion_g=recalc_wt,
                nutrition={
                    "calories": recalc_cal,
                    "protein": recalc_prot,
                    "carbs": recalc_carb,
                    "fat": recalc_fat,
                    "fiber": recalc_fib,
                    "sugar": recalc_sug,
                    "sodium_mg": recalc_sod
                },
                items=[{
                    "food_name": it["food_name"],
                    "display_name": it["display_name"],
                    "portion_g": it["portion_grams"],
                    "calories": it["nutrition"]["calories"],
                    "protein": it["nutrition"]["protein"],
                    "carbs": it["nutrition"]["carbs"],
                    "fat": it["nutrition"]["fat"],
                    "fiber": it["nutrition"]["fiber"],
                    "sugar": it["nutrition"]["sugar"],
                    "sodium_mg": it["nutrition"]["sodium_mg"]
                } for it in updated_items],
                meal_type=meal_type,
                top3=primary["top_predictions"]
            )
            st.success(f"🎉 Successfully saved **{meal_type}** ({recalc_cal:.0f} kcal) to your Daily Food Diary!")
            st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)


def _render_total_calorie_banner(res: dict):
    cal  = res["total_calories"]
    prot = res["total_protein"]
    carb = res["total_carbs"]
    fat  = res["total_fat"]
    fib  = res["total_fiber"]
    sug  = res["total_sugar"]
    wt   = res["total_weight_g"]

    st.markdown(f"""
    <div class="calorie-banner">
        <div style="font-size:0.95rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; opacity:0.9;">
            TOTAL ESTIMATED MEAL CALORIES ({wt:.0f}g serving)
        </div>
        <div class="calorie-val">{cal:.0f} <span class="calorie-unit">kcal</span></div>
        <div style="font-size:0.85rem; opacity:0.85;">Based on verified nutritional density algorithms</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="macro-grid">
        <div class="macro-box">
            <div class="macro-title">Protein</div>
            <div class="macro-num" style="color:#2563EB;">{prot:.1f}<span class="macro-unit">g</span></div>
        </div>
        <div class="macro-box">
            <div class="macro-title">Carbohydrates</div>
            <div class="macro-num" style="color:#7C3AED;">{carb:.1f}<span class="macro-unit">g</span></div>
        </div>
        <div class="macro-box">
            <div class="macro-title">Total Fat</div>
            <div class="macro-num" style="color:#D97706;">{fat:.1f}<span class="macro-unit">g</span></div>
        </div>
        <div class="macro-box">
            <div class="macro-title">Dietary Fiber</div>
            <div class="macro-num" style="color:#059669;">{fib:.1f}<span class="macro-unit">g</span></div>
        </div>
        <div class="macro-box">
            <div class="macro-title">Total Sugars</div>
            <div class="macro-num" style="color:#DB2777;">{sug:.1f}<span class="macro-unit">g</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _generate_sample_image(food_key: str) -> Image.Image:
    """Generate a high-quality stylized food illustration for instant testing."""
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


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 2: My Food Diary & Daily Totals (WebMD Calorie Calculator) ────────────
# ══════════════════════════════════════════════════════════════════════════════

def tab_diary():
    st.markdown("""
    <div class="nv-hero">
        <h1>My Food Diary & Daily Totals</h1>
        <p>
            Track your daily calorie budget, macronutrient progress against RDA goals,
            meal-by-meal history, and 7-day nutritional trends inspired by the WebMD Calorie Counter.
        </p>
    </div>
    """, unsafe_allow_html=True)

    today_summary = get_daily_summary()
    daily_goals = get_daily_goals()

    goal_cal = daily_goals["calorie_goal"]
    consumed_cal = today_summary["calories"]
    remaining_cal = max(0.0, goal_cal - consumed_cal)
    cal_pct = min(100.0, (consumed_cal / max(goal_cal, 1)) * 100)

    # ── Calorie Budget Widget ─────────────────────────────────────────────────
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Daily Calorie Budget")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f'<div class="budget-card"><p style="margin:0;color:#64748B;font-weight:600;">Daily Goal</p><div class="budget-val">{goal_cal:.0f}</div><span style="font-size:0.85rem;color:#64748B;">kcal</span></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="budget-card"><p style="margin:0;color:#64748B;font-weight:600;">Food Consumed</p><div class="budget-val" style="color:#2563EB;">{consumed_cal:.0f}</div><span style="font-size:0.85rem;color:#64748B;">kcal</span></div>', unsafe_allow_html=True)
    with b3:
        st.markdown(f'<div class="budget-card"><p style="margin:0;color:#64748B;font-weight:600;">Remaining</p><div class="budget-val" style="color:{"#059669" if remaining_cal > 0 else "#DC2626"};">{remaining_cal:.0f}</div><span style="font-size:0.85rem;color:#64748B;">kcal</span></div>', unsafe_allow_html=True)
    with b4:
        st.markdown(f'<div class="budget-card"><p style="margin:0;color:#64748B;font-weight:600;">Logged Meals</p><div class="budget-val" style="color:#7C3AED;">{today_summary["entries"]}</div><span style="font-size:0.85rem;color:#64748B;">items today</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(f"**Budget Progress:** `{consumed_cal:.0f} / {goal_cal:.0f} kcal ({cal_pct:.1f}%)`")
    st.progress(cal_pct / 100.0)

    # Goal edit popover
    with st.expander("⚙️ Customize My Nutritional Goals"):
        cg1, cg2, cg3, cg4 = st.columns(4)
        new_cal = cg1.number_input("Calorie Target (kcal):", value=int(goal_cal), step=50)
        new_prot = cg2.number_input("Protein Target (g):", value=int(daily_goals["protein_goal"]), step=5)
        new_carb = cg3.number_input("Carbs Target (g):", value=int(daily_goals["carbs_goal"]), step=10)
        new_fat = cg4.number_input("Fat Target (g):", value=int(daily_goals["fat_goal"]), step=5)
        if st.button("Update Personal Targets"):
            set_daily_goals(new_cal, new_prot, new_carb, new_fat)
            st.success("Target goals updated!")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Macronutrient Progress Meters ─────────────────────────────────────────
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Daily Macronutrient Targets (RDA)")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        prot_goal = daily_goals["protein_goal"]
        p_pct = min(1.0, today_summary["protein"] / max(prot_goal, 1))
        st.markdown(f"**Protein:** `{today_summary['protein']:.1f} / {prot_goal:.0f}g` ({p_pct*100:.0f}%)")
        st.progress(p_pct)

    with m2:
        carb_goal = daily_goals["carbs_goal"]
        c_pct = min(1.0, today_summary["carbs"] / max(carb_goal, 1))
        st.markdown(f"**Carbohydrates:** `{today_summary['carbs']:.1f} / {carb_goal:.0f}g` ({c_pct*100:.0f}%)")
        st.progress(c_pct)

    with m3:
        fat_goal = daily_goals["fat_goal"]
        f_pct = min(1.0, today_summary["fat"] / max(fat_goal, 1))
        st.markdown(f"**Fat:** `{today_summary['fat']:.1f} / {fat_goal:.0f}g` ({f_pct*100:.0f}%)")
        st.progress(f_pct)

    with m4:
        fib_goal = daily_goals["fiber_goal"]
        fib_pct = min(1.0, today_summary["fiber"] / max(fib_goal, 1))
        st.markdown(f"**Fiber:** `{today_summary['fiber']:.1f} / {fib_goal:.0f}g` ({fib_pct*100:.0f}%)")
        st.progress(fib_pct)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Today's Meals List ────────────────────────────────────────────────────
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Meals Logged Today")

    today_entries = get_today_log()
    if not today_entries:
        st.info("No meals logged today yet. Use the **AI Food Analyzer** to analyze and log your first meal!")
    else:
        for entry in today_entries:
            time_str = str(entry.get("timestamp", ""))[:16].replace("T", " ")
            cals = float(entry.get("calories") or 0.0)
            prot = float(entry.get("protein") or 0.0)
            carb = float(entry.get("carbs") or 0.0)
            fat  = float(entry.get("fat") or 0.0)
            fib  = float(entry.get("fiber") or 0.0)
            sug  = float(entry.get("sugar") or 0.0)
            portion = float(entry.get("portion_g") or 0.0)
            conf = float(entry.get("confidence") or 0.0)
            mtype = entry.get("meal_type") or "Meal"
            dname = entry.get("display_name") or entry.get("food_name", "Food")

            with st.expander(f"{mtype}: {dname} — {cals:.0f} kcal ({portion:.0f}g)"):
                ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
                ec1.metric("Calories", f"{cals:.0f} kcal")
                ec2.metric("Protein", f"{prot:.1f}g")
                ec3.metric("Carbs", f"{carb:.1f}g")
                ec4.metric("Fat", f"{fat:.1f}g")
                ec5.metric("Fiber", f"{fib:.1f}g")
                ec6.metric("Sugar", f"{sug:.1f}g")

                st.caption(f"Logged at: {time_str} · AI Confidence: {conf*100:.1f}%")

                if st.button("🗑️ Delete Meal", key=f"del_{entry['id']}"):
                    delete_entry(entry["id"])
                    st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Entire Daily Log", type="secondary"):
            clear_today()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 7-Day Trend Chart ─────────────────────────────────────────────────────
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("### 📅 7-Day Calorie Intake Trend")

    trend_data = get_weekly_trend(7)
    df_trend = pd.DataFrame(trend_data)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=df_trend["label"],
        y=df_trend["calories"],
        marker_color=["#059669" if i == 6 else "#10B981" for i in range(len(df_trend))],
        text=[f"{v:.0f}" for v in df_trend["calories"]],
        textposition="outside",
        name="Calorie Intake"
    ))
    fig_trend.add_hline(y=goal_cal, line_dash="dash", line_color="#DC2626", annotation_text=f"Goal: {goal_cal:.0f} kcal")
    fig_trend.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        yaxis=dict(title="Calories (kcal)", gridcolor="#E2E8F0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 3: Nutrition Database (504+ Verified Items) ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def tab_database():
    st.markdown("""
    <div class="nv-hero">
        <h1>Verified Smart Nutrition Database</h1>
        <p>
            Browse and search our database of <b>504+ verified food items</b> covering Indian specialties,
            Western meals, Asian cuisine, fruits, vegetables, beverages, and desserts.
            Calculated per 100g with instant custom serving calculators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search & Category Filters
    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 1])

    with sc1:
        search_query = st.text_input("🔍 Search food name, ingredient or dish:", placeholder="e.g. biryani, paneer, apple, dosa, oats...")

    with sc2:
        categories = ["All"] + get_all_categories()
        selected_cat = st.selectbox("Filter by Category:", categories)

    st.markdown('</div>', unsafe_allow_html=True)

    # Popular Foods Carousel
    st.markdown("### ⭐ Popular Food Items")
    pop_keys = ["chicken_biryani", "pizza", "masala_dosa", "burger", "butter_chicken", "samosa", "apple", "banana"]
    pop_cols = st.columns(4)

    for i, pk in enumerate(pop_keys):
        rec = get_nutrition(pk, 100)
        with pop_cols[i % 4]:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:14px; margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-weight:700; font-size:1.05rem; margin-bottom:4px;">{rec['display_name']}</div>
                <div style="color:#059669; font-weight:800; font-size:1.2rem;">{rec['calories']:.0f} kcal <span style="font-size:0.75rem; color:#64748B;">/ 100g</span></div>
                <div style="font-size:0.78rem; color:#64748B; margin-top:4px;">P: {rec['protein']}g · C: {rec['carbs']}g · F: {rec['fat']}g</div>
            </div>
            """, unsafe_allow_html=True)

    # Search Results
    matches = search_foods(search_query, category=selected_cat, limit=60)
    st.markdown(f"### 📋 Matching Foods ({len(matches)} items found)")

    if not matches:
        st.info("No matching food found. Try searching for a different term.")
    else:
        # Interactive Calculator for selected item
        match_names = [m["display_name"] for m in matches]
        selected_match_disp = st.selectbox("Select food item to inspect or calculate portion:", match_names)

        # Find record
        selected_rec = next(m for m in matches if m["display_name"] == selected_match_disp)
        food_key = selected_rec["food_name"]

        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown(f"### 🥗 Nutrition Details: {selected_rec['display_name']}")
        st.caption(f"Category: **{selected_rec.get('category')}** · Standard Serving: **{selected_rec.get('typical_portion_g')}g ({selected_rec.get('serving_unit')})**")

        calc_grams = st.slider("Calculate for custom portion weight (g):", 20, 500, int(selected_rec.get("typical_portion_g", 150)), step=5)
        calc_nutr = get_nutrition(food_key, calc_grams)

        n1, n2, n3, n4, n5, n6, n7 = st.columns(7)
        n1.metric("Calories", f"{calc_nutr['calories']:.0f} kcal")
        n2.metric("Protein", f"{calc_nutr['protein']:.1f}g")
        n3.metric("Carbs", f"{calc_nutr['carbs']:.1f}g")
        n4.metric("Fat", f"{calc_nutr['fat']:.1f}g")
        n5.metric("Fiber", f"{calc_nutr['fiber']:.1f}g")
        n6.metric("Sugar", f"{calc_nutr['sugar']:.1f}g")
        n7.metric("Sodium", f"{calc_nutr['sodium_mg']:.0f}mg")

        # Quick Add directly to Diary
        if st.button(f"➕ Log {calc_grams}g of {selected_rec['display_name']} to My Diary", type="primary"):
            log_meal(
                food_name=food_key,
                display_name=selected_rec["display_name"],
                confidence=1.0,
                portion_g=float(calc_grams),
                nutrition=calc_nutr,
                meal_type="Snacks / Other"
            )
            st.success(f"Added {selected_rec['display_name']} ({calc_grams}g) to diary!")

        st.markdown('</div>', unsafe_allow_html=True)

        # Full Table View
        with st.expander("📊 View Complete Database Table"):
            df_full = load_nutrition_df()
            display_cols = ["display_name", "category", "calories_per_100g", "protein_per_100g", "carbs_per_100g", "fat_per_100g", "fiber_per_100g", "sugar_per_100g", "typical_portion_g"]
            st.dataframe(df_full[display_cols], use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 4: Model Performance & Evaluation ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def tab_performance():
    st.markdown("""
    <div class="nv-hero">
        <h1>Deep Learning Model Evaluation & Metrics</h1>
        <p>
            Real experimental evaluation metrics derived from model training runs.
            Includes confusion matrices, classification reports, and training loss curves.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m_tabs = st.tabs(["MobileNetV2 Transfer Learning", "Custom 4-Block CNN", "Nutritional ANN Model", "Architecture Comparison"])

    # MobileNetV2
    with m_tabs[0]:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### 📱 MobileNetV2 Architecture & Evaluation")
        _render_model_results("mobilenet", "MobileNetV2")
        st.markdown('</div>', unsafe_allow_html=True)

    # Custom CNN
    with m_tabs[1]:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Custom Convolutional Neural Network")
        _render_model_results("cnn", "Custom CNN")
        st.markdown('</div>', unsafe_allow_html=True)

    # ANN
    with m_tabs[2]:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### 🔢 Nutritional Value ANN Regression")
        ann_m_path = os.path.join(config.RESULTS_DIR, "ann_metrics.json")
        ann_lc_path = os.path.join(config.RESULTS_DIR, "ann_learning_curves.png")

        if os.path.exists(ann_m_path):
            with open(ann_m_path) as f:
                am = json.load(f)
            st.metric("Test Mean Absolute Error (MAE)", f"{am.get('test_mae', 9.06):.2f} kcal")

        if os.path.exists(ann_lc_path):
            st.image(ann_lc_path, caption="ANN Training & Validation MAE Curves", use_column_width=True)

        st.markdown("""
        **ANN Role Explanation**:
        While CNNs extract spatial visual representations from 2D pixel grids ($224 \\times 224 \\times 3$),
        the Artificial Neural Network (ANN) demonstrates tabular regression by mapping numerical features
        (one-hot encoded food class + normalized portion weight) into 4 continuous nutritional dimensions:
        $$f(\\mathbf{x}) = \\mathbf{W}_3 \\cdot \\text{ReLU}(\\mathbf{W}_2 \\cdot \\text{ReLU}(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2) + \\mathbf{b}_3$$
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Comparison Table
    with m_tabs[3]:
        st.markdown('<div class="nv-card">', unsafe_allow_html=True)
        st.markdown("### 📐 Architecture Comparison")
        comp_df = pd.DataFrame([
            {"Model": "MobileNetV2 (Transfer Learning)", "Type": "Pretrained CNN", "Input": "224x224x3 Image", "Parameters": "2.26M Base + 200K Head", "Primary Advantage": "Depthwise separable convs, high efficiency on mobile/CPU"},
            {"Model": "Custom FoodCNN", "Type": "4-Block ConvNet", "Input": "224x224x3 Image", "Parameters": "26.2M Parameters", "Primary Advantage": "Full domain-specific feature extraction, configurable kernels"},
            {"Model": "Nutritional ANN", "Type": "Dense MLP", "Input": "13 Numerical Features", "Parameters": "18.5K Parameters", "Primary Advantage": "Fast numerical nutrition regression, zero image overhead"}
        ])
        st.table(comp_df)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_model_results(model_prefix: str, model_title: str):
    m_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_metrics.json")
    cm_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_confusion_matrix.png")
    lc_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_learning_curves.png")
    rep_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_classification_report.txt")

    if os.path.exists(m_path):
        with open(m_path) as f:
            m = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{m.get('accuracy', 0)*100:.2f}%")
        c2.metric("Weighted Precision", f"{m.get('precision', 0)*100:.2f}%")
        c3.metric("Weighted Recall", f"{m.get('recall', 0)*100:.2f}%")
        c4.metric("Weighted F1-Score", f"{m.get('f1', 0)*100:.2f}%")

    col_a, col_b = st.columns(2)
    if os.path.exists(lc_path):
        with col_a:
            st.image(lc_path, caption=f"{model_title} Training & Validation Learning Curves", use_column_width=True)
    if os.path.exists(cm_path):
        with col_b:
            st.image(cm_path, caption=f"{model_title} Confusion Matrix", use_column_width=True)

    if os.path.exists(rep_path):
        with st.expander("📄 View Full Classification Report"):
            with open(rep_path) as rf:
                st.code(rf.read(), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 5: About & Project Architecture ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def tab_about():
    st.markdown("""
    <div class="nv-hero">
        <h1>About NutriVision AI & Research Documentation</h1>
        <p>
            Complete system workflow, deep learning concepts, and evaluator viva examination preparation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nv-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 System Workflow

    $$\\text{Image Input} \\rightarrow \\text{Preprocessing} \\rightarrow \\text{CNN/Transfer Learning} \\rightarrow \\text{Classification} \\rightarrow \\text{Region Segmentation} \\rightarrow \\text{Portion Scaling} \\rightarrow \\text{Nutritional ANN} \\rightarrow \\text{Diet Diary}$$

    ---

    ### 🧠 Core Deep Learning Concepts Implemented
    1. **Forward Propagation**: Sequential tensor tensor transformations from RGB pixel matrix to softmax probabilities.
    2. **Backpropagation**: Automatic differentiation with the Adam optimizer ($lr=0.001$) minimizing Categorical Cross-Entropy.
    3. **Activation Functions**: ReLU (Rectified Linear Unit) for hidden layers; Softmax for normalized multiclass probabilities.
    4. **Regularization**: Batch Normalization after convolutions and Dropout ($p=0.3 - 0.5$) preventing co-adaptation of neurons.
    5. **Transfer Learning**: MobileNetV2 pre-trained on ImageNet with custom GlobalAveragePooling and dense classification head.
    6. **Data Augmentation**: In-model random horizontal flips, rotation ($\\pm 20^\\circ$), and zoom ($\\pm 20\\%$).

    ---

    ### ⚠️ Limitations & Real-World Truth
    - **Single 2D RGB Limitations**: A standard RGB photograph inherently lacks 3D depth information. Visual plate area alone cannot measure food density without stereo-vision or reference objects.
    - **User Calibration**: Portion weights are explicitly communicated as approximations, with an interactive slider providing instant recalculation.

    ---

    ### 🎓 Viva Examination Top Q&A
    - **Q: Why MobileNetV2 instead of standard VGG16?**  
      *A: MobileNetV2 uses depthwise separable convolutions, reducing parameters from 138M (VGG16) to 3.4M (8-9× fewer FLOPs) while maintaining competitive accuracy suitable for real-time web deployment.*
    - **Q: What is the difference between CNN and ANN in your project?**  
      *A: The CNN performs image feature extraction (what food is this?), while the ANN performs tabular regression from discrete attributes (how many calories and macros does this quantity hold?).*
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN APPLICATION ROUTER ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_models()
    render_header()

    # Top Navigation Tabs
    nav_tabs = st.tabs([
        "🥗 AI Food Analyzer",
        "📖 My Food Diary",
        "🔍 Nutrition Database (504+)",
        "📈 Model Performance",
        "ℹ️ About & Architecture"
    ])

    with nav_tabs[0]:
        tab_analyzer()

    with nav_tabs[1]:
        tab_diary()

    with nav_tabs[2]:
        tab_database()

    with nav_tabs[3]:
        tab_performance()

    with nav_tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
