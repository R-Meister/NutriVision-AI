"""
app.py — NutriVision AI: Food Image Analyzer & Calorie Counter
Minimal application shell. Page implementations live in `ui/`.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import streamlit as st

import config  # noqa: F401  (ensures directories exist)
from models.inference import get_mobilenet_model, get_cnn_model, get_ann_model
from ui.theme import apply_theme, page_header, show_flash
from ui.analyzer import tab_analyzer
from ui.diary import tab_diary
from ui.database import tab_database
from ui.performance import tab_performance
from ui.about import tab_about

st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load and cache the MobileNet, optional CNN, and nutrition ANN."""
    return get_mobilenet_model(), get_cnn_model(), get_ann_model()


def render_header() -> None:
    page_header(
        "NutriVision AI",
        "Food recognition and nutrition tracking",
    )
    st.caption("12 supported foods · 504 nutrition profiles · 91.1% top-1 accuracy")
    st.divider()


def main() -> None:
    load_models()
    render_header()
    show_flash()

    page = st.segmented_control(
        "Page",
        ["Analyze", "Diary", "Food database", "Model", "About"],
        default="Analyze",
        key="nav_page",
        label_visibility="collapsed",
        width="stretch",
    )
    page = page or "Analyze"

    if page == "Analyze":
        tab_analyzer()
    elif page == "Diary":
        tab_diary()
    elif page == "Food database":
        tab_database()
    elif page == "Model":
        tab_performance()
    elif page == "About":
        tab_about()


if __name__ == "__main__":
    main()
