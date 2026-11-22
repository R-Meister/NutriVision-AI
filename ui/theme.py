"""Shared, deliberately small visual layer for NutriVision AI."""
import re

import streamlit as st

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")

_CSS = """
<style>
:root {
  --muted: #64748B;
  --line: #E2E8F0;
  --accent: #0F766E;
}
[data-testid="stHeader"] {
  display: none;
}
.block-container {
  padding-top: 1.25rem;
  padding-bottom: 3rem;
  padding-left: clamp(1rem, 2.5vw, 2.5rem);
  padding-right: clamp(1rem, 2.5vw, 2.5rem);
  max-width: 100% !important;
  width: 100% !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}
[data-testid="stMain"] {
  margin-left: 0 !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
h1 {
  font-weight: 650;
  letter-spacing: -0.025em;
  margin-bottom: 0.1rem;
}
h2, h3 {
  font-weight: 600;
  letter-spacing: -0.01em;
}
[data-testid="stMetricValue"] {
  font-size: 1.55rem;
}
[data-testid="stMetricLabel"] {
  color: var(--muted);
}
.small {
  color: var(--muted);
  font-size: 0.86rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--line);
}
[data-testid="stProgressBarTrack"] {
  background-color: #DDE3EA !important;
}
[data-testid="stProgressBarTrack"] > div {
  background-color: var(--accent) !important;
}
</style>
"""


def clean_name(value: str) -> str:
    """Remove food emojis from database display names for a calmer UI."""
    return " ".join(_EMOJI.sub("", str(value)).split())



def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def section_title(title: str) -> None:
    st.subheader(title)


def set_flash(message: str) -> None:
    """Store a one-shot success message that survives a rerun."""
    st.session_state["_flash_message"] = message


def show_flash() -> None:
    message = st.session_state.pop("_flash_message", None)
    if message:
        st.success(message)

