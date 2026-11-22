"""Food diary page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import page_header, clean_name, section_title
from utils.database import (get_weekly_trend, get_today_log, get_daily_summary,
                            get_daily_goals, set_daily_goals, delete_entry,
                            clear_today)

_ACCENT = "#0F766E"


def _as_float(value, default: float = 0.0) -> float:
    """Coerce a DB value to float, tolerating None and bad data."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(consumed: float, goal: float) -> float:
    """Progress in [0, 1]; never divides by zero."""
    if goal <= 0:
        return 1.0 if consumed > 0 else 0.0
    return min(1.0, max(0.0, consumed / goal))


def _macro_progress(label: str, consumed: float, goal: float, unit: str) -> None:
    pct = _ratio(consumed, goal)
    st.caption(f"{label} · {consumed:.0f} / {goal:.0f} {unit} ({pct * 100:.0f}%)")
    st.progress(pct)


def _trend_chart(trend_data: list[dict], goal_cal: float) -> go.Figure:
    df = pd.DataFrame(trend_data)
    labels = df["label"] if not df.empty else []
    calories = df["calories"] if not df.empty else []
    # Highlight the most recent day (today) with the accent colour.
    colors = [_ACCENT if i == len(df) - 1 else "#CBD5E1" for i in range(len(df))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=calories,
        marker_color=colors,
        text=[f"{v:.0f}" if v else "" for v in calories],
        textposition="outside",
        textfont=dict(size=11, color="#64748B"),
        hovertemplate="%{x}<br>%{y:.0f} kcal<extra></extra>",
        name="Calories",
    ))
    fig.add_hline(
        y=goal_cal,
        line_dash="dot",
        line_width=1,
        line_color="#94A3B8",
        annotation_text=f"Goal {goal_cal:.0f}",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#64748B"),
    )
    fig.update_layout(
        height=300,
        margin=dict(t=24, b=8, l=8, r=8),
        bargap=0.45,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", size=12),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False,
            rangemode="tozero",
            tickfont=dict(size=11),
        ),
    )
    return fig


def tab_diary():
    page_header("Diary", "Daily calorie budget, macro targets, and logged meals.")

    today_summary = get_daily_summary()
    daily_goals = get_daily_goals()

    goal_cal = _as_float(daily_goals["calorie_goal"])
    consumed_cal = _as_float(today_summary["calories"])
    remaining_cal = goal_cal - consumed_cal
    cal_pct = _ratio(consumed_cal, goal_cal)

    # ── Daily budget ──────────────────────────────────────────────────────────
    with st.container(border=True):
        section_title("Calorie budget")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Goal", f"{goal_cal:.0f} kcal")
        b2.metric("Consumed", f"{consumed_cal:.0f} kcal")
        b3.metric("Remaining", f"{remaining_cal:.0f} kcal")
        b4.metric("Logged items", f"{int(today_summary.get('entries') or 0)}")

        if remaining_cal < 0:
            st.caption(f"Over goal by {abs(remaining_cal):.0f} kcal")
        st.progress(cal_pct, text=f"{consumed_cal:.0f} of {goal_cal:.0f} kcal · {cal_pct * 100:.0f}%")

    # ── Goals ─────────────────────────────────────────────────────────────────
    with st.expander("Goals", key="diary_goals_expander"):
        with st.form("diary_goals_form", border=False):
            g1, g2, g3 = st.columns(3)
            new_cal = g1.number_input(
                "Calories (kcal)", min_value=0.0, step=50.0,
                value=_as_float(daily_goals["calorie_goal"]), key="goal_calories",
            )
            new_prot = g2.number_input(
                "Protein (g)", min_value=0.0, step=5.0,
                value=_as_float(daily_goals["protein_goal"]), key="goal_protein",
            )
            new_carb = g3.number_input(
                "Carbs (g)", min_value=0.0, step=10.0,
                value=_as_float(daily_goals["carbs_goal"]), key="goal_carbs",
            )
            g4, g5, _ = st.columns(3)
            new_fat = g4.number_input(
                "Fat (g)", min_value=0.0, step=5.0,
                value=_as_float(daily_goals["fat_goal"]), key="goal_fat",
            )
            new_fiber = g5.number_input(
                "Fiber (g)", min_value=0.0, step=5.0,
                value=_as_float(daily_goals["fiber_goal"]), key="goal_fiber",
            )

            if st.form_submit_button("Save goals", type="primary"):
                set_daily_goals(new_cal, new_prot, new_carb, new_fat, new_fiber)
                st.rerun()

    # ── Macro targets ─────────────────────────────────────────────────────────
    with st.container(border=True):
        section_title("Macro targets")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _macro_progress("Protein", _as_float(today_summary["protein"]),
                            _as_float(daily_goals["protein_goal"]), "g")
        with m2:
            _macro_progress("Carbs", _as_float(today_summary["carbs"]),
                            _as_float(daily_goals["carbs_goal"]), "g")
        with m3:
            _macro_progress("Fat", _as_float(today_summary["fat"]),
                            _as_float(daily_goals["fat_goal"]), "g")
        with m4:
            _macro_progress("Fiber", _as_float(today_summary["fiber"]),
                            _as_float(daily_goals["fiber_goal"]), "g")

    # ── Today's meals ─────────────────────────────────────────────────────────
    section_title("Today")
    today_entries = get_today_log()

    if not today_entries:
        st.caption("No meals logged today. Analyze a meal to add it to your diary.")
    else:
        for entry in today_entries:
            entry_id = entry.get("id")
            time_str = str(entry.get("timestamp", ""))[:16].replace("T", " ")
            cals = _as_float(entry.get("calories"))
            prot = _as_float(entry.get("protein"))
            carb = _as_float(entry.get("carbs"))
            fat = _as_float(entry.get("fat"))
            fib = _as_float(entry.get("fiber"))
            sug = _as_float(entry.get("sugar"))
            portion = _as_float(entry.get("portion_g"))
            conf = _as_float(entry.get("confidence"))
            mtype = entry.get("meal_type") or "Meal"
            dname = clean_name(entry.get("display_name") or entry.get("food_name") or "Food")

            with st.expander(f"{mtype} · {dname} — {cals:.0f} kcal"):
                ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
                ec1.metric("Calories", f"{cals:.0f}")
                ec2.metric("Protein", f"{prot:.1f} g")
                ec3.metric("Carbs", f"{carb:.1f} g")
                ec4.metric("Fat", f"{fat:.1f} g")
                ec5.metric("Fiber", f"{fib:.1f} g")
                ec6.metric("Sugar", f"{sug:.1f} g")

                st.caption(f"Logged {time_str} · {portion:.0f} g portion · confidence {conf * 100:.1f}%")

                if st.button("Delete meal", key=f"diary_del_{entry_id}"):
                    delete_entry(entry_id)
                    st.rerun()

    # ── Trend ─────────────────────────────────────────────────────────────────
    section_title("Last 7 days")
    trend_data = get_weekly_trend(7)
    st.plotly_chart(
        _trend_chart(trend_data, goal_cal),
        width="stretch",
        config={"displayModeBar": False, "staticPlot": False},
    )
    week_total = sum(_as_float(day.get("calories")) for day in trend_data)
    active_days = sum(1 for day in trend_data if _as_float(day.get("calories")) > 0)
    st.caption(f"{week_total:.0f} kcal logged across {active_days} of 7 days · "
               f"average {week_total / 7:.0f} kcal/day")

    # ── Footer actions ────────────────────────────────────────────────────────
    st.divider()
    if today_entries:
        st.button("Clear today's log", key="diary_clear_today", on_click=clear_today)
