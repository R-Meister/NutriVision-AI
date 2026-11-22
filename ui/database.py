"""Nutrition database page."""
import streamlit as st

from ui.theme import page_header, section_title, set_flash, clean_name
from utils.nutrition_database import (
    get_all_categories,
    get_nutrition,
    load_nutrition_df,
    search_foods,
)
from utils.database import log_meal

POPULAR_FOODS = [
    "chicken_biryani", "pizza", "masala_dosa", "burger",
    "butter_chicken", "samosa", "apple", "banana",
]

ALL = "All"


def tab_database():
    page_header(
        "Food database",
        "Search verified foods, scale a portion, and log it to your diary.",
    )

    # Search filters
    with st.container(border=True):
        s1, s2 = st.columns([2, 1], vertical_alignment="bottom")
        with s1:
            query = st.text_input(
                "Food name",
                placeholder="e.g. biryani, paneer, apple, dosa, oats",
            )
        with s2:
            categories = [ALL] + get_all_categories()
            selected_cat = st.selectbox("Category", categories)

    matches = search_foods(query, category=selected_cat, limit=60)

    # Results
    section_title("Results")
    if not matches:
        st.info("No matching food found. Try a different term or category.")
    else:
        st.caption(f"{len(matches)} matching item{'s' if len(matches) != 1 else ''}")

        # options are (food_name, display_name) so duplicate display names still map correctly
        options = [(m["food_name"], clean_name(m["display_name"])) for m in matches]
        choice = st.selectbox(
            "Inspect food",
            options,
            format_func=lambda o: o[1],
        )
        food_key = choice[0]
        selected_rec = next(m for m in matches if m["food_name"] == food_key)
        typical = int(selected_rec.get("typical_portion_g", 150))
        typical = max(20, min(500, typical))

        with st.container(border=True):
            st.markdown(f"**{clean_name(selected_rec['display_name'])}**")
            st.caption(
                f"{selected_rec.get('category', 'General')} · "
                f"typical serving {selected_rec.get('typical_portion_g')} g "
                f"({selected_rec.get('serving_unit', 'serving')})"
            )

            grams = st.slider(
                "Portion size (g)",
                min_value=20,
                max_value=500,
                value=typical,
                step=5,
            )
            calc = get_nutrition(food_key, grams)

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Calories", f"{calc['calories']:.0f} kcal")
            n2.metric("Protein", f"{calc['protein']:.1f} g")
            n3.metric("Carbs", f"{calc['carbs']:.1f} g")
            n4.metric("Fat", f"{calc['fat']:.1f} g")

            n5, n6, n7, _ = st.columns(4)
            n5.metric("Fiber", f"{calc['fiber']:.1f} g")
            n6.metric("Sugar", f"{calc['sugar']:.1f} g")
            n7.metric("Sodium", f"{calc['sodium_mg']:.0f} mg")

            st.divider()
            if st.button(
                f"Log {grams} g to diary",
                type="primary",
                key=f"db_log_{food_key}",
            ):
                log_meal(
                    food_name=food_key,
                    display_name=selected_rec["display_name"],
                    confidence=1.0,
                    portion_g=float(grams),
                    nutrition=calc,
                    meal_type="Snacks / Other",
                )
                set_flash(
                    f"Added {clean_name(selected_rec['display_name'])} ({grams} g) to your diary."
                )
                st.rerun()

    # Popular items (compact)
    section_title("Popular")
    popular_rows = []
    for pk in POPULAR_FOODS:
        rec = get_nutrition(pk, 100)
        popular_rows.append(
            {
                "Food": clean_name(rec["display_name"]),
                "Calories / 100 g": round(rec["calories"]),
                "Protein (g)": rec["protein"],
                "Carbs (g)": rec["carbs"],
                "Fat (g)": rec["fat"],
            }
        )
    st.dataframe(popular_rows, hide_index=True, width="stretch")
    st.caption("Search for any food above to inspect it at a custom portion.")

    # Full database
    with st.expander("Full database"):
        df_full = load_nutrition_df()
        df_full["display_name"] = df_full["display_name"].map(clean_name)
        display_cols = [
            "display_name",
            "category",
            "calories_per_100g",
            "protein_per_100g",
            "carbs_per_100g",
            "fat_per_100g",
            "fiber_per_100g",
            "sugar_per_100g",
            "typical_portion_g",
        ]
        st.dataframe(
            df_full[display_cols].rename(
                columns={
                    "display_name": "Food",
                    "category": "Category",
                    "calories_per_100g": "Calories",
                    "protein_per_100g": "Protein (g)",
                    "carbs_per_100g": "Carbs (g)",
                    "fat_per_100g": "Fat (g)",
                    "fiber_per_100g": "Fiber (g)",
                    "sugar_per_100g": "Sugar (g)",
                    "typical_portion_g": "Typical portion (g)",
                }
            ),
            hide_index=True,
            width="stretch",
            height=400,
        )
