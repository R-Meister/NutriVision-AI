"""
nutrition_database.py
Comprehensive nutritional database containing 504+ verified food items
based on USDA FoodData Central and Indian Food Composition Tables (IFCT).
Provides fast querying, portion scaling, macronutrient & micronutrient calculation.
"""
import pandas as pd
import numpy as np
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

JSON_DB_PATH = os.path.join(config.DATA_DIR, "nutrition_database.json")
CSV_DB_PATH  = config.NUTRITION_CSV

# Load or initialize database cache
_NUTRITION_CACHE = None

def get_database() -> dict:
    """Load and return master food dictionary."""
    global _NUTRITION_CACHE
    if _NUTRITION_CACHE is not None:
        return _NUTRITION_CACHE

    if os.path.exists(JSON_DB_PATH):
        try:
            with open(JSON_DB_PATH, "r", encoding="utf-8") as f:
                _NUTRITION_CACHE = json.load(f)
                return _NUTRITION_CACHE
        except Exception as e:
            print(f"[nutrition_database] Error loading json: {e}")

    # Fallback to CSV if json fails
    if os.path.exists(CSV_DB_PATH):
        try:
            df = pd.read_csv(CSV_DB_PATH)
            db = {}
            for _, row in df.iterrows():
                db[row["food_name"]] = row.to_dict()
            _NUTRITION_CACHE = db
            return _NUTRITION_CACHE
        except Exception as e:
            print(f"[nutrition_database] Error loading csv: {e}")

    # Built-in fallback
    _NUTRITION_CACHE = {
        "pizza": {"food_name": "pizza", "display_name": "🍕 Cheese Pizza", "category": "Western & Fast Food", "calories_per_100g": 266, "protein_per_100g": 11, "carbs_per_100g": 33, "fat_per_100g": 10, "fiber_per_100g": 2.3, "sugar_per_100g": 3.6, "sodium_mg_per_100g": 598, "serving_unit": "slice", "typical_portion_g": 125},
        "biryani": {"food_name": "biryani", "display_name": "🍛 Fragrant Biryani", "category": "Indian Breads & Rice", "calories_per_100g": 198, "protein_per_100g": 8, "carbs_per_100g": 28, "fat_per_100g": 7, "fiber_per_100g": 1.8, "sugar_per_100g": 1.0, "sodium_mg_per_100g": 420, "serving_unit": "plate", "typical_portion_g": 250},
        "burger": {"food_name": "burger", "display_name": "🍔 Classic Burger", "category": "Western & Fast Food", "calories_per_100g": 295, "protein_per_100g": 17, "carbs_per_100g": 24, "fat_per_100g": 14, "fiber_per_100g": 1.5, "sugar_per_100g": 4.2, "sodium_mg_per_100g": 490, "serving_unit": "burger", "typical_portion_g": 180},
    }
    return _NUTRITION_CACHE


def normalize_food_name(food_name: str) -> str:
    """Normalize input food name to matching key."""
    if not food_name:
        return "pizza"
    clean = str(food_name).lower().strip().replace(" ", "_").replace("-", "_")
    db = get_database()
    if clean in db:
        return clean
    # Try fuzzy / substring match
    for k in db.keys():
        if clean in k or k in clean:
            return k
    return clean


def get_food_record(food_name: str) -> dict:
    """Return raw per-100g food record."""
    db = get_database()
    key = normalize_food_name(food_name)
    if key in db:
        return db[key]
    # Return default generic profile
    return {
        "food_name": food_name,
        "display_name": f"🍽️ {food_name.replace('_', ' ').title()}",
        "category": "General Food",
        "calories_per_100g": 180.0,
        "protein_per_100g": 6.0,
        "carbs_per_100g": 24.0,
        "fat_per_100g": 6.0,
        "fiber_per_100g": 2.0,
        "sugar_per_100g": 2.0,
        "sodium_mg_per_100g": 300.0,
        "serving_unit": "serving",
        "typical_portion_g": 150
    }


def get_nutrition(food_name: str, portion_grams: float) -> dict:
    """
    Return complete scaled nutritional profile for given portion weight.

    Formula
    -------
    Nutrient = (portion_grams / 100.0) * nutrient_per_100g
    """
    rec = get_food_record(food_name)
    factor = float(portion_grams) / 100.0

    return {
        "food_name":     rec.get("food_name", food_name),
        "display_name":  rec.get("display_name", food_name.title()),
        "category":      rec.get("category", "General"),
        "portion_grams": round(float(portion_grams), 1),
        "serving_unit":  rec.get("serving_unit", "g"),
        "calories":      round(rec.get("calories_per_100g", 180) * factor, 1),
        "protein":       round(rec.get("protein_per_100g", 6) * factor, 1),
        "carbs":         round(rec.get("carbs_per_100g", 24) * factor, 1),
        "fat":           round(rec.get("fat_per_100g", 6) * factor, 1),
        "fiber":         round(rec.get("fiber_per_100g", 2) * factor, 1),
        "sugar":         round(rec.get("sugar_per_100g", 2) * factor, 1),
        "sodium_mg":     round(rec.get("sodium_mg_per_100g", 300) * factor, 1),
    }


def get_typical_portion(food_name: str) -> int:
    """Return typical serving size in grams."""
    rec = get_food_record(food_name)
    return int(rec.get("typical_portion_g", config.DEFAULT_PORTION))


def get_display_name(food_name: str) -> str:
    """Return formatted emoji display title."""
    rec = get_food_record(food_name)
    return rec.get("display_name", food_name.replace("_", " ").title())


def get_all_categories() -> list[str]:
    """Return sorted unique food categories."""
    db = get_database()
    cats = sorted(list(set(v.get("category", "General") for v in db.values())))
    return cats


def search_foods(query: str = "", category: str = "All", limit: int = 50) -> list[dict]:
    """Search food database by query and category."""
    db = get_database()
    results = []
    q = query.lower().strip() if query else ""

    for rec in db.values():
        cat_match = (category == "All") or (rec.get("category") == category)
        if not cat_match:
            continue

        if not q:
            results.append(rec)
        else:
            name = rec.get("food_name", "").lower()
            disp = rec.get("display_name", "").lower()
            if q in name or q in disp:
                results.append(rec)

        if len(results) >= limit:
            break

    return results


def load_nutrition_df() -> pd.DataFrame:
    """Return complete database as a Pandas DataFrame."""
    db = get_database()
    return pd.DataFrame(list(db.values()))


if __name__ == "__main__":
    db = get_database()
    print(f"Loaded {len(db)} food items.")
    print("Sample nutrition for 250g Chicken Biryani:")
    print(get_nutrition("chicken_biryani", 250))
