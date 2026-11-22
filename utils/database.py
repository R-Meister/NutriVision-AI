"""
database.py
SQLite-based food diary and daily nutritional tracker.
Supports multi-item meals, macro goals, and meal-type categorisation.
"""
import sqlite3, json, os, sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _connect():
    return sqlite3.connect(config.DB_PATH)


def init_db():
    """Initialize SQLite tables for meals, individual items, and daily goals."""
    conn = _connect()
    cur  = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS food_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            meal_type    TEXT    DEFAULT 'Meal',
            food_name    TEXT    NOT NULL,
            display_name TEXT,
            confidence   REAL,
            portion_g    REAL,
            calories     REAL,
            protein      REAL,
            carbs        REAL,
            fat          REAL,
            fiber        REAL    DEFAULT 0,
            sugar        REAL    DEFAULT 0,
            sodium_mg    REAL    DEFAULT 0,
            items_json   TEXT,
            top3_json    TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_goal (
            date         TEXT PRIMARY KEY,
            calorie_goal REAL DEFAULT 2000,
            protein_goal REAL DEFAULT 75,
            carbs_goal   REAL DEFAULT 250,
            fat_goal     REAL DEFAULT 65,
            fiber_goal   REAL DEFAULT 30
        );
    """)
    conn.commit()

    # Migration: check and add any missing columns in existing database
    cur.execute("PRAGMA table_info(food_log)")
    existing_cols = [row[1] for row in cur.fetchall()]
    new_cols = {
        "meal_type": "TEXT DEFAULT 'Meal'",
        "fiber": "REAL DEFAULT 0",
        "sugar": "REAL DEFAULT 0",
        "sodium_mg": "REAL DEFAULT 0",
        "items_json": "TEXT"
    }
    for col_name, col_def in new_cols.items():
        if col_name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE food_log ADD COLUMN {col_name} {col_def}")
                conn.commit()
            except Exception as e:
                print(f"[db migration food_log] {col_name}: {e}")

    # Migration for daily_goal table
    cur.execute("PRAGMA table_info(daily_goal)")
    existing_goal_cols = [row[1] for row in cur.fetchall()]
    new_goal_cols = {
        "protein_goal": "REAL DEFAULT 75",
        "carbs_goal": "REAL DEFAULT 250",
        "fat_goal": "REAL DEFAULT 65",
        "fiber_goal": "REAL DEFAULT 30"
    }
    for col_name, col_def in new_goal_cols.items():
        if col_name not in existing_goal_cols:
            try:
                cur.execute(f"ALTER TABLE daily_goal ADD COLUMN {col_name} {col_def}")
                conn.commit()
            except Exception as e:
                print(f"[db migration daily_goal] {col_name}: {e}")
    conn.close()


def log_meal(food_name: str, display_name: str, confidence: float,
             portion_g: float, nutrition: dict, items: list = None,
             meal_type: str = "Meal", top3: list = None):
    """Insert a logged meal into SQLite database."""
    conn = _connect()
    top3_json = json.dumps(top3 or [])
    items_json = json.dumps(items or [{
        "food_name": food_name,
        "display_name": display_name,
        "portion_g": portion_g,
        "calories": nutrition.get("calories", 0),
        "protein": nutrition.get("protein", 0),
        "carbs": nutrition.get("carbs", 0),
        "fat": nutrition.get("fat", 0),
        "fiber": nutrition.get("fiber", 0),
        "sugar": nutrition.get("sugar", 0),
        "sodium_mg": nutrition.get("sodium_mg", 0),
    }])

    conn.execute(
        """INSERT INTO food_log
           (timestamp, meal_type, food_name, display_name, confidence, portion_g,
            calories, protein, carbs, fat, fiber, sugar, sodium_mg, items_json, top3_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(),
            meal_type,
            food_name,
            display_name,
            float(confidence),
            float(portion_g),
            float(nutrition.get("calories", 0)),
            float(nutrition.get("protein", 0)),
            float(nutrition.get("carbs", 0)),
            float(nutrition.get("fat", 0)),
            float(nutrition.get("fiber", 0)),
            float(nutrition.get("sugar", 0)),
            float(nutrition.get("sodium_mg", 0)),
            items_json,
            top3_json
        )
    )
    conn.commit()
    conn.close()


def get_today_log() -> list[dict]:
    """Return all entries logged today."""
    today = date.today().isoformat()
    conn  = _connect()
    cur   = conn.cursor()
    cur.execute(
        "SELECT * FROM food_log WHERE timestamp LIKE ? ORDER BY id DESC",
        (f"{today}%",)
    )
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "timestamp", "meal_type", "food_name", "display_name", "confidence",
            "portion_g", "calories", "protein", "carbs", "fat", "fiber", "sugar",
            "sodium_mg", "items_json", "top3_json"]
    return [dict(zip(cols, r)) for r in rows]


def get_history(limit: int = 50) -> list[dict]:
    """Return most recent entries."""
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM food_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "timestamp", "meal_type", "food_name", "display_name", "confidence",
            "portion_g", "calories", "protein", "carbs", "fat", "fiber", "sugar",
            "sodium_mg", "items_json", "top3_json"]
    return [dict(zip(cols, r)) for r in rows]


def get_daily_summary(target_date: str = None) -> dict:
    """Return total macros for given date (defaults to today)."""
    if target_date is None:
        target_date = date.today().isoformat()
    conn = _connect()
    cur  = conn.cursor()
    cur.execute(
        """SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat),
                  SUM(fiber), SUM(sugar), SUM(sodium_mg), COUNT(*)
           FROM food_log WHERE timestamp LIKE ?""",
        (f"{target_date}%",)
    )
    row = cur.fetchone()
    conn.close()
    return {
        "date":      target_date,
        "calories":  round(row[0] or 0.0, 1),
        "protein":   round(row[1] or 0.0, 1),
        "carbs":     round(row[2] or 0.0, 1),
        "fat":       round(row[3] or 0.0, 1),
        "fiber":     round(row[4] or 0.0, 1),
        "sugar":     round(row[5] or 0.0, 1),
        "sodium_mg": round(row[6] or 0.0, 1),
        "entries":   int(row[7] or 0),
    }


def set_daily_goals(calories: float = 2000, protein: float = 75,
                     carbs: float = 250, fat: float = 65, fiber: float = 30,
                     target_date: str = None):
    """Set custom nutritional targets."""
    if target_date is None:
        target_date = date.today().isoformat()
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO daily_goal
           (date, calorie_goal, protein_goal, carbs_goal, fat_goal, fiber_goal)
           VALUES (?,?,?,?,?,?)""",
        (target_date, calories, protein, carbs, fat, fiber)
    )
    conn.commit()
    conn.close()


def get_daily_goals(target_date: str = None) -> dict:
    """Get active targets for target_date."""
    if target_date is None:
        target_date = date.today().isoformat()
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("SELECT calorie_goal, protein_goal, carbs_goal, fat_goal, fiber_goal FROM daily_goal WHERE date=?", (target_date,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "calorie_goal": float(row[0]),
            "protein_goal": float(row[1]),
            "carbs_goal":   float(row[2]),
            "fat_goal":     float(row[3]),
            "fiber_goal":   float(row[4]),
        }
    return {
        "calorie_goal": 2000.0,
        "protein_goal": 75.0,
        "carbs_goal":   250.0,
        "fat_goal":     65.0,
        "fiber_goal":   30.0,
    }


def delete_entry(entry_id: int):
    """Delete a log entry by id."""
    conn = _connect()
    conn.execute("DELETE FROM food_log WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()


def clear_today():
    """Clear today's logs."""
    today = date.today().isoformat()
    conn  = _connect()
    conn.execute("DELETE FROM food_log WHERE timestamp LIKE ?", (f"{today}%",))
    conn.commit()
    conn.close()


def get_weekly_trend(days: int = 7) -> list[dict]:
    """Return day-by-day calories for the past n days."""
    results = []
    for i in range(days - 1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        s = get_daily_summary(d)
        results.append({
            "date": d,
            "label": (date.today() - timedelta(days=i)).strftime("%a, %b %d"),
            "calories": s["calories"],
            "protein": s["protein"],
            "carbs": s["carbs"],
            "fat": s["fat"],
            "entries": s["entries"]
        })
    return results


# Initialize database upon import
init_db()
