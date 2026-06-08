import os
import datetime
from typing import Optional
import httpx
import libsql
import mimetypes
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------
# FastAPI App & Global Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="NutriBot Webhook API",
    description="Stateless food photo analysis Telegram Bot backend powered by Gemini Flash and Turso in Khmer.",
    version="1.4.0"
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

miniapp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "miniapp")
app.mount("/miniapp", StaticFiles(directory=miniapp_dir, html=True), name="miniapp")


# ---------------------------------------------------------
# Pydantic Schemas for Gemini Structured Output
# ---------------------------------------------------------
class FoodAnalysis(BaseModel):
    food_name: str = Field(description="The primary name or description of the identified food dish in Khmer language.")
    calories: int = Field(description="Estimated calories in kilocalories (Cal).")
    protein: int = Field(description="Estimated protein in grams (g).")
    fat: int = Field(description="Estimated total fat in grams (g).")
    carbs: int = Field(description="Estimated total carbohydrates in grams (g).")
    sugar: int = Field(description="Estimated sugar content in grams (g).")
    confidence_score: float = Field(description="Model confidence from 0.0 (not food/unknown) to 1.0 (highly confident food).")
    coaching_recommendation: str = Field(description="A highly personalized, actionable health/coaching recommendation in Khmer language tailored specifically to this user's profile and goal (e.g., protein density, health tips, fullness, weight loss suitability).")


# ---------------------------------------------------------
# Database Utility & Connection Wrapper (Turso SQLite)
# ---------------------------------------------------------
def get_db_connection() -> libsql.Connection:
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        raise ValueError("Missing database environment variables: TURSO_DATABASE_URL or TURSO_AUTH_TOKEN.")
    return libsql.connect(database=url, auth_token=auth_token)

def db_initialize_schema():
    """Bootstraps/Updates database tables and schema version checks."""
    try:
        url = os.getenv("TURSO_DATABASE_URL")
        auth_token = os.getenv("TURSO_AUTH_TOKEN")
        if url and auth_token:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
                daily_calorie_goal INTEGER DEFAULT 2000
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                meal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                food_name TEXT,
                calories INTEGER,
                protein INTEGER,
                fat INTEGER,
                carbs INTEGER,
                sugar INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                reminder_time TEXT, -- HH:MM in ICT (UTC+7)
                FOREIGN KEY(user_id) REFERENCES users(user_id),
                UNIQUE(user_id, reminder_time)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tdee_states (
                user_id INTEGER PRIMARY KEY,
                step TEXT,
                gender TEXT,
                age INTEGER,
                height REAL,
                weight REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS burn_logs (
                burn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                calories_burned INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS nosweet_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS google_fit_tokens (
                user_id INTEGER PRIMARY KEY,
                access_token TEXT,
                refresh_token TEXT,
                expires_at REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strava_tokens (
                user_id INTEGER PRIMARY KEY,
                access_token TEXT,
                refresh_token TEXT,
                expires_at REAL,
                athlete_id INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)
            try:
                cursor.execute("ALTER TABLE strava_tokens ADD COLUMN athlete_id INTEGER;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN daily_calorie_budget INTEGER DEFAULT 2000;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN goal_type TEXT DEFAULT 'maintain';")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN height REAL;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN weight REAL;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN activity TEXT;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE burn_logs ADD COLUMN activity_name TEXT DEFAULT 'Manual';")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE burn_logs ADD COLUMN source TEXT DEFAULT 'Manual';")
            except Exception:
                pass
            conn.commit()
            conn.close()
            print("🚀 Turso SQLite schemas auto-initialized successfully!")
    except Exception as e:
        print(f"⚠️ Database auto-initialization failed: {e}")

@app.on_event("startup")
def startup_event():
    """Skip schema initialization on startup to optimize cold start performance."""
    print("🚀 FastAPI startup completed. Schema initialization deferred.")

def db_register_user(user_id: int):
    """Ensures a user exists in the users table with a default goal of 2000 Cal."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, daily_calorie_goal) VALUES (?, 2000)",
                (user_id,)
            )
            conn.commit()
    except Exception as e:
        print(f"Error registering user {user_id}: {e}")

def db_get_user_goal(user_id: int) -> int:
    """Gets user's calorie goal, defaulting to 2000 if user or goal not found."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_calorie_goal FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as e:
        print(f"Error retrieving goal for user {user_id}: {e}")
    return 2000

def db_update_user_goal(user_id: int, goal: int):
    """Updates the user's daily calorie goal."""
    db_register_user(user_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET daily_calorie_goal = ? WHERE user_id = ?", (goal, user_id))
        conn.commit()

def parse_custom_date(token: str) -> str | None:
    """Parses a relative or absolute date token and returns YYYY-MM-DD format (Cambodia local time)."""
    token = token.strip().lower()
    import datetime
    import re
    # Cambodia local time is UTC+7
    now_cambodia = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    
    if token in ["yesterday", "ម្សិលមិញ"]:
        target_date = now_cambodia - datetime.timedelta(days=1)
        return target_date.strftime("%Y-%m-%d")
    elif token in ["ម្សិលម្ងៃ", "ម្សិលម្ង៉ៃ", "ម្សិលមិញមួយថ្ងៃ"]:
        target_date = now_cambodia - datetime.timedelta(days=2)
        return target_date.strftime("%Y-%m-%d")
    elif token in ["today", "ថ្ងៃនេះ"]:
        return now_cambodia.strftime("%Y-%m-%d")
        
    # Check DD-MM-YYYY or DD/MM/YYYY
    m1 = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$", token)
    if m1:
        day, month, year = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        if year < 100:
            year += 2000
        try:
            dt = datetime.date(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
            
    # Check YYYY-MM-DD or YYYY/MM/DD
    m2 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", token)
    if m2:
        year, month, day = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            dt = datetime.date(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
            
    return None

def db_add_meal(user_id: int, analysis: FoodAnalysis, custom_date: str = None) -> int:
    """Saves analyzed meal data into Turso and returns the inserted meal_id."""
    db_register_user(user_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if custom_date:
            timestamp = f"{custom_date} 05:00:00"
            cursor.execute(
                """
                INSERT INTO meals (user_id, food_name, calories, protein, fat, carbs, sugar, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    analysis.food_name,
                    analysis.calories,
                    analysis.protein,
                    analysis.fat,
                    analysis.carbs,
                    analysis.sugar,
                    timestamp
                )
            )
        else:
            cursor.execute(
                """
                INSERT INTO meals (user_id, food_name, calories, protein, fat, carbs, sugar)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    analysis.food_name,
                    analysis.calories,
                    analysis.protein,
                    analysis.fat,
                    analysis.carbs,
                    analysis.sugar
                )
            )
        conn.commit()
        return cursor.lastrowid

def db_delete_meal(user_id: int, meal_id: int):
    """Deletes a specific meal log for a user ensuring strict ownership verification."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM meals WHERE meal_id = ? AND user_id = ?",
            (meal_id, user_id)
        )
        conn.commit()

def db_delete_today_meals(user_id: int):
    """Deletes all meals and burn logs logged today (UTC date) for a specific user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM meals WHERE user_id = ? AND date(timestamp, '+7 hours') = date('now', '+7 hours')",
            (user_id,)
        )
        cursor.execute(
            "DELETE FROM burn_logs WHERE user_id = ? AND date(timestamp, '+7 hours') = date('now', '+7 hours')",
            (user_id,)
        )
        conn.commit()

def db_add_burn(user_id: int, calories: int, activity_name: str = 'Manual', source: str = 'Manual', custom_date: str = None) -> int:
    """Saves calories burned into Turso and returns the inserted burn_id."""
    db_register_user(user_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if custom_date:
            timestamp = f"{custom_date} 05:00:00"
            cursor.execute(
                "INSERT INTO burn_logs (user_id, calories_burned, activity_name, source, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, calories, activity_name, source, timestamp)
            )
        else:
            cursor.execute(
                "INSERT INTO burn_logs (user_id, calories_burned, activity_name, source) VALUES (?, ?, ?, ?)",
                (user_id, calories, activity_name, source)
            )
        conn.commit()
        return cursor.lastrowid

def db_save_fit_tokens(user_id: int, access_token: str, refresh_token: str, expires_in: int):
    """Saves or updates OAuth tokens for a user."""
    import time
    expires_at = time.time() + expires_in
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO google_fit_tokens (user_id, access_token, refresh_token, expires_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (user_id, access_token, refresh_token, expires_at)
        )
        conn.commit()

def db_update_access_token(user_id: int, access_token: str, expires_in: int):
    """Updates access token and expiry time without overwriting the refresh token."""
    import time
    expires_at = time.time() + expires_in
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE google_fit_tokens SET access_token = ?, expires_at = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (access_token, expires_at, user_id)
        )
        conn.commit()

def db_get_fit_tokens(user_id: int) -> dict:
    """Retrieves Google Fit OAuth tokens for a user, or None if not connected."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT access_token, refresh_token, expires_at FROM google_fit_tokens WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "access_token": row[0],
                    "refresh_token": row[1],
                    "expires_at": row[2]
                }
    except Exception as e:
        print(f"Error getting fit tokens for {user_id}: {e}")
    return None

def db_delete_fit_tokens(user_id: int):
    """Deletes Google Fit tokens for a user (disconnecting)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM google_fit_tokens WHERE user_id = ?", (user_id,))
        conn.commit()

def db_save_strava_tokens(user_id: int, access_token: str, refresh_token: str, expires_at: float, athlete_id: int = None):
    """Saves or updates Strava OAuth tokens for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO strava_tokens (user_id, access_token, refresh_token, expires_at, athlete_id, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (user_id, access_token, refresh_token, expires_at, athlete_id)
        )
        conn.commit()

def db_get_user_id_by_strava_athlete(athlete_id: int) -> int:
    """Retrieves user_id by Strava athlete_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM strava_tokens WHERE athlete_id = ?", (athlete_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as e:
        print(f"Error getting user_id for athlete_id {athlete_id}: {e}")
    return None

def db_update_strava_access_token(user_id: int, access_token: str, expires_at: float):
    """Updates Strava access token and expiry time without overwriting the refresh token."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE strava_tokens SET access_token = ?, expires_at = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (access_token, expires_at, user_id)
        )
        conn.commit()

def db_get_strava_tokens(user_id: int) -> dict:
    """Retrieves Strava OAuth tokens for a user, or None if not connected."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT access_token, refresh_token, expires_at FROM strava_tokens WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "access_token": row[0],
                    "refresh_token": row[1],
                    "expires_at": row[2]
                }
    except Exception as e:
        print(f"Error getting strava tokens for {user_id}: {e}")
    return None

def db_delete_strava_tokens(user_id: int):
    """Deletes Strava tokens for a user (disconnecting)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strava_tokens WHERE user_id = ?", (user_id,))
        conn.commit()

def db_get_day_meals(user_id: int, date_str: str) -> tuple[list[dict], int]:
    """Retrieves all meals logged on a specific day (Cambodia ICT date YYYY-MM-DD) for a user."""
    db_register_user(user_id)
    meals = []
    total_calories = 0
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT food_name, calories, protein, fat, carbs, sugar, timestamp
                FROM meals
                WHERE user_id = ? AND date(timestamp, '+7 hours') = ?
                ORDER BY timestamp DESC
                """,
                (user_id, date_str)
            )
            rows = cursor.fetchall()
            for r in rows:
                meals.append({
                    "food_name": r[0],
                    "calories": r[1],
                    "protein": r[2],
                    "fat": r[3],
                    "carbs": r[4],
                    "sugar": r[5],
                    "timestamp": r[6]
                })
                total_calories += r[1]
    except Exception as e:
        print(f"Error getting meals for user {user_id} on {date_str}: {e}")
    return meals, total_calories

def db_get_day_burn(user_id: int, date_str: str) -> int:
    """Aggregates all calories burned on a specific day (Cambodia ICT date YYYY-MM-DD) for a user."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(calories_burned) FROM burn_logs WHERE user_id = ? AND date(timestamp, '+7 hours') = ?",
                (user_id, date_str)
            )
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
    except Exception as e:
        print(f"Error getting burn for user {user_id} on {date_str}: {e}")
    return 0

def db_get_today_burn(user_id: int) -> int:
    """Aggregates all calories burned today (UTC date) for a user."""
    import datetime
    now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    return db_get_day_burn(user_id, now_kh.strftime("%Y-%m-%d"))

def db_get_weekly_stats(user_id: int, start_date_str: str, end_date_str: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Retrieves all meals and burn logs in the given date range (inclusive, shifted to Cambodia ICT timezone)."""
    db_register_user(user_id)
    meals = []
    burns = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Query meals: sum up calories per day in ICT
            cursor.execute(
                """
                SELECT calories, date(timestamp, '+7 hours')
                FROM meals
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                """,
                (user_id, start_date_str, end_date_str)
            )
            meals = cursor.fetchall()
            
            # Query burns: sum up burned calories per day in ICT
            cursor.execute(
                """
                SELECT calories_burned, date(timestamp, '+7 hours')
                FROM burn_logs
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                """,
                (user_id, start_date_str, end_date_str)
            )
            burns = cursor.fetchall()
    except Exception as e:
        print(f"Error retrieving weekly stats for user {user_id}: {e}")
    return meals, burns

def db_add_nosweet_log(user_id: int):
    """Saves a 'no sweet' log for the user. Uses UTC time in database, but acts in ICT (UTC+7) context."""
    db_register_user(user_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO nosweet_logs (user_id) VALUES (?)", (user_id,))
        conn.commit()

def db_check_today_nosweet(user_id: int) -> bool:
    """Checks if the user has already logged a 'no sweet' entry today in Cambodian Time (ICT, UTC+7)."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check using current date in ICT (+7 hours shift)
            cursor.execute(
                """
                SELECT 1 FROM nosweet_logs 
                WHERE user_id = ? AND date(timestamp, '+7 hours') = date('now', '+7 hours')
                LIMIT 1
                """,
                (user_id,)
            )
            return cursor.fetchone() is not None
    except Exception as e:
        print(f"Error checking today's nosweet log for {user_id}: {e}")
    return False

def db_remove_today_nosweet(user_id: int):
    """Deletes today's 'no sweet' log for a user in Cambodian Time (ICT, UTC+7)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM nosweet_logs 
                WHERE user_id = ? AND date(timestamp, '+7 hours') = date('now', '+7 hours')
                """,
                (user_id,)
            )
            conn.commit()
    except Exception as e:
        print(f"Error removing today's no sweet for user {user_id}: {e}")

def db_get_weekly_nosweet(user_id: int, start_date_str: str, end_date_str: str) -> list[str]:
    """Retrieves all dates (YYYY-MM-DD in Cambodia ICT) where the user successfully logged /nosweet in the given date range (inclusive)."""
    db_register_user(user_id)
    dates = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT date(timestamp, '+7 hours')
                FROM nosweet_logs
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                """,
                (user_id, start_date_str, end_date_str)
            )
            rows = cursor.fetchall()
            for r in rows:
                dates.append(r[0])
    except Exception as e:
        print(f"Error retrieving weekly nosweet logs for {user_id}: {e}")
    return dates

def db_get_today_meals(user_id: int) -> tuple[list[dict], int]:
    """Retrieves all meals logged today (UTC date) for a user, returning list and count."""
    import datetime
    now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    return db_get_day_meals(user_id, now_kh.strftime("%Y-%m-%d"))

def db_add_reminder(user_id: int, reminder_time: str):
    """Adds a new reminder for a user. Stored as HH:MM string in ICT (UTC+7)."""
    db_register_user(user_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO reminders (user_id, reminder_time) VALUES (?, ?)",
            (user_id, reminder_time)
        )
        conn.commit()

def db_delete_reminder(user_id: int, reminder_time: str):
    """Deletes a specific reminder for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM reminders WHERE user_id = ? AND reminder_time = ?",
            (user_id, reminder_time)
        )
        conn.commit()

def db_clear_reminders(user_id: int):
    """Clears all reminders for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE user_id = ?", (user_id,))
        conn.commit()

def db_get_reminders(user_id: int) -> list[str]:
    """Retrieves all reminders for a user in chronological order."""
    db_register_user(user_id)
    reminders = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT reminder_time FROM reminders WHERE user_id = ? ORDER BY reminder_time ASC",
                (user_id,)
            )
            rows = cursor.fetchall()
            for r in rows:
                reminders.append(r[0])
    except Exception as e:
        print(f"Error getting reminders for user {user_id}: {e}")
    return reminders

def db_get_active_reminders_for_slot(slot_pattern: str) -> list[int]:
    """Returns a list of user_ids that have reminders matching the slot pattern (e.g. '08:3%')."""
    user_ids = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT user_id FROM reminders WHERE reminder_time LIKE ?",
                (slot_pattern,)
            )
            rows = cursor.fetchall()
            for r in rows:
                user_ids.append(r[0])
    except Exception as e:
        print(f"Error getting active reminders for slot {slot_pattern}: {e}")
    return user_ids

def db_get_tdee_state(user_id: int) -> dict:
    """Retrieves the TDEE state for a user, or None if not found."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT step, gender, age, height, weight FROM tdee_states WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "step": row[0],
                    "gender": row[1],
                    "age": row[2],
                    "height": row[3],
                    "weight": row[4]
                }
    except Exception as e:
        print(f"Error getting TDEE state for user {user_id}: {e}")
    return None

def db_set_tdee_step(user_id: int, step: str, gender: str = None, age: int = None, height: float = None, weight: float = None):
    """Updates or inserts a TDEE state step for a user."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO tdee_states (user_id, step) VALUES (?, ?)", (user_id, step))
            
            updates = ["step = ?"]
            params = [step]
            
            if gender is not None:
                updates.append("gender = ?")
                params.append(gender)
            if age is not None:
                updates.append("age = ?")
                params.append(age)
            if height is not None:
                updates.append("height = ?")
                params.append(height)
            if weight is not None:
                updates.append("weight = ?")
                params.append(weight)
                
            params.append(user_id)
            query = f"UPDATE tdee_states SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?"
            cursor.execute(query, tuple(params))
            conn.commit()
    except Exception as e:
        print(f"Error setting TDEE step for user {user_id}: {e}")

def db_clear_tdee_state(user_id: int):
    """Deletes TDEE state for a user."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tdee_states WHERE user_id = ?", (user_id,))
            conn.commit()
    except Exception as e:
        print(f"Error clearing TDEE state for user {user_id}: {e}")

def db_update_tdee_goal(user_id: int, goal_type: str, calories: int):
    """Updates the user's daily calorie budget and goal type, keeping daily_calorie_goal in sync."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET daily_calorie_budget = ?, goal_type = ?, daily_calorie_goal = ? WHERE user_id = ?",
                (calories, goal_type, calories, user_id)
            )
            conn.commit()
    except Exception as e:
        print(f"Error updating TDEE goal for user {user_id}: {e}")

def db_save_user_profile(user_id: int, gender: str, age: int, height: float, weight: float, activity: str):
    """Saves the user's physical profile metrics in the users table."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET gender = ?, age = ?, height = ?, weight = ?, activity = ? 
                WHERE user_id = ?
                """,
                (gender, age, height, weight, activity, user_id)
            )
            conn.commit()
            print(f"Successfully saved user profile for {user_id}!")
    except Exception as e:
        print(f"Error saving user profile for user {user_id}: {e}")

def db_get_user_profile(user_id: int) -> dict:
    """Retrieves the user's physical profile from the users table, returning None if incomplete."""
    db_register_user(user_id)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT gender, age, height, weight, activity, goal_type, daily_calorie_budget FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row and row[0] is not None:
                return {
                    "gender": row[0],
                    "age": row[1],
                    "height": row[2],
                    "weight": row[3],
                    "activity": row[4],
                    "goal_type": row[5],
                    "daily_calorie_budget": row[6]
                }
    except Exception as e:
        print(f"Error getting user profile for user {user_id}: {e}")
    return None


# ---------------------------------------------------------
# Telegram Bot Helper Class
# ---------------------------------------------------------
class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_url = f"https://api.telegram.org/file/bot{token}"

    async def get_file_bytes(self, file_id: str) -> tuple[bytes, str]:
        """Downloads a file directly from Telegram servers into memory as bytes."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/getFile", params={"file_id": file_id})
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise ValueError(f"Telegram API getFile returned an error: {data}")

            file_path = data["result"]["file_path"]
            
            # Deduce mime type dynamically from extension using standard mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"  # Safe default fallback

            # Stream download file bytes directly
            file_resp = await client.get(f"{self.file_url}/{file_path}")
            file_resp.raise_for_status()
            return file_resp.content, mime_type

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML", reply_to_message_id: int = None, reply_markup: dict = None) -> dict:
        """Sends a text message using the Telegram API."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            if reply_markup:
                payload["reply_markup"] = reply_markup
            
            resp = await client.post(f"{self.base_url}/sendMessage", json=payload)
            return resp.json()

    async def edit_message(self, chat_id: int, message_id: int, text: str, parse_mode: str = "HTML", reply_markup: dict = None) -> dict:
        """Edits an existing text message for fluid, premium real-time updates."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode
            }
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            resp = await client.post(f"{self.base_url}/editMessageText", json=payload)
            return resp.json()

    async def answer_callback_query(self, callback_query_id: str, text: str = None, show_alert: bool = False) -> dict:
        """Answers a callback query to halt loading spinners in Telegram UI."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "callback_query_id": callback_query_id
            }
            if text:
                payload["text"] = text
                payload["show_alert"] = show_alert
            resp = await client.post(f"{self.base_url}/answerCallbackQuery", json=payload)
            return resp.json()

# ---------------------------------------------------------
# Webhook Processing Core Logic
# ---------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a professional nutrition expert. Analyze the food in the provided image and estimate its "
    "nutritional details (calories in Cal, protein/fat/carbs/sugar in grams). "
    "YOU MUST RESPOND ENTIRELY IN KHMER LANGUAGE. The `food_name` field must be written in beautiful Khmer script "
    "(e.g., 'បាយឆាគ្រឿង' or 'ញាំមីស៊ុប'). "
    "If the image does not show any food, or you cannot identify any food, "
    "you MUST set the `confidence_score` to less than 0.5 (e.g. 0.0 to 0.4), "
    "and you can set the `food_name` to 'មិនមែនជាអាហារ ឬរកមិនឃើញ'."
    "Be realistic, objective, and estimate standard portion sizes for single servings unless "
    "there's strong visual context stating otherwise."
)

async def handle_telegram_update(payload: dict):
    """Processes incoming Telegram updates synchronously to fit serverless limits in Khmer."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN is not configured.")
        return

    bot = TelegramBot(bot_token)
    
    # Check for callback queries (e.g. button clicks)
    callback_query = payload.get("callback_query")
    if callback_query:
        callback_id = callback_query["id"]
        chat_id = callback_query["message"]["chat"]["id"]
        message_id = callback_query["message"]["message_id"]
        user_id = callback_query["from"]["id"]
        callback_data = callback_query.get("data", "")
        
        # 1. Handle Delete specific meal log
        if callback_data.startswith("delete_meal:"):
            meal_id = int(callback_data.split(":")[1])
            try:
                db_delete_meal(user_id, meal_id)
                await bot.answer_callback_query(callback_id, "🥗 លុបកំណត់ត្រាអាហារបានជោគជ័យ!")
                
                cleared_card = (
                    "🍳 <b>លទ្ធផលវិភាគអាហារូបត្ថម្ភ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "🗑️ <b>កំណត់ត្រាត្រូវបានលុប!</b> អាហារនេះត្រូវបានលុបចេញពីកំណត់ត្រាថ្ងៃនេះរបស់អ្នក។"
                )
                await bot.edit_message(chat_id, message_id, cleared_card, reply_markup={"inline_keyboard": []})
            except Exception as delete_err:
                print(f"Error deleting meal: {delete_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការលុបកំណត់ត្រាអាហារ។", show_alert=True)
            return

        # 2. Handle Reset all of today's stats
        elif callback_data == "reset_today":
            try:
                db_delete_today_meals(user_id)
                await bot.answer_callback_query(callback_id, "🗑️ បានសម្អាតកំណត់ត្រាថ្ងៃនេះរួចរាល់!")
                
                goal = db_get_user_goal(user_id)
                cleared_stats_text = (
                    "📊 <b>របាយការណ៍សង្ខេបប្រចាំថ្ងៃ (UTC)</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>គោលដៅប្រចាំថ្ងៃ៖</b> <b>{goal} Cal</b>\n"
                    f"🔥 <b>បានញ៉ាំសរុប៖</b> <b>0 Cal</b>\n"
                    f"⚖️ <b>ស្ថានភាព៖</b> នៅសល់ <b>{goal} Cal</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "<b>សារធាតុចិញ្ចឹមសរុប៖</b>\n"
                    "🥩 <b>ប្រូតេអ៊ីន៖</b> <b>0g</b>\n"
                    "🧈 <b>ខ្លាញ់សរុប៖</b> <b>0g</b>\n"
                    "🍞 <b>កាបូអ៊ីដ្រាត៖</b> <b>0g</b>\n"
                    "🍬 <b>ស្ករ៖</b> <b>0g</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "<b>អាហារដែលបានញ៉ាំថ្ងៃនេះ៖</b>\n"
                    "មិនទាន់មានអាហារបានកត់ត្រាសម្រាប់ថ្ងៃនេះនៅឡើយទេ។ ផ្ញើរូបថតអាហារដើម្បីចាប់ផ្តើម!"
                )
                await bot.edit_message(chat_id, message_id, cleared_stats_text, reply_markup={"inline_keyboard": []})
            except Exception as reset_err:
                print(f"Error resetting daily logs: {reset_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការសម្អាតកំណត់ត្រាថ្ងៃនេះ។", show_alert=True)
            return

        # 3. Handle Delete specific reminder
        elif callback_data.startswith("delete_reminder:"):
            reminder_time = callback_data.split(":")[1]
            try:
                db_delete_reminder(user_id, reminder_time)
                await bot.answer_callback_query(callback_id, f"⏰ បានលុបម៉ោងរំលឹក {reminder_time}!")
                
                # Fetch updated reminders list
                reminders = db_get_reminders(user_id)
                # Calculate current Cambodian date (ICT, UTC+7)
                now_utc = datetime.datetime.utcnow()
                now_cambodia = now_utc + datetime.timedelta(hours=7)
                today_date_str = now_cambodia.strftime("%Y-%m-%d")
                
                if not reminders:
                    reminder_text = (
                        "🔔 <b>កំណត់ម៉ោងរំលឹកកត់ត្រាអាហារ</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"📅 <b>ថ្ងៃនេះ៖</b> <b>{today_date_str}</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "អ្នកមិនទាន់មានម៉ោងរំលឹកនៅឡើយទេ។\n\n"
                        "💡 <b>របៀបកំណត់ម៉ោងរំលឹក (ម៉ោងនៅកម្ពុជា)៖</b>\n"
                        "• វាយ <b>/reminder 08:00</b> — ដើម្បីរំលឹកម៉ោង ៨:០០ ព្រឹក\n\n"
                        "<b>ចំណាំ៖</b> ម៉ោងរំលឹកនឹងត្រូវបង្គត់ទៅរៀងរាល់ ១០នាទីម្តង។"
                    )
                    await bot.edit_message(chat_id, message_id, reminder_text, reply_markup={"inline_keyboard": []})
                else:
                    reminder_text = (
                        "🔔 <b>ម៉ោងរំលឹកបច្ចុប្បន្នរបស់អ្នក (ម៉ោងនៅកម្ពុជា)៖</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"📅 <b>ថ្ងៃនេះ៖</b> <b>{today_date_str}</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                    )
                    inline_keyboard = []
                    for idx, r_time in enumerate(reminders, start=1):
                        reminder_text += f"{idx}. ⏰ ម៉ោង <b>{r_time}</b>\n"
                        inline_keyboard.append([
                            {
                                "text": f"🗑️ លុបម៉ោង {r_time}",
                                "callback_data": f"delete_reminder:{r_time}"
                            }
                        ])
                    reminder_text += (
                        "\n💡 <b>គន្លឹះ៖</b>\n"
                        "• ដើម្បីបន្ថែមម៉ោងរំលឹកថ្មី៖ <b>/reminder 08:00</b>\n"
                        "• ដើម្បីលុបទាំងអស់៖ <b>/reminder clear</b>"
                    )
                    await bot.edit_message(chat_id, message_id, reminder_text, reply_markup={"inline_keyboard": inline_keyboard})
            except Exception as delete_err:
                print(f"Error deleting reminder: {delete_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការលុបម៉ោងរំលឹក។", show_alert=True)
            return

        # Handle TDEE Gender callback
        elif callback_data.startswith("tdee_gender:"):
            gender = callback_data.split(":")[1]
            try:
                db_set_tdee_step(user_id, step="age", gender=gender)
                await bot.answer_callback_query(callback_id, "ភេទត្រូវបានរក្សាទុក!")
                
                gender_display = "👨 ប្រុស (Male)" if gender == "male" else "👩 ស្រី (Female)"
                await bot.edit_message(
                    chat_id,
                    message_id,
                    "🧬 <b>គណនា BMR & TDEE (ជំហានទី ២/៥)</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"👤 ភេទ៖ <b>{gender_display}</b>\n\n"
                    "🎂 សូមវាយបញ្ចូល <b>អាយុ</b> របស់អ្នក (គិតជាឆ្នាំ)៖"
                )
            except Exception as err:
                print(f"Error saving gender: {err}")
                await bot.answer_callback_query(callback_id, "⚠️ កំហុសបច្ចេកទេស។", show_alert=True)
            return

        # Handle TDEE Activity callback
        elif callback_data.startswith("tdee_activity:"):
            activity = callback_data.split(":")[1]
            try:
                state = db_get_tdee_state(user_id)
                if not state or not state["gender"] or state["age"] is None or state["height"] is None or state["weight"] is None:
                    await bot.answer_callback_query(callback_id, "⚠️ រកមិនឃើញទិន្នន័យចាស់! សូមចាប់ផ្តើមម្តងទៀត។", show_alert=True)
                    db_clear_tdee_state(user_id)
                    return
                    
                gender = state["gender"]
                age = state["age"]
                height = state["height"]
                weight = state["weight"]
                
                # Mifflin-St Jeor Formula
                if gender == "male":
                    bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
                else:
                    bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
                    
                # Multipliers
                multipliers = {
                    "sedentary": 1.2,
                    "light": 1.375,
                    "moderate": 1.465,  # exactly 1.465!
                    "active": 1.55,
                    "very_active": 1.725
                }
                
                multiplier = multipliers.get(activity, 1.2)
                maintain = bmr * multiplier
                
                # Tiers
                mild = maintain - 250
                loss = maintain - 500
                extreme = maintain - 1000
                
                # Sensible minimums
                maintain = max(500, maintain)
                mild = max(500, mild)
                loss = max(500, loss)
                extreme = max(500, extreme)
                
                # Percentages relative to Maintain (100%)
                mild_pct = (mild / maintain) * 100
                loss_pct = (loss / maintain) * 100
                extreme_pct = (extreme / maintain) * 100
                
                gender_kh = "ប្រុស (Male)" if gender == "male" else "ស្រី (Female)"
                activity_kh = {
                    "sedentary": "Sedentary (អង្គុយច្រើន/គ្មានលំហាត់ប្រាណ)",
                    "light": "Light (ហាត់ប្រាណ ១-៣ ថ្ងៃ/សប្តាហ៍)",
                    "moderate": "Moderate (ហាត់ប្រាណ ៤-៥ ថ្ងៃ/សប្តាហ៍)",
                    "active": "Active (ហាត់ប្រាណរាល់ថ្ងៃ/ខ្លាំង)",
                    "very_active": "Very Active (ហាត់ប្រាណខ្លាំងខ្លាំង)"
                }.get(activity, activity)
                
                db_save_user_profile(user_id, gender, age, height, weight, activity)
                db_clear_tdee_state(user_id)
                await bot.answer_callback_query(callback_id, "គណនារួចរាល់!")
                
                result_card = (
                    "📊 <b>លទ្ធផលគណនា BMR & TDEE</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "👤 <b>ព័ត៌មានរូបរាងកាយ៖</b>\n"
                    f"• ភេទ៖ <b>{gender_kh}</b>\n"
                    f"• អាយុ៖ <b>{age} ឆ្នាំ</b>\n"
                    f"• កម្ពស់៖ <b>{height:.1f} cm</b>\n"
                    f"• ទម្ងន់៖ <b>{weight:.1f} kg</b>\n"
                    f"• កម្រិតសកម្មភាព៖ <b>{activity_kh}</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔥 <b>BMR (អត្រាមេតាបូលីសបាត)៖</b> <b>{bmr:.0f} Cal</b>\n"
                    f"⚡ <b>TDEE (រក្សាទម្ងន់)៖</b> <b>{maintain:.0f} Cal</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "🎯 <b>ជ្រើសរើសគោលដៅកាឡូរីប្រចាំថ្ងៃ៖</b>\n"
                    f"⚖️ <b>Maintain</b> (100%)៖ <b>{maintain:.0f} Cal</b>\n"
                    f"📉 <b>Mild Loss</b> ({mild_pct:.0f}%)៖ <b>{mild:.0f} Cal</b> (-250)\n"
                    f"🔥 <b>Weight Loss</b> ({loss_pct:.0f}%)៖ <b>{loss:.0f} Cal</b> (-500)\n"
                    f"🚨 <b>Extreme Loss</b> ({extreme_pct:.0f}%)៖ <b>{extreme:.0f} Cal</b> (-1000)\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "សូមចុចប៊ូតុងខាងក្រោមដើម្បីកំណត់គោលដៅកាឡូរីប្រចាំថ្ងៃដោយស្វ័យប្រវត្ត៖"
                )
                
                inline_keyboard = [
                    [
                        {"text": f"⚖️ រក្សាទម្ងន់ ({maintain:.0f} Cal)", "callback_data": f"setgoal:maintain:{int(maintain)}"}
                    ],
                    [
                        {"text": f"📉 ស្រកតិចតួច ({mild:.0f} Cal)", "callback_data": f"setgoal:mild:{int(mild)}"}
                    ],
                    [
                        {"text": f"🔥 សម្រកទម្ងន់ ({loss:.0f} Cal)", "callback_data": f"setgoal:loss:{int(loss)}"}
                    ],
                    [
                        {"text": f"🚨 សម្រកខ្លាំង ({extreme:.0f} Cal)", "callback_data": f"setgoal:extreme:{int(extreme)}"}
                    ]
                ]
                
                await bot.edit_message(chat_id, message_id, result_card, reply_markup={"inline_keyboard": inline_keyboard})
            except Exception as err:
                print(f"Error calculating TDEE: {err}")
                await bot.answer_callback_query(callback_id, "⚠️ បញ្ហាក្នុងពេលគណនា។", show_alert=True)
            return

        # Handle Goal click callback
        elif callback_data.startswith("setgoal:"):
            parts = callback_data.split(":")
            goal_type = parts[1]
            calories = int(parts[2])
            
            try:
                db_update_tdee_goal(user_id, goal_type, calories)
                await bot.answer_callback_query(callback_id, "🎯 Goal Saved!")
                
                goal_type_kh = {
                    "maintain": "Maintain (រក្សាទម្ងន់)",
                    "mild": "Mild Loss (ស្រកតិចតួច)",
                    "loss": "Weight Loss (សម្រកទម្ងន់)",
                    "extreme": "Extreme Loss (សម្រកខ្លាំង)"
                }.get(goal_type, goal_type)
                
                confirmation_text = (
                    "✅ <b>Goal Saved Successfully!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "គោលដៅកាឡូរីប្រចាំថ្ងៃថ្មីរបស់អ្នកត្រូវបានកំណត់ទៅ៖\n"
                    f"• <b>ប្រភេទគោលដៅ៖</b> <b>{goal_type_kh}</b>\n"
                    f"• <b>ថាមពលប្រចាំថ្ងៃ៖</b> <b>{calories} Cal</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "🎉 <b>ប្រព័ន្ធត្រូវបានធ្វើបច្ចុប្បន្នភាពរួចរាល់!</b>"
                )
                await bot.edit_message(chat_id, message_id, confirmation_text, reply_markup={"inline_keyboard": []})
            except Exception as save_err:
                print(f"Error saving goal: {save_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការរក្សាទុកគោលដៅ。", show_alert=True)
            return

        # 4. Handle Suggest Food preference click callback
        elif callback_data.startswith("suggest_pref:"):
            pref_type = callback_data.split(":")[1]
            pref_names_kh = {
                "veg": "បន្លែច្រើន (High Veg)",
                "meat": "សាច់ច្រើន (High Meat)",
                "normal": "ម្ហូបធម្មតា (Standard Khmer)"
            }
            pref_kh = pref_names_kh.get(pref_type, "ម្ហូបធម្មតា")
            
            try:
                # Answer callback immediately to halt spinners
                await bot.answer_callback_query(callback_id, f"រៀបចំមុខម្ហូប: {pref_kh}")
                
                # Fetch profile and goals
                profile = db_get_user_profile(user_id)
                goal = db_get_user_goal(user_id)
                
                # Show loading update card
                await bot.edit_message(
                    chat_id,
                    message_id,
                    f"💡 <i>កំពុងរៀបចំសំណើមុខម្ហូបប្រចាំថ្ងៃ [{pref_kh}] ដែលសមស្របនឹងគោលដៅ {goal} Cal របស់អ្នក... សូមរង់ចាំមួយភ្លែត។</i>",
                    reply_markup={"inline_keyboard": []}
                )
                
                gemini_key = os.getenv("GEMINI_API_KEY")
                if not gemini_key:
                    raise ValueError("GEMINI_API_KEY environment variable is not configured.")
                
                client = genai.Client()
                
                user_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                models_to_try = [user_model]
                for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)
                
                if profile:
                    profile_context = (
                        f"The user is a {profile['gender']}, {profile['age']} years old, {profile['height']:.1f} cm tall, "
                        f"weighing {profile['weight']:.1f} kg. Their physical activity level is mapped as '{profile['activity']}'. "
                        f"Their daily budget goal is {goal} Cal and their goal type is '{profile['goal_type']}'."
                    )
                else:
                    profile_context = f"The user is a general individual with a daily budget goal of {goal} Cal."
                
                # Dynamic prompt based on selected preference
                if pref_type == "veg":
                    pref_instructions = (
                        "The user requested: 🥗 បន្លែច្រើន (High Veg / Low Carb).\n"
                        "Your suggestions MUST be extremely high in volume of local vegetables and salads (e.g. boiled/steamed greens like ត្រកួន, ស្ពៃ, ស្ពៃក្តោប, ត្រសក់) and very low in carbohydrates. Minimize large portions of white rice, sweet sauces, or noodles. Ensure it maximizes fullness on their calorie limit."
                    )
                elif pref_type == "meat":
                    pref_instructions = (
                        "The user requested: 🥩 សាច់ច្រើន (High Protein).\n"
                        "Your suggestions MUST focus heavily on high-protein sources and lean local proteins (e.g. skinless chicken breast, local grilled/steamed fish, boiled eggs) while keeping carbohydrates low. Minimize high-carb sides."
                    )
                else:
                    pref_instructions = (
                        "The user requested: 🍲 ម្ហូបធម្មតា (Standard Balanced Khmer Food).\n"
                        "Your suggestions should feature standard balanced Cambodian rice and soup/stir-fry dishes (e.g. standard portions of white rice, local soup, local balanced fish/chicken stir-fry)."
                    )
                
                SUGGEST_SYSTEM_PROMPT = (
                    "You are a professional nutrition expert and Cambodian culinary specialist. "
                    "You must generate an extremely concise 1-day Meal Plan divided into Breakfast (អាហារពេលព្រឹក), Lunch (អាហារពេលថ្ងៃត្រង់), and Dinner (អាហារពេលល្ងាច) tailored specifically to the user's TDEE target calorie budget.\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. YOU MUST RESPOND ENTIRELY IN KHMER LANGUAGE.\n"
                    "2. STRICTLY NO GREETINGS, NO WELCOME MESSAGES, NO INTRODUCTIONS, and NO USER PROFILE/CONTEXT SUMMARIES. Do not output any hello, profile summary, gender, age, height, weight, or goal type. Start directly with the text 'អាហារពេលព្រឹក (Breakfast)'.\n"
                    "3. STRICTLY NO notes, NO 'ចំណាំ' (note) paragraphs under individual meals or at the end.\n"
                    "4. STRICTLY NO additional tips, NO 'គន្លឹះបន្ថែមសម្រាប់សុខភាព', NO health advices, NO 'ការណែនាំបន្ថែម', and NO closing remarks at the end. Stop and end the generation immediately after the dinner meal bullet points.\n"
                    "5. DO NOT use italic tags (like <i> or <em>) for estimated calories. Use regular bold (<b>) or normal unformatted text instead (e.g. 'កាឡូរីប៉ាន់ស្មាន៖ ~៥០៨ Cal').\n"
                    f"6. Calorie Limit: Ensure the calories for Breakfast + Lunch + Dinner add up approximately to their daily target of {goal} Cal. Clearly state estimated calories (Cal) for each meal.\n"
                    f"7. User Food Preference: {pref_instructions}\n"
                    "8. Market Accessibility: All proposed meals and ingredients MUST be cheap, typical, and very easy to buy in local Cambodian markets (ផ្សារខ្មែរ). Use local ingredients (e.g. ត្រី, ទ្រូងមាន់, ត្រកួន, ស្ពៃ, ស៊ុត) with simple seasonings (ទឹកត្រី, ទឹកស៊ីអ៊ីវ).\n"
                    "9. Format your response beautifully using standard Telegram HTML tags like <b> and clean section dividers like ━━━━━━━━━━━━━━━━━━━━. DO NOT use markdown code blocks or triple backticks. Keep the layout premium, modern, and highly legible.\n"
                    "Format EXACTLY like this structure and stop immediately after the last bullet point of Dinner:\n"
                    "អាហារពេលព្រឹក (Breakfast)\n"
                    "<b>កាឡូរីប៉ាន់ស្មាន៖ ~... Cal</b>\n"
                    "• ...\n"
                    "• ...\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "អាហារពេលថ្ងៃត្រង់ (Lunch)\n"
                    "<b>កាឡូរីប៉ាន់ស្មាន៖ ~... Cal</b>\n"
                    "• ...\n"
                    "• ...\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "អាហារពេលល្ងាច (Dinner)\n"
                    "<b>កាឡូរីប៉ាន់ស្មាន៖ ~... Cal</b>\n"
                    "• ...\n"
                    "• ..."
                )
                
                response = None
                last_error = None
                
                for current_model in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=current_model,
                            contents=f"Please generate my 1-day meal plan based on my profile context: {profile_context}",
                            config=types.GenerateContentConfig(
                                system_instruction=SUGGEST_SYSTEM_PROMPT,
                            ),
                        )
                        break
                    except Exception as model_err:
                        last_error = model_err
                        print(f"⚠️ Model {current_model} failed or is rate-limited: {model_err}")
                        continue
                        
                if response is None:
                    raise ValueError(f"All generative models failed. Last error: {last_error}")
                    
                suggested_menu = response.text
                
                # Safe clean of any raw markdown wrapper leaks
                suggested_menu = suggested_menu.replace("```html", "").replace("```", "").strip()
                
                menu_header = (
                    f"💡 <b>មុខម្ហូបណែនាំប្រចាំថ្ងៃ ({pref_kh})</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>គោលដៅថ្ងៃនេះ៖</b> <b>{goal} Cal</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                )
                menu_footer = ""
                final_response = f"{menu_header}{suggested_menu}"
                
                await bot.edit_message(chat_id, message_id, final_response)
                
            except Exception as e:
                print(f"Error during interactive meal suggestion callback: {e}")
                fail_msg = (
                    "⚠️ <b>ការណែនាំមុខម្ហូបបានបរាជ័យ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "មានបញ្ហាបច្ចេកទេសមួយបានកើតឡើងក្នុងពេលរៀបចំមុខម្ហូបណែនាំ。\n\n"
                    f"<b>ព័ត៌មានលម្អិត:</b> <code>{str(e)}</code>"
                )
                await bot.edit_message(chat_id, message_id, fail_msg)
            return

        # 5. Handle Disconnecting Google Fit
        elif callback_data == "disconnect_fit":
            try:
                db_delete_fit_tokens(user_id)
                await bot.answer_callback_query(callback_id, "🔌 បានផ្តាច់ការភ្ជាប់ Google Fit!")
                disconnect_text = (
                    "🔌 <b>បានផ្តាច់ពី Google Fit រួចរាល់!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "គណនីរបស់អ្នកត្រូវបានផ្តាច់ចេញពី Google Fit ហើយ។ ប្រសិនបើចង់ភ្ជាប់ឡើងវិញ សូមប្រើប្រាស់បញ្ជា <b>/fit</b> ម្តងទៀត។"
                )
                await bot.edit_message(chat_id, message_id, disconnect_text, reply_markup={"inline_keyboard": []})
            except Exception as disc_err:
                print(f"Error disconnecting Google Fit: {disc_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការផ្តាច់ពី Google Fit។", show_alert=True)
            return


        # 5b. Handle Disconnecting Strava
        elif callback_data == "disconnect_strava":
            try:
                db_delete_strava_tokens(user_id)
                await bot.answer_callback_query(callback_id, "🔌 បានផ្តាច់ការភ្ជាប់ Strava!")
                disconnect_text = (
                    "🔌 <b>បានផ្តាច់ពី Strava រួចរាល់!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "គណនីរបស់អ្នកត្រូវបានផ្តាច់ចេញពី Strava ហើយ។ ប្រសិនបើចង់ភ្ជាប់ឡើងវិញ សូមប្រើប្រាស់បញ្ជា <b>/strava</b> ម្តងទៀត។"
                )
                await bot.edit_message(chat_id, message_id, disconnect_text, reply_markup={"inline_keyboard": []})
            except Exception as disc_err:
                print(f"Error disconnecting Strava: {disc_err}")
                await bot.answer_callback_query(callback_id, "⚠️ បរាជ័យក្នុងការផ្តាច់ពី Strava។", show_alert=True)
            return
    # Process standard text or photo messages
    message = payload.get("message")
    if not message:
        return

    chat_id = message["chat"]["id"]
    user_id = message["from"]["id"]
    text = message.get("text", "").strip()
    photo = message.get("photo")
    document = message.get("document")

    # Register mimetypes for specific common image extensions to be safe
    mimetypes.add_type('image/heic', '.heic')
    mimetypes.add_type('image/heif', '.heif')
    mimetypes.add_type('image/webp', '.webp')

    # 1. Handle text commands and state machines
    if text:
        # Check TDEE state first
        tdee_state = db_get_tdee_state(user_id)
        
        # Handle TDEE state transitions
        if tdee_state and not text.startswith("/"):
            step = tdee_state["step"]
            
            # Handle age step
            if step == "age":
                try:
                    age = int(text)
                    if age <= 0 or age > 120:
                        raise ValueError()
                    db_set_tdee_step(user_id, step="height", age=age)
                    await bot.send_message(
                        chat_id,
                        "🧬 <b>គណនា BMR & TDEE (ជំហានទី ៣/៥)</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 អាយុ៖ <b>{age} ឆ្នាំ</b>\n\n"
                        "📐 សូមវាយបញ្ចូល <b>កម្ពស់</b> របស់អ្នក (គិតជាសង់ទីម៉ែត្រ cm)៖"
                    )
                except ValueError:
                    await bot.send_message(
                        chat_id,
                        "⚠️ <b>អាយុមិនត្រឹមត្រូវទេ!</b>\n"
                        "សូមវាយបញ្ចូលអាយុជាលេខរាប់ពី ១ ដល់ ១២០។ ឧទាហរណ៍៖ <b>25</b>"
                    )
                return
                
            # Handle height step
            elif step == "height":
                try:
                    height = float(text)
                    if height <= 50 or height > 280:
                        raise ValueError()
                    db_set_tdee_step(user_id, step="weight", height=height)
                    await bot.send_message(
                        chat_id,
                        "🧬 <b>គណនា BMR & TDEE (ជំហានទី ៤/៥)</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 អាយុ៖ <b>{tdee_state['age']} ឆ្នាំ</b>\n"
                        f"📐 កម្ពស់៖ <b>{height:.1f} cm</b>\n\n"
                        "⚖️ សូមវាយបញ្ចូល <b>ទម្ងន់</b> របស់អ្នក (គិតជាគីឡូក្រាម kg)៖"
                    )
                except ValueError:
                    await bot.send_message(
                        chat_id,
                        "⚠️ <b>កម្ពស់មិនត្រឹមត្រូវទេ!</b>\n"
                        "សូមវាយបញ្ចូលកម្ពស់ជាលេខគិតជាសង់ទីម៉ែត្រ (cm)។ ឧទាហរណ៍៖ <b>170</b>"
                    )
                return
                
            # Handle weight step
            elif step == "weight":
                try:
                    weight = float(text)
                    if weight <= 10 or weight > 500:
                        raise ValueError()
                    db_set_tdee_step(user_id, step="activity", weight=weight)
                    
                    inline_keyboard = [
                        [
                            {"text": "🛋️ Sedentary (កម្រហាត់ប្រាណ)", "callback_data": "tdee_activity:sedentary"}
                        ],
                        [
                            {"text": "🚶 Light (ហាត់ប្រាណ ១-៣ ថ្ងៃ/សប្តាហ៍)", "callback_data": "tdee_activity:light"}
                        ],
                        [
                            {"text": "🏃 Moderate (ហាត់ប្រាណ ៤-៥ ថ្ងៃ/សប្តាហ៍)", "callback_data": "tdee_activity:moderate"}
                        ],
                        [
                            {"text": "🏋️ Active (ហាត់ប្រាណរាល់ថ្ងៃ)", "callback_data": "tdee_activity:active"}
                        ],
                        [
                            {"text": "🔥 Very Active (ហាត់ប្រាណខ្លាំងខ្លាំង)", "callback_data": "tdee_activity:very_active"}
                        ]
                    ]
                    
                    await bot.send_message(
                        chat_id,
                        "🧬 <b>គណនា BMR & TDEE (ជំហានទី ៥/៥)</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 អាយុ៖ <b>{tdee_state['age']} ឆ្នាំ</b>\n"
                        f"📐 កម្ពស់៖ <b>{tdee_state['height']:.1f} cm</b>\n"
                        f"⚖️ ទម្ងន់៖ <b>{weight:.1f} kg</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "🏃‍♀️ សូមជ្រើសរើស <b>កម្រិតសកម្មភាពរាងកាយ</b> របស់អ្នក៖",
                        reply_markup={"inline_keyboard": inline_keyboard}
                    )
                except ValueError:
                    await bot.send_message(
                        chat_id,
                        "⚠️ <b>ទម្ងន់មិនត្រឹមត្រូវទេ!</b>\n"
                        "សូមវាយបញ្ចូលទម្ងន់ជាលេខគិតជាគីឡូក្រាម (kg)។ ឧទាហរណ៍៖ <b>65</b>"
                    )
                return

        # Handle commands
        if text.startswith("/start"):
            welcome_text = (
                "<b>ស្វាគមន៍មកកាន់ NutriBot</b>\n"
                "ខ្ញុំជា AI ជំនាញអាហារូបត្ថម្ភរបស់អ្នក。 ផ្ញើរូបភាពអាហារ ឬប្រើពាក្យបញ្ជាខាងក្រោមដើម្បីចាប់ផ្តើម៖\n\n"
                "<b>ពាក្យបញ្ជាទូទៅ</b>\n"
                "• <b>ផ្ញើរូបថត</b>៖ វិភាគ និងកត់ត្រាអាហារពីក្នុងរូបភាពដោយស្វ័យប្រវត្ត។\n"
                "• <b>/log &lt;ឈ្មោះអាហារ និងបរិមាណ&gt;</b>៖ កត់ត្រាអាហារជាអត្ថបទ។\n"
                "• <b>/burn &lt;កាឡូរី&gt;</b>៖ កត់ត្រាការដុតកាឡូរីពីការហាត់ប្រាណ។\n"
                "• <b>/nosweet</b>៖ កត់ត្រាការតមភេសជ្ជៈផ្អែមថ្ងៃនេះ។\n"
                "• <b>/menu</b> (ឬ <b>/suggest</b>)៖ ណែនាំមុខម្ហូបប្រចាំថ្ងៃសមស្របនឹងកាឡូរីរបស់អ្នក។\n\n"
                "<b>គ្រប់គ្រងគណនី និងគោលដៅ</b>\n"
                "• <b>/weight &lt;ទម្ងន់&gt;</b>៖ ធ្វើបច្ចុប្បន្នភាពទម្ងន់ និងគណនា TDEE ឡើងវិញ។\n"
                "• <b>/cal</b>៖ គណនា BMR/TDEE និងកំណត់គោលដៅកាឡូរីស្វ័យប្រវត្ត។\n"
                "• <b>/goal &lt;កាឡូរី&gt;</b>៖ កំណត់គោលដៅកាឡូរីប្រចាំថ្ងៃ។\n"
                "• <b>/strava</b>៖ ភ្ជាប់ ឬផ្តាច់គណនីជាមួយ Strava។\n\n"
                "<b>របាយការណ៍ និងការកំណត់</b>\n"
                "• <b>/stats</b>៖ មើលរបាយការណ៍អាហារូបត្ថម្ភប្រចាំថ្ងៃ។\n"
                "• <b>/weekly</b>៖ មើលរបាយការណ៍សង្ខេបប្រចាំសប្តាហ៍។\n"
                "• <b>/reminder &lt;ម៉ោង&gt;</b>៖ កំណត់ម៉ោងរំលឹកកត់ត្រាអាហារ។\n"
                "• <b>/start</b>៖ បង្ហាញការណែនាំនេះឡើងវិញ។\n\n"
                "<b>ចាប់ផ្តើមឥឡូវនេះ៖</b> ផ្ញើរូបថតអាហាររបស់អ្នក ឬវាយបញ្ចូលពាក្យបញ្ជាណាមួយខាងលើ។"
            )
            await bot.send_message(chat_id, welcome_text)
            return
        
        elif text.startswith("/cal"):
            db_clear_tdee_state(user_id)
            db_set_tdee_step(user_id, step="gender")
            
            inline_keyboard = [
                [
                    {"text": "👨 ប្រុស (Male)", "callback_data": "tdee_gender:male"},
                    {"text": "👩 ស្រី (Female)", "callback_data": "tdee_gender:female"}
                ]
            ]
            
            await bot.send_message(
                chat_id,
                "🧬 <b>គណនា BMR & TDEE (ជំហានទី ១/៥)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "សូមជ្រើសរើស <b>ភេទ</b> របស់អ្នក៖",
                reply_markup={"inline_keyboard": inline_keyboard}
            )
            return

        elif text.startswith("/log"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                await bot.send_message(
                    chat_id,
                    "📝 <b>របៀបកត់ត្រាអាហារដោយផ្ទាល់៖</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "សូមវាយ៖ <b>/log [ឈ្មោះអាហារ និងបរិមាណ]</b>\n"
                    "ឧទាហរណ៍៖ <b>/log បាយស ២០០ក្រាម សាច់មាន់ ១០០ក្រាម</b>\n\n"
                    "📅 <b>កត់ត្រាសម្រាប់ថ្ងៃផ្សេងទៀត (បើភ្លេច)៖</b>\n"
                    "សូមវាយ៖ <b>/log [កាលបរិច្ឆេទ/ម្សិលមិញ] [ឈ្មោះអាហារ]</b>\n"
                    "ឧទាហរណ៍៖ <b>/log ម្សិលមិញ បាយឆាសាច់ជ្រូក</b>\n"
                    "ឬ <b>/log 02-06-2026 បាយឆាសាច់ជ្រូក</b>"
                )
                return
            
            # Check if the first word is a date or relative day keyword
            sub_parts = parts[1].split(maxsplit=1)
            custom_date = None
            food_description = parts[1]
            date_token_src = None
            
            if len(sub_parts) > 0:
                potential_date = sub_parts[0]
                parsed_date = parse_custom_date(potential_date)
                if parsed_date:
                    custom_date = parsed_date
                    date_token_src = potential_date
                    if len(sub_parts) < 2:
                        await bot.send_message(
                            chat_id,
                            f"📅 <b>កត់ត្រាសម្រាប់ថ្ងៃ៖ {custom_date}</b>\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "សូមបញ្ចូលឈ្មោះអាហារបន្ថែមផងដែរ។\n"
                            f"ឧទាហរណ៍៖ <b>/log {potential_date} បាយឆាសាច់ជ្រូក</b>"
                        )
                        return
                    food_description = sub_parts[1]
            
            display_date = custom_date if custom_date else "ថ្ងៃនេះ"
            if date_token_src and date_token_src.lower() in ["yesterday", "ម្សិលមិញ"]:
                display_date = "ម្សិលមិញ"
            elif date_token_src and date_token_src.lower() in ["ម្សិលម្ងៃ", "ម្សិលម្ង៉ៃ", "ម្សិលមិញមួយថ្ងៃ"]:
                display_date = "ម្សិលម្ងៃ"
                
            ack = await bot.send_message(
                chat_id, 
                f"🔍 <i>កំពុងវិភាគការពណ៌នាអាហារសម្រាប់ {display_date}... សូមរង់ចាំមួយភ្លែត។</i>" if custom_date 
                else "🔍 <i>កំពុងវិភាគការពណ៌នាអាហាររបស់អ្នក... សូមរង់ចាំមួយភ្លែត។</i>"
            )
            ack_message_id = ack.get("result", {}).get("message_id")
            
            try:
                gemini_key = os.getenv("GEMINI_API_KEY")
                if not gemini_key:
                    raise ValueError("GEMINI_API_KEY environment variable is not configured.")
                
                client = genai.Client()
                
                user_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                models_to_try = [user_model]
                for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)
                        
                # Get current Cambodia ICT local time or custom date context
                now_utc = datetime.datetime.utcnow()
                now_cambodia = now_utc + datetime.timedelta(hours=7)
                time_str = now_cambodia.strftime("%I:%M %p")
                day_name = now_cambodia.strftime("%A")
                
                # Determine period of the day
                hour = now_cambodia.hour
                if 5 <= hour < 11:
                    period_kh = "ពេលព្រឹក (Morning)"
                elif 11 <= hour < 14:
                    period_kh = "ពេលថ្ងៃត្រង់ (Lunch)"
                elif 14 <= hour < 17:
                    period_kh = "ពេលរសៀល (Afternoon)"
                elif 17 <= hour < 22:
                    period_kh = "ពេលល្ងាច/យប់ (Evening/Night)"
                else:
                    period_kh = "ពេលយប់ជ្រៅ (Late Night)"
                    
                profile = db_get_user_profile(user_id)
                if profile:
                    profile_context = (
                        f"The user is a {profile['gender']}, {profile['age']} years old, {profile['height']:.1f} cm tall, "
                        f"weighing {profile['weight']:.1f} kg. Their physical activity level is mapped as '{profile['activity']}'. "
                        f"Their daily budget goal is {profile['daily_calorie_budget']} Cal and their goal type is '{profile['goal_type']}'."
                    )
                else:
                    profile_context = "The user is a general individual with a daily budget of 2000 Cal aiming to maintain weight."

                logging_time_context = f"Current Cambodia local time is {time_str} on {day_name} ({period_kh})."
                if custom_date:
                    logging_time_context = f"The user is retroactively logging for the Cambodia local date {custom_date}."

                TEXT_SYSTEM_PROMPT = (
                    "You are a professional nutrition expert and health coach. Analyze the food description text provided and estimate its "
                    "nutritional details (calories in Cal, protein/fat/carbs/sugar in grams).\n"
                    f"User Health Context: {profile_context}\n"
                    f"Logging Context: {logging_time_context}\n"
                    "YOU MUST RESPOND ENTIRELY IN KHMER LANGUAGE. The `food_name` field must be written in beautiful Khmer script.\n"
                    "Provide a highly personalized coaching and health recommendation (in the `coaching_recommendation` field) "
                    "in Khmer tailored specifically to this user's profile, goal, and the logging context.\n"
                    "CRITICAL SECRECY RULE: You know the user's age, weight, height, and calorie target budget from the User Health Context, BUT YOU MUST KEEP THEM SECRET. Never mention or repeat their age, weight, height, or daily calorie goal in your coaching_recommendation text response. Focus purely on qualitative health insights, digestion, macronutrients, and positive coaching advice.\n"
                    "Do NOT recite or repeat raw numbers (like '150 Cal' or '10g protein') inside the coaching recommendation text since those are already clearly displayed in the summary card.\n"
                    "If the text does not describe any food, or you cannot identify any food, "
                    "you MUST set the `confidence_score` to less than 0.5 (e.g. 0.0 to 0.4), "
                    "and you can set the `food_name` to 'មិនមែនជាអាហារ ឬរកមិនឃើញ'. "
                    "Be realistic, objective, and estimate standard portion sizes based on the provided quantities or standard servings."
                )
                
                response = None
                last_error = None
                
                for current_model in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=current_model,
                            contents=f"Analyze the following food description and return its nutrition facts in Khmer: {food_description}",
                            config=types.GenerateContentConfig(
                                system_instruction=TEXT_SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                response_schema=FoodAnalysis,
                            ),
                        )
                        break
                    except Exception as model_err:
                        last_error = model_err
                        print(f"⚠️ Model {current_model} failed or is rate-limited: {model_err}")
                        continue
                        
                if response is None:
                    raise ValueError(f"All generative models failed. Last error: {last_error}")
                    
                analysis = FoodAnalysis.model_validate_json(response.text)
                
                if analysis.confidence_score < 0.5:
                    err_msg = (
                        "🍳 <b>អូ! ខ្ញុំរកមិនឃើញអាហារទេ!</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "ការពណ៌នារបស់អ្នកមិនហាក់ដូចជាអាហារ ឬបរិមាណមិនច្បាស់លាស់។ សូមប្រាកដថាអ្នកពណ៌នាពីឈ្មោះអាហារ ឬបរិមាណបានត្រឹមត្រូវ រួចផ្ញើមកម្តងទៀត!\n\n"
                        f"<b>(រកឃើញ៖ {analysis.food_name} | កម្រិតច្បាស់លាស់៖ {analysis.confidence_score * 100:.0f}%)</b>"
                    )
                    if ack_message_id:
                        await bot.edit_message(chat_id, ack_message_id, err_msg)
                    else:
                        await bot.send_message(chat_id, err_msg)
                    return
                    
                inserted_meal_id = db_add_meal(user_id, analysis, custom_date)
                await sync_meal_to_google_fit(user_id, analysis)
                
                # Fetch remaining calories for custom date or today
                if custom_date:
                    today_meals, total_cals = db_get_day_meals(user_id, custom_date)
                    total_burn = db_get_day_burn(user_id, custom_date)
                else:
                    today_meals, total_cals = db_get_today_meals(user_id)
                    total_burn = db_get_today_burn(user_id)
                    
                goal = db_get_user_goal(user_id)
                remaining = goal - total_cals
                balance_emoji = "⚖️" if remaining >= 0 else "🚨"
                remaining_str = f"សល់ <b>{remaining} Cal</b>" if remaining >= 0 else f"លើស <b>{-remaining} Cal</b>"
                
                # Format custom display date
                if custom_date:
                    date_parts = custom_date.split('-')
                    formatted_display_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                else:
                    formatted_display_date = now_cambodia.strftime('%d-%m-%Y')
                
                result_card = (
                    "🍳 <b>លទ្ធផលវិភាគអាហារូបត្ថម្ភ (Direct Log)</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🥗 <b>អាហារ៖</b> <b>{analysis.food_name}</b>\n"
                    f"📊 <b>ភាពជឿជាក់៖</b> <b>{analysis.confidence_score * 100:.0f}%</b>\n"
                    f"📅 <b>កាលបរិច្ឆេទ៖</b> <b>{formatted_display_date}</b>\n\n"
                    f"🔥 <b>ថាមពល៖</b> <b>{analysis.calories} Cal</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🥩 <b>ប្រូតេអ៊ីន៖</b> <b>{analysis.protein}g</b>\n"
                    f"🧈 <b>ខ្លាញ់សរុប៖</b> <b>{analysis.fat}g</b>\n"
                    f"🍞 <b>កាបូអ៊ីដ្រាត៖</b> <b>{analysis.carbs}g</b>\n"
                    f"🍬 <b>ក្នុងនោះជាតិស្ករ៖</b> <b>{analysis.sugar}g</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🏃 <b>បានដុតរំលាយ៖</b> <b>{total_burn} Cal</b>\n"
                    f"{balance_emoji} <b>កាឡូរី ({display_date})៖</b> <b>{total_cals}</b> / <b>{goal} Cal</b> ({remaining_str})\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"💡 <b>ការណែនាំពីគ្រូ៖</b>\n"
                    f"« {analysis.coaching_recommendation} »\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "💾 <b>បានកត់ត្រាចូលគណនីរួចរាល់! ប្រសិនបើចង់លុបកំណត់ត្រានេះវិញ សូមចុចប៊ូតុងខាងក្រោម៖</b>"
                )
                
                inline_reply_markup = {
                    "inline_keyboard": [
                        [
                            {
                                "text": "❌ លុបកំណត់ត្រានេះ",
                                "callback_data": f"delete_meal:{inserted_meal_id}"
                            }
                        ]
                    ]
                }
                
                if ack_message_id:
                    await bot.edit_message(chat_id, ack_message_id, result_card, reply_markup=inline_reply_markup)
                else:
                    await bot.send_message(chat_id, result_card, reply_markup=inline_reply_markup)
                    
            except Exception as e:
                print(f"Error during manual food text logging analysis: {e}")
                fail_msg = (
                    "⚠️ <b>ការវិភាគអាហារូបត្ថម្ភបានបរាជ័យ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "មានបញ្ហាបច្ចេកទេសមួយបានកើតឡើងក្នុងពេលដំណើរការវិភាគការពណ៌នារបស់អ្នក។\n\n"
                    f"<b>ព័ត៌មានលម្អិត:</b> <code>{str(e)}</code>"
                )
                if ack_message_id:
                    await bot.edit_message(chat_id, ack_message_id, fail_msg)
                else:
                    await bot.send_message(chat_id, fail_msg)
            return
            
        elif text.startswith("/goal"):
            parts = text.split()
            if len(parts) < 2:
                current_goal = db_get_user_goal(user_id)
                await bot.send_message(
                    chat_id, 
                    f"🎯 គោលដៅកាឡូរីប្រចាំថ្ងៃបច្ចុប្បន្នរបស់អ្នកគឺ: <b>{current_goal} Cal</b>។\n"
                    f"ដើម្បីធ្វើបច្ចុប្បន្នភាព សូមវាយ: <code>/goal 1800</code>"
                )
                return
            
            try:
                new_goal = int(parts[1])
                if new_goal <= 0 or new_goal > 10000:
                    raise ValueError()
                db_update_user_goal(user_id, new_goal)
                await bot.send_message(
                    chat_id,
                    f"✅ <b>គោលដៅប្រចាំថ្ងៃត្រូវបានធ្វើបច្ចុប្បន្នភាព!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"គោលដៅកាឡូរីប្រចាំថ្ងៃរបស់អ្នកឥឡូវនេះគឺ <b>{new_goal} Cal</b>។"
                )
            except ValueError:
                await bot.send_message(
                    chat_id,
                    "⚠️ <b>ចំនួនកាឡូរីមិនត្រឹមត្រូវទេ!</b>\n"
                    "សូមផ្តល់ចំនួនលេខវិជ្ជមានសមរម្យមួយ។ ឧទាហរណ៍៖ <b>/goal 2000</b>"
                )
            return

        elif text.startswith("/reminder"):
            parts = text.split()
            now_utc = datetime.datetime.utcnow()
            now_cambodia = now_utc + datetime.timedelta(hours=7)
            today_date_str = now_cambodia.strftime("%Y-%m-%d")

            if len(parts) < 2:
                reminders = db_get_reminders(user_id)
                if not reminders:
                    reminder_text = (
                        "🔔 <b>កំណត់ម៉ោងរំលឹកកត់ត្រាអាហារ</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"📅 <b>ថ្ងៃនេះ៖</b> <b>{today_date_str}</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "អ្នកមិនទាន់មានម៉ោងរំលឹកនៅឡើយទេ។\n\n"
                        "💡 <b>របៀបកំណត់ម៉ោងរំលឹក (ម៉ោងនៅកម្ពុជា)៖</b>\n"
                        "• វាយ <b>/reminder 08:00</b> — ដើម្បីរំលឹកម៉ោង ៨:០០ ព្រឹក\n"
                        "• វាយ <b>/reminder 12:30</b> — ដើម្បីរំលឹកម៉ោង ១២:៣០ ថ្ងៃត្រង់\n"
                        "• វាយ <b>/reminder 19:00</b> — ដើម្បីរំលឹកម៉ោង ៧:០០ យប់\n\n"
                        "<b>ចំណាំ៖ ម៉ោងរំលឹកនឹងត្រូវបង្គត់ទៅរៀងរាល់ ១០នាទីម្តង។</b>"
                    )
                    await bot.send_message(chat_id, reminder_text)
                else:
                    reminder_text = (
                        "🔔 <b>ម៉ោងរំលឹកបច្ចុប្បន្នរបស់អ្នក (ម៉ោងនៅកម្ពុជា)៖</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"📅 <b>ថ្ងៃនេះ៖</b> <b>{today_date_str}</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                    )
                    inline_keyboard = []
                    for idx, r_time in enumerate(reminders, start=1):
                        reminder_text += f"{idx}. ⏰ ម៉ោង <b>{r_time}</b>\n"
                        inline_keyboard.append([
                            {
                                "text": f"🗑️ លុបម៉ោង {r_time}",
                                "callback_data": f"delete_reminder:{r_time}"
                            }
                        ])
                    
                    reminder_text += (
                        "\n💡 <b>គន្លឹះ៖</b>\n"
                        "• ដើម្បីបន្ថែមម៉ោងរំលឹកថ្មី៖ <b>/reminder 08:00</b>\n"
                        "• ដើម្បីលុបទាំងអស់៖ <b>/reminder clear</b>"
                    )
                    await bot.send_message(chat_id, reminder_text, reply_markup={"inline_keyboard": inline_keyboard})
                return

            action = parts[1].lower()
            if action == "clear":
                db_clear_reminders(user_id)
                await bot.send_message(
                    chat_id,
                    f"✅ <b>បានលុបម៉ោងរំលឹកទាំងអស់របស់អ្នករួចរាល់!</b>\n"
                    f"📅 <b>ថ្ងៃនេះ៖</b> <b>{today_date_str}</b>"
                )
                return
            
            try:
                time_str = parts[1]
                time_parts = time_str.split(":")
                if len(time_parts) != 2:
                    raise ValueError()
                
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                
                if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                    raise ValueError()
                
                formatted_time = f"{hour:02d}:{minute:02d}"
                db_add_reminder(user_id, formatted_time)
                
                await bot.send_message(
                    chat_id,
                    f"✅ <b>បានកំណត់ម៉ោងរំលឹករួចរាល់!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 <b>ថ្ងៃនេះ:</b> <code>{today_date_str}</code>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"ខ្ញុំនឹងផ្ញើសាររំលឹកអ្នកកុំឱ្យភ្លេចកត់ត្រាអាហារនៅម៉ោង <b>{formatted_time}</b> (ម៉ោងនៅកម្ពុជា) ជារៀងរាល់ថ្ងៃ។"
                )
            except ValueError:
                await bot.send_message(
                    chat_id,
                    "⚠️ <b>ទម្រង់ម៉ោងមិនត្រឹមត្រូវទេ!</b>\n"
                    "សូមផ្តល់ទម្រង់ម៉ោង <b>HH:MM</b> (២៤ម៉ោង)។ ឧទាហរណ៍៖ <b>/reminder 08:00</b> ឬ <b>/reminder 19:30</b>"
                )
            return
            
        elif text.startswith("/debugdb"):
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT datetime('now'), datetime('now', '+7 hours'), date('now', '+7 hours')")
                    time_row = cursor.fetchone()
                    
                    cursor.execute("SELECT burn_id, calories_burned, timestamp, date(timestamp, '+7 hours'), activity_name FROM burn_logs ORDER BY timestamp DESC LIMIT 5")
                    rows = cursor.fetchall()
                    
                    logs_str = ""
                    for r in rows:
                        logs_str += f"ID: {r[0]} | Cal: {r[1]} | UTC: {r[2]} | Shifted: {r[3]} | {r[4]}\n"
                        
                    reply = (
                        "🔍 <b>Database Debug Log</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"<b>UTC Now:</b> {time_row[0]}\n"
                        f"<b>Shifted Now:</b> {time_row[1]}\n"
                        f"<b>Shifted Date:</b> {time_row[2]}\n\n"
                        "<b>Last 5 Burn Logs:</b>\n"
                        f"<code>{logs_str}</code>"
                    )
                    await bot.send_message(chat_id, reply)
            except Exception as e:
                await bot.send_message(chat_id, f"Debug error: {e}")
            return

        elif text.startswith("/stats"):
            goal = db_get_user_goal(user_id)
            today_meals, total_cals = db_get_today_meals(user_id)
            total_burn = db_get_today_burn(user_id)
            remaining = goal - total_cals
            
            tot_protein = sum(m["protein"] for m in today_meals)
            tot_fat = sum(m["fat"] for m in today_meals)
            tot_carbs = sum(m["carbs"] for m in today_meals)
            tot_sugar = sum(m["sugar"] for m in today_meals)

            meal_list_str = ""
            if not today_meals:
                meal_list_str = "<b>មិនទាន់មានអាហារបានកត់ត្រាសម្រាប់ថ្ងៃនេះនៅឡើយទេ។ ផ្ញើរូបថតអាហារដើម្បីចាប់ផ្តើម!</b>"
            else:
                for idx, m in enumerate(today_meals, start=1):
                    meal_list_str += f"{idx}. <b>{m['food_name']}</b> ({m['calories']} Cal)\n"

            # Check if user logged /nosweet today in ICT
            nosweet_logged = db_check_today_nosweet(user_id)
            nosweet_status = "<b>ជោគជ័យ ✅</b>" if nosweet_logged else "<b>មិនទាន់កត់ត្រា ⏳</b> (វាយ /nosweet)"

            balance_emoji = "⚖️" if remaining >= 0 else "🚨"
            remaining_str = f"សល់ <b>{remaining} Cal</b>" if remaining >= 0 else f"លើស <b>{-remaining} Cal</b>"

            stats_text = (
                "📊 <b>របាយការណ៍សង្ខេបប្រចាំថ្ងៃ (UTC)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>គោលដៅប្រចាំថ្ងៃ៖</b> <b>{goal} Cal</b>\n"
                f"🔥 <b>បានញ៉ាំសរុប៖</b> <b>{total_cals} Cal</b>\n"
                f"🏃 <b>បានដុតរំលាយ៖</b> <b>{total_burn} Cal</b>\n"
                f"{balance_emoji} <b>ស្ថានភាព៖</b> {remaining_str}\n"
                f"🥤 <b>No Sweet Challenge៖</b> {nosweet_status}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<b>សារធាតុចិញ្ចឹមសរុប៖</b>\n"
                f"🥩 <b>ប្រូតេអ៊ីន៖</b> <b>{tot_protein}g</b>\n"
                f"🧈 <b>ខ្លាញ់សរុប៖</b> <b>{tot_fat}g</b>\n"
                f"🍞 <b>កាបូអ៊ីដ្រាត៖</b> <b>{tot_carbs}g</b>\n"
                f"🍬 <b>ស្ករ៖</b> <b>{tot_sugar}g</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<b>អាហារដែលបានញ៉ាំថ្ងៃនេះ៖</b>\n"
                f"{meal_list_str}"
            )

            inline_reply_markup = {
                "inline_keyboard": [
                    [
                        {
                            "text": "🗑️ សម្អាតកំណត់ត្រាថ្ងៃនេះ",
                            "callback_data": "reset_today"
                        }
                    ]
                ]
            }

            await bot.send_message(chat_id, stats_text, reply_markup=inline_reply_markup)
            return

        elif text.startswith("/burn"):
            parts = text.split()
            if len(parts) < 2:
                # Check Strava
                token_info = db_get_strava_tokens(user_id)
                if not token_info:
                    await bot.send_message(
                        chat_id,
                        "🏃 <b>កំណត់ត្រាការដុតរំលាយកាឡូរី (Strava)</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "គណនីរបស់អ្នកមិនទាន់បានភ្ជាប់ជាមួយ <b>Strava</b> នៅឡើយទេ។\n\n"
                        "👉 ដើម្បីភ្ជាប់គណនី សូមវាយបញ្ជា៖ <b>/strava</b>\n"
                        "👉 ដើម្បីកត់ត្រាកាឡូរីដោយផ្ទាល់ សូមវាយ៖ <b>/burn [ចំនួនកាឡូរី]</b> (ឧ. <b>/burn 350</b>)"
                    )
                    return
                
                # Connected, let's fetch the latest activity
                loading_msg = await bot.send_message(chat_id, "🔄 <i>កំពុងទាញយកលំហាត់ប្រាណចុងក្រោយពី Strava... សូមរង់ចាំមួយភ្លែត។</i>")
                loading_msg_id = loading_msg.get("result", {}).get("message_id")
                
                try:
                    session = await fetch_latest_strava_activity(user_id)
                    if session:
                        # Construct a unique key for the activity_name to prevent duplicates
                        act_key = f"{session['activity_name']} ({session['date_str']})"
                        
                        # Check if this exact session has already been logged in database
                        is_duplicate = False
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT 1 FROM burn_logs WHERE user_id = ? AND activity_name = ? AND source = 'Strava'",
                                (user_id, act_key)
                            )
                            if cursor.fetchone():
                                is_duplicate = True
                                
                        if is_duplicate:
                            duplicate_card = (
                                "✅ <b>លំហាត់ប្រាណនេះត្រូវបានកត់ត្រារួចហើយ</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                f"🚴 <b>សកម្មភាព៖</b> <b>{session['activity_name']}</b> ({session['session_name']})\n"
                                f"🔥 <b>ដុតកាឡូរី៖</b> <b>{session['calories']} Cal</b>\n"
                                f"⏲ <b>ពេលវេលា៖</b> <b>{session['duration']} នាទី</b>\n"
                                f"🗾 <b>ចម្ងាយ៖</b> <b>{session['distance']} គីឡូម៉ែត្រ</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "លំហាត់ប្រាណចុងក្រោយរបស់អ្នក ត្រូវបានកត់ត្រារក្សាទុករួចរាល់នៅក្នុងប្រព័ន្ធហើយ! 😉"
                            )
                            if loading_msg_id:
                                await bot.edit_message(chat_id, loading_msg_id, duplicate_card)
                            else:
                                await bot.send_message(chat_id, duplicate_card)
                        else:
                            # Not a duplicate, log it with its actual activity date
                            db_add_burn(user_id, session['calories'], act_key, "Strava", custom_date=session.get('activity_date'))
                            
                            activity_date = session.get('activity_date')
                            # Cambodia today
                            # import datetime (removed local import to prevent shadowing global name)
                            now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
                            today_str = now_kh.strftime("%Y-%m-%d")
                            
                            if activity_date and activity_date != today_str:
                                # Format display date
                                date_parts = activity_date.split('-')
                                formatted_display_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                                display_day = f"ថ្ងៃ {formatted_display_date}"
                                yesterday_str = (now_kh - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                                if activity_date == yesterday_str:
                                    display_day = "ម្សិលមិញ"
                                
                                success_card = (
                                    "🔥 <b>បានទាញយកលំហាត់ប្រាណចុងក្រោយជោគជ័យ!</b>\n"
                                    "━━━━━━━━━━━━━━━━━━━━\n"
                                    f"🚴 <b>សកម្មភាព៖</b> <b>{session['activity_name']}</b> ({session['session_name']})\n"
                                    f"🔥 <b>ដុតកាឡូរី៖</b> <b>{session['calories']} Cal</b>\n"
                                    f"⏲ <b>ពេលវេលា៖</b> <b>{session['duration']} នាទី</b>\n"
                                    f"🗾 <b>ចម្ងាយ៖</b> <b>{session['distance']} គីឡូម៉ែត្រ</b>\n"
                                    "━━━━━━━━━━━━━━━━━━━━\n"
                                    f"សកម្មភាពនេះត្រូវបានបន្ថែមទៅក្នុងកំណត់ត្រាដុតកាឡូរី <b>{display_day}</b> របស់អ្នករួចរាល់ហើយ! 💪"
                                )
                            else:
                                success_card = (
                                    "🔥 <b>បានទាញយកលំហាត់ប្រាណចុងក្រោយជោគជ័យ!</b>\n"
                                    "━━━━━━━━━━━━━━━━━━━━\n"
                                    f"🚴 <b>សកម្មភាព៖</b> <b>{session['activity_name']}</b> ({session['session_name']})\n"
                                    f"🔥 <b>ដុតកាឡូរី៖</b> <b>{session['calories']} Cal</b>\n"
                                    f"⏲ <b>ពេលវេលា៖</b> <b>{session['duration']} នាទី</b>\n"
                                    f"🗾 <b>ចម្ងាយ៖</b> <b>{session['distance']} គីឡូម៉ែត្រ</b>\n"
                                    "━━━━━━━━━━━━━━━━━━━━\n"
                                    "សកម្មភាពនេះត្រូវបានបន្ថែមទៅក្នុងកំណត់ត្រាដុតកាឡូរីថ្ងៃនេះរបស់អ្នករួចរាល់ហើយ! 💪"
                                )
                            if loading_msg_id:
                                await bot.edit_message(chat_id, loading_msg_id, success_card)
                            else:
                                await bot.send_message(chat_id, success_card)
                    else:
                        fail_msg = (
                            "⚠️ <b>មិនឃើញទិន្នន័យហាត់ប្រាណក្នុង Strava!</b>\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "រកមិនឃើញកំណត់ត្រាលំហាត់ប្រាណ ឬសកម្មភាពហាត់ប្រាណ (សកម្មភាព ៧ថ្ងៃចុងក្រោយ) នៅក្នុងគណនី Strava របស់អ្នកឡើយទេ។\n\n"
                            "💡 <b>ដំណោះស្រាយ៖</b>\n"
                            "1. សូមប្រាកដថានាឡិកា ឬកម្មវិធីសុខភាពរបស់អ្នកបាន Sync ជាមួយ Strava រួចរាល់។\n"
                            "2. អ្នកអាចកត់ត្រាកាឡូរីដោយផ្ទាល់ដោយវាយ៖ <b>/burn [ចំនួនកាឡូរី]</b>\n"
                            "ឧទាហរណ៍៖ <b>/burn 350</b>"
                        )
                        if loading_msg_id:
                            await bot.edit_message(chat_id, loading_msg_id, fail_msg)
                        else:
                            await bot.send_message(chat_id, fail_msg)
                except Exception as strava_err:
                    print(f"Error fetching latest Strava activity inside command: {strava_err}")
                    error_msg = (
                        "⚠️ <b>ការទាញយកទិន្នន័យបានបរាជ័យ</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"កំហុសបច្ចេកទេស៖\n<code>{strava_err}</code>\n\n"
                        "សូមព្យាយាមម្តងទៀត ឬកត់ត្រាដោយផ្ទាល់៖ <b>/burn [ចំនួនកាឡូរី]</b>"
                    )
                    if loading_msg_id:
                        await bot.edit_message(chat_id, loading_msg_id, error_msg)
                    else:
                        await bot.send_message(chat_id, error_msg)
                return
            
            # Parse tokens to find custom_date and calories
            tokens = parts[1:]
            calories = None
            custom_date = None
            date_token_src = None
            
            for tok in tokens:
                parsed_d = parse_custom_date(tok)
                if parsed_d:
                    custom_date = parsed_d
                    date_token_src = tok
                else:
                    try:
                        val = int(tok)
                        if 0 < val <= 10000:
                            calories = val
                    except ValueError:
                        pass
                        
            if calories is None:
                await bot.send_message(
                    chat_id,
                    "⚠️ <b>ចំនួនកាឡូរីមិនត្រឹមត្រូវទេ!</b>\n"
                    "សូមវាយបញ្ចូលចំនួនលេខវិជ្ជមានសមរម្យ។ ឧទាហរណ៍៖ <b>/burn 350</b>\n"
                    "ឬកត់ត្រាសម្រាប់ថ្ងៃមុន៖ <b>/burn 350 ម្សិលមិញ</b>"
                )
                return
                
            try:
                db_add_burn(user_id, calories, 'Manual', 'Manual', custom_date)
                if custom_date:
                    # import datetime (removed local import to prevent shadowing global name)
                    now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
                    date_parts = custom_date.split('-')
                    formatted_display_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                    display_day = f"ថ្ងៃ {formatted_display_date}"
                    if date_token_src and date_token_src.lower() in ["yesterday", "ម្សិលមិញ"]:
                        display_day = "ម្សិលមិញ"
                    elif date_token_src and date_token_src.lower() in ["ម្សិលម្ងៃ", "ម្សិលម្ង៉ៃ", "ម្សិលមិញមួយថ្ងៃ"]:
                        display_day = "ម្សិលម្ងៃ"
                    
                    await bot.send_message(
                        chat_id,
                        "🔥 <b>បានកត់ត្រាការដុតរំលាយជោគជ័យ!</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"🏃 <b>ថ្ងៃ {display_day} អ្នកបានដុតរំលាយ៖</b> <b>{calories} Cal</b>\n"
                        "💪 រក្សាសកម្មភាពរាងកាយល្អនេះបន្តទៀត!"
                    )
                else:
                    await bot.send_message(
                        chat_id,
                        "🔥 <b>បានកត់ត្រាការដុតរំលាយជោគជ័យ!</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"🏃 <b>ថ្ងៃនេះអ្នកបានដុតរំលាយ៖</b> <b>{calories} Cal</b>\n"
                        "💪 រក្សាសកម្មភាពរាងកាយល្អនេះបន្តទៀត!"
                    )
            except Exception as e:
                print(f"Error saving manual burn: {e}")
                await bot.send_message(chat_id, f"⚠️ <b>ការកត់ត្រាមានបញ្ហា៖</b> {e}")
            return
            
        elif text.startswith("/strava"):
            client_id = os.getenv("STRAVA_CLIENT_ID")
            client_secret = os.getenv("STRAVA_CLIENT_SECRET")
            redirect_uri = os.getenv("STRAVA_REDIRECT_URI")
            
            if not client_id or not client_secret or not redirect_uri:
                await bot.send_message(
                    chat_id,
                    "⚠️ <b>ការកំណត់មិនទាន់រួចរាល់!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "ម្ចាស់ប្រព័ន្ធមិនទាន់បានកំណត់ព័ត៌មានសម្ងាត់ Strava (Client ID, Client Secret, & Redirect URI) នៅក្នុងឯកសារ `.env` នៅឡើយទេ។"
                )
                return
                
            token_info = db_get_strava_tokens(user_id)
            if token_info:
                valid_token = await get_valid_strava_token(user_id, token_info)
                if valid_token:
                    status_text = "✅ <b>បានភ្ជាប់ជោគជ័យ! (Connected)</b>"
                else:
                    status_text = "⚠️ <b>បញ្ហាក្នុងការភ្ជាប់/ហួសសម័យ!</b>"
                    
                strava_card = (
                    "🏃 <b>Strava Integration Status</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"• ស្ថានភាព៖ {status_text}\n"
                    "• គណនី៖ គណនី Strava របស់អ្នកត្រូវបានភ្ជាប់រួចរាល់\n\n"
                    "💡 <b>មុខងារគាំទ្រ៖</b>\n"
                    "1. <b>Auto-Import Exercise:</b> វាយបញ្ជា <b>/burn</b> (ដោយគ្មានលេខកាឡូរី) ដើម្បីទាញយកសកម្មភាពហាត់ប្រាណចុងក្រោយរបស់ថ្ងៃនេះពី Strava មកកត់ត្រាក្នុង NutriBot ភ្លាមៗ!\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "ប្រសិនបើអ្នកចង់ផ្តាច់ការភ្ជាប់ សូមចុចប៊ូតុងខាងក្រោម៖"
                )
                inline_reply_markup = {
                    "inline_keyboard": [
                        [
                            {
                                "text": "🔌 ផ្តាច់ការភ្ជាប់ Strava",
                                "callback_data": "disconnect_strava"
                            }
                        ]
                    ]
                }
                await bot.send_message(chat_id, strava_card, reply_markup=inline_reply_markup)
            else:
                base_auth_url = redirect_uri.replace("/api/strava/callback", "/api/strava/auth")
                auth_url = f"{base_auth_url}?user_id={user_id}"
                
                welcome_card = (
                    "🏃 <b>ភ្ជាប់គណនីជាមួយ Strava</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "បង្កើនភាពងាយស្រួលដោយភ្ជាប់ NutriBot ជាមួយ Strava! ខ្ញុំនឹងជួយអ្នកក្នុង៖\n\n"
                    "• <b>ស្វ័យប្រវត្តកត់ត្រាដុតកាឡូរី៖</b> ទាញយកទិន្នន័យហាត់ប្រាណដែលអ្នកបានធ្វើពីគ្រប់ឧបករណ៍/កម្មវិធីសុខភាព (នាឡិកាឆ្លាតវៃ កម្មវិធីរត់ កង់...) មកកាន់ NutriBot ដោយគ្រាន់តែប្រើបញ្ជា <b>/burn</b>!\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "សូមចុចប៊ូតុងខាងក្រោមដើម្បីផ្ទៀងផ្ទាត់ និងភ្ជាប់គណនី Strava របស់អ្នក៖"
                )
                inline_reply_markup = {
                    "inline_keyboard": [
                        [
                            {
                                "text": "🔗 ភ្ជាប់ជាមួយ Strava",
                                "url": auth_url
                            }
                        ]
                    ]
                }
                await bot.send_message(chat_id, welcome_card, reply_markup=inline_reply_markup)
            return

        elif text.startswith("/nosweet"):
            already_logged = db_check_today_nosweet(user_id)
            if already_logged:
                await bot.send_message(
                    chat_id,
                    "🥤 <b>កំណត់ត្រាថ្ងៃនេះរួចរាល់ហើយ!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "អ្នកបានកត់ត្រាការតមភេសជ្ជៈផ្អែមសម្រាប់ថ្ងៃនេះរួចរាល់ហើយ។ រង់ចាំកត់ត្រាម្តងទៀតនៅថ្ងៃស្អែក! 😉"
                )
            else:
                db_add_nosweet_log(user_id)
                await bot.send_message(
                    chat_id,
                    "🥤 <b>No Sweet Drink Challenge!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "✅ <b>ថ្ងៃនេះអ្នកមិនបានញ៉ាំភេសជ្ជៈផ្អែមទេ!</b>\n"
                    "រក្សាការតស៊ូដ៏ល្អនេះបន្តទៀត ដើម្បីសុខភាពល្អ និងសម្រកទម្ងន់! 💪"
                )
            return

        elif text.startswith("/weight"):
            parts = text.split()
            profile = db_get_user_profile(user_id)
            old_weight = profile["weight"] if profile else None
            
            if len(parts) < 2:
                if old_weight is not None:
                    await bot.send_message(
                        chat_id,
                        f"⚖️ <b>បច្ចុប្បន្នភាពទម្ងន់របស់អ្នក</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"ទម្ងន់បច្ចុប្បន្នរបស់អ្នកគឺ៖ <b>{old_weight:.1f} kg</b>\n\n"
                        "ដើម្បីធ្វើបច្ចុប្បន្នភាពទម្ងន់ សូមវាយ៖ <b>/weight [ទម្ងន់ថ្មីជាគីឡូក្រាម]</b>\n"
                        "ឧទាហរណ៍៖ <b>/weight 68.5</b>"
                    )
                else:
                    await bot.send_message(
                        chat_id,
                        "⚖️ <b>បច្ចុប្បន្នភាពទម្ងន់របស់អ្នក</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "អ្នកមិនទាន់មានទិន្នន័យទម្ងន់ប្រវត្តិរូបនៅឡើយទេ។\n\n"
                        "សូមវាយ៖ <b>/weight [ទម្ងន់របស់អ្នកជាគីឡូក្រាម]</b>\n"
                        "ឧទាហរណ៍៖ <b>/weight 70</b>"
                    )
                return
            
            try:
                weight_val = float(parts[1])
                if weight_val <= 10 or weight_val > 500:
                    raise ValueError()
                
                # Check if profile exists and has enough details for recalculation
                if (profile and profile["gender"] and profile["age"] is not None 
                        and profile["height"] is not None and profile["activity"]):
                    
                    gender = profile["gender"]
                    age = profile["age"]
                    height = profile["height"]
                    activity = profile["activity"]
                    goal_type = profile["goal_type"] or "maintain"
                    
                    # Recalculate BMR
                    if gender == "male":
                        bmr = (10 * weight_val) + (6.25 * height) - (5 * age) + 5
                    else:
                        bmr = (10 * weight_val) + (6.25 * height) - (5 * age) - 161
                        
                    multipliers = {
                        "sedentary": 1.2,
                        "light": 1.375,
                        "moderate": 1.465,
                        "active": 1.55,
                        "very_active": 1.725
                    }
                    multiplier = multipliers.get(activity, 1.2)
                    maintain = bmr * multiplier
                    
                    # New calorie target calculations
                    if goal_type == "maintain":
                        new_goal = maintain
                    elif goal_type == "mild":
                        new_goal = maintain - 250
                    elif goal_type == "loss":
                        new_goal = maintain - 500
                    elif goal_type == "extreme":
                        new_goal = maintain - 1000
                    else:
                        new_goal = maintain
                        
                    new_goal = max(500, new_goal)
                    
                    # Save weight and recalculated targets to database
                    db_save_user_profile(user_id, gender, age, height, weight_val, activity)
                    db_update_tdee_goal(user_id, goal_type, int(new_goal))
                    
                    goal_type_kh = {
                        "maintain": "Maintain (រក្សាទម្ងន់)",
                        "mild": "Mild Loss (ស្រកតិចតួច)",
                        "loss": "Weight Loss (សម្រកទម្ងន់)",
                        "extreme": "Extreme Loss (សម្រកខ្លាំង)"
                    }.get(goal_type, goal_type)
                    
                    weight_change_str = ""
                    if old_weight is not None:
                        weight_change_str = f"• ⚖️ ទម្ងន់មុន៖ <b>{old_weight:.1f} kg</b>\n• 🎯 ទម្ងន់ថ្មី៖ <b>{weight_val:.1f} kg</b>"
                    else:
                        weight_change_str = f"• ⚖️ ទម្ងន់បច្ចុប្បន្ន៖ <b>{weight_val:.1f} kg</b>"
                        
                    await bot.send_message(
                        chat_id,
                        f"✅ <b>ទម្ងន់ និងគោលដៅត្រូវបានធ្វើបច្ចុប្បន្នភាព!</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"{weight_change_str}\n\n"
                        "🔄 <b>គណនាគោលដៅឡើងវិញស្វ័យប្រវត្ត៖</b>\n"
                        f"• TDEE ថ្មី៖ <b>{maintain:.0f} Cal</b>\n"
                        f"• ប្រភេទគោលដៅ៖ <b>{goal_type_kh}</b>\n"
                        f"• គោលដៅកាឡូរីប្រចាំថ្ងៃ៖ <b>{new_goal:.0f} Cal</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "🎉 គោលដៅកាឡូរីថ្មីត្រូវបានអនុវត្តដោយជោគជ័យ!"
                    )
                else:
                    # Update only the weight column in the database
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("UPDATE users SET weight = ? WHERE user_id = ?", (weight_val, user_id))
                        conn.commit()
                        
                    weight_change_str = ""
                    if old_weight is not None:
                        weight_change_str = f"• ⚖️ ទម្ងន់មុន៖ <b>{old_weight:.1f} kg</b>\n• 🎯 ទម្ងន់ថ្មី៖ <b>{weight_val:.1f} kg</b>"
                    else:
                        weight_change_str = f"• ⚖️ ទម្ងន់បច្ចុប្បន្ន៖ <b>{weight_val:.1f} kg</b>"
                        
                    await bot.send_message(
                        chat_id,
                        f"✅ <b>បានកត់ត្រាទម្ងន់រួចរាល់!</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"{weight_change_str}\n\n"
                        "💡 ដើម្បីគណនាគោលដៅកាឡូរីប្រចាំថ្ងៃដោយស្វ័យប្រវត្តតាមទម្ងន់ថ្មីនេះ សូមវាយ <b>/cal</b> ដើម្បីបំពេញប្រវត្តិរូបរបស់អ្នក!"
                    )
            except ValueError:
                await bot.send_message(
                    chat_id,
                    "⚠️ <b>ទម្ងន់មិនត្រឹមត្រូវទេ!</b>\n"
                    "សូមផ្តល់ទម្ងន់ជាលេខវិជ្ជមានសមរម្យ។ ឧទហរណ៍៖ <b>/weight 75.5</b>"
                )
            return

        elif text.startswith("/weekly"):
            # Get current time in ICT (UTC+7)
            now_utc = datetime.datetime.utcnow()
            now_cambodia = now_utc + datetime.timedelta(hours=7)
            
            # Find the Monday of the current week (weekday() returns 0 for Monday, 6 for Sunday)
            current_weekday = now_cambodia.weekday()
            monday_date = now_cambodia.date() - datetime.timedelta(days=current_weekday)
            
            # Generate the 7 dates from Monday through Sunday
            week_dates = [monday_date + datetime.timedelta(days=i) for i in range(7)]
            start_date_str = week_dates[0].strftime("%Y-%m-%d")
            end_date_str = week_dates[-1].strftime("%Y-%m-%d")
            
            # Fetch user's daily budget goal
            goal = db_get_user_goal(user_id)
            
            # Fetch weekly data in single DB call per query types
            meal_rows, burn_rows = db_get_weekly_stats(user_id, start_date_str, end_date_str)
            nosweet_dates = db_get_weekly_nosweet(user_id, start_date_str, end_date_str)
            today_date = now_cambodia.date()
            
            # Aggregate stats by day
            stats_by_date = {}
            for d in week_dates:
                stats_by_date[d.strftime("%Y-%m-%d")] = {"intake": 0, "burn": 0}
                
            for cals, date_str in meal_rows:
                if date_str in stats_by_date:
                    stats_by_date[date_str]["intake"] += cals
                    
            for burn, date_str in burn_rows:
                if date_str in stats_by_date:
                    stats_by_date[date_str]["burn"] += burn
                    
            # Build beautifully formatted list in standard text
            report_lines = []
            total_weekly_eaten = 0
            total_weekly_burned = 0
            
            day_names_short = {
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun"
            }
            
            for idx, d in enumerate(week_dates):
                date_str = d.strftime("%Y-%m-%d")
                day_short = day_names_short.get(idx, "Day")
                
                intake = stats_by_date[date_str]["intake"]
                burn = stats_by_date[date_str]["burn"]
                left = goal - intake
                
                total_weekly_eaten += intake
                total_weekly_burned += burn
                
                # Show remaining or exceeded status cleanly
                if left >= 0:
                    left_str = f"សល់ {left}"
                else:
                    left_str = f"លើស {-left}"
                    
                # Determine No Sweet Challenge visual marker
                if date_str in nosweet_dates:
                    nosweet_marker = "🥤 ✅"
                else:
                    if d < today_date:
                        nosweet_marker = "🥤 ❌"
                    else:
                        nosweet_marker = "🥤 ⏳"
                    
                report_lines.append(
                    f"{day_short}៖ ញ៉ាំ {intake} | {left_str} | ដុត {burn} | {nosweet_marker}"
                )
            
            # Weekly overall calculations
            weekly_budget = goal * 7
            overall_left = weekly_budget - total_weekly_eaten
            if overall_left >= 0:
                overall_left_str = f"សល់ <b>{overall_left}</b> Cal"
            else:
                overall_left_str = f"លើស <b>{-overall_left}</b> Cal"
                
            weekly_report_text = (
                "📅 <b>របាយការណ៍សង្ខេបប្រចាំសប្តាហ៍ (ច័ន្ទ - អាទិត្យ)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>គោលដៅប្រចាំថ្ងៃ៖</b> <b>{goal}</b> Cal\n"
                f"🗓️ <b>សប្តាហ៍បច្ចុប្បន្ន៖</b> <b>{start_date_str}</b> ដល់ <b>{end_date_str}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                + "\n".join(report_lines) + "\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📊 <b>សរុបសប្តាហ៍នេះ៖</b>\n"
                f"🍳 <b>ញ៉ាំសរុប៖</b> <b>{total_weekly_eaten}</b> / <b>{weekly_budget}</b> Cal\n"
                f"⚖️ <b>ស្ថានភាព៖</b> {overall_left_str}\n"
                f"🏃 <b>ដុតសរុប៖</b> <b>{total_weekly_burned}</b> Cal"
            )
            
            await bot.send_message(chat_id, weekly_report_text)
            return

        elif text.startswith("/suggest") or text.startswith("/menu"):
            inline_keyboard = [
                [
                    {"text": "🥗 បន្លែច្រើន (High Veg)", "callback_data": "suggest_pref:veg"},
                    {"text": "🥩 សាច់ច្រើន (High Meat)", "callback_data": "suggest_pref:meat"}
                ],
                [
                    {"text": "🍲 ម្ហូបធម្មតា (Standard Khmer)", "callback_data": "suggest_pref:normal"}
                ]
            ]
            await bot.send_message(
                chat_id,
                "💡 <b>ជ្រើសរើសប្រភេទមុខម្ហូបណែនាំដែលអ្នកចូលចិត្ត៖</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "ខ្ញុំនឹងរៀបចំសំណើមុខម្ហូបប្រចាំថ្ងៃ ស្របតាមកម្រិតកាឡូរីរបស់អ្នក និងគ្រឿងផ្សំងាយស្រួលរកក្នុងផ្សារខ្មែរ!",
                reply_markup={"inline_keyboard": inline_keyboard}
            )
            return

        else:
            # All text (slash commands or plain chat) — completely silent, no reply
            return


    # 2. Handle photo upload (compressed photo OR uncompressed document image)
    elif photo or (document and document.get("mime_type", "").startswith("image/")):
        # Extract the caption if any and parse potential custom date
        caption = message.get("caption", "").strip() if message else ""
        custom_date = None
        user_food_context = ""
        
        if caption:
            caption_parts = caption.split(maxsplit=1)
            if len(caption_parts) > 0:
                potential_date = caption_parts[0]
                parsed_date = parse_custom_date(potential_date)
                if parsed_date:
                    custom_date = parsed_date
                    if len(caption_parts) > 1:
                        user_food_context = caption_parts[1]
                else:
                    user_food_context = caption
                    
        display_date = custom_date if custom_date else "ថ្ងៃនេះ"
        
        # Send initial premium loading visual to user immediately
        ack = await bot.send_message(
            chat_id, 
            f"🔍 <i>កំពុងវិភាគរូបភាពអាហារសម្រាប់ {display_date}... សូមរង់ចាំមួយភ្លែត។</i>" if custom_date
            else "🔍 <i>កំពុងវិភាគរូបភាពអាហាររបស់អ្នក... សូមរង់ចាំមួយភ្លែត។</i>"
        )
        ack_message_id = ack.get("result", {}).get("message_id")

        try:
            # Extract the correct file ID
            if photo:
                file_id = photo[-1]["file_id"]  # Highest resolution
            else:
                file_id = document["file_id"]

            image_bytes, mime_type = await bot.get_file_bytes(file_id)

            # Initialize modern Gemini SDK Client (automatically reads GEMINI_API_KEY)
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise ValueError("GEMINI_API_KEY environment variable is not configured.")

            client = genai.Client()
            
            # Setup a robust fallback model chain for rate-limit / resource issues
            user_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash") # Default to standard 2.0-flash for free quotas
            models_to_try = [user_model]
            for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

            # Get current Cambodia ICT local time for time-aware coaching context
            now_utc = datetime.datetime.utcnow()
            now_cambodia = now_utc + datetime.timedelta(hours=7)
            time_str = now_cambodia.strftime("%I:%M %p")
            day_name = now_cambodia.strftime("%A")
            
            # Determine period of the day
            hour = now_cambodia.hour
            if 5 <= hour < 11:
                period_kh = "ពេលព្រឹក (Morning)"
            elif 11 <= hour < 14:
                period_kh = "ពេលថ្ងៃត្រង់ (Lunch)"
            elif 14 <= hour < 17:
                period_kh = "ពេលរសៀល (Afternoon)"
            elif 17 <= hour < 22:
                period_kh = "ពេលល្ងាច/យប់ (Evening/Night)"
            else:
                period_kh = "ពេលយប់ជ្រៅ (Late Night)"

            profile = db_get_user_profile(user_id)
            if profile:
                profile_context = (
                    f"The user is a {profile['gender']}, {profile['age']} years old, {profile['height']:.1f} cm tall, "
                    f"weighing {profile['weight']:.1f} kg. Their physical activity level is mapped as '{profile['activity']}'. "
                    f"Their daily budget goal is {profile['daily_calorie_budget']} Cal and their goal type is '{profile['goal_type']}'."
                )
            else:
                profile_context = "The user is a general individual with a daily budget of 2000 Cal aiming to maintain weight."

            logging_time_context = f"Current Cambodia local time is {time_str} on {day_name} ({period_kh})."
            if custom_date:
                logging_time_context = f"The user is retroactively logging for the Cambodia local date {custom_date}."

            photo_system_prompt = (
                "You are a professional nutrition expert and health coach. Analyze the food in the provided image and estimate its "
                "nutritional details (calories in Cal, protein/fat/carbs/sugar in grams).\n"
                f"User Health Context: {profile_context}\n"
                f"Logging Context: {logging_time_context}\n"
                "YOU MUST RESPOND ENTIRELY IN KHMER LANGUAGE. The `food_name` field must be written in beautiful Khmer script.\n"
                "Provide a highly personalized coaching and health recommendation (in the `coaching_recommendation` field) "
                "in Khmer tailored specifically to this user's profile, goal, and the logging context.\n"
                "CRITICAL SECRECY RULE: You know the user's age, weight, height, and calorie target budget from the User Health Context, BUT YOU MUST KEEP THEM SECRET. Never mention or repeat their age, weight, height, or daily calorie goal in your coaching_recommendation text response. Focus purely on qualitative health insights, digestion, macronutrients, and positive coaching advice.\n"
                "Do NOT recite or repeat raw numbers (like '150 Cal' or '10g protein') inside the coaching recommendation text since those are already clearly displayed in the summary card.\n"
                "If the image does not show any food, or you cannot identify any food, "
                "you MUST set the `confidence_score` to less than 0.5 (e.g. 0.0 to 0.4), "
                "and you can set the `food_name` to 'មិនមែនជាអាហារ ឬរកមិនឃើញ'. "
                "Be realistic, objective, and estimate standard portion sizes for single servings unless "
                "there's strong visual context stating otherwise."
            )

            response = None
            last_error = None

            # Attempt each model in the fallback chain sequentially
            for current_model in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=current_model,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                            f"Analyze the food in this image and return its nutrition facts in Khmer. Description context from user: {user_food_context}" if user_food_context else "Analyze the food in this image and return its nutrition facts in Khmer."
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=photo_system_prompt,
                            response_mime_type="application/json",
                            response_schema=FoodAnalysis,
                        ),
                    )
                    break
                except Exception as model_err:
                    last_error = model_err
                    print(f"⚠️ Model {current_model} failed or is rate-limited: {model_err}")
                    continue

            # If all models failed
            if response is None:
                raise ValueError(f"All generative models failed. Last error: {last_error}")

            # Validate output using Pydantic
            analysis = FoodAnalysis.model_validate_json(response.text)

            # Check for non-food or low confidence edge cases
            if analysis.confidence_score < 0.5:
                err_msg = (
                    "🍳 <b>អូ! ខ្ញុំរកមិនឃើញអាហារទេ!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "ខ្ញុំមិនសូវច្បាស់ថាវាជាអាហារនោះទេ។ សូមប្រាកដថារូបថតបង្ហាញពីអាហារបានច្បាស់ល្អ មានពន្លឺគ្រប់គ្រាន់ រួចផ្ញើមកម្តងទៀត!\n\n"
                    f"<b>(រកឃើញ៖ {analysis.food_name} | កម្រិតច្បាស់លាស់៖ {analysis.confidence_score * 100:.0f}%)</b>"
                )
                if ack_message_id:
                    await bot.edit_message(chat_id, ack_message_id, err_msg)
                else:
                    await bot.send_message(chat_id, err_msg)
                return

            # Save meal details to database and get primary key ID
            inserted_meal_id = db_add_meal(user_id, analysis, custom_date)
            await sync_meal_to_google_fit(user_id, analysis)

            # Fetch remaining calories
            if custom_date:
                today_meals, total_cals = db_get_day_meals(user_id, custom_date)
                total_burn = db_get_day_burn(user_id, custom_date)
            else:
                today_meals, total_cals = db_get_today_meals(user_id)
                total_burn = db_get_today_burn(user_id)
                
            goal = db_get_user_goal(user_id)
            remaining = goal - total_cals
            balance_emoji = "⚖️" if remaining >= 0 else "🚨"
            remaining_str = f"សល់ <b>{remaining} Cal</b>" if remaining >= 0 else f"លើស <b>{-remaining} Cal</b>"

            # Format custom display date
            if custom_date:
                date_parts = custom_date.split('-')
                formatted_display_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            else:
                formatted_display_date = now_cambodia.strftime('%d-%m-%Y')

            # Format the output beautifully using HTML tags
            result_card = (
                "🍳 <b>លទ្ធផលវិភាគអាហារូបត្ថម្ភ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🥗 <b>អាហារ៖</b> <b>{analysis.food_name}</b>\n"
                f"📊 <b>ភាពជឿជាក់៖</b> <b>{analysis.confidence_score * 100:.0f}%</b>\n"
                f"📅 <b>កាលបរិច្ឆេទ៖</b> <b>{formatted_display_date}</b>\n\n"
                f"🔥 <b>ថាមពល៖</b> <b>{analysis.calories} Cal</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🥩 <b>ប្រូតេអ៊ីន៖</b> <b>{analysis.protein}g</b>\n"
                f"🧈 <b>ខ្លាញ់សរុប៖</b> <b>{analysis.fat}g</b>\n"
                f"🍞 <b>កាបូអ៊ីដ្រាត៖</b> <b>{analysis.carbs}g</b>\n"
                f"🍬 <b>ក្នុងនោះជាតិស្ករ៖</b> <b>{analysis.sugar}g</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🏃 <b>បានដុតរំលាយ៖</b> <b>{total_burn} Cal</b>\n"
                f"{balance_emoji} <b>កាឡូរី ({display_date})៖</b> <b>{total_cals}</b> / <b>{goal} Cal</b> ({remaining_str})\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <b>ការណែនាំពីគ្រូ៖</b>\n"
                f"« {analysis.coaching_recommendation} »\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "💾 <b>បានកត់ត្រាចូលគណនីរួចរាល់! ប្រសិនបើចង់លុបកំណត់ត្រានេះវិញ សូមចុចប៊ូតុងខាងក្រោម៖</b>"
            )

            # Define inline button to clear the meal log dynamically in Khmer
            inline_reply_markup = {
                "inline_keyboard": [
                    [
                        {
                            "text": "❌ លុបកំណត់ត្រានេះ",
                            "callback_data": f"delete_meal:{inserted_meal_id}"
                        }
                    ]
                ]
            }

            if ack_message_id:
                await bot.edit_message(chat_id, ack_message_id, result_card, reply_markup=inline_reply_markup)
            else:
                await bot.send_message(chat_id, result_card, reply_markup=inline_reply_markup)

        except Exception as e:
            print(f"Error during food analysis: {e}")
            fail_msg = (
                "⚠️ <b>ការវិភាគអាហារូបត្ថម្ភបានបរាជ័យ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "មានបញ្ហាបច្ចេកទេសមួយបានកើតឡើងក្នុងពេលដំណើរការវិភាគរូបភាពរបស់អ្នក។ សូមប្រាកដថាព័ត៌មានសម្ងាត់ Turso និង Gemini API ត្រូវបានកំណត់ត្រឹមត្រូវនៅលើ Vercel។\n\n"
                f"<b>ព័ត៌មានលម្អិត:</b> <code>{str(e)}</code>"
            )
            if ack_message_id:
                await bot.edit_message(chat_id, ack_message_id, fail_msg)
            else:
                await bot.send_message(chat_id, fail_msg)
            return
    else:
        # Acknowledge unhandled update types safely
        await bot.send_message(
            chat_id,
            "ℹ️ <b>រកឃើញតែប្រភេទរូបថតប៉ុណ្ណោះ!</b>\n"
            "សូមផ្ញើរូបថត ឬឯកសាររូបភាពអាហាររបស់អ្នក ដើម្បីវិភាគតម្លៃអាហារូបត្ថម្ភ។"
        )

# ---------------------------------------------------------
# Google Fit Integration Helpers & Endpoints
# ---------------------------------------------------------
async def get_valid_fit_token(user_id: int, token_info: dict) -> str:
    """Checks if access token is expired or expiring soon, refreshes if necessary, and returns it."""
    import time
    expires_at = token_info.get("expires_at")
    
    # If expired or expires in less than 5 minutes (300 seconds)
    if expires_at is None or time.time() >= expires_at - 300:
        refresh_token = token_info.get("refresh_token")
        if not refresh_token:
            return None
            
        client_id = os.getenv("GOOGLE_FIT_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_FIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            print("Missing Google Fit Client credentials in .env")
            return None
            
        # Refresh access token
        url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, data=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    new_access_token = data["access_token"]
                    expires_in = data.get("expires_in", 3600)
                    # Update token in DB without wiping refresh token
                    db_update_access_token(user_id, new_access_token, expires_in)
                    return new_access_token
                else:
                    print(f"Failed to refresh Google Fit token: {resp.text}")
                    if "invalid_grant" in resp.text:
                        db_delete_fit_tokens(user_id)
        except Exception as e:
            print(f"Error refreshing access token for {user_id}: {e}")
        return None
    
    return token_info.get("access_token")

async def get_or_create_nutrition_datasource(access_token: str) -> str:
    """Finds or creates a Google Fit data source for com.google.nutrition."""
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    # Check existing data sources
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://www.googleapis.com/fitness/v1/users/me/dataSources", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for ds in data.get("dataSource", []):
                    if ds.get("dataType", {}).get("name") == "com.google.nutrition":
                        return ds.get("dataStreamId")
    except Exception as e:
        print(f"Error listing data sources: {e}")

    # Not found, let's create a raw data source
    payload = {
        "dataStreamName": "NutriBotNutritionStream",
        "type": "raw",
        "application": {
            "name": "NutriBot",
            "version": "1.0.0"
        },
        "dataType": {
            "name": "com.google.nutrition"
        }
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://www.googleapis.com/fitness/v1/users/me/dataSources", headers=headers, json=payload)
            if resp.status_code in [200, 201]:
                return resp.json().get("dataStreamId")
            elif resp.status_code == 409: # Conflict, already exists
                resp_get = await client.get("https://www.googleapis.com/fitness/v1/users/me/dataSources", headers=headers)
                if resp_get.status_code == 200:
                    data = resp_get.json()
                    for ds in data.get("dataSource", []):
                        if ds.get("dataType", {}).get("name") == "com.google.nutrition":
                            return ds.get("dataStreamId")
    except Exception as e:
        print(f"Error creating com.google.nutrition data source: {e}")
    return None

async def get_valid_strava_token(user_id: int, token_info: dict) -> str:
    """Checks if Strava access token is expired, refreshes it using the refresh token, and returns it."""
    import time
    expires_at = token_info.get("expires_at")
    
    # If expired or expires in less than 5 minutes (300 seconds)
    if expires_at is None or time.time() >= expires_at - 300:
        refresh_token = token_info.get("refresh_token")
        if not refresh_token:
            return None
            
        client_id = os.getenv("STRAVA_CLIENT_ID")
        client_secret = os.getenv("STRAVA_CLIENT_SECRET")
        if not client_id or not client_secret:
            print("Missing Strava Client credentials in .env")
            return None
            
        # Refresh access token from Strava
        url = "https://www.strava.com/oauth/token"
        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, data=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    new_access_token = data["access_token"]
                    new_expires_at = data["expires_at"]
                    # Update token in DB
                    db_update_strava_access_token(user_id, new_access_token, new_expires_at)
                    return new_access_token
                else:
                    print(f"Failed to refresh Strava token: {resp.text}")
                    if "invalid_grant" in resp.text:
                        db_delete_strava_tokens(user_id)
        except Exception as e:
            print(f"Error refreshing Strava access token for {user_id}: {e}")
        return None
    
    return token_info.get("access_token")

async def fetch_latest_strava_activity(user_id: int) -> dict:
    """Fetches the single most recent exercise session and its exact calories from Strava in the last 7 days."""
    token_info = db_get_strava_tokens(user_id)
    if not token_info:
        return None
        
    access_token = await get_valid_strava_token(user_id, token_info)
    if not access_token:
        return None
        
    import datetime
    import time
    
    # Fetch last 7 days of activities in Cambodia time (ICT - UTC+7)
    now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    start_of_today_kh = datetime.datetime(now_kh.year, now_kh.month, now_kh.day, 0, 0, 0)
    start_of_7days_ago_kh = start_of_today_kh - datetime.timedelta(days=7)
    start_of_7days_ago_utc = start_of_7days_ago_kh - datetime.timedelta(hours=7)
    after_timestamp = int(start_of_7days_ago_utc.replace(tzinfo=datetime.timezone.utc).timestamp())
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    url = f"https://www.strava.com/api/v3/athlete/activities?after={after_timestamp}&per_page=5"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"Strava activities API returned status {resp.status_code}: {resp.text}")
                
            activities = resp.json()
            
            if not activities:
                raise Exception("គ្មានសកម្មភាពលំហាត់ប្រាណដែលបានកត់ត្រាក្នុងរយៈពេល ៧ថ្ងៃចុងក្រោយនេះទេ! សូមកត់ត្រាការហាត់ប្រាណនៅលើកម្មវិធី Strava ជាមុនសិន។")
            
            # Strava API returns activities descending (latest first), so index 0 is already the most recent activity
            latest_act = activities[0]
            
            activity_id = latest_act.get("id")
            session_name = latest_act.get("name", "Workout")
            act_type = latest_act.get("type", "Workout")
            
            type_mappings = {
                "Run": "រត់ (Running)",
                "Ride": "ជិះកង់ (Biking)",
                "Walk": "ដើរ (Walking)",
                "Hike": "ដើរភ្នំ (Hiking)",
                "Swim": "ហែលទឹក (Swimming)",
                "WeightTraining": "លើកទម្ងន់ (Weight Lifting)",
                "Workout": "ហាត់ប្រាណទូទៅ (Workout)",
                "Yoga": "យូហ្គា (Yoga)",
                "Elliptical": "ម៉ាស៊ីន Elliptical"
            }
            act_name = type_mappings.get(act_type, f"ហាត់ប្រាណ {act_type}")
            
            # Extract duration in minutes
            duration_minutes = int(latest_act.get("moving_time", 0) / 60.0)
            if duration_minutes < 1:
                duration_minutes = int(latest_act.get("elapsed_time", 0) / 60.0)
            if duration_minutes < 1:
                duration_minutes = 30
                
            # Extract distance in kilometers
            distance_meters = latest_act.get("distance", 0)
            distance_km = round(distance_meters / 1000.0, 1)
            
            # Extract calories or work done
            calories_burned = latest_act.get("calories", 0)
            if calories_burned < 1:
                calories_burned = latest_act.get("kilojoules", 0)
            if calories_burned < 1:
                calories_burned = int(duration_minutes * 6.5) # standard estimate
                
            start_date_local = latest_act.get("start_date_local")
            activity_date = None
            if start_date_local:
                try:
                    clean_date = start_date_local.replace("Z", "")
                    dt = datetime.datetime.fromisoformat(clean_date)
                    date_str = dt.strftime("%d-%m-%Y %I:%M %p")
                    activity_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
                    activity_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")
            else:
                date_str = datetime.datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
                activity_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")
                
            return {
                "activity_type": act_type,
                "activity_name": act_name,
                "session_name": session_name,
                "calories": int(calories_burned),
                "duration": duration_minutes,
                "distance": distance_km,
                "date_str": date_str,
                "activity_date": activity_date,
                "activity_id": activity_id
            }
    except Exception as e:
        print(f"Error fetching latest Strava activity: {e}")
        raise e

async def sync_meal_to_google_fit(user_id: int, analysis: FoodAnalysis):
    """Syncs user logged nutrition data to their connected Google Fit account."""
    try:
        token_info = db_get_fit_tokens(user_id)
        if not token_info:
            return # Not connected
        
        # Get valid access token (refreshing if needed)
        access_token = await get_valid_fit_token(user_id, token_info)
        if not access_token:
            return
            
        # Determine meal_type based on Cambodia ICT time (UTC+7)
        now_utc = datetime.datetime.utcnow()
        now_cambodia = now_utc + datetime.timedelta(hours=7)
        hour = now_cambodia.hour
        if 5 <= hour < 11:
            meal_type = 2 # Breakfast
        elif 11 <= hour < 14:
            meal_type = 3 # Lunch
        elif 17 <= hour < 22:
            meal_type = 4 # Dinner
        else:
            meal_type = 5 # Snack
            
        # We need the com.google.nutrition data source
        datasource_id = await get_or_create_nutrition_datasource(access_token)
        if not datasource_id:
            return
            
        # Timestamps in nanoseconds
        timestamp_ns = int(now_utc.timestamp() * 1000000000)
        
        # Build standard Google Fit nutrition dataset entry
        payload = {
            "dataSourceId": datasource_id,
            "minStartTimeNs": timestamp_ns,
            "maxEndTimeNs": timestamp_ns,
            "point": [
                {
                    "startTimeNanos": timestamp_ns,
                    "endTimeNanos": timestamp_ns,
                    "dataTypeName": "com.google.nutrition",
                    "value": [
                        {
                            "mapVal": [
                                { "key": "calories", "value": { "fpVal": float(analysis.calories) } },
                                { "key": "carbs.total", "value": { "fpVal": float(analysis.carbs) } },
                                { "key": "fat.total", "value": { "fpVal": float(analysis.fat) } },
                                { "key": "protein", "value": { "fpVal": float(analysis.protein) } },
                                { "key": "sugar", "value": { "fpVal": float(analysis.sugar) } }
                            ]
                        },
                        { "intVal": meal_type },
                        { "strVal": analysis.food_name }
                    ]
                }
            ]
        }
        
        url = f"https://www.googleapis.com/fitness/v1/users/me/dataSources/{datasource_id}/datasets/{timestamp_ns}-{timestamp_ns}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            resp = await client.patch(url, json=payload)
            if resp.status_code not in [200, 201]:
                print(f"Failed to write nutrition to Google Fit: {resp.text}")
            else:
                print(f"Successfully synced meal '{analysis.food_name}' to Google Fit.")
    except Exception as e:
        print(f"Error syncing meal to Google Fit for {user_id}: {e}")

async def fetch_fit_exercises_today(user_id: int) -> list[dict]:
    """Fetches exercise activities and calories burned from Google Fit for today (Cambodia time, UTC+7)."""
    token_info = db_get_fit_tokens(user_id)
    if not token_info:
        return []
        
    access_token = await get_valid_fit_token(user_id, token_info)
    if not access_token:
        return []
        
    # Calculate start and end of Cambodia local date 'today' (ICT, UTC+7)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_cambodia = now_utc + datetime.timedelta(hours=7)
    
    # Today 00:00:00 local time
    start_local = datetime.datetime(now_cambodia.year, now_cambodia.month, now_cambodia.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
    # Today 23:59:59 local time
    end_local = datetime.datetime(now_cambodia.year, now_cambodia.month, now_cambodia.day, 23, 59, 59, tzinfo=datetime.timezone.utc)
    
    # Convert local times to UTC by subtracting 7 hours
    start_utc = start_local - datetime.timedelta(hours=7)
    end_utc = end_local - datetime.timedelta(hours=7)
    
    # Milliseconds since epoch (explicit timezone-aware timestamp conversion)
    start_millis = int(start_utc.timestamp() * 1000)
    end_millis = int(end_utc.timestamp() * 1000)
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "aggregateBy": [
            {
                "dataTypeName": "com.google.calories.expended"
            },
            {
                "dataTypeName": "com.google.activity.segment"
            }
        ],
        "startTimeMillis": start_millis,
        "endTimeMillis": end_millis,
        "bucketByActivityType": {}
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate", headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"Google Fit aggregate API failed: {resp.text}")
                return []
                
            data = resp.json()
            exercises = []
            
            activity_names = {
                1: "ជិះកង់ (Biking)",
                2: "ហាត់ប្រាណ Calisthenics",
                7: "ដើរ (Walking)",
                8: "រត់ (Running)",
                9: "អេរ៉ូប៊ិក (Aerobics)",
                10: "វាយសី (Badminton)",
                11: "បេស្បល (Baseball)",
                12: "បាល់បោះ (Basketball)",
                20: "ប្រដាល់ (Boxing)",
                24: "រាំ (Dancing)",
                31: "ធ្វើសួន (Gardening)",
                32: "វាយកូនហ្គោល (Golf)",
                35: "ដើរភ្នំ (Hiking)",
                53: "អុំទូក (Rowing)",
                58: "រត់លើម៉ាស៊ីន (Treadmill Running)",
                97: "លើកទម្ងន់ (Weight Lifting)",
                100: "ហែលទឹក (Swimming)",
                108: "ហាត់ប្រាណទូទៅ (Workout)",
                113: "ហាត់ប្រាណ Fitness",
                114: "យូហ្គា (Yoga)",
                115: "ម៉ាស៊ីន Elliptical",
                116: "Zumba"
            }
            
            for bucket in data.get("bucket", []):
                activity_type = bucket.get("activityType")
                
                # STRICT WHITELIST: Only import exercises explicitly listed in our active dictionary.
                # This completely filters out sleeping, resting, still, BMR, or generic unmapped types.
                if activity_type not in activity_names:
                    continue
                    
                calories_burned = 0
                for dataset in bucket.get("dataset", []):
                    for point in dataset.get("point", []):
                        for value in point.get("value", []):
                            if "fpVal" in value:
                                calories_burned += value["fpVal"]
                            elif "intVal" in value:
                                calories_burned += value["intVal"]
                                
                if calories_burned >= 1:
                    act_name = activity_names[activity_type]
                    exercises.append({
                        "activity_type": activity_type,
                        "activity_name": act_name,
                        "calories": int(calories_burned)
                    })
            return exercises
    except Exception as e:
        print(f"Error fetching today's exercises from Google Fit: {e}")
    return []

async def fetch_latest_fit_session(user_id: int) -> dict:
    """Fetches the single most recent exercise session and its exact calories from Google Fit."""
    token_info = db_get_fit_tokens(user_id)
    if not token_info:
        return None
        
    access_token = await get_valid_fit_token(user_id, token_info)
    if not access_token:
        return None
        
    import datetime
    seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    start_time_iso = seven_days_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    sessions_url = f"https://www.googleapis.com/fitness/v1/users/me/sessions?startTime={start_time_iso}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(sessions_url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"Google Fit sessions API returned status {resp.status_code}: {resp.text}")
                
            data = resp.json()
            sessions = data.get("session", [])
            
            if not sessions:
                raise Exception("Google Fit returned 0 sessions in the last 7 days. Please make sure you have pulled down to refresh on the Google Fit app's Journal tab to force a sync.")
                
            activity_names = {
                1: "ជិះកង់ (Biking)",
                2: "ហាត់ប្រាណ Calisthenics",
                7: "ដើរ (Walking)",
                8: "រត់ (Running)",
                9: "អេរ៉ូប៊ិក (Aerobics)",
                10: "វាយសី (Badminton)",
                11: "បេស្បល (Baseball)",
                12: "បាល់បោះ (Basketball)",
                20: "ប្រដាល់ (Boxing)",
                24: "រាំ (Dancing)",
                31: "ធ្វើសួន (Gardening)",
                32: "វាយកូនហ្គោល (Golf)",
                35: "ដើរភ្នំ (Hiking)",
                53: "អុំទូក (Rowing)",
                58: "រត់លើម៉ាស៊ីន (Treadmill Running)",
                97: "លើកទម្ងន់ (Weight Lifting)",
                100: "ហែលទឹក (Swimming)",
                108: "ហាត់ប្រាណទូទៅ (Workout)",
                113: "ហាត់ប្រាណ Fitness",
                114: "យូហ្គា (Yoga)",
                115: "ម៉ាស៊ីន Elliptical",
                116: "Zumba"
            }
            
            valid_sessions = []
            for s in sessions:
                act_type = s.get("activityType")
                if act_type in activity_names and s.get("endTimeMillis") is not None:
                    valid_sessions.append(s)
                    
            if not valid_sessions:
                session_list = []
                for s in sessions[:5]:
                    name = s.get("name", "Unknown")
                    act_val = s.get("activityType", "Unknown")
                    has_end = "Yes" if s.get("endTimeMillis") is not None else "No"
                    session_list.append(f"• '{name}' (Type: {act_val}, Finished: {has_end})")
                raise Exception(
                    f"Found {len(sessions)} sessions, but none matched whitelisted activity types or were finished.\n"
                    + "\n".join(session_list)
                )
                
            valid_sessions.sort(key=lambda x: int(x.get("endTimeMillis", 0)), reverse=True)
            latest_session = valid_sessions[0]
            
            start_ms = int(latest_session.get("startTimeMillis", 0))
            end_ms = int(latest_session.get("endTimeMillis", 0))
            act_type = latest_session["activityType"]
            act_name = activity_names[act_type]
            session_name = latest_session.get("name", act_name)
            
            # If the session has 0 or extremely short duration, expand the query range to get calories
            query_start = start_ms
            query_end = end_ms
            if query_end - query_start < 60000: # less than 1 minute
                # Expand search window to 1 hour before the session end to catch any data points written
                query_start = query_end - 3600 * 1000
                
            cal_payload = {
                "aggregateBy": [
                    {"dataTypeName": "com.google.calories.expended"},
                    {"dataTypeName": "com.google.distance.delta"}
                ],
                "startTimeMillis": query_start,
                "endTimeMillis": query_end,
                "bucketByActivityType": {}
            }
            
            cal_resp = await client.post("https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate", headers=headers, json=cal_payload)
            calories_burned = 0
            distance_meters = 0
            if cal_resp.status_code == 200:
                cal_data = cal_resp.json()
                for bucket in cal_data.get("bucket", []):
                    datasets = bucket.get("dataset", [])
                    if len(datasets) > 0:
                        for point in datasets[0].get("point", []):
                            for value in point.get("value", []):
                                calories_burned += value.get("fpVal", value.get("intVal", 0))
                    if len(datasets) > 1:
                        for point in datasets[1].get("point", []):
                            for value in point.get("value", []):
                                distance_meters += value.get("fpVal", value.get("intVal", 0))
            
            duration_minutes = (end_ms - start_ms) / 60000.0
            display_duration = int(duration_minutes) if duration_minutes >= 1.0 else 30
            
            if calories_burned < 1:
                calories_burned = int(display_duration * 6.5)
                
            end_dt_utc = datetime.datetime.utcfromtimestamp(end_ms / 1000.0)
            end_dt_kh = end_dt_utc + datetime.timedelta(hours=7)
            date_str = end_dt_kh.strftime("%d-%m-%Y %I:%M %p")
            
            return {
                "activity_type": act_type,
                "activity_name": act_name,
                "session_name": session_name,
                "calories": int(calories_burned),
                "duration": display_duration,
                "distance": round(distance_meters / 1000.0, 1),
                "date_str": date_str,
                "end_ms": end_ms
            }
    except Exception as e:
        print(f"Error fetching latest Google Fit session: {e}")
        raise e

@app.get("/api/fit/auth")
async def fit_auth(user_id: int):
    """Generates and redirects to the Google Fit OAuth 2.0 Consent Screen."""
    client_id = os.getenv("GOOGLE_FIT_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_FIT_REDIRECT_URI")
    
    from fastapi.responses import JSONResponse
    if not client_id or not redirect_uri:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Google Fit credentials are not properly configured in .env."}
        )
        
    scopes = [
        "https://www.googleapis.com/auth/fitness.activity.read",
        "https://www.googleapis.com/auth/fitness.nutrition.write",
        "https://www.googleapis.com/auth/fitness.nutrition.read"
    ]
    scope_str = " ".join(scopes)
    
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope_str}"
        f"&state={user_id}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(auth_url)

@app.get("/api/fit/callback")
async def fit_callback(code: str, state: str):
    """Handles OAuth callback, exchanges authorization code for tokens, and displays a premium HTML confirmation page."""
    user_id = int(state)
    client_id = os.getenv("GOOGLE_FIT_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_FIT_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_FIT_REDIRECT_URI")
    
    from fastapi.responses import HTMLResponse, RedirectResponse
    
    if not code or not state:
        return HTMLResponse(content="<h2>❌ Parameter មិនត្រឹមត្រូវ!</h2>", status_code=400)
        
    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, data=payload)
            if resp.status_code == 200:
                data = resp.json()
                access_token = data["access_token"]
                refresh_token = data.get("refresh_token")
                expires_in = data.get("expires_in", 3600)
                
                existing = db_get_fit_tokens(user_id)
                final_refresh_token = refresh_token if refresh_token else (existing["refresh_token"] if existing else None)
                
                if not final_refresh_token:
                    auth_url = f"/api/fit/auth?user_id={user_id}"
                    return RedirectResponse(auth_url)
                
                db_save_fit_tokens(user_id, access_token, final_refresh_token, expires_in)
                
                success_html = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Google Fit Connected</title>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
                    <style>
                        :root {
                            --bg: #0b0f19;
                            --panel: rgba(255, 255, 255, 0.05);
                            --border: rgba(255, 255, 255, 0.08);
                            --glow: #3b82f6;
                            --success: #10b981;
                            --text: #f3f4f6;
                            --text-muted: #9ca3af;
                        }
                        * { box-sizing: border-box; margin: 0; padding: 0; }
                        body {
                            font-family: 'Inter', sans-serif;
                            background: var(--bg);
                            color: var(--text);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            min-height: 100vh;
                            overflow: hidden;
                            perspective: 1000px;
                        }
                        .container {
                            background: var(--panel);
                            border: 1px solid var(--border);
                            backdrop-filter: blur(20px);
                            border-radius: 24px;
                            padding: 40px;
                            width: 90%;
                            max-width: 440px;
                            text-align: center;
                            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 40px rgba(59, 130, 246, 0.1);
                            transform: translateY(0);
                            animation: floatIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                        }
                        .icon-wrap {
                            position: relative;
                            width: 90px;
                            height: 90px;
                            margin: 0 auto 30px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .icon-bg {
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            background: rgba(16, 185, 129, 0.15);
                            border-radius: 50%;
                            transform: scale(0.8);
                            animation: pulse 2s infinite ease-in-out;
                        }
                        .success-icon {
                            font-size: 45px;
                            z-index: 2;
                        }
                        h1 {
                            font-size: 24px;
                            font-weight: 700;
                            margin-bottom: 12px;
                            letter-spacing: -0.5px;
                            color: #ffffff;
                        }
                        p {
                            font-size: 15px;
                            color: var(--text-muted);
                            line-height: 1.6;
                            margin-bottom: 30px;
                        }
                        .btn {
                            display: inline-block;
                            width: 100%;
                            padding: 14px;
                            border: none;
                            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                            border-radius: 12px;
                            color: #ffffff;
                            font-weight: 600;
                            font-size: 15px;
                            text-decoration: none;
                            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
                            cursor: pointer;
                            transition: transform 0.2s, box-shadow 0.2s;
                        }
                        .btn:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
                        }
                        @keyframes floatIn {
                            from { transform: translateY(40px); opacity: 0; }
                            to { transform: translateY(0); opacity: 1; }
                        }
                        @keyframes pulse {
                            0% { transform: scale(0.9); opacity: 0.8; }
                            50% { transform: scale(1.15); opacity: 0.4; }
                            100% { transform: scale(0.9); opacity: 0.8; }
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="icon-wrap">
                            <div class="icon-bg"></div>
                            <div class="success-icon">🎉</div>
                        </div>
                        <h1>Google Fit ភ្ជាប់បានជោគជ័យ!</h1>
                        <p>គណនី NutriBot របស់អ្នកឥឡូវនេះត្រូវបានភ្ជាប់ទៅកាន់ Google Fit រួចរាល់ហើយ។ អ្នកអាចបិទទំព័រនេះ និងត្រឡប់ទៅកាន់ Telegram Bot វិញដើម្បីបន្តប្រើប្រាស់។</p>
                        <button class="btn" onclick="window.close()">រួចរាល់</button>
                    </div>
                </body>
                </html>
                """
                return HTMLResponse(content=success_html)
            else:
                return HTMLResponse(content=f"<h2>❌ ការដោះដូរ Token បានបរាជ័យ!</h2><p>{resp.text}</p>", status_code=500)
    except Exception as e:
        return HTMLResponse(content=f"<h2>❌ កំហុសបច្ចេកទេស!</h2><p>{str(e)}</p>", status_code=500)

# ---------------------------------------------------------
# Strava OAuth Endpoints
# ---------------------------------------------------------
@app.get("/api/strava/auth")
async def strava_auth(user_id: int):
    """Generates and redirects to the Strava OAuth 2.0 Consent Screen."""
    client_id = os.getenv("STRAVA_CLIENT_ID")
    redirect_uri = os.getenv("STRAVA_REDIRECT_URI")
    
    from fastapi.responses import JSONResponse
    if not client_id or not redirect_uri:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Strava credentials are not properly configured in .env."}
        )
        
    auth_url = (
        "https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=activity:read_all"
        f"&state={user_id}"
        f"&approval_prompt=force"
    )
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(auth_url)

@app.get("/api/strava/callback")
async def strava_callback(code: str, state: str, scope: str = None):
    """Handles OAuth callback, exchanges authorization code for tokens, and displays a premium HTML confirmation page."""
    user_id = int(state)
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    redirect_uri = os.getenv("STRAVA_REDIRECT_URI")
    
    from fastapi.responses import HTMLResponse
    
    if not code or not state:
        return HTMLResponse(content="<h2>❌ Parameter មិនត្រឹមត្រូវ!</h2>", status_code=400)
        
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, data=payload)
            if resp.status_code == 200:
                data = resp.json()
                access_token = data["access_token"]
                refresh_token = data.get("refresh_token")
                expires_at = data.get("expires_at")
                
                athlete_id = None
                if "athlete" in data and isinstance(data["athlete"], dict):
                    athlete_id = data["athlete"].get("id")
                
                db_save_strava_tokens(user_id, access_token, refresh_token, expires_at, athlete_id)
                
                # Send a message to the user confirming successful sync!
                try:
                    await bot.send_message(
                        user_id,
                        "🎉 <b>ភ្ជាប់គណនីជាមួយ Strava ជោគជ័យ! (Connected)</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "គណនី Strava របស់អ្នកត្រូវបានភ្ជាប់ទៅកាន់ NutriBot រួចរាល់ហើយ!\n\n"
                        "👉 វាយបញ្ជា៖ <b>/burn</b> ដើម្បីទាញយកសកម្មភាពហាត់ប្រាណចុងក្រោយរបស់អ្នកពី Strava មកកត់ត្រាក្នុង NutriBot ភ្លាមៗ!"
                    )
                except Exception as tg_err:
                    print(f"Failed to send Strava Telegram confirmation to user {user_id}: {tg_err}")
                
                success_html = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Strava Connected</title>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
                    <style>
                        :root {
                            --bg: #0b0f19;
                            --panel: rgba(255, 255, 255, 0.05);
                            --border: rgba(255, 255, 255, 0.08);
                            --glow: #fc4c02; /* Strava Orange */
                            --success: #10b981;
                            --text: #f3f4f6;
                            --text-muted: #9ca3af;
                        }
                        * { box-sizing: border-box; margin: 0; padding: 0; }
                        body {
                            font-family: 'Inter', sans-serif;
                            background: var(--bg);
                            color: var(--text);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            min-height: 100vh;
                            overflow: hidden;
                            perspective: 1000px;
                        }
                        .container {
                            background: var(--panel);
                            border: 1px solid var(--border);
                            backdrop-filter: blur(20px);
                            border-radius: 24px;
                            padding: 40px;
                            width: 90%;
                            max-width: 440px;
                            text-align: center;
                            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 40px rgba(252, 76, 2, 0.15);
                            transform: translateY(0);
                            animation: floatIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                        }
                        .icon-wrap {
                            position: relative;
                            width: 90px;
                            height: 90px;
                            margin: 0 auto 30px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .icon-bg {
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            background: rgba(252, 76, 2, 0.15);
                            border-radius: 50%;
                            transform: scale(0.8);
                            animation: pulse 2s infinite ease-in-out;
                        }
                        .success-icon {
                            font-size: 45px;
                            z-index: 2;
                        }
                        h1 {
                            font-size: 24px;
                            font-weight: 700;
                            margin-bottom: 12px;
                            letter-spacing: -0.5px;
                            color: #ffffff;
                        }
                        p {
                            font-size: 15px;
                            color: var(--text-muted);
                            line-height: 1.6;
                            margin-bottom: 30px;
                        }
                        .btn {
                            display: inline-block;
                            width: 100%;
                            padding: 14px;
                            border: none;
                            background: linear-gradient(135deg, #fc4c02 0%, #e34402 100%);
                            border-radius: 12px;
                            color: #ffffff;
                            font-weight: 600;
                            font-size: 15px;
                            text-decoration: none;
                            box-shadow: 0 4px 15px rgba(252, 76, 2, 0.3);
                            cursor: pointer;
                            transition: transform 0.2s, box-shadow 0.2s;
                        }
                        .btn:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 20px rgba(252, 76, 2, 0.4);
                        }
                        @keyframes floatIn {
                            from { transform: translateY(40px); opacity: 0; }
                            to { transform: translateY(0); opacity: 1; }
                        }
                        @keyframes pulse {
                            0% { transform: scale(0.9); opacity: 0.8; }
                            50% { transform: scale(1.15); opacity: 0.4; }
                            100% { transform: scale(0.9); opacity: 0.8; }
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="icon-wrap">
                            <div class="icon-bg"></div>
                            <div class="success-icon">🍊</div>
                        </div>
                        <h1>Strava ភ្ជាប់បានជោគជ័យ!</h1>
                        <p>គណនី NutriBot របស់អ្នកឥឡូវនេះត្រូវបានភ្ជាប់ទៅកាន់ Strava រួចរាល់ហើយ។ អ្នកអាចបិទទំព័រនេះ និងត្រឡប់ទៅកាន់ Telegram Bot វិញដើម្បីបន្តប្រើប្រាស់。</p>
                        <button class="btn" onclick="window.close()">រួចរាល់</button>
                    </div>
                </body>
                </html>
                """
                return HTMLResponse(content=success_html)
            else:
                return HTMLResponse(content=f"<h2>❌ ការដោះដូរ Token បានបរាជ័យ!</h2><p>{resp.text}</p>", status_code=500)
    except Exception as e:
        return HTMLResponse(content=f"<h2>❌ កំហុសបច្ចេកទេស!</h2><p>{str(e)}</p>", status_code=500)

# ---------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------
@app.post("/api/webhook")
async def telegram_webhook(request: Request):
    """Processes incoming Telegram bot webhook updates."""
    try:
        payload = await request.json()
        # Handle the update safely
        await handle_telegram_update(payload)
    except Exception as e:
        print(f"Unhandled error in webhook route: {e}")
    # Always return a 200 OK to Telegram immediately to prevent webhook retry loops
    return {"status": "ok"}

@app.get("/api/setup")
async def setup_webhook(request: Request):
    """Utility route to bind this deployment's endpoint to Telegram Webhook."""
    # Ensure database schema is bootstrapped/updated
    db_initialize_schema()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN environment variable is not configured."}
    
    # Dynamically extract host domain to build the webhook callback URL
    host = request.headers.get("host")
    if not host:
        return {"ok": False, "error": "Could not determine host from request headers."}
    
    protocol = "https"
    if "localhost" in host or "127.0.0.1" in host:
        protocol = "http"
        
    webhook_url = f"{protocol}://{host}/api/webhook"
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.telegram.org/bot{token}/setWebhook",
            params={"url": webhook_url}
        )
        data = resp.json()
        
    return {
        "ok": data.get("ok", False),
        "webhook_url": webhook_url,
        "telegram_response": data
    }

@app.get("/api/cron_reminders")
async def vercel_cron_reminders(request: Request):
    """Vercel cron endpoint that triggers reminders to active users based on Cambodian Time slot."""
    cron_header = request.headers.get("x-vercel-cron")
    is_prod = os.getenv("VERCEL_ENV") == "production"
    if is_prod and cron_header != "1":
         return {"ok": False, "error": "Unauthorized. This endpoint is secured for Vercel Cron only."}
         
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
         return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured."}
         
    bot = TelegramBot(bot_token)
    
    # Calculate current slot in Cambodian Time (UTC + 7)
    now_utc = datetime.datetime.utcnow()
    now_cambodia = now_utc + datetime.timedelta(hours=7)
    slot_hour = now_cambodia.hour
    slot_minute_start = (now_cambodia.minute // 10) * 10
    
    # Format pattern, e.g. "08:3%"
    slot_pattern = f"{slot_hour:02d}:{slot_minute_start // 10}%"
    
    # Query all users with active reminders in this slot
    user_ids = db_get_active_reminders_for_slot(slot_pattern)
    
    reminded_count = 0
    failed_count = 0
    
    reminder_message = (
        "🔔 <b>រំលឹកកត់ត្រាអាហារ!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "កុំភ្លេចកត់ត្រារបបអាហាររបស់អ្នកថ្ងៃនេះណា! សូមផ្ញើរូបភាពអាហារ ឬសរសេររៀបរាប់ពីអ្វីដែលអ្នកបានញ៉ាំមកខ្ញុំឥឡូវនេះ។"
    )
    
    for uid in user_ids:
        try:
            await bot.send_message(uid, reminder_message)
            reminded_count += 1
        except Exception as send_err:
            print(f"Error sending cron reminder to user {uid}: {send_err}")
            failed_count += 1
            
    return {
        "status": "success",
        "slot_pattern": slot_pattern,
        "cambodia_time_checked": now_cambodia.strftime("%Y-%m-%d %H:%M:%S"),
        "reminded_count": reminded_count,
        "failed_count": failed_count
    }

@app.get("/api/health")
async def health_check():
    """Diagnostic API check verifying credentials and connections."""
    results = {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat(), "checks": {}}
    
    # 1. Check Turso connection
    try:
        url = os.getenv("TURSO_DATABASE_URL")
        auth_token = os.getenv("TURSO_AUTH_TOKEN")
        if not url or not auth_token:
            results["checks"]["turso"] = "missing_credentials"
            results["status"] = "unhealthy"
        else:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            results["checks"]["turso"] = "connected"
    except Exception as e:
        results["checks"]["turso"] = f"failed: {e}"
        results["status"] = "unhealthy"
        
    # 2. Check Gemini Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    results["checks"]["gemini"] = "present" if gemini_key else "missing"
    if not gemini_key:
        results["status"] = "unhealthy"
        
    # 3. Check Telegram Token
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    results["checks"]["telegram"] = f"present (suffix: ...{telegram_token[-5:]})" if telegram_token else "missing"
    if not telegram_token:
        results["status"] = "unhealthy"
        
    return results

async def fetch_specific_strava_activity(user_id: int, activity_id: int) -> dict:
    """Fetches a specific completed activity details from Strava by its activity ID."""
    token_info = db_get_strava_tokens(user_id)
    if not token_info:
        return None
        
    access_token = await get_valid_strava_token(user_id, token_info)
    if not access_token:
        return None
        
    import datetime
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"Strava activity details API returned status {resp.status_code}: {resp.text}")
                
            act = resp.json()
            
            activity_id = act.get("id")
            session_name = act.get("name", "Workout")
            act_type = act.get("type", "Workout")
            
            type_mappings = {
                "Run": "រត់ (Running)",
                "Ride": "ជិះកង់ (Biking)",
                "Walk": "ដើរ (Walking)",
                "Hike": "ដើរភ្នំ (Hiking)",
                "Swim": "ហែលទឹក (Swimming)",
                "WeightTraining": "លើកទម្ងន់ (Weight Lifting)",
                "Workout": "ហាត់ប្រាណទូទៅ (Workout)",
                "Yoga": "យូហ្គា (Yoga)",
                "Elliptical": "ម៉ាស៊ីន Elliptical"
            }
            act_name = type_mappings.get(act_type, f"ហាត់ប្រាណ {act_type}")
            
            # Extract duration in minutes
            duration_minutes = int(act.get("moving_time", 0) / 60.0)
            if duration_minutes < 1:
                duration_minutes = int(act.get("elapsed_time", 0) / 60.0)
            if duration_minutes < 1:
                duration_minutes = 30
                
            # Extract distance in kilometers
            distance_meters = act.get("distance", 0)
            distance_km = round(distance_meters / 1000.0, 1)
            
            # Extract calories
            calories_burned = act.get("calories", 0)
            if calories_burned < 1:
                calories_burned = act.get("kilojoules", 0)
            if calories_burned < 1:
                calories_burned = int(duration_minutes * 6.5)
                
            start_date_local = act.get("start_date_local")
            activity_date = None
            if start_date_local:
                try:
                    clean_date = start_date_local.replace("Z", "")
                    dt = datetime.datetime.fromisoformat(clean_date)
                    date_str = dt.strftime("%d-%m-%Y %I:%M %p")
                    activity_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
                    activity_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")
            else:
                date_str = datetime.datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
                activity_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")
                
            return {
                "activity_type": act_type,
                "activity_name": act_name,
                "session_name": session_name,
                "calories": int(calories_burned),
                "duration": duration_minutes,
                "distance": distance_km,
                "date_str": date_str,
                "activity_date": activity_date,
                "activity_id": activity_id
            }
    except Exception as e:
        print(f"Error fetching specific Strava activity: {e}")
        raise e

@app.get("/api/strava/setup_webhook")
async def setup_strava_webhook(request: Request):
    """One-click setup endpoint to register Strava Push Subscription Webhook."""
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    
    from fastapi.responses import JSONResponse
    if not client_id or not client_secret:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Strava credentials are not properly configured in .env."}
        )
        
    host = request.headers.get("host")
    protocol = "https"
    if "localhost" in host or "127.0.0.1" in host:
        protocol = "http"
        
    callback_url = f"{protocol}://{host}/api/strava/webhook"
    verify_token = "NutriBotStravaVerifyToken123!"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Fetch any existing subscriptions
            get_url = f"https://www.strava.com/api/v3/push_subscriptions?client_id={client_id}&client_secret={client_secret}"
            get_resp = await client.get(get_url)
            
            deleted_count = 0
            if get_resp.status_code == 200:
                subs = get_resp.json()
                if isinstance(subs, list):
                    for sub in subs:
                        sub_id = sub.get("id")
                        if sub_id:
                            del_url = f"https://www.strava.com/api/v3/push_subscriptions/{sub_id}?client_id={client_id}&client_secret={client_secret}"
                            del_resp = await client.delete(del_url)
                            if del_resp.status_code in [200, 204]:
                                deleted_count += 1
                                
            # 2. Register the fresh webhook
            post_url = "https://www.strava.com/api/v3/push_subscriptions"
            payload = {
                "client_id": client_id,
                "client_secret": client_secret,
                "callback_url": callback_url,
                "verify_token": verify_token
            }
            
            resp = await client.post(post_url, data=payload)
            if resp.status_code in [200, 201]:
                return JSONResponse(content={
                    "ok": True, 
                    "message": "Strava webhook registered successfully!", 
                    "deleted_previous_subscriptions": deleted_count,
                    "data": resp.json()
                })
            else:
                return JSONResponse(
                    status_code=resp.status_code,
                    content={
                        "ok": False, 
                        "message": "Failed to register Strava webhook.", 
                        "deleted_previous_subscriptions": deleted_count,
                        "details": resp.text
                    }
                )
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.get("/api/strava/webhook")
async def strava_webhook_challenge(request: Request):
    """Handles Strava's verification handshake (GET request)."""
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    from fastapi.responses import JSONResponse
    if mode == "subscribe" and token == "NutriBotStravaVerifyToken123!":
        return JSONResponse(content={"hub.challenge": challenge})
        
    return JSONResponse(status_code=400, content={"error": "Invalid verification token or parameters"})

@app.post("/api/strava/webhook")
async def strava_webhook_event(request: Request):
    """Handles background workout event triggers (POST request) from Strava."""
    payload = await request.json()
    print(f"Received Strava webhook payload: {payload}")
    
    object_type = payload.get("object_type")
    aspect_type = payload.get("aspect_type")
    
    if object_type == "activity" and aspect_type == "create":
        activity_id = payload.get("object_id")
        owner_id = payload.get("owner_id")
        
        user_id = db_get_user_id_by_strava_athlete(owner_id)
        if user_id:
            try:
                session = await fetch_specific_strava_activity(user_id, activity_id)
                if session:
                    act_key = f"{session['activity_name']} ({session['date_str']})"
                    
                    is_duplicate = False
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT 1 FROM burn_logs WHERE user_id = ? AND activity_name = ? AND source = 'Strava'",
                            (user_id, act_key)
                        )
                        if cursor.fetchone():
                            is_duplicate = True
                            
                    if not is_duplicate:
                        db_add_burn(user_id, session['calories'], act_key, "Strava", custom_date=session.get('activity_date'))
                        
                        activity_date = session.get('activity_date')
                        import datetime
                        now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
                        today_str = now_kh.strftime("%Y-%m-%d")
                        
                        if activity_date and activity_date != today_str:
                            date_parts = activity_date.split('-')
                            formatted_display_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                            display_day = f"ថ្ងៃ {formatted_display_date}"
                            yesterday_str = (now_kh - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                            if activity_date == yesterday_str:
                                display_day = "ម្សិលមិញ"
                                
                            success_card = (
                                "⚡ <b>លំហាត់ប្រាណត្រូវបាន Sync ស្វ័យប្រវត្តពី Strava!</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                f"🚴 <b>សកម្មភាព៖</b> <b>{session['activity_name']}</b> ({session['session_name']})\n"
                                f"🔥 <b>ដុតកាឡូរី៖</b> <b>{session['calories']} Cal</b>\n"
                                f"⏲ <b>ពេលវេលា៖</b> <b>{session['duration']} នាទី</b>\n"
                                f"🗾 <b>ចម្ងាយ៖</b> <b>{session['distance']} គីឡូម៉ែត្រ</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                f"លំហាត់ប្រាណថ្មីរបស់អ្នក ត្រូវបានបញ្ចូលទៅក្នុងកំណត់ត្រា <b>{display_day}</b> ដោយស្វ័យប្រវត្ត! 💪"
                            )
                        else:
                            success_card = (
                                "⚡ <b>លំហាត់ប្រាណត្រូវបាន Sync ស្វ័យប្រវត្តពី Strava!</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                f"🚴 <b>សកម្មភាព៖</b> <b>{session['activity_name']}</b> ({session['session_name']})\n"
                                f"🔥 <b>ដុតកាឡូរី៖</b> <b>{session['calories']} Cal</b>\n"
                                f"⏲ <b>ពេលវេលា៖</b> <b>{session['duration']} នាទី</b>\n"
                                f"🗾 <b>ចម្ងាយ៖</b> <b>{session['distance']} គីឡូម៉ែត្រ</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "លំហាត់ប្រាណថ្មីរបស់អ្នក ត្រូវបានបញ្ចូលទៅក្នុងកំណត់ត្រាថ្ងៃនេះដោយស្វ័យប្រវត្ត! 💪"
                            )
                        await bot.send_message(user_id, success_card)
                        print(f"Successfully auto-synced webhook activity {activity_id} for user {user_id}")
            except Exception as e:
                print(f"Error processing Strava webhook activity {activity_id} for user {user_id}: {e}")
                
    return {"status": "ok"}

# ---------------------------------------------------------
# Telegram Bot Mini App (TMA) Endpoints
# ---------------------------------------------------------

@app.get("/api/tma/dashboard")
async def tma_get_dashboard(user_id: int):
    # Ensure the user is registered (1 transaction, done inline)
    import datetime
    now_kh = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    date_str = now_kh.strftime("%Y-%m-%d")
    
    profile = None
    goal = 2000
    goal_type = "maintain"
    today_meals = []
    total_cals = 0
    total_burn = 0
    no_sweet_today = False
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure user exists in users table with a default goal of 2000 Cal
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, daily_calorie_goal) VALUES (?, 2000)",
                (user_id,)
            )
            conn.commit()
            
            # 1. Query user profile
            cursor.execute(
                "SELECT gender, age, height, weight, activity, goal_type, daily_calorie_budget FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row and row[0] is not None:
                profile = {
                    "gender": row[0],
                    "age": row[1],
                    "height": row[2],
                    "weight": row[3],
                    "activity": row[4],
                    "goal_type": row[5],
                    "daily_calorie_budget": row[6]
                }
                goal_type = row[5] or "maintain"
                goal = row[6] or 2000
            else:
                # If profile not set, try to get just the goal
                cursor.execute("SELECT daily_calorie_goal FROM users WHERE user_id = ?", (user_id,))
                goal_row = cursor.fetchone()
                if goal_row:
                    goal = goal_row[0] or 2000
            
            # 2. Query today's meals
            cursor.execute(
                """
                SELECT meal_id, food_name, calories, protein, fat, carbs, sugar, timestamp
                FROM meals
                WHERE user_id = ? AND date(timestamp, '+7 hours') = ?
                ORDER BY timestamp DESC
                """,
                (user_id, date_str)
            )
            rows = cursor.fetchall()
            for r in rows:
                today_meals.append({
                    "meal_id": r[0],
                    "food_name": r[1],
                    "calories": r[2],
                    "protein": r[3],
                    "fat": r[4],
                    "carbs": r[5],
                    "sugar": r[6],
                    "timestamp": r[7]
                })
                total_cals += r[2]
                
            # 3. Query today's burn
            cursor.execute(
                "SELECT SUM(calories_burned) FROM burn_logs WHERE user_id = ? AND date(timestamp, '+7 hours') = ?",
                (user_id, date_str)
            )
            burn_row = cursor.fetchone()
            if burn_row and burn_row[0] is not None:
                total_burn = int(burn_row[0])
                
            # 4. Check today's nosweet challenge
            cursor.execute(
                """
                SELECT 1 FROM nosweet_logs 
                WHERE user_id = ? AND date(timestamp, '+7 hours') = ?
                LIMIT 1
                """,
                (user_id, date_str)
            )
            no_sweet_today = cursor.fetchone() is not None
            
    except Exception as e:
        print(f"Error loading dashboard data for user {user_id}: {e}")
        
    return {
        "user_id": user_id,
        "goal": goal,
        "goal_type": goal_type,
        "profile": profile,
        "today_meals": today_meals,
        "total_cals": total_cals,
        "total_burn": total_burn,
        "no_sweet_today": no_sweet_today
    }

class TMAMealRequest(BaseModel):
    user_id: int
    food_description: str
    custom_date: Optional[str] = None

@app.post("/api/tma/meal")
async def tma_add_meal(req: TMAMealRequest):
    # Process the food description using Gemini and save it to database
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"ok": False, "error": "Gemini API key is not configured."}
        
    client = genai.Client()
    user_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    models_to_try = [user_model]
    for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)
    
    profile = db_get_user_profile(req.user_id)
    if profile:
        profile_context = (
            f"The user is a {profile['gender']}, {profile['age']} years old, {profile['height']:.1f} cm tall, "
            f"weighing {profile['weight']:.1f} kg. Their physical activity level is mapped as '{profile['activity']}'. "
            f"Their daily budget goal is {profile['daily_calorie_budget']} Cal and their goal type is '{profile['goal_type']}'."
        )
    else:
        profile_context = "The user is a general individual with a daily budget of 2000 Cal aiming to maintain weight."
        
    logging_time_context = "The user is logging a meal via their Telegram Mini App."
    if req.custom_date:
        logging_time_context = f"The user is retroactively logging for the Cambodia local date {req.custom_date}."

    TEXT_SYSTEM_PROMPT = (
        "You are a professional nutrition expert and health coach. Analyze the food description text provided and estimate its "
        "nutritional details (calories in Cal, protein/fat/carbs/sugar in grams).\n"
        f"User Health Context: {profile_context}\n"
        f"Logging Context: {logging_time_context}\n"
        "YOU MUST RESPOND ENTIRELY IN KHMER LANGUAGE. The `food_name` field must be written in beautiful Khmer script.\n"
        "Provide a highly personalized coaching and health recommendation (in the `coaching_recommendation` field) "
        "in Khmer tailored specifically to this user's profile, goal, and the logging context.\n"
        "CRITICAL SECRECY RULE: You know the user's age, weight, height, and calorie target budget from the User Health Context, BUT YOU MUST KEEP THEM SECRET. Never mention or repeat their age, weight, height, or daily calorie goal in your coaching_recommendation text response. Focus purely on qualitative health insights, digestion, macronutrients, and advice.\n"
        "Do NOT recite or repeat raw numbers (like '150 Cal' or '10g protein') inside the coaching recommendation text.\n"
        "If the text does not describe any food, or you cannot identify any food, "
        "you MUST set the `confidence_score` to less than 0.5 (e.g. 0.0 to 0.4), "
        "and you can set the `food_name` to 'មិនមែនជាអាហារ ឬរកមិនឃើញ'."
    )
    
    response = None
    errors = []
    for current_model in models_to_try:
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=f"Analyze the following food description and return its nutrition facts in Khmer: {req.food_description}",
                config=types.GenerateContentConfig(
                    system_instruction=TEXT_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=FoodAnalysis,
                ),
            )
            break
        except Exception as model_err:
            errors.append(f"{current_model}: {str(model_err)}")

    if not response:
        return {"ok": False, "error": f"Gemini Error list: {'; '.join(errors)}"}
        
    try:
        analysis = FoodAnalysis.model_validate_json(response.text)
        if analysis.confidence_score < 0.5:
            return {"ok": False, "error": "រកមិនឃើញអាហារ ឬបរិមាណមិនច្បាស់លាស់។"}
            
        inserted_meal_id = db_add_meal(req.user_id, analysis, req.custom_date)
        return {
            "ok": True,
            "meal": {
                "meal_id": inserted_meal_id,
                "food_name": analysis.food_name,
                "calories": analysis.calories,
                "protein": analysis.protein,
                "fat": analysis.fat,
                "carbs": analysis.carbs,
                "sugar": analysis.sugar,
                "coaching_recommendation": analysis.coaching_recommendation
            }
        }
    except Exception as e:
        print(f"Error in TMA add meal validation/saving: {e}")
        return {"ok": False, "error": str(e)}

class TMABurnRequest(BaseModel):
    user_id: int
    calories: int
    activity_name: str = 'Manual'
    custom_date: Optional[str] = None

@app.post("/api/tma/burn")
async def tma_add_burn(req: TMABurnRequest):
    try:
        if req.calories <= 0 or req.calories > 10000:
            return {"ok": False, "error": "កាឡូរីមិនត្រឹមត្រូវ។"}
            
        db_add_burn(req.user_id, req.calories, req.activity_name, 'Manual', req.custom_date)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class TMAWeightRequest(BaseModel):
    user_id: int
    weight: float

@app.post("/api/tma/weight")
async def tma_update_weight(req: TMAWeightRequest):
    try:
        profile = db_get_user_profile(req.user_id)
        if not profile:
            db_register_user(req.user_id)
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO users (user_id, gender, age, height, weight, activity, goal_type, daily_calorie_budget)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (req.user_id, "male", 25, 170, req.weight, "moderate", "weight_loss", 2000)
                )
                conn.commit()
        else:
            gender = profile["gender"]
            height = profile["height"]
            age = profile["age"]
            activity = profile["activity"]
            goal_type = profile["goal_type"]
            
            if gender == "male":
                bmr = 10 * req.weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * req.weight + 6.25 * height - 5 * age - 161
                
            multiplier = 1.2
            if activity == 'sedentary': multiplier = 1.2
            elif activity == 'light': multiplier = 1.375
            elif activity == 'moderate': multiplier = 1.465
            elif activity == 'active': multiplier = 1.55
            elif activity == 'very_active': multiplier = 1.725
            
            offset = 0
            if goal_type == 'mild_loss': offset = -250
            elif goal_type == 'weight_loss': offset = -500
            elif goal_type == 'extreme_loss': offset = -1000
            
            new_goal = max(1200, int(bmr * multiplier) + offset)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET weight = ?, daily_calorie_budget = ? WHERE user_id = ?",
                    (req.weight, new_goal, req.user_id)
                )
                conn.commit()
                
        return {"ok": True, "new_goal": db_get_user_goal(req.user_id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class TMAProfileRequest(BaseModel):
    user_id: int
    gender: str
    age: int
    height: float
    weight: float
    activity: str
    goal_type: str

@app.post("/api/tma/profile")
async def tma_update_profile(req: TMAProfileRequest):
    try:
        if req.gender == "male":
            bmr = 10 * req.weight + 6.25 * req.height - 5 * req.age + 5
        else:
            bmr = 10 * req.weight + 6.25 * req.height - 5 * req.age - 161
            
        multiplier = 1.2
        if req.activity == 'sedentary': multiplier = 1.2
        elif req.activity == 'light': multiplier = 1.375
        elif req.activity == 'moderate': multiplier = 1.465
        elif req.activity == 'active': multiplier = 1.55
        elif req.activity == 'very_active': multiplier = 1.725
        
        offset = 0
        if req.goal_type == 'mild_loss': offset = -250
        elif req.goal_type == 'weight_loss': offset = -500
        elif req.goal_type == 'extreme_loss': offset = -1000
        
        new_goal = max(1200, int(bmr * multiplier) + offset)
        
        db_register_user(req.user_id)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO users (user_id, gender, age, height, weight, activity, goal_type, daily_calorie_budget)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (req.user_id, req.gender, req.age, req.height, req.weight, req.activity, req.goal_type, new_goal)
            )
            conn.commit()
            
        return {"ok": True, "new_goal": new_goal}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class TMANosweetRequest(BaseModel):
    user_id: int
    no_sweet: bool

@app.post("/api/tma/nosweet")
async def tma_toggle_nosweet(req: TMANosweetRequest):
    try:
        if req.no_sweet:
            db_add_nosweet_log(req.user_id)
        else:
            db_remove_today_nosweet(req.user_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.delete("/api/tma/delete_meal")
async def tma_delete_meal(user_id: int, meal_id: int):
    try:
        db_delete_meal(user_id, meal_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/tma/weekly")
async def tma_get_weekly(user_id: int):
    import datetime
    now_utc = datetime.datetime.utcnow()
    now_cambodia = now_utc + datetime.timedelta(hours=7)
    
    current_weekday = now_cambodia.weekday()
    monday_date = now_cambodia.date() - datetime.timedelta(days=current_weekday)
    week_dates = [monday_date + datetime.timedelta(days=i) for i in range(7)]
    start_date_str = week_dates[0].strftime("%Y-%m-%d")
    end_date_str = week_dates[-1].strftime("%Y-%m-%d")
    
    goal = db_get_user_goal(user_id)
    
    days_data = []
    day_names_kh = {
        0: "ច័ន្ទ",
        1: "អង្គារ",
        2: "ពុធ",
        3: "ព្រហស្បតិ៍",
        4: "សុក្រ",
        5: "សៅរ៍",
        6: "អាទិត្យ"
    }
    day_names_en = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun"
    }
    
    for idx, d in enumerate(week_dates):
        days_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "day_name_en": day_names_en.get(idx),
            "day_name_kh": day_names_kh.get(idx),
            "eaten": 0,
            "burned": 0,
            "no_sweet": False,
            "meals": []
        })
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Fetch meals for the week
            cursor.execute(
                """
                SELECT meal_id, food_name, calories, protein, fat, carbs, sugar, timestamp, date(timestamp, '+7 hours')
                FROM meals
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                ORDER BY timestamp DESC
                """,
                (user_id, start_date_str, end_date_str)
            )
            meal_rows = cursor.fetchall()
            
            # 2. Fetch burns for the week
            cursor.execute(
                """
                SELECT calories_burned, date(timestamp, '+7 hours')
                FROM burn_logs
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                """,
                (user_id, start_date_str, end_date_str)
            )
            burn_rows = cursor.fetchall()
            
            # 3. Fetch nosweet challenge logs for the week
            cursor.execute(
                """
                SELECT DISTINCT date(timestamp, '+7 hours')
                FROM nosweet_logs
                WHERE user_id = ? AND date(timestamp, '+7 hours') >= ? AND date(timestamp, '+7 hours') <= ?
                """,
                (user_id, start_date_str, end_date_str)
            )
            nosweet_rows = cursor.fetchall()
            nosweet_dates = {row[0] for row in nosweet_rows}
            
            # Process meals
            for r in meal_rows:
                meal_id, food_name, calories, protein, fat, carbs, sugar, timestamp, m_date = r
                # Find matching day
                for day in days_data:
                    if day["date"] == m_date:
                        day["eaten"] += calories
                        
                        # Parse time
                        time_str = "12:00"
                        if timestamp:
                            try:
                                parts = timestamp.split(" ")
                                if len(parts) > 1:
                                    time_str = parts[1][:5]
                            except Exception:
                                pass
                        
                        day["meals"].append({
                            "meal_id": meal_id,
                            "food_name": food_name,
                            "calories": calories,
                            "protein": protein,
                            "fat": fat,
                            "carbs": carbs,
                            "sugar": sugar,
                            "time": time_str
                        })
            
            # Process burns
            for cals, b_date in burn_rows:
                for day in days_data:
                    if day["date"] == b_date:
                        day["burned"] += cals
            
            # Process nosweet
            for day in days_data:
                day["no_sweet"] = day["date"] in nosweet_dates
                
    except Exception as e:
        print(f"Error retrieving weekly summary for user {user_id}: {e}")
        
    return {
        "user_id": user_id,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "daily_goal": goal,
        "days": days_data
    }

class TMACustomMealRequest(BaseModel):
    user_id: int
    food_name: str
    calories: int
    protein: int
    fat: int
    carbs: int
    sugar: int
    custom_date: Optional[str] = None

@app.post("/api/tma/custom_meal")
async def tma_add_custom_meal(req: TMACustomMealRequest):
    try:
        analysis = FoodAnalysis(
            food_name=req.food_name,
            calories=req.calories,
            protein=req.protein,
            fat=req.fat,
            carbs=req.carbs,
            sugar=req.sugar,
            confidence_score=1.0,
            coaching_recommendation="បន្ថែមដោយផ្ទាល់ពីការស្វែងរក"
        )
        inserted_id = db_add_meal(req.user_id, analysis, req.custom_date)
        return {"ok": True, "meal_id": inserted_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/tma/search_food")
async def tma_search_food(user_id: int, query: str):
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"ok": False, "error": "Gemini API key is not configured."}
        
    client = genai.Client()
    user_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    models_to_try = [user_model]
    for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)
    
    TEXT_SYSTEM_PROMPT = (
        "You are a professional nutrition expert and calorie dictionary. Estimate the average nutritional facts "
        "for a standard normal single serving portion of the food queried by the user.\n"
        "The output must include estimated calories in Cal, and protein, fat, carbs, sugar in grams.\n"
        "YOU MUST RESPOND ENTIRELY IN KHMER. The `food_name` field must be written in beautiful Khmer script.\n"
        "Provide a useful brief coaching advice recommendation (in the `coaching_recommendation` field) "
        "in Khmer explaining the health benefits, macro distribution, or typical portion sizing of this item.\n"
        "If the query is not a food item or you cannot find it, set `confidence_score` below 0.5."
    )
    
    response = None
    errors = []
    for current_model in models_to_try:
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=f"Search nutrition details for food: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=TEXT_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=FoodAnalysis,
                ),
            )
            break
        except Exception as e:
            errors.append(f"{current_model}: {str(e)}")

    if not response:
        return {"ok": False, "error": f"Gemini Error list: {'; '.join(errors)}"}

    try:
        analysis = FoodAnalysis.model_validate_json(response.text)
        if analysis.confidence_score < 0.5:
            return {"ok": False, "error": "រកមិនឃើញអាហារ ឬព័ត៌មានមិនច្បាស់លាស់។"}
            
        return {
            "ok": True,
            "food": {
                "food_name": analysis.food_name,
                "calories": analysis.calories,
                "protein": analysis.protein,
                "fat": analysis.fat,
                "carbs": analysis.carbs,
                "sugar": analysis.sugar,
                "coaching_recommendation": analysis.coaching_recommendation
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/")
async def root_index():
    """Simple aesthetic landing page confirming serverless function status."""
    return {
        "message": "NutriBot Telegram Webhook FastAPI Khmer Backend is online!",
        "endpoints": {
            "webhook": "/api/webhook (POST only)",
            "setup": "/api/setup (GET to bind webhook)",
            "health": "/api/health (GET to check credentials)",
            "cron_reminders": "/api/cron_reminders (GET triggered by Vercel Cron)",
            "strava_setup_webhook": "/api/strava/setup_webhook (GET to register Strava webhook)"
        }
    }
