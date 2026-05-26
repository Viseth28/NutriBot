import os
import sys
import libsql

def init_db():
    print("Checking database environment variables...")
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if not url or not auth_token:
        print("❌ Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in the environment.")
        sys.exit(1)
        
    print(f"Connecting to database at {url}...")
    try:
        conn = libsql.connect(database=url, auth_token=auth_token)
        cursor = conn.cursor()
        
        print("Creating table 'users' if not exists...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
            daily_calorie_goal INTEGER DEFAULT 2000
        );
        """)
        
        print("Creating table 'meals' if not exists...")
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
        
        conn.commit()
        conn.close()
        print("✨ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_db()
