import sqlite3
from pathlib import Path

def create_database(sqlite_url: Path, sql_file: Path) -> bool:
    # Check if the SQL file exists
    if not sql_file.exists():
        print(f"Error: The SQL file '{sql_file.name}' does not exist.")
        return False
    
    # Check if the database file already exists. If so, delete it.
    if sqlite_url.exists():
        sqlite_url.unlink()
        print(f"Removed existing database: {sqlite_url.name}")
        
    # Ensure parent directory exists
    if not sqlite_url.parent.exists():
        sqlite_url.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory for database: {sqlite_url.parent}")

    try:
        # Establish a connection (creates the DB file if not exists)
        conn = sqlite3.connect(sqlite_url)
        cursor = conn.cursor()
        print(f"Successfully connected to database '{sqlite_url.name}'.")

        # Open and read the .sql file
        with sql_file.open("r") as f:
            sql_script = f.read()
            
        # Execute the entire script
        cursor.executescript(sql_script)
        print(f"Successfully executed script from '{sql_file.name}'.")

        # Commit and close
        conn.commit()
        conn.close()
        return True
    
    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
        return False


def main():
    from databases.utils import sql_config
    sqlite_url: Path = sql_config.get("url")
    sql_file: Path = sql_config.get("schema_path")

    if create_database(sqlite_url, sql_file):
        print("Database setup completed successfully.")
    else:
        print("Database setup failed.")
