import sqlite3
import os

def create_database(db_file: str, sql_file: str) -> bool:
    # Check if the SQL file exists
    if not os.path.exists(sql_file):
        print(f"Error: The SQL file '{sql_file}' does not exist.")
        return False
    
    # Check if the database file already exists. If so, delete it.
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database file: {db_file}")
        
    # Check if the directory for the database file exists, if not create it
    db_dir = os.path.dirname(db_file)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created directory for database: {db_dir}")

    try:
        # Establish a connection. This will create the database file.
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print(f"Successfully connected to database '{db_file}'.")

        # Open and read the .sql file
        with open(sql_file, 'r') as f:
            sql_script = f.read()
            
        # Execute the entire script
        cursor.executescript(sql_script)
        print(f"Successfully executed script from '{sql_file}'.")

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        return True
    
    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
        return False

def main():
    from databases.helpers import sql_config
    db_file = sql_config['db_path']
    sql_file = sql_config['schema_path']
    if create_database(db_file, sql_file):
        print("Database setup completed successfully.")
    else:
        print("Database setup failed.")