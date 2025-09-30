import pandas as pd
import sqlite3
from pathlib import Path

def populate_database(csv_file: Path, sqlite_url: Path):
    # Check if CSV exists
    if not csv_file.exists():
        print(f"Error: The file '{csv_file}' does not exist.")
        return
    
    # Check if DB exists
    if not sqlite_url.exists():
        print(f"Error: The database file '{sqlite_url}' does not exist.")
        print("Run `uv run prepare_db` -> to create the database and tables first.")
        return

    print(f"Populating '{sqlite_url.name}' from '{csv_file.name}'...")

    # Load data
    df = pd.read_csv(csv_file)
    print(f"Data loaded with {len(df)} records.")
    
    print("Transforming data for each table...")
    # Products table
    products_df = df[["StockCode", "Description"]].drop_duplicates(subset=["StockCode"])
    print(f"Prepared {len(products_df)} unique records for the Products table.")

    # Invoices table
    invoices_df = df[["InvoiceNo", "InvoiceDate", "CustomerID", "Country"]].drop_duplicates(subset=["InvoiceNo"])
    print(f"Prepared {len(invoices_df)} unique records for the Invoices table.")

    # InvoiceItems table
    invoice_items_df = df[["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]]
    print(f"Prepared {len(invoice_items_df)} records for the InvoiceItems table.")

    print("Inserting data into the database...")
    try:
        with sqlite3.connect(sqlite_url) as conn:
            print(f"Connecting to '{sqlite_url}'...")
            
            # Insert data into tables
            products_df.to_sql("Products", conn, if_exists="replace", index=False)
            print("Data successfully inserted into 'Products' table.")

            invoices_df.to_sql("Invoices", conn, if_exists="replace", index=False)
            print("Data successfully inserted into 'Invoices' table.")

            invoice_items_df.to_sql("InvoiceItems", conn, if_exists="replace", index=False)
            print("Data successfully inserted into 'InvoiceItems' table.")

        print("Database population complete.")

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
        print("Insertion failed. This might be because the script was already run and data exists (Primary Key constraint failed).")
        print("Re-run the script after `uv run prepare_db` to reset the database.")


def main():
    from databases.utils import sql_config, data_config
    sample_csv: Path = data_config.get("sample_path")
    sqlite_url: Path = sql_config.get("url")
    populate_database(sample_csv, sqlite_url)
