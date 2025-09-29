import pandas as pd
import sqlite3
import os

def populate_database(csv_file: str, db_file: str) -> None:
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        return
    
    if not os.path.exists(db_file):
        print(f"Error: The database file '{db_file}' does not exist.")
        print("Run uv run prepare_db -> to create the database and tables first.")
        return
    
    
    print(f"Populating of '{db_file}' from '{csv_file}'")
    
    df = pd.read_csv(csv_file)
    print(f"data is loaded with {len(df)} records.")
    
    print("Transforming data for each table...")
    # For the Products table
    products_df = df[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'])
    print(f"Prepared {len(products_df)} unique records for the Products table.")

    # For the Invoices table
    invoices_df = df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country']].drop_duplicates(subset=['InvoiceNo'])
    print(f"Prepared {len(invoices_df)} unique records for the Invoices table.")

    # For the InvoiceItems table (no drops needed here, as we need all line items)
    invoice_items_df = df[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice']]
    print(f"Prepared {len(invoice_items_df)} records for the InvoiceItems table.")

    print("Inserting data into the database...")
    try:
        with sqlite3.connect(db_file) as conn:
            print(f"Connecting to '{db_file}'...")
            
            # Uses 'append' to add data.
            # Important thing on checking:
            # Using 'append' can lead to duplicates if the script is run multiple times.
            #  Use 'replace' if we want to start fresh each time.
            
            products_df.to_sql('Products', conn, if_exists='append', index=False)
            print("Data successfully inserted into 'Products' table.")
            
            invoices_df.to_sql('Invoices', conn, if_exists='append', index=False)
            print("Data successfully inserted into 'Invoices' table.")
            
            invoice_items_df.to_sql('InvoiceItems', conn, if_exists='append', index=False)
            print("Data successfully inserted into 'InvoiceItems' table.")

        print("Database population complete.")

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
        print("Insertion failed. This might be because the script was already run and data exists (Primary Key constraint failed).")
        print("Re-run the script, run uv run prepare_db first to reset the database.")

def main():
    from databases.helpers import sql_config, data_config
    sample_csv = data_config['sample_file']
    db_file = sql_config['db_path']
    populate_database(sample_csv, db_file)