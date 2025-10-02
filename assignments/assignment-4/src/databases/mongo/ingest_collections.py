import pandas as pd
import argparse
from databases.utils import data_config, load_csv, datetime_parser
from databases.mongo import get_mongo_client, ingest_docs

def build_docs(df: pd.DataFrame):
    # Aggregate items by invoice
    items_agg = df.groupby(['InvoiceNo', 'StockCode']).agg(
        Description=('Description', 'first'),
        Quantity=('Quantity', 'sum'),
        UnitPrice=('UnitPrice', 'max')
    ).reset_index()
    nested_items = (
        items_agg.groupby('InvoiceNo')[['StockCode', 'Description', 'Quantity', 'UnitPrice']]
        .apply(lambda x: x.to_dict('records'))
        .reset_index(name='Items')
    )

    # Transaction-centric: Invoice with CustomerID reference only (no Country embedded)
    invoice_meta_transaction = df[['InvoiceNo', 'InvoiceDate', 'CustomerID']].drop_duplicates(subset=['InvoiceNo'])
    transaction_docs = pd.merge(invoice_meta_transaction, nested_items, on='InvoiceNo')
    transaction_docs['InvoiceDate'] = transaction_docs['InvoiceDate'].dt.to_pydatetime()
    transaction_docs = transaction_docs.to_dict('records')
    
    # Transaction-centric: Customer reference collection (CustomerID + Country only)
    transactions_customers_docs = df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID']).to_dict('records')

    # Customer-centric: Invoice with full metadata for embedding
    invoice_meta_customer = df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country']].drop_duplicates(subset=['InvoiceNo'])
    invoice_df = pd.merge(invoice_meta_customer, nested_items, on='InvoiceNo')
    invoice_df['InvoiceDate'] = invoice_df['InvoiceDate'].dt.to_pydatetime()
    
    # Customer-centric: Nested invoices within customer
    nested_invoices = (
        invoice_df.groupby('CustomerID')[['InvoiceNo', 'InvoiceDate', 'Items']]
        .apply(lambda x: x.to_dict('records'))
        .reset_index(name='Invoices')
    )
    customer_meta = df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
    customer_centric_docs = pd.merge(customer_meta, nested_invoices, on='CustomerID').to_dict('records')

    return transaction_docs, transactions_customers_docs, customer_centric_docs

def ingest_data_local(transactions: list[dict], transactions_customers: list[dict], customers: list[dict], 
                      transactions_collection, transactions_customers_collection, customers_collection, 
                      transaction_batch_size: int = 1000, customer_batch_size: int = 500):
    print("Ingesting data for local deployment (both transaction-centric and customer-centric approaches)")
    
    # Create indexes
    transactions_collection.create_index('InvoiceNo', unique=True)
    transactions_collection.create_index('CustomerID')  # For lookups
    transactions_customers_collection.create_index('CustomerID', unique=True)
    customers_collection.create_index('CustomerID', unique=True)
    
    print("Ingesting transaction-centric documents...")
    print("  - Transactions collection (invoices with CustomerID reference)...")
    ingest_docs(transactions_collection, 'InvoiceNo', transactions, transaction_batch_size)
    
    print("  - Transactions_customers collection (customer reference data)...")
    ingest_docs(transactions_customers_collection, 'CustomerID', transactions_customers, customer_batch_size)
    
    print("Ingesting customer-centric documents...")
    ingest_docs(customers_collection, 'CustomerID', customers, customer_batch_size)

def ingest_data_atlas(transactions: list[dict], transactions_customers: list[dict], 
                      transactions_collection, transactions_customers_collection, 
                      transaction_batch_size: int = 1000, customer_batch_size: int = 500):
    print("Ingesting data for Atlas deployment (transaction-centric approach only)")
    
    # Create indexes
    transactions_collection.create_index('InvoiceNo', unique=True)
    transactions_collection.create_index('CustomerID')  # For lookups
    transactions_customers_collection.create_index('CustomerID', unique=True)
    
    print("Ingesting transaction-centric documents...")
    print("  - Transactions collection (invoices with CustomerID reference)...")
    ingest_docs(transactions_collection, 'InvoiceNo', transactions, transaction_batch_size)
    
    print("  - Transactions_customers collection (customer reference data)...")
    ingest_docs(transactions_customers_collection, 'CustomerID', transactions_customers, customer_batch_size)

def main():
    parser = argparse.ArgumentParser(description='Ingest retail data into MongoDB')
    parser.add_argument('--local', action='store_true', help='Use local MongoDB deployment')
    parser.add_argument('--atlas', action='store_true', help='Use MongoDB Atlas deployment')
    
    args = parser.parse_args()
    
    if not args.local and not args.atlas:
        print("Please specify --local or --atlas")
        return
    
    if args.local and args.atlas:
        print("Please specify only one deployment option")
        return
    
    # Determine deployment
    deployment = "local" if args.local else "atlas"
    
    # Get MongoDB client and collections
    client, db, transactions_collection, transactions_customers_collection, customers_collection = get_mongo_client(deployment)
    
    if not client:
        print("Failed to connect to MongoDB")
        return
    
    # Load and process data
    csv_path = data_config.get('dataset_path')
    df = load_csv(csv_path)
    df = datetime_parser(df, 'InvoiceDate')
    print(f"Total Records: {len(df)} rows.")

    invoice_count = df['InvoiceNo'].nunique()
    customer_count = df['CustomerID'].nunique()
    print(f"Unique invoices found: {invoice_count}")
    print(f"Unique customers found: {customer_count}")

    # Build documents
    transaction_docs, transactions_customers_docs, customer_centric_docs = build_docs(df)
    if len(transaction_docs) != invoice_count:
        print("Warning: built transaction docs count differs from unique invoice count.")
    if len(transactions_customers_docs) != customer_count:
        print("Warning: built transactions_customers docs count differs from unique customer count.")
    if len(customer_centric_docs) != customer_count:
        print("Warning: built customer_centric docs count differs from unique customer count.")

    # Ingest based on deployment
    if deployment == "local":
        ingest_data_local(transaction_docs, transactions_customers_docs, customer_centric_docs, 
                         transactions_collection, transactions_customers_collection, customers_collection)
    else:  # atlas
        ingest_data_atlas(transaction_docs, transactions_customers_docs,
                         transactions_collection, transactions_customers_collection)
    
    print(f"Data ingestion to {deployment} completed successfully!")