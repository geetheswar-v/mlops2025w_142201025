import pandas as pd
from databases.utils import data_config, load_csv, datetime_parser
from databases.mongo import customers_collection, transactions_collection, ingest_docs

def build_docs(df: pd.DataFrame):
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

    invoice_meta = df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country']].drop_duplicates(subset=['InvoiceNo'])
    invoice_df = pd.merge(invoice_meta, nested_items, on='InvoiceNo')

    invoice_df['InvoiceDate'] = invoice_df['InvoiceDate'].dt.to_pydatetime()
    nested_invoices = (
        invoice_df.groupby('CustomerID')[['InvoiceNo', 'InvoiceDate', 'Items']]
        .apply(lambda x: x.to_dict('records'))
        .reset_index(name='Invoices')
    )
    customer_meta = df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
    customer_df = pd.merge(customer_meta, nested_invoices, on='CustomerID')

    invoice_docs = invoice_df.to_dict('records')
    customer_docs = customer_df.to_dict('records')

    return invoice_docs, customer_docs

def ingest_data(invoices: list[dict], customers: list[dict], invoice_batch_size: int = 1000, customer_batch_size: int = 500):
    transactions_collection.create_index('InvoiceNo', unique=True)
    customers_collection.create_index('CustomerID', unique=True)
    ingest_docs(transactions_collection, 'InvoiceNo', invoices, invoice_batch_size)
    ingest_docs(customers_collection, 'CustomerID', customers, customer_batch_size)
    
def main():
    csv_path = data_config.get('dataset_path')
    df = load_csv(csv_path)
    df = datetime_parser(df, 'InvoiceDate')
    print(f"Total Records: {len(df)} rows.")

    invoice_count = df['InvoiceNo'].nunique()
    customer_count = df['CustomerID'].nunique()
    print(f"Unique invoices found: {invoice_count}")
    print(f"Unique customers found: {customer_count}")

    invoice_docs, customer_docs = build_docs(df)
    if len(invoice_docs) != invoice_count:
        print("Warning: built invoice docs count differs from unique invoice count.")
    if len(customer_docs) != customer_count:
        print("Warning: built customer docs count differs from unique customer count.")

    ingest_data(invoice_docs, customer_docs)