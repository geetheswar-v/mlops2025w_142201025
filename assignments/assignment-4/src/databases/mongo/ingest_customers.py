import pandas as pd
from pymongo import MongoClient, ReplaceOne
from databases.utils import data_config, mongodb_config, load_csv, datetime_parser

def build_customer_docs(df: pd.DataFrame) -> list[dict]:
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

    invoice_meta = df[['InvoiceNo', 'InvoiceDate', 'CustomerID']].drop_duplicates(subset=['InvoiceNo'])
    invoices_df = pd.merge(invoice_meta, nested_items, on='InvoiceNo')
    invoices_df['InvoiceDate'] = invoices_df['InvoiceDate'].dt.to_pydatetime()
    
    nested_invoices = (
        invoices_df.groupby('CustomerID')[['InvoiceNo', 'InvoiceDate', 'Items']]
        .apply(lambda x: x.to_dict('records'))
        .reset_index(name='Invoices')
    )
    customer_meta = df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
    
    final_df = pd.merge(customer_meta, nested_invoices, on='CustomerID')
    
    docs = final_df.to_dict('records')
    return docs


def ingest_customers(customers: list[dict], batch_size: int = 500):
    uri = mongodb_config.get('uri', 'mongodb://localhost:27017')
    db_name = mongodb_config.get('db_name', 'retail')
    coll_name = mongodb_config.get('customers_collection', 'customers')

    client = MongoClient(uri)
    db = client[db_name]
    coll = db[coll_name]

    coll.create_index('CustomerID', unique=True)

    ops = []
    processed = 0
    for doc in customers:
        ops.append(ReplaceOne({'CustomerID': doc['CustomerID']}, doc, upsert=True))
        if len(ops) >= batch_size:
            coll.bulk_write(ops, ordered=False)
            processed += len(ops)
            print(f"Upsert batch complete: total processed {processed}")
            ops = []

    if ops:
        coll.bulk_write(ops, ordered=False)
        processed += len(ops)

    print(f"Done. Total customers processed (inserted or updated): {processed}.")


def main():
    csv_path = data_config.get('dataset_path')
    df = load_csv(csv_path)
    df = datetime_parser(df, 'InvoiceDate')
    print(f"Total Records: {len(df)} rows.")

    customer_count = df['CustomerID'].nunique()
    print(f"Unique customers found: {customer_count}")

    customer_documents = build_customer_docs(df)
    if len(customer_documents) != customer_count:
        print(f"Warning: built docs count ({len(customer_documents)}) differs from unique customer count ({customer_count}).")

    ingest_customers(customer_documents)


if __name__ == '__main__':
    main()