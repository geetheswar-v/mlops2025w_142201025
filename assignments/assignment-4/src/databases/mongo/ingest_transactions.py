import pandas as pd
from pymongo import MongoClient, ReplaceOne
from databases.utils import data_config, mongodb_config, load_csv, datetime_parser


def build_transactions_docs(df: pd.DataFrame) -> list[dict]:
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

    final_df = pd.merge(invoice_meta, nested_items, on='InvoiceNo')

    final_df['InvoiceDate'] = final_df['InvoiceDate'].dt.to_pydatetime()
    final_df['InvoiceNo'] = final_df['InvoiceNo'].astype(str)
    final_df['CustomerID'] = final_df['CustomerID'].astype(int)
    
    docs = final_df.to_dict('records')

    return docs


def ingest_transactions(transactions: list[dict], batch_size: int = 1000):
    uri = mongodb_config.get('uri', 'mongodb://localhost:27017')
    db_name = mongodb_config.get('db_name', 'retail')
    coll_name = mongodb_config.get('transactions_collection', 'transactions')

    client = MongoClient(uri)
    db = client[db_name]
    coll = db[coll_name]

    coll.create_index('InvoiceNo', unique=True)

    ops = []
    processed = 0
    for doc in transactions:
        ops.append(ReplaceOne({'InvoiceNo': doc['InvoiceNo']}, doc, upsert=True))
        if len(ops) >= batch_size:
            coll.bulk_write(ops, ordered=False)
            processed += len(ops)
            print(f"Upsert batch complete: total processed {processed}")
            ops = []

    if ops:
        coll.bulk_write(ops, ordered=False)
        processed += len(ops)

    print(f"Done. Total invoices processed (inserted or updated): {processed}.")


def main():
    csv_path = data_config.get('dataset_path')
    df = load_csv(csv_path)
    df = datetime_parser(df, 'InvoiceDate')
    print(f"Total Records: {len(df)} rows.")

    invoice_count = df['InvoiceNo'].nunique()
    print(f"Unique invoices found: {invoice_count}")

    transactions_documents = build_transactions_docs(df)
    if len(transactions_documents) != invoice_count:
        print("Warning: built docs count differs from unique invoice count.")

    ingest_transactions(transactions_documents)


if __name__ == '__main__':
    main()
