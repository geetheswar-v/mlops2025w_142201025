import pandas as pd
from pymongo import MongoClient, ReplaceOne
from databases.utils import data_config, mongodb_config, load_csv, datetime_parser


def build_docs(df: pd.DataFrame) -> list[dict]:
    docs: list[dict] = []
    grouped = df.groupby('InvoiceNo')
    print(f"Grouping into {len(grouped)} unique invoices...")

    for invoice_no, g in grouped:
        invoice_date = g['InvoiceDate'].iloc[0]
        customer_id = int(g['CustomerID'].iloc[0])
        country = g['Country'].iloc[0]

        items = g[['StockCode', 'Description', 'Quantity', 'UnitPrice']].to_dict(orient='records')

        docs.append({
            'InvoiceNo': str(invoice_no),
            'InvoiceDate': invoice_date.to_pydatetime() if hasattr(invoice_date, 'to_pydatetime') else invoice_date,
            'CustomerID': customer_id,
            'Country': country,
            'Items': items
        })

    print(f"Built {len(docs)} transaction documents")
    return docs


def ingest(transactions: list[dict], batch_size: int = 1000):
    uri = mongodb_config.get('url')
    db_name = mongodb_config.get('db_name')
    coll_name = mongodb_config.get('transactions_collection')
    
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

    transactions = build_docs(df)
    if len(transactions) != invoice_count:
        print("Warning: built docs count differs from unique invoice count.")

    ingest(transactions)


if __name__ == '__main__':
    main()
