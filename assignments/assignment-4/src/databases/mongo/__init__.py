from pymongo import MongoClient, ReplaceOne
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
from databases.utils import mongodb_config

# Establish a reusable client connection
def get_mongo_client():
    try:
        client = MongoClient(mongodb_config['url'], 
                             maxPoolSize=50,
                             minPoolSize=10,
                             serverSelectionTimeoutMS=5000
                            )
        # Test the connection
        client.admin.command('ping')
        db = client[mongodb_config['db_name']]
        transactions_collection = db[mongodb_config['transactions_collection']]
        customers_collection = db[mongodb_config['customers_collection']]
        print("Successfully connected to MongoDB.")
        return client, db, transactions_collection, customers_collection
    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        return None, None, None, None

client, db, transactions_collection, customers_collection = get_mongo_client()

# Decorator to handle MongoDB errors
def handle_mongo_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionFailure as e:
            print(f"Connection error: {e}")
            return None
        except OperationFailure as e:
            print(f"Operation error: {e}")
            return None
        except BulkWriteError as e:
            print(f"Bulk write error: {e.details}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    return wrapper

# ingest docs into a collection
@handle_mongo_errors
def ingest_docs(collection, replace_id: str, docs: list[dict], batch_size: int = 1000):
    if collection is None:
        print("No valid MongoDB collection provided.")
        return

    ops = []
    processed = 0
    for doc in docs:
        ops.append(ReplaceOne({replace_id: doc[replace_id]}, doc, upsert=True))
        if len(ops) >= batch_size:
            collection.bulk_write(ops, ordered=False)
            processed += len(ops)
            print(f"Inserted batch: total processed {processed}")
            ops = []

    if ops:
        collection.bulk_write(ops, ordered=False)
        processed += len(ops)

    print(f"Done. Total documents processed (inserted): {processed}.")