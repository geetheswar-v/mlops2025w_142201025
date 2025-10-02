from pymongo import MongoClient, ReplaceOne, InsertOne, UpdateOne, DeleteOne
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
from databases.utils import mongo_config

def get_mongo_client(deployment: str = "local"):
    try:
        config = mongo_config[deployment]
        
        client = MongoClient(
            config['url'],
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
            retryWrites=True,
            w='majority'
        )
        # Test the connection
        client.admin.command('ping')
        db = client[config['db_name']]
        transactions_collection = db[config['transactions_collection']]
        transactions_customers_collection = db[config['transactions_customers_collection']]
        customers_collection = db[config['customers_collection']]
        
        deployment_name = "MongoDB Atlas" if deployment == "atlas" else "Local MongoDB"
        print(f"Successfully connected to {deployment_name} with connection pooling.")
        return client, db, transactions_collection, transactions_customers_collection, customers_collection
    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB ({deployment}): {e}")
        return None, None, None, None, None

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