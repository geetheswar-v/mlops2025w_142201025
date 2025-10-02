import tomllib
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load and parse the TOML configuration file (from Assignment 3)
def load_toml(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The Config {file_path} does not exist.")
    
    with open(file_path, 'rb') as file:
        data = tomllib.load(file)
    return data

config = load_toml('config.toml')


## Data Configuration
data_raw = config.get("data", {})
data_path = Path(data_raw.get("path", "data"))
data_config = {
    "dataset_url": data_raw.get("url"),
    "data_path": data_path,
    "dataset_path": data_path / data_raw.get("data_file", "retail.csv"),
    "sample_path": data_path / data_raw.get("sample_file", "sample_retail.csv"),
    "sample_size": int(data_raw.get("sample_size", 5000)),
    "random_state": int(data_raw.get("random_state", 42)),
}


## SQLite Configuration
sql_raw = config.get("sqlite", {})
sql_config = {
    "url": Path(sql_raw.get("url", "data/retail.db")),
    "schema": sql_raw.get("schema", "retail.db"),
    "schema_path": Path(sql_raw.get("schema_path", "sql/schema.sql")),
}


## MongoDB Configuration
mongo_raw = config.get("mongo", {
    "db_name": "retail",
    "transactions_collection": "transactions",
    "customers_collection": "customers"
})

mongo_local = mongo_raw.get("local", {
    "url": "mongodb://localhost:27017"
})

atlas_raw = mongo_raw.get("atlas", {
    "host": "retailcluster.fwdi4ka.mongodb.net",
    "retry_writes": "true",
    "writes": "majority",
    "app_name": "RetailCluster"
})

# Construct the MongoDB Atlas connection string using environment variables for security
atlas_password = os.getenv("ATLAS_PASSWORD", "")
atlas_username = os.getenv("ATLAS_USERNAME", "")
atlas_url = f"mongodb+srv://{atlas_username}:{atlas_password}@{atlas_raw.get('host', 'retailcluster.fwdi4ka.mongodb.net')}/?retryWrites={atlas_raw.get('retry_writes', 'true')}&w={atlas_raw.get('writes', 'majority')}&appName={atlas_raw.get('app_name', 'RetailCluster')}"

mongo_base_config = {
    "db_name": mongo_raw.get("db_name", "retail"),
    "transactions_collection": mongo_raw.get("transactions_collection", "transactions"),
    "transactions_customers_collection": mongo_raw.get("transactions_customers_collection", "transactions_customers"),
    "customers_collection": mongo_raw.get("customers_collection", "customers"),
}

mongo_config = {
    "local": {**mongo_base_config, "url": mongo_local.get("url", "mongodb://localhost:27017")},
    "atlas": {**mongo_base_config, "url": atlas_url}
}