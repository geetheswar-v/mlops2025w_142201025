import tomllib
import os
from pathlib import Path

# Load and parse the TOML configuration file (from Assignment 3)
def load_toml(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The Config {file_path} does not exist.")
    
    with open(file_path, 'rb') as file:
        data = tomllib.load(file)
    return data

config = load_toml('config.toml')

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


sql_raw = config.get("sqlite", {})
sql_config = {
    "url": Path(sql_raw.get("url", "data/retail.db")),
    "schema": sql_raw.get("schema", "retail.db"),
    "schema_path": Path(sql_raw.get("schema_path", "sql/schema.sql")),
}

mongo_raw = config.get("mongodb", {})
mongodb_config = {
    "url": mongo_raw.get("url", "mongodb://localhost:27017"),
    "db_name": mongo_raw.get("db_name", "retail"),
    "transactions_collection": mongo_raw.get("transactions_collection", "transactions"),
    "customers_collection": mongo_raw.get("customers_collection", "customers"),
}