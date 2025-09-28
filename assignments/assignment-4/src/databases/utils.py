import tomllib
import os

# Load and parse the TOML configuration file (from Assignment 3)
def load_toml(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The Config {file_path} does not exist.")
    
    with open(file_path, 'rb') as file:
        data = tomllib.load(file)
    return data

config = load_toml('config.toml')
data_config, sql_config = config['data'], config['sqlite']