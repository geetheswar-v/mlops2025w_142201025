from pathlib import Path
import pandas as pd
import os

def convert_to_csv(excel_path: Path, data_filepath: Path):
    df = pd.read_excel(excel_path)
    df.to_csv(data_filepath, index=False)
    
def load_csv(path: Path) -> pd.DataFrame:
    # check if file exists
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df

def preprocess_and_save_sample(data_filepath: Path, sample_filepath: Path, sample_size: int = 1000, random_state: int = 42):
    df = load_csv(data_filepath)
    df = preprocess_dataset(df)
    print(f"Data pre. {len(df)} records remaining after preprocessing.")
    df.to_csv(data_filepath, index=False)

    sample_df = df.sample(n=sample_size, random_state=random_state)
    sample_df.to_csv(sample_filepath, index=False)
    
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Some inovice records have missing CustomerID and other critical fields
    df.dropna(subset=["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate"], inplace=True)
    
    # Found some duplicate records
    df.drop_duplicates(inplace=True)
    
    # Convert CustomerID to int
    df = df.copy()
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Ensure StockCode is valid
    # has_valid_length = df['StockCode'].str.len() >= 5
    # is_numeric_prefix = df['StockCode'].str[:5].str.isdigit()
    # df = df[has_valid_length & is_numeric_prefix].copy()
    # No more Checking validity of StockCode as per new findings
    
    return df

def datetime_parser(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[~df[date_col].isna()]
    return df

