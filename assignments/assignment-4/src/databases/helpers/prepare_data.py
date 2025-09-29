import shutil
import zipfile
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

def download_file(url: str, destination: Path):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    start_time = time.time()
    with destination.open("wb") as f, tqdm(
        unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"Downloading {destination.name}"
    ) as bar:
        for chunk in response.iter_content((64 * 1024)):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
                elapsed = time.time() - start_time
                if elapsed > 0:
                    bar.set_postfix( time=f"{elapsed:.1f}s")


def extract_excel(zip_path: Path, temp_dir: Path) -> Path:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)

    return next(temp_dir.rglob("*.xls*"))

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Some inovice records have missing CustomerID
    df = df.dropna(subset=['InvoiceNo', 'StockCode', 'CustomerID'])
    
    # Ensure StockCode is valid
    # has_valid_length = df['StockCode'].str.len() >= 5
    # is_numeric_prefix = df['StockCode'].str[:5].str.isdigit()
    # df = df[has_valid_length & is_numeric_prefix].copy()
    
    # No more Checking validity of StockCode as per new findings
    
    # Convert CustomerID to int
    df = df.copy()
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    return df

def convert_to_csv(excel_path: Path, data_filepath: str):
    df = pd.read_excel(excel_path)
    df.to_csv(data_filepath, index=False)
    
def preprocess_and_save_sample(data_filepath: str, sample_filepath: str, sample_size: int = 1000, random_state: int = 42):
    df = pd.read_csv(data_filepath)
    df = preprocess_dataset(df)
    print(f"Data cleaned. {len(df)} records remaining after preprocessing.")
    df.to_csv(data_filepath, index=False)

    sample_df = df.sample(n=sample_size, random_state=random_state)
    sample_df.to_csv(sample_filepath, index=False)

def main():
    from databases.helpers.utils import data_config

    dataset_url = data_config['url']
    data_dir = Path(data_config['path'])
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "online_retail.zip"
    temp_dir = data_dir / "temp"

    print("Downloading dataset...")
    # download_file(dataset_url, zip_path)

    print("Extracting Excel file...")
    excel_path = extract_excel(zip_path, temp_dir)

    print("Converting to CSV...")
    convert_to_csv(excel_path, data_config['data_file'])
    
    print("Preprocessing and saving sample...")
    preprocess_and_save_sample(data_config['data_file'], 
                               data_config['sample_file'], 
                               sample_size=data_config['sample_size'], 
                               random_state=data_config['random_state']
                               )

    print("Cleaning up...")
    zip_path.unlink()
    shutil.rmtree(temp_dir)

    print("Done! Data saved in data/ directory.")


if __name__ == "__main__":
    main()