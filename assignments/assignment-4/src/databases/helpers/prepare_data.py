import shutil
import zipfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

from databases.utils import data_config, convert_to_csv, preprocess_and_save_sample

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

def main():
    dataset_url = data_config.get('dataset_url')
    print(f"Dataset URL: {dataset_url}")
    
    data_dir = data_config.get('data_path')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "online_retail.zip"
    temp_dir = data_dir / "temp"
    
    csv_path = data_config.get('dataset_path')
    sample_path = data_config.get('sample_path')
    
    sample_size = data_config.get('sample_size')
    random_state = data_config.get('random_state')

    print("Downloading dataset...")
    download_file(dataset_url, zip_path)

    print("Extracting Excel file...")
    excel_path = extract_excel(zip_path, temp_dir)

    print("Converting to CSV...")
    convert_to_csv(excel_path, csv_path)
    
    print("Preprocessing and saving sample...")
    preprocess_and_save_sample(csv_path, 
                               sample_path, 
                               sample_size=sample_size, 
                               random_state=random_state
                               )

    print("Cleaning up...")
    zip_path.unlink()
    shutil.rmtree(temp_dir)

    print(f"Done! Data saved in {data_dir} directory.")


if __name__ == "__main__":
    main()