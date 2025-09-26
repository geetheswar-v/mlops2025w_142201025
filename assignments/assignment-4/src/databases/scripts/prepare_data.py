import shutil
import zipfile
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from databases.utils import DATA_DIR, DATASET_URL

sample_size = 2000

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


def convert_to_csv(excel_path: Path, output_dir: Path, sample_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(excel_path)

    df.to_csv(output_dir / "data.csv", index=False)
    df.sample(n=min(sample_size, len(df)), random_state=42).to_csv(
        output_dir / "sample.csv", index=False
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "online_retail.zip"
    temp_dir = DATA_DIR / "temp"

    print("Downloading dataset...")
    download_file(DATASET_URL, zip_path)

    print("Extracting Excel file...")
    excel_path = extract_excel(zip_path, temp_dir)

    print("Converting to CSV...")
    convert_to_csv(excel_path, DATA_DIR, sample_size)

    print("Cleaning up...")
    zip_path.unlink()
    shutil.rmtree(temp_dir)

    print("Done! Data saved in data/ directory.")


if __name__ == "__main__":
    main()