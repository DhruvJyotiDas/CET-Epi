import os
import zipfile
import requests
from pathlib import Path


def download_file(url, save_path):
    print(f"Downloading from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {save_path}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def setup_directories():
    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path


def download_chickenpox_dataset():
    raw_path = setup_directories()

    # UCI dataset direct CSV (cleanest)
    data_url = "https://archive.ics.uci.edu/static/public/580/hungarian+chickenpox+cases.zip"

    zip_path = raw_path / "chickenpox.zip"

    # Download ZIP
    download_file(data_url, zip_path)

    # Extract
    extract_zip(zip_path, raw_path)

    # Rename for consistency
    for file in raw_path.iterdir():
        if "chickenpox" in file.name.lower() and file.suffix == ".csv":
            new_name = raw_path / "hungary_chickenpox.csv"
            file.rename(new_name)
            print(f"Renamed to {new_name}")

    # Create EDGE FILE manually (VERY IMPORTANT)
    edge_file = raw_path / "hungary_county_edges.csv"

    if not edge_file.exists():
        print("Creating default county graph...")

        # simple ring + neighbor graph (fallback)
        edges = []
        n = 20

        for i in range(n):
            edges.append((i, (i + 1) % n))
            edges.append(((i + 1) % n, i))

            edges.append((i, (i + 2) % n))
            edges.append(((i + 2) % n, i))

        with open(edge_file, "w") as f:
            for e in edges:
                f.write(f"{e[0]},{e[1]}\n")

        print(f"Created edge file: {edge_file}")

    print("\n✅ Dataset ready in data/raw/")


if __name__ == "__main__":
    download_chickenpox_dataset()
