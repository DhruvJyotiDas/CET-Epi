import os
import zipfile
import requests

DATA_DIR = "data/raw"
URL = "https://archive.ics.uci.edu/static/public/580/hungarian+chickenpox+cases.zip"
ZIP_PATH = os.path.join(DATA_DIR, "chickenpox.zip")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading from {URL}...")
    r = requests.get(URL)

    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

    print(f"Saved to {ZIP_PATH}")


def extract():
    print("Extracting...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Extraction done.")

    print("\nFiles extracted:")
    for f in os.listdir(DATA_DIR):
        print(" -", f)


if __name__ == "__main__":
    download()
    extract()
    print("\n✅ Dataset ready")
