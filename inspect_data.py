import os
import pandas as pd

folders = [
    "raw_data/2023",
    "raw_data/2025"
]

for folder in folders:

    if not os.path.exists(folder):
        print(f"folder not found: {folder}")
        continue

    print(f"\n{folder}")

    for file in os.listdir(folder):

        if file.endswith(".csv"):

            path = os.path.join(folder, file)
            df = pd.read_csv(path, nrows=3)

            print(f"\n--- {file} ---")
            print(f"columns: {list(df.columns)}")
            print(df.head(3).to_string())