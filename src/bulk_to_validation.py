import sys
import os
import pandas as pd


pcpp_2025_format = [
    ("GPU", "video-card.csv", "name", "price", ["chipset"]),
    ("CPU", "cpu.csv", "name", "price", []),
    ("RAM", "memory.csv", "name", "price", ["speed"]),
    ("Storage", "internal-hard-drive.csv", "name", "price", ["capacity"]),
]

pcpp_2023_format = [
    ("GPU", "gpus_detailed.csv", "Name", "Price", ["Chipset"]),
    ("CPU", "cpus_detailed.csv", "Name", "Price", []),
    ("RAM", "memory_detailed.csv", "Name", "Price", ["Speed"]),
    ("Storage", "storage_detailed.csv", "Name", "Price", ["Capacity"]),
]


def parse_price(x):

    if pd.isna(x):
        return None

    s = str(x).replace("$", "").replace(",", "").strip()

    try:
        v = float(s)

        if v <= 0:
            return None

        return v

    except ValueError:
        return None


def main():

    if len(sys.argv) < 3:
        print("usage: python src/bulk_to_validation.py <snapshot_folder> <date_YYYY-MM-DD> [output_path]")
        print()
        print("  snapshot_folder: a directory with PCPartPicker-format CSVs.")
        print("  date_YYYY-MM-DD: the snapshot date to align with synthetic series.")
        print("  output_path:     defaults to validation_data_bulk.csv")
        print()
        print("Auto-detects PCPartPicker 2023 'detailed' format (gpus_detailed.csv etc.)")
        print("and 2025 lowercase format (video-card.csv etc.).")
        return

    folder = sys.argv[1]
    date = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "validation_data_bulk.csv"

    if not os.path.isdir(folder):
        print(f"folder not found: {folder}")
        return

    files = pcpp_2025_format

    if any(os.path.exists(os.path.join(folder, f[1])) for f in pcpp_2023_format):
        files = pcpp_2023_format
        print("detected 2023-style 'detailed' file naming")

    else:
        print("using 2025-style lowercase file naming")

    rows = []

    for cat, fname, name_col, price_col, extra_cols in files:

        path = os.path.join(folder, fname)

        if not os.path.exists(path):
            print(f"  skipping {cat}: {path} not found")
            continue

        df = pd.read_csv(path)

        if name_col not in df.columns or price_col not in df.columns:
            print(f"  skipping {cat}: required columns {name_col}/{price_col} not in {path}")
            continue

        keep = [name_col, price_col] + [c for c in extra_cols if c in df.columns]
        df = df[keep].copy()
        df["price_clean"] = df[price_col].map(parse_price)
        df = df.dropna(subset=["price_clean", name_col])

        for _, row in df.iterrows():

            parts = [str(row[name_col]).strip()]

            for c in extra_cols:

                if c in row and not pd.isna(row[c]):
                    parts.append(str(row[c]).strip())

            query = " ".join(parts).lower()

            rows.append({
                "query": query,
                "date": date,
                "external_price": round(float(row["price_clean"]), 2),
                "source": f"bulk snapshot ({cat}) - {os.path.basename(folder)}",
                "notes": "",
            })

        print(f"  {cat}: {len(df):,} rows ingested")

    if not rows:
        print("\nNo rows found. Make sure your CSVs match the expected naming.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df):,} validation rows to {out_path}")
    print(f"Now run: python src/external_validation.py {out_path}")


if __name__ == "__main__":
    main()
