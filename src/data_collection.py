import pandas as pd

# these 4 components im pulling from each csv 2023 and 2025 these dicts map each csv column name so they all match before trying to laod the data and clean it
# not all these are one to one because both csvs have differing columns per component

gpu_2023_cols = {
    "Name": "model",
    "Price": "price",
    "Manufacturer": "brand",
    "Chipset": "chipset",
    "Memory": "memory",
    "TDP": "tdp",
    "Core Clock": "core_clock"
}

cpu_2023_cols = {
    "Name": "model",
    "Price": "price",
    "Manufacturer": "brand",
    "Core Count": "core_count",
    "TDP": "tdp",
    "Microarchitecture": "microarchitecture"
}

ram_2023_cols = {
    "Name": "model",
    "Price": "price",
    "Manufacturer": "brand",
    "Speed": "speed",
    "Modules": "modules",
    "CAS Latency": "cas_latency",
    "First Word Latency": "first_word_latency"
}

storage_2023_cols = {
    "Name": "model",
    "Price": "price",
    "Manufacturer": "brand",
    "Capacity": "capacity",
    "Type": "type",
    "Form Factor": "form_factor",
    "Interface": "interface"
}

gpu_2025_cols = {
    "name": "model",
    "price": "price",
    "chipset": "chipset",
    "memory": "memory",
    "core_clock": "core_clock"
}

cpu_2025_cols = {
    "name": "model",
    "price": "price",
    "core_count": "core_count",
    "tdp": "tdp",
    "microarchitecture": "microarchitecture"
}

ram_2025_cols = {
    "name": "model",
    "price": "price",
    "speed": "speed",
    "modules": "modules",
    "cas_latency": "cas_latency",
    "first_word_latency": "first_word_latency"
}

storage_2025_cols = {
    "name": "model",
    "price": "price",
    "capacity": "capacity",
    "type": "type",
    "form_factor": "form_factor",
    "interface": "interface"
}

# function is used to parse the price column in the 2023 csv it converts the value into a float
# otherwise makes it nan

def parse_price_2023(x):

    if pd.isna(x):
        return float("nan")

    s = str(x).replace("$", "").replace(",", "").strip()

    try:
        v = float(s)
    except ValueError:
        return float("nan")

    if v == 0:
        return float("nan")

    return v


# this function is used to normalize the csv data by renaming the columns to match the colmap dict
# 2025 csvs dont have a brand col so adds empty ones to keep consistency
# also cleans the price column by converting to float
# if the value is 0 it makes it nan same as 2023
# 2025 prices are already formatted where df cleaning can be done with simple fucntions

def normalize_csv(path, category, year, colmap):

    df = pd.read_csv(path)
    df = df[list(colmap.keys())].rename(columns=colmap)

    if "brand" not in df.columns:
        df["brand"] = pd.NA

    df["raw_first_word"] = df["model"].astype(str).str.split().str[0]

    for col in df.columns:

        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if year == 2023:
        df["price"] = df["price"].map(parse_price_2023)

    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price"] = df["price"].mask(df["price"] == 0)

    df = df[df["price"].notna() & (df["price"] > 0)]

    df["category"] = category
    df["year"] = year

    return df

# takes every csv and normalizes it then concats them into a single df
# lowercases the model column so 2023 and 2025 names match 
# prints the number of rows in each csv and the total number of rows in the combined df

def main():

    files = [
        ("2023 GPUs", "raw_data/2023/gpus_detailed.csv", "GPU", 2023, gpu_2023_cols),
        ("2023 CPUs", "raw_data/2023/cpus_detailed.csv", "CPU", 2023, cpu_2023_cols),
        ("2023 RAM", "raw_data/2023/memory_detailed.csv", "RAM", 2023, ram_2023_cols),
        ("2023 Storage", "raw_data/2023/storage_detailed.csv", "Storage", 2023, storage_2023_cols),
        ("2025 GPUs", "raw_data/2025/video-card.csv", "GPU", 2025, gpu_2025_cols),
        ("2025 CPUs", "raw_data/2025/cpu.csv", "CPU", 2025, cpu_2025_cols),
        ("2025 RAM", "raw_data/2025/memory.csv", "RAM", 2025, ram_2025_cols),
        ("2025 Storage", "raw_data/2025/internal-hard-drive.csv", "Storage", 2025, storage_2025_cols),
    ]

    parts = []

    for label, path, category, year, colmap in files:

        df = normalize_csv(path, category, year, colmap)
        parts.append(df)

        print(f"{label}: {len(df)} rows")

    combined = pd.concat(parts, ignore_index=True)
    combined["model"] = combined["model"].str.lower().str.strip()

    combined.to_csv("cleaned_data/combined_parts.csv", index=False)

    print(f"Total rows: {len(combined)}")
    print("Wrote cleaned_data/combined_parts.csv")

    print("\nIngesting external GPU price data (HardwareDealsCo)")
    import gpu_real_data
    gpu_real_data.main()

    print("\nIngesting external drive price data (HardwareDealsCo)")
    import drive_real_data
    drive_real_data.main()

    print("\nIngesting external RAM price data (HardwareDealsCo)")
    import ram_real_data
    ram_real_data.main()
