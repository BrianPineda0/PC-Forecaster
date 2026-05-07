import re
import pandas as pd

# regex that gets rid of workstation gpus got rid of these because they skew prices and arent relevant to the project
workstation_gpus = re.compile(r"quadro|firepro|tesla|rtx a\d|radeon pro|radeon vii pro", re.IGNORECASE,)


# normalize profuct brtand names for consistency
def clean_product_name(name):

    if pd.isna(name):
        return name

    text = str(name)
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return pd.NA if text == "" else text


# gets the number of vram from gpus makes it a float
def process_gpu_vram(mem):

    if pd.isna(mem):
        return float("nan")

    text = str(mem).strip()
    match = re.search(r"([\d.]+)\s*GB", text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return float("nan")


# gets the tdp from cpu makes it a float
def process_watts(w):

    if pd.isna(w):
        return float("nan")

    text = str(w).strip()
    match = re.search(r"([\d.]+)\s*W", text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return float("nan")


# takes ddr type and mhz
def process_ram_speed(speed):

    if pd.isna(speed):
        return float("nan"), float("nan")

    text = str(speed).strip()

    match = re.match(r"DDR(\d+)[- ](\d+)", text, re.IGNORECASE)

    if match:
        return float(match.group(1)), float(match.group(2))

    match = re.match(r"(\d+),(\d+)", text)

    if match:
        return float(match.group(1)), float(match.group(2))

    return float("nan"), float("nan")


# takes in number of sticks and gb size
def process_ram_stick(modules):

    if pd.isna(modules):
        return float("nan"), float("nan")

    text = str(modules).strip()

    match = re.match(r"(\d+)\s*x\s*(\d+)", text, re.IGNORECASE)

    if match:
        return float(match.group(1)), float(match.group(2))

    match = re.match(r"(\d+),(\d+)", text)

    if match:
        return float(match.group(1)), float(match.group(2))

    return float("nan"), float("nan")


# takes the gpu clock speed in mhz
def process_gpu_clock_speed(clock):

    if pd.isna(clock):
        return float("nan")

    text = str(clock).strip()
    match = re.search(r"([\d.]+)\s*MHz", text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return float("nan")


# takes the ram latency
def process_ram_latency(latency):

    if pd.isna(latency):
        return float("nan")

    text = str(latency).strip()
    match = re.search(r"([\d.]+)\s*ns", text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return float("nan")


# normalizes storage metric by converting tbs to gbs by multiplying by 1000
def process_storage_capacity(cap):

    if pd.isna(cap):
        return float("nan")

    text = str(cap).strip()

    match = re.search(r"([\d.]+)\s*TB", text, re.IGNORECASE)

    if match:
        return float(match.group(1)) * 1000

    match = re.search(r"([\d.]+)\s*GB", text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return float("nan")


# takes the storage interface and returns the type and pcie gen
def process_storage_interface(inter):

    if pd.isna(inter):
        return pd.NA, float("nan")

    text = str(inter).strip().lower()

    if "m.2" in text and "pcie" in text:

        match = re.search(r"pcie\s*(\d+(?:\.\d+)?)", text)
        gen = float(int(float(match.group(1)))) if match else float("nan")
        return "M.2 PCIe", gen

    if "m.2" in text and "sata" in text:
        return "M.2 SATA", float("nan")

    if "msata" in text:
        return "M.2 SATA", float("nan")

    if "sata" in text:
        return "SATA", float("nan")

    if "sas" in text:
        return "SAS", float("nan")

    return "Other", float("nan")

# takes the storage type and returns its hdd, ssd, or hybrid
def normalize_storage_type(type):

    if pd.isna(type):
        return pd.NA

    text = str(type).strip().lower()

    if "ssd" in text or "nvme" in text or "nand" in text:
        return "SSD"

    if "hybrid" in text:
        return "Hybrid"

    if re.match(r"^\d+(\s*rpm)?$", text.replace(" ", ""), re.IGNORECASE):
        return "HDD"

    return "SSD"


# builds a brand map using brands from 2023 to fill up the brand column in the 2025 df
def build_brand_map(df):

    branded = df[(df["year"] == 2023) & df["brand"].notna()].copy()
    branded["fw"] = branded["model"].astype(str).str.split().str[0].str.lower()

    return (branded.groupby("fw")["brand"]
            .agg(lambda x: x.mode().iloc[0])
            .str.strip()
            .to_dict())


# infers the brand column using the brand map
def infer_brand_column(df, brand_map):

    fw = df["raw_first_word"].astype(str)
    canonical = fw.str.lower().map(brand_map)

    return canonical.fillna(fw).where(fw.notna() & (fw != "nan"), pd.NA)

# build unique keys by taking aspects of a component in order to differentiate different parts with similar names but different specs
def build_component_id(df):

    ids = df["model"].copy().astype(str)

    gpu = df["category"] == "GPU"

    ids[gpu] = (df.loc[gpu, "model"].fillna("").astype(str)+ " | " + df.loc[gpu, "chipset"].fillna("").astype(str))

    sto = df["category"] == "Storage"

    def cap_label(gb):

        if pd.isna(gb):
            return ""

        gb = float(gb)

        if gb >= 1000:
            return f"{gb/1000:.1f}TB"

        return f"{int(gb)}GB"

    sto_type = df.loc[sto, "storage_type"].fillna("Unknown").astype(str)

    ids[sto] = (df.loc[sto, "model"].fillna("").astype(str) + " | " + df.loc[sto, "capacity_gb"].map(cap_label) + " | " + sto_type)

    ram = df["category"] == "RAM"

    def ram_label(row):

        ddr = row["ddr_gen"]
        speed = row["speed_mhz"]
        count = row["module_count"]
        size = row["module_size_gb"]

        parts = []

        if not pd.isna(ddr):
            parts.append(f"DDR{int(ddr)}")

        if not pd.isna(speed):
            parts.append(f"{int(speed)}MHz")

        if not pd.isna(count) and not pd.isna(size):
            parts.append(f"{int(count)}x{int(size)}GB")

        return " | ".join(parts) if parts else ""

    ram_suffix = df[ram].apply(ram_label, axis=1)

    ids[ram] = (df.loc[ram, "model"].fillna("").astype(str) + " | " + ram_suffix)

    ids = ids.str.rstrip(" |").str.strip()

    return ids


# takes individual component prices computes the IQR and removes outliers 3x greater than the iqr
def drop_product_outliers(df):

    def flag_outliers(group):

        if len(group) < 2:
            return pd.Series(False, index=group.index)

        if len(group) == 2:

            lo, hi = group["price"].min(), group["price"].max()

            if lo > 0 and hi / lo > 5:
                return group["price"] == hi

            return pd.Series(False, index=group.index)

        median = group["price"].median()
        q1 = group["price"].quantile(0.25)
        q3 = group["price"].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return pd.Series(False, index=group.index)

        outlier = (group["price"] - median).abs() > 3 * iqr

        if outlier.all():
            return pd.Series(False, index=group.index)

        return outlier

    outlier_mask = (
        df.groupby(["component_id", "year"], group_keys=False)
        .apply(flag_outliers)
    )

    n_dropped = outlier_mask.sum()
    print(f"Dropped {n_dropped} within-group price outliers")

    return df[~outlier_mask]


# computes the iqr for the entire category rather than a specific product used to rid of outliers within the category
def compute_price_caps_per_category(df):

    caps = {}

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        prices = df[df["category"] == cat]["price"].dropna()

        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1

        cap = q3 + 5 * iqr
        caps[cat] = cap

        print(f"  {cat}: Q1=${q1:.0f}  Q3=${q3:.0f}  IQR=${iqr:.0f}  cap=${cap:.0f}")

    return caps



# does the heavy lifting of the cleaning pipeline
# takes the raw merged csv from data_collection and turns it into the cleaned dataset

def main():

    df = pd.read_csv("cleaned_data/combined_parts.csv")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df["model"] = df["model"].map(clean_product_name)
    before = len(df)

    df = df[df["model"].notna()]
    print(f"Dropped {before - len(df)} rows with empty model name")

    before = len(df)
    df = df[df["price"].notna() & (df["price"] > 0)]
    print(f"Dropped {before - len(df)} rows with null/zero price")

    print("Computing price caps using IQR:")
    max_prices = compute_price_caps_per_category(df)

    before = len(df)

    for cat, cap in max_prices.items():
        df = df[~((df["category"] == cat) & (df["price"] > cap))]

    print(f"Dropped {before - len(df)} price-outlier rows")

    before = len(df)
    workstation_mask = (
        (df["category"] == "GPU")
        & df["chipset"].notna()
        & df["chipset"].str.contains(workstation_gpus, regex=True)
    )
    df = df[~workstation_mask]
    print(f"Dropped {before - len(df)} workstation GPU rows")

    for col in ["memory_gb", "tdp_watts", "core_clock_mhz", "ddr_gen", "speed_mhz",
                "module_count", "module_size_gb", "first_word_latency_ns",
                "capacity_gb", "storage_type", "interface_type", "pcie_gen"]:
        df[col] = pd.NA

    gpu_idx = df.index[df["category"] == "GPU"]
    cpu_idx = df.index[df["category"] == "CPU"]
    ram_idx = df.index[df["category"] == "RAM"]
    sto_idx = df.index[df["category"] == "Storage"]

    df.loc[gpu_idx, "memory_gb"] = df.loc[gpu_idx, "memory"].map(process_gpu_vram)
    df.loc[gpu_idx, "tdp_watts"] = df.loc[gpu_idx, "tdp"].map(process_watts)
    df.loc[gpu_idx, "core_clock_mhz"] = df.loc[gpu_idx, "core_clock"].map(process_gpu_clock_speed)
    df.loc[cpu_idx, "tdp_watts"] = df.loc[cpu_idx, "tdp"].map(process_watts)

    speed_parsed = df.loc[ram_idx, "speed"].map(process_ram_speed)
    modules_parsed = df.loc[ram_idx, "modules"].map(process_ram_stick)
    df.loc[ram_idx, "ddr_gen"] = [v[0] for v in speed_parsed]
    df.loc[ram_idx, "speed_mhz"] = [v[1] for v in speed_parsed]
    df.loc[ram_idx, "module_count"] = [v[0] for v in modules_parsed]
    df.loc[ram_idx, "module_size_gb"] = [v[1] for v in modules_parsed]
    df.loc[ram_idx, "first_word_latency_ns"] = df.loc[ram_idx, "first_word_latency"].map(process_ram_latency)

    df.loc[sto_idx, "capacity_gb"] = df.loc[sto_idx, "capacity"].map(process_storage_capacity)
    df.loc[sto_idx, "storage_type"] = df.loc[sto_idx, "type"].map(normalize_storage_type)

    filled = df.loc[sto_idx, "storage_type"].notna().sum()
    print(f"storage_type coverage: {filled} / {len(sto_idx)}")

    if "interface" in df.columns:
        parsed_inter = df.loc[sto_idx, "interface"].map(process_storage_interface)
        df.loc[sto_idx, "interface_type"] = [v[0] for v in parsed_inter]
        df.loc[sto_idx, "pcie_gen"] = [v[1] for v in parsed_inter]

    sata_fallback = (
        (df["category"] == "Storage")
        & (df["year"] == 2023)
        & df["interface_type"].isna()
        & df["form_factor"].notna()
        & df["form_factor"].str.contains(r"2\.5|3\.5", regex=True, na=False)
    )
    df.loc[sata_fallback, "interface_type"] = "SATA"
    print(f"Inferred SATA interface_type for {sata_fallback.sum()} 2023 storage rows")

    brand_map = build_brand_map(df)
    missing_mask = df["brand"].isna()
    df.loc[missing_mask, "brand"] = infer_brand_column(df.loc[missing_mask], brand_map)
    inferred = missing_mask.sum() - df["brand"].isna().sum()
    still_null = df["brand"].isna().sum()
    print(f"Brand inferred for {inferred} rows; {still_null} still unknown")

    df = df.drop(columns=["raw_first_word"])

    df["brand"] = df["brand"].str.lower()

    df = df.reset_index(drop=True)
    df["component_id"] = build_component_id(df)
    print(f"Unique component_ids: {df['component_id'].nunique()}")

    tdp_from_23 = (
        df[(df["category"] == "GPU") & (df["year"] == 2023) & df["tdp_watts"].notna()]
        .groupby("component_id")["tdp_watts"]
        .first()
    )
    gpu_25_null = (df["category"] == "GPU") & (df["year"] == 2025) & df["tdp_watts"].isna()
    df.loc[gpu_25_null, "tdp_watts"] = df.loc[gpu_25_null, "component_id"].map(tdp_from_23)
    filled_tdp = int(gpu_25_null.sum() - ((df["category"] == "GPU") & (df["year"] == 2025) & df["tdp_watts"].isna()).sum())
    print(f"Cross-filled GPU TDP for {filled_tdp} 2025 rows from matched 2023 data")

    df = drop_product_outliers(df)
    df = df.reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["component_id", "year", "price"])
    print(f"Dropped {before - len(df)} exact duplicate rows")

    agg = (
        df.groupby(["component_id", "category", "year"], dropna=False)["price"]
        .median()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["component_id", "category"],
        columns="year",
        values="price",
        aggfunc="first",
    )
    if 2023 not in pivot.columns or 2025 not in pivot.columns:
        pivot = pivot.reindex(columns=[2023, 2025])

    p23, p25 = pivot[2023], pivot[2025]
    both = p23.notna() & p25.notna()
    change = p25 - p23
    pct = (change / p23 * 100).where(p23 != 0)

    pair_stats = pd.DataFrame({"price_change": change, "price_change_pct": pct})
    pair_stats = pair_stats[both].reset_index()

    bad = pair_stats["price_change_pct"].abs() > 200
    print(f"Removed {bad.sum()} bad matches over 200% change")
    pair_stats = pair_stats[~bad]

    df = df.drop(columns=["price_change", "price_change_pct"], errors="ignore")
    df = df.merge(pair_stats, on=["component_id", "category"], how="left")

    matched_ids = df[df["price_change"].notna()]["component_id"].nunique()
    unmatched_rows = df["price_change"].isna().sum()

    print(f"\nFinal shape: {df.shape}")
    print(f"Matched component_ids: {matched_ids}")
    print(f"Unmatched rows: {unmatched_rows}")

    print("\nCategory breakdown:")
    print(df.groupby(["category", "year"]).size().unstack(fill_value=0).to_string())

    print(f"\nBrand null count: {df['brand'].isna().sum()}")

    df.to_csv("cleaned_data/combined_parts_cleaned.csv", index=False)
    print("\nWrote cleaned_data/combined_parts_cleaned.csv")
