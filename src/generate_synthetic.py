import os
import numpy as np
import pandas as pd

import gpu_real_data
import drive_real_data


seed = 42

dates = pd.date_range("2023-01", periods=29, freq="MS")
real_gpu_path = "cleaned_data/gpu_chipset_real_prices.csv"
real_drive_path = "cleaned_data/drive_real_prices.csv"
real_ram_path = "cleaned_data/ram_real_prices.csv"

events = [
    ("post_crypto_deflation", "2023-01", "2023-06", {"GPU": -6}),
    ("ai_demand_wave", "2023-03", "2023-12", {"GPU": +8, "RAM": +4}),
    ("nand_oversupply", "2023-01", "2023-08", {"Storage": -8}),
    ("ram_shortage", "2023-10", "2024-04", {"RAM": +12}),
    ("black_friday_2023", "2023-11", "2023-11", {"GPU": -7, "CPU": -7, "RAM": -7, "Storage": -7}),
    ("ai_investment_surge", "2024-01", "2024-07", {"GPU": +10, "RAM": +5}),
    ("new_cpu_gen_releases", "2024-08", "2024-12", {"CPU": -5}),
    ("black_friday_2024", "2024-11", "2024-11", {"GPU": -6, "CPU": -6, "RAM": -6, "Storage": -6}),
    ("rtx5000_launch_pressure", "2025-01", "2025-05", {"GPU": -9}),
]

seasonal = {
    1: 1.02, 2: 1.00, 3: 1.00, 4: 0.99, 5: 0.99, 6: 0.98,
    7: 0.97, 8: 0.97, 9: 0.99, 10: 1.00, 11: 1.01, 12: 1.02,
}


# takes the cleaned parts dataframe and builds one row per component
# with the 2023 and 2025 anchor prices used as endpoints for the synthetic series
# also keeps the spec columns needed later for matching to real bucket prices

def build_anchors(df):

    spec_cols = [
        "model", "category", "brand",
        "storage_type", "interface_type", "capacity_gb",
        "ddr_gen", "speed_mhz", "module_count", "module_size_gb",
    ]
    available = [c for c in spec_cols if c in df.columns]

    meta = df.groupby("component_id")[available].first().reset_index()

    agg = df.groupby(["component_id", "year"])["price"].median().reset_index()

    pivot = agg.pivot(index="component_id", columns="year", values="price").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={2023: "price_2023", 2025: "price_2025"})

    pivot = pivot.dropna(subset=["price_2023", "price_2025"])

    return pivot.merge(meta, on="component_id", how="left")


# turns the events list into a per-month multiplier per category
# each event window adds or subtracts a percent to the category trend

def build_shock_table():

    categories = ["GPU", "CPU", "RAM", "Storage"]
    effects = {cat: pd.Series(0.0, index=dates) for cat in categories}

    for _label, start, end, cat_effects in events:

        mask = (dates >= start) & (dates <= end)

        for cat, pct in cat_effects.items():
            effects[cat] = effects[cat] + mask.astype(float) * pct

    return {cat: 1.0 + effects[cat] / 100.0 for cat in categories}


# tags each month with which named events were active
# this column ends up in the price_history csv so i can see why a month spiked

def build_event_labels():

    label_map = {d: [] for d in dates}

    for label, start, end, _ in events:

        for d in dates[(dates >= start) & (dates <= end)]:
            label_map[d].append(label)

    return pd.Series({d: ",".join(v) for d, v in label_map.items()})


# the actual synthetic price formula
# trend between 2023 and 2025 anchors times event shock times seasonality times noise
# returns 29 monthly prices for one component

def generate_series(price_2023, price_2025, category, shock_tables, event_labels, rng):

    n = len(dates)
    t = np.linspace(0.0, 1.0, n)

    trend = price_2023 + t * (price_2025 - price_2023)
    shock = shock_tables[category].values
    season = np.array([seasonal[d.month] for d in dates])
    noise = rng.normal(0.0, 0.04, n)

    prices = trend * shock * season * (1.0 + noise)
    prices = np.clip(prices, 1.0, None)

    return np.round(prices, 2), event_labels.values


# quick category-level sanity print so i can see the synthetic trends moved the right direction

def print_summary(history):

    print("\nAverage price change by category (Jan 2023 to May 2025):")

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        sub = history[(history["category"] == cat) & (history["data_source"] == "synthetic")]
        p_start = sub[sub["date"] == dates[0]]["price"].mean()
        p_end = sub[sub["date"] == dates[-1]]["price"].mean()
        pct = (p_end - p_start) / p_start * 100

        print(f"  {cat}: ${p_start:.2f} to ${p_end:.2f} ({pct:.1f}%)")


# pulls the real gpu chipset prices csv if it exists
# returns none if the real-data ingest has not been run yet

def load_real_gpu_lookup():

    if not os.path.exists(real_gpu_path):
        print(f"  no real GPU prices found at {real_gpu_path}; GPU pipeline falls back to synthetic only")
        return None

    df = pd.read_csv(real_gpu_path)
    df["month"] = pd.to_datetime(df["month"])

    return df


# pulls the chipset name out of a gpu component_id like "asus tuf | RTX 4070"
# then runs it through the same normalizer used on the real data so they match

def chipset_from_component_id(cid):

    if not isinstance(cid, str) or "|" not in cid:
        return None

    raw = cid.split("|", 1)[1].strip()
    return gpu_real_data.normalize_chipset(raw)


# the gpu side of the real-data bridge
# for each project gpu component matched to a real chipset
# scales the chipset's monthly real price by an aib_premium ratio
# so a $700 asus rog stays priced higher than a $500 generic of the same chipset

def build_real_gpu_rows(anchors, real_df):

    if real_df is None or real_df.empty:
        return []

    real_by_chipset = {
        chip: g.set_index("month")["real_median_price"].to_dict()
        for chip, g in real_df.groupby("chipset")
    }

    real_months = sorted(real_df["month"].unique())
    anchor_month = pd.Timestamp(real_months[0])

    rows = []
    matched_count = 0
    unmatched_chipsets = set()

    gpu_anchors = anchors[anchors["category"] == "GPU"].copy()
    gpu_anchors["chipset_norm"] = gpu_anchors["component_id"].map(chipset_from_component_id)

    for _, row in gpu_anchors.iterrows():

        chip = row["chipset_norm"]

        if chip is None or chip not in real_by_chipset:

            if chip is not None:
                unmatched_chipsets.add(chip)

            continue

        chip_prices = real_by_chipset[chip]

        if anchor_month not in chip_prices:
            continue

        anchor_real = chip_prices[anchor_month]

        if anchor_real <= 0:
            continue

        aib_premium = float(row["price_2025"]) / float(anchor_real)

        for m in real_months:

            if m not in chip_prices:
                continue

            blended = chip_prices[m] * aib_premium

            rows.append({
                "component_id": row["component_id"],
                "model": row["model"],
                "category": "GPU",
                "brand": row["brand"],
                "date": pd.Timestamp(m),
                "price": round(float(blended), 2),
                "event_label": "",
                "data_source": "real_blended",
            })

        matched_count += 1

    print(f"  GPU real-data matching: {matched_count}/{len(gpu_anchors)} components matched a real chipset series")

    if unmatched_chipsets:
        print(f"  unmatched chipsets ({len(unmatched_chipsets)}): {sorted(list(unmatched_chipsets))[:8]}{'...' if len(unmatched_chipsets) > 8 else ''}")

    return rows


# loads the drive bucket-month real prices csv if it exists

def load_real_drive_lookup():

    if not os.path.exists(real_drive_path):
        print(f"  no real drive prices found at {real_drive_path}; Storage pipeline falls back to synthetic only")
        return None

    df = pd.read_csv(real_drive_path)
    df["month"] = pd.to_datetime(df["month"])

    return df


# loads the ram bucket-month real prices csv if it exists

def load_real_ram_lookup():

    if not os.path.exists(real_ram_path):
        print(f"  no real RAM prices found at {real_ram_path}; RAM pipeline falls back to synthetic only")
        return None

    df = pd.read_csv(real_ram_path)
    df["month"] = pd.to_datetime(df["month"])

    return df


# builds the drive bucket key for a project storage component
# maps "M.2 PCIe" to "NVMe" so the project side matches the hardwaredealsco side
# returns none if any required field is missing

def drive_bucket_from_row(row):

    stype = row.get("storage_type")
    iface = row.get("interface_type")
    cap = row.get("capacity_gb")

    if pd.isna(stype) or pd.isna(iface) or pd.isna(cap):
        return None

    if iface == "M.2 PCIe":
        real_iface = "NVMe"

    elif iface == "SATA":
        real_iface = "SATA"

    else:
        return None

    cap_t = drive_real_data.capacity_tier(cap)

    if cap_t is None or cap_t <= 0:
        return None

    return f"{stype} {real_iface} {cap_t}GB"


# builds the ram bucket key for a project ram component
# kit total = module_count times module_size_gb so it lines up with hardwaredealsco totals

def ram_bucket_from_row(row):

    ddr = row.get("ddr_gen")
    spd = row.get("speed_mhz")
    mc = row.get("module_count")
    ms = row.get("module_size_gb")

    if pd.isna(ddr) or pd.isna(spd) or pd.isna(mc) or pd.isna(ms):
        return None

    if ddr <= 0 or spd <= 0 or mc <= 0 or ms <= 0:
        return None

    kit = int(round(float(mc) * float(ms)))

    if kit <= 0:
        return None

    return f"DDR{int(ddr)} {int(spd)} {kit}GB"


# the drive side of the real-data bridge
# matches project ssds and hdds to their bucket and scales by brand_premium
# same idea as the gpu version but using drive_bucket as the join key

def build_real_drive_rows(anchors, real_df):

    if real_df is None or real_df.empty:
        return []

    real_by_bucket = {
        b: g.set_index("month")["real_median_price"].to_dict()
        for b, g in real_df.groupby("drive_bucket")
    }

    real_months = sorted(real_df["month"].unique())
    anchor_month = pd.Timestamp(real_months[0])

    rows = []
    matched_count = 0

    storage_anchors = anchors[anchors["category"] == "Storage"].copy()
    storage_anchors["bucket"] = storage_anchors.apply(drive_bucket_from_row, axis=1)

    for _, row in storage_anchors.iterrows():

        bucket = row["bucket"]

        if bucket is None or bucket not in real_by_bucket:
            continue

        bucket_prices = real_by_bucket[bucket]

        if anchor_month not in bucket_prices:
            continue

        anchor_real = bucket_prices[anchor_month]

        if anchor_real <= 0:
            continue

        brand_premium = float(row["price_2025"]) / float(anchor_real)

        for m in real_months:

            if m not in bucket_prices:
                continue

            blended = bucket_prices[m] * brand_premium

            rows.append({
                "component_id": row["component_id"],
                "model": row["model"],
                "category": "Storage",
                "brand": row["brand"],
                "date": pd.Timestamp(m),
                "price": round(float(blended), 2),
                "event_label": "",
                "data_source": "real_blended",
            })

        matched_count += 1

    print(f"  Storage real-data matching: {matched_count}/{len(storage_anchors)} components matched a real drive bucket")

    return rows


# the ram side of the real-data bridge
# uses ram_bucket as the join key and applies the same brand_premium scaling

def build_real_ram_rows(anchors, real_df):

    if real_df is None or real_df.empty:
        return []

    real_by_bucket = {
        b: g.set_index("month")["real_median_price"].to_dict()
        for b, g in real_df.groupby("ram_bucket")
    }

    real_months = sorted(real_df["month"].unique())
    anchor_month = pd.Timestamp(real_months[0])

    rows = []
    matched_count = 0

    ram_anchors = anchors[anchors["category"] == "RAM"].copy()
    ram_anchors["bucket"] = ram_anchors.apply(ram_bucket_from_row, axis=1)

    for _, row in ram_anchors.iterrows():

        bucket = row["bucket"]

        if bucket is None or bucket not in real_by_bucket:
            continue

        bucket_prices = real_by_bucket[bucket]

        if anchor_month not in bucket_prices:
            continue

        anchor_real = bucket_prices[anchor_month]

        if anchor_real <= 0:
            continue

        brand_premium = float(row["price_2025"]) / float(anchor_real)

        for m in real_months:

            if m not in bucket_prices:
                continue

            blended = bucket_prices[m] * brand_premium

            rows.append({
                "component_id": row["component_id"],
                "model": row["model"],
                "category": "RAM",
                "brand": row["brand"],
                "date": pd.Timestamp(m),
                "price": round(float(blended), 2),
                "event_label": "",
                "data_source": "real_blended",
            })

        matched_count += 1

    print(f"  RAM real-data matching: {matched_count}/{len(ram_anchors)} components matched a real RAM bucket")

    return rows


# builds the full price_history csv
# generates synthetic monthly rows for every matched component
# then appends real_blended rows for gpus storage and ram via the bridge functions

def main():

    rng = np.random.default_rng(seed)

    df = pd.read_csv("cleaned_data/combined_parts_cleaned.csv")

    anchors = build_anchors(df)
    counts = anchors["category"].value_counts().to_dict()
    print(f"Matched components: {len(anchors):,}  {counts}")

    shock_tables = build_shock_table()
    event_labels = build_event_labels()

    records = []

    for _, row in anchors.iterrows():

        prices, labels = generate_series(
            row["price_2023"],
            row["price_2025"],
            row["category"],
            shock_tables,
            event_labels,
            rng,
        )

        chunk = pd.DataFrame({
            "component_id": row["component_id"],
            "model": row["model"],
            "category": row["category"],
            "brand": row["brand"],
            "date": dates,
            "price": prices,
            "event_label": labels,
            "data_source": "synthetic",
        })

        records.append(chunk)

    real_df = load_real_gpu_lookup()
    real_rows = build_real_gpu_rows(anchors, real_df)

    if real_rows:
        records.append(pd.DataFrame(real_rows))

    real_drive_df = load_real_drive_lookup()
    real_drive_rows = build_real_drive_rows(anchors, real_drive_df)

    if real_drive_rows:
        records.append(pd.DataFrame(real_drive_rows))

    real_ram_df = load_real_ram_lookup()
    real_ram_rows = build_real_ram_rows(anchors, real_ram_df)

    if real_ram_rows:
        records.append(pd.DataFrame(real_ram_rows))

    history = pd.concat(records, ignore_index=True)
    history = history[["component_id", "model", "category", "brand", "date", "price", "event_label", "data_source"]]

    history.to_csv("cleaned_data/price_history.csv", index=False)

    n_synth = (history["data_source"] == "synthetic").sum()
    n_real = (history["data_source"] == "real_blended").sum()
    real_by_cat = history[history["data_source"] == "real_blended"]["category"].value_counts().to_dict()
    print(f"Saved: cleaned_data/price_history.csv  ({len(history):,} rows: {n_synth:,} synthetic + {n_real:,} real-blended {real_by_cat})")

    print_summary(history)
