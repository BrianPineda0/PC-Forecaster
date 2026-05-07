import os
import re
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import numpy as np


REPO_API = "https://api.github.com/repos/HardwareDealsCo/drive-deals/contents/historical-drive-deals-data"
RAW_BASE = "https://raw.githubusercontent.com/HardwareDealsCo/drive-deals/main/historical-drive-deals-data"
LOCAL_DIR = "raw_data/drive_deals_daily"
OUT_PATH = "cleaned_data/drive_real_prices.csv"
CONDITION_FILTER = "New"
MIN_SAMPLE = 3


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.json$")


_CAPACITY_TIERS = [128, 256, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]


# pulls YYYY-MM-DD out of a filename like DRIVE-2026-05-07.json or 2025-09-18.json
# the repo uses both formats so the regex handles both

def extract_date(filename):

    m = _DATE_RE.search(filename)
    return m.group(1) if m else None


# snaps a drive capacity to the closest standard tier within 10%
# this fixes the binary vs decimal mismatch (1024gb and 1000gb both map to 1000gb)
# match rate jumped from 65% to 91% after adding this

def capacity_tier(capacity_gb):

    if capacity_gb is None:
        return None

    try:
        cap = float(capacity_gb)

    except (TypeError, ValueError):
        return None

    if cap <= 0:
        return None

    best = None
    best_diff = None

    for t in _CAPACITY_TIERS:

        diff = abs(cap - t) / t

        if diff <= 0.10 and (best_diff is None or diff < best_diff):
            best = t
            best_diff = diff

    if best is not None:
        return best

    return int(round(cap))


# builds the bucket key like "SSD NVMe 1000GB"
# this is the join key the bridge uses to match project storage components to real prices

def normalize_drive_bucket(drive_type, interface_type, capacity_gb):

    if drive_type is None or interface_type is None or capacity_gb is None:
        return None

    cap = capacity_tier(capacity_gb)

    if cap is None or cap <= 0:
        return None

    return f"{drive_type} {interface_type} {cap}GB"


# asks github for the list of daily json files in the drive-deals repo

def list_remote_files():

    req = urllib.request.Request(REPO_API + "?per_page=1000", headers={"User-Agent": "cs210-pipeline"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))

    return [x["name"] for x in items if x["name"].endswith(".json")]


# downloads one daily json into the local cache folder

def download_file(name):

    url = f"{RAW_BASE}/{name}"
    target = os.path.join(LOCAL_DIR, name)
    req = urllib.request.Request(url, headers={"User-Agent": "cs210-pipeline"})

    with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as f:
        f.write(resp.read())

    return target


# incremental sync: only downloads daily files not already cached
# falls back to local-only if github is unreachable

def sync_local_cache():

    os.makedirs(LOCAL_DIR, exist_ok=True)

    try:
        remote = list_remote_files()

    except urllib.error.URLError as e:
        print(f"  warning: cannot reach GitHub ({e}); using local cache only")
        return sorted(f for f in os.listdir(LOCAL_DIR) if f.endswith(".json"))

    local = set(os.listdir(LOCAL_DIR))
    missing = [n for n in remote if n not in local]

    if missing:
        print(f"  downloading {len(missing)} new daily file(s) from HardwareDealsCo drive-deals")

        for i, name in enumerate(missing, 1):

            try:
                download_file(name)

            except urllib.error.URLError as e:
                print(f"    failed {name}: {e}")
                continue

            if i % 25 == 0:
                print(f"    {i}/{len(missing)}...")

            time.sleep(0.05)

    return sorted(remote)


# reads each daily drive json into rows
# filters to "New" condition only and dedupes if a date has multiple file variants

def load_daily_listings(files):

    rows = []
    seen_dates = set()

    for name in files:

        path = os.path.join(LOCAL_DIR, name)

        if not os.path.exists(path):
            continue

        date_str = extract_date(name)

        if date_str is None:
            continue

        if date_str in seen_dates:
            continue

        seen_dates.add(date_str)

        try:

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        for d in data:

            condition = d.get("condition")
            price_str = d.get("price")

            if condition != CONDITION_FILTER or price_str is None:
                continue

            bucket = normalize_drive_bucket(d.get("drive_type"), d.get("interface_type"), d.get("capacity_gb"))

            if bucket is None:
                continue

            try:
                price = float(price_str)

            except (TypeError, ValueError):
                continue

            if price <= 0:
                continue

            rows.append({
                "date": date_str,
                "drive_bucket": bucket,
                "drive_type": d.get("drive_type"),
                "interface_type": d.get("interface_type"),
                "capacity_gb": capacity_tier(d.get("capacity_gb")),
                "price": price,
                "brand": d.get("brand"),
            })

    return pd.DataFrame(rows)


# groups listings by drive_bucket and month, takes the median
# 1.5x iqr filter and 3-listings minimum per month

def aggregate_to_monthly(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    grouped = df.groupby(["drive_bucket", "drive_type", "interface_type", "capacity_gb", "month"])

    out_rows = []

    for (bucket, dtype, iface, cap, month), g in grouped:

        prices = g["price"].values

        if len(prices) < MIN_SAMPLE:
            continue

        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = prices[(prices >= low) & (prices <= high)]

        if len(clean) < MIN_SAMPLE:
            clean = prices

        out_rows.append({
            "drive_bucket": bucket,
            "drive_type": dtype,
            "interface_type": iface,
            "capacity_gb": cap,
            "month": month.strftime("%Y-%m-%d"),
            "real_median_price": round(float(np.median(clean)), 2),
            "real_mean_price": round(float(np.mean(clean)), 2),
            "sample_size": int(len(clean)),
            "raw_sample_size": int(len(prices)),
        })

    return pd.DataFrame(out_rows).sort_values(["drive_bucket", "month"]).reset_index(drop=True)


# syncs new daily drive json files, loads them, aggregates to monthly, writes the bucket-month csv

def main():

    files = sync_local_cache()

    if not files:
        print("  no daily drive files available; skipping real drive data ingest")
        return

    df = load_daily_listings(files)
    print(f"  loaded {len(df):,} new-condition drive listings from {df['date'].nunique()} days")

    monthly = aggregate_to_monthly(df)

    os.makedirs("cleaned_data", exist_ok=True)
    monthly.to_csv(OUT_PATH, index=False)

    print(f"  Saved: {OUT_PATH}  ({len(monthly):,} bucket-month rows, {monthly['drive_bucket'].nunique()} buckets, {monthly['month'].nunique()} months)")


if __name__ == "__main__":
    main()
