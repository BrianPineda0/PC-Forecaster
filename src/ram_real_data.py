import os
import re
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import numpy as np


REPO_API = "https://api.github.com/repos/HardwareDealsCo/ram-deals/contents/historical-ram-deals-data"
RAW_BASE = "https://raw.githubusercontent.com/HardwareDealsCo/ram-deals/main/historical-ram-deals-data"
LOCAL_DIR = "raw_data/ram_deals_daily"
OUT_PATH = "cleaned_data/ram_real_prices.csv"
CONDITION_FILTER = "New"
FORM_FACTOR_FILTER = "Desktop"
MIN_SAMPLE = 3


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.json$")
_SPEED_RE = re.compile(r"DDR(\d+)[- ]?(\d+)?", re.IGNORECASE)


# pulls YYYY-MM-DD out of a filename like RAM-2026-05-07.json or 2025-09-18.json

def extract_date(filename):

    m = _DATE_RE.search(filename)
    return m.group(1) if m else None


# pulls ddr generation and speed mhz out of a string like "DDR5-6400"

def parse_speed(speed_str):

    if not isinstance(speed_str, str):
        return None, None

    m = _SPEED_RE.search(speed_str)

    if not m or m.group(2) is None:
        return None, None

    return int(m.group(1)), int(m.group(2))


# builds the bucket key like "DDR5 6400 32GB" for desktop ram only
# this is the join key the bridge uses to match project ram to real prices

def normalize_ram_bucket(speed_str, capacity_gb, form_factor):

    if form_factor != FORM_FACTOR_FILTER:
        return None, None, None, None

    ddr_gen, speed_mhz = parse_speed(speed_str)

    if ddr_gen is None or speed_mhz is None:
        return None, None, None, None

    if capacity_gb is None:
        return None, None, None, None

    try:
        cap = int(round(float(capacity_gb)))

    except (TypeError, ValueError):
        return None, None, None, None

    if cap <= 0:
        return None, None, None, None

    return f"DDR{ddr_gen} {speed_mhz} {cap}GB", ddr_gen, speed_mhz, cap


# asks github for the list of daily json files in the ram-deals repo

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
        print(f"  downloading {len(missing)} new daily file(s) from HardwareDealsCo ram-deals")

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


# reads each daily ram json into rows
# filters to "New" condition + "Desktop" form factor only

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

            bucket, ddr_gen, speed_mhz, cap = normalize_ram_bucket(d.get("speed"), d.get("capacity_gb"), d.get("form_factor"))

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
                "ram_bucket": bucket,
                "ddr_gen": ddr_gen,
                "speed_mhz": speed_mhz,
                "capacity_gb": cap,
                "price": price,
                "brand": d.get("brand"),
            })

    return pd.DataFrame(rows)


# groups listings by ram_bucket and month, takes the median
# 1.5x iqr filter and 3-listings minimum per month

def aggregate_to_monthly(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    grouped = df.groupby(["ram_bucket", "ddr_gen", "speed_mhz", "capacity_gb", "month"])

    out_rows = []

    for (bucket, gen, speed, cap, month), g in grouped:

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
            "ram_bucket": bucket,
            "ddr_gen": gen,
            "speed_mhz": speed,
            "capacity_gb": cap,
            "month": month.strftime("%Y-%m-%d"),
            "real_median_price": round(float(np.median(clean)), 2),
            "real_mean_price": round(float(np.mean(clean)), 2),
            "sample_size": int(len(clean)),
            "raw_sample_size": int(len(prices)),
        })

    return pd.DataFrame(out_rows).sort_values(["ram_bucket", "month"]).reset_index(drop=True)


# syncs new daily ram json files, loads them, aggregates to monthly, writes the bucket-month csv

def main():

    files = sync_local_cache()

    if not files:
        print("  no daily RAM files available; skipping real RAM data ingest")
        return

    df = load_daily_listings(files)
    print(f"  loaded {len(df):,} new-condition desktop RAM listings from {df['date'].nunique()} days")

    monthly = aggregate_to_monthly(df)

    os.makedirs("cleaned_data", exist_ok=True)
    monthly.to_csv(OUT_PATH, index=False)

    print(f"  Saved: {OUT_PATH}  ({len(monthly):,} bucket-month rows, {monthly['ram_bucket'].nunique()} buckets, {monthly['month'].nunique()} months)")


if __name__ == "__main__":
    main()
