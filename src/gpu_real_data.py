import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import numpy as np


REPO_API = "https://api.github.com/repos/HardwareDealsCo/gpu-deals/contents/historical-gpu-deals-data"
RAW_BASE = "https://raw.githubusercontent.com/HardwareDealsCo/gpu-deals/main/historical-gpu-deals-data"
LOCAL_DIR = "raw_data/gpu_deals_daily"
OUT_PATH = "cleaned_data/gpu_chipset_real_prices.csv"
CONDITION_FILTER = "New"
MIN_SAMPLE = 3


_TOKEN_FIX = {
    "rtx": "RTX",
    "gtx": "GTX",
    "rx": "RX",
    "arc": "Arc",
    "ti": "Ti",
    "xt": "XT",
    "xtx": "XTX",
    "super": "Super",
    "geforce": "",
    "radeon": "",
    "nvidia": "",
    "amd": "",
    "lhr": "",
    "ultra": "",
    "founders": "",
    "edition": "",
}


# cleans up the chipset name from the hardwaredealsco listings
# strips prefixes like geforce/radeon, fixes casing, corrects mislabels (gtx 5070 -> rtx 5070)
# this is the normalizer that lets the project chipsets join the external chipsets

def normalize_chipset(name):

    if name is None:
        return None

    s = str(name).strip()

    if not s:
        return None

    tokens = []

    for raw in s.split():

        low = raw.lower().strip(",.():")

        if not low:
            continue

        if low in _TOKEN_FIX:
            mapped = _TOKEN_FIX[low]

            if mapped:
                tokens.append(mapped)

            continue

        tokens.append(raw.upper() if raw.isupper() else raw)

    out = " ".join(tokens).strip()

    if out.startswith("GTX ") and len(out) > 4 and out[4].isdigit():

        try:
            num = int("".join(c for c in out.split()[1] if c.isdigit())[:4])

            if num >= 2000:
                out = "RTX" + out[3:]

        except ValueError:
            pass

    return " ".join(out.split())


# asks github for the list of daily json files in the gpu-deals repo

def list_remote_files():

    req = urllib.request.Request(REPO_API, headers={"User-Agent": "cs210-pipeline"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))

    return [x["name"] for x in items if x["name"].startswith("GPU-") and x["name"].endswith(".json")]


# downloads one daily json into the local cache folder

def download_file(name):

    url = f"{RAW_BASE}/{name}"
    target = os.path.join(LOCAL_DIR, name)
    req = urllib.request.Request(url, headers={"User-Agent": "cs210-pipeline"})

    with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as f:
        f.write(resp.read())

    return target


# incremental sync: only downloads daily files that aren't already cached
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
        print(f"  downloading {len(missing)} new daily file(s) from HardwareDealsCo")

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


# reads every daily json and turns each listing into a row
# filters to "New" condition only since used market is too noisy

def load_daily_listings(files):

    rows = []

    for name in files:

        path = os.path.join(LOCAL_DIR, name)

        if not os.path.exists(path):
            continue

        date_str = name.replace("GPU-", "").replace(".json", "")

        try:

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        for d in data:

            chipset = normalize_chipset(d.get("chipset"))
            condition = d.get("condition")
            price_str = d.get("price")

            if chipset is None or condition is None or price_str is None:
                continue

            if condition != CONDITION_FILTER:
                continue

            try:
                price = float(price_str)

            except (TypeError, ValueError):
                continue

            if price <= 0:
                continue

            rows.append({
                "date": date_str,
                "chipset": chipset,
                "price": price,
                "brand": d.get("brand"),
                "vram_gb": d.get("vram_gb"),
            })

    return pd.DataFrame(rows)


# groups listings by chipset and month, takes the median
# 1.5x iqr filter drops outliers, requires 3+ listings per month so one weird price doesn't dominate

def aggregate_to_monthly(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    grouped = df.groupby(["chipset", "month"])

    out_rows = []

    for (chipset, month), g in grouped:

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
            "chipset": chipset,
            "month": month.strftime("%Y-%m-%d"),
            "real_median_price": round(float(np.median(clean)), 2),
            "real_mean_price": round(float(np.mean(clean)), 2),
            "sample_size": int(len(clean)),
            "raw_sample_size": int(len(prices)),
        })

    return pd.DataFrame(out_rows).sort_values(["chipset", "month"]).reset_index(drop=True)


# syncs new daily json files, loads them, aggregates to monthly, writes the chipset-month csv

def main():

    files = sync_local_cache()

    if not files:
        print("  no daily GPU files available; skipping real GPU data ingest")
        return

    df = load_daily_listings(files)
    print(f"  loaded {len(df):,} new-condition GPU listings from {df['date'].nunique()} days")

    monthly = aggregate_to_monthly(df)

    os.makedirs("cleaned_data", exist_ok=True)
    monthly.to_csv(OUT_PATH, index=False)

    print(f"  Saved: {OUT_PATH}  ({len(monthly):,} chipset-month rows, {monthly['chipset'].nunique()} chipsets, {monthly['month'].nunique()} months)")


if __name__ == "__main__":
    main()
