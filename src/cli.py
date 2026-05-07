import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


db_path = "pcparts.db"
forecast_path = "cleaned_data/future_price_predictions.csv"

spec_features = {
    "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
    "CPU": ["core_count", "tdp_watts"],
    "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
    "Storage": ["capacity_gb"],
}


# strips non-ascii characters so component names with weird symbols print cleanly in the terminal

def safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")


# creates the user_observations and prediction_adjustments tables on first run
# these store the adaptive correction state so it survives between sessions

def ensure_user_tables():

    conn = sqlite3.connect(db_path)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            component_id TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            observed_price REAL NOT NULL,
            predicted_price REAL,
            ratio REAL,
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_adjustments (
            component_id TEXT PRIMARY KEY,
            multiplier REAL NOT NULL,
            n_observations INTEGER NOT NULL,
            last_updated TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()


# fuzzy search the components table by component_id substring
# require_forecast=True filters to components that have a 12-month forecast row

def find_components(query, require_forecast=False):

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT c.component_id, c.category, c.brand, c.model
        FROM components c
        WHERE LOWER(c.component_id) LIKE ?
        ORDER BY c.category, c.model
    """, conn, params=[f"%{query.lower()}%"])

    conn.close()

    if require_forecast and not df.empty:

        try:
            forecast_ids = set(pd.read_csv(forecast_path)["component_id"].unique())
            df = df[df["component_id"].isin(forecast_ids)].reset_index(drop=True)

        except FileNotFoundError:
            pass

    return df


# pulls the monthly price history for one component from sqlite

def get_synth(component_id):

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        "SELECT date, synthetic_price FROM synthetic_prices WHERE component_id = ? ORDER BY date",
        conn, params=[component_id]
    )

    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    return df


# pulls the 12-month forecast for one component from the forecast csv

def get_forecast(component_id):

    df = pd.read_csv(forecast_path)
    df = df[df["component_id"] == component_id].copy()

    if df.empty:
        return df

    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    df = df.rename(columns={"forecast_month": "date", "predicted_price": "price"})

    return df[["date", "price"]].sort_values("date").reset_index(drop=True)


# gets the user's stored calibration multiplier for one component
# returns 1.0 if no observations yet so the prediction is unchanged

def get_adjustment(component_id):

    conn = sqlite3.connect(db_path)

    row = conn.execute(
        "SELECT multiplier, n_observations FROM prediction_adjustments WHERE component_id = ?",
        [component_id]
    ).fetchone()

    conn.close()

    if row is None:
        return 1.0, 0

    return float(row[0]), int(row[1])


# shows up to 15 search matches and lets the user pick one by number

def pick_component(matches):

    print(f"\nFound {len(matches)} matching components.")

    n_show = min(15, len(matches))

    for i in range(n_show):
        row = matches.iloc[i]
        print(f"  {i}: [{row['category']}] {safe(row['model'])[:60]}")
        print(f"      id: {safe(row['component_id'])[:90]}")

    if len(matches) > 15:
        print(f"  ... and {len(matches) - 15} more (narrow your query for the rest)")

    pick = input("\nPick a number (or blank to cancel): ").strip()

    if not pick:
        return None

    try:
        i = int(pick)

        if i < 0 or i >= len(matches):
            print("out of range")
            return None

        return matches.iloc[i]["component_id"]

    except ValueError:
        print("invalid number")
        return None


# option 1 - prints high-level project stats (row counts + headline metrics)

def show_summary():

    conn = sqlite3.connect(db_path)

    n_components = conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    n_observed = conn.execute("SELECT COUNT(*) FROM observed_prices").fetchone()[0]
    n_synthetic = conn.execute("SELECT COUNT(*) FROM synthetic_prices").fetchone()[0]
    n_user_obs = conn.execute("SELECT COUNT(*) FROM user_observations").fetchone()[0]
    n_adj = conn.execute("SELECT COUNT(*) FROM prediction_adjustments").fetchone()[0]

    by_cat = pd.read_sql_query("""
        SELECT category, COUNT(*) AS n FROM components GROUP BY category ORDER BY category
    """, conn)

    conn.close()

    print()
    print("Project: Predicting the Optimal Time to Build a PC Using Hardware Price Trends")
    print(f"  Components stored:           {n_components:,}")
    print(f"  Observed (real) prices:      {n_observed:,}")
    print(f"  Synthetic monthly prices:    {n_synthetic:,}")
    print(f"  User observations recorded:  {n_user_obs}")
    print(f"  Active prediction calibrations: {n_adj}")

    print("\nComponents by category:")

    for _, row in by_cat.iterrows():
        print(f"  {row['category']}: {row['n']:,}")

    print("\nRegression metrics (single-split):")
    print(pd.read_csv("cleaned_data/model_metrics.csv").to_string(index=False))

    cls = pd.read_csv("cleaned_data/classification_metrics.csv")
    rf = cls[cls["model"] == "Random Forest"][["category", "accuracy", "macro_f1"]]
    print("\nClassification metrics (Random Forest spec-to-tier):")
    print(rf.to_string(index=False))


# option 2 - search a component, then run analyze_component on the pick

def lookup_component():

    q = input("\nComponent query (e.g. 'rtx 4070', 'ryzen 7 7700x'): ").strip()

    if not q:
        return

    matches = find_components(q)

    if matches.empty:
        print(f"No matches for '{q}'")
        return

    cid = pick_component(matches)

    if cid is None:
        return

    analyze_component(cid)


# pulls the 2023 + 2025 real observed prices for one component

def get_observed(component_id):

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT year, observed_price_median
        FROM observed_prices
        WHERE component_id = ?
        ORDER BY year
    """, conn, params=[component_id])

    conn.close()

    return df


# shows everything known about one component
# observed prices, full price history, forecast, plus any user calibration in effect

def analyze_component(component_id):

    syn = get_synth(component_id)
    fc = get_forecast(component_id)
    obs = get_observed(component_id)
    mult, n_obs = get_adjustment(component_id)

    if syn.empty and fc.empty and obs.empty:
        print(f"No data for {safe(component_id)}")
        return

    if not syn.empty:
        syn = syn.copy()
        syn["synthetic_price"] = syn["synthetic_price"] * mult

    if not fc.empty:
        fc = fc.copy()
        fc["price"] = fc["price"] * mult

    print(f"\n{safe(component_id)}")

    if mult != 1.0:
        print(f"  user calibration applied: x{mult:.4f}  (based on {n_obs} observation(s))")

    if not obs.empty:

        print(f"\n  Real observed prices:")

        for _, row in obs.iterrows():
            print(f"    {int(row['year'])}: ${row['observed_price_median']:.2f}  (median across listings)")

        if len(obs) == 1:
            print(f"    (component only present in {int(obs.iloc[0]['year'])} - excluded from synthetic series)")

    if not syn.empty:
        i_min = syn["synthetic_price"].idxmin()
        i_max = syn["synthetic_price"].idxmax()
        syn_start = syn["date"].min().strftime("%b %Y")
        syn_end = syn["date"].max().strftime("%b %Y")

        print(f"\n  Price history ({syn_start} - {syn_end}, synthetic + real_blended):")
        print(f"    cheapest: ${syn.loc[i_min, 'synthetic_price']:.2f} on {syn.loc[i_min, 'date'].strftime('%Y-%m-%d')}")
        print(f"    priciest: ${syn.loc[i_max, 'synthetic_price']:.2f} on {syn.loc[i_max, 'date'].strftime('%Y-%m-%d')}")

    if not fc.empty:
        i_min = fc["price"].idxmin()
        i_max = fc["price"].idxmax()
        fc_start = fc["date"].min().strftime("%b %Y")
        fc_end = fc["date"].max().strftime("%b %Y")

        print(f"\n  Forecast ({fc_start} - {fc_end}):")
        print(f"    cheapest: ${fc.loc[i_min, 'price']:.2f} on {fc.loc[i_min, 'date'].strftime('%Y-%m-%d')}")
        print(f"    priciest: ${fc.loc[i_max, 'price']:.2f} on {fc.loc[i_max, 'date'].strftime('%Y-%m-%d')}")

    elif not syn.empty:
        print(f"\n  No forecast (this component is in the matched set but forecast data missing)")

    elif not obs.empty and len(obs) == 1:
        print(f"\n  No forecast: component only present in one year, so it can't be projected forward.")


# option 3 - the headline feature
# given a component, looks at the forecast and tells the user buy now or wait

def buy_recommender():

    q = input("\nComponent query: ").strip()

    if not q:
        return

    matches = find_components(q, require_forecast=True)

    if matches.empty:
        print(f"No forecastable matches for '{q}' (only matched components have forecasts).")
        print("Try: lookup-component (option 2) for unmatched components.")
        return

    cid = pick_component(matches)

    if cid is None:
        return

    fc = get_forecast(cid)
    syn = get_synth(cid)
    mult, n_obs = get_adjustment(cid)

    if fc.empty:
        print("No forecast available for this component (it may not have been in the matched set)")
        return

    fc = fc.copy()
    fc["price"] = fc["price"] * mult

    today = pd.Timestamp.today().normalize()
    this_month = today.to_period("M")

    future = fc[fc["date"].dt.to_period("M") >= this_month]

    if future.empty:
        print("Forecast horizon is fully in the past - re-run the pipeline to refresh predictions.")
        future = fc.tail(1)

    current_idx = future.index[0]
    current_price = float(future.loc[current_idx, "price"])
    current_date = future.loc[current_idx, "date"]

    cheapest_idx = future["price"].idxmin()
    cheapest_price = float(future.loc[cheapest_idx, "price"])
    cheapest_date = future.loc[cheapest_idx, "date"]

    priciest_idx = future["price"].idxmax()
    priciest_price = float(future.loc[priciest_idx, "price"])
    priciest_date = future.loc[priciest_idx, "date"]

    savings = current_price - cheapest_price
    pct_savings = savings / current_price * 100 if current_price > 0 else 0
    months_to_cheapest = (cheapest_date.to_period("M") - current_date.to_period("M")).n

    print(f"\n{safe(cid)}")

    if mult != 1.0:
        print(f"  user calibration applied: x{mult:.4f}  (n={n_obs})")

    print(f"\n  Nearest forecast month: {current_date.strftime('%B %Y')}  ${current_price:.2f}")
    print(f"  Cheapest forecast month: {cheapest_date.strftime('%B %Y')}  ${cheapest_price:.2f}")
    print(f"  Most expensive forecast month: {priciest_date.strftime('%B %Y')}  ${priciest_price:.2f}")

    print(f"\n  Potential savings by waiting until cheapest: ${savings:.2f} ({pct_savings:.1f}%)")

    if not syn.empty:

        syn_with_mult = syn.copy()
        syn_with_mult["synthetic_price"] = syn_with_mult["synthetic_price"] * mult
        hist_min = float(syn_with_mult["synthetic_price"].min())

        if cheapest_price <= hist_min * 1.02:
            historical_context = f"  Forecast cheapest is at or below the historical low of ${hist_min:.2f} - strong buying signal."

        elif cheapest_price <= hist_min * 1.10:
            historical_context = f"  Forecast cheapest is within 10% of the historical low of ${hist_min:.2f}."

        else:
            historical_context = f"  Forecast cheapest is ${cheapest_price - hist_min:.2f} above the historical low of ${hist_min:.2f}."

        print(historical_context)

    print(f"\n  RECOMMENDATION:")

    if months_to_cheapest <= 1:
        print(f"    BUY NOW - current month is at or near the projected minimum.")

    elif pct_savings >= 8:
        print(f"    WAIT until {cheapest_date.strftime('%B %Y')} - projected savings of ${savings:.2f} ({pct_savings:.1f}%).")

    elif pct_savings >= 3:
        print(f"    MARGINAL WAIT - savings of {pct_savings:.1f}% projected, weigh against opportunity cost.")

    else:
        print(f"    BUY NOW - projected savings of {pct_savings:.1f}% are too small to justify waiting.")


# option 4 - the adaptive correction loop
# user submits a real observed price, system computes ratio vs prediction
# stores a calibration multiplier so future predictions for that component get scaled

def submit_observation():

    print("\nSubmit a real-world price observation. The system will store it and adjust")
    print("future predictions for that component to match what you actually saw.")
    print("(only matched components with forecasts can be calibrated this way)")

    q = input("\nComponent query: ").strip()

    if not q:
        return

    matches = find_components(q, require_forecast=True)

    if matches.empty:
        print(f"No forecastable matches for '{q}'.")
        return

    cid = pick_component(matches)

    if cid is None:
        return

    price_str = input("Observed price in USD: ").strip()

    try:
        price = float(price_str)

        if price <= 0:
            print("price must be positive")
            return

    except ValueError:
        print("invalid price")
        return

    date_str = input("Observation date YYYY-MM-DD (blank = today): ").strip()

    if not date_str:
        date_str = datetime.today().strftime("%Y-%m-%d")

    try:
        obs_date = pd.Timestamp(date_str)

    except Exception:
        print("invalid date")
        return

    syn = get_synth(cid)
    fc = get_forecast(cid)
    mult, n_obs = get_adjustment(cid)

    raw_pred = None
    pred_source = None

    if not syn.empty:

        nearest = (syn["date"] - obs_date).abs().idxmin()
        days_off = abs((syn.loc[nearest, "date"] - obs_date).days)

        if days_off <= 31:
            raw_pred = float(syn.loc[nearest, "synthetic_price"])
            pred_source = f"synthetic series ({syn.loc[nearest, 'date'].strftime('%Y-%m-%d')}, {days_off} days off)"

    if raw_pred is None and not fc.empty:

        nearest = (fc["date"] - obs_date).abs().idxmin()
        days_off = abs((fc.loc[nearest, "date"] - obs_date).days)

        if days_off <= 31:
            raw_pred = float(fc.loc[nearest, "price"])
            pred_source = f"forecast ({fc.loc[nearest, 'date'].strftime('%Y-%m-%d')}, {days_off} days off)"

    if raw_pred is None:
        print(f"No prediction available within 31 days of {date_str}")
        return

    adjusted_pred = raw_pred * mult
    ratio = price / raw_pred
    pct_off = (ratio - 1) * 100

    print(f"\n  Raw model prediction:    ${raw_pred:.2f}  (from {pred_source})")

    if mult != 1.0:
        print(f"  Currently calibrated to: ${adjusted_pred:.2f}  (multiplier x{mult:.4f}, n={n_obs})")

    print(f"  You observed:            ${price:.2f}")
    print(f"  Ratio (vs raw):          {ratio:.4f}  ({pct_off:+.1f}%)")

    if abs(pct_off) < 5:
        print(f"  Within 5% - model was already accurate on this component.")

    elif abs(pct_off) < 15:
        print(f"  Moderate disagreement - recording and updating calibration.")

    else:
        print(f"  Large disagreement - model was off by more than 15%, recording and updating calibration.")

    conn = sqlite3.connect(db_path)

    conn.execute("""
        INSERT INTO user_observations (component_id, observation_date, observed_price, predicted_price, ratio, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [cid, date_str, price, round(raw_pred, 4), round(ratio, 6), datetime.now().isoformat()])

    rows = conn.execute("""
        SELECT ratio FROM user_observations WHERE component_id = ?
    """, [cid]).fetchall()

    n = len(rows)
    new_mult = float(np.mean([r[0] for r in rows]))

    conn.execute("""
        INSERT INTO prediction_adjustments (component_id, multiplier, n_observations, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(component_id) DO UPDATE SET
            multiplier = excluded.multiplier,
            n_observations = excluded.n_observations,
            last_updated = excluded.last_updated
    """, [cid, round(new_mult, 6), n, datetime.now().isoformat()])

    conn.commit()
    conn.close()

    print(f"\n  Stored. Calibration multiplier for this component is now x{new_mult:.4f}")
    print(f"  All future predictions for this component (in this CLI) will be scaled by that factor.")


# option 5 - lists recent user observations and active calibrations

def view_observations():

    conn = sqlite3.connect(db_path)

    obs = pd.read_sql_query("""
        SELECT id, component_id, observation_date, observed_price, predicted_price, ratio, recorded_at
        FROM user_observations
        ORDER BY recorded_at DESC
        LIMIT 25
    """, conn)

    if obs.empty:
        print("\nNo user observations yet. Use option 4 to add one.")
        conn.close()
        return

    print(f"\n{len(obs)} most recent observations:")

    for _, row in obs.iterrows():
        print(f"  [{row['observation_date']}] {safe(row['component_id'])[:55]}  obs=${row['observed_price']:.2f}  pred=${row['predicted_price']:.2f}  ratio={row['ratio']:.3f}")

    adj = pd.read_sql_query("""
        SELECT component_id, multiplier, n_observations, last_updated
        FROM prediction_adjustments
        ORDER BY last_updated DESC
    """, conn)

    conn.close()

    if not adj.empty:

        print(f"\nActive prediction calibrations ({len(adj)}):")

        for _, row in adj.iterrows():
            print(f"  {safe(row['component_id'])[:60]}  x{row['multiplier']:.4f}  n={row['n_observations']}")


# option 6 - wipes all user observations and calibrations after typing YES

def reset_calibrations():

    confirm = input("\nThis deletes ALL user observations and calibration multipliers. Type YES to confirm: ").strip()

    if confirm != "YES":
        print("cancelled")
        return

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM user_observations")
    conn.execute("DELETE FROM prediction_adjustments")
    conn.commit()
    conn.close()

    print("All user observations and calibrations cleared.")


# option 9 - dumps every metric file: regression, real-data backtests, importance, classification

def view_metrics():

    print("\n=== Regression metrics (single train/test split) ===")
    print(pd.read_csv("cleaned_data/model_metrics.csv").to_string(index=False))

    print("\n=== Per-category MAE ===")
    pc = pd.read_csv("cleaned_data/model_metrics_per_category.csv")
    print(pc.pivot(index="model", columns="category", values="MAE").to_string())

    print("\n=== Real-data backtests (honest out-of-time evaluation) ===")
    print("Train: synthetic + early real_blended.  Test: last 3 months of real-blended prices.")

    for cat, fname in [("GPU", "real_gpu_backtest_metrics"),
                       ("Storage", "real_storage_backtest_metrics"),
                       ("RAM", "real_ram_backtest_metrics")]:

        path = f"cleaned_data/{fname}.csv"

        if not os.path.exists(path):
            print(f"  {cat}: (file not found - re-run model_training.py)")
            continue

        df = pd.read_csv(path)
        print(f"\n  {cat} (n_test={int(df['n_test'].iloc[0]):,}):")
        print(df[["model", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))

    print("\n=== Permutation feature importance ===")
    print(pd.read_csv("cleaned_data/feature_importance.csv").to_string(index=False))

    print("\n=== Top 10 hardest test predictions ===")
    hp = pd.read_csv("cleaned_data/hardest_predictions.csv")

    for _, row in hp.iterrows():
        print(f"  [{row['category']}] {safe(row['model'])[:48]}  actual=${row['mean_actual']:.2f}  pred=${row['mean_pred']:.2f}  err=${row['mean_abs_error']:.2f}")

    print("\n=== Spec-only regression metrics (cross-sectional, real prices) ===")

    if os.path.exists("cleaned_data/spec_regression_metrics.csv"):
        sr = pd.read_csv("cleaned_data/spec_regression_metrics.csv")
        print(sr.to_string(index=False))

    print("\n=== Classification (Random Forest macro-F1 by category) ===")
    cls = pd.read_csv("cleaned_data/classification_metrics.csv")
    rf = cls[cls["model"] == "Random Forest"][["category", "accuracy", "macro_f1", "macro_precision", "macro_recall"]]
    print(rf.to_string(index=False))


# option 7 - prints the 10 pipeline stages in order

def view_pipeline():

    stages = [
        "1. data_collection.py    - merge 2023 + 2025 PCPartPicker CSVs, then ingest real daily prices",
        "                           from HardwareDealsCo for GPU, SSD, and RAM (Sep 2025 - May 2026)",
        "2. data_cleaning.py      - IQR caps, spec parsing, component_id, cross-year matching",
        "3. generate_synthetic.py - 29-month synthetic series with 9 encoded events + bridge real",
        "                           bucket prices into per-SKU 'real_blended' rows for matched components",
        "4. database_setup.py     - SQLite schema (components, observed/synthetic prices,",
        "                           gpu/drive/ram_real_prices tables) + 5 sample SQL queries",
        "5. model_training.py     - LR/DT/RF on full price history, TimeSeriesSplit CV,",
        "                           permutation importance, then 3 real-data backtests (GPU/Storage/RAM)",
        "                           that hold out the last 3 real months for honest evaluation",
        "6. estimator.py          - 3-tier (budget/mid/high) build cost forecast + k-means tier discovery",
        "7. value_metrics.py      - $/GB VRAM, $/core, $/GB metrics on REAL observed prices",
        "8. spec_classifier.py    - multi-class price-tier classifier (cheap/mid/expensive) on REAL",
        "                           observed prices, with stratified k-fold CV and calibration curves",
        "9. spec_regression.py    - cross-sectional regression: predict 2025 price from specs",
        "10. data_visualization.py - 6 core plots + classifier diagnostics, with event windows shaded",
    ]

    print("\nPipeline runs end to end with: python src/main.py\n")

    for s in stages:
        print(s)


# option 8 - shows where each pipeline stage gets its data
# useful for confirming the real_blended rows landed in the synthetic_prices table

def view_data_lineage():

    conn = sqlite3.connect(db_path)

    n_comp = conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    n_obs = conn.execute("SELECT COUNT(*) FROM observed_prices").fetchone()[0]
    n_syn = conn.execute("SELECT COUNT(*) FROM synthetic_prices").fetchone()[0]

    matched = conn.execute("""
        SELECT COUNT(DISTINCT component_id) FROM observed_prices
        WHERE component_id IN (SELECT component_id FROM observed_prices WHERE year = 2023)
          AND component_id IN (SELECT component_id FROM observed_prices WHERE year = 2025)
    """).fetchone()[0]

    n_real_blended = conn.execute(
        "SELECT COUNT(*) FROM synthetic_prices WHERE data_source = 'real_blended'"
    ).fetchone()[0]

    real_blended_by_cat = pd.read_sql_query("""
        SELECT c.category, COUNT(*) AS n
        FROM synthetic_prices s JOIN components c ON s.component_id = c.component_id
        WHERE s.data_source = 'real_blended'
        GROUP BY c.category ORDER BY c.category
    """, conn)

    n_gpu_real = conn.execute("SELECT COUNT(*) FROM gpu_chipset_real_prices").fetchone()[0]
    n_drive_real = conn.execute("SELECT COUNT(*) FROM drive_real_prices").fetchone()[0]
    n_ram_real = conn.execute("SELECT COUNT(*) FROM ram_real_prices").fetchone()[0]

    conn.close()

    print("\nData lineage and usage:")
    print(f"  Raw rows ingested (2023 + 2025): 10,071")
    print(f"  After cleaning: ~9,452 rows in cleaned_data/combined_parts_cleaned.csv")
    print(f"  Components in DB: {n_comp:,}")
    print(f"  Real observed prices in DB: {n_obs:,}  (used in classifier + value metrics)")
    print(f"  Components matched in BOTH 2023 and 2025: {matched:,}")
    print(f"  Synthetic monthly rows: {n_syn:,}  (synthetic + real_blended combined)")
    print(f"    of which real_blended: {n_real_blended:,}")

    if not real_blended_by_cat.empty:

        for _, row in real_blended_by_cat.iterrows():
            print(f"      {row['category']}: {row['n']:,}")

    print(f"  External real-price tables (HardwareDealsCo daily ingest, Sep 2025 - May 2026):")
    print(f"    gpu_chipset_real_prices: {n_gpu_real:,} chipset-month rows")
    print(f"    drive_real_prices:       {n_drive_real:,} bucket-month rows")
    print(f"    ram_real_prices:         {n_ram_real:,} bucket-month rows")
    print()
    print("  Where each pipeline stage gets its data:")
    print("    Stage 5 (regression)    -> synthetic + real_blended price history")
    print("    Stage 5 (real backtest) -> real_blended rows only (last 3 months held out)")
    print("    Stage 6 (estimator)     -> forecast (synthetic-derived) + k-means on real specs")
    print("    Stage 7 (value)         -> REAL observed prices (PCPartPicker, no synthetic)")
    print("    Stage 8 (classifier)    -> REAL observed prices (PCPartPicker, no synthetic)")
    print("    Stage 9 (spec regress)  -> REAL observed 2025 prices (cross-sectional)")
    print()
    print("  Unmatched components (only one year): present in observed_prices but not synthetic.")
    print("  They contribute to Stage 7-9 but are excluded from the time-series regression.")


# option 11 - the full build cost projector
# user picks one gpu + cpu + ram + storage, the tool shows total cost per month with savings

def custom_build_cost():

    print("\nProject your full build cost across the next 12 forecast months.")
    print("Pick one component each for GPU, CPU, RAM, Storage. Skip a category by leaving blank.")

    chosen = {}

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        print(f"\n--- {cat} ---")
        q = input(f"Search query for {cat} (blank to skip): ").strip()

        if not q:
            print(f"  skipping {cat}")
            continue

        matches = find_components(q, require_forecast=True)
        matches = matches[matches["category"] == cat].reset_index(drop=True)

        if matches.empty:
            print(f"  no forecastable {cat} matches for '{q}'")
            continue

        cid = pick_component(matches)

        if cid is None:
            continue

        chosen[cat] = cid

    if not chosen:
        print("\nNo components selected.")
        return

    monthly = {}

    for cat, cid in chosen.items():

        fc = get_forecast(cid)
        mult, _ = get_adjustment(cid)

        if fc.empty:
            continue

        for _, row in fc.iterrows():

            d = row["date"].strftime("%Y-%m-%d")

            if d not in monthly:
                monthly[d] = {}

            monthly[d][cat] = float(row["price"]) * mult

    if not monthly:
        print("\nNo forecast data available for any selected component.")
        return

    df = pd.DataFrame(monthly).T.reset_index().rename(columns={"index": "month"})

    cat_cols = [c for c in chosen.keys() if c in df.columns]
    df["total"] = df[cat_cols].sum(axis=1)
    df = df.sort_values("month").reset_index(drop=True)

    print(f"\nProjected monthly cost for your build ({', '.join(chosen.keys())}):\n")

    header_parts = ["month       "] + [f"{c:>10}" for c in cat_cols] + [f"{'total':>10}"]
    print("".join(header_parts))

    for _, row in df.iterrows():

        line = [f"{row['month']}"]

        for c in cat_cols:
            line.append(f"{row[c]:>10.2f}")

        line.append(f"{row['total']:>10.2f}")
        print("  ".join(line))

    cheapest_idx = df["total"].idxmin()
    priciest_idx = df["total"].idxmax()
    cheapest = df.loc[cheapest_idx]
    priciest = df.loc[priciest_idx]
    savings = priciest["total"] - cheapest["total"]
    pct = savings / priciest["total"] * 100 if priciest["total"] > 0 else 0

    print(f"\nCheapest month: {cheapest['month']}  total ${cheapest['total']:.2f}")
    print(f"Most expensive: {priciest['month']}  total ${priciest['total']:.2f}")
    print(f"Savings by timing right: ${savings:.2f} ({pct:.1f}%)")

    save = input("\nSave this to cleaned_data/my_build_forecast.csv? (y/N): ").strip().lower()

    if save == "y":
        df.to_csv("cleaned_data/my_build_forecast.csv", index=False)
        print("Saved: cleaned_data/my_build_forecast.csv")


# option 12 - nearest neighbors on standardized spec vectors
# helps when a search by name fails because the model name is messy

def find_similar():

    print("\nFind components similar in spec-space to a given component (k-nearest-neighbors on standardized spec features).")

    q = input("\nComponent query: ").strip()

    if not q:
        return

    matches = find_components(q)

    if matches.empty:
        print(f"No matches for '{q}'")
        return

    cid = pick_component(matches)

    if cid is None:
        return

    target_row = matches[matches["component_id"] == cid].iloc[0]
    cat = target_row["category"]
    feats = spec_features.get(cat)

    if feats is None:
        print(f"No spec features defined for category {cat}")
        return

    conn = sqlite3.connect(db_path)

    cat_df = pd.read_sql_query(f"""
        SELECT c.component_id, c.brand, c.model, c.category,
               c.memory_gb, c.tdp_watts, c.core_clock_mhz,
               c.core_count,
               c.ddr_gen, c.speed_mhz, c.module_count, c.module_size_gb,
               c.capacity_gb,
               o.observed_price_median AS price_2025
        FROM components c
        LEFT JOIN observed_prices o ON c.component_id = o.component_id AND o.year = 2025
        WHERE c.category = ?
    """, conn, params=[cat])

    conn.close()

    cat_df = cat_df.dropna(subset=feats).reset_index(drop=True)

    if len(cat_df) < 6:
        print(f"Not enough {cat} components with full specs to run nearest-neighbor")
        return

    if cid not in cat_df["component_id"].values:
        print(f"Target component dropped due to missing spec features.")
        return

    X = StandardScaler().fit_transform(cat_df[feats].values)

    k = min(6, len(cat_df))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    target_pos = cat_df.index[cat_df["component_id"] == cid][0]
    distances, indices = nn.kneighbors(X[target_pos].reshape(1, -1))

    print(f"\nTarget: {safe(cid)}")

    for f in feats:
        v = target_row.get(f) if f in target_row.index else cat_df.loc[target_pos, f]

        if pd.notna(v):
            print(f"  {f} = {v}")

    print(f"\nTop {k - 1} nearest neighbors in spec space:")

    for rank, (dist, pos) in enumerate(zip(distances[0][1:], indices[0][1:]), start=1):

        nbr = cat_df.iloc[pos]
        spec_str = ", ".join(f"{f}={nbr[f]}" for f in feats if pd.notna(nbr[f]))
        price_str = f"  $2025 median ${nbr['price_2025']:.2f}" if pd.notna(nbr["price_2025"]) else ""
        print(f"  {rank}. {safe(nbr['model'])[:50]}  dist={dist:.3f}  ({spec_str}){price_str}")


# option 10 - lists the png plots and optionally opens one in the default image viewer

def list_visuals():

    if not os.path.isdir("visuals"):
        print("\nvisuals/ directory not found")
        return

    files = sorted(os.listdir("visuals"))
    files = [f for f in files if f.endswith(".png")]

    print(f"\n{len(files)} plots in visuals/:")

    for f in files:
        size_kb = os.path.getsize(os.path.join("visuals", f)) // 1024
        print(f"  visuals/{f}   ({size_kb} KB)")

    pick = input("\nOpen one in your default image viewer? Enter filename or blank to skip: ").strip()

    if not pick:
        return

    full = os.path.join("visuals", pick)

    if not os.path.exists(full):
        print(f"file not found: {full}")
        return

    if sys.platform == "win32":
        os.startfile(full)

    elif sys.platform == "darwin":
        os.system(f'open "{full}"')

    else:
        os.system(f'xdg-open "{full}"')


# the main menu loop - keeps showing options until the user types q

def menu_loop():

    print()
    print("CS 210 Data Management Project - Interactive Console")

    while True:

        print()
        print("  1) project summary (counts, metrics)")
        print("  2) look up a component (history + forecast)")
        print("  3) buy now or wait? recommendation")
        print("  4) submit a real observation (adaptive correction)")
        print("  5) view user observations + active calibrations")
        print("  6) reset all user observations + calibrations")
        print("  7) view pipeline stages")
        print("  8) view data lineage (what data was used where)")
        print("  9) view all metrics (regression + classification)")
        print(" 10) list visualizations (and optionally open one)")
        print(" 11) build my PC (custom 4-part build cost across forecast months)")
        print(" 12) find similar components (nearest-neighbor on spec vectors)")
        print("  q) quit")

        choice = input("\n> ").strip().lower()

        try:

            if choice == "1":
                show_summary()

            elif choice == "2":
                lookup_component()

            elif choice == "3":
                buy_recommender()

            elif choice == "4":
                submit_observation()

            elif choice == "5":
                view_observations()

            elif choice == "6":
                reset_calibrations()

            elif choice == "7":
                view_pipeline()

            elif choice == "8":
                view_data_lineage()

            elif choice == "9":
                view_metrics()

            elif choice == "10":
                list_visuals()

            elif choice == "11":
                custom_build_cost()

            elif choice == "12":
                find_similar()

            elif choice in ("q", "quit", "exit", ""):
                print("bye")
                break

            else:
                print("invalid choice")

        except FileNotFoundError as e:
            print(f"missing file: {e}.  Make sure you have run python src/main.py first.")

        except Exception as e:
            print(f"error: {safe(str(e))}")


# entry point - sets up the user tables then drops into the menu loop

def main():
    ensure_user_tables()
    menu_loop()


if __name__ == "__main__":
    main()
