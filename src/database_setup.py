import os
import sqlite3
import pandas as pd


real_gpu_path = "cleaned_data/gpu_chipset_real_prices.csv"
real_drive_path = "cleaned_data/drive_real_prices.csv"
real_ram_path = "cleaned_data/ram_real_prices.csv"


ddl = """
CREATE TABLE IF NOT EXISTS components (
    component_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    brand TEXT,
    model TEXT NOT NULL,
    memory_gb REAL,
    core_clock_mhz REAL,
    tdp_watts REAL,
    core_count REAL,
    microarchitecture TEXT,
    ddr_gen REAL,
    speed_mhz REAL,
    module_count REAL,
    module_size_gb REAL,
    capacity_gb REAL,
    storage_type TEXT,
    interface_type TEXT,
    pcie_gen REAL
);

CREATE TABLE IF NOT EXISTS observed_prices (
    component_id TEXT NOT NULL REFERENCES components(component_id),
    year INTEGER NOT NULL,
    observed_price_median REAL NOT NULL,
    PRIMARY KEY (component_id, year)
);

CREATE TABLE IF NOT EXISTS synthetic_prices (
    component_id TEXT NOT NULL REFERENCES components(component_id),
    date TEXT NOT NULL,
    synthetic_price REAL NOT NULL,
    event_label TEXT,
    data_source TEXT NOT NULL DEFAULT 'synthetic',
    PRIMARY KEY (component_id, date)
);

CREATE TABLE IF NOT EXISTS gpu_chipset_real_prices (
    chipset TEXT NOT NULL,
    month TEXT NOT NULL,
    real_median_price REAL NOT NULL,
    real_mean_price REAL,
    sample_size INTEGER,
    raw_sample_size INTEGER,
    PRIMARY KEY (chipset, month)
);

CREATE TABLE IF NOT EXISTS drive_real_prices (
    drive_bucket TEXT NOT NULL,
    drive_type TEXT,
    interface_type TEXT,
    capacity_gb INTEGER,
    month TEXT NOT NULL,
    real_median_price REAL NOT NULL,
    real_mean_price REAL,
    sample_size INTEGER,
    raw_sample_size INTEGER,
    PRIMARY KEY (drive_bucket, month)
);

CREATE TABLE IF NOT EXISTS ram_real_prices (
    ram_bucket TEXT NOT NULL,
    ddr_gen INTEGER,
    speed_mhz INTEGER,
    capacity_gb INTEGER,
    month TEXT NOT NULL,
    real_median_price REAL NOT NULL,
    real_mean_price REAL,
    sample_size INTEGER,
    raw_sample_size INTEGER,
    PRIMARY KEY (ram_bucket, month)
);
"""


# turns the cleaned parts dataframe into one row per component for the components table

def build_components(df):

    spec_cols = [
        "category", "brand", "model",
        "memory_gb", "core_clock_mhz", "tdp_watts",
        "core_count", "microarchitecture",
        "ddr_gen", "speed_mhz", "module_count", "module_size_gb",
        "capacity_gb", "storage_type", "interface_type", "pcie_gen",
    ]

    return (
        df.groupby("component_id")[spec_cols]
        .first()
        .reset_index()
    )


# medians the real pcpartpicker prices per component per year for the observed_prices table

def build_observed_prices(df):

    return (
        df.groupby(["component_id", "year"])["price"]
        .median()
        .round(2)
        .reset_index()
        .rename(columns={"price": "observed_price_median"})
    )


# preps the price_history dataframe for the synthetic_prices table
# keeps the data_source column so synthetic vs real_blended rows stay marked

def build_synthetic_prices(hist):

    cols = ["component_id", "date", "price", "event_label"]

    if "data_source" in hist.columns:
        cols.append("data_source")

    out = hist[cols].rename(columns={"price": "synthetic_price"})

    if "data_source" not in out.columns:
        out["data_source"] = "synthetic"

    return out


# wipes the old tables and recreates them from the ddl above
# done this way so the script can be rerun without leftover state

def create_db(conn):

    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS ram_real_prices; "
                      "DROP TABLE IF EXISTS drive_real_prices; "
                      "DROP TABLE IF EXISTS gpu_chipset_real_prices; "
                      "DROP TABLE IF EXISTS synthetic_prices; "
                      "DROP TABLE IF EXISTS observed_prices; "
                      "DROP TABLE IF EXISTS components;")
    cur.executescript(ddl)
    conn.commit()


# tiny helper that pushes a dataframe into a sqlite table

def load_table(conn, df, table):

    df.to_sql(table, conn, if_exists="append", index=False)


# prints how many rows landed in each table so i can sanity check the load

def verify_counts(conn):

    print("\nRow counts after insert:")

    for table in ("components", "observed_prices", "synthetic_prices", "gpu_chipset_real_prices", "drive_real_prices", "ram_real_prices"):

        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")


queries = [
    (
        "Q1: avg price by category per month",
        """
        SELECT c.category, s.date, ROUND(AVG(s.synthetic_price), 2) AS avg_price
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        GROUP BY c.category, s.date
        ORDER BY c.category, s.date
        LIMIT 32;
        """,
    ),
    (
        "Q2: cheapest month per category",
        """
        SELECT category, date, avg_price
        FROM (
            SELECT c.category, s.date, ROUND(AVG(s.synthetic_price), 2) AS avg_price,
                RANK() OVER (PARTITION BY c.category ORDER BY AVG(s.synthetic_price)) AS rnk
            FROM synthetic_prices s
            JOIN components c ON s.component_id = c.component_id
            GROUP BY c.category, s.date
        )
        WHERE rnk = 1;
        """,
    ),
    (
        "Q3: 2023 vs 2025 median prices",
        """
        SELECT c.category,
            ROUND(AVG(CASE WHEN o.year = 2023 THEN o.observed_price_median END), 2) AS avg_2023,
            ROUND(AVG(CASE WHEN o.year = 2025 THEN o.observed_price_median END), 2) AS avg_2025,
            ROUND(
                (AVG(CASE WHEN o.year = 2025 THEN o.observed_price_median END) -
                 AVG(CASE WHEN o.year = 2023 THEN o.observed_price_median END)) /
                 AVG(CASE WHEN o.year = 2023 THEN o.observed_price_median END) * 100, 1
            ) AS pct_change
        FROM observed_prices o
        JOIN components c ON o.component_id = c.component_id
        GROUP BY c.category
        ORDER BY c.category;
        """,
    ),
    (
        "Q4: top 10 price increases",
        """
        SELECT c.category, c.model,
            ROUND(s_start.synthetic_price, 2) AS jan2023_price,
            ROUND(s_end.synthetic_price, 2) AS may2025_price,
            ROUND(s_end.synthetic_price - s_start.synthetic_price, 2) AS price_increase
        FROM synthetic_prices s_start
        JOIN synthetic_prices s_end
            ON s_start.component_id = s_end.component_id
           AND s_start.date = '2023-01-01'
           AND s_end.date = '2025-05-01'
        JOIN components c ON s_start.component_id = c.component_id
        WHERE s_end.synthetic_price > s_start.synthetic_price
        ORDER BY price_increase DESC
        LIMIT 10;
        """,
    ),
    (
        "Q5: top 10 price drops",
        """
        SELECT c.category, c.model,
            ROUND(s_start.synthetic_price, 2) AS jan2023_price,
            ROUND(s_end.synthetic_price, 2) AS may2025_price,
            ROUND(s_start.synthetic_price - s_end.synthetic_price, 2) AS price_drop
        FROM synthetic_prices s_start
        JOIN synthetic_prices s_end
            ON s_start.component_id = s_end.component_id
           AND s_start.date = '2023-01-01'
           AND s_end.date = '2025-05-01'
        JOIN components c ON s_start.component_id = c.component_id
        WHERE s_end.synthetic_price < s_start.synthetic_price
        ORDER BY price_drop DESC
        LIMIT 10;
        """,
    ),
]


# runs the 5 demo sql queries from the queries list above and prints results
# these are the queries the rubric wants for the sql portion of the project

def run_sample_queries(conn):

    for title, sql in queries:

        print(f"\n{title}")

        cur = conn.execute(sql)
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]

        print("  " + " | ".join(col_names))

        for row in rows:
            print("  " + " | ".join(str(v) for v in row))


# builds the whole sqlite database from the cleaned csvs
# loads components observed and synthetic plus the 3 external real-price tables
# then runs the demo queries

def main():

    df_clean = pd.read_csv("cleaned_data/combined_parts_cleaned.csv")
    df_hist = pd.read_csv("cleaned_data/price_history.csv")

    components = build_components(df_clean)
    observed_prices = build_observed_prices(df_clean)
    synthetic_prices = build_synthetic_prices(df_hist)

    real_gpu = pd.read_csv(real_gpu_path) if os.path.exists(real_gpu_path) else pd.DataFrame()
    real_drive = pd.read_csv(real_drive_path) if os.path.exists(real_drive_path) else pd.DataFrame()
    real_ram = pd.read_csv(real_ram_path) if os.path.exists(real_ram_path) else pd.DataFrame()

    print(f"  components: {len(components):,} rows")
    print(f"  observed_prices: {len(observed_prices):,} rows")
    print(f"  synthetic_prices: {len(synthetic_prices):,} rows")

    if len(real_gpu):
        print(f"  gpu_chipset_real_prices: {len(real_gpu):,} rows")

    if len(real_drive):
        print(f"  drive_real_prices: {len(real_drive):,} rows")

    if len(real_ram):
        print(f"  ram_real_prices: {len(real_ram):,} rows")

    conn = sqlite3.connect("pcparts.db")

    create_db(conn)
    load_table(conn, components, "components")
    load_table(conn, observed_prices, "observed_prices")
    load_table(conn, synthetic_prices, "synthetic_prices")

    if len(real_gpu):
        load_table(conn, real_gpu, "gpu_chipset_real_prices")

    if len(real_drive):
        load_table(conn, real_drive, "drive_real_prices")

    if len(real_ram):
        load_table(conn, real_ram, "ram_real_prices")

    conn.commit()

    verify_counts(conn)
    run_sample_queries(conn)

    conn.close()
    print("\nDone. Database saved to: pcparts.db")
