import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


categories = ["GPU", "CPU", "RAM", "Storage"]
cat_color = {"GPU": "red", "CPU": "blue", "RAM": "green", "Storage": "orange"}
months_short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

event_windows = [
    ("post-crypto", "2023-01", "2023-06"),
    ("AI demand wave", "2023-03", "2023-12"),
    ("NAND oversupply", "2023-01", "2023-08"),
    ("RAM shortage", "2023-10", "2024-04"),
    ("BF 2023", "2023-11", "2023-11"),
    ("AI surge", "2024-01", "2024-07"),
    ("new CPU gen", "2024-08", "2024-12"),
    ("BF 2024", "2024-11", "2024-11"),
    ("RTX 5000 launch", "2025-01", "2025-05"),
]


# tiny helper that opens a sqlite connection

def get_conn():
    return sqlite3.connect("pcparts.db")


# pulls the avg synthetic price per category per month for the trend chart

def load_synthetic_trends():

    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT c.category, s.date, AVG(s.synthetic_price) AS avg_price
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        GROUP BY c.category, s.date
        ORDER BY c.category, s.date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# pulls the avg forecast price per category per month for the trend chart

def load_future_trends():

    df = pd.read_csv("cleaned_data/future_price_predictions.csv")
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])

    return (
        df.groupby(["category", "forecast_month"])["predicted_price"]
        .mean().round(2).reset_index()
        .rename(columns={"forecast_month": "date", "predicted_price": "avg_price"})
    )


# pulls the per-component price change between 2023 and 2025 for the histogram

def load_price_changes():

    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            c.category,
            ROUND(
                (o2.observed_price_median - o1.observed_price_median)
                / o1.observed_price_median * 100, 1
            ) AS pct_change
        FROM observed_prices o1
        JOIN observed_prices o2
            ON o1.component_id = o2.component_id AND o2.year = 2025
        JOIN components c ON o1.component_id = c.component_id
        WHERE o1.year = 2023
          AND o1.observed_price_median >= 5
    """, conn)
    conn.close()
    return df


# pulls the avg price per category per calendar month for the cheapest-month chart

def load_seasonal_averages():

    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            c.category,
            CAST(strftime('%m', s.date) AS INTEGER) AS month,
            AVG(s.synthetic_price) AS avg_price
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        GROUP BY c.category, month
        ORDER BY c.category, month
    """, conn)
    conn.close()
    return df


# picks one interesting component per category for the sample curves chart
# tries to find ones with big price swings so the chart actually shows movement

def select_sample_components():

    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            o1.component_id,
            c.category,
            c.model,
            c.brand,
            o1.observed_price_median  AS price_2023,
            o2.observed_price_median  AS price_2025,
            ROUND(
                (o2.observed_price_median - o1.observed_price_median)
                / o1.observed_price_median * 100, 1
            ) AS pct_change
        FROM observed_prices o1
        JOIN observed_prices o2
            ON o1.component_id = o2.component_id AND o2.year = 2025
        JOIN components c ON o1.component_id = c.component_id
        WHERE o1.year = 2023
          AND o1.observed_price_median >= 30
          AND o2.observed_price_median >= 30
    """, conn)
    conn.close()

    selected = []

    for cat in categories:

        sub = df[df["category"] == cat]
        selected.append(sub.nlargest(1, "pct_change"))

        if cat == "GPU":
            selected.append(sub.nsmallest(1, "pct_change"))

    return pd.concat(selected, ignore_index=True)


# pulls the full synthetic price history for the chosen sample components

def load_component_history(component_ids):

    placeholders = ",".join("?" * len(component_ids))
    conn = get_conn()
    df = pd.read_sql_query(f"""
        SELECT s.component_id, s.date, s.synthetic_price,
               c.category, c.model, c.brand
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        WHERE s.component_id IN ({placeholders})
        ORDER BY s.component_id, s.date
    """, conn, params=component_ids)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# pulls the 12-month forecast for the chosen sample components

def load_component_future(component_ids):

    df = pd.read_csv("cleaned_data/future_price_predictions.csv")
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    return df[df["component_id"].isin(component_ids)].copy()


# draws the category price history chart with shaded event windows

def plot_price_trends(synthetic, future):

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for cat in categories:

        color = cat_color[cat]
        hist = synthetic[synthetic["category"] == cat].sort_values("date")
        pred = future[future["category"] == cat].sort_values("date")

        ax.plot(hist["date"], hist["avg_price"], color=color, label=cat)

        if not pred.empty:
            ax.plot(pred["date"], pred["avg_price"], color=color, linestyle="--", label=f"{cat} (forecast)")

    cutoff = pd.Timestamp("2025-06-01")
    ax.axvline(cutoff, color="gray", linestyle=":")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    levels = [ymin + span * 0.97, ymin + span * 0.91, ymin + span * 0.85]

    for i, (label, start, end) in enumerate(event_windows):

        s = pd.Timestamp(start)
        e = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        y_use = levels[i % len(levels)]

        if start == end:
            x = s + pd.Timedelta(days=15)
            ax.axvline(x, color="gray", linestyle=":", alpha=0.5)
            ax.text(x, y_use, label, ha="center", va="top",
                    fontsize=7, color="gray", rotation=90)

        else:
            ax.axvspan(s, e, color="gray", alpha=0.05)
            ax.text(s + (e - s) / 2, y_use, label,
                    ha="center", va="top", fontsize=7, color="dimgray")

    ax.set_title("Average Component Prices Over Time (2023-2026) — gray bands mark encoded market events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Price ($)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("visuals/category_price_history.png")
    plt.close()
    print("Saved: category_price_history.png")


# draws the histogram of % price changes from 2023 to 2025 across components

def plot_price_distribution(changes):

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, cat in zip(axes, categories):

        sub = changes[changes["category"] == cat]["pct_change"].dropna()
        color = cat_color[cat]
        median_val = sub.median()

        ax.hist(sub, bins=30, color=color, alpha=0.7)
        ax.axvline(0, color="black", linestyle="-", alpha=0.5)
        ax.axvline(median_val, color="black", linestyle="--", label=f"Median: {median_val:.1f}%")

        ax.set_title(cat)
        ax.set_xlabel("% Price Change (2023 to 2025)")
        ax.set_ylabel("Number of Components")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Distribution of Component Price Changes (2023 to 2025)")
    plt.tight_layout()

    plt.savefig("visuals/price_change_histogram.png")
    plt.close()
    print("Saved: price_change_histogram.png")


# draws the sample component curves chart with synthetic past and forecast future

def plot_sample_curves(history, future, meta):

    n = len(meta)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.6))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(meta.iterrows()):

        ax = axes[idx]
        cid = row["component_id"]
        color = cat_color[row["category"]]

        hist = history[history["component_id"] == cid].sort_values("date")
        ax.plot(hist["date"], hist["synthetic_price"], color=color, label="synthetic")

        pred = future[future["component_id"] == cid].sort_values("forecast_month")

        if not pred.empty:
            ax.plot(pred["forecast_month"], pred["predicted_price"], color=color, linestyle="--", label="forecast")

        p23 = row["price_2023"]
        p25 = row["price_2025"]
        ax.axhline(p23, color="green", linestyle=":", label=f"2023 ${p23:.0f}")
        ax.axhline(p25, color="red", linestyle=":", label=f"2025 ${p25:.0f}")

        title_model = str(row["model"])[:45]
        ax.set_title(f"[{row['category']}] {title_model}", fontsize=9)
        ax.set_ylabel("Price ($)")
        plt.setp(ax.get_xticklabels(), rotation=25, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Sample Component Price Curves (Synthetic + Forecast)")
    plt.tight_layout()

    plt.savefig("visuals/sample_component_curves.png")
    plt.close()
    print("Saved: sample_component_curves.png")


# draws the cheapest-month-by-category bar chart

def plot_best_month(seasonal):

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, cat in zip(axes, categories):

        sub = seasonal[seasonal["category"] == cat].sort_values("month")
        months = sub["month"].tolist()
        prices = sub["avg_price"].tolist()

        best_idx = prices.index(min(prices))
        bar_colors = ["lightgray"] * len(months)
        bar_colors[best_idx] = cat_color[cat]

        ax.bar([months_short[m - 1] for m in months], prices, color=bar_colors)

        ax.set_title(cat)
        ax.set_ylabel("Avg Price ($)")
        ax.set_ylim(min(prices) * 0.92, max(prices) * 1.065)
        ax.grid(alpha=0.3)

    fig.suptitle("Best Month to Buy by Category")
    plt.tight_layout()

    plt.savefig("visuals/cheapest_month_by_category.png")
    plt.close()
    print("Saved: cheapest_month_by_category.png")


# pulls spec + price data for one category to compute the spec/price correlation matrix

def load_corr_data(category, spec_cols):

    conn = get_conn()
    cols = ", ".join(["o.observed_price_median AS price"] + [f"c.{s}" for s in spec_cols])
    df = pd.read_sql_query(f"""
        SELECT {cols}
        FROM components c
        JOIN observed_prices o ON c.component_id = o.component_id
        WHERE c.category = '{category}'
    """, conn)
    conn.close()
    return df


# draws the spec/price correlation heatmap per category

def plot_correlation_heatmap():

    cat_specs = {
        "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
        "CPU": ["core_count", "tdp_watts"],
        "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
        "Storage": ["capacity_gb", "pcie_gen"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, (cat, spec_cols) in zip(axes, cat_specs.items()):

        df = load_corr_data(cat, spec_cols)
        df = df.dropna()
        corr = df.corr()

        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=35, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.set_title(f"{cat} (n={len(df)})")

        for i in range(len(corr.columns)):

            for j in range(len(corr.columns)):

                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax)

    fig.suptitle("Price vs Spec Correlation by Category")
    plt.tight_layout()
    plt.savefig("visuals/spec_price_correlation.png")
    plt.close()
    print("Saved: spec_price_correlation.png")


# overlays synthetic gpu averages on top of the real chipset prices
# this is the chart that visually validates the bridge

def plot_real_vs_synthetic_gpu():

    import gpu_real_data

    conn = get_conn()

    real_chip = pd.read_sql_query("""
        SELECT chipset, month, real_median_price, sample_size
        FROM gpu_chipset_real_prices
        ORDER BY chipset, month
    """, conn)

    if real_chip.empty:
        conn.close()
        print("Skipped real_vs_synthetic_gpu: no real chipset data in DB")
        return

    real_chip["month"] = pd.to_datetime(real_chip["month"])

    gpu_synth = pd.read_sql_query("""
        SELECT s.component_id, s.date, s.synthetic_price, s.data_source
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        WHERE c.category = 'GPU' AND s.data_source = 'synthetic'
    """, conn)
    conn.close()

    gpu_synth["date"] = pd.to_datetime(gpu_synth["date"])

    def chipset_from_cid(cid):

        if not isinstance(cid, str) or "|" not in cid:
            return None

        return gpu_real_data.normalize_chipset(cid.split("|", 1)[1].strip())

    gpu_synth["chipset_norm"] = gpu_synth["component_id"].map(chipset_from_cid)

    chipset_totals = (
        real_chip.groupby("chipset")["sample_size"].sum()
        .sort_values(ascending=False)
    )

    pick = []

    for chip in chipset_totals.index:

        if (real_chip["chipset"] == chip).sum() >= 6 and (gpu_synth["chipset_norm"] == chip).any():
            pick.append(chip)

        if len(pick) == 4:
            break

    if not pick:
        print("Skipped real_vs_synthetic_gpu: no chipsets matched between real and synthetic data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, chip in zip(axes.flat, pick):

        real_sub = real_chip[real_chip["chipset"] == chip].sort_values("month")

        synth_sub = (
            gpu_synth[gpu_synth["chipset_norm"] == chip]
            .groupby("date")["synthetic_price"].mean()
            .sort_index()
        )

        if not synth_sub.empty:
            ax.plot(synth_sub.index, synth_sub.values, color="gray", linewidth=2, label="synthetic interpolation (avg AIB)")

        ax.plot(real_sub["month"], real_sub["real_median_price"], color="red", linewidth=2, marker="o", markersize=4, label="real chipset median")

        ax.set_title(chip)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Real GPU prices vs synthetic interpolation")
    plt.tight_layout()
    plt.savefig("visuals/real_vs_synthetic_gpu.png")
    plt.close()
    print("Saved: real_vs_synthetic_gpu.png")


# builds every chart used in the report and saves them to visuals/

def main():

    synthetic = load_synthetic_trends()
    future_cat = load_future_trends()
    changes = load_price_changes()
    seasonal = load_seasonal_averages()
    sample_meta = select_sample_components()

    cids = sample_meta["component_id"].tolist()
    comp_history = load_component_history(cids)
    comp_future = load_component_future(cids)

    print(f"  synthetic trend rows: {len(synthetic):,}")
    print(f"  future trend rows: {len(future_cat):,}")
    print(f"  price change rows: {len(changes):,}")
    print(f"  seasonal rows: {len(seasonal):,}")
    print(f"  sample components: {len(sample_meta)}")

    for _, r in sample_meta.iterrows():
        print(f"    [{r['category']}] {r['model'][:55]} ({r['pct_change']:.1f}%, ${r['price_2023']:.0f} to ${r['price_2025']:.0f})")

    plot_price_trends(synthetic, future_cat)
    plot_price_distribution(changes)
    plot_sample_curves(comp_history, comp_future, sample_meta)
    plot_best_month(seasonal)
    plot_correlation_heatmap()
    plot_real_vs_synthetic_gpu()

    print("\nAll plots saved to visuals/")
