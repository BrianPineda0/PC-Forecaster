import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


# pulls components plus their real observed prices from sqlite

def load_components_with_prices():

    conn = sqlite3.connect("pcparts.db")
    df = pd.read_sql_query("""
        SELECT c.component_id, c.category, c.brand, c.model,
               c.memory_gb, c.core_count, c.tdp_watts,
               c.module_count, c.module_size_gb, c.capacity_gb,
               o.year, o.observed_price_median AS price
        FROM components c
        JOIN observed_prices o ON c.component_id = o.component_id
    """, conn)
    conn.close()
    return df


# computes the per-unit value metric for each category
# $/gb vram for gpu, $/core for cpu, $/gb for ram and storage

def compute_metrics(df):

    gpu = df[(df["category"] == "GPU") & (df["memory_gb"] > 0)].copy()
    gpu["value_metric"] = gpu["price"] / gpu["memory_gb"]
    gpu["metric_label"] = "$/GB VRAM"

    cpu = df[(df["category"] == "CPU") & (df["core_count"] > 0)].copy()
    cpu["value_metric"] = cpu["price"] / cpu["core_count"]
    cpu["metric_label"] = "$/core"

    ram = df[(df["category"] == "RAM") & df["module_count"].notna() & df["module_size_gb"].notna()].copy()
    ram["total_gb"] = ram["module_count"] * ram["module_size_gb"]
    ram = ram[ram["total_gb"] > 0]
    ram["value_metric"] = ram["price"] / ram["total_gb"]
    ram["metric_label"] = "$/GB"

    storage = df[(df["category"] == "Storage") & (df["capacity_gb"] > 0)].copy()
    storage["value_metric"] = storage["price"] / storage["capacity_gb"]
    storage["metric_label"] = "$/GB"

    return pd.concat([gpu, cpu, ram, storage], ignore_index=True)


# prints best and worst value components per category

def print_summary(metrics):

    print("\nPrice-to-performance metrics:\n")

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        sub = metrics[metrics["category"] == cat]

        if sub.empty:
            continue

        label = sub["metric_label"].iloc[0]
        print(f"{cat} ({label})")
        print(f"  components analyzed: {len(sub):,}")
        print(f"  average: ${sub['value_metric'].mean():.2f}")
        print(f"  median: ${sub['value_metric'].median():.2f}")
        print(f"  best 3 (lowest = better value):")

        for _, row in sub.nsmallest(3, "value_metric").iterrows():
            name = str(row["model"])[:50]
            print(f"    {name} ${row['value_metric']:.3f} ({row['year']})")

        print(f"  worst 3 (highest = poor value):")

        for _, row in sub.nlargest(3, "value_metric").iterrows():
            name = str(row["model"])[:50]
            print(f"    {name} ${row['value_metric']:.3f} ({row['year']})")

        print()


# draws the per-unit value chart per category

def plot_metrics(metrics):

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    colors = {"GPU": "red", "CPU": "blue", "RAM": "green", "Storage": "orange"}

    for ax, cat in zip(axes, ["GPU", "CPU", "RAM", "Storage"]):

        sub = metrics[metrics["category"] == cat]

        if sub.empty:
            continue

        years = sorted(sub["year"].unique())
        data = [sub[sub["year"] == y]["value_metric"].values for y in years]
        median = sub["value_metric"].median()

        ax.boxplot(data, tick_labels=[str(y) for y in years], patch_artist=True,
                   boxprops=dict(facecolor=colors[cat], alpha=0.5))

        label = sub["metric_label"].iloc[0]
        ax.set_title(f"{cat} ({label})")
        ax.set_ylabel(label)
        ax.axhline(median, color="gray", linestyle="--", alpha=0.7,
                   label=f"overall median ${median:.2f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Price-to-Performance Distribution by Category and Year")
    plt.tight_layout()
    plt.savefig("visuals/value_per_unit.png")
    plt.close()
    print("Saved: value_per_unit.png")


# computes per-unit value metrics on real prices, prints summary, saves the chart

def main():

    df = load_components_with_prices()
    metrics = compute_metrics(df)

    print_summary(metrics)

    keep_cols = ["component_id", "category", "brand", "model",
                 "year", "price", "value_metric", "metric_label"]
    metrics[keep_cols].to_csv("cleaned_data/value_metrics.csv", index=False)
    print("Saved: value_metrics.csv")

    plot_metrics(metrics)
