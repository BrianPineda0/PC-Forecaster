import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


tier_rules = {
    "GPU": {"budget": (0, 8), "mid": (8, 12), "high": (12, 9999)},
    "CPU": {"budget": (0, 6), "mid": (6, 8), "high": (8, 9999)},
    "RAM": {"budget": (0, 16), "mid": (16, 32), "high": (32, 9999)},
    "Storage": {"budget": (0, 500), "mid": (500, 1000), "high": (1000, 999999)},
}


# pulls components and the forecast csv, merges them so each predicted month has its specs

def load_data():

    conn = sqlite3.connect("pcparts.db")
    components = pd.read_sql_query("SELECT * FROM components", conn)
    conn.close()

    forecast = pd.read_csv("cleaned_data/future_price_predictions.csv")

    components = components.drop(columns=["category"])

    merged = forecast.merge(components, on="component_id", how="left")

    return merged


# maps each component into budget mid or high based on its key spec
# vram for gpu, core_count for cpu, kit gb for ram, capacity_gb for storage

def assign_tier(row):

    cat = row["category"]

    if cat == "GPU":
        spec = row["memory_gb"]

    elif cat == "CPU":
        spec = row["core_count"]

    elif cat == "RAM":

        count = row["module_count"]
        size = row["module_size_gb"]

        if pd.isna(count) or pd.isna(size):
            return None

        spec = count * size

    elif cat == "Storage":
        spec = row["capacity_gb"]

    else:
        return None

    if pd.isna(spec):
        return None

    rules = tier_rules[cat]

    for tier_name, (low, high) in rules.items():

        if low <= spec < high:
            return tier_name

    return None


# for each forecast month, averages the predicted price within each tier
# then sums tiers across categories to get a full pc build cost per month

def build_cost_table(df):

    df["tier"] = df.apply(assign_tier, axis=1)

    df = df.dropna(subset=["tier"])

    grouped = (
        df.groupby(["tier", "category", "forecast_month"])["predicted_price"]
        .median()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["tier", "forecast_month"],
        columns="category",
        values="predicted_price",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={
        "CPU": "cpu_cost",
        "GPU": "gpu_cost",
        "RAM": "ram_cost",
        "Storage": "storage_cost",
    })

    pivot["total_cost"] = (
        pivot["cpu_cost"]
        + pivot["gpu_cost"]
        + pivot["ram_cost"]
        + pivot["storage_cost"]
    )

    return pivot


# prints cheapest and priciest forecast month per tier and the savings

def print_summary(table):

    months = sorted(table["forecast_month"].unique())
    span_label = f"{pd.Timestamp(months[0]).strftime('%b %Y')} - {pd.Timestamp(months[-1]).strftime('%b %Y')}"
    print(f"\nBuild cost estimates ({span_label}):\n")

    for tier in ["budget", "mid", "high"]:

        sub = table[table["tier"] == tier].sort_values("forecast_month")

        if sub.empty:
            continue

        cheapest = sub.loc[sub["total_cost"].idxmin()]
        priciest = sub.loc[sub["total_cost"].idxmax()]
        savings = priciest["total_cost"] - cheapest["total_cost"]

        print(f"{tier} build")
        print(f"  Cheapest month: {cheapest['forecast_month']} total ${cheapest['total_cost']:.2f}")
        print(f"    CPU ${cheapest['cpu_cost']:.2f}, GPU ${cheapest['gpu_cost']:.2f}, "
              f"RAM ${cheapest['ram_cost']:.2f}, Storage ${cheapest['storage_cost']:.2f}")
        print(f"  Most expensive: {priciest['forecast_month']} total ${priciest['total_cost']:.2f}")
        print(f"  Savings by timing right: ${savings:.2f}")
        print()


# draws build cost over time per tier on one chart
# this is the build_cost_by_tier.png plot the report references

def plot_build_costs(table):

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"budget": "green", "mid": "blue", "high": "red"}

    for tier in ["budget", "mid", "high"]:

        sub = table[table["tier"] == tier].sort_values("forecast_month")

        if sub.empty:
            continue

        ax.plot(
            sub["forecast_month"], sub["total_cost"],
            marker="o",
            label=tier.capitalize(),
            color=colors[tier],
        )

    months = sorted(table["forecast_month"].unique())
    span_label = f"{pd.Timestamp(months[0]).strftime('%b %Y')} - {pd.Timestamp(months[-1]).strftime('%b %Y')}"
    ax.set_title(f"Projected PC Build Cost by Tier ({span_label})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Build Cost ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("visuals/build_cost_by_tier.png")
    plt.close()
    print("Saved: build_cost_by_tier.png")


# runs k-means with k=3 on the spec features per category
# compares the kmeans clusters to my hand-coded tier rules
# this shows where my rules agree with what the data naturally clusters into and where they don't

def compare_data_driven_tiers(df):

    feature_map = {
        "GPU": ["memory_gb", "tdp_watts"],
        "CPU": ["core_count", "tdp_watts"],
        "RAM": ["total_gb"],
        "Storage": ["capacity_gb"],
    }

    df = df.drop_duplicates("component_id").copy()
    df["total_gb"] = df["module_count"] * df["module_size_gb"]
    df["coded_tier"] = df.apply(assign_tier, axis=1)

    print("\nData-driven tier discovery (k-means, k=3) vs hand-coded thresholds:")

    rows = []

    for cat, feats in feature_map.items():

        sub = df[df["category"] == cat].dropna(subset=feats + ["coded_tier"]).copy()

        if len(sub) < 30:
            continue

        X = StandardScaler().fit_transform(sub[feats].values)
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        sub["raw_cluster"] = km.fit_predict(X)

        order = sub.groupby("raw_cluster")[feats[0]].mean().sort_values().index.tolist()
        remap = {raw: rank for rank, raw in enumerate(order)}
        tier_name = {0: "budget", 1: "mid", 2: "high"}
        sub["data_tier"] = sub["raw_cluster"].map(remap).map(tier_name)

        agree = (sub["data_tier"] == sub["coded_tier"]).mean()
        print(f"  {cat} ({len(sub)} components): {agree*100:.1f}% agreement")

        for tier in ["budget", "mid", "high"]:

            members = sub[sub["data_tier"] == tier]

            if members.empty:
                continue

            ranges = ", ".join(f"{f}=[{members[f].min():.0f}-{members[f].max():.0f}]" for f in feats)
            print(f"    {tier} cluster (n={len(members)}): {ranges}")

            rows.append({
                "category": cat,
                "tier": tier,
                "cluster_size": len(members),
                "spec_summary": ranges,
                "agreement_with_hand_coded": round(agree, 4),
            })

    pd.DataFrame(rows).to_csv("cleaned_data/tier_clusters.csv", index=False)
    print("Saved: cleaned_data/tier_clusters.csv")


# builds the build cost forecast across budget mid and high tiers
# saves the cost table and the chart then runs the kmeans tier comparison

def main():

    df = load_data()
    table = build_cost_table(df)

    print_summary(table)

    table.to_csv("cleaned_data/build_cost_estimates.csv", index=False)
    print("Saved: build_cost_estimates.csv")

    plot_build_costs(table)

    compare_data_driven_tiers(df)
