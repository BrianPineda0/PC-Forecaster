import pandas as pd
import sqlite3

df = pd.read_csv("cleaned_data/combined_parts.csv")

# brand counts
print(df["brand"].value_counts().to_string())

# columns and shape
print(df.columns.tolist())
print(df.shape)
print(df.head())

# null counts per col
print(df.isna().sum())

# rows by cat and year
print(df.groupby(["category", "year"]).size())

# gpus missing tdp
gpus = df[df["category"] == "GPU"]
print(gpus[gpus["tdp"].isna()].shape)
print(gpus[gpus["year"] == 2025]["tdp"].isna().sum())

# rows where brand didnt get inferred
null_brand = df[df["brand"].isna()]
print(null_brand["model"].head(30).tolist())

# 2025 brand discovery - first word frequency for unbranded rows. used this to build brand_keywords
null_brands = df[(df["year"] == 2025) & df["brand"].isna()].copy()
null_brands["first_word"] = null_brands["model"].str.split().str[0].str.lower()
print(null_brands["first_word"].value_counts().head(50).to_string())

# storage type / interface / pcie counts in cleaned data
clean = pd.read_csv("cleaned_data/combined_parts_cleaned.csv")
print(clean["storage_type"].value_counts(dropna=False))
print(clean["interface_type"].value_counts(dropna=False))
print(clean["pcie_gen"].value_counts(dropna=False))

# look up rtx 4070 specs
rtx = clean[clean["model"].str.contains("rtx 4070", na=False)]
print(rtx[["model", "year", "price", "memory_gb", "tdp_watts"]])

# top 15 most expensive parts
print(clean.sort_values("price", ascending=False).head(15)[["model","category","year","price"]])

# iqr per category
for cat in ["GPU","CPU","RAM","Storage"]:
    p = clean[clean["category"]==cat]["price"]
    print(cat, p.quantile(0.25), p.quantile(0.75))

# synthetic data shape and date range
hist = pd.read_csv("cleaned_data/price_history.csv")
print(hist.shape)
print(hist["component_id"].nunique())
print(hist["date"].min(), hist["date"].max())

# component counts by category from db
conn = sqlite3.connect("pcparts.db")
print(pd.read_sql_query("select category, count(*) from components group by category", conn))
conn.close()

# 4070 synthetic prices from db
conn = sqlite3.connect("pcparts.db")
q = "select * from synthetic_prices where component_id like '%4070%' limit 20"
print(pd.read_sql_query(q, conn))
conn.close()
