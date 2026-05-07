import sqlite3
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


spec_features = {
    "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
    "CPU": ["core_count", "tdp_watts"],
    "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
    "Storage": ["capacity_gb"],
}


# mean absolute percentage error - same metric used in model_training

def mape(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0

    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


# pulls components and their real observed prices from sqlite

def load_real_prices():

    conn = sqlite3.connect("pcparts.db")

    df = pd.read_sql_query("""
        SELECT c.component_id, c.category,
               c.memory_gb, c.tdp_watts, c.core_clock_mhz,
               c.core_count,
               c.ddr_gen, c.speed_mhz, c.module_count, c.module_size_gb,
               c.capacity_gb,
               o.year, o.observed_price_median AS price
        FROM components c
        JOIN observed_prices o ON c.component_id = o.component_id
    """, conn)

    conn.close()

    return df


# cross-sectional regression on real prices
# trains on 2023 component+price pairs and tests on 2025 pairs
# more honest than the synthetic split because both train and test are real

def main():

    df = load_real_prices()
    print(f"Loaded {len(df):,} real component-year price observations")

    rows = []
    print()

    for cat, feats in spec_features.items():

        sub = df[df["category"] == cat].dropna(subset=feats + ["price"]).copy()

        train = sub[sub["year"] == 2023]
        test = sub[sub["year"] == 2025]

        if len(train) < 30 or len(test) < 30:
            print(f"  {cat}: skipping (train={len(train)}, test={len(test)})")
            continue

        X_train = train[feats].values
        y_train = train["price"].values
        X_test = test[feats].values
        y_test = test["price"].values

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        }

        print(f"{cat}  (train 2023={len(train):,}, test 2025={len(test):,}, features={feats}):")

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), 1.0, None)

            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            m = mape(y_test, preds)

            rows.append({
                "category": cat,
                "model": name,
                "n_train_2023": len(train),
                "n_test_2025": len(test),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE_pct": round(m, 2),
                "R2": round(r2, 4),
            })

            print(f"  {name}: MAE=${mae:.2f}, RMSE=${rmse:.2f}, MAPE={m:.2f}%, R2={r2:.4f}")

        print()

    out = pd.DataFrame(rows)
    out.to_csv("cleaned_data/spec_regression_metrics.csv", index=False)
    print("Saved: cleaned_data/spec_regression_metrics.csv")

    print("\nFraming: this is a cross-sectional regression on REAL observed prices.")
    print("Train on 2023 component+price pairs, test on held-out 2025 component+price pairs.")
    print("Compare these numbers with model_metrics.csv (synthetic time-series regression on the SAME features in different framings).")
