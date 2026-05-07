import sqlite3
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


train_cutoff = 24
forecast_months = 12

feature_cols = [
    "time_step", "sin_month", "cos_month", "event_active",
    "price_2023", "price_2025", "trend_per_step", "expected_price",
    "spec_value",
    "cat_CPU", "cat_GPU", "cat_RAM", "cat_Storage",
]

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, random_state=42, n_jobs=-1),
}


# pulls the price history and anchors out of sqlite
# anchors are used later to build the expected_price feature

def load_data():

    conn = sqlite3.connect("pcparts.db")

    df = pd.read_sql_query("""
        SELECT
            s.component_id,
            s.date,
            s.synthetic_price,
            s.event_label,
            s.data_source,
            c.category,
            c.memory_gb,
            c.core_count,
            c.capacity_gb,
            c.module_count,
            c.module_size_gb
        FROM synthetic_prices s
        JOIN components c ON s.component_id = c.component_id
        ORDER BY s.component_id, s.date
    """, conn)

    obs = pd.read_sql_query("""
        SELECT component_id, year, observed_price_median
        FROM observed_prices
        WHERE year IN (2023, 2025)
    """, conn)

    conn.close()

    anchors = (
        obs.pivot(index="component_id", columns="year", values="observed_price_median")
        .reset_index()
    )
    anchors.columns.name = None
    anchors = anchors.rename(columns={2023: "price_2023", 2025: "price_2025"})
    return df, anchors


# adds the model's input features to the dataframe
# time step, sin/cos month, event flag, anchors, expected_price baseline, spec_value, category one-hots

def build_features(df, anchors):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["time_step"] = (df["date"].dt.year - 2023) * 12 + (df["date"].dt.month - 1)
    df["month"] = df["date"].dt.month

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["event_active"] = (df["event_label"].fillna("") != "").astype(int)

    cat_dummies = pd.get_dummies(df["category"], prefix="cat", dtype=float)
    df = pd.concat([df, cat_dummies], axis=1)

    for col in ["cat_CPU", "cat_GPU", "cat_RAM", "cat_Storage"]:

        if col not in df.columns:
            df[col] = 0.0

    ram_gb = df["module_count"].fillna(0) * df["module_size_gb"].fillna(0)

    df["spec_value"] = np.select(
        [df["category"] == "GPU",
         df["category"] == "CPU",
         df["category"] == "Storage"],
        [df["memory_gb"].fillna(0),
         df["core_count"].fillna(0),
         df["capacity_gb"].fillna(0)],
        default=ram_gb,
    )

    df = df.merge(anchors, on="component_id", how="left")

    df["trend_per_step"] = (df["price_2025"] - df["price_2023"]) / 28
    df["expected_price"] = df["price_2023"] + df["time_step"] * df["trend_per_step"]

    return df


# mean absolute percentage error
# easier to read across categories than raw mae since gpu and ram have very different price ranges

def mape(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0

    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


# splits the test predictions back out by category
# so i can see if the model does great on gpu but worse on ram

def category_breakdown(model_name, test, y_test, preds):

    rows = []

    for cat in ["CPU", "GPU", "RAM", "Storage"]:

        mask = (test["category"] == cat).values

        if mask.sum() < 5:
            continue

        mae = mean_absolute_error(y_test[mask], preds[mask])
        rmse = mean_squared_error(y_test[mask], preds[mask]) ** 0.5
        r2 = r2_score(y_test[mask], preds[mask])
        m = mape(y_test[mask], preds[mask])

        rows.append({
            "model": model_name,
            "category": cat,
            "n_test": int(mask.sum()),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE_pct": round(m, 2),
            "R2": round(r2, 4),
        })

    return rows


# the main train/test split
# trains on jan 2023 through dec 2024 and tests on jan 2025 through may 2025
# also runs the naive baseline for an honesty check

def split_and_eval(df):

    required = feature_cols + ["synthetic_price"]
    df_clean = df.dropna(subset=required)

    train = df_clean[df_clean["time_step"] < train_cutoff]
    test = df_clean[df_clean["time_step"] >= train_cutoff]

    X_train = train[feature_cols].values
    y_train = train["synthetic_price"].values
    X_test = test[feature_cols].values
    y_test = test["synthetic_price"].values

    print(f"\nTrain: {len(X_train):,} rows (Jan 2023 - Dec 2024, {len(train['component_id'].unique()):,} components)")
    print(f"Test: {len(X_test):,} rows (Jan 2025 - May 2025, {len(test['component_id'].unique()):,} components)")

    metrics = {}
    per_cat_rows = []
    pred_by_model = {}
    print()

    naive_pred = np.clip(test["expected_price"].values, 1.0, None)
    nm = mean_absolute_error(y_test, naive_pred)
    nr = mean_squared_error(y_test, naive_pred) ** 0.5
    nrr = r2_score(y_test, naive_pred)
    n_mape = mape(y_test, naive_pred)

    metrics["Naive (expected_price)"] = {
        "MAE": round(nm, 2), "RMSE": round(nr, 2),
        "MAPE_pct": round(n_mape, 2), "R2": round(nrr, 4),
    }
    print(f"Naive (expected_price): MAE={nm:.2f}, RMSE={nr:.2f}, MAPE={n_mape:.2f}%, R2={nrr:.4f}")

    per_cat_rows.extend(category_breakdown("Naive (expected_price)", test, y_test, naive_pred))
    pred_by_model["Naive (expected_price)"] = naive_pred

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1.0, None)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        m = mape(y_test, preds)

        metrics[name] = {
            "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE_pct": round(m, 2), "R2": round(r2, 4),
        }
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={m:.2f}%, R2={r2:.4f}")

        per_cat_rows.extend(category_breakdown(name, test, y_test, preds))
        pred_by_model[name] = preds

    real_models = {k: v for k, v in metrics.items() if k != "Naive (expected_price)"}
    best_name = max(real_models, key=lambda k: real_models[k]["R2"])

    naive_mae = metrics["Naive (expected_price)"]["MAE"]
    best_mae = metrics[best_name]["MAE"]
    lift = naive_mae - best_mae

    print(f"\nBest ML model: {best_name} (R2={metrics[best_name]['R2']})")
    print(f"  MAE lift over naive baseline: ${lift:.2f} ({lift / naive_mae * 100:.1f}%)")

    return metrics, best_name, per_cat_rows, test, pred_by_model


# timeseriessplit cross validation so the test set is always after the train set
# regular k-fold would let the model peek into the future which is not ok for prices

def cv_eval(df):

    df_clean = df.dropna(subset=feature_cols + ["synthetic_price"])
    df_clean = df_clean.sort_values(["time_step", "component_id"])

    X = df_clean[feature_cols].values
    y = df_clean["synthetic_price"].values

    tscv = TimeSeriesSplit(n_splits=5)

    print("\n5-fold TimeSeriesSplit CV (per-fold MAE):")

    for name, model in models.items():

        maes = []

        for train_idx, test_idx in tscv.split(X):

            model.fit(X[train_idx], y[train_idx])
            preds = np.clip(model.predict(X[test_idx]), 1.0, None)
            maes.append(mean_absolute_error(y[test_idx], preds))

        fold_str = ", ".join(f"${m:.2f}" for m in maes)
        print(f"  {name}: [{fold_str}]  mean=${np.mean(maes):.2f}  std=${np.std(maes):.2f}")


# projects 12 months forward from today using the best model
# clamps predictions so tree models don't go off the rails when extrapolating

def forecast_future(df, best_name):

    df_clean = df.dropna(subset=feature_cols + ["synthetic_price"])
    X_full = df_clean[feature_cols].values
    y_full = df_clean["synthetic_price"].values

    model = models[best_name]
    model.fit(X_full, y_full)

    per_component = (
        df_clean[[
            "component_id", "category",
            "price_2023", "price_2025", "trend_per_step",
            "spec_value",
            "cat_CPU", "cat_GPU", "cat_RAM", "cat_Storage",
        ]]
        .drop_duplicates("component_id")
        .reset_index(drop=True)
    )

    forecast_start = (pd.Timestamp.today().to_period("M") + 1).to_timestamp()
    future_dates = pd.date_range(forecast_start, periods=forecast_months, freq="MS")
    chunks = []

    for d in future_dates:

        t = (d.year - 2023) * 12 + (d.month - 1)
        month = d.month

        chunk = per_component.copy()
        chunk["time_step"] = t
        chunk["sin_month"] = np.sin(2 * np.pi * month / 12)
        chunk["cos_month"] = np.cos(2 * np.pi * month / 12)
        chunk["event_active"] = 0
        chunk["expected_price"] = chunk["price_2023"] + t * chunk["trend_per_step"]
        chunk["forecast_month"] = d.strftime("%Y-%m-%d")
        chunks.append(chunk)

    future_df = pd.concat(chunks, ignore_index=True).dropna(subset=feature_cols)

    raw_preds = model.predict(future_df[feature_cols].values)

    p25 = future_df["price_2025"].values
    p23 = future_df["price_2023"].values
    higher_anchor = np.maximum(p23, p25)

    low_bound = np.maximum(1.0, 0.20 * p25)
    high_bound = 2.0 * higher_anchor

    preds = np.clip(raw_preds, low_bound, high_bound)
    future_df["predicted_price"] = np.round(preds, 2)

    n_floored = int((preds == low_bound).sum())
    n_capped = int((preds == high_bound).sum())

    if n_floored or n_capped:
        print(f"  forecast clamps applied: {n_floored} rows floored at 20% of 2025 anchor, {n_capped} rows capped at 2x higher anchor")

    return future_df[["component_id", "category", "forecast_month", "predicted_price"]]


# prints the cheapest and priciest forecast month per category
# this is what feeds the cli's buy now or wait recommendation

def print_forecast_summary(forecast_df):

    monthly_avg = (
        forecast_df.groupby(["category", "forecast_month"])["predicted_price"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"predicted_price": "avg_price"})
    )

    months = sorted(monthly_avg["forecast_month"].unique())
    span_label = f"{pd.Timestamp(months[0]).strftime('%b %Y')} - {pd.Timestamp(months[-1]).strftime('%b %Y')}"
    print(f"\nAverage predicted price by category ({span_label}):")
    overall = monthly_avg.groupby("category")["avg_price"].mean().round(2)

    for cat, price in overall.items():
        print(f"  {cat}: ${price:.2f}")

    print("\nCheapest projected month by category:")
    cheapest = monthly_avg.loc[monthly_avg.groupby("category")["avg_price"].idxmin()]

    for _, row in cheapest.iterrows():
        print(f"  {row['category']}: {row['forecast_month']} avg ${row['avg_price']:.2f}")

    print("\nMost expensive projected month by category:")
    priciest = monthly_avg.loc[monthly_avg.groupby("category")["avg_price"].idxmax()]

    for _, row in priciest.iterrows():
        print(f"  {row['category']}: {row['forecast_month']} avg ${row['avg_price']:.2f}")


# prints the tree-based feature importance for the best model

def print_feature_importance(best_name):

    model = models[best_name]

    if hasattr(model, "feature_importances_"):

        imps = sorted(zip(feature_cols, model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
        print(f"\nTop feature importances ({best_name}):")

        for feat, imp in imps[:8]:
            print(f"  {feat}: {imp:.4f}")

    elif hasattr(model, "coef_"):

        coefs = sorted(zip(feature_cols, model.coef_),
                       key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop feature coefficients ({best_name}):")

        for feat, coef in coefs[:8]:
            print(f"  {feat}: {coef:.4f}")


# permutation importance shuffles each feature one at a time and measures the error change
# this is how i caught that expected_price was doing most of the work

def print_permutation_importance(best_name, X_test, y_test):

    model = models[best_name]

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error",
    )

    order = np.argsort(result.importances_mean)[::-1]

    print(f"\nPermutation feature importance ({best_name}, drop in -MAE when feature is shuffled):")

    rows = []

    for idx in order[:8]:

        feat = feature_cols[idx]
        mean = result.importances_mean[idx]
        std = result.importances_std[idx]
        print(f"  {feat}: {mean:.4f} +/- {std:.4f}")

        rows.append({"feature": feat, "perm_importance_mean": round(float(mean), 6),
                     "perm_importance_std": round(float(std), 6)})

    pd.DataFrame(rows).to_csv("cleaned_data/feature_importance.csv", index=False)
    print("Saved: cleaned_data/feature_importance.csv")


# finds the components the model got the most wrong on
# usually exotic high-end skus with thin training data

def hardest_predictions(test, preds, best_name):

    df = test[["component_id", "category", "synthetic_price", "date"]].copy()
    df["predicted"] = preds
    df["abs_error"] = (df["synthetic_price"] - df["predicted"]).abs()

    per_comp = (
        df.groupby(["component_id", "category"])
        .agg(mean_actual=("synthetic_price", "mean"),
             mean_pred=("predicted", "mean"),
             mean_abs_error=("abs_error", "mean"))
        .reset_index()
    )

    components = pd.read_csv("cleaned_data/combined_parts_cleaned.csv",
                             usecols=["component_id", "model"]).drop_duplicates("component_id")
    per_comp = per_comp.merge(components, on="component_id", how="left")

    per_comp = per_comp.sort_values("mean_abs_error", ascending=False)
    hardest = per_comp.head(10).reset_index(drop=True)

    out_cols = ["component_id", "model", "category", "mean_actual", "mean_pred", "mean_abs_error"]
    hardest = hardest[out_cols].copy()
    hardest["mean_actual"] = hardest["mean_actual"].round(2)
    hardest["mean_pred"] = hardest["mean_pred"].round(2)
    hardest["mean_abs_error"] = hardest["mean_abs_error"].round(2)

    print(f"\nTop 10 hardest-to-predict components in test ({best_name}):")

    for _, row in hardest.iterrows():
        name = str(row["model"])[:50]
        print(f"  [{row['category']}] {name}  actual=${row['mean_actual']:.2f}  pred=${row['mean_pred']:.2f}  err=${row['mean_abs_error']:.2f}")

    hardest.to_csv("cleaned_data/hardest_predictions.csv", index=False)
    print("Saved: cleaned_data/hardest_predictions.csv")


# the honest evaluation
# trains on synthetic + early real_blended rows
# tests on the last 3 months of real prices for the given category
# this is the credible accuracy claim because the test labels are real prices the model never saw

def real_data_backtest(df, category):

    if "data_source" not in df.columns:
        print(f"\nReal-{category} backtest: data_source column missing; skipping")
        return

    real_rows = df[(df["data_source"] == "real_blended") & (df["category"] == category)].copy()

    if real_rows.empty:
        print(f"\nReal-{category} backtest: no real-blended {category} rows in dataset; skipping")
        return

    real_rows = real_rows.dropna(subset=feature_cols + ["synthetic_price"])

    if real_rows.empty:
        return

    months = sorted(real_rows["date"].unique())
    n_months = len(months)

    if n_months < 4:
        print(f"\nReal-{category} backtest: only {n_months} months of real data; skipping")
        return

    holdout_n = max(1, n_months // 3)
    holdout_months = set(months[-holdout_n:])

    train = df.dropna(subset=feature_cols + ["synthetic_price"])
    train = train[~train["date"].isin(holdout_months)]

    holdout = real_rows[real_rows["date"].isin(holdout_months)]

    print(f"\nReal-{category} backtest:")
    print(f"train:{len(train):,} rows (all categories, excluding {category} real months {sorted([pd.Timestamp(m).strftime('%Y-%m') for m in holdout_months])})")
    print(f"test:{len(holdout):,} real-blended {category} rows from held-out months")

    X_train = train[feature_cols].values
    y_train = train["synthetic_price"].values
    X_test = holdout[feature_cols].values
    y_test = holdout["synthetic_price"].values

    rows = []

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1.0, None)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        m = mape(y_test, preds)

        rows.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(m, 2), "R2": round(r2, 4), "n_test": len(y_test)})
        print(f"  {name}: MAE=${mae:.2f}, RMSE=${rmse:.2f}, MAPE={m:.2f}%, R2={r2:.4f}")

    out_path = f"cleaned_data/real_{category.lower()}_backtest_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# trains all 3 regressors, evaluates them, runs the real-data backtests
# then writes the 12-month forecast to future_price_predictions.csv

def main():

    df_raw, anchors = load_data()
    print(f"{len(df_raw):,} synthetic price rows, {df_raw['component_id'].nunique():,} components")

    df = build_features(df_raw, anchors)

    metrics, best_name, per_cat_rows, test, pred_by_model = split_and_eval(df)

    metrics_df = (
        pd.DataFrame(metrics)
        .T.reset_index()
        .rename(columns={"index": "model"})
    )
    metrics_df.to_csv("cleaned_data/model_metrics.csv", index=False)
    print("\nSaved: cleaned_data/model_metrics.csv")

    per_cat_df = pd.DataFrame(per_cat_rows)
    per_cat_df.to_csv("cleaned_data/model_metrics_per_category.csv", index=False)
    print("Saved: cleaned_data/model_metrics_per_category.csv")

    print("\nPer-category MAE:")

    pivot = per_cat_df.pivot(index="model", columns="category", values="MAE")
    print(pivot.to_string())

    cv_eval(df)

    print_feature_importance(best_name)

    X_test = test[feature_cols].values
    y_test = test["synthetic_price"].values
    print_permutation_importance(best_name, X_test, y_test)

    hardest_predictions(test, pred_by_model[best_name], best_name)

    real_data_backtest(df, "GPU")
    real_data_backtest(df, "Storage")
    real_data_backtest(df, "RAM")

    fc_start = (pd.Timestamp.today().to_period("M") + 1).to_timestamp()
    fc_end = fc_start + pd.DateOffset(months=forecast_months - 1)
    print(f"\nforecasting {fc_start.strftime('%b %Y')} - {fc_end.strftime('%b %Y')} using {best_name}")
    forecast_df = forecast_future(df, best_name)
    print(f"  {len(forecast_df):,} rows  ({forecast_df['component_id'].nunique():,} components x {forecast_months} months)")

    print_forecast_summary(forecast_df)

    forecast_df.to_csv("cleaned_data/future_price_predictions.csv", index=False)
    print("\nSaved: cleaned_data/future_price_predictions.csv")

    print(f"\nBest model: {best_name}")
    print(f"  MAE: ${metrics[best_name]['MAE']:.2f}")
    print(f"  RMSE: ${metrics[best_name]['RMSE']:.2f}")
    print(f"  R2: {metrics[best_name]['R2']:.4f}")
