import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, redirect, url_for

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(root, "pcparts.db")
cleaned_dir = os.path.join(root, "cleaned_data")
visuals_dir = os.path.join(root, "visuals")
forecast_path = os.path.join(cleaned_dir, "future_price_predictions.csv")

spec_features = {
    "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
    "CPU": ["core_count", "tdp_watts"],
    "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
    "Storage": ["capacity_gb"],
}

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))


@app.context_processor
def inject_globals():
    return {"deployed": os.environ.get("DEPLOYED") == "1"}


# creates the user_observations and prediction_adjustments tables on first run
# same schema the cli used so existing calibration state carries over

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


# fuzzy substring search on component_id, optionally filter to ones with a forecast row

def find_components(query, require_forecast=False, category=None, limit=50):

    conn = sqlite3.connect(db_path)

    sql = """
        SELECT component_id, category, brand, model
        FROM components
        WHERE LOWER(component_id) LIKE ?
    """
    params = [f"%{query.lower()}%"]

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY category, model LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    if require_forecast and not df.empty and os.path.exists(forecast_path):
        forecast_ids = set(pd.read_csv(forecast_path)["component_id"].unique())
        df = df[df["component_id"].isin(forecast_ids)].reset_index(drop=True)

    return df


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


# monthly synthetic + real_blended price history for one component

def get_synth(component_id):

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        "SELECT date, synthetic_price, data_source FROM synthetic_prices WHERE component_id = ? ORDER BY date",
        conn, params=[component_id]
    )

    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df


# 12-month forecast rows for one component

def get_forecast(component_id):

    if not os.path.exists(forecast_path):
        return pd.DataFrame()

    df = pd.read_csv(forecast_path)
    df = df[df["component_id"] == component_id].copy()

    if df.empty:
        return df

    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    df = df.rename(columns={"forecast_month": "date", "predicted_price": "price"})
    return df[["date", "price"]].sort_values("date").reset_index(drop=True)


# 2023 + 2025 real pcpartpicker observed prices for one component

def get_observed(component_id):

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        "SELECT year, observed_price_median FROM observed_prices WHERE component_id = ? ORDER BY year",
        conn, params=[component_id]
    )

    conn.close()
    return df


# one row per component with the full spec row used everywhere

def get_component_row(component_id):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM components WHERE component_id = ?", conn, params=[component_id])
    conn.close()

    if df.empty:
        return None

    return df.iloc[0].to_dict()


# ============================================================
# routes
# ============================================================


# home / project summary - replaces cli option 1

@app.route("/")
def summary():

    conn = sqlite3.connect(db_path)

    n_components = conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    n_observed = conn.execute("SELECT COUNT(*) FROM observed_prices").fetchone()[0]
    n_synthetic = conn.execute("SELECT COUNT(*) FROM synthetic_prices").fetchone()[0]
    n_real_blended = conn.execute("SELECT COUNT(*) FROM synthetic_prices WHERE data_source = 'real_blended'").fetchone()[0]
    n_user_obs = conn.execute("SELECT COUNT(*) FROM user_observations").fetchone()[0]
    n_adj = conn.execute("SELECT COUNT(*) FROM prediction_adjustments").fetchone()[0]

    by_cat = pd.read_sql_query(
        "SELECT category, COUNT(*) AS n FROM components GROUP BY category ORDER BY category",
        conn,
    )

    conn.close()

    counts = {
        "components": n_components,
        "observed": n_observed,
        "synthetic": n_synthetic,
        "real_blended": n_real_blended,
        "user_obs": n_user_obs,
        "calibrations": n_adj,
    }

    by_cat_rows = by_cat.to_dict("records")

    reg = pd.read_csv(os.path.join(cleaned_dir, "model_metrics.csv")).to_dict("records")

    cls_path = os.path.join(cleaned_dir, "classification_metrics.csv")
    cls_rf = []

    if os.path.exists(cls_path):
        cls = pd.read_csv(cls_path)
        rf = cls[cls["model"] == "Random Forest"][["category", "accuracy", "macro_f1"]]
        cls_rf = rf.to_dict("records")

    real_backtest = []

    for cat, fname in [("GPU", "real_gpu_backtest_metrics"),
                       ("Storage", "real_storage_backtest_metrics"),
                       ("RAM", "real_ram_backtest_metrics")]:

        path = os.path.join(cleaned_dir, f"{fname}.csv")

        if not os.path.exists(path):
            continue

        sub = pd.read_csv(path)
        rf_row = sub[sub["model"] == "Random Forest"]

        if rf_row.empty:
            continue

        row = rf_row.iloc[0].to_dict()
        row["category"] = cat
        real_backtest.append(row)

    return render_template("summary.html",
                           counts=counts,
                           by_cat=by_cat_rows,
                           regression=reg,
                           classification=cls_rf,
                           real_backtest=real_backtest)


# component browser - search bar + result table - replaces cli option 2 entry

@app.route("/components")
def components():

    q = request.args.get("q", "").strip()
    cat = request.args.get("cat", "").strip() or None
    results = []

    if q:
        matches = find_components(q, category=cat, limit=200)
        results = matches.to_dict("records")

    return render_template("components.html", q=q, cat=cat, results=results)


# fast json endpoint used by autocomplete-style inputs (build planner)

@app.route("/api/search")
def api_search():

    q = request.args.get("q", "").strip()
    cat = request.args.get("cat", "").strip() or None
    require_forecast = request.args.get("forecast", "0") == "1"

    if not q:
        return jsonify([])

    matches = find_components(q, require_forecast=require_forecast, category=cat, limit=25)
    return jsonify(matches.to_dict("records"))


# one component detail page - combines cli options 2, 3, 4, 12

@app.route("/components/<path:component_id>")
def component_detail(component_id):

    row = get_component_row(component_id)

    if row is None:
        abort(404)

    syn = get_synth(component_id)
    fc = get_forecast(component_id)
    obs = get_observed(component_id)
    mult, n_obs = get_adjustment(component_id)

    # build chart series (history + forecast in one chart)

    history_series = []

    if not syn.empty:
        adjusted = syn.copy()
        adjusted["synthetic_price"] = adjusted["synthetic_price"] * mult
        history_series = [
            {"date": d.strftime("%Y-%m-%d"), "price": float(p), "source": s}
            for d, p, s in zip(adjusted["date"], adjusted["synthetic_price"], adjusted["data_source"])
        ]

    forecast_series = []

    if not fc.empty:
        fc = fc.copy()
        fc["price"] = fc["price"] * mult
        forecast_series = [
            {"date": d.strftime("%Y-%m-%d"), "price": float(p)}
            for d, p in zip(fc["date"], fc["price"])
        ]

    # recommendation

    recommendation = None

    if not fc.empty:

        today = pd.Timestamp.today().normalize()
        this_month = today.to_period("M")
        future = fc[fc["date"].dt.to_period("M") >= this_month]

        if future.empty:
            future = fc.tail(1)

        current = future.iloc[0]
        cheapest = future.loc[future["price"].idxmin()]
        priciest = future.loc[future["price"].idxmax()]
        savings = float(current["price"] - cheapest["price"])
        pct = round((savings / float(current["price"]) * 100), 1) if current["price"] > 0 else 0.0
        months_to_cheapest = (cheapest["date"].to_period("M") - current["date"].to_period("M")).n

        hist_min = None
        hist_context = None

        if not syn.empty:
            hist_min = float((syn["synthetic_price"] * mult).min())

            if cheapest["price"] <= hist_min * 1.02:
                hist_context = f"Forecast cheapest is at or below the historical low of ${hist_min:.2f} - strong buying signal."
            elif cheapest["price"] <= hist_min * 1.10:
                hist_context = f"Forecast cheapest is within 10% of the historical low of ${hist_min:.2f}."
            else:
                hist_context = f"Forecast cheapest is ${float(cheapest['price']) - hist_min:.2f} above the historical low of ${hist_min:.2f}."

        if months_to_cheapest <= 1:
            verdict = "buy"
            verdict_text = "BUY NOW"
            verdict_reason = "current month is at or near the projected minimum."
        elif pct >= 8:
            verdict = "wait"
            verdict_text = f"WAIT until {cheapest['date'].strftime('%B %Y')}"
            verdict_reason = f"projected savings of ${savings:.2f} ({pct:.1f}%)."
        elif pct >= 3:
            verdict = "marginal"
            verdict_text = "MARGINAL WAIT"
            verdict_reason = f"savings of {pct:.1f}% projected, weigh against opportunity cost."
        else:
            verdict = "buy"
            verdict_text = "BUY NOW"
            verdict_reason = f"projected savings of {pct:.1f}% are too small to justify waiting."

        recommendation = {
            "verdict": verdict,
            "verdict_text": verdict_text,
            "verdict_reason": verdict_reason,
            "current_date": current["date"].strftime("%B %Y"),
            "current_price": float(current["price"]),
            "cheapest_date": cheapest["date"].strftime("%B %Y"),
            "cheapest_price": float(cheapest["price"]),
            "priciest_date": priciest["date"].strftime("%B %Y"),
            "priciest_price": float(priciest["price"]),
            "savings": savings,
            "pct_savings": pct,
            "hist_min": hist_min,
            "hist_context": hist_context,
        }

    # similar components in spec space

    similar = []
    cat = row["category"]
    feats = spec_features.get(cat)

    if feats is not None:

        conn = sqlite3.connect(db_path)

        cat_df = pd.read_sql_query("""
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

        if len(cat_df) >= 6 and component_id in cat_df["component_id"].values:

            X = StandardScaler().fit_transform(cat_df[feats].values)
            k = min(6, len(cat_df))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X)

            target_pos = cat_df.index[cat_df["component_id"] == component_id][0]
            distances, indices = nn.kneighbors(X[target_pos].reshape(1, -1))

            for dist, pos in zip(distances[0][1:], indices[0][1:]):

                nbr = cat_df.iloc[pos]
                spec_str = ", ".join(f"{f}={nbr[f]}" for f in feats if pd.notna(nbr[f]))
                similar.append({
                    "component_id": nbr["component_id"],
                    "model": nbr["model"],
                    "distance": float(dist),
                    "specs": spec_str,
                    "price_2025": float(nbr["price_2025"]) if pd.notna(nbr["price_2025"]) else None,
                })

    observed_rows = obs.to_dict("records")

    # spec display rows per category
    spec_display = []
    spec_labels = {
        "memory_gb": ("VRAM", "GB"),
        "tdp_watts": ("TDP", "W"),
        "core_clock_mhz": ("Core clock", "MHz"),
        "core_count": ("Cores", ""),
        "microarchitecture": ("Microarchitecture", ""),
        "ddr_gen": ("Generation", ""),
        "speed_mhz": ("Speed", "MHz"),
        "module_count": ("Modules", ""),
        "module_size_gb": ("Module size", "GB"),
        "capacity_gb": ("Capacity", "GB"),
        "storage_type": ("Type", ""),
        "interface_type": ("Interface", ""),
        "pcie_gen": ("PCIe gen", ""),
    }

    cat_spec_order = {
        "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
        "CPU": ["core_count", "tdp_watts", "microarchitecture"],
        "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
        "Storage": ["capacity_gb", "storage_type", "interface_type", "pcie_gen"],
    }

    for key in cat_spec_order.get(row["category"], []):
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        label, unit = spec_labels.get(key, (key, ""))
        if isinstance(v, float) and v.is_integer():
            v_str = f"{int(v)}"
        else:
            v_str = str(v)
        if key == "ddr_gen":
            v_str = f"DDR{int(float(v))}"
        spec_display.append({"label": label, "value": v_str, "unit": unit})

    has_prediction = bool(history_series or forecast_series)

    return render_template("component.html",
                           component=row,
                           calibration={"multiplier": mult, "n_observations": n_obs},
                           history_series=history_series,
                           forecast_series=forecast_series,
                           observed=observed_rows,
                           recommendation=recommendation,
                           similar=similar,
                           specs=spec_display,
                           has_prediction=has_prediction)


# POST handler for option 4 - submit a real observation

@app.route("/components/<path:component_id>/observe", methods=["POST"])
def submit_observation(component_id):

    row = get_component_row(component_id)

    if row is None:
        abort(404)

    try:
        price = float(request.form.get("price", ""))

        if price <= 0:
            return redirect(url_for("component_detail", component_id=component_id) + "?err=bad_price")

    except ValueError:
        return redirect(url_for("component_detail", component_id=component_id) + "?err=bad_price")

    date_str = request.form.get("date", "").strip()

    if not date_str:
        date_str = datetime.today().strftime("%Y-%m-%d")

    try:
        obs_date = pd.Timestamp(date_str)

    except Exception:
        return redirect(url_for("component_detail", component_id=component_id) + "?err=bad_date")

    syn = get_synth(component_id)
    fc = get_forecast(component_id)

    raw_pred = None

    if not syn.empty:

        nearest = (syn["date"] - obs_date).abs().idxmin()
        days_off = abs((syn.loc[nearest, "date"] - obs_date).days)

        if days_off <= 31:
            raw_pred = float(syn.loc[nearest, "synthetic_price"])

    if raw_pred is None and not fc.empty:

        nearest = (fc["date"] - obs_date).abs().idxmin()
        days_off = abs((fc.loc[nearest, "date"] - obs_date).days)

        if days_off <= 31:
            raw_pred = float(fc.loc[nearest, "price"])

    if raw_pred is None or raw_pred <= 0:
        return redirect(url_for("component_detail", component_id=component_id) + "?err=no_pred")

    ratio = price / raw_pred

    conn = sqlite3.connect(db_path)

    conn.execute("""
        INSERT INTO user_observations (component_id, observation_date, observed_price, predicted_price, ratio, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [component_id, date_str, price, round(raw_pred, 4), round(ratio, 6), datetime.now().isoformat()])

    rows = conn.execute("SELECT ratio FROM user_observations WHERE component_id = ?", [component_id]).fetchall()
    n = len(rows)
    new_mult = float(np.mean([r[0] for r in rows]))

    conn.execute("""
        INSERT INTO prediction_adjustments (component_id, multiplier, n_observations, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(component_id) DO UPDATE SET
            multiplier = excluded.multiplier,
            n_observations = excluded.n_observations,
            last_updated = excluded.last_updated
    """, [component_id, round(new_mult, 6), n, datetime.now().isoformat()])

    conn.commit()
    conn.close()

    return redirect(url_for("component_detail", component_id=component_id) + "?ok=1")


# observation history and active calibrations - cli options 5 + 6

@app.route("/observations")
def observations():

    conn = sqlite3.connect(db_path)

    obs = pd.read_sql_query("""
        SELECT id, component_id, observation_date, observed_price, predicted_price, ratio, recorded_at
        FROM user_observations
        ORDER BY recorded_at DESC
        LIMIT 100
    """, conn)

    adj = pd.read_sql_query("""
        SELECT component_id, multiplier, n_observations, last_updated
        FROM prediction_adjustments
        ORDER BY last_updated DESC
    """, conn)

    conn.close()

    return render_template("observations.html",
                           observations=obs.to_dict("records"),
                           calibrations=adj.to_dict("records"))


# POST handler for option 6 - reset
# disabled when DEPLOYED=1 so randos on a public url cant wipe everyones calibration

@app.route("/observations/reset", methods=["POST"])
def reset_observations():

    if os.environ.get("DEPLOYED") == "1":
        abort(403)

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM user_observations")
    conn.execute("DELETE FROM prediction_adjustments")
    conn.commit()
    conn.close()
    return redirect(url_for("observations"))


# pipeline stages + data lineage - cli options 7 + 8

@app.route("/pipeline")
def pipeline():

    stages = [
        ("1", "data_collection.py", "Merges 2023 + 2025 PCPartPicker CSVs, then ingests real daily prices from HardwareDealsCo for GPU, SSD, and RAM."),
        ("2", "data_cleaning.py", "IQR caps, spec parsing, component_id construction, cross-year matching."),
        ("3", "generate_synthetic.py", "29-month synthetic series with 9 encoded events, then bridges real bucket prices into per-SKU real_blended rows."),
        ("4", "database_setup.py", "SQLite schema with 6 tables + 5 sample queries."),
        ("5", "model_training.py", "LR / DT / RF + naive baseline + TimeSeriesSplit CV + permutation importance, then 3 real-data backtests (GPU / Storage / RAM)."),
        ("6", "estimator.py", "Budget / mid / high tier build cost forecast + k-means tier discovery."),
        ("7", "value_metrics.py", "$/GB VRAM, $/core, $/GB metrics on real observed prices."),
        ("8", "spec_classifier.py", "Multi-class spec-to-tier classifier with stratified CV + calibration curves."),
        ("9", "spec_regression.py", "Cross-sectional regression on real prices: train on 2023, test on 2025."),
        ("10", "data_visualization.py", "Plots in visuals/, with annotated event windows and a synthetic-vs-real GPU overlay."),
    ]

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

    lineage = {
        "components": n_comp,
        "observed": n_obs,
        "synthetic_total": n_syn,
        "matched_components": matched,
        "real_blended_total": n_real_blended,
        "real_blended_by_cat": real_blended_by_cat.to_dict("records"),
        "gpu_real": n_gpu_real,
        "drive_real": n_drive_real,
        "ram_real": n_ram_real,
    }

    return render_template("pipeline.html", stages=stages, lineage=lineage)


# all the metrics in one place - cli option 9

@app.route("/metrics")
def metrics():

    reg = pd.read_csv(os.path.join(cleaned_dir, "model_metrics.csv")).to_dict("records")

    per_cat_path = os.path.join(cleaned_dir, "model_metrics_per_category.csv")
    per_cat_pivot = []

    if os.path.exists(per_cat_path):
        pc = pd.read_csv(per_cat_path)
        pivot = pc.pivot(index="model", columns="category", values="MAE").reset_index()
        per_cat_pivot = pivot.to_dict("records")
        per_cat_cols = [c for c in pivot.columns if c != "model"]
    else:
        per_cat_cols = []

    real_backtests = {}

    for cat, fname in [("GPU", "real_gpu_backtest_metrics"),
                       ("Storage", "real_storage_backtest_metrics"),
                       ("RAM", "real_ram_backtest_metrics")]:

        path = os.path.join(cleaned_dir, f"{fname}.csv")

        if os.path.exists(path):
            real_backtests[cat] = pd.read_csv(path).to_dict("records")

    fi_path = os.path.join(cleaned_dir, "feature_importance.csv")
    feature_importance = pd.read_csv(fi_path).to_dict("records") if os.path.exists(fi_path) else []

    hp_path = os.path.join(cleaned_dir, "hardest_predictions.csv")
    hardest = pd.read_csv(hp_path).to_dict("records") if os.path.exists(hp_path) else []

    sr_path = os.path.join(cleaned_dir, "spec_regression_metrics.csv")
    spec_reg = pd.read_csv(sr_path).to_dict("records") if os.path.exists(sr_path) else []

    cls_path = os.path.join(cleaned_dir, "classification_metrics.csv")
    classification = pd.read_csv(cls_path).to_dict("records") if os.path.exists(cls_path) else []

    return render_template("metrics.html",
                           regression=reg,
                           per_cat=per_cat_pivot,
                           per_cat_cols=per_cat_cols,
                           real_backtests=real_backtests,
                           feature_importance=feature_importance,
                           hardest=hardest,
                           spec_regression=spec_reg,
                           classification=classification)


# visualization gallery - cli option 10

@app.route("/visuals")
def visuals():

    if not os.path.isdir(visuals_dir):
        return render_template("visuals.html", images=[])

    files = sorted(f for f in os.listdir(visuals_dir) if f.endswith(".png"))
    images = []

    for f in files:

        path = os.path.join(visuals_dir, f)
        size_kb = os.path.getsize(path) // 1024
        nice = f.replace("_", " ").replace(".png", "")
        images.append({"filename": f, "size_kb": size_kb, "title": nice})

    return render_template("visuals.html", images=images)


# serves png plots from visuals/

@app.route("/visuals/<path:filename>")
def visual_file(filename):

    return send_from_directory(visuals_dir, filename)


# this month - best buys + biggest projected drops, ranked across all forecasted components

@app.route("/deals")
def deals():

    cat_filter = request.args.get("cat", "").strip() or None

    if not os.path.exists(forecast_path):
        return render_template("deals.html", buys=[], waits=[], cat=cat_filter, n_total=0)

    fc = pd.read_csv(forecast_path)
    fc["forecast_month"] = pd.to_datetime(fc["forecast_month"])

    conn = sqlite3.connect(db_path)

    comp = pd.read_sql_query(
        "SELECT component_id, brand, model FROM components",
        conn,
    )

    hist = pd.read_sql_query(
        "SELECT component_id, MIN(synthetic_price) AS hist_min FROM synthetic_prices GROUP BY component_id",
        conn,
    )

    adj = pd.read_sql_query(
        "SELECT component_id, multiplier FROM prediction_adjustments",
        conn,
    )

    conn.close()

    fc = fc.merge(comp, on="component_id", how="left")
    fc = fc.merge(adj, on="component_id", how="left")
    fc["multiplier"] = fc["multiplier"].fillna(1.0)
    fc["predicted_price"] = fc["predicted_price"] * fc["multiplier"]

    this_month = pd.Timestamp.today().normalize().to_period("M")
    fc = fc[fc["forecast_month"].dt.to_period("M") >= this_month]

    if fc.empty:
        return render_template("deals.html", buys=[], waits=[], cat=cat_filter, n_total=0)

    rows = []

    for cid, sub in fc.groupby("component_id"):

        sub = sub.sort_values("forecast_month").reset_index(drop=True)
        current = float(sub.iloc[0]["predicted_price"])

        if current <= 0:
            continue

        idx_min = int(sub["predicted_price"].idxmin())
        cheapest = float(sub.iloc[idx_min]["predicted_price"])
        cheapest_date = sub.iloc[idx_min]["forecast_month"]
        months_to_cheapest = (cheapest_date.to_period("M") - sub.iloc[0]["forecast_month"].to_period("M")).n
        savings = current - cheapest
        pct = (savings / current * 100) if current > 0 else 0.0

        rows.append({
            "component_id": cid,
            "category": sub.iloc[0]["category"],
            "brand": sub.iloc[0]["brand"],
            "model": sub.iloc[0]["model"],
            "current": current,
            "cheapest": cheapest,
            "cheapest_date": cheapest_date,
            "months_to_cheapest": months_to_cheapest,
            "savings": savings,
            "pct_savings": pct,
        })

    df = pd.DataFrame(rows)

    df = df.merge(hist, on="component_id", how="left")
    df = df.merge(adj.rename(columns={"multiplier": "_m"}), on="component_id", how="left")
    df["_m"] = df["_m"].fillna(1.0)
    df["hist_min"] = df["hist_min"] * df["_m"]
    df = df.drop(columns="_m")

    df["pct_above_hist"] = ((df["current"] - df["hist_min"]) / df["hist_min"] * 100).where(df["hist_min"] > 0)

    if cat_filter:
        df = df[df["category"] == cat_filter]

    n_total = len(df)

    if n_total == 0:
        return render_template("deals.html", buys=[], waits=[], cat=cat_filter, n_total=0)

    buy_pool = df[df["pct_savings"] < 3.0].copy()
    buy_pool["buy_score"] = buy_pool["pct_above_hist"].fillna(0) + buy_pool["pct_savings"]
    buy_pool = buy_pool.sort_values("buy_score").head(12)

    wait_pool = df[df["pct_savings"] >= 5.0].copy()
    wait_pool = wait_pool.sort_values("pct_savings", ascending=False).head(12)

    def serialize(d):

        return [{
            "component_id": r["component_id"],
            "category": r["category"],
            "brand": r["brand"] if pd.notna(r["brand"]) else "",
            "model": r["model"] if pd.notna(r["model"]) else r["component_id"],
            "current": float(r["current"]),
            "cheapest": float(r["cheapest"]),
            "cheapest_date": r["cheapest_date"].strftime("%b %Y"),
            "months_to_cheapest": int(r["months_to_cheapest"]),
            "savings": float(r["savings"]),
            "pct_savings": float(r["pct_savings"]),
            "hist_min": float(r["hist_min"]) if pd.notna(r["hist_min"]) else None,
            "pct_above_hist": float(r["pct_above_hist"]) if pd.notna(r["pct_above_hist"]) else None,
        } for _, r in d.iterrows()]

    return render_template("deals.html",
                           buys=serialize(buy_pool),
                           waits=serialize(wait_pool),
                           cat=cat_filter,
                           n_total=n_total)


# side-by-side forecast overlay for 2-4 components

@app.route("/compare")
def compare():

    cids = [c for c in request.args.getlist("c") if c.strip()][:4]
    series = []
    summary = []

    if cids:

        today = pd.Timestamp.today().normalize()
        this_month = today.to_period("M")

        for cid in cids:

            row = get_component_row(cid)

            if row is None:
                continue

            syn = get_synth(cid)
            fc = get_forecast(cid)
            mult, _ = get_adjustment(cid)

            hist_points = []
            fc_points = []

            if not syn.empty:
                adj_syn = syn.copy()
                adj_syn["synthetic_price"] = adj_syn["synthetic_price"] * mult
                hist_points = [{"date": d.strftime("%Y-%m-%d"), "price": float(p)}
                               for d, p in zip(adj_syn["date"], adj_syn["synthetic_price"])]

            if not fc.empty:
                fc_adj = fc.copy()
                fc_adj["price"] = fc_adj["price"] * mult
                fc_points = [{"date": d.strftime("%Y-%m-%d"), "price": float(p)}
                             for d, p in zip(fc_adj["date"], fc_adj["price"])]

            series.append({
                "component_id": cid,
                "category": row.get("category"),
                "brand": row.get("brand"),
                "model": row.get("model"),
                "history": hist_points,
                "forecast": fc_points,
            })

            if not fc.empty:

                fc_future = fc.copy()
                fc_future["price"] = fc_future["price"] * mult
                fc_future = fc_future[fc_future["date"].dt.to_period("M") >= this_month]

                if fc_future.empty:
                    fc_future = fc.tail(1).copy()
                    fc_future["price"] = fc_future["price"] * mult

                current = float(fc_future.iloc[0]["price"])
                idx_min = int(fc_future["price"].idxmin())
                cheapest = float(fc_future.loc[idx_min, "price"])
                cheapest_date = fc_future.loc[idx_min, "date"]
                savings = current - cheapest
                pct = (savings / current * 100) if current > 0 else 0.0

                summary.append({
                    "component_id": cid,
                    "category": row.get("category"),
                    "model": row.get("model"),
                    "brand": row.get("brand"),
                    "current": current,
                    "cheapest": cheapest,
                    "cheapest_date": cheapest_date.strftime("%b %Y"),
                    "savings": savings,
                    "pct_savings": pct,
                })

    return render_template("compare.html", cids=cids, series=series, summary=summary)


# build planner - cli option 11

@app.route("/build")
def build():

    prefill = {}

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        cid = (request.args.get(cat) or "").strip()

        if not cid:
            continue

        row = get_component_row(cid)

        if row is None:
            continue

        prefill[cat] = {"component_id": cid, "model": row.get("model") or "(no model name)"}

    return render_template("build.html", prefill=prefill)


@app.route("/api/build", methods=["POST"])
def api_build():

    payload = request.get_json(force=True, silent=True) or {}
    chosen = {}

    for cat in ["GPU", "CPU", "RAM", "Storage"]:

        cid = (payload.get(cat) or "").strip()

        if cid:
            chosen[cat] = cid

    if not chosen:
        return jsonify({"error": "no components selected"}), 400

    monthly = {}
    used = {}

    for cat, cid in chosen.items():

        row = get_component_row(cid)
        fc = get_forecast(cid)
        mult, _ = get_adjustment(cid)

        if row is None or fc.empty:
            continue

        used[cat] = {"component_id": cid, "model": row.get("model"), "brand": row.get("brand")}

        for _, fr in fc.iterrows():

            d = fr["date"].strftime("%Y-%m-%d")

            if d not in monthly:
                monthly[d] = {}

            monthly[d][cat] = float(fr["price"]) * mult

    if not monthly:
        return jsonify({"error": "no forecast data available for selected components"}), 400

    rows = []

    for d in sorted(monthly.keys()):

        entry = {"month": d}
        total = 0.0

        for cat in used.keys():

            v = monthly[d].get(cat)

            if v is None:
                continue

            entry[cat] = round(v, 2)
            total += v

        entry["total"] = round(total, 2)
        rows.append(entry)

    totals = [r["total"] for r in rows]
    cheapest_i = int(np.argmin(totals))
    priciest_i = int(np.argmax(totals))
    savings = totals[priciest_i] - totals[cheapest_i]
    pct = (savings / totals[priciest_i] * 100) if totals[priciest_i] > 0 else 0.0

    return jsonify({
        "rows": rows,
        "used": used,
        "cheapest": rows[cheapest_i],
        "priciest": rows[priciest_i],
        "savings": round(savings, 2),
        "pct_savings": round(pct, 1),
    })


def main():

    ensure_user_tables()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    local = host in ("127.0.0.1", "localhost")
    app.run(host=host, port=port, debug=local)


if __name__ == "__main__":
    main()
