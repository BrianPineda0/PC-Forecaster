import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)


spec_features = {
    "GPU": ["memory_gb", "tdp_watts", "core_clock_mhz"],
    "CPU": ["core_count", "tdp_watts"],
    "RAM": ["ddr_gen", "speed_mhz", "module_count", "module_size_gb"],
    "Storage": ["capacity_gb"],
}

classes = ["cheap", "mid", "expensive"]


# pulls components + their real observed prices out of sqlite
# the classifier uses real prices not synthetic so it works on actual market data

def load_components():

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


# splits prices into 3 equal-size groups per category
# this is the cheap/mid/expensive label per category instead of one universal cutoff

def label_terciles(prices):
    return pd.qcut(prices, q=3, labels=classes, duplicates="drop")


# fits one classifier and reports accuracy + macro f1 + macro precision + macro recall + per-class f1

def fit_eval(name, clf, X_train, y_train, X_test, y_test):

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)

    per_f1 = f1_score(y_test, preds, average=None, labels=classes, zero_division=0)

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "f1_cheap": round(per_f1[0], 4),
        "f1_mid": round(per_f1[1], 4),
        "f1_expensive": round(per_f1[2], 4),
    }

    return metrics, preds


# 5-fold stratified cv per classifier to check if the test result was a fluke
# stratified keeps the cheap/mid/expensive ratio balanced inside each fold

def stratified_cv(classifiers, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}

    for name, clf in classifiers.items():

        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        out[name] = (float(scores.mean()), float(scores.std()))

    return out


# draws probability calibration curves per category
# checks whether the model's confidence actually matches the true rate of being right

def plot_calibration_grid(per_cat_data):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    class_colors = {"cheap": "green", "mid": "orange", "expensive": "red"}

    for ax, (cat, payload) in zip(axes, per_cat_data.items()):

        y_test = payload["y_test"]
        proba = payload["proba"]

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")

        for i, c in enumerate(classes):

            y_binary = (y_test == c).astype(int)

            if y_binary.sum() < 5:
                continue

            prob_true, prob_pred = calibration_curve(y_binary, proba[:, i], n_bins=8, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", color=class_colors[c], label=c)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{cat}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Random Forest Probability Calibration by Category")
    plt.tight_layout()
    plt.savefig("visuals/classification_calibration.png")
    plt.close()
    print("Saved: classification_calibration.png")


# 2x2 grid of confusion matrices one per category
# shows where the model confuses cheap with mid vs cheap with expensive

def plot_confusion_grid(per_cat_data):

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (cat, payload) in zip(axes, per_cat_data.items()):

        cm = payload["cm"]
        n_test = payload["n_test"]
        f1 = payload["macro_f1"]

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{cat} (RF macro-F1 = {f1:.3f}, n_test = {n_test})")

        for i in range(3):

            for j in range(3):

                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

        plt.colorbar(im, ax=ax)

    fig.suptitle("Spec-to-Price-Tier Classification: Confusion Matrices (Random Forest)")
    plt.tight_layout()
    plt.savefig("visuals/classification_confusion.png")
    plt.close()
    print("Saved: classification_confusion.png")


# trains 3 classifiers per category, evaluates them, runs 5-fold cv
# saves the metrics csv and both diagnostic plots

def main():

    df = load_components()
    print(f"Loaded {len(df):,} component-year observations")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1),
    }
Z
    metrics_rows = []
    per_cat_data = {}

    for cat, feats in spec_features.items():

        sub = df[df["category"] == cat].dropna(subset=feats + ["price"]).copy()

        if len(sub) < 60:
            print(f"  {cat}: skipping ({len(sub)} usable rows)")
            continue

        sub["price_tier"] = label_terciles(sub["price"])
        sub = sub.dropna(subset=["price_tier"])

        X = StandardScaler().fit_transform(sub[feats].values)
        y = sub["price_tier"].astype(str).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        print(f"\n{cat} ({len(sub):,} rows, train {len(X_train):,} / test {len(X_test):,})  features: {feats}")

        rf_preds = None
        rf_proba = None

        for name, clf in classifiers.items():

            metrics, preds = fit_eval(name, clf, X_train, y_train, X_test, y_test)
            metrics["category"] = cat
            metrics_rows.append(metrics)

            print(f"  {name}: acc={metrics['accuracy']:.4f}, macro_F1={metrics['macro_f1']:.4f}, macro_P={metrics['macro_precision']:.4f}, macro_R={metrics['macro_recall']:.4f}")

            if name == "Random Forest":
                rf_preds = preds
                rf_proba = clf.predict_proba(X_test)

        cm = confusion_matrix(y_test, rf_preds, labels=classes)
        rf_metric = next(m for m in metrics_rows if m["category"] == cat and m["model"] == "Random Forest")
        per_cat_data[cat] = {"cm": cm, "n_test": len(X_test), "macro_f1": rf_metric["macro_f1"],
                             "y_test": y_test, "proba": rf_proba}

        cv_results = stratified_cv(classifiers, X, y)
        print(f"  5-fold StratifiedKFold macro-F1:")

        for name, (mean_f1, std_f1) in cv_results.items():
            print(f"    {name}: {mean_f1:.4f} +/- {std_f1:.4f}")

    out_cols = ["category", "model", "accuracy", "macro_f1", "macro_precision", "macro_recall",
                "f1_cheap", "f1_mid", "f1_expensive"]
    metrics_df = pd.DataFrame(metrics_rows)[out_cols]
    metrics_df.to_csv("cleaned_data/classification_metrics.csv", index=False)
    print("\nSaved: cleaned_data/classification_metrics.csv")

    plot_confusion_grid(per_cat_data)
    plot_calibration_grid(per_cat_data)
