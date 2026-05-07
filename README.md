# Predicting the Optimal Time to Build a PC Using Hardware Price Trends

CS 210 Data Management for Data Science project. Uses PCPartPicker snapshots and daily HardwareDealsCo price feeds to model GPU / CPU / RAM / Storage prices and forecast when each part is cheapest over the next 12 months.

## Project Summary

- Data sources: two PCPartPicker snapshots (2023 detailed CSVs and a May 2025 snapshot) plus three open daily-price GitHub repos from HardwareDealsCo
- Cleaned dataset: 9,452 rows after IQR filtering and spec parsing
- Matched components across both PCPartPicker years: 2,107
- Categories: GPU, CPU, RAM, Storage
- Synthetic monthly series: 61,103 rows, Jan 2023 to May 2025
- Real-data bridge: 10,153 real_blended rows, Sep 2025 to May 2026
- Forecast horizon: dynamic, rolls forward 12 months from the run date

## Repository Structure

```text
DataManagement Project/
├── raw_data/
│   ├── 2023/                      # PCPartPicker 2023 detailed CSVs
│   ├── 2025/                      # PCPartPicker May 2025 snapshot
│   ├── gpu_deals_daily/           # HardwareDealsCo daily GPU JSONs (gitignored)
│   ├── drive_deals_daily/         # HardwareDealsCo daily SSD/HDD JSONs (gitignored)
│   └── ram_deals_daily/           # HardwareDealsCo daily RAM JSONs (gitignored)
├── cleaned_data/                  # pipeline output CSVs
├── visuals/                       # plot PNGs
├── pcparts.db                     # SQLite database (built by Stage 4)
├── src/
│   ├── main.py                    # pipeline orchestrator
│   ├── data_collection.py         # Stage 1 — load + merge raw CSVs, then ingest real daily prices
│   ├── gpu_real_data.py           # GPU real-data ingest from HardwareDealsCo
│   ├── drive_real_data.py         # SSD/HDD real-data ingest from HardwareDealsCo
│   ├── ram_real_data.py           # RAM real-data ingest from HardwareDealsCo
│   ├── data_cleaning.py           # Stage 2 — IQR caps, spec parsing, component_id, matching
│   ├── generate_synthetic.py      # Stage 3 — synthetic series + real-data bridge
│   ├── database_setup.py          # Stage 4 — SQLite schema + sample queries
│   ├── model_training.py          # Stage 5 — regression + 3 real-data backtests
│   ├── estimator.py               # Stage 6 — build cost forecast + k-means tiers
│   ├── value_metrics.py           # Stage 7 — $/unit value metrics
│   ├── spec_classifier.py         # Stage 8 — spec→tier classification
│   ├── spec_regression.py         # Stage 9 — cross-sectional regression on real prices
│   ├── data_visualization.py      # Stage 10 — charts
│   └── cli.py                     # interactive console
├── inspect_data.py
├── README.md
└── requirements.txt
```

## Pipeline

1. `data_collection.py` — loads 2023 + 2025 PCPartPicker CSVs into one normalized dataset, then ingests real daily prices from HardwareDealsCo (gpu-deals, drive-deals, ram-deals) into per-bucket monthly medians. Outputs `cleaned_data/combined_parts.csv` plus three real-price CSVs.
2. `data_cleaning.py` — cleans names, applies IQR-based price caps per category, parses spec strings into numeric columns, builds `component_id`, removes outliers, matches components across years. Outputs `cleaned_data/combined_parts_cleaned.csv`.
3. `generate_synthetic.py` — builds the 29-month synthetic price series Jan 2023 to May 2025 using real anchor prices, seasonality, 9 market events, and noise. Then bridges real bucket prices into per-SKU `real_blended` rows for matched GPU / Storage / RAM components. Outputs `cleaned_data/price_history.csv` (71,256 rows).
4. `database_setup.py` — creates `pcparts.db` with six tables: `components`, `observed_prices`, `synthetic_prices` (with `data_source` column), `gpu_chipset_real_prices`, `drive_real_prices`, `ram_real_prices`. Runs 5 sample SQL queries.
5. `model_training.py` — trains Linear Regression / Decision Tree / Random Forest plus a naive baseline. Time-based split + TimeSeriesSplit CV + permutation importance + per-category breakdown. Then runs three out-of-time real-data backtests (GPU, Storage, RAM) holding out the last three months of real prices. Writes forecast and metrics CSVs.
6. `estimator.py` — projects monthly build costs across budget / mid / high tiers using the forecast. Includes k-means tier discovery comparing data-driven clusters against hand-coded thresholds.
7. `value_metrics.py` — per-category $/unit value metrics on real observed prices.
8. `spec_classifier.py` — predicts cheap / mid / expensive tier from spec features using Logistic Regression / Decision Tree / Random Forest. Reports accuracy, macro F1 / precision / recall, per-class F1, plus 5-fold StratifiedKFold CV and probability calibration.
9. `spec_regression.py` — cross-sectional regression on real prices: train on 2023 (component, price) pairs, test on 2025 pairs.
10. `data_visualization.py` — produces plots in `visuals/` with annotated event windows and a synthetic-vs-real GPU overlay.

## How to Run

From the project root:

```bash
python src/main.py
```

## Interactive Console

After the pipeline has run:

```bash
python src/cli.py
```

Menu options:

1. project summary — counts, model metrics, classifier metrics
2. look up a component — observed prices, full price history including real_blended rows, forecast
3. buy now or wait — recommendation engine using the forecast and historical low
4. submit a real observation — feeds in a real price, stores it, and adjusts future predictions for that component (adaptive correction loop)
5. view user observations + active calibrations
6. reset all user observations + calibrations
7. view pipeline stages
8. view data lineage — which stages used real prices vs synthetic vs real_blended
9. view all metrics — regression single-split, real-data backtests, classification
10. list visualizations and optionally open one
11. build my PC — pick one each of GPU/CPU/RAM/Storage and see projected build cost across the 12 forecast months with cheapest month and savings
12. find similar components — k-nearest-neighbor search in standardized spec-feature space

## Key Results

Real-data backtest (Random Forest, 3 months held out as test set, real 2026 prices the model never saw):

| Category | MAE | RMSE | MAPE | R² | n_test |
|----------|-----|------|------|----|--------|
| GPU | $212 | $362 | 23.3% | 0.85 | 559 |
| Storage | $62 | $107 | 21.9% | 0.87 | 1,689 |
| RAM | $56 | $97 | 15.7% | 0.89 | 1,055 |

Random Forest beats Linear Regression by 2–4× MAE on every category in this backtest (LR MAPE: GPU 41.6%, Storage 45.4%, RAM 55.7%).

Spec→tier classification (Random Forest, macro-F1 by category): GPU 0.83, RAM 0.78, CPU 0.70, Storage 0.67. Mid tier is consistently the weakest class (boundary problem).

Cross-sectional spec→price regression on real prices (R²): GPU 0.73, RAM 0.69, CPU 0.51, Storage 0.43.

The synthetic-only train/test split shows R² around 0.985 but the naive baseline gets 0.98 — most of that score is the trend interpolation, not the model. Permutation importance confirmed: shuffling `expected_price` blows up the error; shuffling everything else changes almost nothing. The real-data backtest above is the credible accuracy claim.

## Real Market Story Captured

Across the 9 months of real-price data, SSD NVMe 1TB went from about $78 (Sep 2025) to $183 (May 2026) and DDR5 6000 32GB went from $100 to $390 — the AI-driven DRAM/NAND shortage of late 2025 / early 2026, reflected directly in the bridged price history.

## Limitations

- Synthetic R² inflation: the model leans heavily on `expected_price`, so the synthetic split is not a credible accuracy claim.
- CPU stays synthetic-only because there is no daily real-price source for CPUs in HardwareDealsCo.
- Brand-premium scale mismatch: PCPartPicker 2025 anchor prices and HardwareDealsCo September 2025 anchors sit at slightly different market levels, which can deflate or inflate the bridged prices for SKUs whose prices spiked between the two reference points.
- Older GPU "New" listings (RX 580, RTX 2080) are sometimes leftover specialty stock at non-current prices.
- RAM matching is partly limited by missing module fields in the project DB (per-module vs kit-total mismatch).

## Tech Stack

- Python 3
- pandas, numpy, matplotlib, scikit-learn
- SQLite (via `sqlite3` module)

## Course Context

CS 210 Data Management for Data Science. Project title: Predicting the Optimal Time to Build a PC Using Hardware Price Trends.
