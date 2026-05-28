# PC Forecaster

Tells you the right month to buy each part of your next build. Pulls historical PCPartPicker snapshots and daily HardwareDealsCo price feeds, fits a Random Forest model with out-of-time backtests, and projects 12 months of forecasted prices per component — then turns that into a buy-or-wait call and a stacked monthly cost projection for any GPU / CPU / RAM / Storage combination you pick.

## Quick Start

From the project root:

```bash
pip install -r requirements.txt
python src/main.py    # only on first run, or to refresh forecasts
python src/app.py
```

Open **http://127.0.0.1:5000**.

Pages:

- **Summary** — counts, headline metrics, real-data backtest at a glance.
- **Components** — fuzzy search across the catalogue (6,000+ parts).
- **Component detail** — specs, price history with the 12-month forecast overlaid, a buy-or-wait verdict, similar parts by spec, and a form to submit a price you actually saw (which scales future predictions for that part).
- **Build Planner** — pick one of each category and get a stacked-cost projection across the next 12 months with the cheapest month called out.
- **Metrics** — every evaluation number the pipeline produced.
- **Visuals** — gallery of all plots.
- **Pipeline** — what each stage does and where its data comes from.
- **Observations** — your submitted prices and active calibration multipliers.

## What's Inside

- **Data sources.** Two PCPartPicker snapshots (2023 detailed CSVs, May 2025 snapshot) plus three open daily-price GitHub repos from HardwareDealsCo.
- **Cleaned catalogue.** 9,452 rows after IQR filtering and spec parsing → 6,078 unique components.
- **Cross-year matches.** 2,107 components appear in both 2023 and 2025 listings, enough to build a per-SKU time series.
- **Synthetic monthly series.** 61,103 rows covering Jan 2023 – May 2025, anchored to real prices with seasonality and nine encoded market events.
- **Real-data bridge.** 10,153 real-blended rows covering Sep 2025 – May 2026 from HardwareDealsCo daily feeds.
- **Forecast horizon.** Rolls forward 12 months from the date `model_training.py` last ran.

## Repository Layout

```text
PC Forecaster/
├── raw_data/
│   ├── 2023/                      # PCPartPicker 2023 detailed CSVs
│   ├── 2025/                      # PCPartPicker May 2025 snapshot
│   ├── gpu_deals_daily/           # HardwareDealsCo daily GPU JSONs   (gitignored)
│   ├── drive_deals_daily/         # HardwareDealsCo daily SSD/HDD     (gitignored)
│   └── ram_deals_daily/           # HardwareDealsCo daily RAM         (gitignored)
├── cleaned_data/                  # pipeline output CSVs
├── visuals/                       # plot PNGs
├── pcparts.db                     # SQLite database
├── src/
│   ├── app.py                     # Flask web app (the UI)
│   ├── main.py                    # pipeline orchestrator
│   ├── data_collection.py         # load + merge raw CSVs, ingest real daily prices
│   ├── gpu_real_data.py           # GPU ingest from HardwareDealsCo
│   ├── drive_real_data.py         # SSD/HDD ingest from HardwareDealsCo
│   ├── ram_real_data.py           # RAM ingest from HardwareDealsCo
│   ├── data_cleaning.py           # IQR caps, spec parsing, matching
│   ├── generate_synthetic.py      # synthetic series + real-data bridge
│   ├── database_setup.py          # SQLite schema + sample queries
│   ├── model_training.py          # regression + 3 real-data backtests
│   ├── estimator.py               # build cost forecast + k-means tiers
│   ├── value_metrics.py           # $/unit value metrics
│   ├── spec_classifier.py         # spec → tier classification
│   ├── spec_regression.py         # cross-sectional regression on real prices
│   ├── data_visualization.py      # charts
│   ├── templates/                 # Jinja templates
│   └── static/                    # CSS + assets
├── requirements.txt
└── README.md
```

## How the Pipeline Works

Run `python src/main.py` from the project root and stages execute in order:

1. **data_collection** — loads 2023 + 2025 PCPartPicker CSVs into one normalized dataset, then pulls daily prices from HardwareDealsCo (gpu-deals, drive-deals, ram-deals) and aggregates them into per-bucket monthly medians.
2. **data_cleaning** — cleans names, applies IQR-based price caps per category, parses spec strings into numeric columns, builds `component_id`, drops outliers, matches components across years.
3. **generate_synthetic** — builds the 29-month synthetic price series Jan 2023 – May 2025 using real anchor prices, seasonality, and nine encoded market events. Then bridges real bucket prices into per-SKU `real_blended` rows for matched GPU / Storage / RAM components.
4. **database_setup** — creates `pcparts.db` with six tables (`components`, `observed_prices`, `synthetic_prices`, plus three external real-price tables) and runs five sample SQL queries.
5. **model_training** — trains Linear Regression / Decision Tree / Random Forest plus a naive baseline. Time-based split, TimeSeriesSplit CV, permutation importance, per-category breakdown. Then three out-of-time real-data backtests (GPU, Storage, RAM) holding out the last three months of real prices.
6. **estimator** — projects monthly build costs across budget / mid / high tiers. Includes k-means tier discovery comparing data-driven clusters against hand-coded thresholds.
7. **value_metrics** — per-category $/unit metrics on real observed prices ($/GB VRAM, $/core, $/GB).
8. **spec_classifier** — predicts cheap / mid / expensive tier from spec features. Reports accuracy, macro F1 / P / R, per-class F1, plus 5-fold stratified CV and probability calibration.
9. **spec_regression** — cross-sectional regression on real prices: train on 2023 pairs, test on 2025 pairs.
10. **data_visualization** — produces plots in `visuals/` with annotated market events and a synthetic-vs-real GPU overlay.

## Key Results

Real-data backtest (Random Forest, last three months of real prices held out as the test set):

| Category | MAE | RMSE | MAPE | R² | n_test |
|----------|-----|------|------|----|--------|
| GPU | $212 | $362 | 23.2% | 0.85 | 559 |
| Storage | $62 | $107 | 21.9% | 0.87 | 1,689 |
| RAM | $56 | $97 | 15.7% | 0.89 | 1,055 |

Random Forest beats Linear Regression on every category — 1.55× on GPU, 2.0× on Storage, 3.5× on RAM (LR MAPE: GPU 41.5%, Storage 45.4%, RAM 55.7%).

Spec → tier classifier (Random Forest macro-F1): GPU 0.83, RAM 0.78, CPU 0.70, Storage 0.67. The mid tier is consistently the weakest class (boundary problem).

Cross-sectional spec → price regression on real prices (R²): GPU 0.73, RAM 0.69, CPU 0.51, Storage 0.43.

## What the Real Data Captured

Across the nine months of real-price coverage, SSD NVMe 1TB went from about $78 (Sep 2025) to $183 (May 2026) and DDR5 6000 32GB went from $100 to $390 — the AI-driven DRAM/NAND shortage of late 2025 / early 2026, reflected directly in the bridged price history.

## Limitations

- The synthetic train/test split is not a credible accuracy claim — the `expected_price` feature does most of the work, so the naive baseline matches the ML models there. The real-data backtest is the number to trust.
- CPU stays synthetic-only — HardwareDealsCo has no daily CPU price feed, so the CPU forecast is purely extrapolated from the 2023–2025 anchors.
- There's a three-month gap (Jun–Aug 2025) between the synthetic series end and the real-blended start.
- PCPartPicker 2025 anchor prices and HardwareDealsCo Sep 2025 anchors sit at slightly different market levels, so the bridged prices can be a few percent off for SKUs whose prices spiked between those two reference points.
- Older GPU "New" listings (RX 580, RTX 2080) are sometimes leftover specialty stock at non-current prices.

## Tech Stack

- Python 3
- Flask + Jinja2 for the web app, Chart.js for interactive charts
- pandas, numpy, matplotlib, scikit-learn for the pipeline
- SQLite via the `sqlite3` standard module
