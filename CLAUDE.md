# Agent briefing — PC Forecaster

Tells you the right month to buy each part of a PC build. Pulls historical PCPartPicker snapshots and daily HardwareDealsCo price feeds, fits a Random Forest with out-of-time backtests, projects 12 months of forecast prices per component, and exposes the results through a Flask web app.

This file is the orientation doc for any AI agent working on the repo. Read it before touching code.

## Architecture in one paragraph

The project is a 10-stage data pipeline plus a web UI. `src/main.py` runs the pipeline end-to-end. Each stage reads CSVs (or SQLite tables) the previous stage wrote — no module imports another except for the orchestrator and a couple of helper imports for the real-data ingest. Final outputs are a SQLite DB (`pcparts.db`), 13 CSVs in `cleaned_data/`, and 10 PNGs in `visuals/`. `src/app.py` (Flask) reads those outputs and serves the seven-page web UI. It never re-runs the pipeline at request time.

```
raw_data/    →  pipeline (src/main.py)  →  pcparts.db + cleaned_data/*.csv + visuals/*.png
                                                    ↓
                                              src/app.py  →  browser
```

## Data semantics — the non-obvious parts

Most of the schema is self-explanatory. These are the parts that bite if you don't know them.

- **`component_id`** is constructed in `data_cleaning.py:build_component_id` as `model | chipset` for GPUs, `model | capacity | type` for Storage, `model | DDRn | speed | NxSize` for RAM, and just `model` for CPUs. It is the join key everywhere downstream. Do not change its construction without re-running the whole pipeline — every CSV and every DB row depends on it.

- **`synthetic_prices.data_source`** has two values: `synthetic` (the 29-month formula-generated series Jan 2023 – May 2025) and `real_blended` (Sep 2025 – May 2026 rows where a real chipset/bucket median was scaled by a per-SKU brand premium). They live in the same table because the regressor consumes both as a single price history. There is a three-month gap (Jun – Aug 2025) with no rows on either side.

- **`expected_price`** is the dominant feature in the model. Permutation importance confirms: shuffling it explodes MAE by ~$340; shuffling anything else moves the error less than $0.02. The naive baseline (predict `expected_price`) is within $1 of the best ML model on the synthetic split. The R² number on `model_metrics.csv` is therefore *not* the credible accuracy claim. The real-data backtest in `real_*_backtest_metrics.csv` is.

- **CPU has no real-data bridge** — HardwareDealsCo has no daily CPU feed. The CPU forecast extrapolates synthetic anchors only. Don't assume real_blended rows exist for CPU.

- **Matched vs unmatched components** — only 2,107 of 6,078 components appear in both 2023 and 2025 listings. Only matched components get a synthetic series and a forecast. The other ~4,000 still show up in search and on detail pages, but with no chart, no recommendation, and no calibration. The web app handles this; new features should too.

## Conventions Brian wants enforced

These come from accumulated feedback. Don't drift from them.

- **Style.** Lowercase module-level constants (no SCREAMING_SNAKE). No `from __future__ import annotations`. No type hints on function signatures. No `pathlib` — plain string paths only. No `if __name__ == "__main__"` block in pipeline-stage modules (only `main.py` has one).
- **Comments.** Only at the start of a function. Never inside the body. Keep them short and explanatory, not narrative. Never add `# AI helped here` or any marker of AI involvement.
- **No docstrings** on functions or modules unless explicitly asked.
- **SQL.** Write it raw. Don't import `textwrap.dedent`. Parameterize values; never f-string user-controlled data into a query. (See `src/data_visualization.py:load_corr_data` for a vestigial bad pattern — values are hardcoded so it's safe in practice, but the pattern is wrong.)
- **Plots.** Default matplotlib styling. No `plt.rcParams.update(...)` global config block. No custom color hex codes — use plain color names (`"red"`, `"blue"`). No bbox annotation boxes, no banner lines, no aligned-column f-string console tables.
- **Web.** Templates use the editorial style established in `static/style.css` — light cream palette, serif headings, sidebar nav, earth-tone tags. Chart.js colors must match the page CSS variables (navy / rust / ochre).

## When to ask before changing

Some calls are domain judgements, not code style. Ask before touching:

- **IQR rule choice** — the per-category cap is Q3 + 5·IQR, intentionally loose. The within-component outlier rule is median ± 3·IQR. Both numbers were tuned by hand against ground-truth lists of known-bad rows.
- **Tier thresholds** — `estimator.py:tier_rules` defines budget / mid / high cutoffs. K-means agreement is in the 70–85% range, so the hand-coded rules are intentionally not data-driven.
- **Model choice and hyperparameters** — Random Forest with `n_estimators=150, max_depth=14, min_samples_leaf=4` was picked after comparing MAE lift over naive on the real-data backtest. Swapping the model means re-running the whole evaluation chain.
- **Schema changes** — `database_setup.py:create_db` drops and recreates all tables on every run. Any new column needs the DDL, the build function, and any downstream reader updated together.
- **Feature engineering in `model_training.py:build_features`** — `expected_price` and `trend_per_step` are derived from the per-component 2023 / 2025 anchor prices in a way the model relies on. Touch with care.

## Gotchas

- **Python 3.14 has a 28-minute cold pandas import** on this machine. Python 3.12 doesn't. If a script feels frozen for 5+ minutes during the first run after reboot, that's why. Subsequent imports are fast.
- **Flask runs in debug mode** (`app.run(debug=True)`). File saves auto-reload templates. Fine for local; never for production.
- **`pcparts.db` is tracked in git.** The pipeline mutates it. Web app writes to `user_observations` and `prediction_adjustments` tables. If you see DB diffs in `git status`, that's expected; review the diff is just data, not schema.
- **`raw_data/{gpu,drive,ram}_deals_daily/` are gitignored** and re-synced on every pipeline run from the public HardwareDealsCo GitHub repos. First run will download a few hundred JSON files.
- **The forecast horizon rolls** — `model_training.py:forecast_future` builds 12 months starting from next month after today. The web app's buy-or-wait logic uses `pd.Timestamp.today()`, so the "now" anchor moves with real time. Re-run the pipeline if the forecast horizon needs to slide forward.

## Where AI-assist works well in this project

- Regex-heavy spec parsing in `data_cleaning.py` (the patterns are tedious but mechanical).
- Template-heavy Jinja work in `src/templates/`.
- Audit passes against the whole codebase — surfacing real bugs (syntax, leakage) that a single developer wouldn't catch in one read.
- Test data generation, sample SQL queries, plot variations.

## Where AI-assist does not work well

- Anything that requires deciding *what counts as a match* between datasets. The brand-premium scaling, the bucket key construction, the workstation-GPU exclusion list — all needed human judgement based on hand-spot-checking output.
- The "would this be a credible claim?" check. The model R² of 0.985 looked impressive until permutation importance showed the feature shuffling story. That kind of skeptical reading is the human's job.

## Quick start

```
pip install -r requirements-pipeline.txt
python src/main.py     # only on first run, or to refresh forecasts
python src/app.py      # http://127.0.0.1:5000
```

`requirements-pipeline.txt` is the full set (scikit-learn/scipy included) the pipeline needs. Root `requirements.txt` is intentionally slim (Flask + pandas + numpy) — it's the Vercel function's dependency list, kept under the serverless size limit. Don't add scikit-learn back to it.

## Deploy (Vercel)

The web app deploys to Vercel as one Python serverless function. `api/index.py` imports the Flask `app` and routes everything through it via `vercel.json`. Two consequences of the serverless model that new work must respect:

- **Read-only filesystem.** Only `/tmp` is writable. `app.py:writable_db_path` copies `pcparts.db` to `/tmp` when the `VERCEL` env var is present, so the observation-write path keeps working (non-durably, reset on cold start — same as the old free host). Any new write target must go to `/tmp` on Vercel.
- **Size limit (~250MB unzipped).** That's why scikit-learn/scipy aren't shipped — the "similar parts" KNN in `app.py` was rewritten in plain numpy (z-score + euclidean). Don't reintroduce sklearn into the web path or the function won't build. Bundled data files are pinned in `vercel.json` `includeFiles`; add new runtime-read files there.

See `README.md` for the full feature tour.
