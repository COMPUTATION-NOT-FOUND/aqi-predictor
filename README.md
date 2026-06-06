# Pearls AQI Predictor

End-to-end AQI forecasting system for Karachi — predicts Air Quality Index for the next 3 days using a fully serverless ML pipeline.

🔗 **Live dashboard:** https://aqi-predictor-clgefbv7tu87ehnluaddsu.streamlit.app/
📄 **Technical report:** [`report/report.pdf`](report/report.pdf) · 📓 **EDA:** [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

> Reviewing this submission? Jump to [How to review this submission](#how-to-review-this-submission) for a map of every deliverable.

## Architecture

```
AQICN API ──┐
             ├─→ Feature Pipeline ─→ MongoDB Atlas ─→ Training Pipeline ─→ Model Registry ─→ Dashboard
OpenWeather ─┘         (hourly)                           (daily)
```

---

## Setup Checklist (Things YOU Must Do)

```
□ Step 1 — Register AQICN API key
           → https://aqicn.org/api/
           → Fill in the form → token sent to your email instantly (free)

□ Step 2 — Register OpenWeather API key
           → https://openweathermap.org/api → Sign Up
           → Go to API Keys tab → copy your key (active in ~2 hours, free tier: 1000 calls/day)

□ Step 3 — Create MongoDB Atlas cluster
           → https://cloud.mongodb.com → Sign Up → Create a free M0 cluster
           → Security → Database Access → Add DB User (note username + password)
           → Network Access → Allow access from anywhere (0.0.0.0/0) for GitHub Actions
           → Connect → Drivers → copy the SRV connection string

□ Step 4 — Create GitHub repository and add Secrets
           → GitHub repo → Settings → Secrets and variables → Actions → New secret
           → Add: AQICN_API_KEY, OPENWEATHER_API_KEY, MONGODB_URI, MONGODB_DB_NAME

□ Step 5 — Copy .env.example → .env and fill in your keys (for local development)
           cp .env.example .env
           # edit .env with your actual keys

□ Step 6 — Install dependencies
           pip install -r requirements.txt

□ Step 7 — Run backfill ONCE to seed the feature store
           python src/backfill/backfill_features.py

□ Step 8 — Run training ONCE to seed the model registry
           python src/training_pipeline/train_model.py

□ Step 9 — Deploy dashboard to Streamlit Community Cloud (free tier)
           → https://share.streamlit.io → Sign in with GitHub → New app
           → Repository: your repo, Branch: main, Main file: app.py
           → Advanced → Secrets: paste MONGODB_URI, MONGODB_DB_NAME,
             AQICN_API_KEY, OPENWEATHER_API_KEY (TOML format)
           → Live deployment for this repo:
             https://aqi-predictor-clgefbv7tu87ehnluaddsu.streamlit.app/
```

After Step 9, everything runs automatically via GitHub Actions. The dashboard reads from MongoDB on every page load.

---

## What Runs Automatically (After Setup)

| Task | Schedule |
|---|---|
| Fetch new AQI + weather data | Every hour |
| Compute features + drift check | Every hour |
| Train all models + run ablation | Daily at 2 AM UTC |
| Champion-Challenger promotion | Daily (end of training) |
| Dashboard refresh | On every page load |

---

## How to review this submission

Every required deliverable and where to find it:

| Deliverable | Where to look |
|---|---|
| **Feature pipeline** (API fetch, feature engineering, time/derived features) | [`src/feature_pipeline/`](src/feature_pipeline/) — `fetch_data.py`, `feature_engineering.py`, `store_features.py` |
| **Historical backfill** (training-data generation) | [`src/backfill/`](src/backfill/) — `backfill_features.py`, `fill_targets.py`, `ablation_backfill.py` |
| **Training pipeline** (models, metrics, registry) | [`src/training_pipeline/`](src/training_pipeline/) — `models.py`, `train_model.py`, `evaluate_model.py`, `register_model.py` |
| **Automated CI/CD** | [`.github/workflows/`](.github/workflows/) — and the repo's **Actions** tab shows the live hourly/daily run history |
| **Web dashboard** | [`app.py`](app.py) + [`src/dashboard/`](src/dashboard/) → **live:** https://aqi-predictor-clgefbv7tu87ehnluaddsu.streamlit.app/ |
| **Advanced analytics** (EDA, SHAP, alerts, drift, ablation) | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) + `src/dashboard/components/{shap_plot,alerts,drift_tab,ablation_tab}.py` |
| **Feature store & model registry** | MongoDB Atlas — see [Accessing the Feature Store & Model Registry](#accessing-the-feature-store--model-registry) |
| **Written report** | [`report/report.pdf`](report/report.pdf) |

> **CI/CD is observable without any credentials:** open the repo on GitHub → **Actions** tab → the *Feature Pipeline* (hourly) and *Training Pipeline* (daily) workflows show their full scheduled run history.

---

## Project Structure

```
aqi-predictor/
├── src/
│   ├── config.py                  # All configuration (FILL IN env vars)
│   ├── feature_pipeline/
│   │   ├── fetch_data.py          # AQICN + OpenWeather API calls
│   │   ├── feature_engineering.py # Feature computation + normalization
│   │   ├── drift_monitor.py       # PSI-based drift detection
│   │   └── store_features.py      # Store features to MongoDB (aqi_features collection)
│   ├── backfill/
│   │   ├── backfill_features.py   # Seed historical data (run once)
│   │   └── ablation_backfill.py   # Compare 5 imputation strategies
│   ├── training_pipeline/
│   │   ├── models.py              # 14 model definitions + search spaces
│   │   ├── train_model.py         # Main training loop
│   │   ├── evaluate_model.py      # RMSE, MAE, R², IoA, Skill Score
│   │   ├── distillation.py        # Ensemble → lightweight MLP
│   │   ├── conformal.py           # Prediction intervals via MAPIE
│   │   ├── ablation_features.py   # Feature group ablation
│   │   └── register_model.py      # Champion-Challenger + MongoDB/GridFS artifact store
│   └── dashboard/
│       ├── app.py                 # Dashboard components (entry point is root app.py)
│       ├── inference.py           # Load model (cached) + predict
│       └── components/
│           ├── forecast_chart.py  # 3-day forecast with uncertainty bands
│           ├── gauge.py           # AQI gauge
│           ├── leaderboard.py     # Model registry table
│           ├── shap_plot.py       # SHAP explainability
│           ├── alerts.py          # Hazardous AQI banner
│           ├── drift_tab.py       # PSI drift heatmap
│           └── ablation_tab.py    # Self-maintained ablation dashboard
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis (exports figures to report/figures/)
├── report/
│   ├── report.tex                 # Comprehensive technical report (LaTeX source)
│   ├── report.pdf                 # Compiled report
│   ├── build.sh                   # pdflatex build script
│   └── figures/                   # EDA charts exported by the notebook
├── .github/workflows/
│   ├── feature_pipeline.yml       # Hourly cron
│   └── training_pipeline.yml      # Daily cron
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dashboard Tabs

1. **Live Forecast** — Current AQI gauge, 3-day forecast with uncertainty bands, hazardous alert
2. **Leaderboard** — All models ranked by RMSE + TOPSIS; shows champion and challengers (LSTM/DistilledMLP are leaderboard-only)
3. **Ablation Study** — Auto-updated charts showing which feature groups improve or harm accuracy
4. **Feature Importance** — SHAP bar chart for the current champion model

---

## Models Trained

| Model | Type | Champion-eligible |
|---|---|---|
| Ridge, Lasso, ElasticNet | Classical Linear | Yes |
| Random Forest, Gradient Boosting | Classical Ensemble | Yes |
| XGBoost, LightGBM, CatBoost | Gradient Boosting | Yes |
| Voting Ensemble, Stacking Ensemble | Hybrid Ensemble | Yes |
| LSTM | Deep Learning | **No** — dashboard has no TensorFlow |
| Distilled MLP | Knowledge Distillation | **No** — dashboard has no TensorFlow |

> **Note:** `forecast_leads` (forecasted weather + PM2.5 at t+24/48/72h) is currently **disabled** — the backfill populates these columns from realized future values, making them identical to the targets (target leak). Re-enable once the backfill is updated to use the Open-Meteo historical-forecast archive (past issued forecasts, not actuals).

---

## Key Design Decisions

- **Ground truth**: AQI computed from raw PM2.5 using the US EPA linear interpolation formula
- **Loss function**: Huber loss for gradient boosters (XGBoost, LightGBM, CatBoost) and LSTM — robust to AQI spike outliers that inflate MSE and suppress IoA
- **Forecasting**: MIMO (direct multi-output) — predicts 24h/48h/72h simultaneously, avoids error accumulation
- **Features**: 50+ engineered features including multi-scale lags (1h–168h / 7-day), rolling statistics (mean, std, min, max over 24h), physics-derived and cross features
- **Normalization**: log(x+1) → RobustScaler for skewed pollutants; scaler fitted on training data only
- **Model protection**: Champion-Challenger gate — a challenger must first clear three hard gates (skill score > 0, *not* overfit, IoA ≥ 0.15), then it is ranked by TOPSIS and only promoted if it beats the current champion's TOPSIS score by ≥1% (prevents a mean-predicting or marginally-different model from churning the champion slot)
- **Champion model**: the best sklearn model by TOPSIS ranking (OOF RMSE, IoA, skill score, MAE). LSTM and DistilledMLP are leaderboard-only challengers — they cannot be promoted because the Streamlit dashboard has no TensorFlow and would fail to unpickle a Keras artifact.

---

## Performance Notes

### Dashboard speed on Streamlit Community Cloud
- **Cold starts (~10–20 s)** occur when the app hasn't been visited recently — Streamlit Community Cloud spins down idle apps.
- **In-session performance** is fast: champion model artifacts and leaderboard metadata are loaded from MongoDB once per session and cached with Streamlit's `@st.cache_data` (metadata: 30-min TTL; SHAP: 60-min TTL). The 3-day forecast is precomputed hourly by the feature pipeline and stored in MongoDB, so the dashboard never runs live inference.
- **Testing locally:** run `./run_local.sh` to launch Streamlit on your machine (`streamlit run app.py`). Requires `.env` with `MONGODB_URI` and API keys.

### Why IoA matters alongside RMSE
RMSE-minimising models can "win" by predicting close to the mean — low error but zero predictive value. IoA (Index of Agreement, Willmott 1981) measures how well the model tracks actual AQI movements. IoA = 1.0 is perfect. The champion promotion gate requires **IoA ≥ 0.15** as a hard floor to keep degenerate (mean-predicting) models off the leaderboard's top slot, while TOPSIS — which weights IoA alongside OOF RMSE, skill score and MAE — does the actual ranking. In practice the promoted champion sits well above the floor (recent runs: IoA ≈ 0.55–0.60).

---

## Accessing the Feature Store & Model Registry

This project uses **MongoDB Atlas** as both the feature store and the model registry — Hopsworks and Vertex AI are *not* used. Because they are live database collections rather than a managed UI, here is how a reviewer can inspect them **without database credentials**:

- **Model registry → the dashboard Leaderboard tab.** Every model from the latest training run, its metrics (RMSE, MAE, R², IoA, skill score, TOPSIS) and the promoted champion are rendered live from the `model_metadata` collection at the [live dashboard](https://aqi-predictor-clgefbv7tu87ehnluaddsu.streamlit.app/). Champion binaries are stored in **GridFS**; the full per-run history is the append-only `training_history` collection.
- **Feature store → the live forecast + the EDA notebook.** The `aqi_features` collection (one row per city/hour, ~50 engineered features + the 24h/48h/72h targets) is what drives both the dashboard's 3-day forecast and [`notebooks/eda.ipynb`](notebooks/eda.ipynb), which loads it via `fetch_training_data()`.
- **Schema** for every collection is documented in the table below and, in full, in [`report/report.pdf`](report/report.pdf).

No live credentials are published. To run the pipelines yourself, provision your own free MongoDB Atlas cluster (Setup Step 3) and set `MONGODB_URI` / `MONGODB_DB_NAME`.

## MongoDB Collections

| Collection | Purpose | Write pattern |
|---|---|---|
| `aqi_features` | Engineered feature rows (one per city/hour) | Upsert by `(city, timestamp)` |
| `model_metadata` | Live leaderboard — current metrics for each model | Upsert by `model_name` (overwritten each run) |
| `training_history` | Full append-only log of every model from every run | `insert_many` — never upsert |
| `oof_predictions` | OOF hindcast predictions for champion model | Upsert |
| `predictions` | Precomputed 3-day forecast (refreshed hourly) | Upsert |
| `ablation_results` | Feature group ablation scores | Upsert |
| `drift_log` / `drift_baseline` | PSI drift tracking | Upsert |
| `feature_overrides` | Ablation-driven feature group toggles | Upsert |

### training_history

Each training run inserts one document per model (including LSTM and DistilledMLP) via `insert_many`. Documents are never overwritten. Fields:

- **Run identity:** `run_id` (`GITHUB_RUN_ID` or a UUID), `trained_at` (shared ISO timestamp for all models in the run), `github_sha`, `github_run_number`
- **Run context:** `feature_groups` (snapshot of enabled/disabled groups), `feature_count`
- **Per-model:** `model_name`, `model_type`, `champion_eligible`, `promoted` (True only for the model promoted that run), `topsis_score`
- **Metrics** spread flat: `rmse`, `oof_rmse`, `ioa`, `skill_score`, `mae`, `rmse_d1/d2/d3`, `overfit_flag`, etc.

Indexes: compound `(model_name, trained_at DESC)` for per-model trend queries; `run_id` for grouping a run's models. No TTL — kept indefinitely.

### Feature schema history

Schema changes to `aqi_features` are handled by re-running the backfill (`src/backfill/backfill_features.py`) which upserts by `(city, timestamp)`.

| Version | Columns added | Migration |
|---------|--------------|----------|
| v1 | Baseline schema (2025) | — |
| v2 | `aqi_lag_48h`, `aqi_lag_72h`, `aqi_lag_168h`, `rolling_min_24h`, `rolling_max_24h`, `aqi_pct_change_24h` | Run `src/backfill/migrate_v1_to_v2.py` once to back-fill with proxies |

`migrate_v1_to_v2.py` back-fills with conservative proxies:
- `aqi_lag_48h / 72h / 168h` → copied from `aqi_lag_24h`
- `rolling_min_24h / max_24h` → copied from `aqi_lag_1h`
- `aqi_pct_change_24h` → `0.0`
