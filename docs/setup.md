# Running your own copy

Everything here runs on free tiers: GitHub Actions, MongoDB Atlas M0 and Streamlit Community Cloud.

## Setup checklist

```
□ Step 1 — Get an AQICN API key
           → https://aqicn.org/api/ (free; the token arrives by email)

□ Step 2 — Get an OpenWeather API key
           → https://openweathermap.org/api → Sign Up → API Keys tab
           (active in about 2 hours; free tier allows 1000 calls/day)

□ Step 3 — Create a MongoDB Atlas cluster
           → https://cloud.mongodb.com → create a free M0 cluster
           → Database Access → add a user
           → Network Access → allow 0.0.0.0/0 (GitHub Actions has no fixed IP)
           → Connect → Drivers → copy the SRV connection string

□ Step 4 — Add GitHub Actions secrets
           → repo Settings → Secrets and variables → Actions
           → AQICN_API_KEY, OPENWEATHER_API_KEY, MONGODB_URI, MONGODB_DB_NAME
           → optional, for failure emails: ALERT_EMAIL, EMAIL_USERNAME, EMAIL_PASSWORD

□ Step 5 — Local environment
           cp .env.example .env      # then fill in your keys

□ Step 6 — Install dependencies
           pip install -r requirements.txt            # dashboard only
           pip install -r requirements-feature.txt    # hourly pipeline
           pip install -r requirements-training.txt   # training (adds torch CPU + Chronos)
           pip install -r requirements-dev.txt        # notebooks + tests

□ Step 7 — Seed the feature store (once)
           python src/backfill/backfill_features.py

□ Step 8 — Seed the model registry (once)
           python src/training_pipeline/train_model.py

□ Step 9 — Deploy the dashboard
           → https://share.streamlit.io → New app → main file: app.py
           → Advanced → Secrets: MONGODB_URI, MONGODB_DB_NAME (TOML format)

□ Step 10 — Enable the workflows
           gh workflow enable feature_pipeline.yml
           gh workflow enable training_pipeline.yml
```

After step 10 everything runs on its own. The daily training job also re-enables both
workflows, so GitHub does not switch them off after 60 days without a commit.

## Reviewer map

Where each part of the original course deliverable lives:

| Deliverable | Where to look |
|---|---|
| Feature pipeline (API fetch, feature engineering) | [`src/feature_pipeline/`](../src/feature_pipeline/): `fetch_data.py`, `feature_engineering.py`, `store_features.py` |
| Historical backfill | [`src/backfill/`](../src/backfill/): `backfill_features.py`, `fill_targets.py`, `ablation_backfill.py` |
| Training pipeline (models, metrics, registry) | [`src/training_pipeline/`](../src/training_pipeline/): `models.py`, `train_model.py`, `evaluate_model.py`, `register_model.py` |
| Automated CI/CD | [`.github/workflows/`](../.github/workflows/) and the repo's **Actions** tab |
| Web dashboard | [`app.py`](../app.py) and [`src/dashboard/`](../src/dashboard/) |
| Advanced analytics (EDA, SHAP, alerts, drift, ablation) | [`notebooks/eda.ipynb`](../notebooks/eda.ipynb) and `src/dashboard/components/` |
| Feature store and model registry | MongoDB Atlas; see the collections table in the [README](../README.md#storage) |
| Written report | [`report/report.pdf`](../report/report.pdf) |

The CI/CD history is visible without credentials: open the repo's **Actions** tab to see the
hourly *Feature Pipeline* and daily *Training Pipeline* runs. The model registry is visible on
the dashboard's Leaderboard tab.

## Feature schema history

Schema changes to `aqi_features` are applied by re-running the backfill, which upserts by `(city, timestamp)`.

| Version | Columns added | Migration |
|---|---|---|
| v1 | Baseline schema (2025) | — |
| v2 | `aqi_lag_48h`, `aqi_lag_72h`, `aqi_lag_168h`, `rolling_min_24h`, `rolling_max_24h`, `aqi_pct_change_24h` | `src/backfill/migrate_v1_to_v2.py` once |

`migrate_v1_to_v2.py` fills the new columns with conservative stand-ins: the 48h/72h/168h lags copy
`aqi_lag_24h`, the 24h min/max copy `aqi_lag_1h`, and `aqi_pct_change_24h` is set to 0.
