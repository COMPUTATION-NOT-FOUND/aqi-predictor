# Pearls AQI Predictor

End-to-end AQI forecasting system for Karachi — predicts Air Quality Index for the next 3 days using a fully serverless ML pipeline.

## Architecture

```
AQICN API ──┐
             ├─→ Feature Pipeline ─→ Hopsworks Feature Store ─→ Training Pipeline ─→ Model Registry ─→ Dashboard
OpenWeather ─┘         (hourly)                                      (daily)
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

□ Step 3 — Create Hopsworks project
           → https://app.hopsworks.ai → Sign Up → Create Project
           → Go to Account Settings → API Key → copy key

□ Step 4 — Create GitHub repository and add Secrets
           → GitHub repo → Settings → Secrets and variables → Actions → New secret
           → Add: AQICN_API_KEY, OPENWEATHER_API_KEY, HOPSWORKS_API_KEY

□ Step 5 — Copy .env.example → .env and fill in your keys (for local development)
           cp .env.example .env
           # edit .env with your actual keys

□ Step 6 — Install dependencies
           pip install -r requirements.txt

□ Step 7 — Run backfill ONCE to seed the feature store
           python src/backfill/backfill_features.py

□ Step 8 — Run training ONCE to seed the model registry
           python src/training_pipeline/train_model.py

□ Step 9 — Deploy dashboard to Render (free tier)
           → https://render.com → New Web Service → Connect GitHub repo
           → Set environment variables (same as .env)
           → Start command: python src/dashboard/app.py
```

After Step 9, everything runs automatically via GitHub Actions.

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

## Project Structure

```
aqi-predictor/
├── src/
│   ├── config.py                  # All configuration (FILL IN env vars)
│   ├── feature_pipeline/
│   │   ├── fetch_data.py          # AQICN + OpenWeather API calls
│   │   ├── feature_engineering.py # Feature computation + normalization
│   │   ├── drift_monitor.py       # PSI-based drift detection
│   │   └── store_features.py      # Push to Hopsworks Feature Store
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
│   │   └── register_model.py      # Champion-Challenger + Hopsworks push
│   └── dashboard/
│       ├── app.py                 # Multi-tab Plotly Dash app
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
│   └── eda.ipynb                  # Exploratory data analysis
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
2. **Model Leaderboard** — All models ranked by RMSE; freeze/rollback controls
3. **Ablation Study** — Auto-updated charts: which features matter, which backfill strategy wins
4. **SHAP Explainability** — Why the model predicted what it predicted
5. **Data Drift** — PSI per feature over time; alerts when distribution shifts

---

## Models Trained

| Model | Type |
|---|---|
| Ridge, Lasso, ElasticNet | Classical Linear |
| Random Forest, Gradient Boosting | Classical Ensemble |
| XGBoost, LightGBM, CatBoost | Gradient Boosting |
| Voting Ensemble, Stacking Ensemble | Hybrid Ensemble |
| LSTM | Deep Learning |
| Distilled MLP | Knowledge Distillation (deployed) |

---

## Key Design Decisions

- **Ground truth**: AQI computed from raw PM2.5 using the US EPA linear interpolation formula
- **Loss function**: Huber loss for gradient boosters (XGBoost, LightGBM, CatBoost) and LSTM — robust to AQI spike outliers that inflate MSE and suppress IoA
- **Forecasting**: MIMO (direct multi-output) — predicts 24h/48h/72h simultaneously, avoids error accumulation
- **Features**: 50+ engineered features including multi-scale lags (1h–168h / 7-day), rolling statistics (mean, std, min, max over 24h), STL decomposition, physics-derived and cross features
- **Normalization**: log(x+1) → RobustScaler for skewed pollutants; scaler fitted on training data only
- **Model protection**: Champion-Challenger gate — new model must beat champion by ≥3% RMSE **and** have IoA ≥ 0.40 (prevents a mean-predicting model from holding the champion slot)
- **Deployed model**: Distilled MLP (fast inference; knowledge distilled from top-3 tree ensemble teachers)

---

## Performance Notes

### Dashboard speed on Render free tier
- **Cold starts (~30–50 s)** are unavoidable on the free tier — containers spin down after 15 min of inactivity
- **In-session performance** is fast: the champion model is downloaded from Hopsworks once per container lifecycle and cached in memory. Subsequent tab switches and 5-minute refresh intervals skip the download entirely

### Why IoA matters alongside RMSE
RMSE-minimising models can "win" by predicting close to the mean — low error but zero predictive value. IoA (Index of Agreement, Willmott 1981) measures how well the model tracks actual AQI movements. IoA = 1.0 is perfect; IoA < 0.5 means the model is worse than predicting the mean. The champion promotion gate requires IoA ≥ 0.40 to prevent low-IoA models from being deployed.
