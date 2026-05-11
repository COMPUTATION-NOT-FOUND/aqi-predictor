#!/usr/bin/env bash
# run_local.sh — Launch the dashboard locally for development & presentation testing
#
# Why this exists
# ---------------
# Render.com's free tier kills worker processes after 120–300 s of slow
# Hopsworks queries, making it hard to tell whether bugs are in the app
# or in the hosting environment.  Run this script locally — it uses
# the same gunicorn config as production but on your machine where there
# is no memory limit or worker timeout.
#
# Prerequisites
# -------------
#   1.  cp .env.example .env  (and fill in your API keys)
#   2.  pip install -r requirements.txt
#   3.  pip install gunicorn          (if not already installed)
#
# Usage
# -----
#   chmod +x run_local.sh
#   ./run_local.sh
#
# Then open http://localhost:8050 in your browser.
#
# To stop:  Ctrl-C

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f ".env" ]; then
    echo "[run_local] Loading environment from .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "[run_local] WARNING: .env not found. Set API keys as environment variables or create .env"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
export PORT="${PORT:-8050}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-mlruns}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pearls AQI Predictor — local development server         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  URL:          http://localhost:${PORT}                     ║"
echo "║  MLflow URI:   ${MLFLOW_TRACKING_URI}                           ║"
echo "║  Hopsworks:    ${HOPSWORKS_PROJECT:-NOT SET}                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    echo "[run_local] Activating virtualenv: $VENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "[run_local] WARNING: venv not found at $VENV_ACTIVATE"
    echo "[run_local]          Create it with: python3 -m venv venv && pip install -r requirements.txt"
fi

# ── Run with gunicorn (same as production) ───────────────────────────────────
# --timeout 300  : matches Procfile — avoids worker kill during slow Hopsworks queries
# --workers 1    : single worker shares the in-memory caches (important!)
# --reload       : auto-restart on code changes (dev convenience)
exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --timeout 300 \
    --reload \
    --log-level info \
    src.dashboard.app:server
