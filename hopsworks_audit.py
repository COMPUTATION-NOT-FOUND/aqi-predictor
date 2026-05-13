"""
Hopsworks audit — run locally to verify feature store + model registry state.
Usage: python hopsworks_audit.py
"""
import os, sys, json
from dotenv import load_dotenv
load_dotenv()

import hopsworks

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")

print(f"\n{'═'*60}")
print(f"  Hopsworks Audit  —  project: {PROJECT}")
print(f"{'═'*60}\n")

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT)
    print(f"{PASS} Connected to Hopsworks project '{project.name}'")
except Exception as e:
    print(f"{FAIL} Could not connect: {e}")
    sys.exit(1)

# ── 1. Feature Store — Feature Groups ────────────────────────────────────────
print(f"\n{'─'*60}")
print("1. FEATURE STORE — Feature Groups")
print(f"{'─'*60}")
try:
    fs = project.get_feature_store()
    fgs = fs.get_feature_groups(name="aqi_features")
    if not fgs:
        # try listing all
        print(f"{WARN} No feature group named 'aqi_features' found")
        print(f"{INFO} Attempting to list all feature groups...")
        try:
            all_fgs = fs._feature_store_api.get_all_feature_groups(fs)
            for fg in all_fgs:
                print(f"  {INFO} FG: {fg.name} v{fg.version}  ({fg.features.__len__()} features)")
        except Exception:
            print(f"  {WARN} Could not list all FGs")
    else:
        for fg in fgs:
            print(f"{PASS} Feature Group: '{fg.name}'  version={fg.version}  features={len(fg.features)}")
            # Count rows
            try:
                df = fg.read()
                print(f"       Rows in store: {len(df)}")
                print(f"       Columns: {list(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
                if len(df) == 0:
                    print(f"  {WARN} Feature group is EMPTY — backfill has not run yet")
                else:
                    print(f"  {PASS} Latest timestamp: {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
            except Exception as e:
                print(f"  {WARN} Could not read FG data: {e}")
except Exception as e:
    print(f"{FAIL} Feature store error: {e}")

# ── 2. Feature Store — Feature Views ─────────────────────────────────────────
print(f"\n{'─'*60}")
print("2. FEATURE STORE — Feature Views")
print(f"{'─'*60}")
try:
    fs = project.get_feature_store()
    fvs = fs.get_feature_views(name="aqi_feature_view")
    if not fvs:
        print(f"{WARN} No feature view named 'aqi_feature_view' found")
    else:
        for fv in fvs:
            print(f"{PASS} Feature View: '{fv.name}'  version={fv.version}")
except Exception as e:
    print(f"{WARN} Feature view check error: {e}")

# ── 3. Model Registry ─────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("3. MODEL REGISTRY — All Models")
print(f"{'─'*60}")
try:
    mr = project.get_model_registry()
    all_models = mr.get_models()
    if not all_models:
        print(f"{FAIL} Model registry is EMPTY — training pipeline has not run yet")
        print(f"  {INFO} Fix: run  python src/training_pipeline/train_model.py")
    else:
        print(f"{INFO} Found {len(all_models)} model version(s):")
        for m in all_models:
            metrics_str = ""
            if m.training_metrics:
                rmse = m.training_metrics.get("rmse", "?")
                metrics_str = f"  rmse={rmse}"
            champion_tag = "  *** CHAMPION ***" if m.name == "aqi_champion" else ""
            print(f"  {PASS} {m.name}  v{m.version}{metrics_str}{champion_tag}")
except Exception as e:
    print(f"{FAIL} Model registry error: {e}")

# ── 4. Champion model specifically ───────────────────────────────────────────
print(f"\n{'─'*60}")
print("4. CHAMPION MODEL CHECK")
print(f"{'─'*60}")
try:
    mr = project.get_model_registry()
    champions = mr.get_models(name="aqi_champion")
    if not champions:
        print(f"{FAIL} 'aqi_champion' does NOT exist in registry")
        print(f"  {INFO} This is why Render shows the error.")
        print(f"  {INFO} Fix options:")
        print(f"         A) Run training locally and push to registry:")
        print(f"              python src/training_pipeline/train_model.py")
        print(f"         B) Trigger the training GitHub Action manually")
        print(f"         C) Manually register a model — see hopsworks_register_manual.py")
    else:
        best = champions[-1]
        print(f"{PASS} 'aqi_champion' EXISTS  — version={best.version}")
        try:
            model_dir = best.download()
            import pathlib
            files = list(pathlib.Path(model_dir).iterdir())
            print(f"  {INFO} Artifact files: {[f.name for f in files]}")
            if not (pathlib.Path(model_dir) / "model.pkl").exists():
                print(f"  {FAIL} model.pkl missing from artifact!")
            else:
                print(f"  {PASS} model.pkl present")
            if not (pathlib.Path(model_dir) / "scaler_bundle.pkl").exists():
                print(f"  {WARN} scaler_bundle.pkl missing (model will run without scaling)")
            else:
                print(f"  {PASS} scaler_bundle.pkl present")
            if (pathlib.Path(model_dir) / "metadata.json").exists():
                import json
                with open(pathlib.Path(model_dir) / "metadata.json") as f:
                    meta = json.load(f)
                print(f"  {INFO} Metadata: {json.dumps(meta, indent=4)}")
        except Exception as e:
            print(f"  {WARN} Could not download/inspect artifact: {e}")
except Exception as e:
    print(f"{FAIL} Champion check error: {e}")

print(f"\n{'═'*60}")
print("Audit complete.")
print(f"{'═'*60}\n")
