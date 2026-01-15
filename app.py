from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "titanic_model"

LOG_PATH = Path("logs/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

_model_cache: Dict[str, Any] = {}

def _get_model(stage: str):
    """Load model from Registry stage (cached)."""
    if stage in _model_cache:
        return _model_cache[stage]
    uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.pyfunc.load_model(uri)
    _model_cache[stage] = model
    return model

def _stage_exists(stage: str) -> bool:
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    return len(versions) > 0

def choose_stage(user_id: int) -> str:
    prod = _stage_exists("Production")
    stg = _stage_exists("Staging")

    if prod and stg:
        return "Production" if (user_id % 2 == 0) else "Staging"
    if prod:
        return "Production"
    if stg:
        return "Staging"
    # совсем крайний случай
    return "Staging"

def log_row(row: Dict[str, Any]) -> None:
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON object expected"}), 400

    user_id = int(payload.get("user_id", 0))
    features = payload.get("features")

    if features is None:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    # features can be dict (single) or list[dict] (batch)
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, list) and all(isinstance(x, dict) for x in features):
        df = pd.DataFrame(features)
    else:
        return jsonify({"error": "'features' must be dict or list of dicts"}), 400

    # ensure no target leakage
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    stage = choose_stage(user_id)
    model = _get_model(stage)

    preds = model.predict(df)
    # convert numpy/pandas to list
    if hasattr(preds, "tolist"):
        preds_out = preds.tolist()
    else:
        preds_out = list(preds)

    # log
    log_row({
        "ts": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "stage": stage,
        "n_rows": len(df),
        "features_json": json.dumps(features, ensure_ascii=False),
        "pred_json": json.dumps(preds_out, ensure_ascii=False),
    })

    return jsonify({"stage": stage, "predictions": preds_out})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
