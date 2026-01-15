from __future__ import annotations

import os
import subprocess
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run(cmd: str):
    subprocess.check_call(cmd, shell=True, cwd=PROJECT_DIR)

def drift_detected() -> bool:
    run("python src/check_drift.py")
    flag_path = os.path.join(PROJECT_DIR, "data", "drift_flag.txt")
    with open(flag_path, "r", encoding="utf-8") as f:
        return f.read().strip() == "1"

with DAG(
    dag_id="mlops_titanic_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "drift", "mlflow"],
) as dag:

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=drift_detected,
    )

    train = PythonOperator(
        task_id="train_with_pycaret",
        python_callable=lambda: run("python src/train_with_pycaret.py"),
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=lambda: run("python src/register_model.py"),
    )

    check_drift >> train >> register
