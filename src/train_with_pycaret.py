from pathlib import Path
import pandas as pd
import mlflow
from pycaret.classification import setup, compare_models, finalize_model, save_model, pull

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "titanic_automl"

def main():
    # Переобучаемся на "новых" данных (current) — имитация прод-сценария
    train = pd.read_csv(DATA_DIR / "current.csv")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="pycaret_train"):
        # ВАЖНО: не используем встроенный pycaret mlflow logger (он конфликтует по версиям)
        setup(
            data=train,
            target="target",
            session_id=42,
            fold=3,
            normalize=True,
            verbose=False,
        )

        best = compare_models(sort="F1")
        final = finalize_model(best)

        # Сохраняем модель локально (PyCaret pipeline)
        model_path_prefix = MODELS_DIR / "best_model"
        save_model(final, str(model_path_prefix))  # создаст best_model.pkl

        # Метрики/таблица сравнения (что вернул PyCaret)
        results_df = pull()
        results_path = MODELS_DIR / "pycaret_compare_results.csv"
        results_df.to_csv(results_path, index=False)

        # Логируем артефакты в MLflow вручную
        mlflow.log_artifact(str(model_path_prefix) + ".pkl", artifact_path="model")
        mlflow.log_artifact(str(results_path), artifact_path="reports")

        # Параметры (минимально)
        mlflow.log_param("dataset", "titanic_openml")
        mlflow.log_param("train_source", "data/current.csv")

        print("Saved model to:", str(model_path_prefix) + ".pkl")
        print("Logged artifacts to MLflow experiment:", EXPERIMENT_NAME)

if __name__ == "__main__":
    main()
