from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "titanic_model"
EXPERIMENT_NAME = "titanic_automl"

class PyCaretWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Грузим через PyCaret, а не pickle напрямую
        from pycaret.classification import load_model

        pkl_path = context.artifacts["model_pkl"]  # .../best_model.pkl
        prefix = str(Path(pkl_path).with_suffix(""))  # .../best_model
        self.model = load_model(prefix)

    def predict(self, context, model_input, params=None):
        import pandas as pd
        from pycaret.classification import predict_model

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        if "target" in model_input.columns:
            model_input = model_input.drop(columns=["target"])

        out = predict_model(self.model, data=model_input)

        # В PyCaret обычно это prediction_label / prediction_score
        if "prediction_label" in out.columns:
            return out["prediction_label"]
        if "Label" in out.columns:
            return out["Label"]

        # fallback: вернём последнюю колонку (на всякий случай)
        return out.iloc[:, -1]

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Train first.")

    run_id = runs[0].info.run_id

    # Скачиваем артефакт модели из последнего run
    local_dir = Path("models")
    local_dir.mkdir(exist_ok=True)
    local_pkl = client.download_artifacts(run_id, "model/best_model.pkl", str(local_dir))
    print("Downloaded artifact to:", local_pkl)

    # Логируем MLflow pyfunc модель в рамках этого же run
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pyfunc.log_model(
            artifact_path="registered_model",
            python_model=PyCaretWrapper(),
            artifacts={"model_pkl": local_pkl},
            pip_requirements=[
                "pycaret",
                "pandas",
                "scikit-learn",
                "lightgbm",
            ],
        )

    mv = mlflow.register_model(model_uri=model_info.model_uri, name=MODEL_NAME)

    # Переводим новую версию в Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Registered {MODEL_NAME} version={mv.version} -> Staging")

if __name__ == "__main__":
    main()
