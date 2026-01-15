import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "titanic_model"

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod:
        raise RuntimeError("No Production version found")
    prod_v = prod[0].version

    # Переводим тот же version в Staging и архивируем старые staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=prod_v,
        stage="Staging",
        archive_existing_versions=True,
    )

    print(f"Staging now points to Production version: v{prod_v}")

if __name__ == "__main__":
    main()
