import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "titanic_model"

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    vers = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not vers:
        raise RuntimeError("No Staging versions found")
    v = vers[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=v,
        stage="Production",
        archive_existing_versions=False,
    )
    print(f"Promoted {MODEL_NAME} v{v} to Production")

if __name__ == "__main__":
    main()
