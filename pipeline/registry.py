import mlflow
import os
from mlflow.tracking import MlflowClient


def register_best_model(experiment_name: str, model_name: str):
    """Find best run and register model."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    # Find best run by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    best_run = runs[0]

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    registered = mlflow.register_model(model_uri, model_name)

    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production"
    )
    return best_run.info.run_id