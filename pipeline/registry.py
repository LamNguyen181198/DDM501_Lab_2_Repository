import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name: str, model_name: str):
    """Find best run and register model."""
    client = MlflowClient()
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)

    #Find best run by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    best_run = runs[0]

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri, model_name)
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    return best_run.info.run_id