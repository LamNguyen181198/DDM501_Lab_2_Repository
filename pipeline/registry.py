"""Registry module: register a trained model in the MLflow Model Registry."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    RMSE_THRESHOLD,
    R2_THRESHOLD,
)


def register_model(run_id: str, metrics: dict) -> None:
    """Register a model from a completed MLflow run if it meets quality gates.

    The model is registered only when:
        - RMSE is below ``RMSE_THRESHOLD``
        - R² is above ``R2_THRESHOLD``

    Args:
        run_id: MLflow run ID that contains the logged model artifact.
        metrics: Evaluation metrics dict with at least 'rmse' and 'r2' keys.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    rmse = metrics.get("rmse", float("inf"))
    r2 = metrics.get("r2", float("-inf"))

    if rmse > RMSE_THRESHOLD or r2 < R2_THRESHOLD:
        print(
            f"[registry] Model did NOT meet quality gates "
            f"(RMSE={rmse:.4f}, R²={r2:.4f}). Skipping registration."
        )
        return

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_tag(
        name=REGISTERED_MODEL_NAME,
        key="stage",
        value="staging",
    )

    print(
        f"[registry] Registered model '{REGISTERED_MODEL_NAME}' "
        f"version {result.version} from run {run_id}."
    )
