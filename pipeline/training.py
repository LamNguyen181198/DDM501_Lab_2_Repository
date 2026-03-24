import logging
import os
import pickle
import tempfile

import mlflow
import mlflow.pyfunc
import pandas as pd
from surprise import SVD, NMF, KNNBasic

logger = logging.getLogger(__name__)

# Define model classes mapping
MODEL_CLASSES = {
    'svd': SVD,
    'nmf': NMF,
    'knn': KNNBasic
}


class SurprisePyfuncModel(mlflow.pyfunc.PythonModel):
    """Wrap a Surprise model so it can be logged as an MLflow pyfunc model."""

    def load_context(self, context):
        with open(context.artifacts["surprise_model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas DataFrame")

        required_cols = {"user_id", "item_id"}
        missing_cols = required_cols - set(model_input.columns)
        if missing_cols:
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(f"Missing required input columns: {missing}")

        preds = [
            self.model.predict(str(row.user_id), str(row.item_id)).est
            for row in model_input.itertuples(index=False)
        ]
        return pd.Series(preds)


def train_model(trainset, model_type: str = 'svd', **params):
    """Train model and log to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_type not in MODEL_CLASSES:
        supported = ", ".join(sorted(MODEL_CLASSES.keys()))
        raise ValueError(f"Unsupported model_type='{model_type}'. Supported: {supported}")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)

        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Initialize and train model
        model_class = MODEL_CLASSES[model_type]
        model = model_class(**params)
        model.fit(trainset)

        # Save and log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path, artifact_path="artifacts")

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SurprisePyfuncModel(),
                artifacts={"surprise_model": model_path},
            )

        run_id = mlflow.active_run().info.run_id
        logger.info("Model training completed. run_id=%s", run_id)
        return model, run_id