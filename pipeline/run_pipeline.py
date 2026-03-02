"""Orchestrator: run all pipeline stages end-to-end."""

from pipeline.data_ingestion import load_data, split_data
from pipeline.preprocessing import preprocess
from pipeline.training import run_training
from pipeline.evaluation import evaluate_model
from pipeline.registry import register_model


def run_pipeline(data_path: str | None = None) -> dict:
    """Execute the full ML pipeline.

    Stages:
        1. Data ingestion  – load and split data
        2. Preprocessing   – scale features
        3. Training        – train model and log with MLflow
        4. Evaluation      – compute and log metrics
        5. Registration    – register model if quality gates pass

    Args:
        data_path: Optional override for the CSV data path.

    Returns:
        Dictionary of evaluation metrics produced by the pipeline.
    """
    print("=" * 50)
    print("Starting ML Pipeline")
    print("=" * 50)

    # Stage 1: Data ingestion
    kwargs = {"data_path": data_path} if data_path else {}
    df = load_data(**kwargs)
    X_train, X_test, y_train, y_test = split_data(df)

    # Stage 2: Preprocessing
    X_train_scaled, X_test_scaled, _ = preprocess(X_train, X_test)

    # Stage 3: Training
    model, run_id = run_training(X_train_scaled, y_train)

    # Stage 4: Evaluation
    metrics = evaluate_model(model, X_test_scaled, y_test, run_id)

    # Stage 5: Model registration
    register_model(run_id, metrics)

    print("=" * 50)
    print("Pipeline completed successfully.")
    print("=" * 50)
    return metrics


if __name__ == "__main__":
    run_pipeline()
