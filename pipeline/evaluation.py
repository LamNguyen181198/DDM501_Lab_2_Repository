import mlflow
import os
from surprise import accuracy
import matplotlib.pyplot as plt

def create_prediction_plot(predictions):
    """Create a plot comparing actual vs predicted ratings."""
    actual = [pred.r_ui for pred in predictions]
    predicted = [pred.est for pred in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual, predicted, alpha=0.5)
    ax.plot([0, 5], [0, 5], 'r--', lw=2)  # Perfect prediction line
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Predicted Rating')
    ax.set_title('Actual vs Predicted Ratings')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def evaluate_model(model, testset, run_id: str):
    """Evaluate model and log metrics to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if not run_id:
        raise ValueError("run_id is required to log evaluation metrics")

    with mlflow.start_run(run_id=run_id):
        # Make predictions
        predictions = model.test(testset)

        # Calculate metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Create and log plots
        fig = create_prediction_plot(predictions)
        mlflow.log_figure(fig, "prediction_distribution.png")
        plt.close(fig)
        
        return {"rmse": rmse, "mae": mae}