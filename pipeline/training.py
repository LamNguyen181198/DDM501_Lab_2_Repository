import mlflow
from surprise import SVD, NMF, KNNBasic
import pickle 

# Define model classes mapping
MODEL_CLASSES = {
    'svd': SVD,
    'nmf': NMF,
    'knn': KNNBasic
}

def train_model(trainset, model_type: str = 'svd', **params):
    """Train model and log to MLflow."""
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)

        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Initialize and train model
        model_class = MODEL_CLASSES[model_type]
        model = model_class(**params)
        model.fit(trainset)

        # Save model artifact
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
            mlflow.log_artifact("model.pkl")
        return model, mlflow.active_run().info.run_id