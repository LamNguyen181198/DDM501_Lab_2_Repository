import mlflow
from pipeline.data_ingestion import load_data, split_data
from pipeline.training import train_model
from pipeline.evaluation import evaluate_model

EXPERIMENT_NAME = "movie_rating_recommendation_v2"

# Define experiment configurations
EXPERIMENTS = [
 {"model_type": "svd", "n_factors": 50, "n_epochs": 20},
 {"model_type": "svd", "n_factors": 100, "n_epochs": 20},
 {"model_type": "svd", "n_factors": 100, "n_epochs": 50},
 {"model_type": "nmf", "n_factors": 50, "n_epochs": 50},
 {"model_type": "knn", "k": 40, "sim_options": {"name": "cosine"}},
]

def run_all_experiments():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data once
    data = load_data()
    trainset, testset = split_data(data)

    results = []
    for config in EXPERIMENTS:
        config_copy = config.copy()
        model_type = config_copy.pop("model_type")
        model, run_id = train_model(trainset, model_type, **config_copy)
        metrics = evaluate_model(model, testset, run_id)
        results.append({"config": config_copy, "metrics": metrics, "run_id": run_id})

    return results