import mlflow
from pipeline.data_ingestion import load_data, split_data
from pipeline.training import train_model
from pipeline.evaluation import evaluate_model
from itertools import product

EXPERIMENT_NAME = "movie_rating_recommendation_v2"


def run_hyperparameter_tuning():
    """Run hyperparameter tuning experiments with different configurations."""
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load and split data once
    print("Loading data...")
    data = load_data()
    trainset, testset = split_data(data)
    
    # Define hyperparameter grid
    param_grid = {
        'model_type': ['svd', 'nmf'],
        'n_factors': [50, 100, 150],
        'n_epochs': [10, 20, 30],
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Running {len(combinations)} experiments...")
    
    best_rmse = float('inf')
    best_params = None
    best_run_id = None
    
    # Run experiments
    for i, params in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Training with params: {params}")
        
        try:
            # Train model
            model, run_id = train_model(
                trainset,
                model_type=params['model_type'],
                n_factors=params['n_factors'],
                n_epochs=params['n_epochs']
            )
            
            # Evaluate model
            metrics = evaluate_model(model, testset, run_id)
            
            # Track best model
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_params = params
                best_run_id = run_id
            
            print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    # Print results
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Run ID: {best_run_id}")
    print("="*60)
    
    return best_run_id, best_params, best_rmse


if __name__ == "__main__":
    print("Starting hyperparameter tuning experiment...")
    best_run_id, best_params, best_rmse = run_hyperparameter_tuning()
    print(f"\nBest model saved with run_id: {best_run_id}")