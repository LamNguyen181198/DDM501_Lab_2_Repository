# Experiment Report - DDM501 Lab 2

## 1. Objective

Train and compare recommendation models for movie rating prediction, then register the best model using MLflow Model Registry.

## 2. Dataset

- Source: Surprise built-in dataset `ml-100k`
- Split strategy: train/test split with `test_size=0.2`, `random_state=42`

## 3. Experiment Setup

- Tracking tool: MLflow
- Experiment name: `movie_rating_recommendation_v2`
- Models considered:
  - SVD
  - NMF
  - KNNBasic

## 4. Hyperparameter Search Space

Example search (from `experiments/run_experiments.py`):

- `model_type`: `svd`, `nmf`
- `n_factors`: 50, 100, 150
- `n_epochs`: 10, 20, 30

## 5. Logged Outputs

Per run, the pipeline logs:

- Parameters:
  - `model_type`
  - model hyperparameters (e.g., `n_factors`, `n_epochs`)
- Metrics:
  - RMSE
  - MAE
- Artifacts:
  - `prediction_distribution.png`
  - serialized model artifact
- Registry-ready MLflow model artifact:
  - `model`

## 6. Best Model Selection

Selection criterion:

- Lowest RMSE across runs in `movie_rating_recommendation_v2`

Registry action:

1. Find best run by RMSE ascending
2. Register model URI: `runs:/<best_run_id>/model`
3. Transition resulting model version to `Production`

Final registered model details:

- Model name: `movie_rating_model`
- Version: `1`
- Stage: `Production`
- Source run: `fc1334f3a5e24e108a2bcd30df3d74ae`

## 7. Airflow Automation

- DAG ID: `movie_rating_training`
- Schedule: `@weekly`
- Task order:
  1. load_data
  2. train_model
  3. evaluate_model
  4. register_model

Intermediate handoff:

- Shared directory for Celery-safe task communication:
  - `/opt/airflow/shared` (container)
  - `./shared` (host)

## 8. Results Table

| Run ID | Model Type | Params | RMSE | MAE | Notes |
|---|---|---|---:|---:|---|
| `fc1334f3a5e24e108a2bcd30df3d74ae` | SVD | `n_factors=100`, `n_epochs=20` | 0.9354 | 0.7385 | Best successful Airflow pipeline run; model registered |

## 9. Discussion

- The best completed pipeline run used the SVD model with `n_factors=100` and `n_epochs=20`.
- The final successful run produced `RMSE=0.9354` and `MAE=0.7385`, which became the basis for model registration.
- The Airflow DAG executed all four stages successfully: data loading, training, evaluation, and model registration.
- The final pipeline stored parameters, metrics, artifacts, and the registered model in MLflow as required by the lab rubric.

## 10. Conclusion

- The selected model was SVD with `n_factors=100` and `n_epochs=20` because it achieved the lowest recorded RMSE among successful runs in `movie_rating_recommendation_v2`.
- The model was registered from URI `runs:/fc1334f3a5e24e108a2bcd30df3d74ae/model` as `movie_rating_model` version `1` and promoted to the `Production` stage.
- Recommended next steps are expanding the hyperparameter search space, comparing additional recommender algorithms, and adding a more formal validation strategy for model selection.
