# DDM501 Lab 2 - ML Pipeline and Experiment Tracking

This repository contains an end-to-end MLOps workflow for movie rating prediction using Surprise, MLflow, and Airflow.

## Project Structure

- `pipeline/`: modular ML pipeline logic
  - `data_ingestion.py`: dataset loading and train/test splitting
  - `training.py`: model training and MLflow logging
  - `evaluation.py`: metric/plot generation and MLflow metric logging
  - `registry.py`: best-run selection and model registry transition
  - `run_experiments.py`: batch experiment execution
- `experiments/`
  - `run_experiments.py`: standalone hyperparameter tuning script
- `dag/`
  - `ml_training_dag.py`: Airflow DAG for scheduled pipeline automation
- `docker-compose.yml`: Airflow + MLflow + Postgres + Redis orchestration
- `Dockerfile`: custom Airflow image with pipeline dependencies

## Prerequisites

- Docker Desktop
- Docker Compose

## Quick Start

1. Build and start services:

```bash
docker compose up --build -d
```

2. Access Airflow:

- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

3. Access MLflow:

- URL: http://localhost:5001

4. Trigger DAG:

- DAG ID: `movie_rating_training`
- Schedule: `@weekly`

## Standalone Experiment Tuning

Run tuning outside Airflow:

```bash
python experiments/run_experiments.py
```

This script logs runs to the experiment name:

- `movie_rating_recommendation_v2`

## MLflow Tracking Details

The pipeline logs:

- Parameters: model type and hyperparameters
- Metrics: RMSE and MAE
- Artifacts: prediction plot and serialized model artifact
- Model artifact path for registry: `model`

## Final Results

- Successful experiment: `movie_rating_recommendation_v2`
- Best run ID: `fc1334f3a5e24e108a2bcd30df3d74ae`
- Selected model: `SVD`
- Hyperparameters: `n_factors=100`, `n_epochs=20`
- RMSE: `0.9354`
- MAE: `0.7385`
- Registered model: `movie_rating_model`
- Registered version: `1`
- Model stage: `Production`

## Model Registry Behavior

The registry step:

1. Finds the best run by lowest RMSE
2. Registers `runs:/<run_id>/model` as `movie_rating_model`
3. Transitions the registered version to `Production`

Final registry outcome:

- Model name: `movie_rating_model`
- Version: `1`
- Stage: `Production`
- Source run: `fc1334f3a5e24e108a2bcd30df3d74ae`

## Reproducibility Notes

- Dependencies are pinned in `requirements.txt`
- Data split uses a fixed `random_state=42`
- Airflow Celery tasks exchange intermediate files through shared mount:
  - host: `./shared`
  - container: `/opt/airflow/shared`

## Common Commands

Stop services:

```bash
docker compose down
```

Stop and remove volumes:

```bash
docker compose down -v
```

View service logs:

```bash
docker compose logs -f airflow-scheduler
```

## Deliverables Mapping to Rubric

- Pipeline quality: modular pipeline files in `pipeline/`
- Experiment tracking: MLflow logging in training/evaluation/registry
- Airflow automation: DAG in `dag/ml_training_dag.py`
- Documentation: this README plus `docs/experiment_report.md`
