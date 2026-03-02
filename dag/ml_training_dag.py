from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Task functions with XCom for data passing
def load_data_fn(**context):
    """Load and prepare data."""
    from pipeline.data_ingestion import load_data, split_data
    import pickle

    data = load_data()
    trainset, testset = split_data(data)

    with open('/tmp/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open('/tmp/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)

    return "Data loaded successfully"


def train_model_fn(**context):
    """Train model and log to MLflow."""
    import pickle
    from pipeline.training import train_model

    with open('/tmp/trainset.pkl', 'rb') as f:
        trainset = pickle.load(f)

    model, run_id = train_model(
        trainset,
        model_type='svd',
        n_factors=100,
        n_epochs=20
    )

    # Save model for next task
    with open('/tmp/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='run_id', value=run_id)
    return run_id


def evaluate_fn(**context):
    """Evaluate model and log metrics."""
    import pickle
    from pipeline.evaluation import evaluate_model

    # Load model and testset
    with open('/tmp/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('/tmp/testset.pkl', 'rb') as f:
        testset = pickle.load(f)

    # Get run_id from previous task
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')

    # Evaluate
    metrics = evaluate_model(model, testset, run_id)
    
    context['ti'].xcom_push(key='metrics', value=metrics)
    return metrics


def register_fn(**context):
    """Register best model to MLflow Model Registry."""
    from pipeline.registry import register_best_model

    # Register best model from experiment
    best_run_id = register_best_model(
        experiment_name="movie_rating_recommendation",
        model_name="movie_rating_model"
    )
    
    print(f"Registered model from run: {best_run_id}")
    return best_run_id


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'movie_rating_training',
    default_args=default_args,
    description='ML Training Pipeline',
    schedule_interval='@weekly',
    catchup=False,
)

# Task definitions
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data_fn,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_fn,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_fn,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_fn,
    dag=dag,
)

# Define dependencies
load_data_task >> train_model_task >> evaluate_task >> register_task