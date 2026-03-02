# Configuration parameters for the ML pipeline

# Data settings
DATA_PATH = "data/wine.csv"
TARGET_COLUMN = "quality"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing settings
SCALER = "standard"  # Options: "standard", "minmax"

# Model settings
MODEL_NAME = "ElasticNet"
ALPHA = 0.5
L1_RATIO = 0.5

# MLflow settings
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "wine-quality"
REGISTERED_MODEL_NAME = "WineQualityModel"

# Evaluation thresholds
RMSE_THRESHOLD = 1.0
R2_THRESHOLD = 0.0
