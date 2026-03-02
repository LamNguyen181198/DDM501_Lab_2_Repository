from surprise import Dataset
from surprise.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)

def load_data(dataset_name: str = 'ml-100k'):
    """Load dataset from surprise built-in datasets."""
    logger.info(f"Loading dataset: {dataset_name}")
    data = Dataset.load_builtin(dataset_name)
    return data

def split_data(data, test_size: float = 0.2):
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={test_size}")
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset