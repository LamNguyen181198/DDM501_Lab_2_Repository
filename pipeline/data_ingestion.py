from surprise import Dataset
from surprise.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_data(dataset_name: str = 'ml-100k'):
    """Load dataset from surprise built-in datasets."""
    logger.info(f"Loading dataset: {dataset_name}")
    data = Dataset.load_builtin(dataset_name, prompt=False)
    return data


def split_data(data, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""
    logger.info(
        "Splitting data with test_size=%s and random_state=%s",
        test_size,
        random_state,
    )
    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    return trainset, testset