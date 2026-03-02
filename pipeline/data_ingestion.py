"""Data ingestion module: load dataset and split into train/test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        data_path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    df = pd.read_csv(data_path)
    print(f"[data_ingestion] Loaded data from '{data_path}': {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train and test feature/label arrays.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"[data_ingestion] Train size: {len(X_train)}, Test size: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test
