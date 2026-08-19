"""
src/data_prep.py — Load, clean, split, and save train/test data
"""

from pathlib import Path

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, DATA_PATH, TRAIN_PATH, TEST_PATH, DATA_DIR

log = structlog.get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("data_loaded", rows=len(df), path=str(path))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True)
    return df


def prepare_data(
    data_path: str = DATA_PATH,
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    target: str = "Churn",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load → clean → split → save.
    Returns train/test splits.
    """
    df = load_data(data_path)
    df = clean_data(df)

    X = df.drop(columns=[target])
    y = df[target]

    log.info("data_stats", churn_rate=f"{y.mean():.1%}", features=X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    log.info("data_split", train_rows=X_train.shape[0], test_rows=X_test.shape[0])

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{target: y_train}).to_csv(train_path, index=False)
    X_test.assign(**{target: y_test}).to_csv(test_path, index=False)
    log.info("splits_saved", train_path=str(train_path), test_path=str(test_path))

    return X_train, X_test, y_train, y_test


def load_splits(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    target: str = "Churn",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load previously saved train/test splits."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    log.info("splits_loaded", train_rows=X_train.shape[0], test_rows=X_test.shape[0])
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.config import setup_logging

    setup_logging()
    prepare_data()
