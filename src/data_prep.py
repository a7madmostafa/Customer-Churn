"""
src/data_prep.py — Load, clean, split, and build a preprocessor
"""

import os
from typing import Tuple
from src.config import RANDOM_STATE
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at '{path}'")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True)
    return df


def prepare_data(path: str, target: str = "Churn", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load → clean → split.
    Returns raw (unprocessed) splits — the Pipeline will handle preprocessing.

    Returns:
        X_train, X_test, y_train, y_test  (all raw DataFrames/Series)
    """
    df = load_data(path)
    df = clean_data(df)

    X = df.drop(columns=[target])
    y = df[target]

    print(f"  Churn rate: {y.mean():.1%}  |  Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


